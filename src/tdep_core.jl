mutable struct Cache{U}
    fc_matrix :: Matrix{U}
    centroids :: Vector{U}
    u_disps :: Vector{U}
    fitted_forces :: Vector{U}
end
function init_cache(T, nat3)
    println("Initializing cache with type $T and size $nat3")
    Cache{T}(zeros(T, nat3, nat3), 
             zeros(T, nat3), 
             zeros(T, nat3),
             zeros(T, nat3))
end
function get_cache(T :: DataType, nat3; noinit = false)
    if !haskey(cache_map, T) 
        if noinit
            return nothing
        end
        cache_map[T] = init_cache(T, nat3)
    end
    cache_map[T]
end

const cache_map = Dict{DataType, Cache}()

@doc raw"""
    tdep_anal!(fc_matrix :: Matrix{T}, centroids :: Vector{T}, ensemble :: StandardEnsemble, kT; apply_asr = true)

Perform the TDEP analysis on the ensemble data.
This routine exploits the analytical solution to the TDEP equations: 
the displacement-displacement correlation matrix.
"""
function tdep_anal!(fc_matrix :: Matrix{T}, centroids :: Vector, ensemble :: StandardEnsemble, kT; apply_asr = true, 
    symmetry_group = nothing) where T
    n_configs = length(ensemble)
    nat = length(ensemble.structures[1])

    # if apply_asr
    #     apply_asr!(ensemble)
    # end

    u_disp = zeros(eltype(centroids), 3nat, n_configs)
    centroids .= 0.0
    for i in 1:n_configs
        for j in 1:nat
            for k in 1:3
                index = (j-1) * 3 + k
                centroids[index] += ensemble.structures[i].positions[k, j]
                u_disp[index, i] = ensemble.structures[i].positions[k,j] 
            end
        end
    end
    centroids ./= n_configs

    # Apply the symmetries on the centroids
    if symmetry_group != nothing
        cell = ustrip.(auconvert.(ensemble.structures[1].cell))
        # println("centroid symmetrization")
        tmp_cent = reshape(centroids, 3, nat)
        # println()
        # println("Before symmetrization: ", tmp_cent')
        # println("Cell: ", cell')
        # println()
        symmetrize_positions!(reshape(centroids, 3, nat), 
                              cell,
                              symmetry_group)
        # println("After symmetrization: ", tmp_cent')
        # println()
    end

    for i in 1:n_configs
        u_disp[:, i] .-= centroids

        # Remove the ASR
        for j in 1:3
            δ = 0 
            for k in 1:nat
                δ += u_disp[(k-1)*3 + j, i]  / nat
            end

            for k in 1:nat
                u_disp[(k-1)*3 + j, i] -= δ
            end
        end
    end




    # Now fit the displacement-displacements
    type = eltype(ustrip.(fc_matrix))
    δrδr_mat = zeros(typeof(zero(type)), 3nat, 3nat)

    for i in 1:n_configs
        for a in 1:3nat
            myval = u_disp[a, i]
            for b in 1:3nat
                δrδr_mat[b, a] += myval * u_disp[b, i] / kT
            end
        end
    end
    δrδr_mat ./= n_configs

    ω, p = eigen(ustrip.(δrδr_mat))
    ω *= unit(δrδr_mat[1])
    fc_matrix .= 0.0
    # Invert the matrix discarding the low energy values
    for μ in 1:3nat
        if ω[μ] > ustrip(auconvert(1e-4u"meV"))
            fc_matrix .+= p[:, μ] * p[:, μ]' ./ ω[μ]

            #@views mul!(fc_matrix, p[:, μ], p[:, μ]', 1.0 / (ω[μ]), 1.0)
        end
    end

    # Impose the symmetries
    if symmetry_group != nothing
        #println("fc matrix symmetrization")
        symmetrize_fc!(fc_matrix, cell, symmetry_group)
    end
end

@doc raw"""
    tdep_fit!(fc_matrix :: Matrix{T}, centroids :: Vector{T}, ensemble :: StandardEnsemble; symmetry_group = nothing, optimizer = LBFGS(), optimizer_options = Optim.options(), cartesian = true)

Fit the force constants and centroids to the ensemble data.

The fit is performed minimizing the least squares error between harmonic forces
and the forces in the ensemble.
The final force constants and centroids are stored in the `fc_matrix` and `centroids` arguments (modified in-place).

The inner algorithm performs the calculation in crystalline space
(easier for symmetry reasons).

If you want to use cartesian coordinates, set `cartesian = true` (default).
In this case both the input and output data will be assumed to be in cartesian coordinates.
"""
function tdep_fit!(fc_matrix :: AbstractMatrix, average_structure :: AbstractMatrix, ensemble :: StandardEnsemble;
        symmetry_group = nothing,
        optimizer = LBFGS(), optimizer_options = Optim.Options(iterations=1000, show_trace = true))
    # Check consistency between sizes
    n_structures = length(ensemble)
    nat3 = size(fc_matrix, 1)
    nat = nat3 ÷ 3

    cell = ustrip.(auconvert.(ensemble.structures[1].cell))

    @assert nat3 == length(average_structure) @assert length(ensemble.structures) == size(ensemble.forces, 3)

    # Get the type of the matrix
    T = eltype(ustrip(fc_matrix))

    fitted_forces = zeros(T, nat3)
    u_disps = zeros(T, nat3)
    centroids = zeros(T, nat3)

    # If the symmetries are provided, extract the generators
    n_params = count_params(nat)
    vector_generators = nothing
    fc_generators = nothing
    if symmetry_group != nothing
        # Impose symmetries on the average structure (wyckoff positions)
        symmetrize_positions!(average_structure, cell, symmetry_group)

        # Generators of the displacements
        vector_generators = AtomicSymmetries.get_vector_generators(symmetry_group, cell)
        @info "Finding the generators of the force constants"
        @time fc_generators = AtomicSymmetries.get_matrix_generators(symmetry_group, cell)

        # Count the number of parameters
        n_params = length(vector_generators) + length(fc_generators)
    end
    params = zeros(T, n_params)

    @info "Fitting the force constants with $n_params parameters"

    # Get the ASR functions
    asr! = ASRConstraint!(3)

    # Convert the ensemble forces and displacements to atomic units
    ensemble_coords = zeros(T, 3*nat, n_structures)
    ensemble_forces = zeros(T, 3*nat, n_structures)
    for i in 1:n_structures
        ensemble_coords[:, i] =  reshape(ustrip.(auconvert.(ensemble.structures[i].positions)), 3*nat, :)
        @views ensemble_forces[:, i] =  reshape(ustrip.(auconvert.(ensemble.forces[:, :, i])), 3*nat, :)
    end

    # Preallocate the views
    ensemble_coords_views = [view(ensemble_coords, :, i) for i in 1:n_structures]
    ensemble_forces_views = [view(ensemble_forces, :, i) for i in 1:n_structures]

    # Fill the initial parameters with the provided matrix
    @info "Converting the matrix to parameters"
    @time mat2param_vect!(params, fc_matrix, centroids, cell; 
                    vector_generators = vector_generators, 
                    fc_generators = fc_generators, 
                    symmetry_group = symmetry_group)

    function least_squares(params :: AbstractArray{T}) :: T where T
        # Unpack the parameters (careful with ForwardDiff)
        # my_cache = get_cache(T, nat3)
        # fc_matrix = my_cache.fc_matrix
        # centroids = my_cache.centroids
        # u_disps = my_cache.u_disps
        # fitted_forces = my_cache.fitted_forces
        fc_matrix = zeros(T, nat3, nat3)
        centroids = zeros(T, nat3)
        u_disps = zeros(T, nat3)
        fitted_forces = zeros(T, nat3)

        @info "Unpacking the parameters"
        @time param_vect2mat!(fc_matrix, centroids, params, cell; 
                        vector_generators = vector_generators, 
                        fc_generators = fc_generators, 
                        symmetry_group = symmetry_group)

        # Apply the ASR!
        asr!(fc_matrix; differentiable=true)
        asr!(centroids)

        # Add the average structure to the centroids
        # (Centroids is a displacement vector from the average structure)
        centroids .+= reshape(average_structure, :)

        # Get the displacements
        least_squares_res = zero(T)
        @info "Computing the least squares"
        @time begin
            for i in 1:n_structures
                # for j in 1:nat
                #     for k in 1:3
                #         h = (j - 1) * 3 + k
                #         #u_disps[h] = ensemble.structures[i].positions[k, j] - centroids[h]
                #         u_disps[h] = ensemble_coords[k, j, i] - centroids[h]
                #     end
                # end
                #@views u_disps .= ensemble_coords[:, i] .- centroids
                #@views broadcast!(-, u_disps, ensemble_coords[:, i], centroids)
                broadcast!(-, u_disps, ensemble_coords_views[i], centroids)
                #fitted_forces .= fc_matrix * u_disps
                mul!(fitted_forces, fc_matrix, u_disps, -1.0, 0.0)
                #@views broadcast!(-, u_disps, ensemble_forces[:, i], fitted_forces)
                broadcast!(-, u_disps, ensemble_forces_views[i], fitted_forces)
                #@views fitted_forces .-= ensemble_forces[:, i]
                least_squares_res += sum(abs2, u_disps)

                # for j in 1:nat
                #     for k in 1:3
                #         h = (j - 1) * 3 + k
                #         #least_squares_res += abs2(fitted_forces[h] - ensemble.forces[k, j, i])
                #         least_squares_res += abs2(fitted_forces[h] - ensemble_forces[k, j, i])
                #     end
                # end
            end
        end
        # println()
        # println("Least squares: ", least_squares_res / n_structures)
        # println()

        least_squares_res / n_structures
    end

    # Prepare the gradient tape
    diff_tape = ReverseDiff.GradientTape(least_squares, params)
    diff_tape_compiled = ReverseDiff.compile(diff_tape)


    back_grad!(grad, params) = ReverseDiff.gradient!(grad, diff_tape_compiled, params)
    results = optimize(least_squares, back_grad!, params, optimizer, optimizer_options)

    # copy back 
    # println("Optimization results: ", Optim.minimizer(results))
    param_vect2mat!(fc_matrix, centroids, Optim.minimizer(results), cell; 
                    vector_generators = vector_generators, 
                    fc_generators = fc_generators, 
                    symmetry_group = symmetry_group)

    # Add the wyckoff positions to the centroids
    centroids .+= reshape(average_structure, :)
    
    # Convert the output to cartesian
    # if cartesian
    #     get_cartesian_coords!(reshape(tmp_cent, 3, :), reshape(centroids, 3, :), cell)
    #     AtomicSymmetries.cart_cryst_matrix_conversion!(tmp_mat, fc_matrix, cell; cart_to_cryst = false)

    #     centroids .= tmp_cent
    #     fc_matrix .= tmp_mat
    # end

    return results, fc_matrix, centroids
end


@doc raw"""
    param_vect2mat!(fc_matrix :: Matrix{T}, centroids :: Vector{T}, params :: Vector{T}, cell ::AbstractMatrix)

Unpack the parameters vector into the force constants matrix and centroids vector.
Inplace modifies fc_matrix and centorids.
"""
function param_vect2mat!(fc_matrix :: AbstractArray{T}, centroids :: AbstractArray{T}, params :: AbstractArray{T}, cell :: AbstractMatrix; 
        vector_generators = nothing,
        fc_generators = nothing,
        symmetry_group = nothing
    ) where T
    if symmetry_group === nothing
        # Unpack the parameters
        count = 1
        nat3 = size(fc_matrix, 1)
        for i in 1:nat3
            for j in i:nat3
                fc_matrix[i, j] = params[count]
                fc_matrix[j, i] = params[count]
                count += 1
            end
        end
        for i in 1:nat3
            centroids[i] = params[count]
            count += 1
        end
    else
        n_vectors = length(vector_generators)
        n_matrices = length(fc_generators)

        if n_vectors > 0
            @views AtomicSymmetries.get_vector_from_generators!(centroids,
                                                         vector_generators, 
                                                         params[1:n_vectors], 
                                                         symmetry_group)
        else
            centroids .= 0
        end
        @views AtomicSymmetries.get_fc_from_generators!(fc_matrix, 
                                                 fc_generators, 
                                                 params[n_vectors+1:end], 
                                                 symmetry_group,
                                                 cell)
    end
end

@doc raw"""
    mat2param_vect!(params :: Vector{T}, fc_matrix :: Matrix{T}, centroids :: Vector{T}, cell :: AbstractMatrix)

Pack the force constants matrix and centroids vector into a parameters vector.
Inplace modifies params.
"""
function mat2param_vect!(params :: AbstractVector, fc_matrix :: AbstractMatrix, centroids :: AbstractVector, cell :: AbstractMatrix; 
        vector_generators = nothing,
        fc_generators = nothing,
        symmetry_group = nothing) 
    if symmetry_group === nothing
        count = 1
        nat3 = size(fc_matrix, 1)
        for i in 1:nat3
            for j in i:nat3
                params[count] = fc_matrix[i, j]
                count += 1
            end
        end
        for i in 1:nat3
            params[count] = centroids[i]
            count += 1
        end
    else
        n_vectors = length(vector_generators)
        n_matrices = length(fc_generators)

        if n_vectors > 0
            @views AtomicSymmetries.get_coefficients_from_fc!(params[1:n_vectors],
                                                              centroids, 
                                                              vector_generators, 
                                                              symmetry_group, 
                                                              cell)
        end
        @views AtomicSymmetries.get_coefficients_from_fc!(params[n_vectors+1:end],
                                                          fc_matrix, 
                                                          fc_generators, 
                                                          symmetry_group,
                                                          cell)
    end

    end

# Get the number of parameters
@doc raw"""
    count_params(nat :: Int)

Return the number of parameters needed to fit the force constants and centroids.
"""
function count_params(nat :: Int)
    nat3 = nat * 3
    n_mat = nat3 * (nat3 + 1) ÷ 2
    n_vec = nat3

    n_mat + n_vec
end
function get_nat3(n_params :: Int) :: Int
    nat3 = Int(sqrt(9 + 8 * n_params) - 3) ÷ 2
    return nat3 
end


function save_model(T :: DataType, fname :: String) 
    my_cache = get_cache(T, 1; noinit = true)
    if my_cache != nothing
        save(fname, Dict("fc_matrix" => my_cache.fc_matrix, "centroids" => my_cache.centroids))
    else
        println("No cache found for type $T")
    end
end
