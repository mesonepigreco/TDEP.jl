
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
    init_params!(fc_matrix :: Matrix{T}, centroids :: Vector{T}, ensemble :: StandardEnsemble)

This model solves the TDEP analitically exploiting the displacement-displacement correlation function.
"""
function tdep_anal!(fc_matrix :: Matrix{T}, centroids :: Vector{T}, ensemble :: StandardEnsemble, kT; apply_asr = true) where T
    n_configs = length(ensemble)
    nat = length(ensemble.structures[1])

    if apply_asr
        apply_asr!(ensemble)
    end

    u_disp = zeros(T, 3nat, n_configs)
    centroids .= 0
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

    for i in 1:n_configs
        u_disp[:, i] .-= centroids
    end


    # Now fit the displacement-displacements
    fc_matrix .= 0
    for i in 1:n_configs
        @views fc_matrix .+= u_disp[:, i] * u_disp[:, i]'
    end

    ω, p = eigen(fc_matrix)
    fc_matrix .= 0
    # Invert the matrix discarding the low energy values
    for μ in 1:3nat
        if ω[μ] > 1e-5
            @views mul!(fc_matrix, p[:, μ], p[:, μ]', 1.0 / (ω[μ] * kT), 1.0)
        end
    end
end

@doc raw"""
    tdep_fit!(fc_matrix :: Matrix{T}, centroids :: Vector{T}, ensemble :: StandardEnsemble)

Fit the force constants and centroids to the ensemble data.

The fit is performed minimizing the least squares error between harmonic forces
and the forces in the ensemble.
The final force constants and centroids are stored in the `fc_matrix` and `centroids` arguments (modified in-place).

"""
function tdep_fit!(fc_matrix :: Matrix{T}, centroids :: Vector{T}, ensemble :: StandardEnsemble;
        optimizer = LBFGS(), optimizer_options = Optim.options(iterations=1000, show_trace = true)) where T
    # Check consistency between sizes
    n_structures = length(ensemble)
    nat3 = size(fc_matrix, 1)
    nat = nat3 ÷ 3
    @assert nat3 == length(centroids)
    @assert length(ensemble.structures) == size(ensemble.forces, 3)

    fitted_forces = zeros(T, nat3)
    u_disps = zeros(T, nat3)
    params = zeros(T, count_params(nat))


    # Fill the initial parameters with the provided matrix
    mat2param_vect!(params, fc_matrix, centroids)

    function least_squares(params :: AbstractArray{T}) :: T where T
        # Unpack the parameters (careful with ForwardDiff)
        my_cache = get_cache(T, nat3)
        fc_matrix = my_cache.fc_matrix
        centroids = my_cache.centroids
        u_disps = my_cache.u_disps
        fitted_forces = my_cache.fitted_forces

        param_vect2mat!(fc_matrix, centroids, params)

        # Get the displacements
        least_squares_res = zero(T)
        for i in 1:n_structures
            for j in 1:nat
                for k in 1:3
                    h = (j - 1) * 3 + k
                    u_disps[h] = ensemble.structures[i].positions[k, j] - centroids[h]
                end
            end
            #fitted_forces .= fc_matrix * u_disps
            mul!(fitted_forces, fc_matrix, u_disps)

            for j in 1:nat
                for k in 1:3
                    h = (j - 1) * 3 + k
                    least_squares_res += abs2(fitted_forces[h] - ensemble.forces[k, j, i])
                end
            end
        end

        least_squares_res
    end

    back_grad!(grad, params) = ReverseDiff.gradient!(grad, least_squares, params)
    results = optimize(least_squares, back_grad!, params, optimizer, optimizer_options)

    # copy back 
    param_vect2mat!(fc_matrix, centroids, params)

    return results
end


@doc raw"""
    param_vect2mat!(fc_matrix :: Matrix{T}, centroids :: Vector{T}, params :: Vector{T})

Unpack the parameters vector into the force constants matrix and centroids vector.
Inplace modifies fc_matrix and centorids.
"""
function param_vect2mat!(fc_matrix :: AbstractArray{T}, centroids :: AbstractArray{T}, params :: AbstractArray{T}) where T
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
end

@doc raw"""
    mat2param_vect!(params :: Vector{T}, fc_matrix :: Matrix{T}, centroids :: Vector{T})

Pack the force constants matrix and centroids vector into a parameters vector.
Inplace modifies params.
"""
function mat2param_vect!(params :: Vector{T}, fc_matrix :: Matrix{T}, centroids :: Vector{T}) where {T}
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
