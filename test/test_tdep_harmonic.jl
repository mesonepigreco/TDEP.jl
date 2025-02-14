using Test
using TDEP
using AtomicSymmetries
using AtomicEnsemble
using DelimitedFiles
using Unitful, UnitfulAtomic
using LinearAlgebra

function test_tdep_harmonic(; verbose=false)
    # Load the data 
    harmonic_fc = readdlm(joinpath(@__DIR__, "data/pbte_fc_unit.txt"))
    forces = readdlm(joinpath(@__DIR__, "data/pbte_f_unit.txt"))
    positions = readdlm(joinpath(@__DIR__, "data/pbte_r_unit.txt"))
    cell = readdlm(joinpath(@__DIR__, "data/pbte_unit_cell.txt"))
    avg_struct = readdlm(joinpath(@__DIR__, "data/pbte_struct.txt"))
    masses = [1.0, 1.0]
    atoms = ["Pb", "Te"]
    atoms_types = [1, 2]

    kT = ustrip(auconvert(100.0u"K" * u"k")) # Temperature in K

    n_structures = size(positions, 2)

    # Get the symmetry group
    crystal_coords = zeros(Float64, 3, 2)
    AtomicSymmetries.get_crystal_coords!(crystal_coords, avg_struct, cell)
    symmetry_group = get_symmetry_group_from_spglib(crystal_coords, cell, atoms_types)

    if verbose
        println("Number of symmetry operations: ", length(symmetry_group))
    end

    # Setup the ensemble
    structures = [Structure(reshape(positions[:, i], 3, :),
                            masses, 
                            cell,
                            atoms) for i in 1:n_structures]

    energies = zeros(Float64, n_structures)
    ensemble = StandardEnsemble(structures, energies, reshape(forces, 3, 2, :))


    # Perform the calculation
    fc_drdr = zeros(Float64, size(harmonic_fc)...)
    fc_tdep = zeros(Float64, size(harmonic_fc)...)
    centroids = copy(avg_struct)

    tdep_anal!(fc_drdr, reshape(avg_struct, :), ensemble, kT;
               symmetry_group=symmetry_group)
    
    # Compare the eigenvalues
    drdr_eigvals = eigvals(fc_drdr)
    orig_eigvals = eigvals(harmonic_fc)

    if verbose
        println("drdr eigenvalues: ", drdr_eigvals)
        println("correct eigenvalues: ", orig_eigvals)
    end

    # Now perform the tdep_fit
    res, fc_tdep, centroids = tdep_fit!(fc_drdr, avg_struct, ensemble;
                                        symmetry_group=symmetry_group)

    tdep_eigvals = eigvals(fc_tdep)
    if verbose
        println("average positions: ", avg_struct')
        println("new centroids: ", reshape(centroids, 3, :)')
        println("tdep eigenvalues: ", tdep_eigvals)
        println("correct eigenvalues: ", orig_eigvals)
    end

    # Let us check if TDEP is really fitting
    # for i in 1:n_structures
    #     udisp = positions[:, i] - reshape(avg_struct, :)

    #     # Compute the force
    #     harm_force = -harmonic_fc * udisp
    #     tdep_force = -fc_tdep * udisp

    #     if verbose
    #         println("Configuration: ", i)
    #         println("Displacement: ", udisp)
    #         println("Harmonic force: ", harm_force)
    #         println("TDEP force: ", tdep_force)
    #         println("Real force: ", forces[:, i])
    #     end
    # end
    #
    # Check the results (ignoring asr modes)
    for i in 4:6
        @test drdr_eigvals[i] ≈ orig_eigvals[i] rtol=0.1
        @test tdep_eigvals[i] ≈ orig_eigvals[i] rtol=0.1
    end



end

if abspath(PROGRAM_FILE) == @__FILE__
    test_tdep_harmonic(verbose=true)
end
