using TDEP
using Optim
using AtomicEnsemble
using FileIO
using JLD2
using Unitful, UnitfulAtomic


function run_tdep()
    ensemble_loc = "ensemble.jld2"
    save_fname = joinpath(@__DIR__, "tdep_results.jld2")
    save_last = joinpath(@__DIR__, "tdep_last.jld2")

    ensemble = load_ensemble(joinpath(@__DIR__, ensemble_loc))
    apply_asr!(ensemble)
    # Get the number of atoms
    nat = length(ensemble.structures[1])

    fc_matrix = zeros(Float64, 3nat, 3nat)
    centroids = zeros(Float64, 3nat)

    # Prepare
    kT = uconvert(u"eV", 450u"K" * UnitfulAtomic.k_au).val
    tdep_anal!(fc_matrix, centroids, ensemble, kT)

    # Load a previous calculation if any
    data = nothing
    if isfile(save_fname)
        data = load(save_fname)
        println("Loading previous calculation from $save_fname")
    elseif isfile(save_last)
        data = load(save_last)
        println("Loading previous calculation from $save_last")
    end
    
    if data != nothing
        fc_matrix .= data["fc_matrix"]
        centroids .= data["centroids"]
    end

    # Use TDEP to fit the force constants
    options = Optim.Options(show_trace = true, show_every = 1, iterations=5000,
                            callback = x -> begin
                                TDEP.save_model(typeof(x.value), save_last)
                                false
                            end)

    tdep_fit!(fc_matrix, centroids, ensemble; optimizer = BFGS(), optimizer_options = options)

    # Save on file
    save(save_fname, Dict("fc_matrix" => fc_matrix, "centroids" => centroids))
end

run_tdep()
