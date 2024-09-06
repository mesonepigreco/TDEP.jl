using TDEP
using Optim
using AtomicEnsemble
using FileIO
using JLD2


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

    # Load a previous calculation if any
    if isfile(save_fname)
        data = load(save_fname)
    elseif isfile(save_last)
        data = load(save_last)
    end
    fc_matrix .= data["fc_matrix"]
    centroids .= data["centroids"]


    # Use TDEP to fit the force constants
    options = Optim.Options(show_trace = true, show_every = 1,
                            callback = x -> begin
                                TDEP.save_model(typeof(x.value), save_last)
                                false
                            end)

    tdep_fit!(fc_matrix, centroids, ensemble; optimizer = LBFGS(), optimizer_options = options)

    # Save on file
    save(save_fname, Dict("fc_matrix" => fc_matrix, "centroids" => centroids))
end

run_tdep()
