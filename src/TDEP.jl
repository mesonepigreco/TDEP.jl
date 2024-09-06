module TDEP

using LinearAlgebra
using ForwardDiff
using ReverseDiff
using Optim
using FileIO
using JLD2
using AtomicEnsemble

include("tdep_core.jl")

export tdep_fit!

end # module TDEP
