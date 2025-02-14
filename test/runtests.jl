using Tests

@testset "PbTe TDEP" begin
    include("test_tdep_harmonic.jl")
    test_tdep_harmonic()
end
