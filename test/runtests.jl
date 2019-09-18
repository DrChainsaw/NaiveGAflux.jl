using NaiveGAflux
using Random
using Logging
using Test

@testset "NaiveGAflux.jl" begin

    mutable struct MockRng <:AbstractRNG
        seq::AbstractVector
        ind::Integer
        MockRng(seq) = new(seq, 0)
    end

    function Random.rand(rng::MockRng)
        rng.ind = rng.ind % length(rng.seq) + 1
        return rng.seq[rng.ind]
    end

    @info "Testing util"
    include("util.jl")

    @info "Testing archspace"
    include("archspace.jl")

    @info "Testing mutation"
    include("mutation.jl")

    @info "Testing README examples"
    include("examples.jl")

    @info "Testing CIFAR10 example"
    include("examples/Cifar10.jl")

end
