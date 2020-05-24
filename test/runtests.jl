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

    mutable struct SeqRng <: AbstractRNG
        ind
        SeqRng(ind) = new(ind)
        SeqRng() = new(0)
    end
    function Random.rand(rng::SeqRng, vec::AbstractArray)
        rng.ind = rng.ind % length(vec) + 1
        return vec[rng.ind]
    end

    @info "Testing util"
    include("util.jl")

    @info "Testing archspace"
    include("archspace.jl")

    @info "Testing mutation"
    include("mutation.jl")

    @info "Testing fitness"
    include("fitness.jl")

    @info "Testing candidate"
    include("candidate.jl")

    @info "Testing evolve"
    include("evolve.jl")

    @info "Testing population"
    include("population.jl")

    @info "Testing iterators"
    include("iterators.jl")

    @info "Testing visualization"
    include("visualization/callbacks.jl")

    @info "Testing README examples"
    include("examples.jl")

    @info "Testing AutoFlux"
    include("app/autoflux.jl")
end
