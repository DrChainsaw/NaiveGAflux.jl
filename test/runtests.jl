using NaiveGAflux, Random, Logging, Test
using NaiveNASlib.Advanced, NaiveNASlib.Extend

@testset "NaiveGAflux.jl" begin

    mutable struct MockRng <:AbstractRNG
        seq::AbstractVector
        ind::Integer
        MockRng(seq) = new(seq, 0)
    end

    function Random.rand(rng::MockRng, type::Type{T}=eltype(rng.seq)) where T
        rng.ind = rng.ind % length(rng.seq) + 1
        return convert(type, rng.seq[rng.ind])
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

    @info "Testing shape"
    include("shape.jl")

    @info "Testing batch size utils"
    include("batchsize.jl")

    @info "Testing iterator mapping"
    include("iteratormaps.jl")

    @info "Testing archspace"
    include("archspace.jl")

    @info "Testing mutation"
    include("mutation/generic.jl")
    include("mutation/graph.jl")
    include("mutation/optimizer.jl")
    include("mutation/iteratormaps.jl")

    @info "Testing crossover"
    include("crossover/graph.jl")
    include("crossover/optimizer.jl")
    include("crossover/iteratormaps.jl")

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

    if VERSION === v"1.9.2"
        @info "Testing README examples"
        include("examples.jl")
    else
        @warn "README examples will only be tested in julia version 1.9.2 due to rng dependency. Skipping..."
    end

    @info "Testing AutoFlux"
    include("app/autoflux.jl")

    @info "Testing documentation"
    import Documenter
    Documenter.doctest(NaiveGAflux)

    @info "Testing AutoOptimiser"
    include("autooptimiser.jl")
end
