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

    @info "Testing shape"
    include("shape.jl")

    @info "Testing archspace"
    include("archspace.jl")

    @warn "Skipping mutation"
    #@info "Testing mutation"
    #include("mutation.jl")

    @warn "Skipping crossover"
    #@info "Testing crossover"
    #include("crossover.jl")

    @warn "Skipping fitness"
    #@info "Testing fitness"
    #include("fitness.jl")

    @warn "Skipping candidate"
    #@info "Testing candidate"
    #include("candidate.jl")

    @warn "Skipping evolve"
    #@info "Testing evolve"
    #include("evolve.jl")

    @warn "Skipping population"
    #@info "Testing population"
    #include("population.jl")

    @warn "Skipping iterators"
    #@info "Testing iterators"
    #include("iterators.jl")

    @warn "Skipping visualization"
    #@info "Testing visualization"
    #include("visualization/callbacks.jl")

    if VERSION === v"1.6.1"
        @warn "Skipping readme examples"
        #@info "Testing README examples"
        #include("examples.jl")
    else
        @warn "README examples will only be tested in julia version 1.6.1 due to rng dependency. Skipping..."
    end

    @warn "Skipping autoflux"
    #@info "Testing AutoFlux"
    #include("app/autoflux.jl")
end
