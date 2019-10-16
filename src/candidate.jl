"""
    AbstractCandidate

Abstract base type for canidates
"""
abstract type AbstractCandidate end

"""
    reset!(c::AbstractCandidate)

Reset state of `c`. Typically needs to be called after evolution to clear old fitness computations.
"""
function reset!(c::AbstractCandidate) end
Base.Broadcast.broadcastable(c::AbstractCandidate) = Ref(c)

"""
    CandidateModel <: Candidate
    CandidateModel(model::CompGraph, optimizer, lossfunction, fitness::AbstractFitness)

A candidate model consisting of a `CompGraph`, an optimizer a lossfunction and a fitness method.
"""
struct CandidateModel <: AbstractCandidate
    graph::CompGraph
    opt
    lossfun
    fitness::AbstractFitness
end

Flux.children(c::CandidateModel) = (c.graph, c.opt, c.lossfun, c.fitness)
Flux.mapchildren(f, c::CandidateModel) = CandidateModel(f(c.graph), f(c.opt), f(c.lossfun), c.fitness)

function Flux.train!(model::CandidateModel, data::AbstractArray{<:Tuple})
    f = instrument(Train(), model.fitness, model.graph)
    loss(x,y) = model.lossfun(f(x), y)
    iloss = instrument(TrainLoss(), model.fitness, loss)
    Flux.train!(iloss, params(model.graph), data, model.opt)
end

Flux.train!(model::CandidateModel, data::Tuple{<:AbstractArray, <:AbstractArray}) = Flux.train!(model, [data])

# Assume iterator in the general case
function Flux.train!(model::CandidateModel, iter)
    for data in iter
        Flux.train!(model, data)
    end
end

fitness(model::CandidateModel) = fitness(model.fitness, instrument(Validate(), model.fitness, model.graph))

reset!(model::CandidateModel) = reset!(model.fitness)

graph(model::CandidateModel) = model.graph


"""
    HostCandidate <: AbstractCandidate
    HostCandidate(c::AbstractCandidate)

Keeps `c` in host memory and transfers to GPU when training or calculating fitness.
"""
struct HostCandidate <: AbstractCandidate
    c::AbstractCandidate
end

Flux.@treelike HostCandidate

function Flux.train!(c::HostCandidate, data)
    Flux.train!(c.c |> gpu, data)
    cleanopt(c) # As optimizer state does not survive transfer from gpu -> cpu
    c.c |> cpu # As some parts, namely CompGraph change internal state when mapping to GPU
    gpu_gc()
end

function fitness(c::HostCandidate)
    fitval = fitness(c.c |> gpu)
    c.c |> cpu # As some parts, namely CompGraph change internal state when mapping to GPU
    gpu_gc()
    return fitval
end

const gpu_gc = if Flux.has_cuarrays()
    function()
        GC.gc()
        CuArrays.reclaim(true)
    end
else
    () -> nothing
end

reset!(c::HostCandidate) = reset!(c.c)
graph(c::HostCandidate) = graph(c.c)

"""
    CacheCandidate <: AbstractCandidate
    CacheCandidate(c::AbstractCandidate)

Caches fitness values produced by `c` until `reset!` is called.

Useful with `HostCandidate` to pervent models from being pushed to/from GPU just to fetch fitness values.
"""
mutable struct CacheCandidate <: AbstractCandidate
    fitnesscache
    c::AbstractCandidate
end
CacheCandidate(c::AbstractCandidate) = CacheCandidate(nothing, c)

Flux.train!(c::CacheCandidate, data) = Flux.train!(c.c, data)

function fitness(c::CacheCandidate)
    if isnothing(c.fitnesscache)
        c.fitnesscache = fitness(c.c)
    end
    return c.fitnesscache
end

function reset!(c::CacheCandidate)
    c.fitnesscache = nothing
    reset!(c.c)
end
graph(c::CacheCandidate) = graph(c.c)


"""
    evolvemodel(m::AbstractMutation{CompGraph}, newfields::Function=deepcopy)

Return a function which maps a `AbstractCandidate c1` to a new `AbstractCandidate c2` where `c2.graph = m(copy(c1.graph))`.

All other fields are mapped through the function `newfields`.

Intended use is together with [`EvolveCandidates`](@ref).
"""
function evolvemodel(m::AbstractMutation{CompGraph}, newfields::Function=deepcopy)
    function copymutate(g::CompGraph)
        ng = copy(g)
        m(ng)
        return ng
    end
    mapcandidate(copymutate, newfields)
end

function mapcandidate(newgraph::Function, newfields::Function=deepcopy)
    mapfield(g::CompGraph) = newgraph(g)
    mapfield(f) = newfields(f)
    return c -> newcand(c, mapfield)
end

newcand(c::CandidateModel, mapfield) = CandidateModel(map(mapfield, getproperty.(c, fieldnames(CandidateModel)))...)
newcand(c::HostCandidate, mapfield) = HostCandidate(newcand(c.c, mapfield))
newcand(c::CacheCandidate, mapfield) = CacheCandidate(newcand(c.c, mapfield))

function clearstate(s) end
clearstate(s::AbstractDict) = foreach(k -> delete!(s, k), keys(s))

cleanopt(o::T) where T = foreach(fn -> clearstate(getfield(o, fn)), fieldnames(T))
cleanopt(o::Flux.Optimise.Optimiser) = foreach(cleanopt, o.os)
cleanopt(c::CandidateModel) = cleanopt(c.opt)
cleanopt(c::HostCandidate) = cleanopt(c.c)
cleanopt(c::CacheCandidate) = cleanopt(c.c)

"""
    AbstractEvolution

Abstract base type for strategies for how to evolve a population into a new population.
"""
abstract type AbstractEvolution end

"""
    evolve!(e::AbstractEvolution, population::AbstractArray{<:AbstractCandidate})

Evolve `population` into a new population. New population may or may not contain same individuals as before.
"""
function evolve! end

"""
    NoOpEvolution <: AbstractEvolution
    NoOpEvolution()

Does not evolve the given population.
"""
struct NoOpEvolution <: AbstractEvolution end
evolve!(::NoOpEvolution, pop) = pop

"""
    AfterEvolution <: AbstractEvolution
    AfterEvolution(evo::AbstractEvolution, fun::Function)

Return `fun(newpop)` where `newpop = evolve!(e.evo, pop)` where `pop` is original population to evolve.
"""
struct AfterEvolution <: AbstractEvolution
    evo::AbstractEvolution
    fun::Function
end

function evolve!(e::AfterEvolution, pop)
    newpop = evolve!(e.evo, pop)
    return e.fun(newpop)
end

"""
    ResetAfterEvolution(evo::AbstractEvolution)

Alias for `AfterEvolution` with `fun == reset!`.
"""
ResetAfterEvolution(evo::AbstractEvolution) = AfterEvolution(evo, resetandreturn)
function resetandreturn(pop)
    foreach(reset!, pop)
    return pop
end

"""
    EliteSelection <: AbstractEvolution
    EliteSelection(nselect::Integer)

Selects the only the `nselect` highest fitness candidates.
"""
struct EliteSelection <: AbstractEvolution
    nselect::Integer
end
evolve!(e::EliteSelection, pop::AbstractArray{<:AbstractCandidate}) = partialsort(pop, 1:e.nselect, by=fitness, rev=true)

"""
    SusSelection <: AbstractEvolution
    SusSelection(nselect::Integer, evo::AbstractEvolution, rng=rng_default)

Selects candidates for further evolution using stochastic universal sampling.
"""
struct SusSelection <: AbstractEvolution
    nselect::Integer
    evo::AbstractEvolution
    rng
    SusSelection(nselect, evo, rng=rng_default) = new(nselect, evo, rng)
end

function evolve!(e::SusSelection, pop::AbstractArray{<:AbstractCandidate})
    csfitness = cumsum(fitness.(pop))

    gridspace = csfitness[end] / e.nselect
    start = rand(e.rng)*gridspace

    selected = similar(pop, e.nselect)
    for i in eachindex(selected)
        candind = findfirst(x -> x >= start + (i-1)*gridspace, csfitness)
        selected[i] = pop[candind]
    end
    return evolve!(e.evo, selected)
end

"""
    CombinedEvolution <: AbstractEvolution
    CombinedEvolution(evos::AbstractArray{<:AbstractEvolution})
    CombinedEvolution(evos::AbstractEvolution...)

Combines the evolved populations from several `AbstractEvolutions` into one population.
"""
struct CombinedEvolution <: AbstractEvolution
    evos::AbstractArray{<:AbstractEvolution}
end
CombinedEvolution(evos::AbstractEvolution...) = CombinedEvolution(collect(evos))
evolve!(e::CombinedEvolution, pop) = mapfoldl(evo -> evolve!(evo, pop), vcat, e.evos)

"""
    EvolveCandidates <: AbstractEvolution
    EvolveCandidates(fun::Function)

Applies `fun` for each candidate in a given population.

Useful with [`evolvemodel`](@ref).
"""
struct EvolveCandidates <: AbstractEvolution
    fun::Function
end
evolve!(e::EvolveCandidates, pop) = map(e.fun, pop)
