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

Flux.functor(c::CandidateModel) = (c.graph, c.opt, c.lossfun), gcl -> CandidateModel(gcl..., c.fitness)

function Flux.train!(model::CandidateModel, data)
    f = instrument(Train(), model.fitness, model.graph)
    loss(x,y) = model.lossfun(f(x), y)
    iloss = instrument(TrainLoss(), model.fitness, loss)
    Flux.train!(iloss, Flux.params(model.graph), data, model.opt)
end

Flux.train!(model::CandidateModel, data::Tuple{<:AbstractArray, <:AbstractArray}) = Flux.train!(model, [data])

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

Flux.@functor HostCandidate

function Flux.train!(c::HostCandidate, data)
    eagermutation(graph(c)) # Optimization: If there is a LazyMutable somewhere, we want it to do its thing now so we don't end up copying the model to the GPU only to then trigger another copy when the mutations are applied.
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

reset!(c::HostCandidate) = reset!(c.c)
graph(c::HostCandidate) = graph(c.c)

const gpu_gc = if CuArrays.functional()
    function()
        GC.gc()
        CuArrays.reclaim()
    end
else
    () -> nothing
end


function eagermutation(x) end
function eagermutation(v::InputVertex) end
eagermutation(g::CompGraph) = eagermutation.(vertices(g::CompGraph))
eagermutation(v::AbstractVertex) = eagermutation(base(v))
eagermutation(v::CompVertex) = eagermutation(v.computation)
eagermutation(m::AbstractMutableComp) = eagermutation(NaiveNASflux.wrapped(m))
eagermutation(m::LazyMutable) = m(NoComp())

struct NoComp end
function (::NaiveNASflux.MutableLayer)(x::NoComp) end



"""
    CacheCandidate <: AbstractCandidate
    CacheCandidate(c::AbstractCandidate)

Caches fitness values produced by `c` until `reset!` is called.

Useful with `HostCandidate` to prevent models from being pushed to/from GPU just to fetch fitness values.
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


nparams(c::AbstractCandidate) = nparams(graph(c))
nparams(g::CompGraph) = mapreduce(prod ∘ size, +, params(g).order)

"""
    evolvemodel(m::AbstractMutation{CompGraph}, newfields::Function=deepcopy)
    evolvemodel(m::AbstractMutation{CompGraph}, om::AbstractMutation{FluxOptimizer}, mapothers=deepcopy)

Return a function which maps a `AbstractCandidate c1` to a new `AbstractCandidate c2` where any `CompGraph`s `g` in `c1` will be m(copy(g))` in `c2`. Same principle is applied to any optimisers if `om` is present.


All other fields are mapped through the function `newfields`.

Intended use is together with [`EvolveCandidates`](@ref).
"""
function evolvemodel(m::AbstractMutation{CompGraph}, mapothers=deepcopy)
    function copymutate(g::CompGraph)
        ng = copy(g)
        m(ng)
        return ng
    end
    mapcandidate(copymutate, mapothers)
end
evolvemodel(m::AbstractMutation{CompGraph}, om::AbstractMutation{FluxOptimizer}, mapothers=deepcopy) = evolvemodel(m, optmap(om, mapothers))

function mapcandidate(mapgraph, mapothers=deepcopy)
    mapfield(g::CompGraph) = mapgraph(g)
    mapfield(f) = mapothers(f)
    return c -> newcand(c, mapfield)
end

newcand(c::CandidateModel, mapfield) = CandidateModel(map(mapfield, getproperty.(c, fieldnames(CandidateModel)))...)
newcand(c::HostCandidate, mapfield) = HostCandidate(newcand(c.c, mapfield))
newcand(c::CacheCandidate, mapfield) = CacheCandidate(newcand(c.c, mapfield))

function clearstate(s) end
clearstate(s::AbstractDict) = foreach(k -> delete!(s, k), keys(s))

cleanopt(o::T) where T = foreach(fn -> clearstate(getfield(o, fn)), fieldnames(T))
cleanopt(o::Flux.Optimiser) = foreach(cleanopt, o.os)
cleanopt(c::CandidateModel) = cleanopt(c.opt)
cleanopt(c::HostCandidate) = cleanopt(c.c)
cleanopt(c::CacheCandidate) = cleanopt(c.c)

"""
    randomlrscale(rfun = BoundedRandomWalk(-1.0, 1.0))

Return a function which scales the learning rate based on the output of `rfun`.

Intended use is to apply the same learning rate scaling for a whole population of models, e.g to have a global learning rate schedule.
"""
randomlrscale(rfun = BoundedRandomWalk(-1.0, 1.0)) = function(x...)
    newopt = ShieldedOpt(Descent(10^rfun(x...)))
    return AddOptimizerMutation(o -> newopt)
end

"""
    global_optimizer_mutation(pop, optfun)

Changes the optimizer of all candidates in `pop`.

The optimizer of each candidate in pop will be changed to `om(optc)` where `optc` is the current optimizer and `om = optfun(pop)`.

Intended to be used with `AfterEvolution` to create things like global learning rate schedules.

See `https://github.com/DrChainsaw/NaiveGAExperiments/blob/master/lamarckism/experiments.ipynb` for some hints as to why this might be needed.
"""
function global_optimizer_mutation(pop, optfun)
    om = optfun(pop)
    map(c -> newcand(c, optmap(om)), pop)
end
