"""
    AbstractCandidate

Abstract base type for canidates
"""
abstract type AbstractCandidate end

"""
    reset!(c::AbstractCandidate)

Reset state of `c`. Typically needs to be called after evolution to clear old fitness computations.
"""
reset!(c::AbstractCandidate) = reset!(wrappedcand(c))
Base.Broadcast.broadcastable(c::AbstractCandidate) = Ref(c)

wrappedcand(c::AbstractCandidate) = c.c
graph(c::AbstractCandidate) = graph(wrappedcand(c))
# This is mainly for FileCandidate to allow for writing the graph back to disk after f is done
graph(c::AbstractCandidate, f) = graph(wrappedcand(c), f)

opt(c::AbstractCandidate) = opt(wrappedcand(c))

"""
    CandidateModel <: Candidate
    CandidateModel(model, optimizer, lossfunction, fitness)

A candidate model consisting of a `CompGraph`, an optimizer a lossfunction and a fitness method.
"""
struct CandidateModel{G,O,L,F} <: AbstractCandidate
    graph::G
    opt::O
    lossfun::L
    fitness::F
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

graph(model::CandidateModel, f=identity) = f(model.graph)

opt(c::CandidateModel) = c.opt

wrappedcand(c::CandidateModel) = error("CandidateModel does not wrap any candidate! Check your base case!")


"""
    HostCandidate <: AbstractCandidate
    HostCandidate(c::AbstractCandidate)

Keeps `c` in host memory and transfers to GPU when training or calculating fitness.
"""
struct HostCandidate{C} <: AbstractCandidate
    c::C
end

Flux.@functor HostCandidate

function Flux.train!(c::HostCandidate, data)
    NaiveNASflux.forcemutation(graph(c)) # Optimization: If there is a LazyMutable somewhere, we want it to do its thing now so we don't end up copying the model to the GPU only to then trigger another copy when the mutations are applied.
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

const gpu_gc = if CuArrays.functional()
    function()
        GC.gc()
        CuArrays.reclaim()
    end
else
    () -> nothing
end

"""
    FileCandidate

Keeps `c` on disk when not in use and just maintains its [`DRef`](@ref).

Experimental feature. May not work as intended!
"""
mutable struct FileCandidate{C, R<:Real, L<:Base.AbstractLock} <: AbstractCandidate
    c::MemPool.DRef
    movedelay::R
    movetimer::Timer
    writelock::L
    function FileCandidate(c::C, movedelay::R) where {C, R}
        cref = MemPool.poolset(c)
        writelock = ReentrantLock()
        movetimer = asynctodisk(cref, movedelay, writelock)
        fc = new{C, R, typeof(writelock)}(cref, movedelay, movetimer, writelock)
        finalizer(gfc -> MemPool.pooldelete(gfc.c), fc)
        return fc
     end
end
FileCandidate(c::AbstractCandidate) = FileCandidate(c, 0.5)

function callcand(f, c::FileCandidate, args...)
    # If timer is running we just stop and restart
    close(c.movetimer)

    # writelock means that the candidate is being moved to disk. We need to wait for it or risk data corruption
    islocked(c.writelock) && @warn "Try to access FileCandidate which is being moved to disk. Consider retuning of movedelay!"
    ret = lock(c.writelock) do
        f(MemPool.poolget(c.c), args...)
    end
    c.movetimer = asynctodisk(c.c, c.movedelay, c.writelock)
    return ret
end

function asynctodisk(r::MemPool.DRef, delay, writelock)
    return Timer(delay) do _
        candtodisk(r, writelock)
    end
end
# Function exists outside of timer for testing purposes basically
function candtodisk(r::MemPool.DRef, writelock)
    lock(writelock) do
        # Not sure why this can even happen, but it does. Bug lurking...
        inmem = MemPool.with_datastore_lock() do
            haskey(MemPool.datastore, r.id) && MemPool.isinmemory(MemPool.datastore[r.id])
        end
        inmem || return

        # Remove file if it exists or else MemPool won't move it
        rm(MemPool.default_path(r); force=true)
        MemPool.movetodisk(r)
    end
end


function Serialization.serialize(s::AbstractSerializer, c::FileCandidate)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    serialize(s, FileCandidate)
    callcand(cc -> serialize(s,cc), c)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{FileCandidate})
    wrapped = deserialize(s)
    return FileCandidate(wrapped)
end

# Mutation needs to be enabled here?
Flux.functor(c::FileCandidate) = callcand(Flux.functor, c)

Flux.train!(c::FileCandidate, data) = callcand(Flux.train!, c, data)
fitness(c::FileCandidate) = callcand(fitness, c)

reset!(c::FileCandidate) = callcand(reset!, c)
wrappedcand(c::FileCandidate) = MemPool.poolget(c.c)
graph(c::FileCandidate, f) = callcand(graph, c, f)
opt(c::FileCandidate) = callcand(opt, c)

"""
    CacheCandidate <: AbstractCandidate
    CacheCandidate(c::AbstractCandidate)

Caches fitness values produced by `c` until `reset!` is called.

Useful with `HostCandidate` to prevent models from being pushed to/from GPU just to fetch fitness values.
"""
mutable struct CacheCandidate{C} <: AbstractCandidate
    fitnesscache
    c::C
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

nparams(c::AbstractCandidate) = nparams(graph(c))
nparams(g::CompGraph) = mapreduce(prod ∘ size, +, params(g).order)

"""
    evolvemodel(m::AbstractMutation{CompGraph}, mapothers=deepcopy)
    evolvemodel(m::AbstractMutation{CompGraph}, om::AbstractMutation{FluxOptimizer}, mapothers=deepcopy)

Return a function which maps a `AbstractCandidate c1` to a new `AbstractCandidate c2` where any `CompGraph`s `g` in `c1` will be m(copy(g))` in `c2`. Same principle is applied to any optimisers if `om` is present.

All other fields are mapped through the function `mapothers` (default `deepcopy`).

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

"""
    evolvemodel(m::AbstractCrossover{CompGraph}, mapothers1=deepcopy, mapothers2=deepcopy)
    evolvemodel(m::AbstractCrossover{CompGraph}, om::AbstractCrossover{FluxOptimizer}, mapothers1=deepcopy, mapothers2=deepcopy)

Return a function which maps a tuple of `AbstractCandidate`s `(c1,c2)` to two new candidates `c1', c2'` where any `CompGraph`s `g1` and `g2` in `c1` and `c2` respectively will be `g1', g2' = m((copy(g1), copy(g2)))` in `c1'` and `c2'` respectively. Same principle applies to any optimisers if `om` is present.

All other fields in `c1` will be mapped through the function `mapothers1` and likewise for `c2` and `mapothers2`.

Intended use is together with [`PairCandidates`](@ref) and [`EvolveCandidates`](@ref).
"""
evolvemodel(m::AbstractCrossover{CompGraph}, mapothers1=deepcopy, mapothers2=deepcopy) = (c1, c2)::Tuple -> begin
    # This allows FileCandidate to write the graph back to disk as we don't want to mutate the orignal candidate.
    # Perhaps align single individual mutation to this pattern for consistency?
    g1 = graph(c1, identity)
    g2 = graph(c2, identity)

    g1, g2 = m((copy(g1), copy(g2)))

    return mapcandidate(g -> g1, mapothers1)(c1), mapcandidate(g -> g2, mapothers2)(c2)
end

evolvemodel(m::AbstractCrossover{CompGraph}, om::AbstractCrossover{FluxOptimizer}, mapothers1=deepcopy, mapothers2=deepcopy) = (c1,c2)::Tuple -> begin
    o1 = opt(c1)
    o2 = opt(c2)

    o1n, o2n = om((o1, o2))

    return evolvemodel(m, optmap(o -> o1n, mapothers1), optmap(o -> o2n, mapothers2))((c1,c2))
end


function mapcandidate(mapgraph, mapothers=deepcopy)
    mapfield(g::CompGraph) = mapgraph(g)
    mapfield(f) = mapothers(f)
    return c -> newcand(c, mapfield)
end

newcand(c::CandidateModel, mapfield) = CandidateModel(map(mapfield, getproperty.(c, fieldnames(CandidateModel)))...)
newcand(c::HostCandidate, mapfield) = HostCandidate(newcand(c.c, mapfield))
newcand(c::CacheCandidate, mapfield) = CacheCandidate(newcand(c.c, mapfield))
newcand(c::FileCandidate, mapfield) = FileCandidate(callcand(newcand, c, mapfield), c.movedelay)

function clearstate(s) end
clearstate(s::AbstractDict) = foreach(k -> delete!(s, k), keys(s))

cleanopt(o::T) where T = foreach(fn -> clearstate(getfield(o, fn)), fieldnames(T))
cleanopt(o::ShieldedOpt) = cleanopt(o.opt)
cleanopt(o::Flux.Optimiser) = foreach(cleanopt, o.os)
cleanopt(c::CandidateModel) = cleanopt(c.opt)
cleanopt(c::FileCandidate) = callcand(cleanopt, c.c)
cleanopt(c::AbstractCandidate) = cleanopt(wrappedcand(c))

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
