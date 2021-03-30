"""
    AbstractCandidate

Abstract base type for candidates
"""
abstract type AbstractCandidate end

"""
    AbstractFitness

Abstract type for fitness functions
"""
abstract type AbstractFitness end


Base.Broadcast.broadcastable(c::AbstractCandidate) = Ref(c)

fitness(c::AbstractCandidate, f::AbstractFitness, gen) = fitness(f, c, gen)

graph(c::AbstractCandidate, f=identity; default=nothing) = f(default)
opt(c::AbstractCandidate; default=nothing) = default
lossfun(c::AbstractCandidate; default=nothing) = default

wrappedcand(::T) where T <: AbstractCandidate = error("$T does not wrap any candidate! Check your base case!")

"""
    AbstractWrappingCandidate <: AbstractCandidate

Abstract base type for candidates which wrap other candidates.

Should implement `wrappedcand(c)` to get lots of stuff handled automatically.
"""
abstract type AbstractWrappingCandidate <: AbstractCandidate end

"""
    reset!(c::AbstractCandidate)

Reset state of `c`. Typically needs to be called after evolution to clear old fitness computations.
"""
reset!(c::AbstractWrappingCandidate) = reset!(wrappedcand(c))

wrappedcand(c::AbstractWrappingCandidate) = c.c
graph(c::AbstractWrappingCandidate) = graph(wrappedcand(c))
# This is mainly for FileCandidate to allow for writing the graph back to disk after f is done
graph(c::AbstractWrappingCandidate, f; kwargs...) = graph(wrappedcand(c), f; kwargs...)
opt(c::AbstractWrappingCandidate; kwargs...) = opt(wrappedcand(c); kwargs...)
lossfun(c::AbstractWrappingCandidate; kwargs...) = lossfun(wrappedcand(c); kwargs...)

"""
    CandidateModel <: Candidate
    CandidateModel(model)

A candidate model consisting of a `CompGraph`, an optimizer a lossfunction and a fitness method.
"""
struct CandidateModel{G} <: AbstractCandidate
    graph::G
end

Flux.@functor CandidateModel

graph(c::CandidateModel, f=identity; kwargs...) = f(c.graph)

"""
    CandidateOptModel <: AbstractCandidate
    CandidateOptModel(model)

A candidate model consisting of a `CompGraph` and an optimizer.
"""
struct CandidateOptModel{G,O} <: AbstractCandidate
    graph::G
    opt::O
end

Flux.@functor CandidateOptModel

graph(c::CandidateOptModel, f=identity; kwargs...) = f(c.graph)
opt(c::CandidateOptModel; kwargs...) = c.opt 

"""
    FileCandidate <: AbstractWrappingCandidate

Keeps `c` on disk when not in use and just maintains its [`DRef`](@ref).

Experimental feature. May not work as intended!
"""
mutable struct FileCandidate{C, R<:Real, L<:Base.AbstractLock} <: AbstractWrappingCandidate
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

fitness(c::FileCandidate, f::AbstractFitness, gen) = callcand(fitness, c, f, gen)

reset!(c::FileCandidate) = callcand(reset!, c)
wrappedcand(c::FileCandidate) = MemPool.poolget(c.c)
graph(c::FileCandidate, f) = callcand(graph, c, f)
opt(c::FileCandidate) = callcand(opt, c)

"""
    struct FittedCandidate{F, C} <: AbstractWrappingCandidate

An `AbstractCandidate` with a computed fitness value. 

Basically a container for results so that fitness does not need to be recomputed to e.g. check stopping conditions. 
"""
struct FittedCandidate{F, C} <: AbstractWrappingCandidate
    gen::Int
    fitness::F
    c::C
end
FittedCandidate(c::AbstractCandidate, f::AbstractFitness, gen) = FittedCandidate(gen, fitness(c, f, gen), c)
FittedCandidate(c::FittedCandidate, f::AbstractFitness, gen) = FittedCandidate(wrappedcand(f), f ,gen)

fitness(c::FittedCandidate) = c.fitness


nparams(c::AbstractCandidate) = graph(c, nparams)
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
newcand(c::CandidateOptModel, mapfield) = CandidateOptModel(map(mapfield, getproperty.(c, fieldnames(CandidateOptModel)))...)
newcand(c::FileCandidate, mapfield) = FileCandidate(callcand(newcand, c, mapfield), c.movedelay)
newcand(c::FittedCandidate, mapfield) = newcand(wrappedcand(c), mapfield)

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
