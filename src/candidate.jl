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

release!(::AbstractCandidate) = nothing
hold!(::AbstractCandidate) = nothing
graph(c::AbstractCandidate, f=identity; default=nothing) = f(default)
opt(c::AbstractCandidate; default=nothing) = default
lossfun(c::AbstractCandidate; default=nothing) = default
fitness(c::AbstractCandidate; default=nothing) = default
generation(c::AbstractCandidate; default=nothing) = default


wrappedcand(::T) where T <: AbstractCandidate = error("$T does not wrap any candidate! Check your base case!")

"""
    AbstractWrappingCandidate <: AbstractCandidate

Abstract base type for candidates which wrap other candidates.

Should implement `wrappedcand(c)` to get lots of stuff handled automatically.
"""
abstract type AbstractWrappingCandidate <: AbstractCandidate end

wrappedcand(c::AbstractWrappingCandidate) = c.c

release!(c::AbstractWrappingCandidate) = release!(wrappedcand(c))
hold!(c::AbstractWrappingCandidate) = hold!(wrappedcand(c))

graph(c::AbstractWrappingCandidate) = graph(wrappedcand(c))
# This is mainly for FileCandidate to allow for writing the graph back to disk after f is done
graph(c::AbstractWrappingCandidate, f; kwargs...) = graph(wrappedcand(c), f; kwargs...)
opt(c::AbstractWrappingCandidate; kwargs...) = opt(wrappedcand(c); kwargs...)
lossfun(c::AbstractWrappingCandidate; kwargs...) = lossfun(wrappedcand(c); kwargs...)
fitness(c::AbstractWrappingCandidate; kwargs...) = fitness(wrappedcand(c); kwargs...)
generation(c::AbstractWrappingCandidate; kwargs...) = generation(wrappedcand(c); kwargs...)


"""
    CandidateModel <: Candidate
    CandidateModel(model)

A candidate model consisting of a `CompGraph`, an optimizer a lossfunction and a fitness method.
"""
struct CandidateModel{G} <: AbstractCandidate
    graph::G
end

Functors.@functor CandidateModel

graph(c::CandidateModel, f=identity; kwargs...) = f(c.graph)

newcand(c::CandidateModel, mapfield) = CandidateModel(map(mapfield, getproperty.(c, fieldnames(CandidateModel)))...)

"""
    CandidateOptModel <: AbstractCandidate
    CandidateOptModel(candidate, optimizer)

A candidate adding an optimizer to another candidate.
"""
struct CandidateOptModel{O, C <: AbstractCandidate} <: AbstractWrappingCandidate
    opt::O
    c::C
end
CandidateOptModel(opt, g::CompGraph) = CandidateOptModel(opt, CandidateModel(g))

function Functors.functor(::Type{<:CandidateOptModel}, c) 
    return (opt=c.opt, c=c.c), function ((newopt, newc),)
        # Optimizers are stateful and having multiple candidates pointing to the same instance is downright scary
        # User would need to go out of its way to make it the same instance (e.g. by using a wrapper type and dispatch on it in CandidateOptModel)
        # Hope that we get stateless optimizers soon
        if newopt === c.opt
            newopt = deepcopy(cleanopt!(newopt))
        end
        CandidateOptModel(newopt, newc)
    end
end

opt(c::CandidateOptModel; kwargs...) = c.opt 

newcand(c::CandidateOptModel, mapfield) = CandidateOptModel(mapfield(c.opt), newcand(wrappedcand(c), mapfield))

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
    hold::Bool
    function FileCandidate(c::C, movedelay::R, hold=false) where {C, R}
        cref = MemPool.poolset(c)
        writelock = ReentrantLock()
        movetimer = hold ? asynctodisk(cref, movedelay, writelock) : Timer(movedelay)
        fc = new{C, R, typeof(writelock)}(cref, movedelay, movetimer, writelock, hold)
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
    ret= lock(c.writelock) do
        f(MemPool.poolget(c.c), args...)
    end
    if !c.hold
        c.movetimer = asynctodisk(c.c, c.movedelay, c.writelock)
    end
    return ret
end

candinmem(c::FileCandidate) = candinmem(c.c)
candinmem(r::MemPool.DRef) =  MemPool.with_datastore_lock() do
    haskey(MemPool.datastore, r.id) && MemPool.isinmemory(MemPool.datastore[r.id])
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
        candinmem(r) || return
        # Remove file if it exists or else MemPool won't move it
        rm(MemPool.default_path(r); force=true)
        MemPool.movetodisk(r)
    end
end

function release!(c::FileCandidate)
    c.hold = false
    release!(wrappedcand(c))
    close(c.movetimer)
    c.movetimer = asynctodisk(c.c, c.movedelay, c.writelock)
    return nothing
end

# wrappedcand will do the work for us to actually hold 
hold!(c::FileCandidate) = hold!(wrappedcand(c))

function Serialization.serialize(s::AbstractSerializer, c::FileCandidate)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    serialize(s, FileCandidate)
    callcand(cc -> serialize(s,cc), c)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{FileCandidate})
    wrapped = deserialize(s)
    return FileCandidate(wrapped)
end

function Functors.functor(::Type{<:FileCandidate}, c) 
    # we can allow the wrapped candidate to be moved to disk while user decides what to do with the
    return (c=callcand(identity, c), movedelay=c.movedelay), function(xs)
        FileCandidate(xs..., c.hold)
    end
end

function wrappedcand(c::FileCandidate) 
    c.hold = true
    callcand(identity, c)
end
# Could also implement getproperty using wrappedcand/callcand, but there is no need for it so not gonna bother now

graph(c::FileCandidate, f) = callcand(graph, c, f)

newcand(c::FileCandidate, mapfield) = FileCandidate(callcand(newcand, c, mapfield), c.movedelay)

"""
    struct FittedCandidate{F, C} <: AbstractWrappingCandidate

An `AbstractCandidate` with a computed fitness value. 

Basically a container for results so that fitness does not need to be recomputed to e.g. check stopping conditions. 

Also useful for fitness smoothing, e.g. with `EwmaFitness` as it gives access to previous fitness value.
"""
struct FittedCandidate{F, C <: AbstractCandidate} <: AbstractWrappingCandidate
    gen::Int
    fitness::F
    c::C
end
FittedCandidate(c::AbstractCandidate, f::AbstractFitness, gen) = FittedCandidate(gen, fitness(f, c), c)
FittedCandidate(c::FittedCandidate, f::AbstractFitness, gen) = FittedCandidate(gen, fitness(f, c), wrappedcand(c))

Functors.@functor FittedCandidate

fitness(c::FittedCandidate; default=nothing) = c.fitness
generation(c::FittedCandidate; default=nothing) = c.gen

# Some fitness functions make use of gen and fitness value :/ 
# This is a bit of an ugly inconsistency that I hope to fix one day is it is not beautiful that the fitness of a 
# new candidate is that same as its ancestor. It is the wanted behaviour for e.g. EwmaFitness though and 
# as of version 0.8.0 fitness does not longer carry any state. 
# Maybe I need to come up with a mechanism for that. As of now, alot of fitness strategies will not work properly 
# if they are not passed a FittedCandidate. Perhaps having some kind of fitness state container in each candidate?
newcand(c::FittedCandidate, mapfield) = FittedCandidate(c.gen, c.fitness, newcand(wrappedcand(c), mapfield))

nparams(c::AbstractCandidate) = graph(c, nparams)
nparams(x) = mapreduce(prod ∘ size, +, params(x).order; init=0)

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
    # TODO: Replace with fmap now that we fully support Functors?
    return c -> newcand(c, mapfield)
end

function clearstate!(s) end
clearstate!(s::AbstractDict) = empty!(s)

cleanopt!(o::T) where T = (foreach(fn -> clearstate!(getfield(o, fn)), fieldnames(T)); return o)
cleanopt!(o::ShieldedOpt) = (cleanopt!(o.opt); return o)
cleanopt!(o::Flux.Optimiser) = (foreach(cleanopt!, o.os); return o)

"""
    randomlrscale(rfun = BoundedRandomWalk(-1.0, 1.0))

Return a function which scales the learning rate based on the output of `rfun`.

Intended use is to apply the same learning rate scaling for a whole population of models, e.g to have a global learning rate schedule.
"""
randomlrscale(rfun = BoundedRandomWalk(-1.0, 1.0)) = function(x...)
    newopt = ShieldedOpt(Flux.Descent(10^rfun(x...)))
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
