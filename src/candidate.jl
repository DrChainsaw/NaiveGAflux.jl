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

"""
    model(c::AbstractCandidate; [default])

Return the model of candidate `c` if `c` has a model, `default` (which defaults to `nothing`) otherwise.
"""
model(::AbstractCandidate; default=nothing) = default

"""
    model(f, c::AbstractCandidate; [default])

Return the result of `f(`[`model(c; default)`]`)`. 
"""
model(f, c::AbstractCandidate; kwargs...) = f(model(c; kwargs...))

"""
    opt(c::AbstractCandidate; [default])

Return the optimizer of candidate `c` if `c` has an optimizer, `default` (which defaults to `nothing`) otherwise.
"""
opt(::AbstractCandidate; default=nothing) = default

"""
    lossfun(c::AbstractCandidate; [default])

Return the loss function of candidate `c` if `c` has a lossfunction, `default` (which defaults to `nothing`) otherwise.
"""
lossfun(::AbstractCandidate; default=nothing) = default

fitness(::AbstractCandidate; default=nothing) = default
generation(::AbstractCandidate; default=nothing) = default
trainiterator(::AbstractCandidate; default=nothing) = default
validationiterator(::AbstractCandidate; default=nothing) = default


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

model(c::AbstractWrappingCandidate) = model(wrappedcand(c))
# This is mainly for FileCandidate to allow for writing the graph back to disk after f is done
model(f, c::AbstractWrappingCandidate; kwargs...) = model(f, wrappedcand(c); kwargs...)
opt(c::AbstractWrappingCandidate; kwargs...) = opt(wrappedcand(c); kwargs...)
lossfun(c::AbstractWrappingCandidate; kwargs...) = lossfun(wrappedcand(c); kwargs...)
fitness(c::AbstractWrappingCandidate; kwargs...) = fitness(wrappedcand(c); kwargs...)
generation(c::AbstractWrappingCandidate; kwargs...) = generation(wrappedcand(c); kwargs...)
trainiterator(c::AbstractWrappingCandidate; kwargs...) = trainiterator(wrappedcand(c); kwargs...)
validationiterator(c::AbstractWrappingCandidate; kwargs...) = validationiterator(wrappedcand(c); kwargs...)

"""
    CandidateModel <: Candidate
    CandidateModel(model)

A candidate model which can be accessed by [`model(c)`](@ref) for `CandidateModel c`.
"""
struct CandidateModel{G} <: AbstractCandidate
    model::G
end

@functor CandidateModel

model(c::CandidateModel; kwargs...) = c.model

newcand(c::CandidateModel, mapfield) = CandidateModel(mapfield(c.model))

"""
    CandidateOptModel <: AbstractCandidate
    CandidateOptModel(optimizer, candidate)

A candidate adding an optimizer to another candidate. The optimizer is accessed by [`opt(c)`] for `CandidateOptModel c`.
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
    CandidateDataIterMap{T<:AbstractIteratorMap, C<:AbstractCandidate}     
    CandidateDataIterMap(itermap::AbstractIteratorMap, c::AbstractCandidate)

Maps training and validation data iterators using `iteratormap` for the wrapped candidate `c`.
    
Useful for searching for hyperparameters related to training and validation data, such as augmentation and batch size.

While one generally don't want to augment the validation data, it is useful to select the largest possible batch size
for validation for speed reasons.
"""
struct CandidateDataIterMap{T<:AbstractIteratorMap, C<:AbstractCandidate} <: AbstractWrappingCandidate
    map::T
    c::C
end

@functor CandidateDataIterMap

trainiterator(c::CandidateDataIterMap; kwargs...) = maptrain(c.map, trainiterator(wrappedcand(c); kwargs...))
validationiterator(c::CandidateDataIterMap; kwargs...) = mapvalidation(c.map, validationiterator(wrappedcand(c); kwargs...))

function newcand(c::CandidateDataIterMap, mapfield) 
    nc =  newcand(wrappedcand(c), mapfield)
    CandidateDataIterMap(apply_mapfield(mapfield, c.map, nc), nc)
end

# Just because BatchSizeIteratorMap needs the model to limit the batch sizes :(
apply_mapfield(f, x, ::AbstractCandidate) = f(x)

"""
    FileCandidate <: AbstractWrappingCandidate
    FileCandidate(c::AbstractCandidate) 

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

function callcand(f, c::FileCandidate, args...; kwargs...)
    # If timer is running we just stop and restart
    close(c.movetimer)

    # writelock means that the candidate is being moved to disk. We need to wait for it or risk data corruption
    islocked(c.writelock) && @warn "Try to access FileCandidate which is being moved to disk. Consider retuning of movedelay!"
    ret= lock(c.writelock) do
        f(MemPool.poolget(c.c), args...; kwargs...)
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

model(f, c::FileCandidate; kwargs...) = callcand(c) do cand
    model(f, cand; kwargs...)
end

newcand(c::FileCandidate, mapfield) = FileCandidate(callcand(newcand, c, mapfield), c.movedelay)

"""
    FittedCandidate{F, C} <: AbstractWrappingCandidate
    FittedCandidate(c::AbstractCandidate, fitnessvalue, generation)
    FittedCandidate(c::AbstractCandidate, fitnessfun::AbstractFitness, generation)

An `AbstractCandidate` with a computed fitness value. Will compute the fitness value if provided an `AbstractFitness`.

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

@functor FittedCandidate

fitness(c::FittedCandidate; default=nothing) = c.fitness
generation(c::FittedCandidate; default=nothing) = c.gen

# Some fitness functions make use of gen and fitness value :/ 
# This is a bit of an ugly inconsistency that I hope to fix one day is it is not beautiful that the fitness of a 
# new candidate is that same as its ancestor. It is the wanted behaviour for e.g. EwmaFitness though and 
# as of version 0.8.0 fitness does not longer carry any state. 
# Maybe I need to come up with a mechanism for that. As of now, alot of fitness strategies will not work properly 
# if they are not passed a FittedCandidate. Perhaps having some kind of fitness state container in each candidate?
newcand(c::FittedCandidate, mapfield) = FittedCandidate(c.gen, c.fitness, newcand(wrappedcand(c), mapfield))

nparams(c::AbstractCandidate) = model(nparams, c)
nparams(x) = mapreduce(prod ∘ size, +, params(x).order; init=0)

"""
    MapType{T, F1, F2}
    MapType{T}(match::F1, nomatch::F2)

Callable struct which returns `match(x)` if `x isa T`, otherwise returns `nomatch(x)`.

Main purpose is to ensure that an `AbstractMutation{T}` or `AbstractCrossover{T}` is
applied to fields which are subtypes of `T` when creating new candidates.
"""
struct MapType{T, F1, F2}
    match::F1
    nomatch::F2
    MapType{T}(match::F1, nomatch::F2) where {T,F1, F2} = new{T,F1,F2}(match, nomatch)
end

(a::MapType{T1})(x::T2) where {T1, T2<:T1} = a.match(x)
(a::MapType)(x) = a.nomatch(x)

MapType(match::AbstractMutation{T}, nomatch) where T = MapType{T}(match, nomatch)
MapType(match::AbstractMutation{CompGraph}, nomatch) = MapType{CompGraph}(match ∘ deepcopy, nomatch) 

function MapType(c::AbstractCrossover{CompGraph}, (c1, c2), (nomatch1, nomatch2))
    g1 = model(c1)
    g2 = model(c2)

    release!(c1)
    release!(c2)

    g1, g2 = c((deepcopy(g1), deepcopy(g2)))
    return MapType{CompGraph}(Returns(g1), nomatch1), MapType{CompGraph}(Returns(g2), nomatch2)
end

function MapType(c::AbstractCrossover{FluxOptimizer}, (c1, c2), (nomatch1, nomatch2))
    o1 = opt(c1)
    o2 = opt(c2)

    o1n, o2n = c((o1, o2))
    return MapType{FluxOptimizer}(Returns(o1n), nomatch1), MapType{FluxOptimizer}(Returns(o2n), nomatch2)
end

"""
    MapCandidate{T, F} 
    MapCandidate(mutations, mapothers::F)

Return a callable struct which maps `AbstractCandidate`s to new `AbstractCandidate`s through `mutations` which is a tuple of 
`AbstractMutation`s or `AbstractCrossover`s. 

Basic purpose is to combine multiple mutations operating on different types into a single mapping function which creates new 
candidates from existing candidates. 

When called as a function with an `AbstractCandidate c` as input, it will map fields `f` in `c` (recursively through any 
wrapped candidates of `c`) satisfying `typeof(f) <: MT` through `m(f)` where `m <: AbstractMutation{MT}` in `mutations`.

All other fields are mapped through the function `mapothers` (default `deepcopy`).

For instance, if `e = MapCandidate(m1, m2)` where `m1 isa AbstractMutation{CompGraph}` and `m2 isa 
AbstractMutation{FluxOptimizer}` then `e(c)` where `c` is a `CandidateOptModel` will create a new `CandidateOptModel`where 
the new model is `m1(model(c))` and the new optimizer is `m2(opt(c))`  

When called as a function with a tuple of two `AbstractCandidate`s as input it will similarly apply crossover between the 
two candidates, returning two new candidates.

Note that all `mutations` must be either `AbstractMutation`s or `AbstractCrossover`s as the resulting function either works
on a single candidate or a pair of candidates.

Furthermore, all `mutations` must operate on different types, i.e there must not be two `AbstractMutation{T}` (or `
AbstractCrossover{T}`) for any type `T`.

Intended use is together with [`EvolveCandidates`](@ref).
"""
struct MapCandidate{T, F}
    mutations::T
    mapothers::F
end

MapCandidate(mutations::AbstractMutation...; mapothers=deepcopy) = MapCandidate(mutations, mapothers)
MapCandidate(mutation::AbstractMutation, mapothers) = MapCandidate(tuple(mutation), mapothers)

function MapCandidate(crossovers::NTuple{N, AbstractCrossover}, mapothers::F) where {N, F} 
    _validate_mutations(crossovers)
    MapCandidate{typeof(crossovers), F}(crossovers, mapothers)
end
function MapCandidate(mutations::NTuple{N, AbstractMutation}, mapothers) where N
    _validate_mutations(mutations)
    mapc = foldr(mutations; init=mapothers) do match, nomatch
        MapType(match, nomatch)
    end
    MapCandidate(mapc, mapothers)
end

function _validate_mutations(mutations)
    seentypes = Set()
    iscrossover = first(mutations) isa AbstractCrossover
    for m in mutations
        _validate_unique_type!(seentypes, m)
        (iscrossover == (m isa AbstractCrossover)) || throw(ArgumentError("Can't mix crossover and mutation in same function! Use different functions instead"))
    end
end

function _validate_unique_type!(seen, ::AbstractMutation{T}) where T
    T in seen && throw(ArgumentError("Got mutation of duplicate type $(T)!"))
    push!(seen, T)
end

(e::MapCandidate{<:MapType})(c) = newcand(c, e.mutations)

function (e::MapCandidate{<:NTuple{N, AbstractCrossover}, F})((c1,c2)) where {N,F}
    # Bleh, CGA to avoid a closure here
    mapc1, mapc2 = let c1 = c1, c2 = c2
        # Whats happening here is probably far from obvious, so here goes:
        # MapType returns a Tuple of MapTypes when called with an AbstractCrossover
        # This is because we create new models, optimisers etc from from both c1 
        # and c2 simultaneously, but when creating candidates we create one new 
        # candidate from another new candidate. 
        foldr(e.mutations; init=(e.mapothers, e.mapothers)) do match, nomatch
            # nomatch is always a Tuple while match is always an AbstractCrossover
            MapType(match, (c1,c2), nomatch)
        end
    end

    return newcand(c1, mapc1), newcand(c2, mapc2)
end


function mapcandidate(mapgraph, mapothers=deepcopy)
    mapfield(g::CompGraph) = mapgraph(g)
    mapfield(f) = mapothers(f)
    # Replace with fmap?
    # Maybe not, because we don't want to descend into models?
    return c -> newcand(c, mapfield)
end

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
