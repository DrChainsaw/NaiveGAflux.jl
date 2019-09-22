"""
    Probability
Represents a probability that something (typically mutation) will happen.

Possible to specify RNG implementation. If not specified, `GLOBAL_RNG` will be used
"""
struct Probability
    p::Real
    rng::AbstractRNG
    function Probability(p::Real, rng)
         @assert 0 <= p <= 1
         return new(p, rng)
    end
end
Probability(p::Real) = Probability(p, rng_default)
Probability(p::Integer) = Probability(p, rng_default)
Probability(p::Integer, rng) = Probability(p / 100.0, rng)

"""
    apply(p::Probability)

Return true with a probability of ´p.p´ (subject to `p.rng` behaviour).
"""
apply(p::Probability) = rand(p.rng) < p.p

"""
    apply(f, p::Probability)

Call `f` with probability `p.p` (subject to `p.rng` behaviour).
"""
apply(f::Function, p::Probability) =  apply(p) && f()

"""
    MutationShield <: DecoratingTrait

Shields its associated vertex from being selected for mutation.

Note that vertex might still be modified if an adjacent vertex is mutated in a way which propagates to a shielded vertex.
"""
struct MutationShield <:DecoratingTrait
    t::MutationTrait
end
NaiveNASlib.base(t::MutationShield) = t.t
allow_mutation(v::AbstractVertex) = allow_mutation(trait(v))
allow_mutation(t::DecoratingTrait) = allow_mutation(base(t))
allow_mutation(::MutationTrait) = true
allow_mutation(::Immutable) = false
allow_mutation(::MutationShield) = false

"""
    AbstractVertexSelection

Abstract type for determining how to select vertices from a `CompGraph`
"""
abstract type AbstractVertexSelection end

"""
    AllVertices

Select all vertices in `g`.
"""
struct AllVertices <:AbstractVertexSelection end
select(::AllVertices, g::CompGraph) = vertices(g)

"""
    FilterMutationAllowed

Filters out only the vertices for which mutation is allowed from another selection.
"""
struct FilterMutationAllowed <:AbstractVertexSelection
    s::AbstractVertexSelection
end
FilterMutationAllowed() = FilterMutationAllowed(AllVertices())
select(s::FilterMutationAllowed, g::CompGraph) = filter(allow_mutation, select(s.s, g))

"""
    ApplyIf <: DecoratingTrait
    ApplyIf(predicate::Function, apply::Function, base::MutationTrait)

Enables calling `apply(v)` for an `AbstractVertex v` which has this trait if 'predicate(v) == true'.

Motivating use case is to have a way to remove vertices which have ended up as noops, e.g. element wise and concatenation vertices with a single input or identity activation functions.
"""
struct ApplyIf <: DecoratingTrait
    predicate::Function
    apply::Function
    base::MutationTrait
end
RemoveIfSingleInput(t) = ApplyIf(v -> length(inputs(v)) == 1, remove!, t)
NaiveNASlib.base(t::ApplyIf) = t.base

check_apply(g::CompGraph) = foreach(check_apply, vertices(g))
check_apply(v::AbstractVertex) = check_apply(trait(v), v)
check_apply(t::DecoratingTrait, v) = check_apply(base(t), v)
check_apply(t::ApplyIf, v) = t.predicate(v) && t.apply(v)
function check_apply(t, v) end


"""
    RepeatPartitionIterator
    RepeatPartitionIterator(base, nrep)

Iteratates over iterators of a subset of size `nrep` elements in `base`.

Generally useful for training all models in a population with the same data in each evolution epoch.

Tailored for situations where iterating over models is more expensive than iterating over data, for example if candidates are stored in host RAM or on disk.

Example training loop:
```julia

for partiter in iter

    for model in population
        for data in partiter
            train!(model, data)
        end
    end

    evolvepopulation(population)
end
```

"""
mutable struct RepeatPartitionIterator{T}
    base::T
    ntake::Int
end
RepeatPartitionIterator(base, nrep) = RepeatPartitionIterator(base, nrep)

function Base.iterate(itr::RepeatPartitionIterator, ndrop=0)
    hasmore(ndrop, itr.base) && return nothing
    return Iterators.take(Iterators.drop(itr.base, ndrop), itr.ntake), ndrop + itr.ntake
end

Base.length(itr::RepeatPartitionIterator) = ceil(Int, length(itr.base) / itr.ntake)
Base.eltype(itr::RepeatPartitionIterator{T}) where T = T

Base.IteratorSize(itr::RepeatPartitionIterator) = Base.IteratorSize(itr.base)
Base.IteratorEltype(itr::RepeatPartitionIterator) = Base.HasEltype()

hasmore(than, itr) = hasmore(Base.IteratorSize(itr), than, itr)
hasmore(::Union{Base.HasLength, Base.HasShape}, than, itr) = than >= length(itr)
hasmore(::Base.IsInfinite, than, itr) = true
# Can maybe work by just taking the first item, but chances are this iterator will anyway be very inefficient
hasmore(::Base.SizeUnknown, than, itr) = error("Iterator can not work with unkown size!")

"""
    cycle(itr, nreps)

An iterator that cycles through `itr nreps` times.
"""
Base.Iterators.cycle(itr, nreps) = Iterators.take(Iterators.cycle(itr), nreps * length(itr))

"""
    MapIterator{F, T}
    MapIterator(f::F, base::T)

Return an iterator over `f` of the values from `base`.
"""
struct MapIterator{F, T}
    f::F
    base::T
end

"""
    Flux.onehotbatch(itr::Base.Iterators.PartitionIterator)

Return an iterator over [`Flux.onehotbatch`](@ref) of the values from `itr`.
"""
Flux.onehotbatch(itr::Base.Iterators.PartitionIterator, labels) = MapIterator(x -> Flux.onehotbatch(x, labels), itr)

"""
    GpuIterator(itr)

Return an iterator which sends values from `itr` to the GPU.
"""
GpuIterator(itr) = MapIterator(gpuitr, itr)
gpuitr(a) = Flux.gpu(a)
gpuitr(a::SubArray) = gpuitr(collect(a))
gpuitr(a::Tuple) = gpuitr.(a)


function Base.iterate(itr::MapIterator)
    valstate = iterate(itr.base)
    isnothing(valstate) && return valstate
    return itr.f(valstate[1]), valstate[2]
end

function Base.iterate(itr::MapIterator, state)
    valstate = iterate(itr.base, state)
    isnothing(valstate) && return valstate
    return itr.f(valstate[1]), valstate[2]
end

Base.length(itr::MapIterator) = length(itr.base)
Base.size(itr::MapIterator) = size(itr.base) # Not guaranteed to be true depending on what f does...

Base.IteratorSize(itr::MapIterator) = Base.IteratorSize(itr.base)
Base.IteratorEltype(itr::MapIterator) = Base.EltypeUnknown() # Don't know what f does...


"""
    BatchIterator{T}
    BatchIterator(base, batchsize)

Return an iterator which iterates `batchsize` samples along the last dimension of `base`.
"""
struct BatchIterator{T}
    base::T
    batchsize::Int
end
#BatchIterator(base::T, batchsize) where T <: AbstractArray = BatchIterator{T}(base, batchsize)

Base.length(itr::BatchIterator) = ceil(Int, size(itr.base)[end] / itr.batchsize)
Base.size(itr::BatchIterator) = tuple(length(itr))

Base.IteratorSize(itr::BatchIterator) = Base.IteratorSize(itr.base)
Base.IteratorEltype(itr::BatchIterator) = Base.IteratorEltype(itr.base)

function Base.iterate(itr::BatchIterator, start=1)
    start > size(itr.base)[end] && return nothing
    stop = min(size(itr.base)[end], start + itr.batchsize-1)
    return batch(itr.base, start,stop), start+stop
end

## I *think* speed matters here, so...
batch(a::AbstractArray{T,1}, start, stop) where T = view(a, start:stop)
batch(a::AbstractArray{T,2}, start, stop) where T = view(a, :,start:stop)
batch(a::AbstractArray{T,3}, start, stop) where T = view(a, :,:,start:stop)
batch(a::AbstractArray{T,4}, start, stop) where T = view(a, :,:,:,start:stop)
function batch(a, start, stop)
    access = collect(UnitRange, axes(itr.base))
    access[itr.dim] = start:stop
    return view(itr.base, access...)
end

Flux.onehotbatch(itr::BatchIterator, labels) = MapIterator(x -> Flux.onehotbatch(x, labels), itr)
