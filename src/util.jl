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
        train!(model, partiter)
    end

    evolvepopulation(population)
end
```

"""
struct RepeatPartitionIterator{T, VS}
    base::Iterators.Stateful{T, VS}
    ntake::Int
end
RepeatPartitionIterator(base, nrep) = RepeatPartitionIterator(Iterators.Stateful(base), nrep)
RepeatPartitionIterator(base::Iterators.Stateful, nrep) = RepeatPartitionIterator(base, nrep)

function Base.iterate(itr::RepeatPartitionIterator, state=nothing)
    length(itr) == 0 && return nothing
    return Iterators.take(RepeatStatefulIterator(itr.base), itr.ntake), nothing
end

Base.length(itr::RepeatPartitionIterator) = ceil(Int, length(itr.base) / itr.ntake)
Base.eltype(itr::RepeatPartitionIterator{T}) where T = T

Base.IteratorSize(itr::RepeatPartitionIterator) = Base.IteratorSize(itr.base)
Base.IteratorEltype(itr::RepeatPartitionIterator) = Base.HasEltype()


"""
    cycle(itr, nreps)

An iterator that cycles through `itr nreps` times.
"""
Base.Iterators.cycle(itr, nreps) = Iterators.take(Iterators.cycle(itr), nreps * length(itr))

struct RepeatStatefulIterator{T, VS}
    base::Iterators.Stateful{T, VS}
    start::VS
    taken::Int
end
RepeatStatefulIterator(base) = RepeatStatefulIterator(base, base.nextvalstate, base.taken)

function Base.iterate(itr::RepeatStatefulIterator, reset=true)
    if reset
        itr.base.nextvalstate = itr.start
        itr.base.taken = itr.taken
    end
    valstate = iterate(itr.base)
    isnothing(valstate) && return valstate
    return valstate[1], false
end

Base.length(itr::RepeatStatefulIterator) = length(itr.base.itr) - itr.taken
Base.eltype(itr::RepeatStatefulIterator) = eltype(itr.base)

Base.IteratorSize(itr::RepeatStatefulIterator) = Base.IteratorSize(itr.base)
Base.IteratorEltype(itr::RepeatStatefulIterator) = Base.HasEltype()


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

Base.length(itr::BatchIterator) = ceil(Int, size(itr.base)[end] / itr.batchsize)
Base.size(itr::BatchIterator) = tuple(length(itr))

Base.IteratorSize(itr::BatchIterator) = Base.IteratorSize(itr.base)
Base.IteratorEltype(itr::BatchIterator) = Base.IteratorEltype(itr.base)

function Base.iterate(itr::BatchIterator, start=1)
    start > size(itr.base)[end] && return nothing
    stop = min(size(itr.base)[end], start + itr.batchsize-1)
    return batch(itr.base, start,stop), start+itr.batchsize
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

Base.print(io::IO, itr::BatchIterator) = print(io, "BatchIterator(size=$(size(itr.base)), batchsize=$(itr.batchsize))")

Flux.onehotbatch(itr::BatchIterator, labels) = MapIterator(x -> Flux.onehotbatch(x, labels), itr)

"""
    FlipIterator{T}
    FlipIterator(base, p::Real=0.5, dim::Int=1)

Flips data from `base` along dimension `dim` with probability `p`.
"""
struct FlipIterator{T}
    p::Probability
    dim::Int
    base::T
end
FlipIterator(base, p::Real=0.5, dim::Int=1) = FlipIterator(Probability(p), dim, base)

Base.length(itr::FlipIterator) = length(itr.base)
Base.size(itr::FlipIterator) = size(itr.base)

Base.IteratorSize(itr::FlipIterator) = Base.IteratorSize(itr.base)
Base.IteratorEltype(itr::FlipIterator) = Base.IteratorEltype(itr.base)

Base.iterate(itr::FlipIterator) = flip(itr, iterate(itr.base))
Base.iterate(itr::FlipIterator, state) = flip(itr, iterate(itr.base, state))

flip(itr::FlipIterator, valstate) = apply(itr.p) ? flip(itr.dim, valstate) : valstate
flip(::Nothing) = nothing
flip(dim::Integer, (data,state)::Tuple) = reverse(data, dims=dim), state



"""
    PersistentArray{T, N} <: AbstractArray{T, N}
    PersistentArray(savedir::String, nr::Integer, generator;suffix=".jls")
    PersistentArray(savedir::String, suffix::String, data::Array)

Simple persistent array. Can be created from serialized data and can be asked to persist its elements.

Note that once initialized, the array is not backed by the serialized data. Adding/deleting files is not reflected in data and vice versa.
"""
struct PersistentArray{T, N} <: AbstractArray{T, N}
    savedir::String
    suffix::String
    data::Array{T,N}
end
function PersistentArray(savedir::String, nr::Integer, generator;suffix=".jls")
    data = map(1:nr) do i
        filename = joinpath(savedir, "$i$suffix")
        isfile(filename) && return deserialize(filename)
        return generator(i)
    end
    return PersistentArray(savedir, suffix, data)
end
function persist(a::PersistentArray)
    mkpath(a.savedir)
    for (i, v) in enumerate(a)
        serialize(filename(a, i), v)
    end
end
filename(a::PersistentArray, i::Int) = joinpath(a.savedir, "$i$(a.suffix)")
Base.rm(a::PersistentArray; force=true, recursive=true) = rm(a.savedir, force=force, recursive=recursive)
Base.rm(a::PersistentArray, i::Int, force=false, recursive=true) = rm(filename(a,i), force=force, recursive=recursive)

Base.size(a::PersistentArray) = size(a.data)
Base.getindex(a::PersistentArray, i::Int) = getindex(a.data, i)
Base.getindex(a::PersistentArray, I::Vararg{Int, N}) where N = getindex(a.data, I...)
Base.setindex!(a::PersistentArray, v, i::Int) = setindex!(a.data, v, i)
Base.setindex!(a::PersistentArray, v, I::Vararg{Int, N}) where N = setindex!(a.data, v, I...)
Base.similar(a::PersistentArray, t::Type{S}, dims::Dims) where S = PersistentArray(a.savedir, a.suffix, similar(a.data,t, dims))
