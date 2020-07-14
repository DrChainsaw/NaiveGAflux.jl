"""
    RepeatPartitionIterator
    RepeatPartitionIterator(base, nrep)

Iteratates over iterators of a subset of size `nrep` elements in `base`.

Generally useful for training all models in a population with the same data in each evolution epoch.

Tailored for situations where iterating over models is more expensive than iterating over data, for example if candidates are stored in host RAM or on disk and needs to be transferred to the GPU for training.

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

function Base.iterate(itr::RepeatPartitionIterator, reset=true)
    if reset
        Iterators.reset!(itr.base, itr.base.itr)
    end
    length(itr) == 0 && return nothing
    return Iterators.take(RepeatStatefulIterator(itr.base), itr.ntake), false
end

Base.length(itr::RepeatPartitionIterator) = ceil(Int, length(itr.base) / itr.ntake)
Base.eltype(itr::RepeatPartitionIterator{T}) where T = T
Base.size(itr::RepeatPartitionIterator) = size(itr.base.itr)


Base.IteratorSize(itr::RepeatPartitionIterator) = Base.IteratorSize(itr.base.itr)
Base.IteratorEltype(itr::RepeatPartitionIterator) = Base.HasEltype()


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
Base.size(itr::RepeatStatefulIterator) = size(itr.base.itr)

Base.IteratorSize(itr::RepeatStatefulIterator) = Base.IteratorSize(itr.base.itr)
Base.IteratorEltype(itr::RepeatStatefulIterator) = Base.HasEltype()

"""
    SeedIterator
    SeedIterator(base; rng=rng_default, seed=rand(rng, UInt32))

Iterator which has the random seed of an `AbstractRNG` as state.

Calls `Random.seed!(rng, seed)` every iteration so that wrapped iterators which depend on `rng` will produce the same sequence.

Useful in conjunction with [`RepeatPartitionIterator`](@ref) and random data augmentation so that all candidates in a generation are trained with identical augmentation.
"""
struct SeedIterator{R <: AbstractRNG,T}
    rng::R
    seed::UInt32
    base::T
end
SeedIterator(base; rng=rng_default, seed=rand(rng, UInt32)) = SeedIterator(rng, UInt32(seed), base)

function Base.iterate(itr::SeedIterator)
    Random.seed!(itr.rng, itr.seed)
    valstate = iterate(itr.base)
    valstate === nothing && return nothing
    val, state = valstate
    return val, (itr.seed+1, state)
end

function Base.iterate(itr::SeedIterator, state)
    seed,basestate = state
    Random.seed!(itr.rng, seed)
    valstate = iterate(itr.base, basestate)
    valstate === nothing && return nothing
    val, state = valstate
    return val, (seed+1, state)
end

Base.length(itr::SeedIterator) = length(itr.base)
Base.eltype(itr::SeedIterator) = eltype(itr.base)
Base.size(itr::SeedIterator) = size(itr.base)

Base.IteratorSize(itr::SeedIterator) = Base.IteratorSize(itr.base)
Base.IteratorEltype(itr::SeedIterator) = Base.IteratorEltype(itr.base)

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
Base.size(itr::MapIterator) = size(Base.IteratorSize(itr.base), itr.base)
Base.size(::Base.IteratorSize, itr) = sizeof(first(itr))
Base.size(::Base.HasShape, itr) = size(itr)
sizeof(a::AbstractArray) = size(a)
sizeof(t::Tuple) = size.(t)


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
batch(s::Singleton, start, stop) = batch(val(s), start, stop)
batch(a::AbstractArray{T,1}, start, stop) where T = view(a, start:stop)
batch(a::AbstractArray{T,2}, start, stop) where T = view(a, :,start:stop)
batch(a::AbstractArray{T,3}, start, stop) where T = view(a, :,:,start:stop)
batch(a::AbstractArray{T,4}, start, stop) where T = view(a, :,:,:,start:stop)
batch(a::AbstractArray{T,5}, start, stop) where T = view(a, :,:,:,:,start:stop)
function batch(a::AbstractArray{T,N}, start, stop) where {T,N}
    get = repeat(Union{Colon, AbstractArray{Int}}[Colon()], N)
    get[end] = start:stop
    return view(a, get...)
end

Base.print(io::IO, itr::BatchIterator) = print(io, "BatchIterator(size=$(size(itr.base)), batchsize=$(itr.batchsize))")

"""
    Flux.onehotbatch(itr::BatchIterator, labels)
    Flux.onehotbatch(itr::ShuffleIterator, labels)

Return an iterator over [`Flux.onehotbatch`](@ref) of the values from `itr`.
"""
Flux.onehotbatch(itr::BatchIterator, labels) = MapIterator(x -> Flux.onehotbatch(x, labels), itr)

"""
    ShuffleIterator{T<:AbstractArray, R<:AbstractRNG}
    ShuffleIterator(data, batchsize, rng=rng_default)

Same as `BatchIterator` but also shuffles `data`. Order is reshuffled each time iteration begins.

Beware: The data is shuffled in place. Provide a copy if the unshuffled data is also needed.
"""
struct ShuffleIterator{T, R<:AbstractRNG}
    base::BatchIterator{T}
    rng::R
end
ShuffleIterator(data, bs, rng=rng_default) = ShuffleIterator(BatchIterator(data, bs), rng)

Base.length(itr::ShuffleIterator) = length(itr.base)
Base.size(itr::ShuffleIterator) = size(itr.base)

Base.IteratorSize(itr::ShuffleIterator) = Base.IteratorSize(itr.base)
Base.IteratorEltype(itr::ShuffleIterator) = Base.IteratorEltype(itr.base)

function Base.iterate(itr::ShuffleIterator)
    shufflelastdim!(itr.rng, itr.base.base)
    return iterate(itr.base)
end
Base.iterate(itr::ShuffleIterator, state) = iterate(itr.base, state)

## I *think* speed matters here, so...
shufflelastdim!(rng, s::Singleton) = shufflelastdim!(rng, val(s))
shufflelastdim!(rng, a::AbstractArray{T,1}) where T = a[:] = a[randperm(rng, size(a,1))]
shufflelastdim!(rng, a::AbstractArray{T,2}) where T = a[:,:] = a[:, randperm(rng, size(a, 2))]
shufflelastdim!(rng, a::AbstractArray{T,3}) where T = a[:,:,:] = a[:,:, randperm(rng, size(a, 3))]
shufflelastdim!(rng, a::AbstractArray{T,4}) where T = a[:,:,:,:] = a[:,:,:,randperm(rng, size(a, 4))]
shufflelastdim!(rng, a::AbstractArray{T,5}) where T = a[:,:,:,:,:] = a[:,:,:,:, randperm(rng, size(a, 5))]
function shufflelastdim!(rng, a::AbstractArray{T,N}) where {T,N}
    set = repeat(Union{Colon, AbstractArray{Int}}[Colon()], N)
    get = repeat(Union{Colon, AbstractArray{Int}}[Colon()], N)
    get[end] = randperm(rng, size(a,N))
    a[set...] = a[get...]
end

Flux.onehotbatch(itr::ShuffleIterator, labels) = MapIterator(x -> Flux.onehotbatch(x, labels), itr)
