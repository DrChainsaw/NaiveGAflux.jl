itergeneration(itr, gen) = itr

"""
    RepeatPartitionIterator
    RepeatPartitionIterator(base, nrep)

Iteratates over iterators of a subset of size `nrep` elements in `base`.

Generally useful for training all models in a population with the same data in each evolution epoch.

Tailored for situations where iterating over models is more expensive than iterating over data, for 
example if candidates are stored in host RAM or on disk and needs to be transferred to the GPU for training.

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
    Base.IteratorSize(itr) !== Base.IsInfinite() && length(itr) == 0 && return nothing
    return Iterators.take(RepeatStatefulIterator(itr.base), itr.ntake), false
end

Base.length(itr::RepeatPartitionIterator) = cld(length(itr.base), itr.ntake)
Base.eltype(itr::RepeatPartitionIterator) = eltype(itr.base)
Base.size(itr::RepeatPartitionIterator) = size(itr.base.itr)


Base.IteratorSize(itr::RepeatPartitionIterator) = Base.IteratorSize(itr.base.itr)
Base.IteratorEltype(itr::RepeatPartitionIterator) = Base.IteratorEltype(itr.base.itr)


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
    val, state = IterTools.@ifsomething iterate(itr.base)
    return val, false
end

Base.length(itr::RepeatStatefulIterator) = length(itr.base.itr) - itr.taken
Base.eltype(itr::RepeatStatefulIterator) = eltype(itr.base)
Base.size(itr::RepeatStatefulIterator) = size(itr.base.itr)

Base.IteratorSize(::Type{RepeatStatefulIterator{T,VS}}) where {T, VS} = Base.IteratorSize(Iterators.Stateful{T,VS})
Base.IteratorEltype(::Type{RepeatStatefulIterator{T,VS}}) where {T, VS} = Base.IteratorEltype(Iterators.Stateful{T,VS})

"""
    StatefulGenerationIter{T, VS}

Uses a `RepeatPartitionIterator` to ensure that the same iterator is returned for the same generation number.
"""
struct StatefulGenerationIter{I, T, VS}
    currgen::Ref{Int}
    curriter::Ref{I}
    iter::RepeatPartitionIterator{T, VS}
end
# TODO : This is a bit of cludge-on-cludge. Try to refactor someday to a more straighforward design, perhaps use LearnBase.getobs
StatefulGenerationIter(iter::RepeatPartitionIterator, gen=0) = StatefulGenerationIter(Ref(gen), Ref(first(iterate(iter))), iter)

function itergeneration(itr::StatefulGenerationIter, gen)
    if gen != itr.currgen[]
        itr.currgen[] = gen
        newiterstate = iterate(itr.iter, false)
        if newiterstate === nothing
            newiterstate = iterate(itr.iter)
        end
        itr.curriter[] = first(newiterstate)
    end
    return itr.curriter[]
end


"""
    SeedIterator
    SeedIterator(base; rng=rng_default, seed=rand(rng, UInt32))

Iterator which has the random seed of an `AbstractRNG` as state.

Calls `Random.seed!(rng, seed)` every iteration so that wrapped iterators which depend on `rng` will produce the same sequence.

Useful in conjunction with [`RepeatPartitionIterator`](@ref) and [`BatchIterator`](@ref) and/or random data augmentation so that all candidates in a generation are trained with identical data.
"""
struct SeedIterator{R <: AbstractRNG,T}
    rng::R
    seed::UInt32
    base::T
end
SeedIterator(base; rng=rng_default, seed=rand(rng, UInt32)) = SeedIterator(rng, UInt32(seed), base)

function Base.iterate(itr::SeedIterator)
    Random.seed!(itr.rng, itr.seed)
    val, state = IterTools.@ifsomething iterate(itr.base)
    return val, (itr.seed+1, state)
end

function Base.iterate(itr::SeedIterator, state)
    seed,basestate = state
    Random.seed!(itr.rng, seed)
    val, state = IterTools.@ifsomething iterate(itr.base, basestate)
    return val, (seed+1, state)
end

Base.length(itr::SeedIterator) = length(itr.base)
Base.eltype(itr::SeedIterator) = eltype(itr.base)
Base.size(itr::SeedIterator) = size(itr.base)

Base.IteratorSize(::Type{SeedIterator{R, T}}) where {R,T}= Base.IteratorSize(T)
Base.IteratorEltype(::Type{SeedIterator{R, T}}) where {R,T} = Base.IteratorEltype(T)

"""
    GpuIterator(itr)

Return an iterator which sends values from `itr` to the GPU.
"""
GpuIterator(itr) = Iterators.map(gpuitr, itr)
gpuitr(a) = Flux.gpu(a)
gpuitr(a::SubArray) = gpuitr(collect(a))
gpuitr(a::Tuple) = gpuitr.(a)

struct NoShuffle end
Random.shuffle(::NoShuffle, x) = x

"""
    BatchIterator{R, D}
    BatchIterator(data, batchsize; shuffle)

Return an iterator which iterates `batchsize` samples along the last dimension of `data` 
or all elements of `data` if `data` is a `Tuple` (e.g `(features, labels)`).

Will shuffle examples if `shuffle` is `true` or an `AbstractRNG`. Shuffling will be 
different each time iteration starts (subject to implementation of shuffle(rng,...)).  
"""
struct BatchIterator{R, D}
    nobs::Int
    batchsize::Int
    rng::R
    data::D
end
BatchIterator(data::Union{AbstractArray, Singleton}, bs::Int; kwargs...) = BatchIterator(size(data)[end], bs, data; kwargs...)
function BatchIterator(data::Tuple, bs; kwargs...) 
    @assert all(x -> size(x)[end] === size(data[1])[end], data) "Mistmatched batch dimensions! Got sizes $(size.(data))"
    BatchIterator(size(data[1])[end], bs, data; kwargs...)
end
function BatchIterator(nobs::Integer, batchsize::Integer, data; shuffle=false)
    return BatchIterator(nobs, batchsize, shufflerng(shuffle), data)
end
shufflerng(b::Bool) = b ? rng_default : NoShuffle()
shufflerng(rng) = rng

ndata(itr::BatchIterator) = ndata(itr.data)
ndata(data::Tuple) = length(data)
ndata(data::AbstractArray) = 1 

function Base.iterate(itr::BatchIterator, inds = shuffle(itr.rng, 1:itr.nobs))
    isempty(inds) && return nothing
    return batch(itr.data, @view(inds[1:min(end, itr.batchsize)])), @view(inds[itr.batchsize+1:end])
end

Base.length(itr::BatchIterator) = cld(itr.nobs, itr.batchsize)
Base.size(itr::BatchIterator) = tuple(length(itr))

Base.IteratorEltype(::Type{BatchIterator{R,D}}) where {R, D} = Base.IteratorEltype(D)

batch(s::Singleton, inds) = batch(val(s), inds)
batch(b::Tuple, inds) = batch.(b, Ref(inds))
batch(a::AbstractArray{T,N}, inds) where {T,N} = selectdim(a, N, inds)

Base.show(io::IO, itr::BatchIterator{R}) where R = print(io, "BatchIterator(size=$(size(itr.data)), batchsize=$(itr.batchsize), shuffle=$(R !== NoShuffle))")
Base.show(io::IO, itr::BatchIterator{R, <:Tuple}) where R = print(io, "BatchIterator(size=$(size.(itr.data)), batchsize=$(itr.batchsize), shuffle=$(R !== NoShuffle))")

"""
    TimedIterator{F,A,I}
    TimedIterator(;timelimit, patience, timeoutaction, accumulate_timeouts, base)

Measures time between iterations and calls `timeoutaction()` if this time is longer than `timelimit` `patience` number of times.

Intended use is to quickly abort training of models which take very long time to train, typically also assigning them a very low fitness in case of a timeout.

By default, calling `timeoutaction()` will not stop the iteration as this would break otherwise convenient functions like `length`, `collect` and `map`. Let `timeoutaction()` return `TimedIteratorStop` to stop iteration.

If `accumulate_timeouts` is `false` then counting will reset when time between iterations is shorter than `timelimit`, otherwise it will not.
""" 
struct TimedIterator{F,A,I}
    timelimit::F
    patience::Int
    timeoutaction::A
    accumulate_timeouts::Bool
    base::I
end
TimedIterator(;timelimit, patience=5, timeoutaction, accumulate_timeouts=false, base) = TimedIterator(timelimit, patience, timeoutaction, accumulate_timeouts, base)
"""
    TimedIteratorStop

Special type for stopping a `TimedIterator` if returned from `timeoutaction()`.
"""
struct TimedIteratorStop
    TimedIteratorStop() = throw(DomainError("You typically do not want to instantiate this as `TimedIterator` expects the type and not a value, i.e `return TimedIteratorStop` instead of `return TimedIteratorStop()`. Call TimedIteratorStop(true) if you really want to instantiate"))
    TimedIteratorStop(::Bool) = new()
end

Base.length(itr::TimedIterator) = length(itr.base)
Base.size(itr::TimedIterator) = size(itr.base)

Base.IteratorSize(::Type{TimedIterator{F,A,I}}) where {F,A,I} = Base.IteratorSize(I)
Base.IteratorEltype(::Type{TimedIterator{F,A,I}}) where {F,A,I}  = Base.IteratorEltype(I)

function Base.iterate(itr::TimedIterator)
    val, bstate = IterTools.@ifsomething iterate(itr.base)
    tstamp = NaN #Always skip first due to compilation times etc
    return val, (tstamp, 0, bstate)
end

function Base.iterate(itr::TimedIterator, (tstamp, ntimeout, bstate)::Tuple)
    duration = time() - tstamp
    # Note reiles on NaN > x === false for all x
    ntimeout = if duration > itr.timelimit 
        ntimeout + 1 
    else
         itr.accumulate_timeouts ? ntimeout : 0
    end

    if ntimeout >= itr.patience
        stop = itr.timeoutaction()
        # Note: we allow users to return e.g. nothing here to continue
        stop === TimedIteratorStop && return nothing
    end

    val, bstate = IterTools.@ifsomething iterate(itr.base, bstate)
    tstamp = time()
    return val, (tstamp, ntimeout, bstate)
end

# itergeneration on timeoutaction just in case user e.g. wants to deepcopy it to avoid some shared state.
itergeneration(itr::TimedIterator, gen) = TimedIterator(itr.timelimit, itr.patience, itergeneration(itr.timeoutaction, gen), itr.accumulate_timeouts, itergeneration(itr.base, gen))