itergeneration(itr, gen) = itr

struct MarkableIteratorState{S}
    state::Union{S, Nothing}
    cnt::Int
end

_nextstate(s::MarkableIteratorState{S}, nextstate) where S = MarkableIteratorState{S}(nextstate, s.cnt+1) 

"""
    MarkableStatefulIterator{S, I}

Stateful iterator that can be marked to remember its current state. Will start from the current state whenever a new iteration begins.

Intended to be used by [`RepeatPartitionIterator`](@ref).
"""
mutable struct MarkableStatefulIterator{S, I}
    const base::I
    laststate::MarkableIteratorState{S}
    markedstate::MarkableIteratorState{S}
end

function MarkableStatefulIterator(itr)
    firstvalstate = iterate(itr)
    S = if isnothing(firstvalstate)
        # Empty iterator. Weird, but ok
        Nothing
    else
        # Discard first value which is a bit wasteful
        # However, this iterator is intended to be used in a context
        # where it is created very seldom and used for a very long time
        # Since val is typically things like a large GPU array it is
        # probably (hopefully) better to not hold on to it.
        val, state = firstvalstate
        typeof(state)
    end

    MarkableStatefulIterator(
        itr,
        MarkableIteratorState{S}(nothing, 0),
        MarkableIteratorState{S}(nothing, 0)
    )
end


function Base.iterate(itr::MarkableStatefulIterator, state = itr.markedstate)
    nextvalstate = if isnothing(state.state)
        iterate(itr.base)
    else
        iterate(itr.base, state.state)
    end

    if isnothing(nextvalstate)
        itr.laststate = _nextstate(state, nothing)
        nothing
    else
        val, nextstate = nextvalstate
        itr.laststate = _nextstate(state, nextstate)
        val, itr.laststate
    end
end

function _reset!(itr::MarkableStatefulIterator{S}) where S
    itr.markedstate = MarkableIteratorState{S}(nothing, 0)
    itr.laststate = MarkableIteratorState{S}(nothing, 0)
end

function _mark!(itr::MarkableStatefulIterator)
    itr.markedstate = itr.laststate
end

Base.eltype(::Type{MarkableStatefulIterator{S, I}}) where {S, I} = eltype(I)
Base.size(itr::MarkableStatefulIterator) = size(itr.base)
Base.length(itr::MarkableStatefulIterator) = length(itr.base) - itr.markedstate.cnt

Base.IteratorSize(::Type{MarkableStatefulIterator{S, I}}) where {S, I} = Base.IteratorSize(I)
Base.IteratorEltype(::Type{MarkableStatefulIterator{S, I}}) where {S, I} = Base.IteratorEltype(I)


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
struct RepeatPartitionIterator{I <: MarkableStatefulIterator}
    base::I
    ntake::Int
end
RepeatPartitionIterator(base, nrep) = RepeatPartitionIterator(MarkableStatefulIterator(base), nrep)
RepeatPartitionIterator(base::MarkableStatefulIterator, nrep) = RepeatPartitionIterator(base, nrep)

function Base.iterate(itr::RepeatPartitionIterator, reset=true)
    if reset
        _reset!(itr.base)
    else
        _mark!(itr.base)
    end
    Base.IteratorSize(itr) !== Base.IsInfinite() && length(itr) == 0 && return nothing
    return Iterators.take(itr.base, itr.ntake), false
end

Base.length(itr::RepeatPartitionIterator) = cld(length(itr.base), itr.ntake)
Base.eltype(::Type{RepeatPartitionIterator{I}}) where {I} = eltype(I)
Base.size(itr::RepeatPartitionIterator) = tuple(length(itr))

Base.IteratorSize(::Type{RepeatPartitionIterator{I}}) where {I} = Base.IteratorSize(I)
Base.IteratorEltype(::Type{RepeatPartitionIterator{I}}) where {I} = Base.IteratorEltype(I)

_mark!(r::RepeatPartitionIterator) = _mark!(r.base)
_reset!(r::RepeatPartitionIterator) = _reset(r.base)

"""
    StatefulGenerationIter{I, R}

Uses a `RepeatPartitionIterator` to ensure that the same iterator is returned for the same generation number.
"""
mutable struct StatefulGenerationIter{I, R}
    currgen::Int
    curriter::I
    const iter::RepeatPartitionIterator{R}
end
# TODO : This is a bit of cludge-on-cludge. Try to refactor someday to a more straighforward design, perhaps use LearnBase.getobs
StatefulGenerationIter(iter::RepeatPartitionIterator, gen=0) = StatefulGenerationIter(gen, first(iterate(iter)), iter)

function itergeneration(itr::StatefulGenerationIter, gen)
    if gen != itr.currgen
        itr.currgen = gen
        newiterstate = iterate(itr.iter, false)
        if newiterstate === nothing
            newiterstate = iterate(itr.iter)
        end
        itr.curriter = first(newiterstate)
    end
    return itr.curriter
end


"""
    SeedIterator
    SeedIterator(base; rng=rng_default, seed=rand(rng, UInt32))

Iterator which has the random seed of an `AbstractRNG` as state.

Calls `Random.seed!(rng, seed)` every iteration so that wrapped iterators which depend on `rng` will produce the same sequence.

Useful in conjunction with [`RepeatPartitionIterator`](@ref) and [`BatchIterator`](@ref) and/or random data augmentation so that 
all candidates in a generation are trained with identical data.
"""
struct SeedIterator{R <: AbstractRNG, T}
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
Base.eltype(::Type{SeedIterator{R,T}}) where {R,T} = eltype(T)
Base.size(itr::SeedIterator) = size(itr.base)

Base.IteratorSize(::Type{SeedIterator{R, T}}) where {R,T}= Base.IteratorSize(T)
Base.IteratorEltype(::Type{SeedIterator{R, T}}) where {R,T} = Base.IteratorEltype(T)

"""
    GpuIterator(itr)

Return an iterator which sends values from `itr` to the GPU.

Will often be used automatically when training a model with parameters on the GPU.
"""
GpuIterator(itr) = Iterators.map(gpuitr, itr) # Iterator.map can't infer eltypes, but we can't either as we don't know for sure what Flux.gpu will do
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

Remember to use a `SeedIterator` if using this with [`RepeatPartitionIterator`](@ref) or
[`StatefulGenerationIter`](@ref).
"""
struct BatchIterator{R, D}
    nobs::Int
    batchsize::Int
    rng::R
    data::D
    function BatchIterator(nobs::Int, batchsize::Int, rng::R, data::D) where {R,D}
        batchsize > 0 || throw(ArgumentError("Batch size must be > 0. Got $(batchsize)!"))
        new{R,D}(nobs, batchsize, rng, data)
    end
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

function Base.iterate(itr::BatchIterator, inds = shuffle(itr.rng, 1:itr.nobs))
    isempty(inds) && return nothing
    return batch(itr.data, @view(inds[1:min(end, itr.batchsize)])), @view(inds[itr.batchsize+1:end])
end

Base.length(itr::BatchIterator) = cld(itr.nobs, itr.batchsize)
Base.eltype(::Type{BatchIterator{R,D}}) where {R,D} = D
Base.eltype(::Type{BatchIterator{R,Singleton{D}}}) where {R,D} = D
Base.eltype(::Type{BatchIterator{R, D}}) where {R <: AbstractRNG, D <: AbstractRange} = Array{eltype(D), ndims(D)}
Base.size(itr::BatchIterator) = tuple(length(itr))

Base.IteratorEltype(::Type{BatchIterator{R,D}}) where {R,D} = Base.IteratorEltype(D)
Base.IteratorEltype(::Type{BatchIterator{R,Singleton{D}}}) where {R,D} = Base.IteratorEltype(D)


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
Base.eltype(::Type{TimedIterator{F,A,I}}) where {F,A,I} = eltype(I)
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

setbatchsize(itr::TimedIterator, batchsize) = TimedIterator(itr.timelimit, itr.patience, itr.timeoutaction, itr.accumulate_timeouts, setbatchsize(itr.base, batchsize))

"""
    ReBatchingIterator{I}
    ReBatchingIterator(base, batchsize)

Return and iterator which iterates `batchsize` samples from `base` where `base` is in itself assumed to provide batches of another batchsize.

Reason for this convoluted construct is to provide a way to use different batch sizes for different models while still allowing all models to 
see the same samples (including data augmentation) in the same order. As we don't want to make assumption about what `base` is, this iterator is used by default. 
    
Implement `setbatchsize(itr::T, batchsize::Int)` for iterator types `T` where it is possible to set the batch size (or create a new iterator).
# Examples
```jldoctest
julia> using NaiveGAflux

julia> itr = ReBatchingIterator(BatchIterator(1:20, 8), 4); # Batch size 8 rebatched to 4

julia> collect(itr)
5-element Vector{Array{Int64}}:
 [1, 2, 3, 4]
 [5, 6, 7, 8]
 [9, 10, 11, 12]
 [13, 14, 15, 16]
 [17, 18, 19, 20]

julia> itr = ReBatchingIterator(BatchIterator((1:20, 21:40), 4), 8); # Batch size 4 rebatched to 8

julia> map(x -> Pair(x...), itr) # Pair to make results a bit easier on the eyes
3-element Vector{Pair{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}}}:
        [1, 2, 3, 4, 5, 6, 7, 8] => [21, 22, 23, 24, 25, 26, 27, 28]
 [9, 10, 11, 12, 13, 14, 15, 16] => [29, 30, 31, 32, 33, 34, 35, 36]
                [17, 18, 19, 20] => [37, 38, 39, 40]
"""
struct ReBatchingIterator{I}
    batchsize::Int
    base::I

    function ReBatchingIterator(batchsize::Int, base::I) where I
        batchsize > 0 || throw(ArgumentError("Batch size must be > 0. Got $(batchsize)!"))
        new{I}(batchsize, base)
    end
end
function ReBatchingIterator(base, batchsize::Int) 
    ReBatchingIterator(batchsize, base)
end

"""
    setbatchsize(itr, batchsize) 

Return an iterator which iterates over the same data in the same order as `itr` with batch size `batchsize`.

Defaults to [`ReBatchingIterator`](@ref) for iterators which don't have a specialized method. 
"""
setbatchsize(itr, batchsize) = ReBatchingIterator(itr, batchsize)

Base.eltype(::Type{ReBatchingIterator{I}}) where I = _rangetoarr(eltype(I))

# We can only know the size if the underlying iterator does not produce partial batches (i.e smaller than the batch size)
Base.IteratorSize(::Type{ReBatchingIterator{I}}) where I = Base.SizeUnknown()
Base.IteratorEltype(::Type{ReBatchingIterator{I}}) where I = Base.IteratorEltype(I) 

_rangetoarr(a) = a
_rangetoarr(t::Type{<:Tuple}) = Tuple{map(_rangetoarr, t.parameters)...}
_rangetoarr(a::Type{<:Array}) = a
_rangetoarr(::Type{<:AbstractArray{T,N}}) where {T,N} = Array{T,N}

function Base.iterate(itr::ReBatchingIterator)
    innerval, innerstate = IterTools.@ifsomething iterate(itr.base)
    innerval, innerstate = _concat_inner(itr.base, itr.batchsize, _collectbatch(innerval), innerstate)
    bitr = BatchIterator(innerval, itr.batchsize)
    outerval, outerstate = IterTools.@ifsomething iterate(bitr)
    return outerval, (bitr, outerstate, innerstate)
end


function Base.iterate(itr::ReBatchingIterator, (bitr, outerstate, innerstate))
    outervalstate = iterate(bitr, outerstate)
    if outervalstate === nothing
        innerval, innerstate = IterTools.@ifsomething iterate(itr.base, innerstate)
        innerval, innerstate = _concat_inner(itr.base, itr.batchsize, _collectbatch(innerval), innerstate)
        bitr = BatchIterator(innerval, itr.batchsize)
        outerval, outerstate = IterTools.@ifsomething iterate(bitr)
        return outerval, (bitr, outerstate, innerstate)
    end
    outerval, outerstate = outervalstate
    return outerval, (bitr, outerstate, innerstate)
end

function _concat_inner(inneritr, batchsize, innerval, innerstate)
    while _innerbatchsize(innerval) < batchsize
        innervalstate = iterate(inneritr, innerstate)
        innervalstate === nothing && break
        innerval, innerstate = _catbatch(innerval, first(innervalstate)), last(innervalstate)
    end
    return innerval, innerstate
end

_catbatch(b1::Tuple, b2::Tuple) = _catbatch.(b1, b2)
_catbatch(b1::AbstractArray{T, N}, b2::AbstractArray{T, N}) where {T,N} = cat(b1, b2; dims=ndims(b1))
_catbatch(::T1, ::T2) where {T1, T2}= throw(DimensionMismatch("Tried to cat incompatible types when rebatching: $T1 vs $T2"))

_collectbatch(b::Tuple) = _collectbatch.(b)
_collectbatch(b::AbstractRange) = collect(b)
_collectbatch(b) = b

_innerbatchsize(t::Tuple) = _innerbatchsize(first(t))
_innerbatchsize(a::AbstractArray) = size(a, ndims(a))
