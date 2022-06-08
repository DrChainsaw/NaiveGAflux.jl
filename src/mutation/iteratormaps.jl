newfrom(im::AbstractIteratorMap) = deepcopy(im)

"""
    TrainBatchSizeMutation{R<:Real, Q, RNG<:AbstractRNG} 
    TrainBatchSizeMutation(l1, l2, quantizeto, rng) 
    TrainBatchSizeMutation(l1, l2, rng::AbstractRNG) 
    TrainBatchSizeMutation(l1, l2, quantizeto)
    TrainBatchSizeMutation(l1, l2)

Mutate the batch size used for training. 
    
Maximum possible relative change is determined by the numbers `l1` and `l2`.

Behaviour depends on `quantizeto` (default `Int`) in the following way. 

If `quantizeto` is a `DataType` (e.g `Int`) then the largest possible increase is `maxrel * batchsize` and the largest possible
 decrease is `minrel * batchsize` where `minrel` and `maxrel` are `l1` and `l2` if `l1 < l2` and `l2` and `l1` otherwise. Note that
 if `(minrel, maxrel)` is not symetric around `0` the mutation will be biased.

More precisely, the new size is `round(quantizeto, (x+1) * batchsize)` where `x` is drawn from `U(minrel, maxrel)`.

If `quantizeto` is a an array or tuple of values then the new size is drawn from `quantizeto` with elements closer 
to the current batch size being more likely.

More precisely, the new size is `quantizeto[i]` where `i = j + round(Int, x * length(quantizeto))` where `x` is drawn from 
`U(minrel, maxrel)` and `j` is the index for which `quantizeto[j]` is the closest to the current batch size.

Use the function `mutate_batchsize` to get a feeling for how different values of `l1`, `l2` and `quantizeto` affect the new batch sizes. 
Note that setting `l1 == l2` means that `x` in the descriptions above will always equal to `l1` (and `l2`) can also be useful in this context.

# Examples
```jldoctest aa; filter=r"\\d*"
julia> using NaiveGAflux

julia> m = TrainBatchSizeMutation(-0.1, 0.1);

julia> m(BatchSizeIteratorMap(16, 32, identity))
BatchSizeIteratorMap{typeof(identity)}(NaiveGAflux.TrainBatchSize(15), NaiveGAflux.ValidationBatchSize(32), identity)

julia> m = TrainBatchSizeMutation(-0.1, 0.1, ntuple(i -> 2^i, 10)); # Quantize to powers of 2

julia> m(BatchSizeIteratorMap(16, 32, identity))
BatchSizeIteratorMap{typeof(identity)}(NaiveGAflux.TrainBatchSize(32), NaiveGAflux.ValidationBatchSize(32), identity)

julia> NaiveGAflux.mutate_batchsize(Int, 16, -0.3, 0.3)
14

julia> NaiveGAflux.mutate_batchsize(Int, 16, -0.3, 0.3)
19

julia> NaiveGAflux.mutate_batchsize(ntuple(i -> 2^i, 10),  16, -0.3, 0.3)
64

julia> NaiveGAflux.mutate_batchsize(ntuple(i -> 2^i, 10),  16, -0.3, 0.3)
8
```
"""
struct TrainBatchSizeMutation{R<:Real, Q, RNG<:AbstractRNG} <: AbstractMutation{AbstractIteratorMap}
    minrel::R
    maxrel::R
    quantizeto::Q
    rng::RNG
    function TrainBatchSizeMutation(l1::R1, l2::R2, quantizeto::Q, rng::RNG) where {R1, R2, Q, RNG} 
        R = promote_type(R1, R2)
        return l1 < l2 ? new{R, Q, RNG}(promote(l1, l2)..., quantizeto, rng) : new{R, Q, RNG}(promote(l2, l1)..., quantizeto, rng)
    end
end
TrainBatchSizeMutation(l1, l2, rng::AbstractRNG=rng_default) = TrainBatchSizeMutation(l1, l2, Int, rng)
TrainBatchSizeMutation(l1, l2, q) = TrainBatchSizeMutation(l1, l2, q, rng_default)

(m::TrainBatchSizeMutation)(im::AbstractIteratorMap) = newfrom(im)
(m::TrainBatchSizeMutation)(im::IteratorMaps) = IteratorMaps(m.(im.maps))

function (m::TrainBatchSizeMutation)(im::BatchSizeIteratorMap) 
    newbs = max(1, mutate_batchsize(m.quantizeto, batchsize(im.tbs), m.minrel, m.maxrel, m.rng)) 
    @set im.tbs = TrainBatchSize(newbs)
end

function mutate_batchsize(quantizeto::DataType, bs, minrel, maxrel, rng=rng_default)
    shift = minrel
    scale = maxrel - minrel
    newbs = bs * (1 + rand(rng) * scale + shift)
    round(quantizeto, newbs)
end

function mutate_batchsize(quantizeto::Union{AbstractArray, Tuple}, bs, args...)
    bs, ind = findmin(x -> abs(bs - x), quantizeto)
    indstep = mutate_batchsize(Int, length(quantizeto), args...) - length(quantizeto)
    newind = clamp(indstep + ind, firstindex(quantizeto), lastindex(quantizeto))
    return quantizeto[newind]
end
