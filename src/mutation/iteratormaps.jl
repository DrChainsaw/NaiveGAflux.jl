
(m::AbstractMutation{<:AbstractIteratorMap})(im::IteratorMaps) = IteratorMaps(m.(im.maps))

newfrom(im::AbstractIteratorMap) = deepcopy(im)

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
TrainBatchSizeMutation(l1,l2,q) = TrainBatchSizeMutation(l1,l2, q, rng_default)

(m::TrainBatchSizeMutation)(im::AbstractIteratorMap) = newfrom(im)
function (m::TrainBatchSizeMutation)(im::BatchSizeIteratorMap) 
    newbs = max(1, mutate_batchsize(m.quantizeto, batchsize(im.tbs), m.minrel, m.maxrel, m.rng)) 
    @set im.tbs = TrainBatchSize(newbs)
end


function mutate_batchsize(quantizeto::DataType, bs, minrel, maxrel, rng)
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
