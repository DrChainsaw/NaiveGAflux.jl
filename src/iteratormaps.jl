
"""
    AbstractIteratorMap

Abstract type for mapping training and validation dataset iterators using `maptrain(im, iter)` and `mapvalidation(im, iter)` respectively where `im` 
is the struct extending `AbstractIteratorMap`.

Main reason for existence is to enable dispatch to `AbstractMutation{AbstractIteratorMap}` and `AbstractCrossover{AbstractIteratorMap}` so that 
strategies for data augmentation and batch size selection can be evolved.
"""
abstract type AbstractIteratorMap end

maptrain(::AbstractIteratorMap, iter) = iter
mapvalidation(::AbstractIteratorMap, iter) = iter

limit_maxbatchsize(im::AbstractIteratorMap, args...; kwargs...) = im
 
"""
    BatchSizeIteratorMap{F} <: AbstractIteratorMap 
    BatchSizeIteratorMap(limitfun, trainbatchsize, validationbatchsize, model)

[AbstractIteratorMap](@ref) which sets the batch size of training and validation iterators to `trainbatchsize` and `validationbatchsize` respectively.
`limitfun` is used to try to ensure that batch sizes are small enough so that training and validating `model` does not risk an out of memory error.
Use [`batchsizeselection`](@ref) to create an appropriate `limitfun`.

# Examples
```jldoctest
julia> using NaiveGAflux

julia> import NaiveGAflux: maptrain, mapvalidation # needed for examples only

julia> bsim = BatchSizeIteratorMap(4, 8, batchsizeselection((32,32,3)));

julia> collect(maptrain(bsim, (1:20,)))
5-element Vector{Vector{Int64}}:
 [1, 2, 3, 4]
 [5, 6, 7, 8]
 [9, 10, 11, 12]
 [13, 14, 15, 16]
 [17, 18, 19, 20]

julia> collect(mapvalidation(bsim, (1:20,)))
3-element Vector{Vector{Int64}}:
 [1, 2, 3, 4, 5, 6, 7, 8]
 [9, 10, 11, 12, 13, 14, 15, 16]
 [17, 18, 19, 20]

julia> map(x -> Pair(x...), maptrain(bsim, ((1:20, 21:40),))) # Pair to make results a bit easier on the eyes
5-element Vector{Pair{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}}}:
     [1, 2, 3, 4] => [21, 22, 23, 24]
     [5, 6, 7, 8] => [25, 26, 27, 28]
  [9, 10, 11, 12] => [29, 30, 31, 32]
 [13, 14, 15, 16] => [33, 34, 35, 36]
 [17, 18, 19, 20] => [37, 38, 39, 40]

julia> map(x -> Pair(x...), maptrain(bsim, BatchIterator((1:20, 21:40),12))) # Pair to make results a bit easier on the eyes
5-element Vector{Pair{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}}}:
     [1, 2, 3, 4] => [21, 22, 23, 24]
     [5, 6, 7, 8] => [25, 26, 27, 28]
  [9, 10, 11, 12] => [29, 30, 31, 32]
 [13, 14, 15, 16] => [33, 34, 35, 36]
 [17, 18, 19, 20] => [37, 38, 39, 40]
```
"""
struct BatchSizeIteratorMap{F} <: AbstractIteratorMap
    tbs::TrainBatchSize
    vbs::ValidationBatchSize
    limitfun::F
end

function BatchSizeIteratorMap(tbs::Integer, vbs::Integer, limitfun)
    BatchSizeIteratorMap(TrainBatchSize(tbs), ValidationBatchSize(vbs), limitfun)
end

maptrain(bs::BatchSizeIteratorMap, iter) = setbatchsize(iter, batchsize(bs.tbs))
mapvalidation(bs::BatchSizeIteratorMap, iter) = setbatchsize(iter, batchsize(bs.vbs))

function limit_maxbatchsize(bsim::BatchSizeIteratorMap, args...; kwargs...) 
    BatchSizeIteratorMap(bsim.limitfun(bsim.tbs, args...; kwargs...), bsim.limitfun(bsim.vbs, args...; kwargs...), bsim.limitfun)
end

"""
    IteratorMaps{T} <: AbstractIteratorMap 
"""
struct IteratorMaps{T<:Tuple} <: AbstractIteratorMap
    maps::T
end
IteratorMaps(x...) = IteratorMaps(x)

maptrain(iws::IteratorMaps, iter) = foldr(maptrain, iws.maps; init=iter)
mapvalidation(iws::IteratorMaps, iter) = foldr(mapvalidation, iws.maps; init=iter)

limit_maxbatchsize(ims::IteratorMaps, args...; kwargs...) = IteratorMaps(map(im -> limit_maxbatchsize(im, args...; kwargs...), ims.maps))