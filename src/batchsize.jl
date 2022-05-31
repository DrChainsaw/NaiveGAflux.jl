generic_batchsizefun_docstring(fname="batchsizefun") = """

`$(fname)` is a function with the following signature:

`$(fname)(model, batchsize; inshape_nobatch, availablebytes)`

It returns the largest batch size not larger than `batchsize` which can be used for `model` without using more than `availablebytes` bytes of memory.
The type of `batchsize` may be used to e.g. determine if one shall account for backwards pass (if `typeof(batchsize) === TrainBatchSize`) or not (if `typeof(batchsize) == ValidationBatchSize`).

"""

generic_batchsizefun_testgraph() = """
julia> v0 = conv2dinputvertex("v0", 3);

julia> v1 = fluxvertex("v1", Conv((3,3), nout(v0) => 8), v0);

julia> graph = CompGraph(v0, v1);
"""

generic_batchsizeselection_example(sbs, kwres...) = """
julia> bs(graph, TrainBatchSize(512); $(first(kwres[1]))availablebytes = 10_000_000) # availablebytes supplied for doctest reasons
$(last(kwres[1]))

julia> bs(graph, TrainBatchSize(512); $(first(kwres[2]))availablebytes = 1000_000_000)
$(last(kwres[2]))

julia> $sbs

julia> sbs(graph, TrainBatchSize(512); $(first(kwres[3]))availablebytes = 10_000_000)
$(last(kwres[3]))

julia> sbs(graph, TrainBatchSize(512); $(first(kwres[4]))availablebytes = 1000_000_000)
$(last(kwres[4]))

julia> bs(graph, ValidationBatchSize(512); $(first(kwres[5]))availablebytes=10_000_000)
$(last(kwres[5]))
"""

# Mostly to enable dispatch when mutating since that happens to be the only way to know what about a candidate to mutate :(
# We make use of types below as well, but that is mostly because they happen to already be there.
struct TrainBatchSize
    size::Int
end
batchsize(bs::TrainBatchSize) = bs.size

struct ValidationBatchSize
    size::Int
end
batchsize(bs::ValidationBatchSize) = bs.size


"""
    BatchSizeSelectionWithDefaultInShape{T, F}
    BatchSizeSelectionWithDefaultInShape(default_inshape)
    BatchSizeSelectionWithDefaultInShape(default_inshape, batchsizefun)

Batch size selection with a default assumed inshape used for estimating valid batch sizes.

$(generic_batchsizefun_docstring())

Returns the result of `batchsizefun` with default value of `inshape_nobatch = default_inshape` when called as a function with valid inputs to `batchsizefun`.

Composable with other batch size selection types which may be used as `batchsizefun`. See examples.

# Examples
```jldoctest
julia> using NaiveGAflux, Flux

julia> import NaiveGAflux: TrainBatchSize, ValidationBatchSize # Needed only for examples

$(generic_batchsizefun_testgraph())
julia> bs = BatchSizeSelectionWithDefaultInShape((32,32,3));

$(generic_batchsizeselection_example(
    "sbs = BatchSizeSelectionWithDefaultInShape((32,32,3), BatchSizeSelectionScaled(0.5));",
    "" => "120",
    "" => "512",
    "" => "60",
    "" => "512",
    "" => "243"))
```
"""
struct BatchSizeSelectionWithDefaultInShape{T, F}
    default_inshape::T
    batchsizefun::F
end
function BatchSizeSelectionWithDefaultInShape(default_inshape) 
        BatchSizeSelectionWithDefaultInShape(default_inshape, limit_maxbatchsize)
end
function (bs::BatchSizeSelectionWithDefaultInShape)(args...; inshape_nobatch=bs.default_inshape ,kwargs...) 
        bs.batchsizefun(args...; inshape_nobatch, kwargs...)
end


"""
    BatchSizeSelectionScaled{F}
    BatchSizeSelectionScaled(scale)
    BatchSizeSelectionScaled(scale, batchsizefun)

Batch size selection with a margin applied when estimating valid batch sizes.

$(generic_batchsizefun_docstring())

Returns the result of `batchsizefun` with default value of `availablebytes = floor(scale * availablebytes)` when called as a function with valid inputs to `batchsizefun`.

Composable with other batch size selection types which may be used as `batchsizefun`. See examples.

# Examples
```jldoctest
julia> using NaiveGAflux, Flux

julia> import NaiveGAflux: TrainBatchSize, ValidationBatchSize # Needed only for examples

$(generic_batchsizefun_testgraph())
julia> bs = BatchSizeSelectionScaled(0.5);

$(generic_batchsizeselection_example(
    "sbs = BatchSizeSelectionScaled(0.5, BatchSizeSelectionWithDefaultInShape((32,32,3)));",
    "inshape_nobatch=(32,32,3), " => "60",
    "inshape_nobatch=(32,32,3), " => "512",
    "" => "60",
    "" => "512",
    "inshape_nobatch=(32,32,3), " => "121"))
```
"""
struct BatchSizeSelectionScaled{F}
    scale::Float64
    batchsizefun::F
end
BatchSizeSelectionScaled(scale::AbstractFloat) = BatchSizeSelectionScaled(scale, limit_maxbatchsize)  
function (bs::BatchSizeSelectionScaled)(args...; availablebytes=_availablebytes(), kwargs...) 
    bs.batchsizefun(args...;availablebytes = floor(Int, bs.scale * availablebytes), kwargs...)
end

"""
    BatchSizeSelectionFromAlternatives{T, F}
    BatchSizeSelectionFromAlternatives(alts)
    BatchSizeSelectionFromAlternatives(alts, batchsizefun)

Batch size selection from a set of available alternatives. Useful for iterators which need to be pre-loaded with batch size, for example the iterators in this package.

$(generic_batchsizefun_docstring())

Returns the largest number in `alts` smaller than the result of `batchsizefun` when called as a function with valid inputs to `batchsizefun`.

Composable with other batch size selection types which may be used as `batchsizefun`. See examples.

# Examples
```jldoctest
julia> using NaiveGAflux, Flux

julia> import NaiveGAflux: TrainBatchSize, ValidationBatchSize # Needed only for examples

$(generic_batchsizefun_testgraph())
julia> bs = BatchSizeSelectionFromAlternatives(2 .^ (0:10));

$(generic_batchsizeselection_example(
    "sbs = BatchSizeSelectionFromAlternatives(2 .^ (0:10), BatchSizeSelectionScaled(0.5));",
    "inshape_nobatch=(32,32,3), " => "64",
    "inshape_nobatch=(32,32,3), " => "512",
    "inshape_nobatch=(32,32,3), " => "32",
    "inshape_nobatch=(32,32,3), " => "512",
    "inshape_nobatch=(32,32,3), " => "128"))
```
"""
struct BatchSizeSelectionFromAlternatives{T, F}
    alts::T
    batchsizefun::F
end
BatchSizeSelectionFromAlternatives(alts) = BatchSizeSelectionFromAlternatives(alts, limit_maxbatchsize)

function (bs::BatchSizeSelectionFromAlternatives)(args...;kwargs...) 
    select_bestfit_smaller(bs.batchsizefun(args...;kwargs...), bs.alts)
end

function select_bestfit_smaller(bs::Integer, alts)
    validalts = filter(<=(bs), alts)
    isempty(validalts) && return 0
    argmin(x -> bs - x, validalts)
end

"""
    BatchSizeSelectionMaxSize{F}
    BatchSizeSelectionMaxSize(uppersize) 
    BatchSizeSelectionMaxSize(uppersize, batchsizefun) 

Batch size selection which always try to select `uppersize`. Basically the strategy to select the largest batchsize which fits in memory.

$(generic_batchsizefun_docstring())

Returns the result of `batchsizefun` but with the `batchsize` as `uppersize` of the same type as `batchsize` (i.e. to differentiate between train size and validation size).

Composable with other batch size selection types which may be used as `batchsizefun`. See examples.

# Examples
```jldoctest
julia> using NaiveGAflux, Flux

julia> import NaiveGAflux: TrainBatchSize, ValidationBatchSize # Needed only for examples

$(generic_batchsizefun_testgraph())
julia> bs = BatchSizeSelectionMaxSize(1024);

$(generic_batchsizeselection_example(
    "sbs = BatchSizeSelectionMaxSize(1024, BatchSizeSelectionScaled(0.5));",
    "inshape_nobatch=(32,32,3), " => "120",
    "inshape_nobatch=(32,32,3), " => "1024",
    "inshape_nobatch=(32,32,3), " => "60",
    "inshape_nobatch=(32,32,3), " => "1024",
    "inshape_nobatch=(32,32,3), " => "243"))
```
"""
struct BatchSizeSelectionMaxSize{F}
    uppersize::Int
    batchsizefun::F
end
BatchSizeSelectionMaxSize(uppersize) = BatchSizeSelectionMaxSize(uppersize, limit_maxbatchsize)
function (bs::BatchSizeSelectionMaxSize)(c, orgbs, args...; kwargs...)
     bs.batchsizefun(c, newbatchsize(orgbs, bs.uppersize), args...; kwargs...)
end
# For strange batch size types which can't be created from just a number
newbatchsize(::T, newsize) where T = T(newsize) 

"""
    batchsizeselection(inshape_nobatch::Tuple; maxmemutil=0.7, uppersize=nothing, alternatives=nothing, batchsizefun=limit_maxbatchsize)

Return a batch size selection callable which may be used to select an appropriate batch size when given a model and 
a suggested batch size.

`inshape_nobatch` is the size of the input without the batch dimension (e.g. 3 values for images) to be assumed. See [`BatchSizeSelectionWithDefaultInShape`](@ref)

$(generic_batchsizefun_docstring())

`maxmemutil` is the maximum memory utilization which typically need to be `< 1` to account for inaccuracies in the estimation. See [`BatchSizeSelectionScaled`](@ref)

If `uppersize` is not `nothing` the maximum possible batchsize smaller or equal to `uppersize` will be used. See [`BatchSizeSelectionMaxSize`](@ref)

If `alternatives` is not nothing, the returned batchsize will be quantized to the closest matching size in `alternatives` which is not bigger than the unquantized batch size. See [`BatchSizeSelectionFromAlternatives`](@ref).

# Examples
```jldoctest
julia> using NaiveGAflux, Flux

julia> import NaiveGAflux: TrainBatchSize, ValidationBatchSize # Needed only for examples

$(generic_batchsizefun_testgraph())
julia> bs = batchsizeselection((32,32,3));

julia> bs(graph, TrainBatchSize(128); availablebytes = 10_000_000) # availablebytes supplied for doctest reasons
84

julia> bs(graph, ValidationBatchSize(128); availablebytes = 10_000_000)
128

julia> bs = batchsizeselection((32,32,3); maxmemutil=0.1);

julia> bs(graph, TrainBatchSize(128); availablebytes = 10_000_000)
12

julia> bs(graph, ValidationBatchSize(128); availablebytes = 10_000_000)
24

julia> bs = batchsizeselection((32,32,3); uppersize=1024);

julia> bs(graph, TrainBatchSize(128); availablebytes = 10_000_000)
84

julia> bs(graph, ValidationBatchSize(128); availablebytes = 10_000_000)
170

julia> bs = batchsizeselection((32,32,3); uppersize=1024, alternatives = 2 .^ (0:10));

julia> bs(graph, TrainBatchSize(128); availablebytes = 10_000_000)
64

julia> bs(graph, ValidationBatchSize(128); availablebytes = 10_000_000)
128
```
"""
function batchsizeselection(inshape_nobatch::Tuple;
                            batchsizefun=limit_maxbatchsize, 
                            maxmemutil=0.7, 
                            uppersize=nothing, 
                            alternatives=nothing)
    bs = BatchSizeSelectionWithDefaultInShape(inshape_nobatch, batchsizefun)
    bs = isnothing(maxmemutil) ? bs : BatchSizeSelectionScaled(maxmemutil, bs)
    bs = isnothing(uppersize) ? bs : BatchSizeSelectionMaxSize(uppersize, bs)
    bs = isnothing(alternatives) ? bs : BatchSizeSelectionFromAlternatives(alternatives, bs)
end

# specialization for CompGraph needed to avoid ambiguity with method that just unwraps an AbstractCandidate :( 
# Consider refactoring
function limit_maxbatchsize(model::CompGraph, bs::TrainBatchSize; inshape_nobatch, availablebytes = _availablebytes())
    min(batchsize(bs), maxtrainbatchsize(model, inshape_nobatch, availablebytes))
end

# specialization for CompGraph needed to avoid ambiguity with method that just unwraps an AbstractCandidate :( 
# Consider refactoring
function limit_maxbatchsize(model::CompGraph, 
                            bs::ValidationBatchSize; 
                            inshape_nobatch,
                            availablebytes = _availablebytes()
                            )
    min(batchsize(bs), maxvalidationbatchsize(model, inshape_nobatch, availablebytes))
end

function maxtrainbatchsize(model, inshape_nobatch, availablebytes=_availablebytes())
    paramsize = mapreduce(ps -> length(ps) * sizeof(eltype(ps)), +, params(model))
    actsize = activationsizes(model, inshape_nobatch) 
    return fld(availablebytes - paramsize, paramsize + 2 * actsize)
end

function maxvalidationbatchsize(model, inshape_nobatch, availablebytes=_availablebytes())
    paramsize = mapreduce(ps -> length(ps) * sizeof(eltype(ps)), +, params(model))
    actsize = activationsizes(model, inshape_nobatch)
    return fld(availablebytes - paramsize, actsize)
end

function activationsizes(model::CompGraph, inshape_nobatch, elemsize = model |> params |> first |> eltype |> sizeof)
    activations = if length(inputs(model)) == 1
        Dict{AbstractVertex, Any}(v => Flux.nil_input(true, inshape_nobatch) for v in inputs(model))
    else
        Dict{AbstractVertex, Any}(v => Flux.nil_input(true, inshape_nobatch)[i] for (i, v) in inputs(model))
    end
    for v in outputs(model) 
        output!(activations, v)
    end

    mapreduce(act -> length(act) * elemsize, +, values(activations))
end

function _availablebytes()
    if CUDA.functional()
        info = CUDA.MemoryInfo()
        info.free_bytes + info.pool_reserved_bytes - info.pool_used_bytes
    else
        Int(Sys.free_memory())
    end
end
