generic_batchsizefun_docstring(fname="batchsizefun") = """

`$(fname)` is a function with the following signature:

`$(fname)(model, batchsize; inshape_nobatch, availablebytes)`

It returns the largest batch size not larger than `batchsize` which can be used for `model` without using more than `availablebytes` bytes of memory.
The type of `batchsize` may be used to e.g. determine if one shall account for backwards pass (if `typeof(batchsize) === TrainBatchSize`) or not (if `typeof(batchsize) == ValidationBatchSize`).

"""

generic_batchsizefun_testgraph() = """
julia> v0 = conv2dinputvertex("v0", 3);

julia> v1 = fluxvertex("v1", Conv((3,3), nout(v0) => 8), v0);

julia> model = CompGraph(v0, v1);
"""

generic_batchsizeselection_example(sbs, kwres...) = """

# availablebytes is automatically computed if omitted, but here we supply it to avoid doctest errors
julia> bs(model, TrainBatchSize(512); $(first(kwres[1]))availablebytes = 10_000_000)
$(last(kwres[1]))

julia> bs(model, TrainBatchSize(512); $(first(kwres[2]))availablebytes = 1000_000_000)
$(last(kwres[2]))

julia> $sbs

julia> sbs(model, TrainBatchSize(512); $(first(kwres[3]))availablebytes = 10_000_000)
$(last(kwres[3]))

julia> sbs(model, TrainBatchSize(512); $(first(kwres[4]))availablebytes = 1000_000_000)
$(last(kwres[4]))

julia> bs(model, ValidationBatchSize(512); $(first(kwres[5]))availablebytes=10_000_000)
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
    BatchSizeSelectionWithDefaultInShape(batchsizefun, default_inshape)

Batch size selection with a default assumed inshape used for estimating valid batch sizes.

$(generic_batchsizefun_docstring())

Return the result of `batchsizefun` with default value of `inshape_nobatch = default_inshape` when called as a function.

Composable with other batch size selection types which may be used as `batchsizefun`. See examples.

# Examples
```jldoctest
julia> using NaiveGAflux

$(generic_batchsizefun_testgraph())

julia> bs = BatchSizeSelectionWithDefaultInShape((32,32,3));

$(generic_batchsizeselection_example(
    "sbs = BatchSizeSelectionWithDefaultInShape(BatchSizeSelectionScaled(0.5), (32,32,3))",
    "" => "120",
    "" => "512",
    "" => "60",
    "" => "512",
    "" => "243"))
```
"""
struct BatchSizeSelectionWithDefaultInShape{T, F}
    batchsizefun::F
    default_inshape::T
end
function BatchSizeSelectionWithDefaultInShape(default_inshape) 
        BatchSizeSelectionWithDefaultInShape(limit_maxbatchsize, default_inshape)
end
function (bs::BatchSizeSelectionWithDefaultInShape)(args...; inshape_nobatch=bs.default_inshape ,kwargs...) 
        bs.batchsizefun(args...; inshape_nobatch, kwargs...)
end


"""
    BatchSizeSelectionScaled{F}
    BatchSizeSelectionScaled(scale)
    BatchSizeSelectionScaled(batchsizefun, scale)

Batch size selection with a margin applied when estimating valid batch sizes.

$(generic_batchsizefun_docstring())

Return the result of `batchsizefun` with default value of `availablebytes = floor(scale * availablebytes)` when called as a function.

Composable with other batch size selection types which may be used as `batchsizefun`. See examples.

# Examples
```jldoctest
julia> using NaiveGAflux

$(generic_batchsizefun_testgraph())

julia> bs = BatchSizeSelectionScaled(0.5);

$(generic_batchsizeselection_example(
    "sbs = BatchSizeSelectionScaled(BatchSizeSelectionWithDefaultInShape((32,32,3)), 0.5);",
    "inshape_nobatch=(32,32,3), " => "60",
    "inshape_nobatch=(32,32,3), " => "512",
    "" => "60",
    "" => "512",
    "inshape_nobatch=(32,32,3), " => "121"))
```
"""
struct BatchSizeSelectionScaled{F}
    batchsizefun::F
    scale::Float64
end
BatchSizeSelectionScaled(scale::AbstractFloat) = BatchSizeSelectionScaled(limit_maxbatchsize, scale)  
function (bs::BatchSizeSelectionScaled)(args...; availablebytes=_availablebytes(), kwargs...) 
    bs.batchsizefun(args...;availablebytes = floor(Int, bs.scale * availablebytes), kwargs...)
end


struct BatchSizeSelectionFromAlternatives{F, T}
    batchsizefun::F
    alts::T
end
BatchSizeSelectionFromAlternatives(alts) = BatchSizeSelectionFromAlternatives(limit_maxbatchsize, alts)

function (bs::BatchSizeSelectionFromAlternatives)(args...;kwargs...) 
    select_bestfit_smaller(bs.batchsizefun(args...;kwargs...), bs.alts)
end

function select_bestfit_smaller(bs::Integer, alts)
    validalts = filter(<=(bs), alts)
    isempty(validalts) && return nothing
    argmin(x -> abs(bs - x), validalts)
end

struct BatchSizeSelectionMaxSize{F}
    batchsizefun::F
    uppersize::Int
end
BatchSizeSelectionMaxSize(uppersize=1024) = BatchSizeSelectionMaxSize(limit_maxbatchsize, uppersize)
function (bs::BatchSizeSelectionMaxSize)(c, ::Any, args...; kwargs...)
     bs.batchsizefun(c, bs.uppersize, args...; kwargs...)
end

function limit_maxbatchsize(model, tbs::TrainBatchSize; inshape_nobatch, availablebytes = _availablebytes())
    min(batchsize(tbs), maxtrainbatchsize(model, inshape_nobatch, availablebytes))
end

function limit_maxbatchsize(model,
                            tbs::ValidationBatchSize; 
                            inshape_nobatch,
                            availablebytes = _availablebytes()
                            )
    min(batchsize(tbs), maxvalidationbatchsize(model, inshape_nobatch, availablebytes))
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
