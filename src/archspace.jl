"""
    AbstractArchSpace

Abstract functor representing an architecture search space.

Architecture spaces define a range of possible hyperparameters for a model architecture. Used to create new models or parts of models.

Return an `AbstractVertex` from the search space when invoked with an input `AbstractVertex`.
"""
abstract type AbstractArchSpace end

"""
    AbstractLayerSpace

Abstract functor representing a search space of layers.

Return a layer in the search space given an input size and (optionally) a random number generator.
"""
abstract type AbstractLayerSpace end

"""
    AbstractParSpace{N, T}

Abstract functor representing a seach space of `N`D parameters of type `T`.

Return a tuple of parameters from the search space when invoked. Random number generator may be supplied.
"""
abstract type AbstractParSpace{N, T} end

"""
    AbstractWeightInit

Abstract type for weight initialization strategies
"""
abstract type AbstractWeightInit end

"""
    DefaultWeightInit <: AbstractWeightInit

Use the layer default weight initialization.
"""
struct DefaultWeightInit <: AbstractWeightInit end

"""
    IdentityWeightInit <: AbstractWeightInit
    IdentityWeightInit()

Initialize weights with an identity mapping.
"""
struct IdentityWeightInit <: AbstractWeightInit end

"""
    PartialIdentityWeightInit <: AbstractWeightInit
    PartialIdentityWeightInit(inoffset, outoffset)

Initialize weights with an identity mapping.

Parameter `inoffset` and `outoffset` shifts the weigths in case `nin != nout`. This is so that a concatenation can also be an identity mapping as a whole.
"""
struct PartialIdentityWeightInit <: AbstractWeightInit
    inoffset::Int
    outoffset::Int
end

"""
    ZeroWeightInit <: AbstractWeightInit
    ZeroWeightInit()

Initialize weights as zeros.

Main use case is to let residual layers be and identity mapping
"""
struct ZeroWeightInit <: AbstractWeightInit end


"""
    BaseLayerSpace

Search space for basic attributes common to all layers.
"""
struct BaseLayerSpace{T}
    nouts::AbstractParSpace{1, <:Integer}
    acts::AbstractParSpace{1, T}
end
BaseLayerSpace(n::Integer, act) = BaseLayerSpace(SingletonParSpace(n), SingletonParSpace(act))
BaseLayerSpace(n::AbstractVector{<:Integer}, act::AbstractVector{T}) where T = BaseLayerSpace(ParSpace(n), ParSpace(act))
outsize(s::BaseLayerSpace, rng=rng_default) = s.nouts(rng)
activation(s::BaseLayerSpace, rng=rng_default) = s.acts(rng)


"""
    SingletonParSpace{N, T} <:AbstractParSpace{N, T}

Singleton search space. Has exactly one value per dimension.
"""
struct SingletonParSpace{N, T} <:AbstractParSpace{N, T}
    p::NTuple{N, T}
end
SingletonParSpace(p::T...) where T = SingletonParSpace(p)
Singleton2DParSpace(p::T) where T = SingletonParSpace(p,p)
(s::SingletonParSpace)(rng=nothing) = s.p
(s::SingletonParSpace{1, T})(rng=nothing) where T = s.p[1]

"""
    ParSpace{N, T} <:AbstractParSpace{N, T}

Search space for parameters.

Return independent uniform random values for all `N` dimensions from the search space when invoked.
"""
struct ParSpace{N, T} <:AbstractParSpace{N, T}
    p::NTuple{N, AbstractVector{T}}
end
ParSpace(p::AbstractVector{T}...) where T = ParSpace(p)
ParSpace1D(p...) = ParSpace(collect(p))
ParSpace2D(p::AbstractVector{T}) where T = ParSpace(p,p)
(s::ParSpace)(rng=rng_default) = rand.((rng,), s.p)
(s::ParSpace{1, T})(rng=rng_default) where T = rand(rng, s.p[1])


struct CoupledParSpace{N, T} <:AbstractParSpace{N, T}
    p::AbstractParSpace{1, T}
end
CoupledParSpace(p::AbstractParSpace{1, T}, N) where T = CoupledParSpace{N}(p)
CoupledParSpace(p::AbstractVector{T}, N) where T = CoupledParSpace{N, T}(ParSpace(p))
function(s::CoupledParSpace{N})(rng=rng_default) where N
    val = s.p(rng)
    return ntuple(i -> val, N)
end

"""
    AbstractPadSpace

Abstract type for generating padding parameters for conv layers.
"""
abstract type AbstractPadSpace end

"""
    SamePad <:AbstractPadSpace

Generates padding parameters so that outputshape == inputshape ./ stride.

Formula from Relationship 14 in http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html
"""
struct SamePad <:AbstractPadSpace end
function (::SamePad)(ks, dilation, rng=nothing)
    # Effective kernel size, including dilation
    ks_eff = @. ks + (ks - 1) * (dilation - 1)
    # How much total padding needs to be applied?
    pad_amt = @. ks_eff - 1
    # In case amount of padding is odd we need to apply different amounts to each side.
    return Tuple(mapfoldl(i -> [ceil(Int, i/2), i ÷ 2], vcat, pad_amt))
end

struct NoPad <:AbstractPadSpace end
(::NoPad)(ks, dilation, rng=nothing) = 0

"""
    NamedLayerSpace <:AbstractLayerSpace

Adds a `name` to an `AbstractLayerSpace`
"""
struct NamedLayerSpace <:AbstractLayerSpace
    name::String
    s::AbstractLayerSpace
end
NaiveNASlib.name(::AbstractLayerSpace) = ""
NaiveNASlib.name(s::NamedLayerSpace) = s.name
function (s::NamedLayerSpace)(in::Integer,rng=rng_default; outsize=missing, wi=DefaultWeightInit())
    ismissing(outsize) && return s.s(in, rng)
    return s.s(in, rng, outsize=outsize, wi=wi)
end

"""
    LoggingLayerSpace <: AbstractLayerSpace
    LoggingLayerSpace(s::AbstractLayerSpace)
    LoggingLayerSpace(level::LogLevel, s::AbstractLayerSpace)
    LoggingLayerSpace(level::LogLevel, msgfun::Function, s::AbstractLayerSpace)

Logs `msgfun(layer)` at loglevel `level` after creating a `layer` from `s`.
"""
struct LoggingLayerSpace <: AbstractLayerSpace
    level::LogLevel
    msgfun::Function
    s::AbstractLayerSpace
end
LoggingLayerSpace(s::AbstractLayerSpace) = LoggingLayerSpace(Logging.Debug, s)
LoggingLayerSpace(level::LogLevel, s::AbstractLayerSpace) = LoggingLayerSpace(level, l -> "Create $l from $(name(s))", s)
NaiveNASlib.name(s::LoggingLayerSpace) = name(s.s)
function (s::LoggingLayerSpace)(in::Integer,rng=rng_default; outsize=missing, wi=DefaultWeightInit())
    layer = ismissing(outsize) ? s.s(in, rng) : s.s(in, rng, outsize=outsize, wi=wi)
    msg = s.msgfun(layer)
    @logmsg s.level msg
    return layer
end

"""
    DenseSpace <:AbstractLayerSpace
    DenseSpace(base::BaseLayerSpace)
    DenseSpace(outsizes, activations)

Search space of Dense layers.
"""
struct DenseSpace <:AbstractLayerSpace
    base::BaseLayerSpace
end
DenseSpace(outsizes, activations) = DenseSpace(BaseLayerSpace(outsizes, activations))
(s::DenseSpace)(in::Integer,rng=rng_default; outsize=outsize(s.base,rng), wi=DefaultWeightInit(), densefun=Dense) = densefun(in, outsize, activation(s.base,rng); denseinitW(wi)...)

denseinitW(::DefaultWeightInit) = ()
denseinitW(::IdentityWeightInit) = (initW = idmapping,)
denseinitW(wi::PartialIdentityWeightInit) = (initW = (args...) -> circshift(idmapping_nowarn(args...),(wi.outoffset, wi.inoffset)),)
denseinitW(::ZeroWeightInit) = (initW = zeros,)

"""
    ConvSpace{N} <:AbstractLayerSpace
    ConvSpace2D(outsizes, activations, ks)
    ConvSpace2D(base::BaseLayerSpace, ks::AbstractVector{<:Integer})
    ConvSpace(outsizes, activations, ks::AbstractVector{<:Integer}...)
    ConvSpace(outsizes, activations, ks::AbstractParSpace)
    ConvSpace(base::BaseLayerSpace, ks::AbstractVector{<:Integer}...)
    ConvSpace(base::BaseLayerSpace, ks::AbstractParSpace)
    ConvSpace(base::BaseLayerSpace, ks::AbstractParSpace, stride::AbstractParSpace, dilation::AbstractParSpace, padding::AbstractPadSpace)


Search space of basic `N`D convolutional layers.
"""
struct ConvSpace{N} <:AbstractLayerSpace
    base::BaseLayerSpace
    kernelsize::AbstractParSpace{N, <:Integer}
    stride::AbstractParSpace
    dilation::AbstractParSpace
    padding::AbstractPadSpace
end
ConvSpace2D(outsizes, activations, ks) = ConvSpace2D(BaseLayerSpace(outsizes, activations), ks)
ConvSpace2D(base::BaseLayerSpace, ks::AbstractVector{<:Integer}) = ConvSpace(base, ks,ks)
ConvSpace(outsizes, activations, ks::AbstractVector{<:Integer}...) = ConvSpace(BaseLayerSpace(outsizes, activations), ks...)
ConvSpace(outsizes, activations, ks::AbstractParSpace) = ConvSpace(BaseLayerSpace(outsizes, activations), ks)
ConvSpace(base::BaseLayerSpace, ks::AbstractVector{<:Integer}...) = ConvSpace(base, ParSpace(ks))
ConvSpace(base::BaseLayerSpace, ks::AbstractParSpace) = ConvSpace(base, ks, SingletonParSpace(1), SingletonParSpace(1), SamePad())

function (s::ConvSpace)(insize::Integer, rng=rng_default; outsize = outsize(s.base, rng), wi=DefaultWeightInit(), convfun = Conv)
    ks = Tuple(s.kernelsize(rng))
    stride = s.stride(rng)
    dilation = s.dilation(rng)
    pad = s.padding(ks, dilation, rng)
    act = activation(s.base, rng)
    return convfun(ks, insize=>outsize, act; pad=pad, stride=stride,dilation=dilation, convinitW(wi)...)
end

convinitW(::DefaultWeightInit) = ()
convinitW(::IdentityWeightInit) = (init=idmapping,)
convinitW(wi::PartialIdentityWeightInit) = (init = (args...) -> circshift(idmapping_nowarn(args...), (0,0,wi.inoffset, wi.outoffset)),)
convinitW(::ZeroWeightInit) = (init=(args...) -> zeros(Float32,args...),)

"""
    BatchNormSpace <:AbstractLayerSpace

Search space of BatchNorm layers.
"""
struct BatchNormSpace <:AbstractLayerSpace
    acts::AbstractParSpace
end
BatchNormSpace(act::Function) = BatchNormSpace(SingletonParSpace(act))
BatchNormSpace(act, acts...) = BatchNormSpace(ParSpace1D(act,acts...))
BatchNormSpace(acts::AbstractVector) = BatchNormSpace(acts...)
(s::BatchNormSpace)(in::Integer, rng=rng_default;outsize=nothing, wi=nothing) = BatchNorm(in, s.acts(rng))

"""
    PoolSpace{N} <:AbstractLayerSpace

Search space of `N`D pooling layers.
"""
struct PoolSpace{N} <:AbstractLayerSpace
    ws::AbstractParSpace{N, <:Integer}
    stride::AbstractParSpace
    pad::AbstractPadSpace
end
PoolSpace2D(ws::AbstractVector{<:Integer}) = PoolSpace(ws,ws)
PoolSpace(ws::AbstractVector{<:Integer}...) = PoolSpace(ParSpace(ws), ParSpace(ws))
PoolSpace(ws::AbstractParSpace, stride::AbstractParSpace) = PoolSpace(ws,stride, NoPad())
function (s::PoolSpace)(in::Integer, rng=rng_default;outsize=nothing, pooltype)
    ws = Tuple(s.ws(rng))
    stride = s.stride(rng)
    pad = s.pad(ws, 1, rng)
    pooltype(ws, stride=stride, pad=pad)
end

"""
    MaxPoolSpace{N} <: AbstractLayerSpace

Search space of `N`D max pooling layers.
"""
struct MaxPoolSpace{N} <: AbstractLayerSpace
    s::PoolSpace{N}
end
(s::MaxPoolSpace)(in::Integer, rng=rng_default;outsize=nothing, wi=nothing) = s.s(in, rng, pooltype=MaxPool)


default_logging() = logged(level=Logging.Debug, info=NameAndIOInfoStr())
"""
    LayerVertexConf

Generic configuration template for computation graph vertices.

Intention is to make it easy to add logging, validation and pruning metrics in an uniform way.
"""
struct LayerVertexConf
    layerfun
    traitfun
end
LayerVertexConf() = LayerVertexConf(ActivationContribution ∘ LazyMutable, validated() ∘ default_logging())
Shielded(base=LayerVertexConf()) = LayerVertexConf(base.layerfun, MutationShield ∘ base.traitfun)


(c::LayerVertexConf)(in::AbstractVertex, l) = mutable(l,in,layerfun=c.layerfun, mutation=IoChange, traitfun=c.traitfun)
(c::LayerVertexConf)(name::String, in::AbstractVertex, l) = mutable(name, l,in,layerfun=c.layerfun, mutation=IoChange, traitfun=c.traitfun)

Base.Broadcast.broadcastable(c::LayerVertexConf) = Ref(c)

"""
    ConcConf

Generic configuration template for concatenation of vertex outputs.
"""
struct ConcConf
    layerfun
    traitfun
end
ConcConf() = ConcConf(ActivationContribution, validated() ∘ default_logging())

(c::ConcConf)(in::AbstractVector{<:AbstractVertex}) = c(in...)
(c::ConcConf)(in::AbstractVertex) = in
(c::ConcConf)(ins::AbstractVertex...) = concat(ins...,mutation=IoChange, traitfun = c.traitfun, layerfun=c.layerfun)
(c::ConcConf)(name::String, in::AbstractVector{<:AbstractVertex}) = c(name, in...)
(c::ConcConf)(name::String, in::AbstractVertex) = in
(c::ConcConf)(name::String, ins::AbstractVertex...) = concat(ins...,mutation=IoChange, traitfun = c.traitfun ∘ named(name), layerfun=c.layerfun)


"""
    LoggingArchSpace <: AbstractArchSpace
    LoggingArchSpace(s::AbstractArchSpace)
    LoggingArchSpace(level::LogLevel, s::AbstractArchSpace)
    LoggingArchSpace(level::LogLevel, msgfun::Function, s::AbstractArchSpace)

Logs `msgfun(vertex)` at loglevel `level` after creating a `vertex` from `s`.
"""
struct LoggingArchSpace <: AbstractArchSpace
    level::LogLevel
    msgfun::Function
    s::AbstractArchSpace
end
LoggingArchSpace(s::AbstractArchSpace) = LoggingArchSpace(Logging.Debug, s)
LoggingArchSpace(level::LogLevel, s::AbstractArchSpace) = LoggingArchSpace(level, v -> "Created $(name(v))", s)
function (s::LoggingArchSpace)(in::AbstractVertex,rng=rng_default; outsize=missing, wi=DefaultWeightInit())
    layer = ismissing(outsize) ? s.s(in, rng) : s.s(in, rng, outsize=outsize, wi=wi)
    @logmsg s.level s.msgfun(layer)
    return layer
end
function (s::LoggingArchSpace)(namestr::String, in::AbstractVertex,rng=rng_default; outsize=missing, wi=DefaultWeightInit())
    layer = ismissing(outsize) ? s.s(namestr, in, rng) : s.s(namestr, in, rng, outsize=outsize, wi=wi)
    @logmsg s.level s.msgfun(layer)
    return layer
end

"""
    VertexSpace <:AbstractArchSpace

Search space of one `AbstractVertex` from one `AbstractLayerSpace`.
"""
struct VertexSpace <:AbstractArchSpace
    conf::LayerVertexConf
    lspace::AbstractLayerSpace
end
VertexSpace(lspace::AbstractLayerSpace) = VertexSpace(LayerVertexConf(), lspace)

(s::VertexSpace)(in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit()) = s.conf(in,  create_layer(outsize, nout(in), s.lspace, wi, rng))
(s::VertexSpace)(namestr::String, in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit()) = s.conf(join(filter(!isempty, [namestr, name(s.lspace)]), "."), in, create_layer(outsize, nout(in), s.lspace, wi, rng))
create_layer(::Missing, insize::Integer, ls::AbstractLayerSpace, wi, rng) = ls(insize, rng, wi=wi)
create_layer(outsize::Integer, insize::Integer, ls::AbstractLayerSpace, wi, rng) = ls(insize, rng, outsize=outsize, wi=wi)

"""
    ArchSpace <:AbstractArchSpace

Search space of `AbstractArchSpace`.
"""
struct ArchSpace <:AbstractArchSpace
    s::AbstractParSpace{1, <:AbstractArchSpace}
end
ArchSpace(l::AbstractLayerSpace;conf=LayerVertexConf()) = ArchSpace(SingletonParSpace(VertexSpace(conf,l)))
ArchSpace(l::AbstractLayerSpace, ls::AbstractLayerSpace...;conf=LayerVertexConf()) = ArchSpace(ParSpace(VertexSpace.(conf,[l,ls...])))

(s::ArchSpace)(in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit()) = s.s(rng)(in, rng, outsize=outsize, wi=wi)
(s::ArchSpace)(name::String, in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit()) = s.s(rng)(name, in, rng, outsize=outsize, wi=wi)

"""
    RepeatArchSpace <:AbstractArchSpace

Search space of repetitions of another `AbstractArchSpace`.

Output of each generated candidate is input to next and the last is returned.

Number of repetitions comes from an `AbstractParSpace`.
"""
struct RepeatArchSpace <:AbstractArchSpace
    s::AbstractArchSpace
    r::AbstractParSpace{1, <:Integer}
end
RepeatArchSpace(s::AbstractArchSpace, r::Integer) = RepeatArchSpace(s, SingletonParSpace(r))
RepeatArchSpace(s::AbstractArchSpace, r::AbstractVector{<:Integer}) = RepeatArchSpace(s, ParSpace(r))

(s::RepeatArchSpace)(in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit()) = foldl((next, i) -> s.s(next, rng, outsize=outsize, wi=repeatinitW(wi, next, outsize)), 1:s.r(rng), init=in)
(s::RepeatArchSpace)(name::String, in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit()) = foldl((next, i) -> s.s(join([name,".", i]), next, rng, outsize=outsize, wi=repeatinitW(wi, next, outsize)), 1:s.r(rng), init=in)

repeatinitW(wi::AbstractWeightInit, invertex, outsize) = wi
repeatinitW(wi::PartialIdentityWeightInit, invertex, outsize) = nout(invertex) == outsize ? IdentityWeightInit() : wi
repeatinitW(wi::PartialIdentityWeightInit, invertex, ::Missing) = wi

"""
    ListArchSpace <:AbstractArchSpace

Search space composed of a list of search spaces.

Basically a more deterministic version of RepeatArchSpace.
"""
struct ListArchSpace <:AbstractArchSpace
    s::AbstractVector{<:AbstractArchSpace}
end
ListArchSpace(s::AbstractArchSpace...) = ListArchSpace(collect(s))
(s::ListArchSpace)(in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit()) = foldl((next, ss) -> ss(next, rng, outsize=outsize, wi=repeatinitW(wi, next, outsize)), s.s, init=in)
(s::ListArchSpace)(name::String, in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit()) = foldl((next, i) -> s.s[i](join([name,".", i]), next, rng, outsize=outsize, wi=repeatinitW(wi, next, outsize)), eachindex(s.s), init=in)


"""
    ForkArchSpace <:AbstractArchSpace

Search space of parallel paths from another `AbstractArchSpace`.

Input vertex is input to a number of paths drawn from an `AbstractParSpace`. Concatenation of paths is output.
"""
struct ForkArchSpace <:AbstractArchSpace
    s::AbstractArchSpace
    p::AbstractParSpace{1, <:Integer}
    c::ConcConf
end
ForkArchSpace(s::AbstractArchSpace, r::Integer; conf=ConcConf()) = ForkArchSpace(s, SingletonParSpace(r), conf)
ForkArchSpace(s::AbstractArchSpace, r::AbstractVector{<:Integer}; conf=ConcConf()) = ForkArchSpace(s, ParSpace(r), conf)

function (s::ForkArchSpace)(in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit())
     # Make sure there are no paths with size 0, which is what happens if np > outsize
     np=min_nomissing(s.p(rng), outsize)
     np == 0 && return in
     outsizes = eq_split(outsize, np)
     return s.c(map(i -> s.s(in, rng, outsize=outsizes[i], wi=forkinitW(wi, outsizes, i)), 1:np))
 end
function (s::ForkArchSpace)(name::String, in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit())
    # Make sure there are no paths with size 0, which is what happens if np > outsize
    np=min_nomissing(s.p(rng), outsize)
    np == 0 && return in
    outsizes = eq_split(outsize, np)
    return s.c(name * ".cat", map(i -> s.s(join([name, ".path", i]), in, rng,outsize=outsizes[i], wi=forkinitW(wi, outsizes, i)), 1:np))
end
min_nomissing(x, ::Missing) = x
min_nomissing(x, y) = min(x,y)

function eq_split(x, n)
    rem = x
    out = Vector(undef, n)
    for i = 1:n
        out[i] = rem ÷ (n-i+1)
        rem -= out[i]
    end
    return out
end

forkinitW(wi::AbstractWeightInit, outsizes, i) = wi
forkinitW(wi::IdentityWeightInit, outsizes, i) = PartialIdentityWeightInit(mapreduce(ii -> outsizes[ii], + , 1:i-1, init=0), 0)

"""
    ResidualArchSpace <:AbstractArchSpace

Turns the wrapped `AbstractArchSpace` into a residual.

Return `x = y + in` where `y` is drawn from the wrapped `AbstractArchSpace` when invoked with `in` as input vertex.
"""
struct ResidualArchSpace <:AbstractArchSpace
    s::AbstractArchSpace
    conf::VertexConf
end
ResidualArchSpace(s::AbstractArchSpace) = ResidualArchSpace(s, VertexConf(traitdecoration = validated() ∘ default_logging(), outwrap = ActivationContribution))
ResidualArchSpace(l::AbstractLayerSpace) = ResidualArchSpace(VertexSpace(l))

(s::ResidualArchSpace)(in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit()) = s.conf >> in + s.s(in, rng,outsize=nout(in), wi=resinitW(wi))
(s::ResidualArchSpace)(name::String, in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit()) = VertexConf(s.conf.mutation, s.conf.traitdecoration ∘ named(name * ".add"), s.conf.outwrap) >> in + s.s(join([name, ".res"]), in, rng,outsize=nout(in), wi=resinitW(wi))

resinitW(wi::AbstractWeightInit) = wi
resinitW(::Union{IdentityWeightInit, PartialIdentityWeightInit}) = ZeroWeightInit()

"""
    FunctionSpace <: AbstractArchSpace
    FunctionSpace(fun::Function, namesuff::String, conf=LayerVertexConf(ActivationContribution, validated() ∘ default_logging()))

Return a `SizeInvariant` vertex representing `fun(x)` when invoked with `in` as input vertex where `x` is output of `in`.
"""
struct FunctionSpace <: AbstractArchSpace
    conf::LayerVertexConf
    funspace::AbstractParSpace
    namesuff::String
end
funspace_default_conf() = LayerVertexConf(ActivationContribution, validated() ∘ default_logging())

FunctionSpace(funs...; namesuff::String, conf=funspace_default_conf()) = FunctionSpace(conf, ParSpace1D(funs...), namesuff)

(s::FunctionSpace)(in::AbstractVertex, rng=rng_default; outsize=nothing, wi=nothing) = funvertex(s, in, rng)
(s::FunctionSpace)(name::String, in::AbstractVertex, rng=rng_default; outsize=nothing, wi=nothing) = funvertex(join([name,s.namesuff]), s, in, rng)

funvertex(s::FunctionSpace, in::AbstractVertex, rng) = invariantvertex(s.conf.layerfun(s.funspace(rng)), in, mutation=IoChange, traitdecoration = s.conf.traitfun)

funvertex(name::String, s::FunctionSpace, in::AbstractVertex, rng) =
invariantvertex(s.conf.layerfun(s.funspace(rng)), in, mutation=IoChange, traitdecoration = s.conf.traitfun ∘ named(name))

"""
    GlobalPoolSpace2D()
    GlobalPoolSpace2D(conf::LayerVertexConf)

Short for `FunctionSpace` with global average or global max pooling.

Also adds a `MutationShield` to prevent the vertex from being removed by default.
"""
GlobalPoolSpace2D() = GlobalPoolSpace2D(LayerVertexConf(ActivationContribution, MutationShield ∘ validated() ∘ default_logging()))
GlobalPoolSpace2D(conf) = FunctionSpace(GlobalPool.(2, (MaxPool, MeanPool))..., namesuff = ".globpool", conf=conf)
# About 50% faster on GPU to create a MeanPool and use it compared to dropdims(mean(x, dims=[1:2]), dims=(1,2)). CBA to figure out why...
struct GlobalPool{N, PT} end
GlobalPool(nd::Int, PT) = GlobalPool{nd, PT}()
(::GlobalPool{N, PT})(x) where {N, PT} = dropdims(PT(size(x)[1:N])(x),dims=Tuple(1:N))

NaiveNASflux.layertype(gp::GlobalPool) = gp
NaiveNASflux.layer(gp::GlobalPool) = gp
NaiveNASlib.minΔninfactor(::GlobalPool) = 1
NaiveNASlib.minΔnoutfactor(::GlobalPool) = 1
function NaiveNASlib.mutate_inputs(::GlobalPool, args...) end
function NaiveNASlib.mutate_outputs(::GlobalPool, args...) end
