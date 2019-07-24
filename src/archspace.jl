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
    BaseLayerSpace

Search space for basic attributes common to all layers.

Return a tuple `(outsize,activation)` from the search space when invoked.
"""
struct BaseLayerSpace{T}
    nouts::AbstractParSpace{1, <:Integer}
    acts::AbstractParSpace{1, T}
end
BaseLayerSpace(n::Integer, act) = BaseLayerSpace(SingletonParSpace(n), SingletonParSpace(act))
BaseLayerSpace(n::AbstractVector{<:Integer}, act::AbstractVector{T}) where T = BaseLayerSpace(ParSpace(n), ParSpace(act))
(s::BaseLayerSpace)(rng=rng_default) = s.nouts(rng), s.acts(rng)


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

"""
    DenseSpace <:AbstractLayerSpace

Search space of Dense layers.
"""
struct DenseSpace <:AbstractLayerSpace
    base::BaseLayerSpace
end
(s::DenseSpace)(in::Integer,rng=rng_default) = Dense(in, s.base()...)

"""
    ConvSpace{N} <:AbstractLayerSpace

Search space of basic `N`D convolutional layers.
"""
struct ConvSpace{N} <:AbstractLayerSpace
    base::BaseLayerSpace
    kernelsize::AbstractParSpace{N, <:Integer}
    stride::AbstractParSpace
    dilation::AbstractParSpace
    padding::AbstractPadSpace
end

ConvSpace2D(base::BaseLayerSpace, ks::AbstractVector{<:Integer}) = ConvSpace(base, ks,ks)
ConvSpace(base::BaseLayerSpace, ks::AbstractVector{<:Integer}...) = ConvSpace(base, ParSpace(ks), SingletonParSpace(1), SingletonParSpace(1), SamePad())

function (s::ConvSpace)(insize::Integer, rng=rng_default; convfun = Conv)
    ks = Tuple(s.kernelsize(rng))
    stride = s.stride(rng)
    dilation = s.dilation(rng)
    pad = s.padding(ks, dilation, rng)
    outsize, act = s.base(rng)
    return convfun(ks, insize=>outsize, act, pad=pad, stride=stride,dilation=dilation)
end

"""
    BatchNormSpace <:AbstractLayerSpace

Search space of BatchNorm layers.
"""
struct BatchNormSpace <:AbstractLayerSpace
    acts::AbstractParSpace
end
BatchNormSpace(act::Function) = BatchNormSpace(SingletonParSpace(act))
BatchNormSpace(act, acts...) = BatchNormSpace(ParSpace1D(act,acts...))
(s::BatchNormSpace)(in::Integer, rng=rng_default) = BatchNorm(in, s.acts(rng))

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
PoolSpace(ws::AbstractParSpace, stride::AbstractParSpace) = PoolSpace(ws,stride, SamePad())
function (s::PoolSpace)(in::Integer, rng=rng_default;pooltype)
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
(s::MaxPoolSpace)(in::Integer, rng=rng_default) = s.s(in, rng, pooltype=MaxPool)


default_logging() = logged(level=Base.CoreLogging.Info, info=NameAndIOInfoStr())
"""
    VertexConf

Generic configuration template for computation graph vertices.

Intention is to make it easy to add logging, validation and pruning metrics in an uniform way.
"""
struct VertexConf
    layerfun
    traitfun
end
VertexConf() = VertexConf(ActivationContribution ∘ LazyMutable, validated() ∘ default_logging())

(c::VertexConf)(in::AbstractVertex, l) = mutable(l,in,layerfun=c.layerfun, mutation=IoChange, traitfun=c.traitfun)
(c::VertexConf)(name::String, in::AbstractVertex, l) = mutable(name, l,in,layerfun=c.layerfun, mutation=IoChange, traitfun=c.traitfun)

Base.Broadcast.broadcastable(c::VertexConf) = Ref(c)

"""
    ConcConf

Generic configuration template for concatenation of vertex outputs.
"""
struct ConcConf
    traitfun
end
ConcConf() = ConcConf(validated() ∘ default_logging())

(c::ConcConf)(in::AbstractVector{<:AbstractVertex}) = c(in...)
(c::ConcConf)(in::AbstractVertex) = in
(c::ConcConf)(ins::AbstractVertex...) = concat(ins...,mutation=IoChange, traitdecoration = c.traitfun)
(c::ConcConf)(name::String, in::AbstractVector{<:AbstractVertex}) = c(name, in...)
(c::ConcConf)(name::String, in::AbstractVertex) = in
(c::ConcConf)(name::String, ins::AbstractVertex...) = concat(ins...,mutation=IoChange, traitdecoration = c.traitfun ∘ named(name))

"""
    VertexSpace <:AbstractArchSpace

Search space of one `AbstractVertex` from one `AbstractLayerSpace`.
"""
struct VertexSpace <:AbstractArchSpace
    conf::VertexConf
    lspace::AbstractLayerSpace
end
VertexSpace(lspace::AbstractLayerSpace) = VertexSpace(VertexConf(), lspace)

(s::VertexSpace)(in::AbstractVertex, rng=rng_default) = s.conf(in, s.lspace(nout(in), rng))
(s::VertexSpace)(name::String, in::AbstractVertex, rng=rng_default) = s.conf(name, in, s.lspace(nout(in), rng))

"""
    ArchSpace <:AbstractArchSpace

Search space of `AbstractArchSpace`.
"""
struct ArchSpace <:AbstractArchSpace
    s::AbstractParSpace{1, <:AbstractArchSpace}
end
ArchSpace(l::AbstractLayerSpace;conf=VertexConf()) = ArchSpace(SingletonParSpace(VertexSpace(conf,l)))
ArchSpace(l::AbstractLayerSpace, ls::AbstractLayerSpace...;conf=VertexConf()) = ArchSpace(ParSpace(VertexSpace.(conf,[l,ls...])))

(s::ArchSpace)(in::AbstractVertex, rng=rng_default) = s.s(rng)(in, rng)
(s::ArchSpace)(name::String, in::AbstractVertex, rng=rng_default) = s.s(rng)(name, in, rng)

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

(s::RepeatArchSpace)(in::AbstractVertex, rng=rng_default) = foldl((next, i) -> s.s(next, rng), 1:s.r(rng), init=in)
(s::RepeatArchSpace)(name::String, in::AbstractVertex, rng=rng_default) = foldl((next, i) -> s.s(join([name,".", i]), next, rng), 1:s.r(rng), init=in)

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

(s::ForkArchSpace)(in::AbstractVertex, rng=rng_default) = s.c(map(i -> s.s(in, rng), 1:s.p(rng)))
(s::ForkArchSpace)(name::String, in::AbstractVertex, rng=rng_default) = s.c(name, map(i -> s.s(join([name, ".path", i]), in, rng), 1:s.p(rng)))
