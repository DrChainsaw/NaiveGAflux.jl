"""
    AbstractArchSpace

Abstract functor representing an architecture search space.

Architecture spaces define a range of possible hyperparameters for a model architecture. Used to create new models or parts of models.

Return an `AbstractVertex` from the search space when invoked with an input `AbstractVertex`.
"""
abstract type AbstractArchSpace end

"""
    AbstractLayerSpace

Abstract functor representing a search space for layers.

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
    ParSpace{N}

Search space for parameters.

Return independent uniform random values for all `N` dimensions from the search space when invoked.
"""
struct ParSpace{N, T} <:AbstractParSpace{N, T}
    p::NTuple{N, AbstractVector{T}}
end
ParSpace(p::AbstractVector{T}...) where T = ParSpace(p)
ParSpace2D(p::AbstractVector{T}) where T = ParSpace(p,p)
(s::ParSpace)(rng=rng_default) = rand.((rng,), s.p)
(s::ParSpace{1, T})(rng=rng_default) where T = rand(rng, s.p[1])

"""
    AbstractPadSpace

Abstract type for generating padding parameters for conv layers.
"""
abstract type AbstractPadSpace end

"""
    SamePad

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
    ConvSpace{N}

Search space of basic `N`D convolutional layers.
"""
struct ConvSpace{N} <:AbstractLayerSpace
    base::BaseLayerSpace
    kernelsize::AbstractParSpace{N, <:Integer}
    stride::AbstractParSpace
    dilation::AbstractParSpace
    padding::AbstractPadSpace
end

Conv2DSpace(base::BaseLayerSpace, ks::AbstractVector{<:Integer}) = ConvSpace(base, ks,ks)
ConvSpace(base::BaseLayerSpace, ks::AbstractVector{<:Integer}...) = ConvSpace(base, ParSpace(ks), SingletonParSpace(1), SingletonParSpace(1), SamePad())

function (s::ConvSpace)(insize::Integer, rng=rng_default; convfun = Conv)
    ks = s.kernelsize(rng)
    stride = s.stride(rng)
    dilation = s.dilation(rng)
    pad = s.padding(ks, dilation, rng)
    outsize, act = s.base(rng)
    return convfun(ks, insize=>outsize, act, pad=pad, stride=stride,dilation=dilation)
end
