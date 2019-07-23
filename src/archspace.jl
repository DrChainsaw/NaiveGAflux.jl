"""
    AbstractArchSpace

Abstract functor representing an architecture search space.

Architecture spaces define a range of possible hyperparameters for a model architecture. Used to create new models or parts of models.

Return an `AbstractVertex` when invoked with an input `AbstractVertex`.
"""
abstract type AbstractArchSpace end

"""
    BaseLayerSpace

Generates basic attributes common to all layers.
"""
struct BaseLayerSpace
    nouts::AbstractVector{<:Integer}
    acts::AbstractVector
end
gen_nout(s::BaseLayerSpace, rng=rng_default) = rand(rng, s.nouts)
gen_act(s::BaseLayerSpace, rng=rng_default) = rand(rng, s.acts)

"""
    AbstractParSpace{N}

Abstract functor representing a seach space of `N`D parameters.

Typically used for convolutional layers, examples are kernel size, strides, dilation and padding.
"""
abstract type AbstractParSpace{N} end

"""
    SingletonParSpace{N} <:AbstractParSpace{N}

Singleton search space.
"""
struct SingletonParSpace{N} <:AbstractParSpace{N}
    p::NTuple{N, <:Integer}
end
SingletonParSpace(p::Integer...) = SingletonParSpace(p)
Singleton2DParSpace(p::Integer) = SingletonParSpace(p,p)
(s::SingletonParSpace)(rng=nothing) = s.p
(s::SingletonParSpace{1})(rng=nothing) = s.p[1]

"""
    RandParSpace{N}

Random search space for parameters.

Generates independent uniform random values for all `N` dimensions.
"""
struct RandParSpace{N} <:AbstractParSpace{N}
    p::NTuple{N, AbstractVector{<:Integer}}
end
RandParSpace(p::AbstractVector{<:Integer}...) = RandParSpace(p)
Rand2DParSpace(p::AbstractVector{<:Integer}) = RandParSpace(p,p)
(s::RandParSpace)(rng=rng_default) = rand.((rng,), s.p)
(s::RandParSpace{1})(rng=rng_default) = rand(rng, s.p[1])

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
    AbstractLayerSpace

Abstract functor representing a search space for layers.

Generates a random layer in the search space given an input size and (optionally) a random number generator.
"""
abstract type AbstractLayerSpace end

"""
    ConvSpace{N}

Generation of basic `N`D convolutional layers.
"""
struct ConvSpace{N} <:AbstractLayerSpace
    base::BaseLayerSpace
    kernelsize::AbstractParSpace{N}
    stride::AbstractParSpace
    dilation::AbstractParSpace
    padding::AbstractPadSpace
end

Conv2DSpace(base::BaseLayerSpace, ks::AbstractVector{<:Integer}) = ConvSpace(base, ks,ks)
ConvSpace(base::BaseLayerSpace, ks::AbstractVector{<:Integer}...) = ConvSpace(base, RandParSpace(ks), SingletonParSpace(1), SingletonParSpace(1), SamePad())

function (s::ConvSpace)(insize::Integer, rng=rng_default; convfun = Conv)
    ks = s.kernelsize(rng)
    stride = s.stride(rng)
    dilation = s.dilation(rng)
    pad = s.padding(ks, dilation, rng)
    outsize = gen_nout(s.base, rng)
    act = gen_act(s.base, rng)
    return convfun(ks, insize=>outsize, act, pad=pad, stride=stride,dilation=dilation)
end
