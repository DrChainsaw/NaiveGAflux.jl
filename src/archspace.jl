"""
    AbstractArchSpace

Abstract base type for an architecture space.

Architecture spaces define a range of possible hyperparameters for a model architecture. Used to create new models or parts of models.
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

Abstract type for generating `N`D parameters, typically for convolutional layers.

Examples are kernel size, strides, dilation and padding.
"""
abstract type AbstractParSpace{N} end

struct FixedNDParSpace{N} <: AbstractParSpace{N}
    p::NTuple{N, <:Integer}
end
FixedNDParSpace(p::Integer...) = FixedNDParSpace(p)
Fixed2DParSpace(p::Integer) = FixedNDParSpace(p,p)
SingletonNDParSpace(p::Integer, N::Integer) = FixedNDParSpace(ntuple(_ -> p, N))
(s::FixedNDParSpace)(rng=nothing) = s.p
(s::FixedNDParSpace{1})(rng=nothing) = s.p[1]

"""
    ParNDSpace{N}

Generates independent uniform random values for all `N` dimensions.
"""
struct ParNDSpace{N} <:AbstractParSpace{N}
    p::NTuple{N, AbstractVector{<:Integer}}
end
ParNDSpace(p::AbstractVector{<:Integer}...) = ParNDSpace(p)
Par2DSpace(p::AbstractVector{<:Integer}) = ParNDSpace(p,p)
(s::ParNDSpace)(rng=rng_default) = rand.((rng,), s.p)
(s::ParNDSpace{1})(rng=rng_default) = rand(rng, s.p[1])

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

Generation of basic `N`D convolutional layers.
"""
struct ConvSpace{N}
    base::BaseLayerSpace
    kernelsize::AbstractParSpace{N}
    stride::AbstractParSpace
    dilation::AbstractParSpace
    padding::AbstractPadSpace
end

Conv2DSpace(base::BaseLayerSpace, ks::AbstractVector{<:Integer}) = ConvSpace(base, ks,ks)
ConvSpace(base::BaseLayerSpace, ks::AbstractVector{<:Integer}...) = ConvSpace(base, ParNDSpace(ks), FixedNDParSpace(1), FixedNDParSpace(1), SamePad())

function (s::ConvSpace)(insize::Integer, rng=rng_default; convfun = Conv)
    ks = s.kernelsize(rng)
    stride = s.stride(rng)
    dilation = s.dilation(rng)
    pad = s.padding(ks, dilation, rng)
    outsize = gen_nout(s.base, rng)
    act = gen_act(s.base, rng)
    return convfun(ks, insize=>outsize, act, pad=pad, stride=stride,dilation=dilation)
end
