"""
    AbstractArchSpec

Abstract base type for architecture specifications.

Architecture specifications define a range of possible hyperparameters for a model architecture. Used to create new models or parts of models.
"""
abstract type AbstractArchSpec end

"""
    BaseLayerSpec

Generates basic attributes common to all layers.
"""
struct BaseLayerSpec
    nouts::AbstractVector{<:Integer}
    acts::AbstractVector
end
gen_nout(s::BaseLayerSpec, rng=rng_default) = rand(rng, s.nouts)
gen_act(s::BaseLayerSpec, rng=rng_default) = rand(rng, s.acts)

"""
    AbstractParSpec{N}

Abstract type for generating `N`D parameters, typically for convolutional layers.

Examples are kernel size, strides, dilation and padding.
"""
abstract type AbstractParSpec{N} end

struct FixedNDParSpec{N} <: AbstractParSpec{N}
    p::NTuple{N, <:Integer}
end
FixedNDParSpec(p::Integer...) = FixedNDParSpec(p)
Fixed2DParSpec(p::Integer) = FixedNDParSpec(p,p)
SingletonNDParSpec(p::Integer, N::Integer) = FixedNDParSpec(ntuple(_ -> p, N))
(s::FixedNDParSpec)(rng=nothing) = s.p
(s::FixedNDParSpec{1})(rng=nothing) = s.p[1]

"""
    ParNDSpec{N}

Generates independent uniform random values for all `N` dimensions.
"""
struct ParNDSpec{N} <:AbstractParSpec{N}
    p::NTuple{N, AbstractVector{<:Integer}}
end
ParNDSpec(p::AbstractVector{<:Integer}...) = ParNDSpec(p)
Par2DSpec(p::AbstractVector{<:Integer}) = ParNDSpec(p,p)
(s::ParNDSpec)(rng=rng_default) = rand.((rng,), s.p)
(s::ParNDSpec{1})(rng=rng_default) = rand(rng, s.p[1])

"""
    AbstractPadSpec

Abstract type for generating padding parameters for conv layers.
"""
abstract type AbstractPadSpec end

"""
    SamePad

Generates padding parameters so that outputshape == inputshape ./ stride.

Formula from Relationship 14 in http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html
"""
struct SamePad <:AbstractPadSpec end
function (::SamePad)(ks, dilation, rng=nothing)
    # Effective kernel size, including dilation
    ks_eff = @. ks + (ks - 1) * (dilation - 1)
    # How much total padding needs to be applied?
    pad_amt = @. ks_eff - 1
    # In case amount of padding is odd we need to apply different amounts to each side.
    return Tuple(mapfoldl(i -> [ceil(Int, i/2), i ÷ 2], vcat, pad_amt))
end

"""
    ConvSpec{N}

Specification for generation of basic `N`D convolutional layers.
"""
struct ConvSpec{N}
    base::BaseLayerSpec
    kernelsize::AbstractParSpec{N}
    stride::AbstractParSpec
    dilation::AbstractParSpec
    padding::AbstractPadSpec
end

Conv2DSpec(base::BaseLayerSpec, ks::AbstractVector{<:Integer}) = ConvSpec(base, ks,ks)
ConvSpec(base::BaseLayerSpec, ks::AbstractVector{<:Integer}...) = ConvSpec(base, ParNDSpec(ks), FixedNDParSpec(1), FixedNDParSpec(1), SamePad())

function (s::ConvSpec)(insize::Integer, rng=rng_default; convfun = Conv)
    ks = s.kernelsize(rng)
    stride = s.stride(rng)
    dilation = s.dilation(rng)
    pad = s.padding(ks, dilation, rng)
    outsize = gen_nout(s.base, rng)
    act = gen_act(s.base, rng)
    return convfun(ks, insize=>outsize, act, pad=pad, stride=stride,dilation=dilation)
end
