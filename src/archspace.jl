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
BaseLayerSpace(outsizes, acts) = BaseLayerSpace(parspaceof(outsizes), parspaceof(acts))
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

"""
    CoupledParSpace{N, T} <:AbstractParSpace{N, T}

Search space for parameters.

Return the same uniformly sampled value for all `N` dimensions from the search space when invoked.
"""
struct CoupledParSpace{N, T} <:AbstractParSpace{N, T}
    p::AbstractParSpace{1, T}
end
CoupledParSpace(p::AbstractParSpace{1, T}, N) where T = CoupledParSpace{N}(p)
CoupledParSpace(p::AbstractVector{T}, N) where T = CoupledParSpace{N, T}(ParSpace(p))
function(s::CoupledParSpace{N})(rng=rng_default) where N
    val = s.p(rng)
    return ntuple(i -> val, N)
end

parspaceof(x) = parspaceof(1, x)
parspaceof(N, x) = parspaceof(Val(N), x)
parspaceof(::Val{N}, s::AbstractParSpace{N}) where N = s
parspaceof(::Val{N}, s::AbstractParSpace{1}) where N = parspaceof(N, s.p[1])
parspaceof(::Val{N}, s::CoupledParSpace{1}) where N = CoupledParSpace(s.p, N)

parspaceof(n::Val{N}, x) where N = parspaceof(n, ntuple(i -> x, N))
parspaceof(::Val{N}, x::NTuple{N}) where N = SingletonParSpace(x...)
parspaceof(::Val{N}, x::NTuple{N, AbstractVector}) where N = ParSpace(x...)

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
    LoggingLayerSpace(s::AbstractLayerSpace; level=Logging.Debug, nextlogfun=() -> PrefixLogger("   "))
    LoggingLayerSpace(msgfun, s::AbstractLayerSpace; level = Logging.Debug, nextlogfun = () -> PrefixLogger("   "))
    LoggingLayerSpace(level::LogLevel, msgfun, nextlogfun, s::AbstractLayerSpace)

Logs `msgfun(layer)` at loglevel `level` after creating a `layer` from `s`.

Calling `nextlogfun()` produces an `AbstractLogger` which will be used when creating `layer` from `s`.

By default, this is used to add a level of indentation to subsequent logging calls which makes logs of hierarchical archspaces easier to read. Set `nextlogfun = () -> current_logger()` to remove this behaviour.
"""
struct LoggingLayerSpace{F,L<:LogLevel,LF,T <: AbstractLayerSpace}  <: AbstractLayerSpace
    msgfun::F
    level::L
    nextlogfun::LF
    s::T
end
LoggingLayerSpace(s::AbstractLayerSpace; level=Logging.Debug, nextlogfun=() -> PrefixLogger("   ")) = LoggingLayerSpace(l -> "Create $l from $(name(s))", s, level=level, nextlogfun=nextlogfun)
LoggingLayerSpace(msgfun, s::AbstractLayerSpace; level = Logging.Debug, nextlogfun = () -> PrefixLogger("   ")) = LoggingLayerSpace(msgfun, level, nextlogfun, s)
NaiveNASlib.name(s::LoggingLayerSpace) = name(s.s)
function (s::LoggingLayerSpace)(in::Integer,rng=rng_default; outsize=missing, wi=DefaultWeightInit())
    layer = with_logger(s.nextlogfun()) do
        ismissing(outsize) ? s.s(in, rng) : s.s(in, rng, outsize=outsize, wi=wi)
    end
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
denseinitW(::IdentityWeightInit) = (init = idmapping,)
denseinitW(wi::PartialIdentityWeightInit) = (init = (args...) -> circshift(idmapping_nowarn(args...),(wi.outoffset, wi.inoffset)),)
denseinitW(::ZeroWeightInit) = (init = zeros,)

"""
    ConvSpace{N} <:AbstractLayerSpace

    ConvSpace{N}(;outsizes, kernelsizes, activations=identity, strides=1, dilations=1, paddings=SamePad(), convfuns=Conv)
    ConvSpace(convfun::AbstractParSpace, base::BaseLayerSpace, ks::AbstractParSpace, stride::AbstractParSpace, dilation::AbstractParSpace, pad)



Search space of `N`D convolutional layers.

Constructor with keyword arguments takes scalars, vectors or `AbstractParSpace`s as inputs.
"""
struct ConvSpace{N} <:AbstractLayerSpace
    convfun::AbstractParSpace
    base::BaseLayerSpace
    kernelsize::AbstractParSpace{N, <:Integer}
    stride::AbstractParSpace
    dilation::AbstractParSpace
    pad::AbstractParSpace
end

ConvSpace{N}(;outsizes, kernelsizes, activations=identity, strides=1, dilations=1, paddings=SamePad(), convfuns=Conv) where N =
ConvSpace(
    parspaceof(convfuns),
    BaseLayerSpace(parspaceof(outsizes), parspaceof(activations)),
    parspaceof(N, kernelsizes),
    parspaceof(N, strides),
    parspaceof(N, dilations),
    parspaceof(paddings))

function (s::ConvSpace)(insize::Integer, rng=rng_default; outsize = outsize(s.base, rng), wi=DefaultWeightInit())
    convfun = s.convfun(rng)
    ks = Tuple(s.kernelsize(rng))
    act = activation(s.base, rng)
    pad = s.pad(rng)
    stride = s.stride(rng)
    dilation = s.dilation(rng)
    return convfun(ks, insize=>outsize, act; pad=pad, stride=stride, dilation=dilation, convinitW(wi)...)
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
    PoolSpace{N}(;windowsizes, strides=1, paddings=SamePad(), poolfun=[MaxPool, MeanPool])

Search space of `N`D pooling layers.

Constructor with keyword arguments takes scalars/tuples, vectors or `AbstractParSpace`s as inputs.
"""
struct PoolSpace{N} <:AbstractLayerSpace
    poolfun::AbstractParSpace
    ws::AbstractParSpace{N, <:Integer}
    stride::AbstractParSpace
    pad::AbstractParSpace
end
PoolSpace{N}(;windowsizes, strides=1, paddings=0, poolfuns=[MaxPool, MeanPool]) where N = PoolSpace{N}(
    parspaceof(poolfuns),
    parspaceof(N, windowsizes),
    parspaceof(N, strides),
    parspaceof(paddings)
)

function (s::PoolSpace)(in::Integer, rng=rng_default;outsize=nothing, wi=nothing)
    poolfun = s.poolfun(rng)
    ws = Tuple(s.ws(rng))
    stride = s.stride(rng)
    pad = s.pad(rng)
    poolfun(ws, stride=stride, pad=pad)
end

default_logging() = logged(level=Logging.Debug, info=NameAndIOInfoStr())
"""
    LayerVertexConf

Generic configuration template for computation graph vertices.

Intention is to make it easy to add logging, validation and pruning metrics in an uniform way.
"""
struct LayerVertexConf{F, T}
    layerfun::F
    traitfun::T
end
LayerVertexConf() = LayerVertexConf(ActivationContribution ∘ LazyMutable, validated() ∘ default_logging())

"""
    Shielded(base=LayerVertexConf(); allowed = tuple())

Create a [`LayerVertexConf`](@ref) which is shielded from mutation.

Keyword `allowed` can be used to supply a tuple (or array) of `AbstractMutation` types to allow.
"""
Shielded(base=LayerVertexConf(); allowed = tuple()) = let Shield(t) = MutationShield(t, allowed...)
    LayerVertexConf(base.layerfun, Shield ∘ base.traitfun)
end


(c::LayerVertexConf)(in::AbstractVertex, l) = mutable(l,in,layerfun=c.layerfun, traitfun=c.traitfun)
(c::LayerVertexConf)(name::String, in::AbstractVertex, l) = mutable(name, l,in,layerfun=c.layerfun, traitfun=c.traitfun)

Base.Broadcast.broadcastable(c::LayerVertexConf) = Ref(c)

"""
    ConcConf

Generic configuration template for concatenation of vertex outputs.
"""
struct ConcConf{F, T}
    layerfun::F
    traitfun::T
end
ConcConf() = ConcConf(ActivationContribution, validated() ∘ default_logging())

(c::ConcConf)(in::AbstractVector{<:AbstractVertex}) = c(in...)
(c::ConcConf)(in::AbstractVertex) = in
(c::ConcConf)(ins::AbstractVertex...) = concat(ins..., traitfun = c.traitfun, layerfun=c.layerfun)
(c::ConcConf)(name::String, in::AbstractVector{<:AbstractVertex}) = c(name, in...)
(c::ConcConf)(name::String, in::AbstractVertex) = in
(c::ConcConf)(name::String, ins::AbstractVertex...) = concat(ins..., traitfun = c.traitfun ∘ named(name), layerfun=c.layerfun)


"""
    LoggingArchSpace <: AbstractArchSpace
    LoggingArchSpace(s::AbstractArchSpace; level=Logging.Debug, nextlogfun=in -> PrefixLogger("   "))
    LoggingArchSpace(msgfun, s::AbstractArchSpace; level=Logging.Debug, nextlogfun=in -> PrefixLogger("   "))
    LoggingArchSpace(msgfun::Function, level::LogLevel, nextlogfun, s::AbstractArchSpace)

Logs `msgfun(vertex)` at loglevel `level` after creating a `vertex` from `s`.

Calling `nextlogfun(in)` where `in` is the input vertex produces an `AbstractLogger` which will be used when creating `vertex` from `s`.

By default, this is used to add a level of indentation to subsequent logging calls which makes logs of hierarchical archspaces easier to read. Set `nextlogfun = e -> current_logger()` to remove this behaviour.
"""
struct LoggingArchSpace{F,L<:LogLevel,LF,T <: AbstractArchSpace} <: AbstractArchSpace
    msgfun::F
    level::L
    nextlogfun::LF
    s::T
end
LoggingArchSpace(s::AbstractArchSpace; level=Logging.Debug, nextlogfun=in -> PrefixLogger("   ")) = LoggingArchSpace(v -> "Created $(name(v))", level, nextlogfun, s)
LoggingArchSpace(msgfun, s::AbstractArchSpace; level=Logging.Debug, nextlogfun=in -> PrefixLogger("   ")) = LoggingArchSpace(msgfun, level, nextlogfun, s)
function (s::LoggingArchSpace)(in::AbstractVertex,rng=rng_default; outsize=missing, wi=DefaultWeightInit())
    layer = with_logger(s.nextlogfun(in)) do
        ismissing(outsize) ? s.s(in, rng) : s.s(in, rng, outsize=outsize, wi=wi)
    end
    @logmsg s.level s.msgfun(layer)
    return layer
end
function (s::LoggingArchSpace)(namestr::String, in::AbstractVertex,rng=rng_default; outsize=missing, wi=DefaultWeightInit())
    layer = with_logger(s.nextlogfun(in)) do
        ismissing(outsize) ? s.s(namestr, in, rng) : s.s(namestr, in, rng, outsize=outsize, wi=wi)
    end
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

Search space of `AbstractArchSpace`s.
"""
struct ArchSpace <:AbstractArchSpace
    s::AbstractParSpace{1, <:AbstractArchSpace}
end
ArchSpace(l::AbstractLayerSpace; conf=LayerVertexConf()) = ArchSpace(VertexSpace(conf,l))
ArchSpace(l::AbstractLayerSpace, ls::AbstractLayerSpace...; conf=LayerVertexConf()) = ArchSpace(VertexSpace.(conf,[l,ls...])...)
ArchSpace(s::AbstractArchSpace) = ArchSpace(SingletonParSpace(s))
ArchSpace(s::AbstractArchSpace, ss::AbstractArchSpace...) = ArchSpace(ParSpace([s, ss...]))

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
    ArchSpaceChain <:AbstractArchSpace

Chains multiple `AbstractArchSpace`s after each other.

Input vertex will be used to generate an output vertex from the first `AbstractArchSpace` in the chain which is then used to generate a next output vertex from the next `AbstractArchSpace` in the chain and so on. The output from the last `AbstractArchSpace` is returned.
"""
struct ArchSpaceChain{S<:AbstractVector{<:AbstractArchSpace}} <:AbstractArchSpace
    s::S
end
ArchSpaceChain(s::AbstractArchSpace...) = ArchSpaceChain(collect(s))
(s::ArchSpaceChain)(in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit()) = foldl((next, ss) -> ss(next, rng, outsize=outsize, wi=repeatinitW(wi, next, outsize)), s.s, init=in)
(s::ArchSpaceChain)(name::String, in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit()) = foldl((next, i) -> s.s[i](join([name,".", i]), next, rng, outsize=outsize, wi=repeatinitW(wi, next, outsize)), eachindex(s.s), init=in)


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

function (s::ResidualArchSpace)(in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit())
    add = s.conf >> in + s.s(in, rng,outsize=nout(in), wi=resinitW(wi))
    return resscale(wi, s.conf, add)
end
function (s::ResidualArchSpace)(name::String, in::AbstractVertex, rng=rng_default; outsize=missing, wi=DefaultWeightInit())
    conf = s.conf
    nconf = @set conf.traitdecoration = conf.traitdecoration ∘ named(name * ".add")
    add = nconf >> in + s.s(join([name, ".res"]), in, rng,outsize=nout(in), wi=resinitW(wi))

    sconf = @set conf.traitdecoration = conf.traitdecoration ∘ named(name * ".add.scale")
    return resscale(wi, sconf, add)
 end

resinitW(wi::AbstractWeightInit) = wi

resscale(wi, conf, v) = v
resscale(::Union{IdentityWeightInit, PartialIdentityWeightInit}, conf, v) = invariantvertex(conf.outwrap(x -> convert(eltype(x), 0.5) .* x), v; traitdecoration= conf.traitdecoration)

"""
    FunctionSpace <: AbstractArchSpace
    FunctionSpace(funs...; namesuff::String, conf=LayerVertexConf(ActivationContribution, validated() ∘ default_logging()))

Return a `SizeInvariant` vertex representing `fun(x)` when invoked with `in` as input vertex where `x` is output of `in` where `fun` is uniformly selected from `funs`.
"""
struct FunctionSpace{C<:LayerVertexConf, P<:AbstractParSpace} <: AbstractArchSpace
    conf::C
    funspace::P
    namesuff::String
end
funspace_default_conf() = LayerVertexConf(ActivationContribution, validated() ∘ default_logging())

FunctionSpace(funs...; namesuff::String, conf=funspace_default_conf()) = FunctionSpace(conf, ParSpace1D(funs...), namesuff)

(s::FunctionSpace)(in::AbstractVertex, rng=rng_default; outsize=nothing, wi=nothing) = funvertex(s, in, rng)
(s::FunctionSpace)(name::String, in::AbstractVertex, rng=rng_default; outsize=nothing, wi=nothing) = funvertex(join([name,s.namesuff]), s, in, rng)

funvertex(s::FunctionSpace, in::AbstractVertex, rng) = invariantvertex(s.conf.layerfun(s.funspace(rng)), in, traitdecoration = s.conf.traitfun)

funvertex(name::String, s::FunctionSpace, in::AbstractVertex, rng) =
invariantvertex(s.conf.layerfun(s.funspace(rng)), in, traitdecoration = s.conf.traitfun ∘ named(name))

"""
    GlobalPoolSpace(Ts...)
    GlobalPoolSpace(conf::LayerVertexConf, Ts...)

Short for `FunctionSpace` with global average or global max pooling.

Also adds a `MutationShield` to prevent the vertex from being removed by default.
"""
GlobalPoolSpace(Ts...) = GlobalPoolSpace(LayerVertexConf(ActivationContribution, MutationShield ∘ validated() ∘ default_logging()), Ts...)
GlobalPoolSpace(conf::LayerVertexConf, Ts...=(MaxPool, MeanPool)...) = FunctionSpace(GlobalPool.(Ts)..., namesuff = ".globpool", conf=conf)
