"""
    Probability
Represents a probability that something (typically mutation) will happen.

Possible to specify RNG implementation. If not specified, `rng_default` will be used.
"""
struct Probability{P<:Real, R<:AbstractRNG}
    p::P
    rng::R
    function Probability(p::P, rng::R) where {P, R}
         @assert 0 <= p <= 1
         return new{P, R}(p, rng)
    end
end
Probability(p::Real) = Probability(p, rng_default)
Probability(p::Integer) = Probability(p, rng_default)
Probability(p::Integer, rng) = p == 1 ? Probability(1.0, rng) : Probability(p / 100.0, rng)

"""
    apply(p::Probability)
    apply(p::Number)

Return true with a probability of ´p.p´ (subject to `p.rng` behaviour).
"""
apply(p::Probability) = rand(p.rng) < p.p
apply(p::Real) = apply(Probability(p))

"""
    apply(f, p::Probability, or = () -> nothing)
    apply(f, p::Real, or = () -> nothing)

Call `f` with probability `p.p` (subject to `p.rng` behaviour). If `f` is not called then call `or`.
"""
apply(f, p::Probability, or = () -> nothing) =  apply(p) ? f() : or()
apply(f, p::Real, or = () -> nothing) = apply(p) ? f() : or()

"""
    MutationShield <: DecoratingTrait
    MutationShield(t, allowed...)

Shields its associated vertex from being selected for mutation.

Any types in `allowed` will be allowed to mutate the vertex if supplied when calling `allow_mutation`.

Note that vertex might still be modified if an adjacent vertex is mutated in a way which propagates to a shielded vertex.
"""
struct MutationShield{T<:MutationTrait, S} <:DecoratingTrait
    t::T
    allowed::S
    MutationShield(t::T, allowed::Tuple) where T = new{T, typeof(allowed)}(t, allowed)
end
MutationShield(t, allowed...) = MutationShield(t, allowed)

NaiveNASlib.base(t::MutationShield) = t.t
allow_mutation(v::AbstractVertex, ms...) = allow_mutation(trait(v), ms...)
allow_mutation(t::DecoratingTrait, ms...) = allow_mutation(base(t), ms...)
allow_mutation(::MutationTrait, ms...) = true
allow_mutation(::Immutable, ms...) = false
allow_mutation(t::MutationShield, ms...) = !isempty(ms) && all(mt -> any(amt -> mt <: amt, t.allowed), typeof.(mutationleaves(ms)))

@functor MutationShield

"""
    AbstractVertexSelection

Abstract type for determining how to select vertices from a `CompGraph` or an array of vertices.
"""
abstract type AbstractVertexSelection end

"""
    AllVertices

Select all vertices in `g`.
"""
struct AllVertices <:AbstractVertexSelection end
select(::AllVertices, g::CompGraph, ms...) = vertices(g)
select(::AllVertices, vs::AbstractArray, ms...) = vs

"""
    FilterMutationAllowed

Filters out only the vertices for which mutation is allowed from another selection.
"""
struct FilterMutationAllowed{S} <:AbstractVertexSelection
    s::S
end
FilterMutationAllowed() = FilterMutationAllowed(AllVertices())
select(s::FilterMutationAllowed, x, ms...) = filter(v -> allow_mutation(v, ms...), select(s.s, x, ms...))

"""
    SelectWithMutation{S, M} <: AbstractVertexSelection
    SelectWithMutation(m::M)
    SelectWithMutation(s::S, m::M)

Adds `m` to the list of `ms` when selecting vertices using `s`.

Useful when calling `select` from inside a function without the context of an `AbstractMutation`.
"""
struct SelectWithMutation{S, M} <: AbstractVertexSelection
    s::S
    m::M
end
SelectWithMutation(m) = SelectWithMutation(FilterMutationAllowed(), m)
select(s::SelectWithMutation, x, ms...) = select(s.s, x, (s.m, ms...))

"""
    ApplyIf <: DecoratingTrait
    ApplyIf(predicate::Function, apply::Function, base::MutationTrait)

Enables calling `apply(v)` for an `AbstractVertex v` which has this trait if 'predicate(v) == true'.

Motivating use case is to have a way to remove vertices which have ended up as noops, e.g. element wise and concatenation vertices with a single input or identity activation functions.
"""
struct ApplyIf{P, A, M<:MutationTrait} <: DecoratingTrait
    predicate::P
    apply::A
    base::M
end
RemoveIfSingleInput(t) = ApplyIf(v -> length(inputs(v)) == 1, remove!, t)
NaiveNASlib.base(t::ApplyIf) = t.base

check_apply(g::CompGraph) = foreach(check_apply, vertices(g))
check_apply(v::AbstractVertex) = check_apply(trait(v), v)
check_apply(t::DecoratingTrait, v) = check_apply(base(t), v)
check_apply(t::ApplyIf, v) = t.predicate(v) && t.apply(v)
function check_apply(t, v) end

@functor ApplyIf

"""
    PersistentArray{T, N} <: AbstractArray{T, N}
    PersistentArray(savedir::String, nr::Integer, generator;suffix=".jls")
    PersistentArray(savedir::String, suffix::String, data::Array)

Simple persistent array. Can be created from serialized data and can be asked to persist its elements using [`persist`](@ref).

Note that once initialized, the array is not backed by the serialized data. Adding/deleting files is not reflected in data and vice versa.
"""
struct PersistentArray{T, N} <: AbstractArray{T, N}
    savedir::String
    suffix::String
    data::Array{T,N}
end
function PersistentArray(savedir::String, nr::Integer, generator;suffix=".jls")
    data = map(1:nr) do i
        filename = joinpath(savedir, "$i$suffix")
        isfile(filename) && return deserialize(filename)
        return generator(i)
    end
    return PersistentArray(savedir, suffix, data)
end
"""
    persist(a::PersistentArray) 

Serializes the elements of `a`, one file per element.
"""
function persist(a::PersistentArray)
    mkpath(a.savedir)
    for (i, v) in enumerate(a)
        serialize(filename(a, i), v)
    end
end
filename(a::PersistentArray, i::Int) = joinpath(a.savedir, "$i$(a.suffix)")
function Base.rm(a::PersistentArray; force=true, recursive=true)
    foreach(i -> rm(a, i; force=force, recursive=recursive), 1:length(a))
    if readdir(a.savedir) |> isempty
        rm(a.savedir; force=force, recursive=recursive)
    end
end
Base.rm(a::PersistentArray, i::Int; force=false, recursive=true) = rm(filename(a,i), force=force, recursive=recursive)

Base.size(a::PersistentArray) = size(a.data)
Base.getindex(a::PersistentArray, i::Int) = getindex(a.data, i)
Base.getindex(a::PersistentArray, I::Vararg{Int, N}) where N = getindex(a.data, I...)
Base.setindex!(a::PersistentArray, v, i::Int) = setindex!(a.data, v, i)
Base.setindex!(a::PersistentArray, v, I::Vararg{Int, N}) where N = setindex!(a.data, v, I...)
Base.similar(a::PersistentArray, t::Type{S}, dims::Dims) where S = PersistentArray(a.savedir, a.suffix, similar(a.data,t, dims))


"""
    BoundedRandomWalk{T <: Real, R <: Function}
    BoundedRandomWalk(lb, ub, rfun =  (x...) -> 0.2randn(rng_default))

Generates steps for a random walk with bounds `[lb, ub]`.

Main use case is with learning rate mutation to prevent it from accidentally drifting to some catastropically high value.

# Examples
```julia-repl
julia> import NaiveGAflux: BoundedRandomWalk

julia> using Random

julia> rng = MersenneTwister(0);

julia> brw = BoundedRandomWalk(-2.34, 3.45, () -> 10randn(rng));

julia> extrema(cumsum([brw() for i in 1:10000]))
(-2.3400000000000007, 3.4500000000000153)
```
"""
struct BoundedRandomWalk{T <: Real, R <: Function}
    lb::T
    ub::T
    state::Base.RefValue{T}
    rfun::R
end
BoundedRandomWalk(lb::T,ub::T, rfun = (x...) -> 0.2randn(rng_default)) where T = BoundedRandomWalk(lb,ub, Ref(zero(ub)), rfun)

function(r::BoundedRandomWalk)(x...)
    y =  ((r.ub - r.lb) * clamp(r.rfun(x...), -1, 1) + (r.ub + r.lb)) / 2 - r.state[]
    r.state[] += y
    return y
 end


"""
    ShieldedOpt{R} <: Flux.Optimise.AbstractOptimiser 
    ShieldedOpt(rule)

Shields `rule` from mutation by `OptimiserMutation`.
"""
struct ShieldedOpt{R<:Optimisers.AbstractRule} <: Optimisers.AbstractRule
    rule::R
end
Optimisers.apply!(r::ShieldedOpt, args...) = Optimisers.apply!(r.rule, args...)
Optimisers.init(r::ShieldedOpt, args...) = Optimisers.init(r.rule, args...)

"""
    ImplicitOpt{R}
    ImplicitOpt(rule)

Sentinel value to be used e.g. for a [`CandidateOptModel`](@ref) or a [`TrainThenFitness`](@ref)
to mark that parameters are updated implicitly when taking gradients.

Main use case is when an `$AutoOptimiser` is used to update the model.
"""
struct ImplicitOpt{R} <: Optimisers.AbstractRule
    rule::R
end
Optimisers.apply!(o::ImplicitOpt, args...) = Optimisers.apply!(o.rule, args...)
Optimisers.init(o::ImplicitOpt, args...) = Optimisers.init(o.rule, args...)

"""
    mergeopts(t::Type{T}, os...) where T
    mergeopts(os::T...)

Merge all optimisers of type `T` in `os` into one optimiser of type `T`.
"""
function mergeopts(::Type{T}, os...) where T
    merged = mergeopts(filter(o -> isa(o, T), os))
    return (filter(o -> !isa(o, T), os)..., merged...)
end
mergeopts(os::Tuple) = tuple(mergeopts(os...))
mergeopts(os::Tuple{}) = os
mergeopts(::Type{T}, os::T...) where T = mergeopts(os...)
mergeopts(os::Optimisers.AbstractRule...) = first(@set os[1].eta = prod(learningrate, os))
mergeopts(os::ShieldedOpt{T}...) where T = ShieldedOpt(only(mergeopts(map(o -> o.rule, os))))
mergeopts(os::WeightDecay...) = WeightDecay(prod(o -> o.gamma, os))

"""
    optmap(fopt, x, felse=identity)
    optmap(fopt, felse=identity)

Return `fopt(x)` if `x` is an optimiser, else return `felse(x)`.

Call without x to return `x -> optmap(fopt, x, felse)`
"""
optmap(fopt, felse=identity) = x -> optmap(fopt, x, felse)
optmap(fopt, x, felse) = felse(x)
optmap(fopt, o::Optimisers.AbstractRule, felse) = fopt(o)

"""
    Singleton{T}
    Singleton(val::T)

Wrapper for `val` which prevents it from being copied if the wrapping singleton is copied using `copy` and `deepcopy`.

Also makes sure that only one unique copy of `val` is created when deserializing a serialized `Singleton`.
"""
struct Singleton{T}
    val::T
    function Singleton(val::T) where T
        s = new{T}(val)
        singletons[val] = s
        return s
    end
end
val(s::Singleton) = s.val

const singletons = WeakKeyDict()
function Serialization.deserialize(s::AbstractSerializer, ::Type{Singleton{T}}) where T
    val = deserialize(s)
    return get!(()-> Singleton(val), singletons, val)
end

Base.deepcopy_internal(s::Singleton, stackdict::IdDict) = s
Base.copy(s::Singleton) = s

Base.iterate(s::Singleton, state...) = iterate(val(s), state)

Base.length(s::Singleton) = length(val(s))
Base.size(s::Singleton) = size(val(s))

Base.IteratorEltype(s::Singleton) = Base.IteratorEltype(val(s))
Base.IteratorSize(s::Singleton) = Base.IteratorSize(val(s))


"""
     PrefixLogger{L<:AbstractLogger, S} <: AbstractLogger
     PrefixLogger(wrappedLogger::L, prefix::S)

Add `prefix` to messages and forwards to `wrappedLogger`.
"""
struct PrefixLogger{L<:AbstractLogger, S} <: AbstractLogger
      wrapped::L
      prefix::S
end
PrefixLogger(prefix::String) = PrefixLogger(current_logger(), prefix)

Logging.min_enabled_level(l::PrefixLogger) = Logging.min_enabled_level(l.wrapped)
Logging.shouldlog(l::PrefixLogger, args...) =  Logging.shouldlog(l.wrapped, args...)
Logging.handle_message(l::PrefixLogger, level, message, args...; kwargs...) = Logging.handle_message(l.wrapped, level, l.prefix * message, args...; kwargs...)

"""
    GlobalPool{PT}
    GlobalPool(PT)

Global pool combined with `Flux.flatten`. Pool type is determined by `PT`, typically `MaxPool` or `MeanPool`.

Useful because in the context of mutations, one most likely want to keep them as one single vertex to avoid mutations happening between them.
"""
struct GlobalPool{PT} end
GlobalPool(PT) = GlobalPool{PT}()

# About 50% faster on GPU to create a MeanPool and use it compared to dropdims(mean(x, dims=[1:2]), dims=(1,2)). CBA to figure out why...
(::GlobalPool{PT})(x::AbstractArray{<:Any, N}) where {N, PT} = Flux.flatten(PT(size(x)[1:N-2])(x))

NaiveNASflux.layertype(gp::GlobalPool) = gp
NaiveNASflux.layer(gp::GlobalPool) = gp

"""
    ninputs(model)
    
Return the number of model inputs.
"""
ninputs(cg::CompGraph) = length(inputs(cg))
# I guess this is not good practice, but I'll fix it the first time someone posts an issue about it :)
ninputs(m) = 1 


"""
    InconsistentAutoOptimiserException{M}
    InconsistentAutoOptimiserException(model)

Thrown when `model` has implicit optimiser state for some but not all trainable parameters.
"""
struct InconsistentAutoOptimiserException{M} <: Exception
    model::M
end

const failmodel = Ref{Any}(nothing)


function Base.showerror(io::IO, e::InconsistentAutoOptimiserException{<:CompGraph}) 
    println(io, "Model has a mix of implicit and explicit optimiser rules! This is currently not supported!")
    failmodel[] = e.model
    _print_implicit_opts(io, e.model)
end
function _print_implicit_opts(io::IO, g::CompGraph)
    for (i, v) in enumerate(vertices(g))
        if find_first_array(layer(v)) !== nothing
            println(io, "vertex ", i, " implicit: ", _is_implicit_opt(v), ", name: ", name(v))
        end
    end
end
_is_implicit_opt(v) = mutateoptimiser!(a -> a.optstate, v)

"""
    check_implicit_optimiser(m::CompGraph)
    check_implicit_optimiser(c::AbstractCandidate)

Returns true if input argument uses implicit optimiser state (e.g $AutoOptimiser).

Throws a $InconsistentAutoOptimiserException if some but not all parameters use implicit optimiser state.
"""
function check_implicit_optimiser(g::CompGraph)
    prev = nothing
    for v in vertices(g)
        if find_first_array(layer(v)) !== nothing
            isimplicit = _is_implicit_opt(v)
            if isnothing(prev)
                prev = isimplicit 
            else
                if prev !== isimplicit
                   throw(InconsistentAutoOptimiserException(g)) 
                end
            end
        end
    end
    isnothing(prev) ? false : prev
end

struct NaiveGAfluxCpuDevice end

function execution_device(model) 
     maybearray = find_first_array(model)
     maybearray === nothing ? NaiveGAfluxCpuDevice() : execution_device(maybearray)
end
execution_device(::AbstractArray) = NaiveGAfluxCpuDevice()

# Just to avoid infinite recursion due to inputs -> outputs -> inputs
find_first_array(g::CompGraph) = find_first_array(vertices(g))
find_first_array(v::AbstractVertex) = find_first_array(layer(v))

find_first_array(model) = find_first_array(Flux.trainable(model))
function find_first_array(x::Union{Tuple, NamedTuple, AbstractArray}) 
    for e in x
        res = find_first_array(e)
        res !== nothing && return res
    end
end
find_first_array(x::AbstractArray{<:Number}) = x


