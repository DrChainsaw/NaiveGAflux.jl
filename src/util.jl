"""
    Probability
Represents a probability that something (typically mutation) will happen.

Possible to specify RNG implementation. If not specified, `GLOBAL_RNG` will be used
"""
struct Probability
    p::Real
    rng::AbstractRNG
    function Probability(p::Real, rng)
         @assert 0 <= p <= 1
         return new(p, rng)
    end
end
Probability(p::Real) = Probability(p, rng_default)
Probability(p::Integer) = Probability(p, rng_default)
Probability(p::Integer, rng) = Probability(p / 100.0, rng)

"""
    apply(p::Probability)
    apply(p::Number)

Return true with a probability of ´p.p´ (subject to `p.rng` behaviour).
"""
apply(p::Probability) = rand(p.rng) < p.p
apply(p::Real) = apply(Probability(p))

"""
    apply(f, p::Probability)
    apply(f, p::Real)

Call `f` with probability `p.p` (subject to `p.rng` behaviour).
"""
apply(f, p::Probability) =  apply(p) && f()
apply(f, p::Real) = apply(p) && f()

"""
    MutationShield <: DecoratingTrait

Shields its associated vertex from being selected for mutation.

Note that vertex might still be modified if an adjacent vertex is mutated in a way which propagates to a shielded vertex.
"""
struct MutationShield <:DecoratingTrait
    t::MutationTrait
end
NaiveNASlib.base(t::MutationShield) = t.t
allow_mutation(v::AbstractVertex) = allow_mutation(trait(v))
allow_mutation(t::DecoratingTrait) = allow_mutation(base(t))
allow_mutation(::MutationTrait) = true
allow_mutation(::Immutable) = false
allow_mutation(::MutationShield) = false
NaiveNASlib.clone(t::MutationShield;cf=clone) = MutationShield(cf(base(t), cf=cf))

"""
    AbstractVertexSelection

Abstract type for determining how to select vertices from a `CompGraph`
"""
abstract type AbstractVertexSelection end

"""
    AllVertices

Select all vertices in `g`.
"""
struct AllVertices <:AbstractVertexSelection end
select(::AllVertices, g::CompGraph) = vertices(g)

"""
    FilterMutationAllowed

Filters out only the vertices for which mutation is allowed from another selection.
"""
struct FilterMutationAllowed <:AbstractVertexSelection
    s::AbstractVertexSelection
end
FilterMutationAllowed() = FilterMutationAllowed(AllVertices())
select(s::FilterMutationAllowed, g::CompGraph) = filter(allow_mutation, select(s.s, g))

"""
    ApplyIf <: DecoratingTrait
    ApplyIf(predicate::Function, apply::Function, base::MutationTrait)

Enables calling `apply(v)` for an `AbstractVertex v` which has this trait if 'predicate(v) == true'.

Motivating use case is to have a way to remove vertices which have ended up as noops, e.g. element wise and concatenation vertices with a single input or identity activation functions.
"""
struct ApplyIf <: DecoratingTrait
    predicate::Function
    apply::Function
    base::MutationTrait
end
RemoveIfSingleInput(t) = ApplyIf(v -> length(inputs(v)) == 1, remove!, t)
NaiveNASlib.base(t::ApplyIf) = t.base

check_apply(g::CompGraph) = foreach(check_apply, vertices(g))
check_apply(v::AbstractVertex) = check_apply(trait(v), v)
check_apply(t::DecoratingTrait, v) = check_apply(base(t), v)
check_apply(t::ApplyIf, v) = t.predicate(v) && t.apply(v)
function check_apply(t, v) end

NaiveNASlib.clone(t::ApplyIf;cf=clone) = ApplyIf(cf(t.predicate, cf=cf), cf(t.apply, cf=cf), cf(base(t), cf=cf))


"""
    PersistentArray{T, N} <: AbstractArray{T, N}
    PersistentArray(savedir::String, nr::Integer, generator;suffix=".jls")
    PersistentArray(savedir::String, suffix::String, data::Array)

Simple persistent array. Can be created from serialized data and can be asked to persist its elements.

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
function persist(a::PersistentArray)
    mkpath(a.savedir)
    for (i, v) in enumerate(a)
        serialize(filename(a, i), v)
    end
end
filename(a::PersistentArray, i::Int) = joinpath(a.savedir, "$i$(a.suffix)")
Base.rm(a::PersistentArray; force=true, recursive=true) = rm(a.savedir, force=force, recursive=recursive)
Base.rm(a::PersistentArray, i::Int, force=false, recursive=true) = rm(filename(a,i), force=force, recursive=recursive)

Base.size(a::PersistentArray) = size(a.data)
Base.getindex(a::PersistentArray, i::Int) = getindex(a.data, i)
Base.getindex(a::PersistentArray, I::Vararg{Int, N}) where N = getindex(a.data, I...)
Base.setindex!(a::PersistentArray, v, i::Int) = setindex!(a.data, v, i)
Base.setindex!(a::PersistentArray, v, I::Vararg{Int, N}) where N = setindex!(a.data, v, I...)
Base.similar(a::PersistentArray, t::Type{S}, dims::Dims) where S = PersistentArray(a.savedir, a.suffix, similar(a.data,t, dims))
