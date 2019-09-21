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

Return true with a probability of ´p.p´ (subject to `p.rng` behaviour).
"""
apply(p::Probability) = rand(p.rng) < p.p

"""
    apply(f, p::Probability)

Call `f` with probability `p.p` (subject to `p.rng` behaviour).
"""
apply(f::Function, p::Probability) =  apply(p) && f()

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


"""
    RepeatPartitionIterator
    RepeatPartitionIterator(base, nrep)

Stateful iterator which repeats a partition of size `nrep` elements in `base` each time iterated over until [`advance!`](@ref) is called.

Useful for training all models in a population with the same data in each evolution epoch.

Calling `advance!(itr)` will advance the state so that the next time `itr` is iterated over it will start from the element after `e` in base where `e` is the last element from previous iteration.
"""
mutable struct RepeatPartitionIterator{T,VS}
    base::T
    curr::VS
    RepeatPartitionIterator(base::Stateful{B, VS}, nrep) where {B,VS} = new{Take{Stateful{B,VS}}, VS}(Take(base, nrep), base.nextvalstate)
end
RepeatPartitionIterator(base::B, nrep) where B = RepeatPartitionIterator(Stateful(base), nrep)

function Base.iterate(itr::RepeatPartitionIterator)
    itr.base.xs.nextvalstate = itr.curr
    return iterate(itr.base)
end
Base.iterate(itr::RepeatPartitionIterator, state) = iterate(itr.base, state)

Base.length(itr::RepeatPartitionIterator) = length(itr.base)
Base.eltype(itr::RepeatPartitionIterator) = eltype(itr.base)

# Workaround for https://github.com/JuliaLang/julia/issues/33349
repeatiter(itr, nreps) = Iterators.take(Iterators.cycle(itr), nreps * length(itr))

"""
    advance!(itr::RepeatPartitionIterator)

Advances itr to the next partition of the wrapped iterator.
"""
advance!(itr::RepeatPartitionIterator) = itr.curr = itr.base.xs.nextvalstate
