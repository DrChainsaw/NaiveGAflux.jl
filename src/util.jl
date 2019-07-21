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
Probability(p::Real) = Probability(p, Random.GLOBAL_RNG)
Probability(p::Integer) = Probability(p, Random.GLOBAL_RNG)
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

# TODO: Fix NaiveNaslib issue #18 
import NaiveNASlib: DecoratingTrait
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
