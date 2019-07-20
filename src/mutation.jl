"""
    AbstractMutation

Generic mutation type.
"""
abstract type AbstractMutation end

"""
    mutate(m::M, t::T) where {M<:AbstractMutation, T}

Mutate `t` using operation `m`.
"""
mutate(::M, ::T) where {M<:AbstractMutation, T} = throw(ArgumentError("$M of $T not implemented!"))

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
apply(p::Probability) = rand(p.rng) > p.p

"""
    apply(f, p::Probability)

Call `f` with probability `p.p` (subject to `p.rng` behaviour).
"""
apply(f::Function, p::Probability) =  apply(p) && f()

struct VertexMutation <:AbstractMutation
    m::AbstractMutation
    p::Probability
end
function mutate(m::VertexMutation, g::CompGraph)
    for v in vertices(g)
        apply(m.p) do
            mutate(m.m, v)
        end
    end
end
