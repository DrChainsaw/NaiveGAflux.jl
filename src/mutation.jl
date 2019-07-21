"""
    AbstractMutation

Generic mutation type.
"""
abstract type AbstractMutation{T} end

"""
    mutate(m::M, t::T) where {M<:AbstractMutation, T}

Mutate `t` using operation `m`.
"""
mutate(::M, ::T) where {M<:AbstractMutation, T} = throw(ArgumentError("$M of $T not implemented!"))

"""
    MutationProbability

Applies a wrapped `AbstractMutation` with a configured probability
"""
struct MutationProbability{T} <:AbstractMutation{T}
    m::AbstractMutation{T}
    p::Probability
end

function mutate(m::MutationProbability{T}, e::T) where T
    apply(m.p) do
        mutate(m.m, e)
    end
end

"""
    VertexMutation

Applies a wrapped `AbstractMutation` for each vertex in a `CompGraph`.
"""
struct VertexMutation <:AbstractMutation{CompGraph}
    m::AbstractMutation{AbstractVertex}
    p::Probability
end
function mutate(m::VertexMutation, g::CompGraph)
    for v in vertices(g)
        apply(m.p) do
            mutate(m.m, v)
        end
    end
end
