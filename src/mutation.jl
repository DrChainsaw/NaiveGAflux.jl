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
    VertexMutation

Applies a wrapped `AbstractMutation` for each vertex in a `CompGraph` with a configured probability.
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
