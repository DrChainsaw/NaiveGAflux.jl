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
    VertexMutation

Applies
"""
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
