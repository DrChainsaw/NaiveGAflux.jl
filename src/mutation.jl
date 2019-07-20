"""
    AbstractMutation

Generic mutation type.
"""
abstract type AbstractMutation end

"""
    mutate(m::M, t::T) where {M<:AbstractMutation, T}

Mutate `t` using operation `m`.
"""
mutate(::M, ::T) where {M<:AbstractMutation, T} = throw(ArgumentError("Mutation $M of $T not implemented!"))
