"""
    AbstractMutation{T}

Abstract type defining a mutation operation on entities of type `T`.

Implementations are expected to be callable using an entity of type `T` as only input.
"""
abstract type AbstractMutation{T} end

"""
    MutationProbability{T} <:AbstractMutation{T}

Applies a wrapped `AbstractMutation` with a configured probability
"""
struct MutationProbability{T} <:AbstractMutation{T}
    m::AbstractMutation{T}
    p::Probability
end

function (m::MutationProbability{T})(e::T) where T
    apply(m.p) do
        m.m(e)
    end
end

"""
    MutationList{T} <: AbstractMutation{T}

Applies all wrapped `AbstractMutation{T}`s to each entity of type `T`.
"""
struct MutationList{T} <: AbstractMutation{T}
    m::AbstractVector{<:AbstractMutation{T}}
end
MutationList(m::AbstractMutation{T}...) where T = MutationList(collect(m))
(m::MutationList)(e::T) where T = foreach(mm -> mm(e), m.m)

"""
    RecordMutation{T} <:AbstractMutation{T}

Records all mutated entities.

Intended use case is to be able to do parameter selection on mutated vertices.
"""
struct RecordMutation{T} <:AbstractMutation{T}
    m::AbstractMutation{T}
    mutated::AbstractVector{T}
end
RecordMutation(m::AbstractMutation{T}) where T = RecordMutation(m, T[])
function (m::RecordMutation{T})(e::T) where T
    push!(m.mutated, e)
    m.m(e)
end

"""
    VertexMutation <:AbstractMutation{CompGraph}

Applies a wrapped `AbstractMutation{AbstractVertex}` for each selected vertex in a `CompGraph`.

Vertices to select is determined by the configured `AbstractVertexSelection`.
"""
struct VertexMutation <:AbstractMutation{CompGraph}
    m::AbstractMutation{AbstractVertex}
    s::AbstractVertexSelection
end
VertexMutation(m::AbstractMutation{AbstractVertex}) = VertexMutation(m, FilterMutationAllowed())
function (m::VertexMutation)(g::CompGraph)
    for v in select(m.s, g)
        m.m(v)
    end
end

"""
    NoutMutation <:AbstractMutation{AbstractVertex}

Mutate the out size of a vertex.

Size is changed by `x * nout(v)` quantized to closest non-zero integer of `minΔnoutfactor(v)` where `x` is drawn from `U(minrel, maxrel)`
"""
struct NoutMutation <:AbstractMutation{AbstractVertex}
    minrel::Real
    maxrel::Real
    rng::AbstractRNG
    NoutMutation(l1,l2, rng) = l1 < l2 ? new(l1, l2, rng) : new(l2,l1, rng)
end
NoutMutation(limit, rng::AbstractRNG=rng_default) = NoutMutation(0, limit, rng)
NoutMutation(l1,l2) = NoutMutation(l1,l2, rng_default)
function (m::NoutMutation)(v::AbstractVertex)
    Δfactor = minΔnoutfactor(v)
    # Missing Δfactor means vertex can't be mutated, for example if it touches an immutable vertex such as an input vertex
    ismissing(Δfactor) && return

    shift = m.minrel
    scale = m.maxrel - m.minrel

    x = rand(m.rng) * scale + shift
    xq = (nout(v) * x) ÷ Δfactor
    Δ = Int(sign(x) * max(Δfactor, abs(xq) * Δfactor))

    Δnout(v, Δ)
end
