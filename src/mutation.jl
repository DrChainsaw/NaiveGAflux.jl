"""
    AbstractMutation{T}

Abstract type defining a mutation operation on entities of type `T`.

Implementations are expected to be callable using an entity of type `T` as only input.
"""
abstract type AbstractMutation{T} end

"""
    MutationProbability{T} <:AbstractMutation{T}
    MutationProbability(m::AbstractMutation{T}, p::Probability)

Applies a wrapped `AbstractMutation` with a configured `Probability`
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
    MutationList(m::AbstractMutation{T}...)

Applies all wrapped `AbstractMutation{T}`s to each entity of type `T`.
"""
struct MutationList{T} <: AbstractMutation{T}
    m::AbstractVector{<:AbstractMutation{T}}
end
MutationList(m::AbstractMutation{T}...) where T = MutationList(collect(m))
(m::MutationList)(e::T) where T = foreach(mm -> mm(e), m.m)

"""
    RecordMutation{T} <:AbstractMutation{T}
    RecordMutation(m::AbstractMutation{T})

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
    VertexMutation(m::AbstractMutation{AbstractVertex}, s::AbstractVertexSelection)
    VertexMutation(m::AbstractMutation{AbstractVertex})

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
    NoutMutation(l1::Real,l2::Real, rng::AbstractRNG)
    NoutMutation(limit, rng::AbstractRNG=rng_default)
    NoutMutation(l1,l2)

Mutate the out size of a vertex.

Size is changed by `x * nout(v)` quantized to closest non-zero integer of `minΔnoutfactor(v)` where `x` is drawn from `U(minrel, maxrel)` where `minrel` and `maxrel` are `l1` and `l2` if `l1 < l2` and `l2` and `l1` otherwise.
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

"""
    AddVertexMutation <:AbstractMutation{AbstractVertex}
    AddVertexMutation(s::AbstractArchSpace, outselect::Function, rng::AbstractRNG)
    AddVertexMutation(s, outselect::Function=identity)
    AddVertexMutation(s, rng::AbstractRNG)

Add insert a vertex from the wrapped `AbstractArchSpace` `s` after a given vertex `v`.

The function `outselect` takes an `AbstractVector{AbstractVertex}` representing the output of `v` and returns an `AbstractVector{AbstractVertex}` which shall be reconnected to the vertex `v'` returned by `s`. Defaults to `identity` meaning all outputs of `v` are reconnected to `v'`.
"""
struct AddVertexMutation <:AbstractMutation{AbstractVertex}
    s::AbstractArchSpace
    outselect::Function
    rng::AbstractRNG
end
AddVertexMutation(s, outselect::Function=identity) = AddVertexMutation(s, outselect, rng_default)
AddVertexMutation(s, rng::AbstractRNG) = AddVertexMutation(s, identity, rng)

(m::AddVertexMutation)(v::AbstractVertex) = insert!(v, vi -> m.s(vi, outsize=nout(vi)), m.outselect)
