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
    LogMutation{T} <:AbstractMutation{T}
    LogMutation(strfun, m::AbstractMutation{T})
    LogMutation(strfun, level::LogLevel, m::AbstractMutation{T})

Logs all mutation operations.

Argument `strfun` maps the mutated entity to the logged string.
"""
struct LogMutation{T} <:AbstractMutation{T}
    strfun
    level::LogLevel
    m::AbstractMutation{T}
end
LogMutation(strfun, m::AbstractMutation{T}) where T = LogMutation(strfun, Logging.Info, m)
function (m::LogMutation{T})(e::T) where T
    @logmsg m.level m.strfun(e)
    m.m(e)
end

"""
    MutationFilter{T} <: AbstractMutation{T}
    MutationFilter(predicate, m)

Applies mutation `m` only for entities `e` for which `predicate(e)` returns true.
"""
struct MutationFilter{T} <: AbstractMutation{T}
    predicate
    m::AbstractMutation{T}
end
function (m::MutationFilter{T})(e::T) where T
    if m.predicate(e)
        m.m(e)
    end
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

    minsize = min(nout(v), minimum(nout.(findterminating(v, inputs))))
    minsize + Δ <= Δfactor && return

    fallback = ΔNout{Relaxed}(v, Δ, LogΔSizeExec(Logging.Warn, "Could not change nout of $v by $(Δ) after relaxation! Vertex not changed!", ΔSizeFailNoOp()))
    strategy = ΔNout{Exact}(v, Δ, LogΔSizeExec(Logging.Warn, "Could not change nout of $v by $(Δ)! Relaxing constraints...", fallback))

    Δsize(strategy , all_in_Δsize_graph(v, Output()))
end

"""
    AddVertexMutation <:AbstractMutation{AbstractVertex}
    AddVertexMutation(s::AbstractArchSpace, outselect::Function, rng::AbstractRNG)
    AddVertexMutation(s, outselect::Function=identity)
    AddVertexMutation(s, rng::AbstractRNG)

Insert a vertex from the wrapped `AbstractArchSpace` `s` after a given vertex `v`.

The function `outselect` takes an `AbstractVector{AbstractVertex}` representing the output of `v` and returns an `AbstractVector{AbstractVertex}` which shall be reconnected to the vertex `v'` returned by `s`. Defaults to `identity` meaning all outputs of `v` are reconnected to `v'`.
"""
struct AddVertexMutation <:AbstractMutation{AbstractVertex}
    s::AbstractArchSpace
    outselect::Function
    rng::AbstractRNG
end
AddVertexMutation(s, outselect::Function=identity) = AddVertexMutation(s, outselect, rng_default)
AddVertexMutation(s, rng::AbstractRNG) = AddVertexMutation(s, identity, rng)

(m::AddVertexMutation)(v::AbstractVertex) = insert!(v, vi -> m.s(name(vi), vi, m.rng, outsize=nout(vi)), m.outselect)

"""
    RemoveVertexMutation <:AbstractMutation{AbstractVertex}
    RemoveVertexMutation(s::RemoveStrategy)
    RemoveVertexMutation()

Remove the given vertex `v` using the configured `RemoveStrategy`.

Default size align strategy is `IncreaseSmaller -> DecreaseBigger -> AlignSizeBoth -> FailAlignSizeWarn -> FailAlignSizeRevert`.

Default reconnect strategy is `ConnectAll`.
"""
struct RemoveVertexMutation <:AbstractMutation{AbstractVertex}
    s::RemoveStrategy
end
RemoveVertexMutation() = RemoveVertexMutation(RemoveStrategy(CheckAligned(CheckNoSizeCycle(ApplyMutation(SelectOutputs(select = SelectDirection(OutSelect{NaiveNASlib.Exact}(LogSelectionFallback("Reverting...", NoutRevert()))),
 valuefun = default_neuronselect, align=IncreaseSmaller(DecreaseBigger(AlignSizeBoth(FailAlignSizeWarn()))))), FailAlignSizeWarn(msgfun = (vin,vout) -> "Can not remove vertex $(name(vin))! Size cycle detected!")))))

(m::RemoveVertexMutation)(v::AbstractVertex) = remove!(v, m.s)

"""
    NeuronSelectMutation{T} <: AbstractMutation{AbstractVertex}
    NeuronSelectMutation(m::AbstractMutation{AbstractVertex})
    NeuronSelectMutation(rankfun, m::AbstractMutation{AbstractVertex})
    NeuronSelectMutation(rankfun::Function, strategy, m::RecordMutation{AbstractVertex})

Selects neurons of vertices mutated by the wrapped `RecordMutation`.

Possible to select ranking method for neurons using `rankfun` which takes a mutated vertex as input and value/utility per neuron (higher is better).

How to select neurons depends a bit on what operation the wrapped `RecordMutation` performs. If not supplied explicitly an attempt to infer it will be made, resulting in an error if not possible.
"""
struct NeuronSelectMutation{T} <: AbstractMutation{AbstractVertex}
    rankfun::Function
    strategy::T
    m::RecordMutation{AbstractVertex}
end
NeuronSelectMutation(rankfun, m::AbstractMutation{AbstractVertex}) = NeuronSelectMutation(rankfun, neuron_select_strategy(m), m)
NeuronSelectMutation(rankfun, strategy, m::AbstractMutation{AbstractVertex}) = NeuronSelectMutation(rankfun, strategy, RecordMutation(m))
NeuronSelectMutation(m::AbstractMutation{AbstractVertex}) = NeuronSelectMutation(default_neuronselect, m)
NeuronSelectMutation(m::RecordMutation{AbstractVertex}) = NeuronSelectMutation(default_neuronselect, neuron_select_strategy(m.m), m)

(m::NeuronSelectMutation)(v::AbstractVertex) = m.m(v)

neuron_select_strategy(::T) where T <:AbstractMutation = error("Neuron select strategy not implemented for $T")
neuron_select_strategy(::RemoveVertexMutation) = Nout()
neuron_select_strategy(::NoutMutation) = Nout()

struct Nout
    s::AbstractSelectionStrategy
end
Nout() = Nout(ApplyAfter())

"""
    select(m::NeuronSelectMutation)

Select neurons for each `AbstractVertex` mutated by `m`.
"""
select(m::NeuronSelectMutation) = select_neurons(m.strategy, m.m.mutated, m.rankfun)

select_neurons(::T, vs, rankfun) where T = error("Neuron select not implemented for $T")
function select_neurons(strategy::Nout, vs::AbstractArray{AbstractVertex}, rankfun::Function)
    vchanged = filter(v -> nout(v) != nout_org(v), vs)
    isempty(vchanged) && return

    vall = unique(mapfoldl(v -> all_in_Δsize_graph(v, Output()), vcat, vchanged))
    Δoutputs(strategy.s, vall, rankfun)
end

default_neuronselect(v) = neuron_value(trait(v), v)

NaiveNASflux.neuron_value(t::DecoratingTrait, v) = neuron_value(base(t), v)
NaiveNASflux.neuron_value(::Immutable, v) = ones(nout(v))
NaiveNASflux.neuron_value(::MutationSizeTrait, v) = clean_values(cpu(neuron_value(v)),v)
clean_values(::Missing, v) = ones(nout_org(v))
# NaN should perhaps be < 0, but since SelectDirection is used, this might lead to inconsistent results as a subset of neurons for a vertex v whose output vertices are not part of the selection (typically because only v's inputs are touched) are selected. As the output vertices are not changed this will lead to a size inconsistency. Cleanest fix might be to separate "touch output" from "touch input" when formulating the output selection problem.
clean_values(a::AbstractArray, v) = replace(a, NaN => 0.01, 0.0 => 0.01, Inf => 0.01, -Inf => 0.01)


"""
    PostMutation{T} <: AbstractMutation{T}
    PostMutation(actions, m::AbstractMutation{T})
    PostMutation(m::AbstractMutation{T}, actions...)

Performs a set of actions after a wrapped `AbstractMutation` is applied.

Actions will be invoked with arguments (m::PostMutation{T}, e::T) where m is the enclosing `PostMutation` and `e` is the mutated entity of type `T`.
"""
struct PostMutation{T} <: AbstractMutation{T}
    actions
    m::AbstractMutation{T}
end
PostMutation(m::AbstractMutation{T}, actions...) where T = PostMutation(actions, m)
PostMutation(action::Function, m::AbstractMutation{T}) where T = PostMutation(m, action)
function (m::PostMutation{T})(e::T) where T
    m.m(e)
    foreach(a -> a(m, e), m.actions)
end


"""
    NeuronSelect()

Search a given `AbstractMutation` hieararchy for `NeuronSelectMutations` and invoke `select` on them
"""
struct NeuronSelect end

(s::NeuronSelect)(m, e) = s(m)
function (s::NeuronSelect)(m) end
(s::NeuronSelect)(m::T) where T <:AbstractMutation = foreach(fn -> s(getfield(m,fn)), fieldnames(T))
(s::NeuronSelect)(a::Union{AbstractVector, Tuple}) = foreach(ae -> s(ae), a)
(s::NeuronSelect)(m::NeuronSelectMutation) = select(m)

"""
    RemoveZeroNout()
    RemoveZeroNout(fallback)

Search for vertices with zero output size and remove them and all of their input vertices if possible to do so witout removing an input or output vertex.

Removal is only possible if a vertex is inside a parallel path which will later be concatenated.
"""
struct RemoveZeroNout
    fallback
end
RemoveZeroNout() = RemoveZeroNout(IncreaseZeroNout())
struct IncreaseZeroNout end

(r::RemoveZeroNout)(m, e) = r(e)
function (r::RemoveZeroNout)(g::CompGraph)
    topoligical_order = vertices(g)
    for v in topoligical_order
        nout(v) == 0 || continue

        # Beware! This turned out to be a lot harder than I first thought.
        # I'm not sure the algorithm works (or realizes that it won't work) for all possible cases.

        # This obviously only works if v is inside a parallel path which will later be concatenated

        # To make sure it is, we look ahead in forward and backward direction.
        # If we don't see 1) an input vertex 2) a vertex without outputs (i.e an output vertex) we are good to go
        fseen, fpoints = findforkpoint(v, topoligical_order, inputs, outputs)
        isempty(fpoints) && continue

        bseen, bpoints = findforkpoint(v, topoligical_order, outputs, inputs)
        isempty(bpoints) && continue

        # Ok, fpoints are all vertices where forks with v in them join and bpoints are all vertices where forks with v in them begin

        # If we start removing all seen vertices which are input to an fpoint until we hit a bpoint we should have removed the fork with v in it, right?
        seen = union(fseen, bseen)
        to_rm = intersect(vcat(inputs.(fpoints)...), seen)
        foreach(fpoint -> remove_all_inputs(fpoint, seen, bpoints), to_rm)
    end
end

function findforkpoint(v::AbstractVertex, topoligical_order, f1=inputs, f2=outputs, seen = vcat(v, f1(v)), points = AbstractVertex[])

    if all(x -> x in seen, f1(v))

        # Check if we came across a vertex which previously was thought to be a forkpoint and remove it if so
        if v in points
            deleteat!(points, indexin([v], points))
        end
        # Always visit in reverse topoligical order to mitigate chasing down paths just because we haven't explored them yet
        nextverts = reverse(topoligical_order[unique(indexin(f2(v), topoligical_order))])
        push!(seen, filter(v2 -> !in(v2, seen), nextverts)...)
        foreach(v2 -> findforkpoint(v2, topoligical_order, f1, f2, seen, points), nextverts)
    elseif !(v in points)
        push!(points, v)
    end

    return seen, points
end

function remove_all_inputs(v, seen, stop)
    v in stop && return
    foreach(vrm -> remove_all_inputs(vrm, seen, stop), intersect(inputs(v), seen))
    remove!(v, RemoveStrategy(ConnectNone(), NoSizeChange()))
end
