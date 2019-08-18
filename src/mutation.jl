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
    minsize + Δ <= 0 && return
    Δnout(v, Δ)
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

(m::AddVertexMutation)(v::AbstractVertex) = insert!(v, vi -> m.s(vi, outsize=nout(vi)), m.outselect)

# TODO: Belongs in NaiveNASlib
struct CheckAligned <:AbstractAlignSizeStrategy
    ifnot
end
CheckAligned() = CheckAligned(IncreaseSmaller())
function NaiveNASlib.prealignsizes(s::CheckAligned, vin, vout, will_rm)
    nout(vin) == NaiveNASlib.tot_nin(vout) && return true
    return NaiveNASlib.prealignsizes(s.ifnot, vin, vout, will_rm)
end

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
RemoveVertexMutation() = RemoveVertexMutation(RemoveStrategy(CheckAligned(IncreaseSmaller(DecreaseBigger(AlignSizeBoth(FailAlignSizeWarn()))))))

(m::RemoveVertexMutation)(v::AbstractVertex) = remove!(v, m.s)

"""
    NeuronSelectMutation{T} <: AbstractMutation{AbstractVertex}
    NeuronSelectMutation(m::AbstractMutation{AbstractVertex})
    NeuronSelectMutation(rankfun, m::AbstractMutation{AbstractVertex})
    NeuronSelectMutation(rankfun::Function, strategy, m::RecordMutation{AbstractVertex})

Selects neurons of vertices mutated by the wrapped `RecordMutation`.

Possible to select ranking method for neurons using `rankfun` which takes a mutated vertex as input and returns indices of selected neurons.

How to select neurons depends a bit on what operation the wrapped `RecordMutation` performs. If not supplied explicitly an attempt to infer it will be made, resulting in an error if not possible.
"""
struct NeuronSelectMutation{T} <: AbstractMutation{AbstractVertex}
    rankfun::Function
    strategy::T
    m::RecordMutation{AbstractVertex}
end
NeuronSelectMutation(rankfun, m::AbstractMutation{AbstractVertex}) = NeuronSelectMutation(rankfun, neuron_select_strategy(m), RecordMutation(m))
NeuronSelectMutation(m::AbstractMutation{AbstractVertex}) = NeuronSelectMutation(default_neuronselect, m)
NeuronSelectMutation(m::RecordMutation{AbstractVertex}) = NeuronSelectMutation(default_neuronselect, neuron_select_strategy(m.m), m)

(m::NeuronSelectMutation)(v::AbstractVertex) = m.m(v)

neuron_select_strategy(::T) where T <:AbstractMutation = error("Neuron select strategy not implemented for $T")
neuron_select_strategy(::RemoveVertexMutation) = RemoveVertex()
neuron_select_strategy(::NoutMutation) = Nout()

struct Nout
    s::AbstractSelectionStrategy
end
Nout() = Nout(NoutExact())
struct RemoveVertex
    s::AbstractSelectionStrategy
end
RemoveVertex() = RemoveVertex(NoutExact())

"""
    select(m::NeuronSelectMutation)

Select neurons for each `AbstractVertex` mutated by `m`.
"""
function select(m::NeuronSelectMutation)
    for v in m.m.mutated
        select_neurons(m.strategy, v, m.rankfun)
    end
end

default_neuronselect(vsel, vvals) = neuron_value(trait(vvals), vvals)


NaiveNASflux.neuron_value(t::DecoratingTrait, v) = neuron_value(base(t), v)
NaiveNASflux.neuron_value(::Immutable, v) = ones(nout(v))
NaiveNASflux.neuron_value(::MutationSizeTrait, v) = neuron_value(v)

select_neurons(::T, v::AbstractVertex, rankfun::Function) where T = error("Neuron select not implemented for $T")
function select_neurons(strategy::Nout, v::AbstractVertex, rankfun::Function)
    v in vcat(outputs.(inputs(v))...) || return # if vertex was removed

    Δout = nout(v) - nout_org(v)

    if Δout != 0
        execute, inds = select_outputs(strategy.s, v, rankfun(v, v))
        if execute
            Δnout(v, inds)
        end
    end
end

function select_neurons(strategy::RemoveVertex, v::AbstractVertex, rankfun::Function)
    # This could be messy due to many different cases:
    # 1: Outsize of input vertex is increased
    # 2: Insize of output vertex is increased
    # 3: Outsize of input vertex is decreased
    # 4: Insize of output vertex is decreased
    # as well as all combinations of two of the above.

    # Handle cases when 1 or 3 has happened:
    # This is easy as it is equivalent to handling a normal Nout change except we pretend that outputs have already been visited when propagating selected indices.
    Δins = nin(v) - nin_org(v)
    skip_outputs = [];
    for i in eachindex(Δins)
        Δins[i] == 0 && continue
        vin = inputs(v)[i]
        # Basic idea: Don't want to change anything (what was) outputs to (now removed) vertex v
        # Complication: In some cases vin was changed due to some other reason than vertex removal
        # To handle this, we search for unchanged vertices and exclude them
        s = NaiveNASlib.VisitState{Vector{Int}}(vin)

        # I,... think this is safe to do. Its only an optimizaion though afaik.
        if !istransparent(v)
            NaiveNASlib.visited_in!.(s, outputs(v))
        end

        execute, inds = select_outputs(strategy.s, vin, rankfun(vin, vin), s.out, s.in)
        if execute
            Δnout(vin, inds, s=s)
        end

        if istransparent(vin)
            append!(skip_outputs, outputs(vin))
        end
    end

    # Handle cases when 2 and 4 has happened
    # Cases 2 and 4 are a bit trickier as it is now the input size which has changed, meaning that nout of all input verteces was not changed. Reason its trickier is because there is no helper method for it (yet).
    # The strategy is to select outputs from the input vertices instead
    # Challenge with that approach is that sizes where not aligned to begin with, so if no special action is taken one will end up with tasks like "select 100 unique values out of these 50 values".
    if nout(v) - nout_org(v) != 0
        for vout in filter(vo -> !in(vo, skip_outputs), unique(outputs(v)))
            select_neurons_nout_org_not_aligned(vout, v, strategy.s, rankfun)
        end
    end
end

istransparent(v::AbstractVertex) = istransparent(trait(v))
istransparent(t::DecoratingTrait) = istransparent(base(t))
istransparent(::SizeAbsorb) = false
istransparent(::NaiveNASlib.SizeTransparent) = true

select_neurons_nout_org_not_aligned(vout, vfrom, strat, rf) = select_neurons_nout_org_not_aligned(trait(vout), vout, vfrom, strat, rf)
select_neurons_nout_org_not_aligned(t::DecoratingTrait, vout, vfrom, strat, rf) = select_neurons_nout_org_not_aligned(base(t), vout, vfrom, strat, rf)
function select_neurons_nout_org_not_aligned(::SizeInvariant, vout, vfrom, strategy, rankfun)
    # Select outputs from vout directly and let them propagate backwards to not risk selecting different outputs from each input vertex
    s = NaiveNASlib.VisitState{Vector{Int}}(vout)
    NaiveNASlib.visited_out!.(s, filter(voi -> voi in inputs(vfrom), inputs(vout)))

    cdict = validouts(vout, Set(s.out), Set(s.in), true)
    execute, inds = select_outputs(NoutMainVar(strategy, NoutRelaxSize(0.01, 2)), vout, rankfun(vout, vout), cdict)

    if execute
        Δnout(vout, inds, s=s)
    end
end
function select_neurons_nout_org_not_aligned(::MutationSizeTrait, vout, vfrom, strategy, rankfun)
    # Little bit trickier as changing nout of vout will not propagate to its inputs
    # Instead, we select output neurons from all its inputs and use this to select inputs
    s = NaiveNASlib.VisitState{Vector{Int}}(vout)
    NaiveNASlib.visited_out!.(s, inputs(vout))

    # Note that one can probably assume that SizeAbsorb only has one input so this might be a bit overgeneralized
    Δ = map(enumerate(inputs(vout))) do (i, voi)

        voi in inputs(vfrom) || return missing

        # out=false because we are actually selecting for vout in the input direction
        cdict = validouts(voi, Set(s.out), Set(s.in), false)
        valid, inds = select_outputs(NoutMainVar(strategy, NoutRelaxSize(0.01, 2)), voi, rankfun(voi, vfrom), cdict)
        return valid ? inds : missing
    end

    if !all(ismissing.(Δ))
        Δnin(vout, Δ..., s=s)
    end

end

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
