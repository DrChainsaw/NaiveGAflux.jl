"""
    AbstractMutation{T}

Abstract type defining a mutation operation on entities of type `T`.

Implementations are expected to be callable using an entity of type `T` as only input.
"""
abstract type AbstractMutation{T} end

"""
    AbstractCrossover{T}

Type alias for `AbstractMutation{Tuple{T,T}}` defining a crossover of two entities of type `T`.

Implementations are expected to be callable using a tuple of two type `T` as only input.
"""
AbstractCrossover{T} = AbstractMutation{Tuple{T,T}}

"""
    DecoratingMutation{T}

Abstract type indicating that the type itself does not perform any mutation but wraps a type which might do.
"""
abstract type DecoratingMutation{T} <: AbstractMutation{T} end
wrapped(m::DecoratingMutation) = m.m

mutationleaves(m::DecoratingMutation) = (mutationleaves(wrapped(m))...,)
mutationleaves(tm::Tuple) = mapreduce(mutationleaves, (t1,t2) -> (t1...,t2...), tm)
mutationleaves(m) = tuple(m)

"""
    MutationProbability{T} <: DecoratingMutation{T}
    MutationProbability(m::AbstractMutation{T}, p::Probability)
    MutationProbability(m::AbstractMutation{T}, p::Number)

Applies `m` with probability `p`.
"""
struct MutationProbability{T} <: DecoratingMutation{T}
    m::AbstractMutation{T}
    p::Probability
end
MutationProbability(m::AbstractMutation{T}, p::Number) where T = MutationProbability(m, Probability(p))
(m::MutationProbability)(e) = apply(() -> m.m(e), m.p, () -> e)

"""
    WeightedMutationProbability{T,F} <: DecoratingMutation{T}
    WeightedMutationProbability(m::AbstractMutation::T, pfun::F)

Applies `m` to an entity `e` with a probability `pfun(e)`.
"""
struct WeightedMutationProbability{T,F} <: DecoratingMutation{T}
    m::AbstractMutation{T}
    pfun::F
end
(m::WeightedMutationProbability)(e) = apply(() -> m.m(e), m.pfun(e), () -> e)

"""
    HighValueMutationProbability(m::AbstractMutation{T}, pbase::Real, rng=rng_default; spread=0.5)

Return a `WeightedMutationProbability` which applies `m` to vertices with an (approximately) average probability of `pbase` and where high `neuron_value` compared to other vertices in same graph means higher probability.

Parameter `spread` can be used to control how much the difference in probability is between high and low values. High spread means high difference while low spread means low difference.
"""
HighValueMutationProbability(m::AbstractMutation{T}, pbase::Real, rng=rng_default;spread=0.5) where T <: AbstractVertex = WeightedMutationProbability(m, weighted_neuron_value_high(pbase, rng,spread=spread))

"""
    LowValueMutationProbability(m::AbstractMutation{T}, pbase::Real, rng=rng_default; spread=2)

Return a `WeightedMutationProbability` which applies `m` to vertices with an (approximately) average probability of `pbase` and where low `neuron_value` compared to other vertices in same graph means higher probability.

Parameter `spread` can be used to control how much the difference in probability is between high and low values. High spread means high difference while low spread means low difference.
"""
LowValueMutationProbability(m::AbstractMutation{T}, pbase::Real, rng=rng_default;spread=2) where T <: AbstractVertex = WeightedMutationProbability(m, weighted_neuron_value_low(pbase, rng, spread=spread))


weighted_neuron_value_high(pbase, rng=rng_default; spread=0.5) = function(v::AbstractVertex)
    ismissing(neuron_value(v)) && return pbase
    return Probability(fixnan(pbase ^ normexp(v, spread), pbase), rng)
end

weighted_neuron_value_low(pbase, rng=rng_default;spread=2) = function(v::AbstractVertex)
    ismissing(neuron_value(v)) && return pbase
    return Probability(fixnan(pbase ^ (1/normexp(v, 1/spread)), pbase), rng)
end

fixnan(x, rep) = isnan(x) ? rep : clamp(x, 0.0, 1.0)

# This is pretty hacky and arbitrary. Change to something better
function normexp(v::AbstractVertex, s)
    allvertices = filter(allow_mutation, all_in_graph(v))
    allvalues = map(vi -> neuron_value(vi), allvertices)
    meanvalues = map(mean, skipmissing(allvalues))
    meanvalue = mean(meanvalues)
    maxvalue = maximum(meanvalues)
    value = mean(neuron_value(v))
    # Basic idea: maxvalue - value means the (to be) exponent is <= 0 while the division seems to normalize so that average of pbase ^ normexp across allvertices is near pbase (no proof!). The factor 2 is just to prevent probability of vertex with maxvalue to be 1.
    return (2maxvalue^s - value^s) / (2maxvalue^s - meanvalue^s)
end


"""
    MutationChain{T} <: DecoratingMutation{T}
    MutationChain(m::AbstractMutation{T}...)

Chains multiple `AbstractMutation{T}`s after each other.

Input entities will be mutated by the first `AbstractMutation{T}` in the chain and the output will be fed into the next `AbstractMutation{T}` in the chain and so on. The output from the last `AbstractMutation{T}` is returned.
"""
struct MutationChain{T} <: DecoratingMutation{T}
    m::Tuple{Vararg{AbstractMutation{T}}}
end
MutationChain(m::AbstractMutation{T}...) where T = MutationChain(m)
(m::MutationChain)(e) = foldl((ei, mi) -> mi(ei), m.m; init=e)

"""
    RecordMutation{T} <: DecoratingMutation{T}
    RecordMutation(m::AbstractMutation{T})

Records all mutated entities.

Intended use case is to be able to do parameter selection on mutated vertices.
"""
struct RecordMutation{T} <: DecoratingMutation{T}
    m::AbstractMutation{T}
    mutated::AbstractVector{T}
end
RecordMutation(m::AbstractMutation{T}) where T = RecordMutation(m, T[])
function (m::RecordMutation)(e)
    em = m.m(e)
    push!(m.mutated, em)
    return em
end
function fetchmutated!(m::RecordMutation)
    mutated = copy(m.mutated)
    deleteat!(m.mutated, eachindex(m.mutated))
    return mutated
end

"""
    LogMutation{T} < :DecoratingMutation{T}
    LogMutation(strfun, m::AbstractMutation{T}; level = Logging.Info, nextlogfun=e -> PrefixLogger("   "))
    LogMutation(strfun, level::LogLevel, nextlogfun, m::AbstractMutation{T})

Logs all mutation operations.

Argument `strfun` maps the mutated entity to the logged string.

Calling `nextlogfun(e)` where `e` is the entity to mutate produces an `AbstractLogger` which will be used when applying `m(e)`.

By default, this is used to add a level of indentation to subsequent logging calls which makes logs of hierarchical mutations (e.g. mutate a CompGraph by applying mutations to some of its vertices) easier to read. Set `nextlogfun = e -> current_logger()` to remove this behaviour.
"""
struct LogMutation{F,L<:LogLevel,LF,T} <: DecoratingMutation{T}
    strfun::F
    level::L
    nextlogfun::LF
    m::AbstractMutation{T}
end
LogMutation(strfun, m::AbstractMutation{T}; level = Logging.Info, nextlogfun=e -> PrefixLogger("   ")) where T = LogMutation(strfun, level, nextlogfun, m)
function (m::LogMutation)(e)
    @logmsg m.level m.strfun(e)
    return with_logger(() -> m.m(e), m.nextlogfun(e))
end

"""
    MutationFilter{T} <: DecoratingMutation{T}
    MutationFilter(predicate, m)

Applies mutation `m` only for entities `e` for which `predicate(e)` returns true.
"""
struct MutationFilter{T} <: DecoratingMutation{T}
    predicate
    m::AbstractMutation{T}
end
function (m::MutationFilter)(e)
    m.predicate(e) && return m.m(e)
    return e
end


"""
    VertexMutation <: DecoratingMutation{CompGraph}
    VertexMutation(m::AbstractMutation{AbstractVertex}, s::AbstractVertexSelection)
    VertexMutation(m::AbstractMutation{AbstractVertex})

Applies a wrapped `AbstractMutation{AbstractVertex}` to each selected vertex in a `CompGraph`.

Vertices to select is determined by the configured `AbstractVertexSelection`.
"""
struct VertexMutation <: DecoratingMutation{CompGraph}
    m::AbstractMutation{AbstractVertex}
    s::AbstractVertexSelection
end
VertexMutation(m::AbstractMutation{AbstractVertex}) = VertexMutation(m, FilterMutationAllowed())
function (m::VertexMutation)(g::CompGraph)
    for v in select(m.s, g, m)
        m.m(v)
    end
    return g
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
    ismissing(Δfactor) && return v

    shift = m.minrel
    scale = m.maxrel - m.minrel

    x = rand(m.rng) * scale + shift
    xq = (nout(v) * x) ÷ Δfactor
    Δ = Int(sign(x) * max(Δfactor, abs(xq) * Δfactor))

    minsize = min(nout(v), minimum(nout.(findterminating(v, inputs))))
    minsize + Δ <= Δfactor && return v

    fallback = ΔNout{Relaxed}(v, Δ, LogΔSizeExec(Logging.Warn, "Could not change nout of $v by $(Δ) after relaxation! Vertex not changed!", ΔSizeFailNoOp()))
    strategy = ΔNout{Exact}(v, Δ, LogΔSizeExec(Logging.Warn, "Could not change nout of $v by $(Δ)! Relaxing constraints...", fallback))

    Δsize(strategy , all_in_Δsize_graph(v, Output()))
    return v
end

"""
    AddVertexMutation <:AbstractMutation{AbstractVertex}
    AddVertexMutation(s::AbstractArchSpace, outselect::Function, WeightInit::AbstractWeightInit, rng::AbstractRNG)
    AddVertexMutation(s, outselect::Function=identity)
    AddVertexMutation(s, rng::AbstractRNG)
    AddVertexMutation(s, wi::AbstractWeightInit)

Insert a vertex from the wrapped `AbstractArchSpace` `s` after a given vertex `v`.

The function `outselect` takes an `AbstractVector{AbstractVertex}` representing the output of `v` and returns an `AbstractVector{AbstractVertex}` which shall be reconnected to the vertex `v'` returned by `s`. Defaults to `identity` meaning all outputs of `v` are reconnected to `v'`.
"""
struct AddVertexMutation <:AbstractMutation{AbstractVertex}
    s::AbstractArchSpace
    outselect::Function
    weightinit::AbstractWeightInit
    rng::AbstractRNG
end
AddVertexMutation(s, outselect::Function=identity) = AddVertexMutation(s, outselect, IdentityWeightInit(), rng_default)
AddVertexMutation(s, rng::AbstractRNG) = AddVertexMutation(s, identity, IdentityWeightInit(), rng)
AddVertexMutation(s, wi::AbstractWeightInit) = AddVertexMutation(s, identity, wi, rng_default)

function (m::AddVertexMutation)(v::AbstractVertex)
    insert!(v, vi -> m.s(name(vi), vi, m.rng, outsize=nout(vi), wi=m.weightinit), m.outselect)
    return v
end

"""
    RemoveVertexMutation <:AbstractMutation{AbstractVertex}
    RemoveVertexMutation(s::RemoveStrategy)
    RemoveVertexMutation()

Remove the given vertex `v` using the configured `RemoveStrategy`.

Default size align strategy is `IncreaseSmaller -> DecreaseBigger -> AlignSizeBoth -> FailAlignSizeWarn -> FailAlignSizeRevert`.

Default reconnect strategy is `ConnectAll`.

Note: High likelyhood of large accuracy degradation after applying this mutation.
"""
struct RemoveVertexMutation <:AbstractMutation{AbstractVertex}
    s::RemoveStrategy
end
RemoveVertexMutation() = RemoveVertexMutation(RemoveStrategy(CheckAligned(CheckNoSizeCycle(ApplyMutation(SelectOutputs(select = SelectDirection(OutSelect{NaiveNASlib.Exact}(LogSelectionFallback("Reverting...", NoutRevert()))),
valuefun = default_neuronselect, align=IncreaseSmaller(DecreaseBigger(AlignSizeBoth(FailAlignSizeWarn()))))), FailAlignSizeWarn(msgfun = (vin,vout) -> "Can not remove vertex $(name(vin))! Size cycle detected!")))))

function (m::RemoveVertexMutation)(v::AbstractVertex)
    remove!(v, m.s)
    return v
end

"""
    AddEdgeMutation <: AbstractMutation{AbstractVertex}
    AddEdgeMutation(p; rng=rng_default, mergefun=default_mergefun(rng=rng), filtfun=no_shapechange, valuefun=default_neuronselect)
    AddEdgeMutation(p::Probability; rng=rng_default, mergefun=default_mergefun(rng=rng), filtfun=no_shapechange, valuefun=default_neuronselect)

Add an edge from a vertex `vi` to another vertex `vo` randomly selected from `vs = filtfun(vi)`.

Higher values of `p` will give more preference to earlier vertices of `vs`.

If `vo` is not capable of having multiple inputs (determined by `singleinput(v) == true`), `vm = mergefun(voi)` where `voi` is a randomly selected input to `vo` will be used instead of `vo` and `vo` will be added as the output of `vm`.

When selecting neurons/outputs after any eventual size change the values `valuefun(v)` will be used to determine the value of each output in vertex `v`. Note that `length(valuefun(v)) == nout_org(v)` must hold.

Note: High likelyhood of large accuracy degradation after applying this mutation.
"""
struct AddEdgeMutation{F1, F2, F3, R} <: AbstractMutation{AbstractVertex}
    mergefun::F1
    filtfun::F2
    valuefun::F3
    p::Probability
    rng::R
end
AddEdgeMutation(p; rng=rng_default, mergefun=default_mergefun(rng=rng), filtfun=no_shapechange, valuefun=default_neuronselect) = AddEdgeMutation(Probability(p, rng), rng=rng, mergefun=mergefun, filtfun=filtfun, valuefun=valuefun)
AddEdgeMutation(p::Probability; rng=rng_default, mergefun=default_mergefun(rng=rng), filtfun=no_shapechange, valuefun=default_neuronselect) = AddEdgeMutation(mergefun, filtfun, valuefun, p, rng)

default_mergefun(pconc = 0.5; rng=rng_default, traitfun = MutationShield ∘ RemoveIfSingleInput ∘ validated() ∘ default_logging(), layerfun = ActivationContribution) = function(vin)
    if rand(rng) > pconc
        return invariantvertex(layerfun(+), vin, traitdecoration=traitfun ∘ named(name(vin) * ".add"))
    end
    return concat(vin ,mutation=IoChange, traitfun = traitfun ∘ named(name(vin) * ".cat"), layerfun=layerfun)
end

function no_shapechange(vi)
    # all_in_graph is not sorted, and we want some kind of topoligical order here so that earlier indices are closer to vi
    allsorted = mapreduce(NaiveNASlib.flatten, vcat, filter(v -> isempty(outputs(v)), all_in_graph(vi))) |> unique
    
    # Vertices which have the same input as vi and are singleinput
    #   Reason is that this will cause a new vertex to be added between the target output vertex vo
    #   and the input vertex to vi (vii) and this is detected as a size cycle which causes 
    #   try_add_edge to fail.
    inouts = filter(singleinput, mapreduce(outputs, vcat, inputs(vi); init=[]))
    # All vertices which are after vi in the topology
    vsafter = setdiff(allsorted, NaiveNASlib.flatten(vi), outputs(vi), inouts)
    
    vitrace = shapetrace(vi) 
    viorder = allΔshapetypes(vitrace)
    viΔshape = squashshapes(vitrace; order=viorder)

    return filter(vsafter) do vafter
        all(inputs(vafter)) do v
        t = shapetrace(v)
        vΔshape = squashshapes(t; order=union(viorder, allΔshapetypes(t)))
        return viΔshape == vΔshape
        end
    end
end

function (m::AddEdgeMutation)(vi::AbstractVertex)
    # All vertices for which it is allowed to add vi as an input
    allverts = filter(allow_mutation, m.filtfun(vi))
    isempty(allverts) && return vi

    # Higher probability to select a vertex close to v is desired behaviour
    # One line less than a for loop => FP wins!!
    selfun(::Nothing, vc) = apply(m.p) ? vc : nothing
    selfun(vs, vd) = vs
    vo = foldl(selfun, allverts, init=nothing)
    vo = isnothing(vo) ? rand(m.rng, allverts) : vo

    try_add_edge(vi, vo, m.mergefun, m.valuefun)
    return vi
end

function try_add_edge(vi, vo, mergefun, valuefun=default_neuronselect)

    # Need to add a vertex which can handle multiple inputs if vo is single input only
    # For cleaning up added vertex if the whole operation fails
    cleanup_failed = () -> nothing
    if singleinput(vo)
        voi = inputs(vo)[1]
        # If the input to vo is capable of multi input we don't need to create a new vertex
        # We must also check that this input does not happen to be an input to vi as this would create a cycle in the graph
        if singleinput(voi) || voi in NaiveNASlib.flatten(vi)
            vm = mergefun(voi)
            # Insert vm between voi and vo, i.e voi -> vo turns into voi -> vm -> vo
            # vs -> [vo] means only add the new vertex between voi and vo as voi could have other outputs
            insert!(voi, vv -> vm, vs -> [vo])
            cleanup_failed = function()
                length(inputs(vm)) > 1 && return
                remove!(vm, RemoveStrategy(NoSizeChange()))
            end
            vo = vm # vm is the one we shall add an edge to
            @debug "Create new vertex for merging $(name(vo))"
        else
            vo = voi
        end
    end
    # This is mainly because FailAlignSizeRevert does not work when the same vertex is input more than once, but it also seems kinda redundant.
    vi in inputs(vo) && return
    @debug "Create edge between $(name(vi)) and $(name(vo))"
    create_edge!(vi, vo, strategy = create_edge_strat(vo, valuefun))
    cleanup_failed()
end
# Need to override this one for strange types e.g. layers which support exactly 2 inputs or something.
singleinput(v) = isempty(inputs(v)) || length(inputs(v)) == 1

create_edge_strat(v::AbstractVertex, valuefun) = create_edge_strat(trait(v), valuefun)
create_edge_strat(d::DecoratingTrait, valuefun) = create_edge_strat(base(d), valuefun)
function create_edge_strat(::SizeInvariant, valuefun)
    alignstrat = IncreaseSmaller(DecreaseBigger(AlignSizeBoth(FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))!"))))

    selectstrat = OutSelect{Exact}(LogSelectionFallback("Reverting...", NoutRevert()))

    okstrat = PostApplyMutation(SelectOutputs(selectstrat, alignstrat, valuefun))

    # Tricky failure case: It is possible that CheckCreateEdgeNoSizeCycle does not detect any size cycle until after the edge has been created. When this happens, we run PostSelectOutputs to revert all size changes before removing the edge. The latter might not be strictly needed (I really don't know actually), but it does stay true to the contract of "no size change if operation does not succeed".
    nokstrat = FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))! Size cycle detected! Reverting...", andthen=PostSelectOutputs(select=NoutRevert(), align=NoSizeChange(), fallback=FailAlignSizeRevert())) # NoutRevert returns success=false, meaning that fallback will be invoked

    return CheckCreateEdgeNoSizeCycle(okstrat, nokstrat)
end
function create_edge_strat(::SizeStack, valuefun)

    alignstrat = PostAlignJuMP(DefaultJuMPΔSizeStrategy(), fallback=FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))!"))

    selectstrat = OutSelect{Exact}(LogSelectionFallback("Reverting...", NoutRevert()))

    okstrat = PostApplyMutation(PostSelectOutputs(selectstrat, alignstrat, valuefun, FailAlignSizeRevert()))

    nokstrat = FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))! Size cycle detected! Reverting...")
    return CheckCreateEdgeNoSizeCycle(okstrat, nokstrat)
end

"""
    RemoveEdgeMutation <: AbstractMutation{AbstractVertex}
    RemoveEdgeMutation(;valuefun=default_neuronselect, rng=rng_default)

Remove an edge from a vertex `vi` to another vertex `vo` randomly selected from `outputs(vi)`.

Vertex `vi` must have more than one output and vertex `vo` must have more than one output for the edge to be removed. Otherwise no change is made.

If there are multiple edges between `vi` and `vo` no change will be made due to NaiveNASlib not being able to revert a failed operation in this case..

When selecting neurons/outputs after any eventual size change the values `valuefun(v)` will be used to determine the value of each output in vertex `v`. Note that `length(valuefun(v)) == nout_org(v)` must hold.

Note: High likelyhood of large accuracy degradation after applying this mutation.
"""
struct RemoveEdgeMutation{F, R} <: AbstractMutation{AbstractVertex}
    valuefun::F
    rng::R
end
RemoveEdgeMutation(;valuefun=default_neuronselect, rng=rng_default) = RemoveEdgeMutation(valuefun, rng)

function (m::RemoveEdgeMutation)(vi::AbstractVertex)
    length(outputs(vi)) < 2 && return vi

    allverts = filter(vo -> length(inputs(vo)) > 1, outputs(vi))

    isempty(allverts) && return vi

    vo = rand(m.rng, allverts)
    sum(inputs(vo) .== vi) > 1 && return vi# Not implemented in NaiveNASlib

    @debug "Remove edge between $(name(vi)) and $(name(vo))"
    remove_edge!(vi, vo, strategy=remove_edge_strat(vo, m.valuefun))
    return vi
end

remove_edge_strat(v::AbstractVertex, valuefun) = remove_edge_strat(trait(v), valuefun)
remove_edge_strat(d::DecoratingTrait, valuefun) = remove_edge_strat(base(d), valuefun)
remove_edge_strat(::SizeInvariant, valuefun) = NoSizeChange()
remove_edge_strat(t::SizeStack, valuefun) = create_edge_strat(t, valuefun)

"""
    KernelSizeMutation{N} <: AbstractMutation{AbstractVertex}
    KernelSizeMutation(Δsizespace::AbstractParSpace{N, Int}; maxsize, pad, rng)
    KernelSizeMutation2D(absΔ::Integer;maxsize, pad, rng)
    KernelSizeMutation(absΔ::Integer...;maxsize, pad, rng)

Mutate the size of filter kernels of convolutional layers.

Note: High likelyhood of large accuracy degradation after applying this mutation.
"""
struct KernelSizeMutation{N,F,P} <: AbstractMutation{AbstractVertex}
    Δsizespace::AbstractParSpace{N, Int}
    maxsize::F
    pad::P
    rng::AbstractRNG
end
KernelSizeMutation(Δsizespace::AbstractParSpace{N, Int}; maxsize = v -> ntuple(i->Inf,N), pad=SamePad(), rng=rng_default) where N = KernelSizeMutation(Δsizespace, maxsize, pad, rng)
KernelSizeMutation2D(absΔ::Integer;maxsize = v -> (Inf,Inf), pad=SamePad(), rng=rng_default) = KernelSizeMutation(absΔ, absΔ, maxsize = maxsize, pad=pad, rng=rng)
KernelSizeMutation(absΔ::Integer...;maxsize = v -> ntuple(i->Inf, length(absΔ)), pad=SamePad(), rng=rng_default) = KernelSizeMutation(ParSpace(UnitRange.(.-absΔ, absΔ));maxsize = maxsize, pad=pad, rng=rng)

function (m::KernelSizeMutation{N})(v::AbstractVertex) where N
    layertype(v) isa FluxConvolutional{N} || return
    l = layer(v)

    currsize = size(NaiveNASflux.weights(l))[1:N]
    Δsize = Int.(clamp.(m.Δsizespace(m.rng), 1 .- currsize, m.maxsize(v) .- currsize)) # ensure new size is > 0 and < maxsize
    # This will eventually boil down to Setfield doing its thing, and that won't be using any convenience constructors
    pad = Flux.calc_padding(typeof(l), m.pad, currsize .+ Δsize, dilation(l), stride(l))
    mutate_weights(v, KernelSizeAligned(Δsize, pad))
    return v
end
dilation(l) = l.dilation
stride(l) = l.stride

"""
    ActivationFunctionMutation{T,R} <: AbstractMutation{AbstractVertex} where {T <: AbstractParSpace{1}, R <: AbstractRNG}
    ActivationFunctionMutation(actspace::AbstractParSpace{1}, rng::AbstractRNG)
    ActivationFunctionMutation(acts...;rng=rng_default)
    ActivationFunctionMutation(acts::AbstractVector;rng=rng_default)

Mutate the activation function of layers which have an activation function.

Note: High likelyhood of large accuracy degradation after applying this mutation.
"""
struct ActivationFunctionMutation{T,R} <: AbstractMutation{AbstractVertex} where {T <: AbstractParSpace{1}, R <: AbstractRNG}
    actspace::T
    rng::R
end
ActivationFunctionMutation(acts...;rng=rng_default) = ActivationFunctionMutation(collect(acts), rng=rng)
ActivationFunctionMutation(acts::AbstractVector;rng=rng_default) = ActivationFunctionMutation(ParSpace(acts), rng)

function (m::ActivationFunctionMutation)(v::AbstractVertex)
    m(layertype(v), v)
    return v
end
function (m::ActivationFunctionMutation)(t, v) end
(m::ActivationFunctionMutation)(::Union{FluxDense, FluxConvolutional}, v) = setlayer(v, (σ = m.actspace(m.rng),))
(m::ActivationFunctionMutation)(::FluxParNorm, v) = setlayer(v, (λ = m.actspace(m.rng),))
function (m::ActivationFunctionMutation)(::FluxRnn, v)
    newcell = setproperties(layer(v).cell, (σ = m.actspace(m.rng),))
    setlayer(v, (cell = newcell,))
end

# TODO: Move to NaiveNASflux??
function setlayer(x, propval) end
setlayer(v::AbstractVertex, propval) = setlayer(base(v), propval)
setlayer(v::CompVertex, propval) = setlayer(v.computation, propval)
setlayer(m::AbstractMutableComp, propval) = setlayer(NaiveNASflux.wrapped(m), propval)
setlayer(m::NaiveNASflux.ResetLazyMutable, propval) = setlayer(m.wrapped, propval)
setlayer(m::NaiveNASflux.MutationTriggered, propval) = setlayer(m.wrapped, propval)
function setlayer(m::MutableLayer, propval)
    m.layer = setproperties(m.layer, propval)
end


"""
    NeuronSelectMutation{T} <: AbstractMutation{AbstractVertex}
    NeuronSelectMutation(m::AbstractMutation{AbstractVertex})
    NeuronSelectMutation(rankfun, m::AbstractMutation{AbstractVertex})
    NeuronSelectMutation(rankfun::Function, strategy, m::RecordMutation{AbstractVertex})

Selects neurons of vertices mutated by the wrapped `RecordMutation`.

Possible to select ranking method for neurons using `rankfun` which takes a mutated vertex as input and value/utility per neuron (higher is better).

How to select neurons depends a bit on what operation the wrapped `RecordMutation` performs. If not supplied explicitly an attempt to infer it will be made, resulting in an error if not possible.
"""
struct NeuronSelectMutation{T, MT} <: AbstractMutation{MT}
    rankfun::Function
    strategy::T
    m::RecordMutation{MT}
end
NeuronSelectMutation(rankfun, m::AbstractMutation) = NeuronSelectMutation(rankfun, neuron_select_strategy(m), m)
NeuronSelectMutation(rankfun, strategy, m::AbstractMutation) = NeuronSelectMutation(rankfun, strategy, RecordMutation(m))
NeuronSelectMutation(m::AbstractMutation) = NeuronSelectMutation(default_neuronselect, m)
NeuronSelectMutation(m::RecordMutation) = NeuronSelectMutation(default_neuronselect, neuron_select_strategy(m.m), m)

(m::NeuronSelectMutation)(v::AbstractVertex) = m.m(v)
(m::NeuronSelectMutation)(vs::NTuple{N, AbstractVertex}) where N = m.m(vs)

neuron_select_strategy(::T) where T <:AbstractMutation = error("Neuron select strategy not implemented for $T")
neuron_select_strategy(::RemoveVertexMutation) = NeuronSelectOut()
neuron_select_strategy(::NoutMutation) =  NeuronSelectOut()
neuron_select_strategy(::AbstractCrossover) = NeuronSelectFlatten()

struct NeuronSelectFlatten{T}
    s::T
end
NeuronSelectFlatten() = NeuronSelectFlatten(NeuronSelectOut())

struct NeuronSelectOut
    s::AbstractSelectionStrategy
end
NeuronSelectOut() = NeuronSelectOut(ApplyAfter())

"""
    select(m::NeuronSelectMutation)

Select neurons for each `AbstractVertex` mutated by `m`.
"""
select(m::NeuronSelectMutation) = select_neurons(m.strategy, fetchmutated!(m.m), m.rankfun)

function select_neurons(s, vs::AbstractArray{NTuple{N, AbstractVertex}}, rankfun) where N
    vsn = reduce((t, r) -> vcat.(t, r), vs)
    map(vsi -> select_neurons(s, vsi, rankfun), vsn) |> all
end
select_neurons(s, v::AbstractVertex, rankfun) = select_neurons(s, [v], rankfun)

function select_neurons(s::NeuronSelectFlatten, vs::AbstractArray{<:AbstractVertex}, rankfun)
    allvs = foldr(NaiveNASlib.flatten, vs, init=AbstractVertex[])
    select_neurons(s.s, allvs, rankfun)
end

function select_neurons(strategy::NeuronSelectOut, vs::AbstractArray{<:AbstractVertex}, rankfun)
    vchanged = filter(vs) do v
        # Some structural operations (create/remove edges/vertices) might result in nout(v) being unchanged but nin of its outputs is changed
        nout(v) != nout_org(v) || any(vo -> nin(vo) != nin_org(vo), outputs(v))
    end
    isempty(vchanged) && return true

    vall = unique(mapreduce(v -> all_in_Δsize_graph(v, Output()), vcat, vchanged))
    Δoutputs(strategy.s, vall, rankfun)
end

default_neuronselect(v) = neuron_value_safe(trait(v), v)

neuron_value_safe(t::DecoratingTrait, v) = neuron_value_safe(base(t), v)
neuron_value_safe(::Immutable, v) = ones(nout(v))
neuron_value_safe(::MutationSizeTrait, v) = clean_values(cpu(neuron_value(v)),v)
clean_values(::Missing, v) = ones(nout_org(v))
# NaN should perhaps be < 0, but since SelectDirection is used, this might lead to inconsistent results as a subset of neurons for a vertex v whose output vertices are not part of the selection (typically because only v's inputs are touched) are selected. As the output vertices are not changed this will lead to a size inconsistency. Cleanest fix might be to separate "touch output" from "touch input" when formulating the output selection problem.
clean_values(a::AbstractArray{T}, v, repval=eps(T)) where T <: AbstractFloat = replace(a, NaN => repval, 0.0 => repval, Inf => repval, -Inf => repval)


"""
    PostMutation{T} <: DecoratingMutation{T}
    PostMutation(actions, m::AbstractMutation{T})
    PostMutation(m::AbstractMutation{T}, actions...)

Performs a set of actions after a wrapped `AbstractMutation` is applied.

Actions will be invoked with arguments (m::PostMutation{T}, e::T) where m is the enclosing `PostMutation` and `e` is the mutated entity of type `T`.
"""
struct PostMutation{T} <: DecoratingMutation{T}
    actions
    m::AbstractMutation{T}
end
PostMutation(m::AbstractMutation{T}, actions...) where T = PostMutation(actions, m)
PostMutation(action::Function, m::AbstractMutation{T}) where T = PostMutation(m, action)
function (m::PostMutation)(e)
    eout = m.m(e)
    foreach(a -> a(m, eout), m.actions)
    return eout
end


"""
    neuronselect(m, e)

Search `AbstractMutation` hieararchy `m` for `NeuronSelectMutations` and invoke `select` on them
"""
neuronselect(m, e) = neuronselect(m)
function neuronselect(m) end
neuronselect(m::T) where T <:AbstractMutation = foreach(fn -> neuronselect(getfield(m,fn)), fieldnames(T))
neuronselect(a::Union{AbstractVector, Tuple}) = foreach(ae -> neuronselect(ae), a)
neuronselect(m::NeuronSelectMutation) = select(m)

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


"""
    struct OptimizerMutation{F} <: AbstractMutation{FluxOptimizer}
    OptimizerMutation(optfun)
    OptimizerMutation(os::Union{Tuple, <:AbstractArray})

Mutatates optimizers not wrapped in `ShieldedOpt` through `optfun`.

Invoked recursively for `Flux.Optimiser`s.
"""
struct OptimizerMutation{F} <: AbstractMutation{FluxOptimizer}
    optfun::F
end
OptimizerMutation(os::Union{Tuple, <:AbstractArray}, rng=rng_default) = OptimizerMutation(o -> rand(rng, os)(learningrate(o)))

"""
    LearningRateMutation(rng=rng_default)

Return an `OptimizerMutation` which mutates the learning rate of optimizers.
"""
LearningRateMutation(rng=rng_default) = OptimizerMutation(o -> nudgelr(o, rng))

(m::OptimizerMutation)(opt::Flux.Optimiser) = Flux.Optimiser(m.(opt.os))
(m::OptimizerMutation)(o::ShieldedOpt) = o;
(m::OptimizerMutation)(o) = m.optfun(o)


nudgelr(o, rng=rng_default) = sameopt(o, nudgelr(learningrate(o), rng))
nudgelr(lr::Number, rng=rng_default) = clamp(lr + (rand(rng) - 0.5) * lr * 0.3, 1e-6, 1.0)

learningrate(o::Flux.Optimiser) = prod(learningrate.(o.os))
learningrate(o::ShieldedOpt) = learningrate(o.opt)
learningrate(o) = o.eta

newlr(o, lrf = nudgelr) = sameopt(o, lrf(learningrate(o)))
sameopt(o, lr) = @set o.eta = lr

"""
    AddOptimizerMutation{F} <: AbstractMutation{FluxOptimizer}

Adds optimizer generated by `optgen(os)` to the set of optimizers where `os` is the existing set.

An attempt to merge optimizers of the same type is made using `mergeopts`.
"""
struct AddOptimizerMutation{F} <: AbstractMutation{FluxOptimizer}
    optgen::F
end
(m::AddOptimizerMutation)(o::ShieldedOpt) = o;
(m::AddOptimizerMutation)(o) = m(Flux.Optimiser([o]))
function (m::AddOptimizerMutation)(opt::Flux.Optimiser)
    newopt = m.optgen(opt)
    return Flux.Optimiser(mergeopts(typeof(newopt), newopt, opt.os...))
end
