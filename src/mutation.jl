"""
    AbstractMutation{T}

Abstract type defining a mutation operation on entities of type `T`.

Implementations are expected to be callable using an entity of type `T` as only input.
"""
abstract type AbstractMutation{T} end

"""
    MutationProbability{T} <:AbstractMutation{T}
    MutationProbability(m::AbstractMutation{T}, p::Probability)
    MutationProbability(m::AbstractMutation{T}, p::Number)

Applies `m` with probability `p`.
"""
struct MutationProbability{T} <:AbstractMutation{T}
    m::AbstractMutation{T}
    p::Probability
end
MutationProbability(m::AbstractMutation{T}, p::Number) where T = MutationProbability(m, Probability(p))

function (m::MutationProbability{T})(e::T) where T
    apply(m.p) do
        m.m(e)
    end
end

"""
    WeightedMutationProbability{T,F} <: AbstractMutation{T}
    WeightedMutationProbability(m::AbstractMutation::T, pfun::F)

Applies `m` to an entity `e` with a probability `pfun(e)`.
"""
struct WeightedMutationProbability{T,F} <: AbstractMutation{T}
    m::AbstractMutation{T}
    pfun::F
end

function (m::WeightedMutationProbability{T})(e::T) where T
    apply(m.pfun(e)) do
        m.m(e)
    end
end

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

(m::AddVertexMutation)(v::AbstractVertex) = insert!(v, vi -> m.s(name(vi), vi, m.rng, outsize=nout(vi), wi=m.weightinit), m.outselect)

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

(m::RemoveVertexMutation)(v::AbstractVertex) = remove!(v, m.s)

"""
    AddEdgeMutation{F1, F2, R} <: AbstractMutation{AbstractVertex}
    AddEdgeMutation(p, rng=rng_default)
    AddEdgeMutation(p::Probability, rng=rng_default)
    AddEdgeMutation(mergefun::F, p::Probability, rng::R)

Add an edge from a vertex `vi` to another vertex `vo` randomly selected from `vs = filtfun(vi)`.

Higher values of `p` will give more preference to earlier vertices of `vs`.

If `vo` is not capable of having multiple inputs (determined by `singleinput(v) == true`), `vm = mergefun(voi)` where `voi` is a randomly selected input to `vo` will be used instead of `vo`.
"""
struct AddEdgeMutation{F1, F2, R} <: AbstractMutation{AbstractVertex}
    mergefun::F1
    filtfun::F2
    p::Probability
    rng::R
end
AddEdgeMutation(p, rng=rng_default) = AddEdgeMutation(Probability(p, rng), rng)
AddEdgeMutation(p::Probability, rng=rng_default) = AddEdgeMutation(default_mergefun(rng), no_shapechange, p, rng)

default_mergefun(rng=rng_default, pconc = 0.5; traitfun = MutationShield ∘ validated() ∘ default_logging(), layerfun = ActivationContribution) = function(vin)
    if rand(rng) > pconc
        return invariantvertex(layerfun(+), vin, traitdecoration=traitfun ∘ named(name(vin) * ".add"))
    end
    return concat(vin ,mutation=IoChange, traitfun = traitfun ∘ named(name(vin) * ".cat"), layerfun=layerfun)
end

function no_shapechange(vi, vc=vi, valid = [])
    if vi != vc
        if vi ∉ inputs(vc)
            valid = vcat(valid, vc)
        end
        layertype(vc) == typeof(globalpooling2d) && return valid
        if layertype(vc) == FluxNoParLayer()
            layer(vc) isa MaxPool && return valid
            layer(vc) isa MeanPool && return valid
        end
    end
    isempty(outputs(vc)) && return valid
    return mapfoldl(vco -> no_shapechange(vi, vco, valid), vcat, outputs(vc))
end

"""
    PostSelectOutputs <: AbstractAlignSizeStrategy
    PostSelectOutputs(;select=OutSelectExact(),align=IncreaseSmaller(), valuefun=v -> ones(nout_org(v)))

Post change alignment strategy which first aligns size using `align`, then select outputs (through `Δoutputs`) using `select` and `valuefun` if size alignment is successful.

Motivation is basically convenience when creating new edges between vertices.
"""
struct PostSelectOutputs <: AbstractAlignSizeStrategy
    selectstrategy::AbstractSelectionStrategy
    alignstrategy::AbstractAlignSizeStrategy
    valuefun::Function
    fallback::AbstractAlignSizeStrategy
end
PostSelectOutputs(;select=OutSelectExact(),align=PostAlignJuMP(), valuefun=v -> ones(nout_org(v)), fallback=FailAlignSizeRevert()) = PostSelectOutputs(select, align, valuefun, fallback)

NaiveNASlib.prealignsizes(s::PostSelectOutputs, vin, vout, will_rm) = NaiveNASlib.prealignsizes(s.alignstrategy, vin, vout, will_rm)
function NaiveNASlib.postalignsizes(s::PostSelectOutputs, vin, vout)
    if NaiveNASlib.postalignsizes(s.alignstrategy, vin, vout)
        vin_all = nout(vin) == nout_org(vin) ? AbstractVertex[] : all_in_Δsize_graph(vin, Output())
        vout_all = nin(vout) == nin_org(vout) ? AbstractVertex[] : all_in_Δsize_graph(vout, Input())

        verts = union(vin_all, vout_all)

        isempty(verts) && return true
        success = Δoutputs(s.selectstrategy, verts, s.valuefun)

        if !success
            return NaiveNASlib.postalignsizes(s.fallback, vin, vout)
        end
        return success
    end
    return false
end

function NaiveNASlib.remove_edge!(from::AbstractVertex, to::AbstractVertex; nr = 1, strategy = NaiveNASlib.default_remove_edge_strat(to))

    NaiveNASlib.prealignsizes(strategy, from, to, v -> false) || return

    in_inds = findall(vx -> vx == from, inputs(to))[nr]
    out_inds =findall(vx -> vx == to, outputs(from))[nr]
    deleteat!(inputs(to), in_inds)
    deleteat!(outputs(from), out_inds)

    NaiveNASlib.add_output!(op(to), trait(to), -nin(to)[in_inds])
    NaiveNASlib.rem_input!(op(to), in_inds...)

    NaiveNASlib.postalignsizes(strategy, from, to)
end
NaiveNASlib.postalignsizes(s::AbstractAlignSizeStrategy, vin, vout) = true


"""
    PostApplyMutation <: AbstractAlignSizeStrategy
    PostApplyMutation()
    PostApplyMutation(strategy::AbstractAlignSizeStrategy)

Post change alignment strategy which first aligns size using `strategy`, then invoke `apply_mutation` if size alignment is successful.

Motivation is basically convenience when creating new edges between vertices.
"""
struct PostApplyMutation <: AbstractAlignSizeStrategy
    strategy::AbstractAlignSizeStrategy
end
PostApplyMutation() = PostApplyMutation(PostSelectOutputs())
NaiveNASlib.prealignsizes(s::PostApplyMutation, vin, vout, will_rm) = begin
    @show NaiveNASlib.prealignsizes(s.strategy, vin, vout, will_rm)
end
function NaiveNASlib.postalignsizes(s::PostApplyMutation, vin, vout)
    if NaiveNASlib.postalignsizes(s.strategy, vin, vout)
        apply_mutation.(all_in_graph(vin))
        return true
    end
    return false
end

function (m::AddEdgeMutation)(vi::AbstractVertex)
    # All vertices for which it is allowed to add vi as an input
    allverts = filter(allow_mutation, m.filtfun(vi))
    isempty(allverts) && return

    # Higher probability to select a vertex close to v is desired behaviour
    # One line less than a for loop => FP wins!!
    selfun(::Nothing, vc) = apply(m.p) ? vc : nothing
    selfun(vs, vd) = vs
    vo = foldl(selfun, allverts, init=nothing)
    vo = vo == nothing ? rand(m.rng, allverts) : vo

    try_add_edge(vi, vo, m.mergefun, m.rng)
end

function try_add_edge(vi, vo, mergefun, rng=rng_default)

    # Need to add a vertex which can handle multiple inputs if vo is single input only
    # For cleaning up added vertex if the whole operation fails

    printsizes("start", vi, vo)

    cleanup_failed = () -> nothing
    if singleinput(vo)
        voi = inputs(vo)[1]
        if singleinput(voi)
            vm = mergefun(voi)
            # Insert vm between voi and vo, i.e voi -> vo turns into voi -> vm -> vo
            insert!(voi, vv -> vm, vs -> [vo])
            cleanup_failed = function()
                length(inputs(vm)) > 1 && return
                remove!(vm, RemoveStrategy(NoSizeChange()))
            end
            vo = vm # vm is the one we shall add an edge to
            @info "Create new vertex for merging $(name(vo))"
        else
            vo = voi
        end
    end
    # This is mainly because FailAlignSizeRevert does not work when the same vertex is input more than once, but it also seems kinda redundant.
    vi in inputs(vo) && return
    @info "Create edge between $(name(vi)) and $(name(vo))"

    printsizes("before edge", vi, vo)

    create_edge!(vi, vo, strategy = add_edge_strat(vo))
    printsizes("after edge", vi, vo)
    cleanup_failed()

    any(vx -> nout(vx) != nout_org(vx), all_in_graph(vi)) && error("Size apply error nout!!!")
    any(vx -> nin(vx) != nin_org(vx), all_in_graph(vi)) && error("Size apply error nin!!!")

    printsizes("done", vi, vo)
    validate_sizes(vo) || error("Validation failed after!!")
end
# Need to override this one for strange types which e.g. layers which support exactly 2 inputs or something.
singleinput(v) = length(inputs(v)) == 1

function printsizes(pref, vi, vo)
    println("$pref vi  nout=$(nout(vi)) nin=$(nin(vi)), nout_org=$(nout_org(vi)), nin_org=$(nin_org(vi))")
    println("$pref vo  nout=$(nout(vo)) nin=$(nin(vo)), nout_org=$(nout_org(vo)), nin_org=$(nin_org(vo))")
    println("$pref voi nout=$(nout.(inputs(vo))), nout_org=$(nout_org.(inputs(vo)))")
    println("$pref op  $(op(vo).outΔ), $(op(vo).inΔ)")
end

add_edge_strat(v::AbstractVertex) = add_edge_strat(trait(v))
add_edge_strat(d::DecoratingTrait) = add_edge_strat(base(d))
function add_edge_strat(::SizeInvariant)
    alignstrat = IncreaseSmaller(DecreaseBigger(AlignSizeBoth(FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))!"))))

    selectstrat = OutSelect{Exact}(LogSelectionFallback("Reverting...", NoutRevert()))

    okstrat = PostApplyMutation(SelectOutputs(selectstrat, alignstrat, default_neuronselect))

    # Tricky failure case: It is possible that CheckCreateEdgeNoSizeCycle does not detect any size cycle until after the edge has been created. When this happens, we run PostSelectOutputs to revert all size changes before removing the edge. The latter might not be strictly needed (I really don't know actually), but it does stay true to the contract of "no size change if operation does not succeed".
    nokstrat = FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))! Size cycle detected! Reverting...", andthen=PostSelectOutputs(select=NoutRevert(), align=NoSizeChange(), fallback=FailAlignSizeRevert())) # NoutRevert returns success=false, meaning that fallback will be invoked

    return CheckCreateEdgeNoSizeCycle(okstrat, nokstrat)
end
function add_edge_strat(::SizeStack)

    alignstrat = PostAlignJuMP(DefaultJuMPΔSizeStrategy(), FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))!"))

    selectstrat = OutSelect{Exact}(LogSelectionFallback("Reverting...", NoutRevert()))

    okstrat = PostApplyMutation(PostSelectOutputs(selectstrat, alignstrat, default_neuronselect, FailAlignSizeRevert()))

    nokstrat = FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))! Size cycle detected!")
    return CheckCreateEdgeNoSizeCycle(okstrat, nokstrat)
end

validate_sizes(v) = validate_sizes(trait(v), v)
validate_sizes(t::DecoratingTrait, v) = validate_sizes(base(t), v)
validate_sizes(::SizeInvariant, v) = unique(nin(v)) == [nout(v)] && unique(nin_org(v)) == [nout_org(v)]
validate_sizes(::SizeStack, v) = sum(nin(v)) == nout(v) && sum(nin_org(v)) == nout_org(v)
validate_sizes(::MutationTrait, v) = true

"""
    KernelSizeMutation{N} <: AbstractMutation{AbstractVertex}
    KernelSizeMutation(Δsizespace::AbstractParSpace{N, Int}; maxsize, pad, rng)
    KernelSizeMutation2D(absΔ::Integer;maxsize, pad, rng)
    KernelSizeMutation(absΔ::Integer...;maxsize, pad, rng)

Mutate the size of filter kernels of convolutional layers.

Note: High likelyhood of large accuracy degradation after applying this mutation.
"""
struct KernelSizeMutation{N,F} <: AbstractMutation{AbstractVertex}
    Δsizespace::AbstractParSpace{N, Int}
    maxsize::F
    padspace::AbstractPadSpace
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
    pad = m.padspace(currsize .+ Δsize, dilation(l))
    mutate_weights(v, KernelSizeAligned(Δsize, pad))
end
dilation(l) = l.dilation

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

(m::ActivationFunctionMutation)(v::AbstractVertex) = m(layertype(v), v)
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
clean_values(a::AbstractArray, v) = replace(a, NaN => 0.0001, 0.0 => 0.0001, Inf => 0.0001, -Inf => 0.0001)


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
