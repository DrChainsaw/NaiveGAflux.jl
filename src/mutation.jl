"""
    AbstractMutation{T}

Abstract type defining a mutation operation on entities of type `T`.

Implementations are expected to be callable using an entity of type `T` as only input.

May also implement a callable accepting a `AbstractVector{<:T}` if it is useful to work on
all items to mutate at once.
"""
abstract type AbstractMutation{T} end

(m::AbstractMutation{T})(es::AbstractVector{<:T}) where T = m.(es)

"""
    AbstractCrossover{T}

Type alias for `AbstractMutation{Tuple{T,T}}` defining a crossover of two entities of type `T`.

Implementations are expected to be callable using a tuple of two type `T` as only input.
"""
const AbstractCrossover{T} = AbstractMutation{Tuple{T,T}}

"""
    DecoratingMutation{T}

Abstract type indicating that the type itself does not perform any mutation but wraps a type which might do.

Must either implement callable method for `AbstractVector{<:T}` or accept keyword arguments `next=wrapped(m)` and
`noop=identity` along with a single `T`.
"""
abstract type DecoratingMutation{T} <: AbstractMutation{T} end
wrapped(m::DecoratingMutation) = m.m

mutationleaves(m::DecoratingMutation) = (mutationleaves(wrapped(m))...,)
mutationleaves(tm::Tuple) = mapreduce(mutationleaves, (t1,t2) -> (t1...,t2...), tm)
mutationleaves(m) = tuple(m)

# Apart from being overengineered this helps protecting against forgetting to handle arrays in a DecoratingMutation
# The next/noop happens to work with most existing DecoratingMutations, but it is a bit arbitrary and in some cases 
# one must implement both the per-element and the vector of elements versions.
function (m::DecoratingMutation{T})(es::AbstractVector{<:T}) where T 
    cnt = Ref(1)
    fornext = Int[]
    next = function(e) 
        push!(fornext, cnt[])
        cnt[] += 1
        e
    end
    noop = function(e)
        cnt[] += 1
        e
    end

    allres = m.(es; next, noop)
    mres = wrapped(m)(es[fornext])

    # Mutation might accidentally widen the type compared to allres and then we can't insert mres into allres.
    # Lets fix that if it happens
    RT = typejoin(eltype(allres), eltype(mres))
    res = RT === eltype(allres) ? allres :  convert(Vector{RT}, allres)

    res[fornext] = mres
    return res
end

"""
    MutationProbability{T} <: DecoratingMutation{T}
    MutationProbability(m::AbstractMutation{T}, p::Probability)
    MutationProbability(m::AbstractMutation{T}, p::Number)

Applies `m` with probability `p`.
"""
struct MutationProbability{T, P<:Probability} <: DecoratingMutation{T}
    m::AbstractMutation{T}
    p::P
end
MutationProbability(m::AbstractMutation{T}, p::Number) where T = MutationProbability(m, Probability(p))
(m::MutationProbability{T})(e::T; next=m.m, noop=identity) where T = apply(() -> next(e), m.p, () -> noop(e))

"""
    WeightedMutationProbability{T,F} <: DecoratingMutation{T}
    WeightedMutationProbability(m::AbstractMutation::T, pfun::F)

Applies `m` to an entity `e` with a probability `pfun(e)`.
"""
struct WeightedMutationProbability{T,F} <: DecoratingMutation{T}
    m::AbstractMutation{T}
    pfun::F
end
(m::WeightedMutationProbability{T})(e::T; next=m.m, noop=identity) where T = apply(() -> next(e), m.pfun(e), () -> noop(e))

"""
    HighValueMutationProbability(m::AbstractMutation{T}, pbase::Real, rng=rng_default; spread=0.5)

Return a `WeightedMutationProbability` which applies `m` to vertices with an (approximately) average probability of `pbase` and where high `neuronutility` compared to other vertices in same graph means higher probability.

Parameter `spread` can be used to control how much the difference in probability is between high and low values. High spread means high difference while low spread means low difference.
"""
HighValueMutationProbability(m::AbstractMutation{T}, pbase::Real, rng=rng_default;spread=0.5) where T <: AbstractVertex = WeightedMutationProbability(m, weighted_neuronutility_high(pbase, rng,spread=spread))

"""
    LowValueMutationProbability(m::AbstractMutation{T}, pbase::Real, rng=rng_default; spread=2)

Return a `WeightedMutationProbability` which applies `m` to vertices with an (approximately) average probability of `pbase` and where low `neuronutility` compared to other vertices in same graph means higher probability.

Parameter `spread` can be used to control how much the difference in probability is between high and low values. High spread means high difference while low spread means low difference.
"""
LowValueMutationProbability(m::AbstractMutation{T}, pbase::Real, rng=rng_default;spread=2) where T <: AbstractVertex = WeightedMutationProbability(m, weighted_neuronutility_low(pbase, rng, spread=spread))


weighted_neuronutility_high(pbase, rng=rng_default; spread=0.5) = function(v::AbstractVertex)
    ismissing(NaiveNASflux.neuronutility(v)) && return pbase
    return Probability(fixnan(pbase ^ normexp(v, spread), pbase), rng)
end

weighted_neuronutility_low(pbase, rng=rng_default;spread=2) = function(v::AbstractVertex)
    ismissing(NaiveNASflux.neuronutility(v)) && return pbase
    return Probability(fixnan(pbase ^ (1/normexp(v, 1/spread)), pbase), rng)
end

fixnan(x, rep) = isnan(x) ? rep : clamp(x, 0.0, 1.0)

# This is pretty hacky and arbitrary. Change to something better
function normexp(v::AbstractVertex, s)
    allvertices = filter(allow_mutation, all_in_graph(v))
    allvalues = map(vi -> NaiveNASflux.neuronutility(vi), allvertices)
    meanvalues = map(mean, skipmissing(allvalues))
    meanvalue = mean(meanvalues)
    maxvalue = maximum(meanvalues)
    value = mean(NaiveNASflux.neuronutility(v))
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
# Identical, but can't use Union due to ambiguity
(m::MutationChain{T})(es::AbstractVector{<:T}) where T = foldl((ei, mi) -> mi(ei), m.m; init=es)
(m::MutationChain{T})(e::T) where T = foldl((ei, mi) -> mi(ei), m.m; init=e)

"""
    RecordMutation{T} <: DecoratingMutation{T}
    RecordMutation(m::AbstractMutation{T})

Records all mutated entities.

Intended use case is to be able to do parameter selection on mutated vertices.
"""
struct RecordMutation{T} <: DecoratingMutation{T}
    m::AbstractMutation{T}
    mutated::Vector{T}
end
RecordMutation(m::AbstractMutation{T}) where T = RecordMutation(m, T[])
function (m::RecordMutation{T})(e::T; next=m.m, noop=identity) where T
    em = next(e)
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
struct LogMutation{T,F,L<:LogLevel,LF} <: DecoratingMutation{T}
    strfun::F
    level::L
    nextlogfun::LF
    m::AbstractMutation{T}
end
LogMutation(strfun, m::AbstractMutation{T}; level = Logging.Info, nextlogfun=e -> PrefixLogger("   ")) where T = LogMutation(strfun, level, nextlogfun, m)
function (m::LogMutation{T})(e::T; next=m.m, noop=identity) where T
    @logmsg m.level m.strfun(e)
    return with_logger(() -> next(e), m.nextlogfun(e))
end

"""
    MutationFilter{T} <: DecoratingMutation{T}
    MutationFilter(predicate, m)

Applies mutation `m` only for entities `e` for which `predicate(e)` returns true.
"""
struct MutationFilter{T,P} <: DecoratingMutation{T}
    predicate::P
    m::AbstractMutation{T}
end
function (m::MutationFilter{T})(e::T; next=m.m, noop=identity) where T
    m.predicate(e) && return next(e)
    return noop(e)
end


"""
    VertexMutation <: DecoratingMutation{CompGraph}
    VertexMutation(m::AbstractMutation{AbstractVertex}, s::AbstractVertexSelection)
    VertexMutation(m::AbstractMutation{AbstractVertex})

Applies a wrapped `AbstractMutation{AbstractVertex}` to each selected vertex in a `CompGraph`.

Vertices to select is determined by the configured `AbstractVertexSelection`.
"""
struct VertexMutation{S<:AbstractVertexSelection} <: DecoratingMutation{CompGraph}
    m::AbstractMutation{AbstractVertex}
    s::S
end
VertexMutation(m::AbstractMutation{AbstractVertex}) = VertexMutation(m, FilterMutationAllowed())
function (m::VertexMutation)(g::CompGraph)
    m.m(select(m.s, g, m))
    return g
end

"""
    NoutMutation <:AbstractMutation{AbstractVertex}
    NoutMutation(l1::Real,l2::Real, rng::AbstractRNG)
    NoutMutation(limit, rng::AbstractRNG=rng_default)
    NoutMutation(l1,l2)

Mutate the out size of a vertex or vector of vertices.

Size is changed by `x * nout(v)` rounded away from from zero where `x` is drawn from `U(minrel, maxrel)` where 
`minrel` and `maxrel` are `l1` and `l2` if `l1 < l2` and `l2` and `l1` otherwise.
"""
struct NoutMutation{R<:Real, RNG<:AbstractRNG} <:AbstractMutation{AbstractVertex}
    minrel::R
    maxrel::R
    rng::RNG
    function NoutMutation(l1::R1, l2::R2, rng::RNG) where {R1, R2, RNG} 
        R = promote_type(R1, R2)
        return l1 < l2 ? new{R, RNG}(promote(l1, l2)..., rng) : new{R, RNG}(promote(l2, l1)..., rng)
    end
end
NoutMutation(limit, rng::AbstractRNG=rng_default) = NoutMutation(0, limit, rng)
NoutMutation(l1,l2) = NoutMutation(l1,l2, rng_default)
(m::NoutMutation)(v::AbstractVertex) = first(m([v]))
function (m::NoutMutation)(vs::AbstractVector{<:AbstractVertex})

    Δs = Dict{AbstractVertex, Int}()
    shift = m.minrel
    scale = m.maxrel - m.minrel

    for v in vs
        terminputs = findterminating(v, inputs)

        # We are basically just searching for Immutable vertices here, allow_mutation(trait(v)) happens to do just that
        any(tv -> allow_mutation(trait(tv)), terminputs) || continue
        
        Δfloat = rand(m.rng) * scale + shift

        Δ = ceil(Int, abs(Δfloat)) *  sign(Δfloat)
        minsize = minimum(nout.(terminputs))
        # Or else we might increase the size despite Δ being negative which would be surprising to a user who has specified 
        # strictly negative size changes
        minsize + Δ <= 0 && continue

        Δs[v] = Δ
    end

    if !isempty(Δs)
        failmsg = (args...) -> "Could not change nout of $(join(NaiveNASlib.nameorrepr.(keys(Δs)), ", ", " and ")) by $(join(values(Δs), ", ", " and ")). No change!"

        strategy = TimeOutAction(;base=ΔNoutRelaxed(Δs), fallback=LogΔSizeExec(failmsg, Logging.Warn, ΔSizeFailNoOp()))

        Δsize!(strategy)
    end
    return vs
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
struct AddVertexMutation{S<:AbstractArchSpace, F, WI<:AbstractWeightInit, RNG<:AbstractRNG} <:AbstractMutation{AbstractVertex}
    s::S
    outselect::F
    weightinit::WI
    rng::RNG
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
struct RemoveVertexMutation{S<:RemoveStrategy} <:AbstractMutation{AbstractVertex}
    s::S
end
function RemoveVertexMutation() 
    alignstrat = IncreaseSmaller(fallback=DecreaseBigger(fallback=AlignSizeBoth(fallback=FailAlignSizeWarn())))
    return RemoveVertexMutation(RemoveStrategy(CheckAligned(CheckNoSizeCycle(alignstrat, FailAlignSizeWarn(msgfun = (vin,vout) -> "Can not remove vertex $(name(vin))! Size cycle detected!")))))
end

function (m::RemoveVertexMutation)(v::AbstractVertex)
    remove!(v, m.s)
    return v
end

default_neuronselect(args...) = NaiveNASlib.defaultutility(args...)

"""
    AddEdgeMutation <: AbstractMutation{AbstractVertex}
    AddEdgeMutation(p; rng=rng_default, mergefun=default_mergefun(rng=rng), filtfun=no_shapechange, valuefun=default_neuronselect)
    AddEdgeMutation(p::Probability; rng=rng_default, mergefun=default_mergefun(rng=rng), filtfun=no_shapechange, valuefun=default_neuronselect)

Add an edge from a vertex `vi` to another vertex `vo` randomly selected from `vs = filtfun(vi)`.

Higher values of `p` will give more preference to earlier vertices of `vs`.

If `vo` is not capable of having multiple inputs (determined by `singleinput(v) == true`), `vm = mergefun(voi)` where `voi` is a randomly selected input to `vo` will be used instead of `vo` and `vo` will be added as the output of `vm`.

When selecting neurons/outputs after any eventual size change the values `valuefun(v)` will be used to determine the value of each output in vertex `v`. Note that `length(valuefun(v)) == nout(v)` must hold.

Note: High likelyhood of large accuracy degradation after applying this mutation.
"""
struct AddEdgeMutation{F1, F2, F3, P<:Probability, RNG} <: AbstractMutation{AbstractVertex}
    mergefun::F1
    filtfun::F2
    valuefun::F3
    p::P
    rng::RNG
end
AddEdgeMutation(p; rng=rng_default, mergefun=default_mergefun(rng=rng), filtfun=no_shapechange, valuefun=default_neuronselect) = AddEdgeMutation(Probability(p, rng), rng=rng, mergefun=mergefun, filtfun=filtfun, valuefun=valuefun)
AddEdgeMutation(p::Probability; rng=rng_default, mergefun=default_mergefun(rng=rng), filtfun=no_shapechange, valuefun=default_neuronselect) = AddEdgeMutation(mergefun, filtfun, valuefun, p, rng)

default_mergefun(pconc = 0.5; rng=rng_default, traitfun = MutationShield ∘ RemoveIfSingleInput ∘ validated() ∘ default_logging(), layerfun = ActivationContribution) = function(vin)
    if rand(rng) > pconc
        return invariantvertex(layerfun(+), vin, traitdecoration=traitfun ∘ named(name(vin) * ".add"))
    end
    return concat(vin, traitfun = traitfun ∘ named(name(vin) * ".cat"), layerfun=layerfun)
end

function no_shapechange(vi)
    # all_in_graph is not sorted, and we want some kind of topoligical order here so that earlier indices are closer to vi
    allsorted = mapreduce(ancestors, vcat, filter(v -> isempty(outputs(v)), all_in_graph(vi))) |> unique
    
    # Vertices which have the same input as vi and are singleinput
    #   Reason is that this will cause a new vertex to be added between the target output vertex vo
    #   and the input vertex to vi (vii) and this is detected as a size cycle which causes 
    #   try_add_edge to fail.
    inouts = filter(singleinput, mapreduce(outputs, vcat, inputs(vi); init=[]))
    # All vertices which are after vi in the topology
    vsafter = setdiff(allsorted, ancestors(vi), outputs(vi), inouts)
    
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
        if singleinput(voi) || voi in ancestors(vi)
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
    warnfailalign = FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))!")
    alignstrat = AlignSizeBoth(;mapstrat=WithValueFun(valuefun), fallback = warnfailalign)
    # Tricky failure case: It is possible that CheckCreateEdgeNoSizeCycle does not detect any size cycle until after the edge has been created?
    sizecyclewarn = FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))! Size cycle detected!") 

    return CheckCreateEdgeNoSizeCycle(ifok=alignstrat, ifnok=sizecyclewarn)
end
function create_edge_strat(::SizeStack, valuefun)
    warnfailalign = FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))!")
    alignstrat = PostAlign(TruncateInIndsToValid(WithValueFun(valuefun, AlignNinToNout(;fallback=ΔSizeFailNoOp()))), fallback=warnfailalign)

    sizecyclewarn = FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))! Size cycle detected!")
    return CheckCreateEdgeNoSizeCycle(ifok=alignstrat, ifnok=sizecyclewarn)
end

"""
    RemoveEdgeMutation <: AbstractMutation{AbstractVertex}
    RemoveEdgeMutation(;valuefun=default_neuronselect, rng=rng_default)

Remove an edge from a vertex `vi` to another vertex `vo` randomly selected from `outputs(vi)`.

Vertex `vi` must have more than one output and vertex `vo` must have more than one output for the edge to be removed. Otherwise no change is made.

If there are multiple edges between `vi` and `vo` no change will be made due to NaiveNASlib not being able to revert a failed operation in this case..

When selecting neurons/outputs after any eventual size change the values `valuefun(v)` will be used to determine the value of each output in vertex `v`. Note that `length(valuefun(v)) == nout(v)` must hold.

Note: High likelyhood of large accuracy degradation after applying this mutation.
"""
struct RemoveEdgeMutation{F, RNG<:AbstractRNG} <: AbstractMutation{AbstractVertex}
    valuefun::F
    rng::RNG
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
    KernelSizeAligned(Δsize, pad)(v)
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
struct ActivationFunctionMutation{T,RNG} <: AbstractMutation{AbstractVertex} where {T <: AbstractParSpace{1}, R <: AbstractRNG}
    actspace::T
    rng::RNG
end
ActivationFunctionMutation(acts...;rng=rng_default) = ActivationFunctionMutation(collect(acts), rng=rng)
ActivationFunctionMutation(acts::AbstractVector;rng=rng_default) = ActivationFunctionMutation(ParSpace(acts), rng)

function (m::ActivationFunctionMutation)(v::AbstractVertex)
    m(layertype(v), v)
    return v
end
function (m::ActivationFunctionMutation)(t, v) end
(m::ActivationFunctionMutation)(::Union{FluxDense, FluxConvolutional}, v) = NaiveNASflux.setlayer!(v, (σ = m.actspace(m.rng),))
(m::ActivationFunctionMutation)(::FluxParNorm, v) = NaiveNASflux.setlayer!(v, (λ = m.actspace(m.rng),))
function (m::ActivationFunctionMutation)(::FluxRnn, v)
    newcell = setproperties(layer(v).cell, (σ = m.actspace(m.rng),))
    NaiveNASflux.setlayer!(v, (cell = newcell,))
end


"""
    PostMutation{T} <: DecoratingMutation{T}
    PostMutation(actions, m::AbstractMutation{T})
    PostMutation(m::AbstractMutation{T}, actions...)

Performs a set of actions after a wrapped `AbstractMutation` is applied.

Actions will be invoked with arguments (m::PostMutation{T}, e::T) where m is the enclosing `PostMutation` and `e` is the mutated entity of type `T`.
"""
struct PostMutation{T,A} <: DecoratingMutation{T}
    actions::A
    m::AbstractMutation{T}
end
PostMutation(m::AbstractMutation{T}, actions...) where T = PostMutation(actions, m)
PostMutation(action::Function, m::AbstractMutation{T}) where T = PostMutation(m, action)
function (m::PostMutation{T})(e::T; next=m.m, noop=identity) where T
    eout = next(e)
    foreach(a -> a(m, eout), m.actions)
    return eout
end

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
(m::OptimizerMutation)(o::FluxOptimizer) = m.optfun(o)


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
(m::AddOptimizerMutation)(o::FluxOptimizer) = m(Flux.Optimiser([o]))
function (m::AddOptimizerMutation)(opt::Flux.Optimiser)
    newopt = m.optgen(opt)
    return Flux.Optimiser(mergeopts(typeof(newopt), newopt, opt.os...))
end
