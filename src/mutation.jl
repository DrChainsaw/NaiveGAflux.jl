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
RemoveVertexMutation() = RemoveVertexMutation(RemoveStrategy(IncreaseSmaller(DecreaseBigger(AlignSizeBoth(FailAlignSizeWarn())))))

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

struct Nout end
struct RemoveVertex end

"""
    select(m::NeuronSelectMutation)

Select neurons for each `AbstractVertex` mutated by `m`.
"""
function select(m::NeuronSelectMutation)
    for v in m.m.mutated
        select_neurons(m.strategy, v, m.rankfun)
    end
end

default_neuronselect(v) = selectvalidouts(v, neuron_value)

select_neurons(::T, v::AbstractVertex, rankfun::Function) where T = error("Neuron select not implemented for $T")
function select_neurons(::Nout, v::AbstractVertex, rankfun::Function, s=NaiveNASlib.VisitState{Vector{Int}}(v))
    # nout_org will fail if op(v) is not IoChange
    # This package is kinda hardcoded to use IoChange, and with some polishing of NaiveNASlib it should be the only possible option
    Δout = nout(v) - nout_org(op(v))

    # Note: Case of Δnout > 0 (size increase) not implemented yet
    if Δout != 0
        inds= rankfun(v)
        Δnout(v, inds, s=s)
    end
end

function select_neurons(::RemoveVertex, v::AbstractVertex, rankfun::Function)
    # This could be messy due to many different cases:
    # 1: Outsize of input vertex is increased
    # 2: Insize of output vertex is increased
    # 3: Outsize of input vertex is decreased
    # 4: Insize of output vertex is decreased
    # as well as all combinations of two of the above.

    # However, the only cases which needs special action are those when 3 has happened and they all warrant the same action: Select neurons from input vertex but don't touch the output vertex

    # When 4 has happened it might be useful to try to select the best input neurons
    # Can one assume that "best output neurons" from input vertex are "best input neurons" even if those "best output neurons" belong to the removed vertex?

    Δins = nin(v) - nin_org(op(v))
    for i in eachindex(Δins)
        Δins[i] >= 0 && continue
        vin = inputs(v)[i]

        s = NaiveNASlib.VisitState{Vector{Int}}(vin)
        NaiveNASlib.visited_in!.(s, outputs(vin))

        select_neurons(Nout(), vin, rankfun, s)
    end
end

# TODO: This method and validouts shall go to NaiveNASlib once they have proven themselves some more

function selectvalidouts(v::AbstractVertex, scorefun::Function)
    score = scorefun(v)
    valouts = NaiveGAflux.validouts(v)

    selected = nothing
    for (vi, mi) in valouts
        Δout = nout(vi) - size(mi, 1)
        # Case 1: Size has decreased, we need to select a subset of the outputs based on the score
        if Δout < 0
            # Step 1: The constraint is that we must pick all values in a row of mi for neuron selection to be consistent. Matrix mi has more than one column if activations from vi is input (possibly through an arbitrary number of size transparent layers such as BatchNorm) to v more than once.
            bestrows = sortperm(vec(sum(score[mi], dims=2)), lt = >)
             # Step 2: Due to negative values being used to indicate insertion of neurons, we must do this:
             # Add each column from mi to selected as a separate array (selected is an array of arrays)
             # Each columns array in selected individually sorted here to keep unnecessary shuffling of neurons to a minimum.
            toadd = [sort(vec(mi[bestrows[1:nout(vi)], i])) for i in 1:size(mi, 2)]
            isnothing(selected) ? selected = toadd : append!(selected, toadd)
        # Case 2: Size has not decreased,
        # We want to insert -1s to show where new columns are to be added
        else
            # This is the reason for selected being an array of arrays: Activations from multiple repeated inputs might be interleaved and we must retain the original neuron ordering without moving the negative values around.
            toadd =  [vec(vcat(mi[:,i], -ones(Int,Δout, 1))) for i in 1:size(mi, 2)]
            isnothing(selected) ? selected = toadd : append!(selected, toadd)
        end
     end

     # Now concatenate all column arrays in selected. Sort them by their first element which due to the above is always the smallest. Note that due to how validouts works, we can always assume that all nonnegative numbers in a column array in selected are either strictly greater than or strictly less than all nonnegative numbers of all other column arrays.
     return foldl(vcat, sort(selected, by = v -> v[1]))
end

validouts(v::AbstractVertex, offs=0, dd=Dict()) = validouts(trait(v), v, offs, dd)
validouts(t::DecoratingTrait, v, offs, dd) = validouts(base(t), v, offs, dd)
function validouts(::SizeStack, v, offs, dd)
    for vin in inputs(v)
        validouts(vin, offs, dd)
        offs += nout_org(op(vin))
    end
    return dd
end
function validouts(::SizeInvariant, v, offs, dd)
    foreach(vin -> validouts(vin, offs, dd), inputs(v))
    return dd
end
function validouts(::SizeAbsorb, v, offs, dd)
    orgsize = nout_org(op(v))
    selectfrom = hcat(get(() -> zeros(Int, orgsize, 0), dd, v), (1:orgsize) .+ offs)
    dd[v] = selectfrom
    return dd
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
