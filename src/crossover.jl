"""
    VertexCrossover{S, PF, CF}
    VertexCrossover(crossover ;selection=FilterMutationAllowed(), pairgen=default_pairgen)
    VertexCrossover(crossover, deviation::Number; selection=FilterMutationAllowed())

Applies `crossover` to each pair of selected vertices from two `CompGraph`s.

Vertices to select from the first graph is determined by `selection` (default [`FilterMutationAllowed`](@Ref) while `pairgen` (default [`default_pairgen`](@Ref)) determines how to pair the selected vertices with vertices in the second graph.

The default pairing function will try to pair vertices which have similar relative topologial order within their graphs. For instance, if the first graph has 5 vertices and the second has 10, it will pair vertex 2 from the first graph with vertex 4 from the second (assuming they are of compatible type). The parameter `deviation` can be used to inject noise in this process so that the pairing will randomly deviate where the magnitude of `deviation` sets how much and how often.

See also [`crossover`](@Ref).
"""
struct VertexCrossover{S, PF, CF} <: AbstractCrossover{CompGraph}
    selection::S
    pairgen::PF
    crossover::CF
end
VertexCrossover(crossover ;selection=FilterMutationAllowed(), pairgen=default_pairgen) = VertexCrossover(selection, pairgen, crossover)
VertexCrossover(crossover, deviation::Number; selection=FilterMutationAllowed()) = VertexCrossover(selection, (vs1,vs2) -> default_pairgen(vs1, vs2, deviation), crossover)

(c::VertexCrossover)((g1,g2)::Tuple) = crossover(g1, g2; selection=c.selection, pairgen=c.pairgen, crossoverfun=c.crossover)


"""
    CrossoverSwap{F, S} <: AbstractCrossover{<:AbstractVertex}
    CrossoverSwap(pairgen, mergefun, selection)
    CrossoverSwap(;pairgen=default_pairgen, mergefun=default_mergefun, selection=FilterMutationAllowed())
    CrossoverSwap(deviation::Number; mergefun=default_mergefun, selection=FilterMutationAllowed())

Swap out a part of one graph with a part of another graph, making sure that the graphs do not become connected in the process.

More concretely, swaps a set of consecutive vertices `vs1` set of consecutive vertices `vs2` returning the swapped `v1` and `v2` respectively if successful or `v1` and `v2` if not.

The last vertex in `vs1` is `v1` and the last output of `vs2` is `v2`. The other members of `vs1` and `vs2` are determined by `pairgen` (default [`default_pairgen`](@Ref)) and `selection` (default [`FilterMutationAllowed`](@Ref)).

If a vertex `v` is not capable of having multiple inputs (determined by `singleinput(v) == true`), `vm = mergefun(vi)` where `vi` is the input to `v` will be used instead of `v` and `v` will be added as the output of `vm` if necessary.

See also [`crossoverswap`](@Ref)

Note: High likelyhood of large accuracy degradation after applying this mutation.
"""
struct CrossoverSwap{F1, F2, S} <: AbstractCrossover{AbstractVertex}
    pairgen::F1
    mergefun::F2
    selection::S
end
CrossoverSwap(;pairgen=default_pairgen, mergefun=default_mergefun, selection=FilterMutationAllowed()) = CrossoverSwap(pairgen, mergefun, selection)
CrossoverSwap(deviation::Number; mergefun=default_mergefun, selection=FilterMutationAllowed()) = CrossoverSwap((vs1,vs2) -> default_pairgen(vs1, vs2, deviation), mergefun, selection)

(c::CrossoverSwap)((v1,v2)::Tuple) = crossoverswap(v1, v2; pairgen = c.pairgen, mergefun=c.mergefun, selection=c.selection)


"""
    crossover(g1::CompGraph, g2::CompGraph; selection=FilterMutationAllowed(), pairgen=default_pairgen, crossoverfun=crossoverswap)

Perform crossover between `g1` and `g2` and return the two children `g1'` and `g2'`.

How the crossover is performed depends on `pairgen` and `crossoverfun` as well as `selection`.

`selection` is used to filter out at which vertices the crossoverpoints may be.

`pairgen` return indices of the potential crossover points to use out of the allowed crossover points. New crossover points will be drawn from pairgen until it returns `nothing`.

`crossoverfun` return the result of the crossover which may be same as inputs depending on implementation.
"""
function crossover(g1::CompGraph, g2::CompGraph; selection=FilterMutationAllowed(), pairgen=default_pairgen, crossoverfun=crossoverswap)

    sel(g) = select(selection, g)

    ninputs1 = length(g1.inputs)
    ninputs2 = length(g2.inputs)
    noutputs1 = length(g1.outputs)
    noutputs2 = length(g2.outputs)

    # pairgen api is very close to the iterator specification. Not sure if things would be easier if it was an iterator instead...
    vs1, vs2 = sel(g1), sel(g2)
    inds = pairgen(vs1, vs2, ind1 = 1)
    while !isnothing(inds)
        ind1, ind2 = inds

        v1, v2 = crossoverfun((vs1[ind1], vs2[ind2]))

        g1 = regraph(v1, ninputs1, noutputs1)
        g2 = regraph(v2, ninputs2, noutputs2)
        vs1, vs2 = sel(g1), sel(g2)

        # Graphs may be different now, so we need to reselect
        inds = pairgen(vs1, vs2, ind1 = ind1+1)
    end
    return g1, g2
end

regraph(v::AbstractVertex) = regraph(all_in_graph(v))
function regraph(vs)
    ins = filter(v -> isempty(inputs(v)), vs)
    outs = filter(v -> isempty(outputs(v)), vs)
    return CompGraph(ins, outs)
end
function regraph(x, ninputs, noutputs)
    g = regraph(x)
    @assert ninputs == length(g.inputs) "Incorrect number of inputs! Expected $ninputs but got $(length(g.inputs))"
    @assert noutputs == length(g.outputs) "Incorrect number of inputs! Expected $noutputs but got $(length(g.outputs))"
    return g
end

"""
    default_pairgen(vs1, vs2, deviation = 0.0; rng=rng_default, compatiblefun = sameactdims, ind1 = rand(rng, eachindex(vs1)))

Return integers `ind1` and `ind2` so that `vs1[ind1]` and `vs2[ind2]` are a suitable pair for crossover.

Input function `compatiblefun(v1,v2)` shall return `true` if `v1` and `v2` can be swapped and is used to determine the set of vertices to select from given.

From the set of compatible vertices in `vs2`, the one which has the smallest relative topologial distance from `vs1[ind2]` is selected. The parameter `devation` can be used to randomly deviate from this where larger magnitude means more deviation.
"""
function default_pairgen(vs1, vs2, deviation = 0.0; rng=rng_default, compatiblefun = sameactdims, ind1 = rand(rng, eachindex(vs1)))
    ind1 > length(vs1) && return nothing

    # Make sure the returned indices map to vertices which are compatible so that output from a convolutional layer does not suddenly end up being input to a dense layer.
    candidate_ind2s = filter(i2 -> compatiblefun(vs1[ind1], vs2[i2]), eachindex(vs2))

    isempty(candidate_ind2s) && return nothing

    order1 = relative_positions(vs1) .+ deviation .* randn(rng, length(vs1))
    order2 = relative_positions(vs2) .+ deviation .* randn(rng, length(vs2))

    ind2 = candidate_ind2s[argmin(abs.(order2[candidate_ind2s] .- order1[ind1]))]
    return ind1, ind2
end

# We also check all size terminating outputs (perhaps sufficient with one) since transparent layers (incorrectly in this case) just report their inputs dims, meaning that a flattening global pool will seem to have conv output dims
sameactdims(v1, v2) =  _sameactdims(v1, v2) && all(Iterators.product(findterminating(v1, outputs), findterminating(v2, outputs))) do (v1o, v2o)
    _sameactdims(v1o, v2o)
end
_sameactdims(v1, v2) = NaiveNASflux.actdim(v1) == NaiveNASflux.actdim(v2) && NaiveNASflux.actrank(v1) == NaiveNASflux.actrank(v2)

relative_positions(arr) = collect(eachindex(arr) / length(arr))

"""
    crossoverswap((v1,v2)::Tuple; pairgen=default_pairgen, selection=FilterMutationAllowed())

Perform [`crossoverswap!`](@Ref) with `v1` and `v2` as output crossover points.

Inputs are selected from the feasible set (as determined by `separablefrom` and `selection`) through the supplied `pairgen` function.

Inputs `v1` and `v2` along with their entire graph are copied before operation is performed and originals are returned if operation is not successful.
"""
crossoverswap((v1, v2)::Tuple; kwargs...) = crossoverswap(v1, v2; kwargs...)
function crossoverswap(v1, v2; pairgen=default_pairgen, mergefun=default_mergefun, selection=FilterMutationAllowed())
    # This is highly annoying: crossoverswap! does multiple remove/create_edge! and is therefore very hard to revert should something go wrong with one of the steps.
    # To mitigate this a backup copy is used. It is however not easy to backup a single vertex as it is connected to all other vertices in the graph, meaning that the whole graph must be copied. Sigh...
    function copyvertex(v)
        g = regraph(v)
        # TODO: Shallow-copy the graph, i.e. copy the metadata but leave things like NN weights the same instance.
        # This should speed up the operation and reduce memory consumption.
        # Someone else most likely wants the graph copied, but in the general case this happens before the crossover operation is even initiated. Lets see how things play out before we try it...
        vs = vertices(copy(g))
        return vs[indexin([v], vertices(g))][], g.inputs
    end
    v1c, ivs1 = copyvertex(v1)
    v2c, ivs2 = copyvertex(v2)

    vs1 = select(selection, separablefrom(v1c, ivs1))
    vs2 = select(selection, separablefrom(v2c, ivs2))

    # From the two sets of separable vertices, find two matching pairs to use as endpoints in input direction
    # Endpoint in output direction is already determined by sel1 and sel2 and is returned by separablefrom as the first element
    inds = pairgen(vs1,vs2)

    isnothing(inds) && return v1, v2

    ind1, ind2 = inds

    success1, success2 = crossoverswap!(vs1[ind1], vs1[1], vs2[ind2], vs2[1]; mergefun=mergefun)

    v1ret = success1 ? v1c : v1
    v2ret = success2 ? v2c : v2

    # Note! Flip v1 and v2 for same reason as success1 and 2 are flipped in crossoverswap!
    return v2ret, v1ret
end


"""
    crossoverswap!(v1::AbstractVertex, v2::AbstractVertex) = crossoverswap!(v1,v1,v2,v2)
    crossoverswap!(vin1::AbstractVertex, vout1::AbstractVertex, vin2::AbstractVertex, vout2::AbstractVertex)

Swap vertices `vin1` to `vout1` with `vin2` and `vout2` so that `vin1` to `vin2` is placed in the same position of the graph as `vin2` to `vout2` and vice versa.

Vertices must come from different graphs.

This operation can fail, leaving one or both graphs in a corrupted state where evaluating them results in an error (typically a `DimensionMismatch` error).

Return a tuple `(success1, success2)` where `success1` is true if `vin2` and `vou2` was successfully swapped in to the graph which previously contained `vin1` and `vout1` and vice versa for `success2`.
"""
crossoverswap!(v1::AbstractVertex, v2::AbstractVertex; kwargs...) = crossoverswap!(v1,v1,v2,v2; kwargs...)

function crossoverswap!(vin1::AbstractVertex, vout1::AbstractVertex, vin2::AbstractVertex, vout2::AbstractVertex; mergefun = default_mergefun, strategy=default_crossoverswap_strategy)

    # If vin1 can only take a single input while vin2 has multiple inputs (or vice versa) we need to add a vertex before which merges the inputs
    vin1, vin2 = check_singleinput!(vin1, vin2, mergefun)

    # Beware: ix and ox are not the same thing!! Check the strip function
    i1, o1 = stripedges!(vin1, vout1)
    i2, o2 = stripedges!(vin2, vout2)

    # success1 mapped to vin2 and vout2 looks backwards, but remember that vin2 and vout2 are the new guys being inserted in everything connected to i1 and o1
    # Returning success status instead of acting as a noop at failure is not very nice, but I could not come up with a way which was 100% to revert a botched attempt and making a backup of vertices before doing any changes is not easy to deal with for the receiver either

    success1 = addinedges!(vin2, i1, strategy)
    success1 && apply_mutation.(all_in_Δsize_graph(vin2, Input()))
    success1 &= success1 && addoutedges!(vout2, o1, strategy)

    success2 = addinedges!(vin1, i2, strategy)
    success2 && apply_mutation.(all_in_Δsize_graph(vin1, Input()))
    success2 &= success2 && addoutedges!(vout1, o2, strategy)

    return success1, success2
end

function check_singleinput!(v1, v2, mergefun)
    singleinputs = singleinput.([v1, v2])

    needmerge1 = singleinputs[1] && !all(singleinputs)
    needmerge2 = singleinputs[2] && !all(singleinputs)

    function addmerge!(v)
        # vs -> [v] means only add the new vertex between vi and v as vi could have other outputs
        insert!(inputs(v)[], vi -> mergefun(vi), vs -> [v])
        return inputs(v)[]
    end
    v1 = needmerge1 ? addmerge!(v1) : v1
    v2 = needmerge2 ? addmerge!(v2) : v2
    return v1, v2
end

#TODO: Add in NaiveNASlib?
struct FailAlignSizeNoOp <: AbstractAlignSizeStrategy end
NaiveNASlib.postalignsizes(s::FailAlignSizeNoOp, vin, vout, pos) = false
NaiveNASlib.prealignsizes(s::FailAlignSizeNoOp, vin, vout, will_rm) = false

# TODO: Change valuefun!!
function default_crossoverswap_strategy(valuefun = v -> ones(nout_org(v)))
    alignstrat = PostAlignJuMP(DefaultJuMPΔSizeStrategy(), fallback=FailAlignSizeWarn(msgfun = (vin,vout) -> "Failed to align sizes when adding edge between $(name(vin)) and $(name(vout)) for crossover. Reverting..."))

    selectstrat = OutSelect{Exact}(OutSelect{Relaxed}(LogSelectionFallback("Reverting...", NoutRevert())))

    return PostSelectOutputs(selectstrat, alignstrat, valuefun, FailAlignSizeRevert())
end

stripedges!(vin, vout) = stripinedges!(vin) ,stripoutedges!(vout)

function stripinedges!(v)
    # Unfortunately remove_edge! also removes metadata, causing subsequent create_edge! to create new neurons instead of keeping old ones.

    # Instead we create a dummyvertex between v and each of its inputs to act as a buffer and remove the edges from v's inputs to the dummyvertex. The dummyvertex will be removed by addinedges!
    i = copy(inputs(v))

    for (ind, vi) in enumerate(i)

        # We do this instead of insert! as it is a bit too smart and recreates all edges if any vi == vj for i != j where vi and vj are input vertices to v and this throws addinedges! off.
        dummy = dummyvertex(vi, trait(v))

        # Manually replace v with dummy as output for vi
        inputs(v)[ind] = dummy
        push!(outputs(dummy), v)
        deleteat!(outputs(vi), findall(vx -> vx == v, outputs(vi)))

        remove_edge!(vi, dummy, strategy = NoSizeChange())
    end
    return i
end


# TODO: All dummyvertices can be invariantvertex?
dummyvertex(v) = dummyvertex(v, trait(v))
dummyvertex(v, t::DecoratingTrait) = dummyvertex(v, base(t))
dummyvertex(v, t) = invariantvertex(identity, v; traitdecoration = t -> NamedTrait(t, "$(name(v)).dummy"))
dummyvertex(v, t::SizeAbsorb) = conc(v; dims=1, traitdecoration = t -> NamedTrait(t, "$(name(v)).dummy"))

function addinedges!(v, ins, strat = default_crossoverswap_strategy)
    dummies = copy(inputs(v))
    outs = copy(dummies)
    connectstrat = AbstractConnectStrategy[ConnectAll() for i in eachindex(dummies)]
    create_edge_strat = [i == length(ins) ? strat() : NoSizeChange() for i in eachindex(ins)]

    #More outs than ins: We want to connect every input so lets just pad outs with v as this is what we want in the end
    while length(outs) < length(ins)
        push!(outs, v)
    end
    # More outs than ins: This means that v had more inputs in its old graph compared to the vertex it was swapped with.
    # No problem really as we only care about connecting the ins and extra dummies can be left hanging with no input before removal. However, this might leave v with larger nout than it really has if not removed before adding inputs.
    while length(ins) < length(dummies)
        #connectstrat[length(outs)] = ConnectNone()
        # TODO: Cleanup patched legacy design!!
        pop!(outs)
        vrm = pop!(dummies)
        pop!(connectstrat)
        remove!(vrm, RemoveStrategy(ConnectNone(), NoSizeChange()))
    end

    success = map((iv, ov, s) -> create_edge!(iv, ov; strategy = s), ins, outs, create_edge_strat) |> all
    return success && map((dv, cs) -> remove!(dv, RemoveStrategy(cs, NoSizeChange())), dummies, connectstrat) |> all
end

function stripoutedges!(v)
    # Similar story as stripinedges to avoid destroying the mutation metadata in the outputs, eventually
    # causing neurons to be unnecessary recreated. Instead, we insert a new dummy neuron which acts as a buffer for
    # which we don't care that it is corrupted as we will anyways remove it.
    insert!(v, dummyvertex, reverse)
    dummy = outputs(v)[]
    remove_edge!(v, dummy; strategy = NoSizeChange())
    return dummy
end

function addoutedges!(v, dummy, strat = default_crossoverswap_strategy)
    #TODO: Try to avoid large negative size changes here, e.g. by doing a Δnout/Δnin before creating the edge?
    success = create_edge!(v, dummy, strategy = strat())
    return success && remove!(dummy, RemoveStrategy(NoSizeChange()))
end

"""
    separablefrom(v)

Return an array of vertices which may be separated from the graph.

More precisely, a connected component which is not connected to the graph can be created if
    1. All output edges from `v` are removed
    2. All input edges from a vertex `v'` in the returned array are removed

The disconncted component has `v'` as first vertex and `v` as last and contains all vertices in between them.

Note that output always contains `v`, i.e it is never empty.
"""
function separablefrom(v, forbidden = AbstractVertex[])
    # Rewrite in a guaranteed to be non-destrucive manner? LightGraphs?
    o = stripoutedges!(v)
    swappable = separablefrom(v, forbidden, AbstractVertex[])
    addoutedges!(v, o, NoSizeChange)
    return swappable
end

function separablefrom(v, forbidden, seen)
    push!(seen, v)
    ins = stripinedges!(v)
    seen_and_dummies = vcat(seen, inputs(v))
    ok = all(vv -> vv in seen_and_dummies && vv ∉ forbidden, all_in_graph(v))
    addinedges!(v, ins, NoSizeChange)
    swappable = mapreduce(vi -> separablefrom(vi, forbidden, seen), vcat, inputs(v), init=[])
    return ok ? vcat(v, swappable) : swappable
end
