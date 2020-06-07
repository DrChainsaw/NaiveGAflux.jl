
"""
    crossover(g1::CompGraph, g2::CompGraph; selection=FilterMutationAllowed(), pairgen=default_pairgen, crossoverfun=crossoverswap_bc)

Perform crossover between `g1` and `g2` and return the two children `g1'` and `g2'`.

How the crossover is performed depends on `pairgen` and `crossoverfun` as well as `selection`.

`selection` is used to filter out at which vertices the crossoverpoints may be.

`pairgen` return indices of the potential crossover points to use out of the allowed crossover points. New crossover points will be drawn from pairgen until it returns `nothing`.

`crossoverfun` return the result of the crossover which may be same as inputs depending on implementation.
"""
function crossover(g1::CompGraph, g2::CompGraph; selection=FilterMutationAllowed(), pairgen=default_pairgen, crossoverfun=crossoverswap_bc)

    # pairgen api is very close to the iterator specifiction. Not sure if things would be easier if it was an iterator instead...
    sel(g) = select(selection, g)

    inds = pairgen(sel(g1), sel(g2), ind1 = 1)
    while !isnothing(inds)
        ind1, ind2 = inds

        # sel1 and sel2 returned only because crossoverfun may be a generic AbstractMutation (e.g. MutationProbability) which by contract returns its inputs if noop
        g1,g2,sel1,sel2 = crossoverfun((g1,g2, g -> sel(g)[ind1], g -> sel(g)[ind2]))

        # Graphs may be different now, so we need to reselect
        inds = pairgen(sel(g1), sel(g2), ind1 = ind1+1)
    end
    return g1, g2
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

    order1 = relative_positions(vs1) .+ deviation .* randn(rng, length(vs1))
    order2 = relative_positions(vs2) .+ deviation .* randn(rng, length(vs2))

    ind2 = candidate_ind2s[argmin(abs.(order2[candidate_ind2s] .- order1[ind1]))]
    return ind1, ind2
end

sameactdims(v1, v2) = NaiveNASflux.actdim(v1) == NaiveNASflux.actdim(v2) && NaiveNASflux.actrank(v1) == NaiveNASflux.actrank(v2)
relative_positions(arr) = collect(eachindex(arr) / length(arr))

"""
    crossoverswap_bc((g1,g2,sel1,sel2)::Tuple; pairgen=default_pairgen)

Perform [`crossoverswap`](@Ref) between a set of vertices from `g1` and a set of vertices from `g2`.

Outputs of the swap is selected by `sel1` and `sel2` while inputs are selected from the feasible set (as determined by `separablefrom`) through the supplied `pairgen` function.

Inputs `g1` and `g2` are copied before operation is performed and originals are returned if operation is not successful.

Function is designed to work inside an `AbstractMutation` (e.g. MutationProbability) which is the reason for the awkward signature and return values.
"""
function crossoverswap_bc((g1,g2,sel1,sel2)::Tuple; pairgen=default_pairgen)
    g1c = copy(g1)
    g2c = copy(g2)

    # Swapping inputs or outputs will replace the whole graph but the CompGraph will have the old inputs causing all kinds of mayhem
    # Instead of trying to detect it and fixing or try to create new graphs it we'll just no allow it
    vs1 = filter(v -> allow_mutation(v) && v ∉ g1c.inputs && v ∉ g1c.outputs, separablefrom(sel1(g1c)))
    vs2 = filter(v -> allow_mutation(v) && v ∉ g2c.inputs && v ∉ g2c.outputs, separablefrom(sel2(g2c)))

    # From the two sets of separable vertices, find two matching pairs to use as endpoints in input direction
    # Endpoint in output direction is already determined by sel1 and sel2 and is returned by separablefrom as the first element
    ind1, ind2 = pairgen(vs1,vs2)

    success1, success2 = crossoverswap!(vs1[ind1], vs1[1], vs2[ind2], vs2[1])

    g1ret = success1 ? g1c : g1
    g2ret = success2 ? g2c : g2

    return g1ret, g2ret, sel1, sel2 #Just to be compatiable with mutation utils, like MutationProbability
end


"""
    crossoverswap!(v1::AbstractVertex, v2::AbstractVertex)
    crossoverswap!(vin1::AbstractVertex, vout1::AbstractVertex, vin2::AbstractVertex, vout2::AbstractVertex)

Swap vertices `vin1` to `vout1` with `vin2` and `vout2` so that `vin1` to `vin2` is placed in the same position of the graph as `vin2` to `vout2` and vice versa.

Vertices may come from different graphs.
"""
crossoverswap!(v1::AbstractVertex, v2::AbstractVertex) = crossoverswap!(v1,v1,v2,v2)

function crossoverswap!(vin1::AbstractVertex, vout1::AbstractVertex, vin2::AbstractVertex, vout2::AbstractVertex)

    # Beware: ix and ox are not the same thing!! Check the strip function
    i1, o1 = stripedges(vin1, vout1)
    i2, o2 = stripedges(vin2, vout2)

    # success1 mapped to vin2 and vout2 looks backwards, but remember that vin2 and vout2 are the new guys being inserted in everything connected to i1 and o1
    # Returning success status instead of acting as a noop at failure is not very nice, but I could not come up with a way which was 100% to revert a botched attempt and making a backup of vertices before doing any changes is not easy to deal with for the receiver either
    success1 = addinedges!(vin2, i1) && addoutedges!(vout2, o1)
    success2 = addinedges!(vin1, i2) && addoutedges!(vout1, o2)

    return success1, success2
end

#TODO: Add in NaiveNASlib?
struct FailAlignSizeNoOp <: AbstractAlignSizeStrategy end
NaiveNASlib.postalignsizes(s::FailAlignSizeNoOp, vin, vout, pos) = false
NaiveNASlib.prealignsizes(s::FailAlignSizeNoOp, vin, vout, will_rm) = false

default_crossoverswap_strategy() = PostAlignJuMP(DefaultJuMPΔSizeStrategy(); fallback = FailAlignSizeWarn(;andthen=FailAlignSizeRevert(), msgfun=(vin,vout) -> "Failed to align sizes when adding edge between $(name(vin)) and $(name(vout)) for crossover. Reverting..."))

stripedges(vin, vout) = stripinedges!(vin) ,stripoutedges!(vout)

stripinedges!(v) = stripinedges!(v, layertype(v))

function stripinedges!(v, lt)
    # Need to copy inputs first and iterate

    i = copy(inputs(v))
    foreach(vi -> remove_edge!(vi, v, strategy=NoSizeChange()), i)
    return i
end
function stripinedges!(v, ::FluxLayer)
    # Unfortunately remove_edge! also removes metadata, causing subsequent create_edge! to create new neurons instead of keeping old ones.

    # Instead we create a dummyvertex between v and each of its inputs to act as a buffer and remove the edges from v's inputs to the dummyvertex. The dummyvertex will be removed by addinedges!
    i = copy(inputs(v))
    for (ind, vi) in enumerate(i)

        # We do this instead of insert! as it is a bit too smart and recreates all edges if any vi == vj for i != j where vi and vj are input vertices to v and this throws addinedges! off.
        dummy = dummyvertex(vi)

        # Manually replace v with dummy as output for vi
        inputs(v)[ind] = dummy
        push!(outputs(dummy), v)
        deleteat!(outputs(vi), findall(vx -> vx == v, outputs(vi)))

        remove_edge!(vi, dummy, strategy = NoSizeChange())
    end
    return i
end


dummyvertex(v) = conc(v; dims=1, traitdecoration = t -> NamedTrait(t, "$(name(v)).dummy"))

function addinedges!(v, ins, strat = default_crossoverswap_strategy)
    dummies = copy(inputs(v))
    outs = copy(dummies)
    connectstrat = AbstractConnectStrategy[ConnectAll() for i in eachindex(outs)]
    create_edge_strat = [i == length(ins) ? strat() : NoSizeChange() for i in eachindex(ins)]

    #More outs than ins: We want to connect every input so lets just pad outs with v as this is what we want in the end
    while length(outs) < length(ins)
        push!(outs, v)
    end
    # More outs than ins: No problem really as we only care about connecting the ins and extra dummies can be left hanging before removal. However, map fails if sizes are not equal.
    while length(ins) < length(outs)
        connectstrat[length(outs)] = ConnectNone()
        pop!(outs)
    end

    ret = map((iv, ov, s) -> create_edge!(iv, ov; strategy = s), ins, outs, create_edge_strat) |> all
    ret &= map((dv, cs) -> remove!(dv, RemoveStrategy(cs, NoSizeChange())), dummies, connectstrat) |> all
    return ret
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
    ret = create_edge!(v, dummy, strategy = strat())
    ret &= remove!(dummy, RemoveStrategy(NoSizeChange()))
    return ret
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
function separablefrom(v)
    # Rewrite in a guaranteed to be non-destrucive manner? LightGraphs?
    o = stripoutedges!(v)
    swappable = separablefrom(v, AbstractVertex[])
    addoutedges!(v, o, NoSizeChange)
    return swappable
end

function separablefrom(v, seen)
    push!(seen, v)
    ins = stripinedges!(v)
    seen_and_dummies = vcat(seen, inputs(v))
    ok = all(vv -> vv in seen_and_dummies, all_in_graph(v))
    addinedges!(v, ins, NoSizeChange)
    swappable = mapreduce(vi -> separablefrom(vi, seen), vcat, inputs(v), init=[])
    return ok ? vcat(v, swappable) : swappable
end
