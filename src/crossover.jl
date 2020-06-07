
"""
    crossoverswap(v1::AbstractVertex, v2::AbstractVertex, strategy = () -> PostAlignJuMP())
    crossoverswap(vin1::AbstractVertex, vout1::AbstractVertex, vin2::AbstractVertex, vout2::AbstractVertex, strategy = () -> PostAlignJuMP())

Swap vertices `vin1` to `vout1` with `vin2` and `vout2` so that `vin1` to `vin2` is placed in the same position of the graph as `vin2` to `vout2` and vice versa.

Vertices may come from different graphs.
"""
crossoverswap(v1::AbstractVertex, v2::AbstractVertex) = crossoverswap(v1,v1,v2,v2)

function crossoverswap(vin1::AbstractVertex, vout1::AbstractVertex, vin2::AbstractVertex, vout2::AbstractVertex)

    # Beware: ix and ox are not the same thing!! Check the strip function
    i1, o1 = stripedges(vin1, vout1)
    i2, o2 = stripedges(vin2, vout2)

    success1 = addinedges!(vin2, i1) && addoutedges!(vout2, o1)
    success2 = addinedges!(vin1, i2) && addoutedges!(vout1, o2)

    return success1, success2
end

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
