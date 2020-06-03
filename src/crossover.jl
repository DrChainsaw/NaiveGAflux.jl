
"""
    crossoverswap(v1::AbstractVertex, v2::AbstractVertex, strategy = () -> PostAlignJuMP())
    crossoverswap(vin1::AbstractVertex, vout1::AbstractVertex, vin2::AbstractVertex, vout2::AbstractVertex, strategy = () -> PostAlignJuMP())

Swap vertices `vin1` to `vout1` with `vin2` and `vout2` so that `vin1` to `vin2` is placed in the same position of the graph as `vin2` to `vout2` and vice versa.

Vertices may come from different graphs.
"""
function crossoverswap(v1::AbstractVertex, v2::AbstractVertex, strategy = () -> PostAlignJuMP())
     crossoverswap(v1,v1,v2,v2, strategy)
     return v1,v2
end
function crossoverswap(vin1::AbstractVertex, vout1::AbstractVertex, vin2::AbstractVertex, vout2::AbstractVertex, strategy = () -> PostAlignJuMP())
    # TODO: Handle failure cases (if possible to fail)
    i1, ininds1, o1, oinds1 = stripedges(vin1, vout1)
    i2, ininds2, o2, oinds2 = stripedges(vin2, vout2)

    addinedges!(vin2, i1, ininds1, strategy())
    addoutedges!(vout2, o1, oinds1, strategy())

    addinedges!(vin1, i2, ininds2, strategy())
    addoutedges!(vout1, o2, oinds2, strategy())

    return vin1, vout1, vin2, vout2
end

stripedges(vin, vout) = stripinedges!(vin)...,stripoutedges!(vout)...

function stripinedges!(v)
    i = copy(inputs(v))
    foreach(iv -> remove_edge!(iv, v; strategy = NoSizeChange()), i)
    return i, 1:length(inputs(v)) # Inds mainly for symmetry with stripoutedges
end

function stripoutedges!(v)
    inds = mapreduce(vcat, unique(outputs(v))) do vo
        findall(voi -> voi == v, inputs(vo))
    end
    o = copy(outputs(v))
    foreach(ov -> remove_edge!(v, ov; strategy = NoSizeChange()), o)
    return o, inds
end

function addinedges!(v, vis, inds, strat)
    # Only need to align sizes after the very last edge (I hope)
    strats = (i == length(vis) ? strat : NoSizeChange() for i in eachindex(vis))
    foreach((iv, pos, s) -> create_edge!(iv, v; pos=pos, strategy=s), vis, inds, strats)
end

function addoutedges!(v, vos, inds, strat)
    # Only need to align sizes after the very last edge (I hope)
    strats = (i == length(vos) ? strat : NoSizeChange() for i in eachindex(vos))
    foreach((ov, pos, s) -> create_edge!(v, ov; pos=pos, strategy=s), vos, inds, strats)
end

"""
    separablefrom(v)

Return an array of vertices for which may be separated from the graph.

More precisely, a connected component which is not connected to the graph can be created if
    1. All output edges from `v` are removed
    2. All input edges from a vertex `v'` in the returned array are removed

The disconncted component has `v'` as first vertex and `v` as last and contains all vertices in between them.

Note that output always contains `v`, i.e it is never empty.
"""
function separablefrom(v)
    # Rewrite in a guaranteed to be non-destrucive manner? LightGraphs?
    o, oinds = stripoutedges!(v)
    swappable = separablefrom(v, AbstractVertex[v])
    addoutedges!(v, o, oinds, NoSizeChange())
    return swappable
end

function separablefrom(v, seen)
    push!(seen, v)
    i, ininds = stripinedges!(v)
    ok = all(vv -> vv in seen, all_in_graph(v))
    addinedges!(v, i, ininds, NoSizeChange())
    swappable = mapreduce(vi -> separablefrom(vi, seen), vcat, inputs(v), init=[])
    return ok ? vcat(v, swappable) : swappable
end
