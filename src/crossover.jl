
"""
    crossoverswap(v1::AbstractVertex, v2::AbstractVertex, strategy = () -> PostAlignJuMP())
    crossoverswap(vin1::AbstractVertex, vout1::AbstractVertex, vin2::AbstractVertex, vout2::AbstractVertex, strategy = () -> PostAlignJuMP())

Swap vertices `vin1` to `vout1` with `vin2` and `vout2` so that `vin1` to `vin2` is placed in the same position of the graph as `vin2` to `vout2` and vice versa.

Vertices may come from different graphs.
"""
function crossoverswap(v1::AbstractVertex, v2::AbstractVertex, strategy = default_crossoverswap_strategy)
     crossoverswap(v1,v1,v2,v2, strategy)
     return v1,v2
end
function crossoverswap(vin1::AbstractVertex, vout1::AbstractVertex, vin2::AbstractVertex, vout2::AbstractVertex, strategy = default_crossoverswap_strategy)

    i1, ininds1, o1, oinds1 = stripedges(vin1, vout1)
    i2, ininds2, o2, oinds2 = stripedges(vin2, vout2)

    function revert(nr)
        nr >= 1 && stripinedges!(vin2)
        nr >= 2 && stripoutedges!(vout2)
        nr >= 3 && stripinedges!(vin1)
        nr >= 4 && stripoutedges!(vout1)

        strat(n) = nr >= n ? PostAlignJuMP() : NoSizeChange()

        addinedges!(vin1, i1, ininds1, strat(1))
        addoutedges!(vout1, o1, oinds1, strat(2))

        addinedges!(vin2, i2, ininds2, strat(3))
        addoutedges!(vout2, o2, oinds2, strat(4))
        return vin1, vout1, vin2, vout2
    end

    addinedges!(vin2, i1, ininds1, strategy()) |> all || return revert(1)
    addoutedges!(vout2, o1, oinds1, strategy()) |> all || return revert(2)

    addinedges!(vin1, i2, ininds2, strategy()) |> all || return revert(3)
    addoutedges!(vout1, o2, oinds2, strategy()) |> all || return revert(4)

    return vin1, vout1, vin2, vout2
end

struct FailAlignSizeNoOp <: AbstractAlignSizeStrategy end
NaiveNASlib.postalignsizes(s::FailAlignSizeNoOp, vin, vout, pos) = false
NaiveNASlib.prealignsizes(s::FailAlignSizeNoOp, vin, vout, will_rm) = false

default_crossoverswap_strategy() = PostAlignJuMP(DefaultJuMPΔSizeStrategy(); fallback = FailAlignSizeWarn(;andthen=FailAlignSizeNoOp(), msgfun=(vin,vout) -> "Failed to align sizes when adding edge between $(name(vin)) and $(name(vout)) for crossover. Reverting..."))

stripedges(vin, vout) = stripinedges!(vin)...,stripoutedges!(vout)...

function stripinedges!(v)
    i = copy(inputs(v))
    foreach(iv -> remove_edge!(iv, v; strategy = NoSizeChange()), i)
    return i, 1:length(i) # Inds mainly for symmetry with stripoutedges
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
    return map((iv, pos, s) -> create_edge!(iv, v; pos=pos, strategy=s), vis, inds, strats)
end

function addoutedges!(v, vos, inds, strat)
    # Only need to align sizes after the very last edge (I hope)
    strats = (i == length(vos) ? strat : NoSizeChange() for i in eachindex(vos))
    return map((ov, pos, s) -> create_edge!(v, ov; pos=pos, strategy=s), vos, inds, strats)
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
