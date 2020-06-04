
"""
    crossoverswap(v1::AbstractVertex, v2::AbstractVertex, strategy = () -> PostAlignJuMP())
    crossoverswap(vin1::AbstractVertex, vout1::AbstractVertex, vin2::AbstractVertex, vout2::AbstractVertex, strategy = () -> PostAlignJuMP())

Swap vertices `vin1` to `vout1` with `vin2` and `vout2` so that `vin1` to `vin2` is placed in the same position of the graph as `vin2` to `vout2` and vice versa.

Vertices may come from different graphs.
"""
function crossoverswap(v1::AbstractVertex, v2::AbstractVertex)
     crossoverswap(v1,v1,v2,v2)
     return v1,v2
end
function crossoverswap(vin1::AbstractVertex, vout1::AbstractVertex, vin2::AbstractVertex, vout2::AbstractVertex)

    # Beware: ix and ox are not the same thing!! Check the strip function
    i1, o1 = stripedges(vin1, vout1)
    i2, o2 = stripedges(vin2, vout2)

    function revert(nr)
        nr >= 1 && stripinedges!(vin2)
        nr >= 2 && stripoutedges!(vout2)
        nr >= 3 && stripinedges!(vin1)
        nr >= 4 && stripoutedges!(vout1)

        strat(n) = nr >= n ? PostAlignJuMP() : NoSizeChange()

        @show out_inds.(op.(filter(ii -> ii isa MutationVertex, i1)))
        @show in_inds.(op.(o1))

        addinedges!(vin1, i1, strat(1))
        addoutedges!(vout1, o1, strat(2))

        addinedges!(vin2, i2, strat(3))
        addoutedges!(vout2, o2, strat(4))
        return vin1, vout1, vin2, vout2
    end

    addinedges!(vin2, i1) |> all || return revert(1)
    addoutedges!(vout2, o1) |> all || return revert(2)

    addinedges!(vin1, i2) |> all || return revert(3)
    addoutedges!(vout1, o2) |> all || return revert(4)

    return vin1, vout1, vin2, vout2
end

struct FailAlignSizeNoOp <: AbstractAlignSizeStrategy end
NaiveNASlib.postalignsizes(s::FailAlignSizeNoOp, vin, vout, pos) = false
NaiveNASlib.prealignsizes(s::FailAlignSizeNoOp, vin, vout, will_rm) = false

default_crossoverswap_strategy() = PostAlignJuMP(DefaultJuMPΔSizeStrategy(); fallback = FailAlignSizeWarn(;andthen=FailAlignSizeNoOp(), msgfun=(vin,vout) -> "Failed to align sizes when adding edge between $(name(vin)) and $(name(vout)) for crossover. Reverting..."))

stripedges(vin, vout) = stripinedges!(vin) ,stripoutedges!(vout)

function stripinedges!(v)
    i = copy(inputs(v))
    foreach(iv -> remove_edge!(iv, v; strategy = NoSizeChange()), i)
    return i
end

function stripoutedges!(v)
    # Does not use the same method as stripinedges as this destroys the mutation metadata in the outputs, eventually
    # causing neurons to be unnecessary recreated. Instead, we insert a new dummy neuron which acts as a buffer for
    # which we don't care that it is corrupted as we will anyways remove it.
    insert!(v, v -> conc(v; dims=1), reverse)
    dummy = outputs(v)[]
    remove_edge!(v, dummy; strategy = NoSizeChange())
    return dummy
end

function addinedges!(v, vis, strat = default_crossoverswap_strategy())
    # Only need to align sizes after the very last edge (I hope)
    strats = (i == length(vis) ? strat : NoSizeChange() for i in eachindex(vis))
    return map((iv, s) -> create_edge!(iv, v; strategy=s), vis, strats)
end

function addoutedges!(v, dummy, strat = default_crossoverswap_strategy())
    create_edge!(v, dummy, strategy = strat)
    ret = remove!(dummy, RemoveStrategy(strat))
    return ret
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
    o = stripoutedges!(v)
    swappable = separablefrom(v, AbstractVertex[v])
    addoutedges!(v, o)
    return swappable
end

function separablefrom(v, seen)
    push!(seen, v)
    ins = stripinedges!(v)
    ok = all(vv -> vv in seen, all_in_graph(v))
    addinedges!(v, ins)
    swappable = mapreduce(vi -> separablefrom(vi, seen), vcat, inputs(v), init=[])
    return ok ? vcat(v, swappable) : swappable
end
