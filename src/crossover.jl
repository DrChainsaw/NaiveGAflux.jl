
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
    i1, o1 = stripedges(vin1, vout1)
    i2, o2 = stripedges(vin2, vout2)

    foreach(iv -> create_edge!(iv, vin2; strategy = strategy()), i1)
    foreach(ov -> create_edge!(vout2, ov; strategy = strategy()), o1)

    foreach(iv -> create_edge!(iv, vin1; strategy = strategy()), i2)
    foreach(ov -> create_edge!(vout1, ov; strategy = strategy()), o2)

    return vin1, vout1, vin2, vout2
end

stripedges(vin, vout) = stripinputs(vin),stripoutputs(vout)

function stripinputs(v)
    i = copy(inputs(v))
    foreach(iv -> remove_edge!(iv, v; strategy = NoSizeChange()), i)
    return i
end

function stripoutputs(v)
    o = copy(outputs(v))
    foreach(ov -> remove_edge!(v, ov; strategy = NoSizeChange()), o)
    return o
end
