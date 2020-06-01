
"""
    function crossoverswap(v1::AbstractVertex, v2::AbstractVertex, strategy = () -> PostAlignJuMP())

Swap vertices `v1` and `v2` so that `v1` is placed in `v2`s position of the graph and vice versa.

Vertices may come from different graphs.
"""
function crossoverswap(v1::AbstractVertex, v2::AbstractVertex, strategy = () -> PostAlignJuMP())
    i1, o1 = stripedges(v1)
    i2, o2 = stripedges(v2)

    foreach(iv -> create_edge!(iv, v2; strategy = strategy()), i1)
    foreach(ov -> create_edge!(v2, ov; strategy = strategy()), o1)

    foreach(iv -> create_edge!(iv, v1; strategy = strategy()), i2)
    foreach(ov -> create_edge!(v1, ov; strategy = strategy()), o2)

    return v1, v2
end

function stripedges(v)
    i, o = copy(inputs(v)), copy(outputs(v))
    foreach(iv -> remove_edge!(iv, v; strategy = NoSizeChange()), i)
    foreach(ov -> remove_edge!(v, ov; strategy = NoSizeChange()), o)
    return i, o
end
