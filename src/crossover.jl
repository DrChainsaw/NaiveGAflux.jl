"""
    VertexCrossover{S, PF, CF}
    VertexCrossover(crossover ;selection=FilterMutationAllowed(), pairgen=default_pairgen)
    VertexCrossover(crossover, deviation::Number; selection=FilterMutationAllowed())

Applies `crossover` to each pair of selected vertices from two `CompGraph`s.

Vertices to select from the first graph is determined by `selection` (default [`FilterMutationAllowed`](@ref) while `pairgen` (default [`default_pairgen`](@ref)) determines how to pair the selected vertices with vertices in the second graph.

The default pairing function will try to pair vertices which have similar relative topologial order within their graphs. For instance, if the first graph has 5 vertices and the second has 10, it will pair vertex 2 from the first graph with vertex 4 from the second (assuming they are of compatible type). The parameter `deviation` can be used to inject noise in this process so that the pairing will randomly deviate where the magnitude of `deviation` sets how much and how often.

See also [`crossover`](@ref).
"""
struct VertexCrossover{S, PF, CF} <: AbstractCrossover{CompGraph}
    selection::S
    pairgen::PF
    crossover::CF
end
VertexCrossover(crossover ;selection=FilterMutationAllowed(), pairgen=default_pairgen) = VertexCrossover(selection, pairgen, crossover)
VertexCrossover(crossover, deviation::Number; selection=FilterMutationAllowed()) = VertexCrossover(selection, (vs1,vs2;ind1=1) -> default_pairgen(vs1, vs2, deviation;ind1=ind1), crossover)

(c::VertexCrossover)((g1,g2)::Tuple) = crossover(g1, g2; selection=SelectWithMutation(c.selection, c), pairgen=c.pairgen, crossoverfun=c.crossover)


"""
    CrossoverSwap{F1, F2, F3, S} <: AbstractCrossover{<:AbstractVertex}
    CrossoverSwap(pairgen, mergefun, strategy, selection)
    CrossoverSwap(;pairgen=default_inputs_pairgen, mergefun=default_mergefun, strategy=default_crossoverswap_strategy, selection=FilterMutationAllowed()) = CrossoverSwap(pairgen, mergefun, selection)
    CrossoverSwap(deviation::Number; mergefun=default_mergefun, strategy=default_crossoverswap_strategy, selection=FilterMutationAllowed()) = CrossoverSwap((vs1,vs2) -> default_inputs_pairgen(vs1, vs2, deviation), mergefun, strategy, selection)

Swap out a part of one graph with a part of another graph, making sure that the graphs do not become connected in the process.

More concretely, swaps a set of consecutive vertices `vs1` set of consecutive vertices `vs2` returning the swapped `v1` and `v2` respectively if successful or `v1` and `v2` if not.

The last vertex in `vs1` is `v1` and the last vertex of `vs2` is `v2`. The other members of `vs1` and `vs2` are determined by `pairgen` (default [`default_inputs_pairgen`](@ref)) and `selection` (default [`FilterMutationAllowed`](@ref)).

If a vertex `v` is not capable of having multiple inputs (determined by `singleinput(v) == true`), `vm = mergefun(vi)` where `vi` is the input to `v` will be used instead of `v` and `v` will be added as the output of `vm` if necessary.

See also [`crossoverswap`](@ref).

Note: High likelyhood of large accuracy degradation after applying this mutation.
"""
struct CrossoverSwap{F1, F2, F3, S} <: AbstractCrossover{AbstractVertex}
    pairgen::F1
    mergefun::F2
    strategy::F3
    selection::S
end
CrossoverSwap(;pairgen=default_inputs_pairgen, mergefun=default_mergefun, strategy=default_crossoverswap_strategy, selection=FilterMutationAllowed()) = CrossoverSwap(pairgen, mergefun, strategy, selection)
CrossoverSwap(deviation::Number; mergefun=default_mergefun, strategy=default_crossoverswap_strategy, selection=FilterMutationAllowed()) = CrossoverSwap((vs1,vs2) -> default_inputs_pairgen(vs1, vs2, deviation), mergefun, strategy, selection)

(c::CrossoverSwap)((v1,v2)::Tuple) = crossoverswap(v1, v2; pairgen = c.pairgen, mergefun=c.mergefun, selection=SelectWithMutation(c.selection, c), strategy=c.strategy)


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

"""
    default_inputs_pairgen(vs1, vs2, args...;kwargs...)

Same as [´default_pairgen`](@ref) except it also ensures that shape changes of feature maps are consistent between the pairs.

Feature map here refers to the shape of inputs to convolutional-type layers (Conv, Pooling) in dimensions other than the batch dimension or the channel dimension.

For example, for 2D convolutions, the arrays may have the shape WxHxCxB where C and B are the channel and batch dimensions respectively. The shape of the feature map in this case is then WxH.

More concretely, if `x` is the shape of a feature maps input to `vin1` and `f1(x)` is the shape of the feature maps output from `vout1` and `f2` describes the same relation between `vin2` and `vout2` then only vertices `vin2'` for which `f1(x) == f2(x) ∀ x` for a selected vertex `vin1` may be returned.

This function assumes that the last vertex in `vs1` and `vs2` are `vout1` and `vout2` respectively.

This prevents issues where graphs become inconsistent due to
1) Feature maps become zero sized
2) Feature maps of different sizes are merged (e.g. concatenated or element wise added)

Note that this is more strict than needed as a change in the feature maps size does not necessarily result in 1) or 2).
"""
function default_inputs_pairgen(vs1, vs2, args...;kwargs...)
    samedimsandshapes(vin1, vin2) = NaiveGAflux.sameactdims(vin1,vin2) && sameoutshape(vin1, last(vs1), vin2, last(vs2))
    return default_pairgen(vs1,vs2, args...;compatiblefun = samedimsandshapes, kwargs...)
end
# We also check all size terminating outputs (perhaps sufficient with one) since transparent layers (incorrectly in this case) just report their inputs dims, meaning that a flattening global pool will seem to have conv output dims
sameactdims(v1, v2) =  _sameactdims(v1, v2) && all(Iterators.product(findterminating(v1, outputs), findterminating(v2, outputs))) do (v1o, v2o)
    _sameactdims(v1o, v2o)
end
_sameactdims(v1, v2) = NaiveNASflux.actdim(v1) == NaiveNASflux.actdim(v2) && NaiveNASflux.actrank(v1) == NaiveNASflux.actrank(v2)

function sameoutshape(vin1, vout1, vin2, vout2)
    tr1 = squashshapes(shapetrace(vout1, vin1))
    tr2 = squashshapes(shapetrace(vout2, vin2))
    return isempty(Δshapediff(tr1,tr2))
end

relative_positions(arr) = collect(eachindex(arr) / length(arr))

"""
    crossoverswap((v1,v2)::Tuple; pairgen=default_inputs_pairgen, selection=FilterMutationAllowed(), kwargs...)

Perform [`crossoverswap!`](@ref) with `v1` and `v2` as output crossover points.

Inputs are selected from the feasible set (as determined by `separablefrom` and `selection`) through the supplied `pairgen` function.

Inputs `v1` and `v2` along with their entire graph are copied before operation is performed and originals are returned if operation is not successful.

Additional keyword arguments will be passed on to [`crossoverswap!`](@ref).
"""
crossoverswap((v1, v2)::Tuple; kwargs...) = crossoverswap(v1, v2; kwargs...)
function crossoverswap(v1, v2; pairgen=default_inputs_pairgen, selection=FilterMutationAllowed(), kwargs...)
    # This is highly annoying: crossoverswap! does multiple remove/create_edge! and is therefore very hard to revert should something go wrong with one of the steps.
    # To mitigate this a backup copy is used. It is however not easy to backup a single vertex as it is connected to all other vertices in the graph, meaning that the whole graph must be copied. Sigh...
    function copyvertex(v)
        g = regraph(v)
        vs = vertices(copy(g))
        return vs[indexin([v], vertices(g))][], filter(vv -> isempty(inputs(vv)), vs)
    end
    v1c, ivs1 = copyvertex(v1)
    v2c, ivs2 = copyvertex(v2)

    vs1 = reverse!(select(selection, separablefrom(v1c, ivs1)))
    vs2 = reverse!(select(selection, separablefrom(v2c, ivs2)))

    # From the two sets of separable vertices, find two matching pairs to use as endpoints in input direction
    # Endpoint in output direction is already determined by sel1 and sel2 and is returned by separablefrom as the first element
    inds = pairgen(vs1, vs2)

    isnothing(inds) && return v1, v2

    ind1, ind2 = inds

    success1, success2 = crossoverswap!(vs1[ind1], vs1[end], vs2[ind2], vs2[end]; kwargs...)

    # Note: Success 1 and 2 are mapped to v2 and v1 respectively as success1 means we successfully inserted v1c into the graph which held v2c and vice versa
    v1ret = success2 ? v1c : v1
    v2ret = success1 ? v2c : v2

    # Note! Flip v1 and v2 for same reason as success1 and 2 are flipped in crossoverswap!
    return v2ret, v1ret
end


"""
    crossoverswap!(v1::AbstractVertex, v2::AbstractVertex) = crossoverswap!(v1,v1,v2,v2)
    crossoverswap!(vin1::AbstractVertex, vout1::AbstractVertex, vin2::AbstractVertex, vout2::AbstractVertex)

Swap vertices `vin1` to `vout1` with `vin2` and `vout2` so that `vin1` to `vin2` is placed in the same position of the graph as `vin2` to `vout2` and vice versa.

Vertices must come from different graphs.

This operation can fail, leaving one or both graphs in a corrupted state where evaluating them results in an error (typically a `DimensionMismatch` error).

Return a tuple `(success1, success2)` where `success1` is true if `vin2` and `vout2` was successfully swapped in to the graph which previously contained `vin1` and `vout1` and vice versa for `success2`.
"""
crossoverswap!(v1::AbstractVertex, v2::AbstractVertex; kwargs...) = crossoverswap!(v1,v1,v2,v2; kwargs...)

function crossoverswap!(vin1::AbstractVertex, vout1::AbstractVertex, vin2::AbstractVertex, vout2::AbstractVertex; mergefun = default_mergefun, strategy=default_crossoverswap_strategy)

    # If vin1 can only take a single input while vin2 has multiple inputs (or vice versa) we need to add a vertex before which merges the inputs
    vin1, vin2 = check_singleinput!(vin1, vin2, mergefun)

    # Beware: ix and ox are not the same type of thing!! Check the strip functions!
    i1, o1 = stripedges!(vin1, vout1)
    i2, o2 = stripedges!(vin2, vout2)

    # success1 mapped to vin2 and vout2 looks backwards, but remember that vin2 and vout2 are the new guys being inserted 
    # in everything connected to i1 and o1
    # Returning success status instead of acting as a noop at failure is not very nice, but I could not come up with a way 
    # which was 100% to revert a botched attempt and making a backup of vertices before doing any changes is not easy to deal 
    # with for the receiver either
    success1 = addinedges!(vin2, i1, strategy)
    success1 &= success1 && addoutedges!(vout2, o1, strategy)

    success2 = addinedges!(vin1, i2, strategy)
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

function default_crossoverswap_strategy(valuefun = NaiveNASlib.default_outvalue)
    warnfailalign = FailAlignSizeWarn(msgfun = (vin,vout) -> "Failed to align sizes for vertices $(name(vin)) and $(name(vout)) for crossover. Attempt aborted!")
    alignstrat = TruncateInIndsToValid(WithValueFun(valuefun, AlignNinToNout(;fallback=ΔSizeFailNoOp())))
    return PostAlign(alignstrat, fallback=warnfailalign)
end

struct NonZeroSizeTrait{T <: NaiveNASlib.MutationTrait} <: NaiveNASlib.DecoratingTrait
    minsize::Int
    base::T
end
NonZeroSizeTrait(base) = NonZeroSizeTrait(1, base)

NaiveNASlib.base(t::NonZeroSizeTrait) = t.base
NaiveNASlib.nin(t::NonZeroSizeTrait, v::AbstractVertex) = max.(t.minsize, nin(base(t), v))
NaiveNASlib.nout(t::NonZeroSizeTrait, v::AbstractVertex) = max(t.minsize, nout(base(t), v))

# Invariant vertex so that removal is trivial.
# Using size absorbing dummies leads to selection of input neurons which dont exist (e.g. take input nr 243 from a layer with 16 inputs).
# Using size stacking dummies leads to neurons being whiped out (e.g. when connecting a 16 neuron output to a 8 neuron input then all 8 are replaced with new neurons).
# We must ensure size is non-zero though, or else NaiveNASlib will not generate any variables for setting the size
dummyvertex(v) = invariantvertex(ActivationContribution(identity), v; traitdecoration = named("$(name(v)).dummy") ∘ NonZeroSizeTrait)

stripedges!(vin, vout) = stripinedges!(vin) ,stripoutedges!(vout)

function stripinedges!(v)
    # We would like to just remove all input edges using remove_edge!, but this messes up the order of inputs to the outputs of v if v is not the only input
    # Instead we add a  dummyvertex after each input to v and disconnect it (which preserves order as we know it has only one input)
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

function addinedges!(v, ins, strat = default_crossoverswap_strategy)
    dummies = copy(inputs(v))
    outs = copy(dummies)

    #More outs than ins: We want to connect every input so lets just pad outs with v as this is what we want in the end
    while length(outs) < length(ins)
        push!(outs, v)
    end
    # More outs than ins: This means that v had more inputs in its old graph compared to the vertex it was swapped with.
    # No problem really as we only care about connecting the ins and extra dummies can be left hanging with no input before removal. However, this might leave v with larger nout than it really has if not removed before adding inputs.
    while length(ins) < length(dummies)
        # If we end up here outs and dummies are just copies of each other so we just pop the same element from both.
        pop!(outs)
        vrm = pop!(dummies)
        remove!(vrm, RemoveStrategy(ConnectNone(), NoSizeChange()))
    end
    create_edge_strat = (i == length(ins) ? strat() : NoSizeChange() for i in eachindex(ins))
    success = map((iv, ov, s) -> create_edge!(iv, ov; strategy = s), ins, outs, create_edge_strat) |> all
    remove_dummy_strat = (i == length(dummies) ? strat() : NoSizeChange() for i in eachindex(dummies))
    return success && map((dv,  s)-> remove!(dv, RemoveStrategy(s)), dummies, remove_dummy_strat) |> all
end

function stripoutedges!(v)
    # Similar story as stripinedges to avoid messing with the input order for outputs to v.
    insert!(v, dummyvertex, reverse) 
    dummy = outputs(v)[]
    remove_edge!(v, dummy; strategy = NoSizeChange())
    return dummy
end

function addoutedges!(v, dummy, strat = default_crossoverswap_strategy)
    # Perhaps try to avoid large negative size changes here, e.g. by doing a Δnout/Δnin before creating the edge?
    # Should perhaps be baked into strat, but then one probably can't use the same strat for addinedges!
    success = create_edge!(v, dummy, strategy = strat())
    return success && remove!(dummy, RemoveStrategy(strat()))
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


"""
    OptimizerCrossover{C} <: AbstractCrossover{FluxOptimizer}
    OptimizerCrossover()
    OptimizerCrossover(crossover)

Apply crossover between optimizers.

Type of crossover is determined by `crossover` (default `optimizerswap`) which when given a a tuple of two optimizers will return the result of the crossover operation as a tuple of optimizers.

Designed to be composable with most utility `AbstractMutation`s as well as with itself. For instance, the following seemingly odd construct will swap components of a [`Flux.Optimiser`](@ref) with a probability of `0.2` per component:

`OptimizerCrossover(MutationProbability(OptimizerCrossover(), 0.2))`

Compare with the following which either swaps all components or none:

`MutationProbability(OptimizerCrossover(), 0.2)`
"""
struct OptimizerCrossover{C} <: AbstractCrossover{FluxOptimizer}
    crossover::C
end
OptimizerCrossover() = OptimizerCrossover(optimizerswap)

"""
    LearningRateCrossover()

Return an `OptimizerCrossover` which will swap learning rates between optimizers but not change anything else.

Does not do anything if any of the optimizers don't have a learning rate (e.g. WeightDecay).
"""
LearningRateCrossover() = OptimizerCrossover(learningrateswap)

EitherIs{T} = Union{Tuple{T, Any}, Tuple{Any,T}}

(oc::OptimizerCrossover)(os) = oc.crossover(os)
(oc::OptimizerCrossover)(os::EitherIs{ShieldedOpt}) = os
function (oc::OptimizerCrossover)((o1,o2)::EitherIs{Flux.Optimiser})
    os1,o1re = optiter(o1)
    os2,o2re = optiter(o2)
    res = oc.crossover.(zip(os1,os2))
    os1n = (t[1] for t in res)
    os2n = (t[2] for t in res)
    return o1re(os1n..., os1[length(os2)+1:end]...), o2re(os2n..., os2[length(os1)+1:end]...)
end

optiter(o) = (o,), (os...) -> os[1]
optiter(o::Flux.Optimiser) = Tuple(o.os), (os...) -> Flux.Optimiser(os...)

optimizerswap((o1, o2)::Tuple) = o2,o1

learningrateswap((o1,o2)::Tuple) = (@set o1.eta = learningrate(o2)) , (@set o2.eta = learningrate(o1))
learningrateswap(os::EitherIs{ShieldedOpt}) = os
learningrateswap(os::EitherIs{WeightDecay}) = os
