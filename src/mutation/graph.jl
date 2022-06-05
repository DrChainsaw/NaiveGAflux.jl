
"""
VertexMutation <: DecoratingMutation{CompGraph}
VertexMutation(m::AbstractMutation{AbstractVertex}, s::AbstractVertexSelection)
VertexMutation(m::AbstractMutation{AbstractVertex})

Applies a wrapped `AbstractMutation{AbstractVertex}` to each selected vertex in a `CompGraph`.

Vertices to select is determined by the configured `AbstractVertexSelection`.
"""
struct VertexMutation{S<:AbstractVertexSelection} <: DecoratingMutation{CompGraph}
m::AbstractMutation{AbstractVertex}
s::S
end
VertexMutation(m::AbstractMutation{AbstractVertex}) = VertexMutation(m, FilterMutationAllowed())
function (m::VertexMutation)(g::CompGraph)
m.m(select(m.s, g, m))
return g
end

"""
NoutMutation <:AbstractMutation{AbstractVertex}
NoutMutation(l1::Real,l2::Real, rng::AbstractRNG)
NoutMutation(limit, rng::AbstractRNG=rng_default)
NoutMutation(l1,l2)

Mutate the out size of a vertex or vector of vertices.

Size is changed by `x * nout(v)` rounded away from from zero where `x` is drawn from `U(minrel, maxrel)` where 
`minrel` and `maxrel` are `l1` and `l2` if `l1 < l2` and `l2` and `l1` otherwise.
"""
struct NoutMutation{R<:Real, RNG<:AbstractRNG} <:AbstractMutation{AbstractVertex}
minrel::R
maxrel::R
rng::RNG
function NoutMutation(l1::R1, l2::R2, rng::RNG) where {R1, R2, RNG} 
    R = promote_type(R1, R2)
    return l1 < l2 ? new{R, RNG}(promote(l1, l2)..., rng) : new{R, RNG}(promote(l2, l1)..., rng)
end
end
NoutMutation(limit, rng::AbstractRNG=rng_default) = NoutMutation(0, limit, rng)
NoutMutation(l1,l2) = NoutMutation(l1,l2, rng_default)
(m::NoutMutation)(v::AbstractVertex) = first(m([v]))
function (m::NoutMutation)(vs::AbstractVector{<:AbstractVertex})

Δs = Dict{AbstractVertex, Int}()
shift = m.minrel
scale = m.maxrel - m.minrel

for v in vs
    terminputs = findterminating(v, inputs)

    # We are basically just searching for Immutable vertices here, allow_mutation(trait(v)) happens to do just that
    any(tv -> allow_mutation(trait(tv)), terminputs) || continue
    
    Δfloat = rand(m.rng) * scale + shift

    Δ = ceil(Int, abs(Δfloat) * nout(v)) *  sign(Δfloat)
    minsize = minimum(nout.(terminputs))
    # Or else we might increase the size despite Δ being negative which would be surprising to a user who has specified 
    # strictly negative size changes
    minsize + Δ <= 0 && continue

    Δs[v] = Δ
end

if !isempty(Δs)
    failmsg = (args...) -> "Could not change nout of $(join(NaiveNASlib.nameorrepr.(keys(Δs)), ", ", " and ")) by $(join(values(Δs), ", ", " and ")). No change!"

    strategy = TimeOutAction(;base=ΔNoutRelaxed(Δs), fallback=LogΔSizeExec(failmsg, Logging.Warn, ΔSizeFailNoOp()))

    Δsize!(strategy)
end
return vs
end

"""
AddVertexMutation <:AbstractMutation{AbstractVertex}
AddVertexMutation(s::AbstractArchSpace, outselect::Function, WeightInit::AbstractWeightInit, rng::AbstractRNG)
AddVertexMutation(s, outselect::Function=identity)
AddVertexMutation(s, rng::AbstractRNG)
AddVertexMutation(s, wi::AbstractWeightInit)

Insert a vertex from the wrapped `AbstractArchSpace` `s` after a given vertex `v`.

The function `outselect` takes an `AbstractVector{AbstractVertex}` representing the output of `v` and returns an `AbstractVector{AbstractVertex}` which shall be reconnected to the vertex `v'` returned by `s`. Defaults to `identity` meaning all outputs of `v` are reconnected to `v'`.
"""
struct AddVertexMutation{S<:AbstractArchSpace, F, WI<:AbstractWeightInit, RNG<:AbstractRNG} <:AbstractMutation{AbstractVertex}
s::S
outselect::F
weightinit::WI
rng::RNG
end
AddVertexMutation(s, outselect::Function=identity) = AddVertexMutation(s, outselect, IdentityWeightInit(), rng_default)
AddVertexMutation(s, rng::AbstractRNG) = AddVertexMutation(s, identity, IdentityWeightInit(), rng)
AddVertexMutation(s, wi::AbstractWeightInit) = AddVertexMutation(s, identity, wi, rng_default)

function (m::AddVertexMutation)(v::AbstractVertex)
insert!(v, vi -> m.s(name(vi), vi, m.rng, outsize=nout(vi), wi=m.weightinit), m.outselect)
return v
end

"""
RemoveVertexMutation <:AbstractMutation{AbstractVertex}
RemoveVertexMutation(s::RemoveStrategy)
RemoveVertexMutation()

Remove the given vertex `v` using the configured `RemoveStrategy`.

Default size align strategy is `IncreaseSmaller -> DecreaseBigger -> AlignSizeBoth -> FailAlignSizeWarn -> FailAlignSizeRevert`.

Default reconnect strategy is `ConnectAll`.

Note: High likelyhood of large accuracy degradation after applying this mutation.
"""
struct RemoveVertexMutation{S<:RemoveStrategy} <:AbstractMutation{AbstractVertex}
s::S
end
function RemoveVertexMutation() 
alignstrat = IncreaseSmaller(fallback=DecreaseBigger(fallback=AlignSizeBoth(fallback=FailAlignSizeWarn(msgfun = (vin,vout) -> "Can not remove vertex $(name(vin))! Could not align sizes of neighbours!"))))
return RemoveVertexMutation(RemoveStrategy(CheckAligned(CheckNoSizeCycle(alignstrat, FailAlignSizeWarn(msgfun = (vin,vout) -> "Can not remove vertex $(name(vin))! Size cycle detected!")))))
end

function (m::RemoveVertexMutation)(v::AbstractVertex)
remove!(v, m.s)
return v
end

default_neuronselect(args...) = NaiveNASlib.defaultutility(args...)

"""
AddEdgeMutation <: AbstractMutation{AbstractVertex}
AddEdgeMutation(p; rng=rng_default, mergefun=default_mergefun(rng=rng), filtfun=no_shapechange, utilityfun=default_neuronselect)
AddEdgeMutation(p::Probability; rng=rng_default, mergefun=default_mergefun(rng=rng), filtfun=no_shapechange, utilityfun=default_neuronselect)

Add an edge from a vertex `vi` to another vertex `vo` randomly selected from `vs = filtfun(vi)`.

Higher values of `p` will give more preference to earlier vertices of `vs`.

If `vo` is not capable of having multiple inputs (determined by `singleinput(v) == true`), `vm = mergefun(voi)` where `voi` is a randomly selected input to `vo` will be used instead of `vo` and `vo` will be added as the output of `vm`.

When selecting neurons/outputs after any eventual size change the output of `utilityfun(v)` will be used to determine the utlity of each output in vertex `v`. Note that `length(utilityfun(v)) == nout(v)` must hold.

Note: High likelyhood of large accuracy degradation after applying this mutation.
"""
struct AddEdgeMutation{F1, F2, F3, P<:Probability, RNG} <: AbstractMutation{AbstractVertex}
mergefun::F1
filtfun::F2
utilityfun::F3
p::P
rng::RNG
end
AddEdgeMutation(p; rng=rng_default, mergefun=default_mergefun(rng=rng), filtfun=no_shapechange, utilityfun=default_neuronselect) = AddEdgeMutation(Probability(p, rng), rng=rng, mergefun=mergefun, filtfun=filtfun, utilityfun=utilityfun)
AddEdgeMutation(p::Probability; rng=rng_default, mergefun=default_mergefun(rng=rng), filtfun=no_shapechange, utilityfun=default_neuronselect) = AddEdgeMutation(mergefun, filtfun, utilityfun, p, rng)

default_mergefun(pconc = 0.5; rng=rng_default, traitfun = MutationShield ∘ RemoveIfSingleInput ∘ validated() ∘ default_logging(), layerfun = ActivationContribution) = function(vin)
if rand(rng) > pconc
    return invariantvertex(layerfun(+), vin, traitdecoration=traitfun ∘ named(name(vin) * ".add"))
end
return concat(vin, traitfun = traitfun ∘ named(name(vin) * ".cat"), layerfun=layerfun)
end

function no_shapechange(vi)
# all_in_graph is not sorted, and we want some kind of topoligical order here so that earlier indices are closer to vi
allsorted = mapreduce(ancestors, vcat, filter(v -> isempty(outputs(v)), all_in_graph(vi))) |> unique

# Vertices which have the same input as vi and are singleinput
#   Reason is that this will cause a new vertex to be added between the target output vertex vo
#   and the input vertex to vi (vii) and this is detected as a size cycle which causes 
#   try_add_edge to fail.
inouts = filter(singleinput, mapreduce(outputs, vcat, inputs(vi); init=[]))
# All vertices which are after vi in the topology
vsafter = setdiff(allsorted, ancestors(vi), outputs(vi), inouts)

vitrace = shapetrace(vi) 
viorder = allΔshapetypes(vitrace)
viΔshape = squashshapes(vitrace; order=viorder)

return filter(vsafter) do vafter
    all(inputs(vafter)) do v
    t = shapetrace(v)
    vΔshape = squashshapes(t; order=union(viorder, allΔshapetypes(t)))
    return viΔshape == vΔshape
    end
end
end

function (m::AddEdgeMutation)(vi::AbstractVertex)
# All vertices for which it is allowed to add vi as an input
allverts = filter(allow_mutation, m.filtfun(vi))
isempty(allverts) && return vi

# Higher probability to select a vertex close to v is desired behaviour
# One line less than a for loop => FP wins!!
selfun(::Nothing, vc) = apply(m.p) ? vc : nothing
selfun(vs, vd) = vs
vo = foldl(selfun, allverts, init=nothing)
vo = isnothing(vo) ? rand(m.rng, allverts) : vo

try_add_edge(vi, vo, m.mergefun, m.utilityfun)
return vi
end

function try_add_edge(vi, vo, mergefun, utilityfun=default_neuronselect)

# Need to add a vertex which can handle multiple inputs if vo is single input only
# For cleaning up added vertex if the whole operation fails
cleanup_failed = () -> nothing
if singleinput(vo)
    voi = inputs(vo)[1]
    # If the input to vo is capable of multi input we don't need to create a new vertex
    # We must also check that this input does not happen to be an input to vi as this would create a cycle in the graph
    if singleinput(voi) || voi in ancestors(vi)
        vm = mergefun(voi)
        # Insert vm between voi and vo, i.e voi -> vo turns into voi -> vm -> vo
        # vs -> [vo] means only add the new vertex between voi and vo as voi could have other outputs
        insert!(voi, vv -> vm, vs -> [vo])
        cleanup_failed = function()
            length(inputs(vm)) > 1 && return
            remove!(vm, RemoveStrategy(NoSizeChange()))
        end
        vo = vm # vm is the one we shall add an edge to
        @debug "Create new vertex for merging $(name(vo))"
    else
        vo = voi
    end
end
# This is mainly because FailAlignSizeRevert does not work when the same vertex is input more than once, but it also seems kinda redundant.
vi in inputs(vo) && return
@debug "Create edge between $(name(vi)) and $(name(vo))"
create_edge!(vi, vo, strategy = create_edge_strat(vo, utilityfun))
cleanup_failed()
end
# Need to override this one for strange types e.g. layers which support exactly 2 inputs or something.
singleinput(v) = isempty(inputs(v)) || length(inputs(v)) == 1

create_edge_strat(v::AbstractVertex, utilityfun) = create_edge_strat(trait(v), utilityfun)
create_edge_strat(d::DecoratingTrait, utilityfun) = create_edge_strat(base(d), utilityfun)
function create_edge_strat(::SizeInvariant, utilityfun)
warnfailalign = FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))!")
alignstrat = AlignSizeBoth(;mapstrat=WithUtilityFun(utilityfun), fallback = warnfailalign)
# Tricky failure case: It is possible that CheckCreateEdgeNoSizeCycle does not detect any size cycle until after the edge has been created?
sizecyclewarn = FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))! Size cycle detected!") 

return CheckCreateEdgeNoSizeCycle(ifok=alignstrat, ifnok=sizecyclewarn)
end
function create_edge_strat(::SizeStack, utilityfun)
warnfailalign = FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))!")
alignstrat = PostAlign(TruncateInIndsToValid(WithUtilityFun(utilityfun, AlignNinToNout(;fallback=ΔSizeFailNoOp()))), fallback=warnfailalign)

sizecyclewarn = FailAlignSizeWarn(msgfun = (vin,vout) -> "Could not align sizes of $(name(vin)) and $(name(vout))! Size cycle detected!")
return CheckCreateEdgeNoSizeCycle(ifok=alignstrat, ifnok=sizecyclewarn)
end

"""
RemoveEdgeMutation <: AbstractMutation{AbstractVertex}
RemoveEdgeMutation(;utilityfun=default_neuronselect, rng=rng_default)

Remove an edge from a vertex `vi` to another vertex `vo` randomly selected from `outputs(vi)`.

Vertex `vi` must have more than one output and vertex `vo` must have more than one output for the edge to be removed. Otherwise no change is made.

If there are multiple edges between `vi` and `vo` no change will be made due to NaiveNASlib not being able to revert a failed operation in this case..

When selecting neurons/outputs after any eventual size change the output of `utilityfun(v)` will be used to determine the utlity of each output in vertex `v`. Note that `length(utilityfun(v)) == nout(v)` must hold.

Note: High likelyhood of large accuracy degradation after applying this mutation.
"""
struct RemoveEdgeMutation{F, RNG<:AbstractRNG} <: AbstractMutation{AbstractVertex}
utilityfun::F
rng::RNG
end
RemoveEdgeMutation(;utilityfun=default_neuronselect, rng=rng_default) = RemoveEdgeMutation(utilityfun, rng)

function (m::RemoveEdgeMutation)(vi::AbstractVertex)
length(outputs(vi)) < 2 && return vi

allverts = filter(vo -> length(inputs(vo)) > 1, outputs(vi))

isempty(allverts) && return vi

vo = rand(m.rng, allverts)
sum(inputs(vo) .== vi) > 1 && return vi# Not implemented in NaiveNASlib

@debug "Remove edge between $(name(vi)) and $(name(vo))"
remove_edge!(vi, vo, strategy=remove_edge_strat(vo, m.utilityfun))
return vi
end

remove_edge_strat(v::AbstractVertex, utilityfun) = remove_edge_strat(trait(v), utilityfun)
remove_edge_strat(d::DecoratingTrait, utilityfun) = remove_edge_strat(base(d), utilityfun)
remove_edge_strat(::SizeInvariant, utilityfun) = NoSizeChange()
remove_edge_strat(t::SizeStack, utilityfun) = create_edge_strat(t, utilityfun)

"""
KernelSizeMutation{N} <: AbstractMutation{AbstractVertex}
KernelSizeMutation(Δsizespace::AbstractParSpace{N, Int}; maxsize, pad, rng)
KernelSizeMutation2D(absΔ::Integer;maxsize, pad, rng)
KernelSizeMutation(absΔ::Integer...;maxsize, pad, rng)

Mutate the size of filter kernels of convolutional layers.

Note: High likelyhood of large accuracy degradation after applying this mutation.

`KernelSizeMutation2D` is a convenience constructor for `KernelSizeMutation(absΔ, absΔ;...)`.
"""
struct KernelSizeMutation{N,F,P} <: AbstractMutation{AbstractVertex}
Δsizespace::AbstractParSpace{N, Int}
maxsize::F
pad::P
rng::AbstractRNG
end
KernelSizeMutation(Δsizespace::AbstractParSpace{N, Int}; maxsize = v -> ntuple(i->Inf,N), pad=SamePad(), rng=rng_default) where N = KernelSizeMutation(Δsizespace, maxsize, pad, rng)
KernelSizeMutation2D(absΔ::Integer;maxsize = v -> (Inf,Inf), pad=SamePad(), rng=rng_default) = KernelSizeMutation(absΔ, absΔ, maxsize = maxsize, pad=pad, rng=rng)
KernelSizeMutation(absΔ::Integer...;maxsize = v -> ntuple(i->Inf, length(absΔ)), pad=SamePad(), rng=rng_default) = KernelSizeMutation(ParSpace(UnitRange.(.-absΔ, absΔ));maxsize = maxsize, pad=pad, rng=rng)

function (m::KernelSizeMutation{N})(v::AbstractVertex) where N
layertype(v) isa FluxConvolutional{N} || return
l = layer(v)

currsize = size(NaiveNASflux.weights(l))[1:N]
Δsize = Int.(clamp.(m.Δsizespace(m.rng), 1 .- currsize, m.maxsize(v) .- currsize)) # ensure new size is > 0 and < maxsize
# This will eventually boil down to Setfield doing its thing, and that won't be using any convenience constructors
pad = Flux.calc_padding(typeof(l), m.pad, currsize .+ Δsize, dilation(l), stride(l))
KernelSizeAligned(Δsize, pad)(v)
return v
end
dilation(l) = l.dilation
stride(l) = l.stride

"""
ActivationFunctionMutation{T,R} <: AbstractMutation{AbstractVertex} where {T <: AbstractParSpace{1}, R <: AbstractRNG}
ActivationFunctionMutation(actspace::AbstractParSpace{1}, rng::AbstractRNG)
ActivationFunctionMutation(acts...;rng=rng_default)
ActivationFunctionMutation(acts::AbstractVector;rng=rng_default)

Mutate the activation function of layers which have an activation function.

Note: High likelyhood of large accuracy degradation after applying this mutation.
"""
struct ActivationFunctionMutation{T,RNG} <: AbstractMutation{AbstractVertex} where {T <: AbstractParSpace{1}, R <: AbstractRNG}
actspace::T
rng::RNG
end
ActivationFunctionMutation(acts...;rng=rng_default) = ActivationFunctionMutation(collect(acts), rng=rng)
ActivationFunctionMutation(acts::AbstractVector;rng=rng_default) = ActivationFunctionMutation(ParSpace(acts), rng)

function (m::ActivationFunctionMutation)(v::AbstractVertex)
m(layertype(v), v)
return v
end
function (m::ActivationFunctionMutation)(t, v) end
(m::ActivationFunctionMutation)(::Union{FluxDense, FluxConvolutional}, v) = NaiveNASflux.setlayer!(v, (σ = m.actspace(m.rng),))
(m::ActivationFunctionMutation)(::FluxParNorm, v) = NaiveNASflux.setlayer!(v, (λ = m.actspace(m.rng),))
function (m::ActivationFunctionMutation)(::FluxRnn, v)
newcell = setproperties(layer(v).cell, (σ = m.actspace(m.rng),))
NaiveNASflux.setlayer!(v, (cell = newcell,))
end


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