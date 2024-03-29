# General disclaimer: This whole little mini-lib feels like a poor mans implementation of basic symbolic arithmetics and therefore it is hidden in here so I can just delete it silently when I finally realize how this should really be handled.
# It probably looks like it is capable of much more than it actually can do so if you stumble upon this, use with care. Any bug report are of course still extremely welcome, especially the kind which would motivate scrapping the whole thing in favour of a real solution.

"""
    ΔShape{N,M}

Abstract type describing how an operation changes the shape of the input.

Designed with convolutional and pooling layers in mind to deal with how parameters like kernel size, padding and stride affect the output size.

Typically has nothing to do with the channel or batch dimensions which are handled through other means.

`N` and `M` are integers and represent the number of dimensions of the input and output shape respectively.

For example, the ΔShapes describing a 2D convolutional layer has `N == M == 2` and only describes how the shape of the feature maps change.
"""
abstract type ΔShape{N,M} end

shapeΔ(s::ΔShape) = s.Δshape
ndimsin(::ΔShape{N,M}) where {N,M} = N
ndimsout(::ΔShape{N,M}) where {N,M} = M

ndimsin(s::Tuple{ΔShape, Vararg{ΔShape}}) = ndimsin(first(s))
ndimsout(s::Tuple{ΔShape, Vararg{ΔShape}}) = ndimsout(last(s))

"""
    fshape(s::ΔShape{N,M}, shape::Tuple{N,Integer})
    fshape(s::Tuple{ΔShape{N}, Vararg{ΔShape}}, shape::NTuple{N, Integer})

Return the output shape after passing input of shape `shape` through `s`.
"""
function fshape end

"""
    ShapeMul{N} <: ΔShape{N,N}
    ShapeMul(Δshape)
    ShapeMul(Δshape::Int...)

Size is multiplied by `Δshape`.
"""
struct ShapeMul{N} <: ΔShape{N,N}
    Δshape::NTuple{N, Int}
end
ShapeMul(Δshape::Int...) = ShapeMul(Δshape)
fshape(s::ShapeMul{N}, shape::NTuple{N, Integer}) where N = shape .* shapeΔ(s)

"""
    ShapeDiv{N} <: ΔShape{N,N}
    ShapeDiv(Δshape)
    ShapeDiv(Δshape::Int...)

Size is divided by `Δshape`, rounding up fractions.
"""
struct ShapeDiv{N} <: ΔShape{N,N}
    Δshape::NTuple{N, Int}
end
ShapeDiv(Δshape::Int...) = ShapeDiv(Δshape)
fshape(s::ShapeDiv{N}, shape::NTuple{N, T}) where {N, T<:Integer} = ceil.(T, shape ./ shapeΔ(s))

"""
    ShapeAdd{N} <: ΔShape{N,N}
    ShapeAdd(Δshape)
    ShapeAdd(Δshape::Int...)

Size is added to `Δshape`.
"""
struct ShapeAdd{N} <: ΔShape{N,N}
    Δshape::NTuple{N, Int}
end
ShapeAdd(Δshape::Int...) = ShapeAdd(Δshape)
fshape(s::ShapeAdd{N}, shape::NTuple{N, Integer}) where N = shape .+ shapeΔ(s)

"""
    ShapeFlatten{N} <: ΔShape{N, 0}

Flattens shape so it no longer exists.

Note that `N` can be `Any` since flatten operations typically don't care about the number of input dimensions.
"""
struct ShapeFlatten{N} <: ΔShape{N, 0} end
ShapeFlatten() = ShapeFlatten{Any}()
fshape(::ShapeFlatten, shape) = tuple()

"""
    RevertΔShape{N,M,T<:ΔShape{N,M}} <: ΔShape{N,M}
    RevertΔShape(Δshape)

Revert operation of `Δshape` typically used when the revert operation is undefined.
"""
struct RevertΔShape{N,M,T<:ΔShape{N,M}} <: ΔShape{N,M}
    Δshape::T
end

fshape(::Tuple{}, shape) = shape
fshape(s::Tuple{ΔShape{N}, Vararg{ΔShape}}, shape::NTuple{N, Integer}) where N = foldr(fshape, reverse(s); init=shape)
fshape(s::AbstractVector{<:ΔShape}, shape::NTuple{N, Integer}) where N = foldr(fshape, reverse(s); init=shape)

"""
    revert(s::ΔShape)
    revert(s::AbstractVector{<:ΔShape}) 

Return a `ΔShape` or tuple of `ΔShape`s which reverts the shape change of `s`, i.e `fshape((s..., revert(s)...), x) == x`.

Warning: Not guaranteed to produce exact results for all `ΔShape` types! For example, the operation performed by `ShapeDiv` is not generally invertible due to rounding.
"""
revert(s) = RevertΔShape(s)
revert(s::RevertΔShape) = shapeΔ(s)
revert(s::ShapeAdd) = ShapeAdd(.-shapeΔ(s))
revert(s::ShapeMul) = ShapeDiv(shapeΔ(s))
revert(s::ShapeDiv) = ShapeMul(shapeΔ(s))
revert(s::Tuple{Vararg{ΔShape}}) = reverse(revert.(s))
revert(s::AbstractVector{<:ΔShape}) = reverse(revert.(s))

"""
    combine(s1::ΔShape,s2::ΔShape)

Return a `ΔShape` or tuple of `ΔShape`s which combines `s1` and `s2`, i.e `fshape((s1,s2), x) == fshape(combine(s1,s2), x)`
"""
combine(s1::ΔShape,s2::ΔShape) = s1,s2
combine(s1::Tuple{Vararg{ΔShape}}, s2::ΔShape) = vcat(collect(s1[1:end-1]), combine(last(s1), s2)...)
combine(s1::ΔShape, s2::Tuple{Vararg{ΔShape}}) = vcat(combine(s1, first(s2))..., collect(s2[2:end]))
combine(s1::AbstractVector{<:ΔShape}, s2::ΔShape) = vcat(s1[1:end-1], combine(last(s1), s2)...)
combine(s1::ΔShape, s2::AbstractVector{<:ΔShape}) = vcat(combine(s1, first(s2))..., s2[2:end])
combine(s1::ShapeAdd{N}, s2::ShapeAdd{N}) where N = tuple(ShapeAdd(shapeΔ(s1) .+ shapeΔ(s2)))
combine(s1::T, s2::T) where T <: Union{ShapeDiv{N}, ShapeMul{N}} where N = tuple(T(shapeΔ(s1) .* shapeΔ(s2)))
# Note: Combining ShapeDiv and ShapeMul not generally safe due to rounding when dividing
combine(s1::ShapeMul{N}, s2::ShapeDiv{N}) where N = isdiv(s1, s2) ? tuple(ShapeMul(shapeΔ(s1) .÷ shapeΔ(s2))) : (s1,s2)

isdiv(s1::ΔShape{N,N},s2::ΔShape{N,N}) where N = all(iszero, shapeΔ(s1) .% shapeΔ(s2))

"""
    swapΔshape(s1, s2)

Return a tuple of `ΔShape`s where `s1` and `s2` have swapped places without changing the shape mapping, i.e `fshape((s1,s2), x) == fshape(swapΔshape(s1,s2), x)`.

Basically a helper function for [`orderΔshapes`](@ref). Probably not useful on its own.
"""
swapΔshape(s1, s2) = s1,s2
swapΔshape(s1::T, s2::T) where T <: ΔShape = s2,s1

swapΔshape(s1::ShapeAdd{N}, s2::ShapeMul{N}) where N = s2, ShapeAdd(shapeΔ(s2) .* shapeΔ(s1))
swapΔshape(s1::ShapeMul{N}, s2::ShapeAdd{N}) where N = isdiv(s2, s1) ? (ShapeAdd(shapeΔ(s2) .÷ shapeΔ(s1)), s1) : (s1,s2)

swapΔshape(s1::ShapeAdd{N}, s2::ShapeDiv{N}) where N = isdiv(s1, s2) ? (s2, ShapeAdd(shapeΔ(s1) .÷ shapeΔ(s2))) : (s1,s2)
swapΔshape(s1::ShapeDiv{N}, s2::ShapeAdd{N}) where N = ShapeAdd(shapeΔ(s1) .* shapeΔ(s2)), s1

"""
    filter_noops(s::ΔShape)
    filter_noops(s::ΔShape...)
    filter_noops(s::Tuple{Vararg{ΔShape}})
    filter_noops(s::AbstractVector{<:ΔShape}) 
    filter_noops(s::ΔShape)

Return a tuple of `ΔShape`s where all identity mappings (e.g things like `ShapeAdd(0)`) are removed.

If called with a single identity mapping an empty tuple is returned.
"""
filter_noops(s::ΔShape) = tuple(s)
filter_noops(s::ΔShape...) = filter_noops(s)
filter_noops(s::Tuple{Vararg{ΔShape}}) = mapreduce(filter_noops, (s1,s2) -> (s1...,s2...), s; init=tuple())
filter_noops(s::Union{ShapeMul, ShapeDiv}) = all(x -> x == 1, shapeΔ(s)) ? tuple() : tuple(s)
filter_noops(s::ShapeAdd) = all(x -> x == 0, shapeΔ(s)) ? tuple() : tuple(s)

"""
    filter_noops(s::AbstractVector{<:ΔShape}) 

Return an array of `ΔShape`s where all identity mappings (e.g things like `ShapeAdd(0)`) are removed.

If called with a single identity mapping an empty array is returned.
"""
filter_noops(s::AbstractVector{<:ΔShape}) = mapreduce(filter_noops, (s1,s2) -> vcat(s1...,s2...), s; init=similar(s, 0))

"""
    orderΔshapes(s::AbstractVector{<:ΔShape}; order=allΔshapetypes(s))

Return a tuple of `ΔShape`s which has the same shape mapping as `s` (i.e `fshape(s, x) == fshape(orderΔshapes(s), x)`) but where `ΔShape`s to the extent possible are ordered according to `order`.

Useful to determine whether two arbitrary sequences of `ΔShape`s result in the same shape mapping for all shapes.

Warning: Sort is not stable due to lazy implementation, i.e `orderΔshapes(orderΔshapes(s; order=someorder);order=someorder)` is not guaranteed to return the same thing as `orderΔshapes(s; order=someorder)`.
"""
orderΔshapes(s::AbstractVector{<:ΔShape}; order=allΔshapetypes(s)) = orderΔshapes!(copy(s); order)
function orderΔshapes!(s::AbstractVector{<:ΔShape}; order=allΔshapetypes(s))
    # Yeah, this is bubble sort :/ Given the constraint that ΔShapes can't always swap places along with the fact that swapping generally changes the swapped elements I couldn't think up and other sorting algo that works
    nlook = length(s)

    atleastoneswap = true

    while atleastoneswap
        nlook -= 1
        atleastoneswap = false
        for i in 1:nlook
            s1,s2 = s[i], s[i+1]
            # Check if s1 shall be before s2 in the ordering
            # Note: We need to "bubble" on equality because s1 might be prevented from bubbling up while s2 isn't.
            if findfirst(st -> s1 isa st, order) >= findfirst(st -> s2 isa st, order)
                s1n, s2n = swapΔshape(s1, s2)
                atleastoneswap |= s1n != s1 
                s[i] = s1n
                s[i+1] = s2n
            end
        end
    end
    return s
end

allΔshapetypes(::T) where T <: ΔShape = T
allΔshapetypes(s::Tuple{Vararg{ΔShape}}) = unique(allΔshapetypes.(s))
allΔshapetypes(s::AbstractVector{<:ΔShape}) = unique(allΔshapetypes.(s))

"""
    squashshapes(s::ΔShape...; order=allΔshapetypes(s))
    squashshapes(s::AbstractVector{<:ΔShape}; order=allΔshapetypes(s))

Return an array of `ΔShape`s with the same shape mapping as `s` (i.e `fshape(s, x) == fshape(squashshapes(s), x)`) with as few `ΔShape`s as possible for the given `order`.

Useful to determine whether two arbitrary sequences of `ΔShape`s result in the same shape mapping for all shapes.
"""
squashshapes(s::ΔShape; kwargs...) = [s]
squashshapes(s::ΔShape...; order=allΔshapetypes(s)) = squashshapes(s; order)
squashshapes(s::Tuple{ΔShape, Vararg{ΔShape}}; order=allΔshapetypes(s)) = squashshapes(collect(s); order)
squashshapes(s::AbstractVector{<:ΔShape}; order=allΔshapetypes(s)) = _squashshapes(orderΔshapes(s; order=order))

_squashshapes(s::ΔShape) = [s]
function _squashshapes(s::AbstractVector{<:ΔShape})
    isempty(s) && return s
    squashed = collect(filter_noops(foldr(combine, s)))
    squashed == s && return s
    isempty(squashed) && return similar(s, 0)
    return _squashshapes(squashed)
end

Δshapediff(s1,s2) = filter_noops(squashshapes(_Δshapediff(s1,s2)))
_Δshapediff(s1::ΔShape{N}, s2::ΔShape{M}) where {N,M} = N == M ? [revert(s2), s1] : [s1,s2]
_Δshapediff(s1::ΔShape{N}, s2::ΔShape{N}) where N = s1 == s2 ? typeof(s1)[] : [revert(s2), s1]
function _Δshapediff(s1::AbstractVector{<:ΔShape}, s2::AbstractVector{<:ΔShape})
    # Pretty crappy heurisic tbh, but I couldn't think of anything better:
    # Step 1: Remove all identical ΔShapes
    # Step 2: Squash shapes and try again
    # Step 3: revert s2 and concat s1

    firstdiff = findfirst(((ss1,ss2),) -> ss1 != ss2, collect(zip(s1,s2)))
    firstdiff = isnothing(firstdiff) ? min(length(s1), length(s2))+1 : firstdiff
    sd1 = s1[firstdiff:end]
    sd2 = s2[firstdiff:end]

    isempty(sd1) && isempty(sd2) && return similar(s1, 0)

    ts1 = allΔshapetypes(sd1)
    ts2 = allΔshapetypes(sd2)
    front = intersect(ts1,ts2)
    back = symdiff(ts1, ts2)

    so1 = squashshapes(sd1; order=vcat(front, back))
    so2 = squashshapes(sd2; order=vcat(front, back))

    (so1 != s1 || so2 != s2) && return _Δshapediff(so1, so2)
    return vcat(revert(so2), so1)
end

"""
    AbstractShapeTrace

Abstract type for tracing shape properties of a graph.

A `MutationVertex v` called with an `AbstractShapeTrace tr` as input will call `visitvertex(tr, v)` or `merge(v, trs...)` if called with more than one `AbstractShapeTrace`.
"""
abstract type AbstractShapeTrace end

"""
    ShapeTrace{T,V1,V2} <: AbstractShapeTrace

Records `ΔShape`s of any visited vertices as well as the origin and destination vertex.
"""
struct ShapeTrace{T,V1,V2} <: AbstractShapeTrace
    origin::V1
    dest::V2
    trace::T
end
ShapeTrace(v) = ShapeTrace(v, v, collect(Δshapes(v)))

allΔshapetypes(t::ShapeTrace) = allΔshapetypes(t.trace)
allΔshapetypes(t::Union{Tuple, <:AbstractArray}) = unique(mapreduce(allΔshapetypes, vcat, t))

squashshapes(t::ShapeTrace; order=allΔshapetypes(t)) = squashshapes(t.trace;order=order)
# TODO: One can probably do better here when parallel paths can't be squashed
# Now the first instance of such a path will basically prevent squashing of any subsequent paths, even if they are not parallel
squashshapes(t::AbstractArray; order=allΔshapetypes(t)) = mapfoldr(tt -> squashshapes(tt;order=order), (t1,t2) -> squashshapes(t1,t2; order=order), t)
function squashshapes(t::AbstractArray{<:ShapeTrace}; order=allΔshapetypes(t))
     squashed = unique(map(tt -> squashshapes(tt;order=order), t))
     length(squashed) == 1 && return only(squashed)
     return squashed # Danger danger! Graph probably only works for one single input shape
end
# This is the reason for "TODO: One can probably do better here when parallel paths can't be squashed" above
squashshapes(s1, s2; order=missing) = [s1, s2]
squashshapes(s1::AbstractVector{<:ΔShape}, s2::AbstractVector{<:ΔShape}; order=allΔshapetypes((s1,s2))) = squashshapes(vcat(s1,s2); order=order)


visitvertex(tr::ShapeTrace, v) = ShapeTrace(tr.origin, v, vcat(tr.trace, Δshapes(v)...))

Base.merge(::AbstractVertex, tr::ShapeTrace) = tr
Base.merge(v::AbstractVertex, trs::ShapeTrace...) = ShapeTrace(v, v, [[(ShapeTrace(t.origin, v, vcat(t.trace, Δshapes(v)...)) for t in trs)...]])

"""
    shapetrace(v::AbstractVertex, vs::AbstractVertex...; trfun = v -> ShapeTrace(v))

Return a `AbstractShapeTrace` (default `ShapeTrace`) between `v` and `vs` where `vs` must be input ascendants to `v`. If `vs` is omitted then all input vertices will be used.
"""
function shapetrace(v::AbstractVertex; trfun = v -> ShapeTrace(v))
    ins = filter(v -> isempty(inputs(v)), ancestors(v))
    memo = Dict{AbstractVertex, Any}(ins .=> trfun.(ins))
    return output!(memo, v)
end

function shapetrace(v::AbstractVertex, vs::AbstractVertex...; trfun = v -> ShapeTrace(v))
    memo = Dict{AbstractVertex, Any}(vs .=> trfun.(vs))
    return output!(memo, v)
end

(v::NaiveNASlib.MutationVertex)(trs::AbstractShapeTrace...) = merge(v, trs...)
(v::NaiveNASlib.MutationVertex)(tr::AbstractShapeTrace) = visitvertex(tr, v)

"""
    Δshapes(v::AbstractVertex)

Return a tuple of `ΔShape`s describing the shape mapping of `v`.

More concretely, if `xs = size(x)[sdims]` then `size(v(x))[sdims] == fshape(Δshapes(v), xs)` where `sdims` are the shape dimensions of `x`, e.g. the height and width in case of 2D convolutions.
"""
Δshapes(v::AbstractVertex) = Δshapes(base(v))
Δshapes(::InputVertex) = ΔShape[]
Δshapes(v::MutationVertex) = Δshapes(trait(v), v)
Δshapes(t::DecoratingTrait, v) = Δshapes(base(t), v)
Δshapes(::MutationSizeTrait, v) = _Δshapes(layertype(v), v)

_Δshapes(t::Any, v) = ΔShape[]

function _Δshapes(::FluxConv{N}, v) where N
    c = layer(v)
    ks = size(NaiveNASflux.weights(c))[1:N]
    Δwindow = Δshape_from_window(c, ks, c.dilation, c.pad)
    Δstride = ShapeDiv(c.stride)
    return Δwindow, Δstride
end

_Δshapes(::FluxNoParLayer, v) = _Δshapes(layer(v), v)
_Δshapes(p::Union{MeanPool, MaxPool}, v) = Δshape_from_window(p, p.k, 1, p.pad), ShapeDiv(p.stride)
_Δshapes(::Type{typeof(Flux.flatten)}, v) = _Δshapes_gp(NaiveNASflux.actdim(v) .- 2)
_Δshapes(::GlobalPool, v) = _Δshapes_gp(NaiveNASflux.actdim(v) .- 2)

Δshape_from_window(l::LT, ws::NTuple{N}, dilation::Integer, pad) where {N, LT} = Δshape_from_window(l, ws, ntuple(i -> dilation, N), pad)
function Δshape_from_window(::LT, ws::NTuple{N}, dilation, pad) where {LT, N}
    padact = length(pad) == N ? 2 .* pad : ntuple(i -> pad[2(i-1)+1] + pad[2(i-1)+2], N)
    padref = ntuple(i -> sum(Flux.calc_padding(LT, SamePad(), tuple(ws[i]), dilation[i], 1)), N)
    return ShapeAdd(padact .- padref)
end

function _Δshapes_gp(shapedims)
    udims = unique(shapedims)
    # Graphs with different shapedims are probably not correct, but there is always the chance that some op in between which fixes this does not have a defined ΔShape.
    length(udims) != 1 && return tuple(ShapeFlatten())
    return tuple(ShapeFlatten{first(udims)}())
end
