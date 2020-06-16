
abstract type ΔShape{N,M} end

shapeΔ(s::ΔShape) = s.Δshape
indims(s::ΔShape{N,M}) where {N,M} = N
outdims(s::ΔShape{N,M}) where {N,M} = M

struct ShapeMul{N} <: ΔShape{N,N}
    Δshape::NTuple{N, <:Integer}
end
ShapeMul(Δshape::Integer...) = ShapeMul(Δshape)
fshape(s::ShapeMul{N}, shape::NTuple{N, <:Integer}) where N = shape .* shapeΔ(s)

struct ShapeDiv{N} <: ΔShape{N,N}
    Δshape::NTuple{N, <:Integer}
end
ShapeDiv(Δshape::Integer...) = ShapeDiv(Δshape)
fshape(s::ShapeDiv{N}, shape::NTuple{N, T}) where {N, T<:Integer} = ceil.(T, shape ./ shapeΔ(s))


struct ShapeAdd{N} <: ΔShape{N,N}
    Δshape::NTuple{N, <:Integer}
end
ShapeAdd(Δshape::Integer...) = ShapeAdd(Δshape)
fshape(s::ShapeAdd{N}, shape::NTuple{N, <:Integer}) where N = shape .+ shapeΔ(s)


fshape(s::Tuple{ΔShape{N}, Vararg{ΔShape}}, shape::NTuple{N, <:Integer}) where N = foldr(fshape, reverse(s); init=shape)


revert(s::ShapeAdd) = ShapeAdd(.-shapeΔ(s))
revert(s::ShapeMul) = ShapeDiv(shapeΔ(s))
revert(s::ShapeDiv) = ShapeMul(shapeΔ(s))
revert(s::NTuple{N, ΔShape}) where N = reverse(revert.(s))

combine(s1,s2) = s1,s2
combine(s1::Tuple{Vararg{ΔShape}}, s2) = (s1[1:end-1]..., combine(last(s1), s2)...)
combine(s1::ΔShape, s2::Tuple{Vararg{ΔShape}}) = (combine(s1, first(s2))..., s2[2:end]...)
combine(s1::ShapeAdd{N}, s2::ShapeAdd{N}) where N = tuple(ShapeAdd(shapeΔ(s1) .+ shapeΔ(s2)))
combine(s1::T, s2::T) where T <: Union{ShapeDiv{N}, ShapeMul{N}} where N = tuple(T(shapeΔ(s1) .* shapeΔ(s2)))
# Note: Combining ShapeDiv and ShapeMul not safe due to truncation when dividing
combine(s1::ShapeMul{N}, s2::ShapeDiv{N}) where N = isdiv(s1, s2) ? tuple(ShapeMul(shapeΔ(s1) .÷ shapeΔ(s2))) : (s1,s2)

isdiv(s1::ΔShape{N,N},s2::ΔShape{N,N}) where N = all(iszero, shapeΔ(s1) .% shapeΔ(s2))


swapΔshape(s1, s2) = s1,s2
swapΔshape(s1::T, s2::T) where T <: ΔShape = s2,s1

swapΔshape(s1::ShapeAdd{N}, s2::ShapeMul{N}) where N = s2, ShapeAdd(shapeΔ(s2) .* shapeΔ(s1))
swapΔshape(s1::ShapeMul{N}, s2::ShapeAdd{N}) where N = isdiv(s2, s1) ? (ShapeAdd(shapeΔ(s2) .÷ shapeΔ(s1)), s1) : (s1,s2)

swapΔshape(s1::ShapeAdd{N}, s2::ShapeDiv{N}) where N = isdiv(s1, s2) ? (s2, ShapeAdd(shapeΔ(s1) .÷ shapeΔ(s2))) : (s1,s2)
swapΔshape(s1::ShapeDiv{N}, s2::ShapeAdd{N}) where N = ShapeAdd(shapeΔ(s1) .* shapeΔ(s2)), s1


filter_noops(ss::ΔShape...) = filter_noops(ss)
filter_noops(ss::Tuple{Vararg{ΔShape}}) = mapreduce(filter_noops, (s1,s2) -> (s1...,s2...), ss)
filter_noops(s::Union{ShapeMul, ShapeDiv}) = all(x -> x == 1, shapeΔ(s)) ? tuple() : tuple(s)
filter_noops(s::ShapeAdd) = all(x -> x == 0, shapeΔ(s)) ? tuple() : tuple(s)

function orderΔshapes(s::Tuple{Vararg{ΔShape}}; order=unique(typeof.(s))) where N
    # Yeah, this is bubble sort :/ Given the constraint that ΔShapes can't always swap places along with the fact that swapping generally changes the swapped elements I couldn't think up and other sorting algo that works
    sprev = tuple()
    snew = s
    nlook = length(s)
    while sprev != snew
        sprev = snew
        nlook -= 1
        for i in 1:nlook
            s1,s2 = snew[i], snew[i+1]
            # Check if s1 shall be before s2 in the ordering
            # Note: We need to "bubble" on equality because s1 might be prevented from bubbling up while s2 isn't.
            if findfirst(st -> s1 isa st, order) >= findfirst(st -> s2 isa st, order)
                snew = (snew[1:i-1]..., swapΔshape(s1,s2)..., snew[i+2:end]...)
            end
        end
    end
    return snew
end

squashshapes(s::ΔShape) = tuple(s)
squashshapes(s::ΔShape...) = squashshapes(s)
squashshapes(s::Tuple{Vararg{ΔShape}}) where N = _squashshapes(orderΔshapes(s))

_squashshapes(s::ΔShape) = tuple(s)
_squashshapes(s::Tuple{ΔShape}) = s
function _squashshapes(s::Tuple{Vararg{ΔShape}})
    squashed = filter_noops(foldr(combine, s)...)
    squashed == s && return s
    isempty(squashed) && return squashed
    return _squashshapes(squashed)
end


abstract type AbstractShapeTraceX end

struct ShapeTraceV0{T,V1,V2} <: AbstractShapeTraceX
    origin::V1
    dest::V2
    trace::T
end
ShapeTraceV0(v) = ShapeTraceV0(v, v, tuple())

function add_op(tr::ShapeTraceV0, v, Δshapes...)
    Δshapes_valid = filter_noops(Δshapes...)
    return ShapeTraceV0(tr.origin, v, (tr.trace..., Δshapes_valid...))
end

Base.merge(v, trs::ShapeTraceV0...) = ShapeTraceV0(v, v, tuple(tuple((ShapeTraceV0(t.origin, v, t.trace) for t in trs)...)))

shapepaths(t::ShapeTraceV0) = t.origin => shapepaths.(t.trace)
shapepaths(x) = x

squashshapes(t::ShapeTraceV0) = squashshapes(t.trace)
squashshapes(t::Tuple) = mapfoldl(squashshapes, squashshapes, t)
function squashshapes(t::Tuple{Vararg{ShapeTraceV0}})
     squashed = unique(map(squashshapes, t))
     length(squashed) == 1 && return first(squashed)
     return Tuple(squashed) # Danger danger! Graph probably only works for one single input shape
end
squashshapes(t1::ShapeTraceV0, t2::ShapeTraceV0) = t1,t2 # Can squash?
squashshapes(t::ShapeTraceV0, s::Tuple{Vararg{ΔShape}}) = squashshapes(t.trace, s)
squashshapes(s::Tuple{Vararg{ΔShape}}, t::ShapeTraceV0) = squashshapes(s, t.trace)
squashshapes(s::Tuple{Vararg{ΔShape}}, t::Tuple{Vararg{ShapeTraceV0}}) = s,t # Danger danger! Graph probably only works for one single input shape
squashshapes(t::Tuple{Vararg{ShapeTraceV0}}, s::Tuple{Vararg{ΔShape}}) = t,s # Danger danger! Graph probably only works for one single input shape
squashshapes(s1::Tuple{Vararg{ΔShape}}, s2::Tuple{Vararg{ΔShape}}) = squashshapes((s1...,s2...))




function shapetrace(v::AbstractVertex; trfun = v -> ShapeTraceV0(v))
    ins = filter(v -> isempty(inputs(v)), NaiveNASlib.flatten(v))
    memo = Dict{AbstractVertex, Any}(ins .=> trfun.(ins))
    return output!(memo, v)
end

function shapetrace(v::AbstractVertex, vs::AbstractVertex...; trfun = v -> ShapeTraceV0(v))
    memo = Dict{AbstractVertex, Any}(vs .=> trfun.(vs))
    return output!(memo, v)
end


(v::NaiveNASlib.AbstractVertex)(trs::AbstractShapeTraceX...) = base(v)(trs...)
(v::NaiveNASlib.MutationVertex)(trs::AbstractShapeTraceX...) = shapequery(trait(v), v, trs...)

shapequery(t::DecoratingTrait, v, trs...) = shapequery(base(t), v, trs...)
# TODO: Not correct in general cases. Just a temp thing until I can get a picture of whether this is whole thing works out or not
shapequery(::NaiveNASlib.SizeTransparent, v, trs...) = merge(v, trs...)

shapequery(::SizeAbsorb, v, trs...) = shapequery(layertype(v), v, trs...)
shapequery(lt, v, trs...) = first(trs)
function shapequery(::FluxConv{N}, v, tr) where N
    c = layer(v)
    ks = size(NaiveNASflux.weights(c))[1:N]
    Δwindow = Δshape_from_window(ks, c.dilation, c.pad)
    Δstride = ShapeDiv(c.stride)
    add_op(tr, v, Δwindow, Δstride)
end

function Δshape_from_window(ws::NTuple{N}, dilation, pad) where N
    padact = length(pad) == N ? 2 .* pad : ntuple(i -> pad[2(i-1)+1] + pad[2(i-1)+2], N)
    padref = ntuple(i -> sum(SamePad()(ws[i], dilation[i])), N)
    return ShapeAdd(padact .- padref)
end
