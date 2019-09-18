struct CandidateModel
    model::CompGraph
    opt
    lossfun
end

"""
    AbstractFitness

Abstract type for fitness functions
"""
abstract type AbstractFitness end

"""
    AbstractFunLabel

Puts a label on a function for use with [`instrument`](@ref).
"""
abstract type AbstractFunLabel end
struct Train <: AbstractFunLabel end
struct Validate <: AbstractFunLabel end

"""
    instrument(l::AbstractFunLabel, s::AbstractFitness, f::Funtion)

Instrument `f` labelled `l` for fitness measurement `s`.

Example is to store the result of `f` for fitness calculation, or to add a measurement of the average time it takes to evalute as used with [`TimeFitness`](@ref)..
"""
instrument(::AbstractFunLabel, ::AbstractFitness, f::Function) = f

"""
    AccuracyFitness <: AbstractFitness
    AccuracyFitness(dataset)

Measure fitness as the accuracy on a dataset.
"""
struct AccuracyFitness <: AbstractFitness
    dataset
end
function fitness(s::AccuracyFitness, f)
    acc,cnt = 0, 0
    for (x,y) in s.dataset
        acc += mean(Flux.onecold(f(x)) .== Flux.onecold(y))
        cnt += 1
    end
    return acc / cnt
end

"""
    MapFitness <: AbstractFitness
    MapFitness(mapping::Function, base::AbstractFitness)

Maps fitness `x` from `base` to `mapping(x)`.
"""
struct MapFitness <: AbstractFitness
    mapping::Function
    base::AbstractFitness
end
fitness(s::MapFitness, f) = fitness(s.base, f) |> s.mapping
instrument(l::AbstractFunLabel,s::MapFitness,f::Function) = instrument(l, s.base, f)

"""
    TimeFitness{T} <: AbstractFitness where T <: AbstractFunLabel
    TimeFitness()

Measure fitness as time to evaluate a function.

Function needs to be instrumented using [`instrument`](@ref).
"""
mutable struct TimeFitness{T} <: AbstractFitness where T <: AbstractFunLabel
    totaltime
    neval
end
TimeFitness(t::T) where T = TimeFitness{T}(0, 0)
fitness(s::TimeFitness, f) = s.totaltime / s.neval

function instrument(::T, s::TimeFitness{T}, f::Function) where T <: AbstractFunLabel
    return function(x...)
        res, t = @timed f(x...)
        s.totaltime += t
        s.neval += 1
        return res
    end
end
