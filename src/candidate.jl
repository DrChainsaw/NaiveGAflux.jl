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

You probably want to use this with a `FitnessCache`.
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
TimeFitness(t::T) where T = TimeFitness{T}(0.0, 0)
fitness(s::TimeFitness, f) = s.totaltime / s.neval

function instrument(::T, s::TimeFitness{T}, f::Function) where T <: AbstractFunLabel
    return function(x...)
        res, t = @timed f(x...)
        s.totaltime += t
        s.neval += 1
        return res
    end
end

"""
    FitnessCache <: AbstractFitness

Caches fitness values so that they don't need to be recomputed.
"""
mutable struct FitnessCache <: AbstractFitness
    wrapped::AbstractFitness
    cache
end
FitnessCache(f::AbstractFitness) = FitnessCache(f, nothing)
function fitness(s::FitnessCache, f)
    if isnothing(s.cache)
        val = fitness(s.wrapped, f)
        s.cache = val
    end
    return s.cache
end


"""
    AbstractCandidate

Abstract base type for canidates
"""
abstract type AbstractCandidate end

"""
    CandidateModel <: Candidate
    CandidateModel(model::CompGraph, optimizer, lossfunction, fitness::AbstractFitness)

A candidate model consisting of a `CompGraph`, an optimizer a lossfunction and a fitness method.
"""
struct CandidateModel <: AbstractCandidate
    graph::CompGraph
    opt
    lossfun
    fitness::AbstractFitness
end

function Flux.train!(model::CandidateModel, data)
    f = instrument(Train(), model.fitness, x -> model.graph(x))
    loss(x,y) = model.lossfun(f(x), y)
    Flux.train!(loss, params(model.graph), data, model.opt)
end

fitness(model::CandidateModel) = fitness(model.fitness, instrument(Validate(), model.fitness, x -> model.graph(x)))

"""
    AbstractEvolution

Abstract base type for strategies for how to evolve a population into a new population
"""
abstract type AbstractEvolution end

"""
    evolve(e::AbstractEvolution, population::AbstractArray{<:AbstractCandidate})

Evolve `population` into a new population. New population may or may not contain same individuals as before.
"""
function evolve end

"""
    EliteSelection <: AbstractEvolution
    EliteSelection(nselect::Integer)

Selects the only the `nselect` highest fitness candidates
"""
struct EliteSelection <: AbstractEvolution
    nselect::Integer
end
evolve(e::EliteSelection, pop::AbstractArray{<:AbstractCandidate}) = partialsort(pop, 1:e.nselect, by=fitness, rev=true)
