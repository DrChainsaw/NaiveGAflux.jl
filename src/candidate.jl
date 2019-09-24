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
struct TrainLoss <: AbstractFunLabel end
struct Validate <: AbstractFunLabel end

"""
    instrument(l::AbstractFunLabel, s::AbstractFitness, f::Funtion)

Instrument `f` for fitness measurement `s`.

Argument `l` gives some context to `f` to enable different instrumentation for different operations.

Example is to use the result of `f` for fitness calculation, or to add a measurement of the average time it takes to evalute as used with [`TimeFitness`](@ref).

Basically a necessary (?) evil which complicates things around it quite a bit.
"""
instrument(::AbstractFunLabel, ::AbstractFitness, f::Function) = f

"""
    reset!(s::AbstractFitness)

Reset all state of `s`. Typically needs to be performed after new candidates are selected.
"""
function reset!(::AbstractFitness) end

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

        acc += mean(Flux.onecold(cpu(f(x))) .== Flux.onecold(cpu(y)))
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
reset!(s::MapFitness) = reset!(s.base)

"""
    TimeFitness{T} <: AbstractFitness where T <: AbstractFunLabel
    TimeFitness(t::T) where T <: AbstractFunLabel

Measure fitness as time to evaluate a function.

Function needs to be instrumented using [`instrument`](@ref).
"""
mutable struct TimeFitness{T} <: AbstractFitness where T <: AbstractFunLabel
    totaltime
    neval
end
TimeFitness(t::T) where T = TimeFitness{T}(0.0, 0)
fitness(s::TimeFitness, f) = s.neval == 0 ? 0 : s.totaltime / s.neval

function instrument(::T, s::TimeFitness{T}, f::Function) where T <: AbstractFunLabel
    return function(x...)
        res, t = @timed f(x...)
        s.totaltime += t
        s.neval += 1
        return res
    end
end

function reset!(s::TimeFitness)
    s.totaltime = 0.0
    s.neval = 0
end

"""
    FitnessCache <: AbstractFitness

Caches fitness values so that they don't need to be recomputed.

Needs to be `reset!` manually when cache is stale (e.g. after training the model some more).
"""
mutable struct FitnessCache <: AbstractFitness
    base::AbstractFitness
    cache
end
FitnessCache(f::AbstractFitness) = FitnessCache(f, nothing)
function fitness(s::FitnessCache, f)
    if isnothing(s.cache)
        val = fitness(s.base, f)
        s.cache = val
    end
    return s.cache
end

function reset!(s::FitnessCache)
    s.cache = nothing
    reset!(s.base)
end

instrument(l::AbstractFunLabel, s::FitnessCache, f::Function) = instrument(l, s.base, f)

"""
    NanGuard{T} <: AbstractFitness where T <: AbstractFunLabel
    NanGuard(base::AbstractFitness, replaceval = 0.0)
    NanGuard(t::T, base::AbstractFitness, replaceval = 0.0) where T <: AbstractFunLabel

Instruments functions labeled with type `T` with a NaN guard.

The NaN guard checks for NaNs or Infs in the output of the instrumented function and

    1. Replaces the them with a configured value (default 0.0).
    2. Prevents the function from being called again. Returns the same output as last time.
    3. Returns fitness value of 0.0 without calling the base fitness function.

Rationale for 2 is that models tend to become very slow to evalute if when producing NaN/Inf.
"""
mutable struct NanGuard{T} <: AbstractFitness where T <: AbstractFunLabel
    base::AbstractFitness
    shield::Bool
    replaceval
    lastout
end
NanGuard(base::AbstractFitness, replaceval = 0.0) = NanGuard{AbstractFunLabel}(base, false, replaceval, nothing)
NanGuard(t::T, base::AbstractFitness, replaceval = 0.0) where T <: AbstractFunLabel = NanGuard{T}(base, false, replaceval, nothing)

fitness(s::NanGuard, f) = s.shield ? 0.0 : fitness(s.base, f)

function reset!(s::NanGuard)
    s.shield = false
    reset!(s.base)
end

function instrument(l::T, s::NanGuard{T}, f::Function) where T <: AbstractFunLabel
    function guard(x...)
        s.shield && return s.lastout

        y = f(x...)
        wasnan, y = checkreplace(isnan, y; replaceval = s.replaceval)
        wasinf, y = checkreplace(isinf, y; replaceval = s.replaceval)

        s.shield = wasnan || wasinf
        s.lastout = y
        if s.shield
            @warn "NaN/Inf detected for function with label $l"
        end
        return s.lastout
    end
    return instrument(l, s.base, guard)
end

checkreplace(f, x::T; replaceval) where T <:Real = f(x) ? (true, T(replaceval)) : (false, x)
checkreplace(f, x::Union{Tuple, AbstractArray}; replaceval) = any(f, x), map(xi -> f(xi) ? replaceval : xi, x)
function checkreplace(f, x::TrackedArray; replaceval)
    invalid = f.(x)
    anyinvalid = any(invalid)
    if anyinvalid
        x.data[invalid] .= replaceval
    end
    return anyinvalid, x
end

"""
    AggFitness <: AbstractFitness
    AggFitness(aggfun::Function, fitnesses::AbstractFitness...)

Aggreagate fitness value from all `fitnesses` using `aggfun`
"""
struct AggFitness <: AbstractFitness
    aggfun::Function
    fitnesses
end
AggFitness(aggfun, fitnesses::AbstractFitness...) = AggFitness(aggfun, fitnesses)

fitness(s::AggFitness, f) = s.aggfun(fitness.(s.fitnesses, f))

reset!(s::AggFitness) = foreach(reset!, s.fitnesses)

instrument(l::AbstractFunLabel, s::AggFitness, f::Function) = foldl((ifun, fit) -> instrument(l, fit, ifun), s.fitnesses, init = f)


"""
    AbstractCandidate

Abstract base type for canidates
"""
abstract type AbstractCandidate end

"""
    reset!(c::AbstractCandidate)

Reset state of `c`. Typically needs to be called after evolution to clear old fitness computations.
"""
function reset!(c::AbstractCandidate) end
Base.Broadcast.broadcastable(c::AbstractCandidate) = Ref(c)

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

Flux.children(c::CandidateModel) = (c.graph, c.opt, c.lossfun, c.fitness)
Flux.mapchildren(f, c::CandidateModel) = CandidateModel(f(c.graph), f(c.opt), f(c.lossfun), c.fitness)

function Flux.train!(model::CandidateModel, data::AbstractArray{<:Tuple})
    f = instrument(Train(), model.fitness, x -> model.graph(x))
    loss(x,y) = model.lossfun(f(x), y)
    iloss = instrument(TrainLoss(), model.fitness, loss)
    Flux.train!(iloss, params(model.graph), data, model.opt)
end

fitness(model::CandidateModel) = fitness(model.fitness, instrument(Validate(), model.fitness, x -> model.graph(x)))

reset!(model::CandidateModel) = reset!(model.fitness)


"""
    HostCandidate <: AbstractCandidate
    HostCandidate(c::AbstractCandidate)

Keeps `c` in host memory and transfers to GPU when training or calculating fitness.
"""
struct HostCandidate <: AbstractCandidate
    c::AbstractCandidate
end

Flux.@treelike HostCandidate

function Flux.train!(c::HostCandidate, data::AbstractArray{<:Tuple})
    Flux.train!(c.c |> gpu, data)
    c.c |> cpu # As some parts, namely CompGraph change internal state when mapping to GPU
end

function fitness(c::HostCandidate)
    fitval = fitness(c.c |> gpu)
    c.c |> cpu # As some parts, namely CompGraph change internal state when mapping to GPU
    return fitval
end

reset!(c::HostCandidate) = reset!(c.c)

"""
    evolvemodel(m::AbstractMutation{CompGraph})

Return a function which maps a `CandidateModel c1` to a new `CandidateModel c2` where `c2.graph = m(copy(c1.graph))`.

All other fields are copied "as is".

Intended use is together with [`EvolveCandidates`](@ref).
"""
function evolvemodel(m::AbstractMutation{CompGraph})
    function copymutate(g::CompGraph)
        ng = copy(g)
        m(ng)
        return ng
    end
    mapmodel(copymutate)
end

function mapmodel(newgraph::Function, newfields::Function=deepcopy)
    mapfield(g::CompGraph) = newgraph(g)
    mapfield(f) = newfields(f)
    # Bleh! This is not sustainable. Will refactor when I find the time
    newcand(c::CandidateModel) = CandidateModel(map(mapfield, getproperty.(c, fieldnames(CandidateModel)))...)
    newcand(c::HostCandidate) = HostCandidate(newcand(c.c))
    return newcand
end

"""
    AbstractEvolution

Abstract base type for strategies for how to evolve a population into a new population.
"""
abstract type AbstractEvolution end

"""
    evolve!(e::AbstractEvolution, population::AbstractArray{<:AbstractCandidate})

Evolve `population` into a new population. New population may or may not contain same individuals as before.
"""
function evolve end

"""
    NoOpEvolution <: AbstractEvolution
    NoOpEvolution()

Does not evolve the given population.
"""
struct NoOpEvolution <: AbstractEvolution end
evolve!(::NoOpEvolution, pop) = pop

"""
    AfterEvolution <: AbstractEvolution
    AfterSelection(evo::AbstractEvolution, fun::Function)

Calls `fun(newpop)` where `newpop = evolve!(e.evo, pop)` where `pop` is original population to evolve.
"""
struct AfterEvolution <: AbstractEvolution
    evo::AbstractEvolution
    fun::Function
end

function evolve!(e::AfterEvolution, pop)
    newpop = evolve!(e.evo, pop)
    e.fun(newpop)
    return newpop
end

"""
    ResetAfterEvolution(evo::AbstractEvolution)

Alias for `AfterEvolution` with `fun == reset!`.
"""
ResetAfterEvolution(evo::AbstractEvolution) = AfterEvolution(evo, np -> reset!.(np))


"""
    EliteSelection <: AbstractEvolution
    EliteSelection(nselect::Integer)

Selects the only the `nselect` highest fitness candidates.
"""
struct EliteSelection <: AbstractEvolution
    nselect::Integer
end
evolve!(e::EliteSelection, pop::AbstractArray{<:AbstractCandidate}) = partialsort(pop, 1:e.nselect, by=fitness, rev=true)

"""
    SusSelection <: AbstractEvolution
    SusSelection(nselect::Integer, evo::AbstractEvolution, rng=rng_default)

Selects candidates for further evolution using stochastic universal sampling.
"""
struct SusSelection <: AbstractEvolution
    nselect::Integer
    evo::AbstractEvolution
    rng
    SusSelection(nselect, evo, rng=rng_default) = new(nselect, evo, rng)
end

function evolve!(e::SusSelection, pop::AbstractArray{<:AbstractCandidate})
    csfitness = cumsum(fitness.(pop))

    gridspace = csfitness[end] / e.nselect
    start = rand(e.rng)*gridspace

    selected = similar(pop, e.nselect)
    for i in eachindex(selected)
        candind = findfirst(x -> x >= start + (i-1)*gridspace, csfitness)
        selected[i] = pop[candind]
    end
    return evolve!(e.evo, selected)
end

"""
    CombinedEvolution <: AbstractEvolution
    CombinedEvolution(evos::AbstractArray{<:AbstractEvolution})
    CombinedEvolution(evos::AbstractEvolution...)

Combines the evolved populations from several `AbstractEvolutions` into one population.
"""
struct CombinedEvolution <: AbstractEvolution
    evos::AbstractArray{<:AbstractEvolution}
end
CombinedEvolution(evos::AbstractEvolution...) = CombinedEvolution(collect(evos))
evolve!(e::CombinedEvolution, pop) = mapfoldl(evo -> evolve!(evo, pop), vcat, e.evos)

"""
    EvolveCandidates <: AbstractEvolution
    EvolveCandidates(fun::Function)

Applies `fun` for each candidate in a given population.

Useful with [`evolvemodel`](@ref).
"""
struct EvolveCandidates <: AbstractEvolution
    fun::Function
end
evolve!(e::EvolveCandidates, pop) = map(e.fun, pop)
