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
    instrument(l::AbstractFunLabel, s::AbstractFitness, f)

Instrument `f` for fitness measurement `s`.

Argument `l` gives some context to `f` to enable different instrumentation for different operations.

Example is to use the result of `f` for fitness calculation, or to add a measurement of the average time it takes to evalute as used with [`TimeFitness`](@ref).

Basically a necessary (?) evil which complicates things around it quite a bit.
"""
instrument(::AbstractFunLabel, ::AbstractFitness, f) = f

"""
    reset!(s::AbstractFitness)

Reset all state of `s`. Typically needs to be performed after new candidates are selected.
"""
function reset!(::AbstractFitness) end

"""
    AccuracyFitness <: AbstractFitness
    AccuracyFitness(dataset)

Measure fitness as the accuracy on a dataset.

You probably want to use this with a `FitnessCache` or a `CacheCandidate`.
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
    mutable struct TrainAccuracyFitness <: AbstractFitness
    TrainAccuracyFitness(drop=0.5)

Measure fitness as the accuracy on the training data set. Beware of overfitting!

Parameter `drop` determines the fraction of examples to drop for fitness measurement. This mitigates the penalty for newly mutated candidates as the first part of the training examples are not used for fitness.

Advantage vs `AccuracyFitness` is that one does not have to run through another data set. Disadvantage is that evolution will likely favour candidates which overfit.
"""
mutable struct TrainAccuracyFitness <: AbstractFitness
    acc::AbstractArray
    ŷ::AbstractArray
    drop::Real
end
TrainAccuracyFitness(drop = 0.5) = TrainAccuracyFitness([], [], drop)
instrument(::Train,s::TrainAccuracyFitness,f) = function(x...)
    ŷ = f(x...)
    s.ŷ = ŷ |> cpu
    return ŷ
end
instrument(::TrainLoss,s::TrainAccuracyFitness,f) = function(x...)
    y = x[2]
    ret = f(x...)
    # Assume above call has also been instrument with Train, so now we have ŷ
    append!(s.acc, Flux.onecold(s.ŷ) .== Flux.onecold(cpu(y)))
    return ret
end
function fitness(s::TrainAccuracyFitness, f)
    @assert !isempty(s.acc) "No accuracy metric reported! Please make sure you have instrumented the correct methods and that training has been run."
    startind = max(1, 1+floor(Int, s.drop * length(s.acc)))
    mean(s.acc[startind:end])
end
function reset!(s::TrainAccuracyFitness)
    s.acc = []
    s.ŷ = []
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
instrument(l::AbstractFunLabel,s::MapFitness,f) = instrument(l, s.base, f)
reset!(s::MapFitness) = reset!(s.base)

"""
    TimeFitness{T} <: AbstractFitness where T <: AbstractFunLabel
    TimeFitness(t::T, nskip=0) where T <: AbstractFunLabel

Measure fitness as time to evaluate a function.

Time for first `nskip` evaluations will be discarded.

Function needs to be instrumented using [`instrument`](@ref).
"""
mutable struct TimeFitness{T} <: AbstractFitness where T <: AbstractFunLabel
    totaltime
    neval::Int
    nskip::Int
end
TimeFitness(t::T, nskip = 0) where T = TimeFitness{T}(0.0, 0, nskip)
fitness(s::TimeFitness, f) = s.neval <= s.nskip ? 0 : s.totaltime / (s.neval-s.nskip)

function instrument(::T, s::TimeFitness{T}, f) where T <: AbstractFunLabel
    return function(x...)
        res, t, bytes, gctime = @timed f(x...)
        # Skip first time(s) e.g. due to compilation
        if s.neval >= s.nskip
            s.totaltime += t - gctime
        end
        s.neval += 1
        return res
    end
end

function reset!(s::TimeFitness)
    s.totaltime = 0.0
    s.neval = 0
end

"""
    SizeFitness <: AbstractFitness
    SizeFitness()

Measure fitness as the total number of parameters in the function to be evaluated.

Note: relies on Flux.params which does not work for functions which have been instrumented through [`instrument`](@ref).

To handle intrumentation, an attempt to extract the size is also made when instrumenting for `Validation`. Whether this works or not depends on the order in which fitness functions are combined.
"""
mutable struct SizeFitness <: AbstractFitness
    size::Int
end
SizeFitness() = SizeFitness(0)

function fitness(s::SizeFitness, f)
    fsize = mapreduce(prod ∘ size, +, params(f).order, init=0)
    # Can't do params(f) as f typically is instrumented
    if fsize == s.size == 0
        @warn "SizeFitness got zero parameters! Check your fitness function!"
    end

    return fsize == 0 ? s.size : fsize
end
function instrument(l::Validate, s::SizeFitness, f)
    s.size = mapreduce(prod ∘ size, +, params(f).order, init=0)
    return f
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

instrument(l::AbstractFunLabel, s::FitnessCache, f) = instrument(l, s.base, f)

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
    lastout::IdDict
end
NanGuard(base::AbstractFitness, replaceval = 0.0) = foldl((b, l) -> NanGuard(l, b, replaceval), (Train(), TrainLoss(), Validate()), init=base)
NanGuard(t::T, base::AbstractFitness, replaceval = 0.0) where T <: AbstractFunLabel = NanGuard{T}(base, false, replaceval, IdDict())

fitness(s::NanGuard, f) = s.shield ? 0.0 : fitness(s.base, f)

function reset!(s::NanGuard)
    s.shield = false
    reset!(s.base)
end

function NaiveGAflux.instrument(l::T, s::NanGuard{T}, f) where T <: NaiveGAflux.AbstractFunLabel
    fi = NaiveGAflux.instrument(l, s.base, f)
    return function(x...)
        if s.shield
            lastout = NaiveGAflux.nograd() do
                get(s.lastout, size.(x), nothing)
            end

            !isnothing(lastout) && return lastout(s.replaceval)
        end
        y = fi(x...)

        wasshield = s.shield
        anynan = NaiveGAflux.nograd() do
            # Broadcast to avoid scalar operations when using CuArrays
            anynan = any(isnan.(y))
            anyinf = any(isinf.(y))

            s.shield = anynan || anyinf
            tt = typeof(y)
            ss = size(y)

            s.lastout[size.(x)] = val -> NaiveGAflux.dummyvalue(tt, ss, val)
            return anynan
        end

        if s.shield
            NaiveGAflux.nograd() do
                if !wasshield
                    badval = anynan ? "NaN" : "Inf"
                    @warn "$badval detected for function with label $l for x of size $(size.(x))"
                end
            end
            return s.lastout[size.(x)](s.replaceval)
        end
        return y
    end
end
instrument(l::AbstractFunLabel, s::NanGuard, f) = instrument(l, s.base, f)


dummyvalue(::Type{<:AT}, shape, val) where AT <: AbstractArray = fill!(similar(AT, shape), val)
dummyvalue(::Type{T}, shape, val) where T <: Number = T(val)
Flux.Zygote.@nograd dummyvalue
# Flux.Zygote.@adjoint function dummyvalue(t, shape, val)
#     return dummyvalue(t, shape, val), _ -> (nothing, 0, 0)
# end

"""
    AggFitness <: AbstractFitness
    AggFitness(aggfun::Function, fitnesses::AbstractFitness...)

Aggreagate fitness value from all `fitnesses` using `aggfun`
"""
struct AggFitness <: AbstractFitness
    aggfun::Function
    fitnesses
end
AggFitness(aggfun) = error("Must supply an aggregation function an at least one fitness")
AggFitness(aggfun, fitnesses::AbstractFitness...) = AggFitness(aggfun, fitnesses)

fitness(s::AggFitness, f) = mapfoldl(ff -> fitness(ff, f), s.aggfun, s.fitnesses)

reset!(s::AggFitness) = foreach(reset!, s.fitnesses)

instrument(l::AbstractFunLabel, s::AggFitness, f) = foldl((ifun, fit) -> instrument(l, fit, ifun), s.fitnesses, init = f)
