
"""
    LogFitness{F, MF} <: AbstractFitness
    LogFitness(fitnesstrategy::AbstractFitness) = LogFitness(;fitnesstrategy)
    LogFitness(;currgen=0, candcnt=0, fitnesstrategy, msgfun=default_fitnessmsgfun)

Logs the fitness of `fitnessstrategy` along with some candiate information.
"""
mutable struct LogFitness{F, MF} <: AbstractFitness
    currgen::Int
    candcnt::Int
    fitstrat::F
    msgfun::MF
end
LogFitness(fitnesstrategy::AbstractFitness) = LogFitness(;fitnesstrategy)
LogFitness(;currgen=0, candcnt=0, fitnesstrategy, msgfun=default_fitnessmsgfun) = LogFitness(currgen, candcnt, fitnesstrategy, msgfun)

function fitness(lf::LogFitness, c::AbstractCandidate, gen)
    f = fitness(lf.fitstrat, c, gen)
    if gen != lf.currgen
        lf.candcnt = 0
        lf.currgen = gen
    end
    lf.candcnt += 1
    lf.msgfun(lf.candcnt, c, f)
    return f
end

function default_fitnessmsgfun(i, c, f; level::Logging.LogLevel = Logging.Info)
    nvs, nps = graph(c, g -> (nv(g), nparams(g)))
    if nps > 1e8
        nps = @sprintf "%5.2fG" (nps / 1e9)
    elseif nps > 1e5
        nps = @sprintf "%5.2fM" (nps / 1e6)
    else
        nps = @sprintf "%5.2fk" (nps / 1e3)
    end
    msg = @sprintf "  Candidate: %i\tvertices: %i\tparams: %s\tfitness: %f" i nvs nps f
    @logmsg level msg
end


"""
    GpuFitness{F} <: AbstractFitness
    GpuFitness(fitnesstrategy)

Move candidates to `gpu` before calculating their fitness according to `fitnesstrategy`.

After fitness has been calculated the candidate is moved back to `cpu`.

Note that if no `gpu` is available this should be a noop.

Note: Only works for mutable models, such as `CompGraph` since it can't change the candidate itself. Consider using a `MutableCandidate` if dealing with immutable models.
"""
struct GpuFitness{F} <: AbstractFitness
    f::F
end
function fitness(s::GpuFitness, c::AbstractCandidate, gen)
    NaiveNASflux.forcemutation(graph(c)) # Optimization: If there is a LazyMutable somewhere, we want it to do its thing now so we don't end up copying the model to the GPU only to then trigger another copy when the mutations are applied.
    fitval = fitness(s.f, c |> gpu, gen)
    c |> cpu # As some parts, namely CompGraph change internal state when mapping to GPU
    gpu_gc()
    return fitval
end

const gpu_gc = if CUDA.functional()
    function()
        GC.gc()
        CUDA.reclaim()
    end
else
    () -> nothing
end

"""
    AccuracyFitness <: AbstractFitness
    AccuracyFitness(dataset)

Measure fitness as the accuracy on a dataset.
"""
struct AccuracyFitness{D} <: AbstractFitness
    dataset::D
end
function fitness(s::AccuracyFitness, c::AbstractCandidate, gen)
    acc,cnt = 0, 0
    model = graph(c)
    for (x,y) in s.dataset
        correct = Flux.onecold(cpu(model(x))) .== Flux.onecold(cpu(y))
        acc += sum(correct)
        cnt += length(correct)
    end
    return acc / cnt
end

"""
    TrainThenFitness{I,L,O,F} <: AbstractFitness
    TrainThenFitness(;dataiter, defaultloss, defaultopt, fitstrat, invalidfitness=0.0)

Measure fitness using `fitstrat` after training the model using `dataiter`.

Loss function and optimizer may be provided by the candidate if `lossfun(c; defaultloss)` 
and `opt(c; defaultopt)` are implemented, otherwise `defaultloss` and `defaultopt` will 
be used.

The data used for training is the result of `itergeneration(dataiter, gen)` where `gen`
is the generation number. This defaults to returning `dataiter` but allows for more 
complex iterators such as [`StatefulGenerationIter`](@gen).

If the model loss is ever `NaN` or `Inf` the training will be stopped and `invalidfitness`
will be returned without calculating the fitness using `fitstrat`. 
"""
struct TrainThenFitness{I,L,O,F, IF} <: AbstractFitness
    dataiter::I
    defaultloss::L
    defaultopt::O
    fitstrat::F
    invalidfitness::IF
end
TrainThenFitness(;dataiter, defaultloss, defaultopt, fitstrat, invalidfitness=0.0) = TrainThenFitness(dataiter, defaultloss, defaultopt, fitstrat, invalidfitness)

function fitness(s::TrainThenFitness, c::AbstractCandidate, gen)
    loss = lossfun(c; default=s.defaultloss)
    model = graph(c)
    ninput = ninputs(model)

    valid = let valid = true
        nanguard = function(data...)
            inputs = data[1:ninput]
            ŷ = model(inputs...)

            y = data[ninput+1:end]
            l = loss(ŷ, y...)   
            
            checkvalid(l) do badval
                @warn "$badval loss detected when training!"
                valid = false
                Flux.stop()
            end
            return l
        end
        iter = itergeneration(s.dataiter, gen)
        o = opt(c; default=s.defaultopt)
        Flux.train!(nanguard, params(model), iter, o)
        cleanopt(o)
        valid
    end
    return valid ? fitness(s.fitstrat, c, gen) : s.invalidfitness
end

function checkvalid(ifnot, x)
    anynan = any(isnan, x)
    anyinf = any(isinf, x)
    
    if anynan || anyinf
        badval = anynan ? "NaN" : "Inf"
        ifnot(badval)
        return false
    end
    return true
end

"""
    TrainAccuracyCandidate{C} <: AbstractWrappingCandidate

Collects training accuracy through a spying loss function. Only intended for use by [`TrainAccuracyFitness`](@ref).
"""
struct TrainAccuracyCandidate{C} <: AbstractWrappingCandidate
    acc::BitVector
    c::C
end
TrainAccuracyCandidate(c::AbstractCandidate) = TrainAccuracyCandidate(falses(0), c)
function lossfun(c::TrainAccuracyCandidate; default=nothing)
    actualloss = lossfun(c.c; default)
    return function(ŷ,y)
        nograd() do
            append!(c.acc, Flux.onecold(cpu(ŷ)) .== Flux.onecold(cpu(y)))
        end
        return actualloss(ŷ, y)
    end
end

"""
    TrainAccuracyFitnessInner <: AbstractFitness

Fetches `acc` from a `TrainAccuracyCandidate`. Only intended for use by [`TrainAccuracyFitness`](@ref).
"""
struct TrainAccuracyFitnessInner{D} <: AbstractFitness 
    drop::D
end

function fitness(s::TrainAccuracyFitnessInner, c::TrainAccuracyCandidate, gen)
    startind = max(1, 1+floor(Int, s.drop * length(c.acc)))
    return mean(@view c.acc[startind:end])
end

"""
    mutable struct TrainAccuracyFitness <: AbstractFitness
    TrainAccuracyFitness(;drop=0.5, kwargs...)

Measure fitness as the accuracy on the training data set. Beware of overfitting!

Parameter `drop` determines the fraction of examples to drop for fitness measurement. This mitigates the penalty for newly mutated candidates as the first part of the training examples are not used for fitness.

Other keyword arguments are passed to `TrainThenFitness` constructor. Note that `fitstrat` should generally be
left to default value.

Advantage vs `AccuracyFitness` is that one does not have to run through another data set. Disadvantage is that evolution will likely favour candidates which overfit.
"""
struct TrainAccuracyFitness{T} <: AbstractFitness
    train::T
end
TrainAccuracyFitness(;drop=0.5, fitstrat=TrainAccuracyFitnessInner(drop), kwargs...,) =  TrainAccuracyFitness( TrainThenFitness(;fitstrat, kwargs...)) 

fitness(s::TrainAccuracyFitness, c::AbstractCandidate, gen) = fitness(s.train, TrainAccuracyCandidate(c), gen)


"""
    MapFitness <: AbstractFitness
    MapFitness(mapping::Function, base::AbstractFitness)

Maps fitness `x` from `base` to `mapping(x)`.
"""
struct MapFitness <: AbstractFitness
    mapping::Function
    base::AbstractFitness
end
fitness(s::MapFitness, c::AbstractCandidate, gen) = fitness(s.base, c, gen) |> s.mapping


"""
    EwmaFitness(base)
    EwmaFitness(α, base)

Computes the exponentially weighted moving average of the fitness of `base`.

Main purpose is to mitigate the effects of fitness noise.

The filter is updated each time `fitness` is called, so practical use requires the fitness to be wrapped in a `CacheFitness`.

See `https://github.com/DrChainsaw/NaiveGAExperiments/blob/master/fitnessnoise/experiments.ipynb` for some hints as to why this might be needed.
"""
EwmaFitness(base) = EwmaFitness(0.5, base)
function EwmaFitness(α, base)
    avgfitness = missing
    ewma = Ewma(α)
    state(x) = avgfitness = NaiveNASflux.agg(ewma, avgfitness, x)
    return MapFitness(state, base)
end


"""
    TimeFitness{T} <: AbstractFitness

Measure fitness as time to evaluate a function.

Time for first `nskip` evaluations will be discarded.

Function needs to be instrumented using [`instrument`](@ref).
"""
struct TimeFitness{T} <: AbstractFitness
    fitstrat::T
end
function fitness(s::TimeFitness, c::AbstractCandidate, gen)
    res, t, bytes, gctime = @timed fitness(s.fitstrat, c, gen)
    return t - gctime, res
end


"""
    SizeFitness <: AbstractFitness
    SizeFitness()

Measure fitness as the total number of parameters in the function to be evaluated.

Note: relies on Flux.params which does not work for functions which have been instrumented through [`instrument`](@ref).

To handle intrumentation, an attempt to extract the size is also made when instrumenting for `Validation`. Whether this works or not depends on the order in which fitness functions are combined.
"""
struct SizeFitness <: AbstractFitness end
fitness(s::SizeFitness, c::AbstractCandidate, gen) = nparams(c)
    

"""
    AggFitness <: AbstractFitness
    AggFitness(aggfun::Function, fitnesses::AbstractFitness...)

Aggreagate fitness value from all `fitnesses` using `aggfun`
"""
struct AggFitness{T} <: AbstractFitness
    aggfun::Function
    fitnesses::T
end
AggFitness(aggfun) = error("Must supply an aggregation function an at least one fitness")
AggFitness(aggfun, fitnesses::AbstractFitness...) = AggFitness(aggfun, fitnesses)

fitness(s::AggFitness, c::AbstractCandidate, gen) = mapfoldl(fs -> fitness(fs, c, gen), s.aggfun, s.fitnesses)
