"""
    fitness(f::AbstractFitness, c::AbstractCandidate)

Compute the fitness metric `f` for candidate `c`.
"""
function fitness(f::AbstractFitness, c::AbstractCandidate)
    hold!(c) # Note: This transfer over to any potential fmapped candidates, e.g. in GpuFitness
    val = _fitness(f, c)
    release!(c)
    return val
end


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

function _fitness(lf::LogFitness, c::AbstractCandidate)
    f = _fitness(lf.fitstrat, c)
    gen = generation(c; default=lf.currgen)
    if gen != lf.currgen
        lf.candcnt = 0
        lf.currgen = gen
    end
    lf.candcnt += 1
    lf.msgfun(lf.candcnt, c, f)
    return f
end

function default_fitnessmsgfun(i, c, f; level::Logging.LogLevel = Logging.Info)
    nvs, nps = model(g -> (nvertices(g), nparams(g)), c)
    if nps > 1e8
        nps = @sprintf "%5.2fG" (nps / 1e9)
    elseif nps > 1e5
        nps = @sprintf "%5.2fM" (nps / 1e6)
    else
        nps = @sprintf "%5.2fk" (nps / 1e3)
    end
    msg = @sprintf " Candidate: %3i\tvertices: %3i\tparams: %s\tfitness: %s" i nvs nps f
    @logmsg level msg
end


"""
    GpuFitness{F} <: AbstractFitness
    GpuFitness(fitnesstrategy)

Move candidates to `gpu` before calculating their fitness according to `fitnesstrategy`.

Copies parameters back to the given candidate after fitness have been computed to ensure that updated parameters from training are used. 
Assumes canidates have parameters on `cpu` for that step.

Note that if no `gpu` is available this should be a noop.
"""
struct GpuFitness{F} <: AbstractFitness
    f::F
end
function _fitness(s::GpuFitness, c::AbstractCandidate)
    cgpu = gpu(c)
    fitval = _fitness(s.f, cgpu)
    # In case parameters changed. Would like to do this some other way, perhaps return the candidate too, or move training to evolve...
    transferstate!(c, cpu(cgpu)) # Can't load CuArray into a normal array
    cgpu = nothing # So we can reclaim the memory
    # Should not be needed according to CUDA docs, but programs seems to hang every now and then if not done.
    # Should revisit every now and then to see if things have changed...
    gpu_gc()
    return fitval
end

transferstate!(to, from) = _transferstate!(to, from)
function transferstate!(to::ActivationContribution, from::ActivationContribution)
    if isempty(to.contribution)
        resize!(to.contribution, length(from.contribution))
    end
    _transferstate!(to, from)
end

function transferstate!(to::T, from::T) where T <: NaiveNASflux.AbstractMutableComp
    if ismutable(to)
        for fn in fieldnames(T)
            setfield!(to, fn, getfield(from, fn))
        end
    else
        _transferstate!(to, from)
    end
end

function _transferstate!(to, from)
    tocs, _ = functor(to)
    fromcs, _ = functor(from)
    @assert length(tocs) == length(fromcs) "Mismatched number of children for $to vs $from"
    for (toc, fromc) in zip(tocs, fromcs)
        transferstate!(toc, fromc)
    end
end
#_transferstate!(to::AbstractArray, from::AbstractArray) = copyto!(to, from)
_transferstate!(to::AbstractArray{<:Number}, from::AbstractArray{<:Number}) = copyto!(to, from)
function _transferstate!(to::T, from::T) where T <:AbstractArray 
    @assert length(to) === length(from) "Mismatched array lenghts´of type "
    foreach(transferstate!, to, from)
end 

const gpu_gc = if CUDA.functional()
    function(full=true)
        GC.gc(full)
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
function _fitness(s::AccuracyFitness, c::AbstractCandidate)

    m = model(c)
    ninput = ninputs(m)

    iter = _fitnessiterator(validationiterator, c, s.dataset)

    acc,cnt = 0.0, 0
    for (data) in iter
        xs = data[1:ninput]
        ys = data[ninput+1:end]

        correct = Flux.onecold(m(xs...)) .== Flux.onecold(ys...)
        acc += sum(correct)
        cnt += length(correct)
    end
    return cnt == 0 ? acc : acc / cnt
end

function _fitnessiterator(f, c::AbstractCandidate, iter)
    geniter = itergeneration(iter, generation(c; default=0))
    canditer = f(c; default=geniter)
    matchdatatype(params(c), canditer)
end

matchdatatype(ps::Flux.Params, iter) = isempty(ps) ? iter : matchdatatype(first(ps), iter)

matchdatatype(::CUDA.CuArray, iter) = GpuIterator(iter)
matchdatatype(::AbstractArray, iter) = iter

"""
    TrainThenFitness{I,L,O,F} <: AbstractFitness
    TrainThenFitness(;dataiter, defaultloss, defaultopt, fitstrat, invalidfitness=0.0)

Measure fitness using `fitstrat` after training the model using `dataiter`.

Loss function and optimizer may be provided by the candidate if `lossfun(c; defaultloss)` 
and `opt(c; defaultopt)` are implemented, otherwise `defaultloss` and `defaultopt` will 
be used.

The data used for training is the result of `itergeneration(dataiter, gen)` where `gen`
is the generation number. This defaults to returning `dataiter` but allows for more 
complex iterators such as [`StatefulGenerationIter`](@ref).

If the model loss is ever `NaN` or `Inf` the training will be stopped and `invalidfitness`
will be returned without calculating the fitness using `fitstrat`. 

Tip: Use `TimedIterator` to stop training of models which take too long to train.
"""
struct TrainThenFitness{I,L,O,F, IF} <: AbstractFitness
    dataiter::I
    defaultloss::L
    defaultopt::O
    fitstrat::F
    invalidfitness::IF
end
TrainThenFitness(;dataiter, defaultloss, defaultopt, fitstrat, invalidfitness=0.0) = TrainThenFitness(dataiter, defaultloss, defaultopt, fitstrat, invalidfitness)

function _fitness(s::TrainThenFitness, c::AbstractCandidate)
    loss = lossfun(c; default=s.defaultloss)
    o = opt(c; default=s.defaultopt)

    iter = _fitnessiterator(trainiterator, c, s.dataiter)

    valid = trainmodel!(loss, model(c), o, iter)
    return valid ? _fitness(s.fitstrat, c) : s.invalidfitness
end


function trainmodel!(lossfun, model, optrule, dataiter)
    ninput = ninputs(model)
    opt_state = Flux.setup(optrule, model)
    for data in dataiter
        inputs = data[1:ninput]
        y = data[ninput+1:end]
        
        l, modelgrads = Flux.withgradient(model) do m
            ŷ = m(inputs...)
            lossfun(ŷ, y...)  
        end 
        
        _lossok(l) || return false

        Flux.update!(opt_state, model, modelgrads[1])
    end
    return true
end

function trainmodel!(lossfun, model, optrule::ImplicitOpt, dataiter)
    ninput = ninputs(model)
    ok = optimisersetup!(optrule, model)
    if !ok 
        throw(ArgumentError("Could not setup implicit optimiser for model $model. Forgot to use $AutoOptimiser when creating vertices?"))
    end

    for data in dataiter
        inputs = data[1:ninput]
        y = data[ninput+1:end]
        
        l, _ = Flux.withgradient() do 
            ŷ = model(inputs...)
            lossfun(ŷ, y...)  
        end 
        
        _lossok(l) || return false
    end
    return true
end

function _lossok(l)
    if isnan(l) || isinf(l)
        badval = isnan(l) ? "NaN" : "Inf"
        @warn "$badval loss detected when training!"
        return false
    end
    true
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

@functor TrainAccuracyCandidate (c,)

"""
    TrainAccuracyFitnessInner <: AbstractFitness

Fetches `acc` from a `TrainAccuracyCandidate`. Only intended for use by [`TrainAccuracyFitness`](@ref).
"""
struct TrainAccuracyFitnessInner{D} <: AbstractFitness 
    drop::D
end

function _fitness(s::TrainAccuracyFitnessInner, c::TrainAccuracyCandidate)
    startind = max(1, 1+floor(Int, s.drop * length(c.acc)))
    return mean(@view c.acc[startind:end])
end

"""
    struct TrainAccuracyFitness <: AbstractFitness
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
TrainAccuracyFitness(;drop=0.5, kwargs...,) =  TrainAccuracyFitness( TrainThenFitness(;fitstrat=TrainAccuracyFitnessInner(drop), kwargs...)) 

_fitness(s::TrainAccuracyFitness, c::AbstractCandidate) = _fitness(s.train, TrainAccuracyCandidate(c))


"""
    MapFitness <: AbstractFitness
    MapFitness(mapping::Function, base::AbstractFitness)

Maps fitness `x` from `base` to `mapping(x)`.
"""
struct MapFitness{F, T} <: AbstractFitness
    mapping::F
    base::T
end
_fitness(s::MapFitness, c::AbstractCandidate) = _fitness(s.base, c) |> s.mapping


"""
    EwmaFitness(base)
    EwmaFitness(α, base)

Computes the exponentially weighted moving average of the fitness of `base`. Assumes that candidates previous fitness metric is available through `fitness(cand)`.

Main purpose is to mitigate the effects of fitness noise.

See `https://github.com/DrChainsaw/NaiveGAExperiments/blob/master/fitnessnoise/experiments.ipynb` for some hints as to why this might be needed.
"""
struct EwmaFitness{F} <: AbstractFitness
    α::Float64
    base::F
end
EwmaFitness(base) = EwmaFitness(0.5, base)

function _fitness(s::EwmaFitness, c::AbstractCandidate)
    prev = fitness(c)
    curr = _fitness(s.base, c)
    return ewma(curr, prev, s.α)
end

ewma(curr, prev, α) = (1 - α) .* curr + α .* prev
ewma(curr, ::Nothing, α) = curr



"""
    TimeFitness{T} <: AbstractFitness

Measure fitness as time to evaluate a function.

Time for first `nskip` evaluations will be discarded.
"""
struct TimeFitness{T} <: AbstractFitness
    fitstrat::T
end
function _fitness(s::TimeFitness, c::AbstractCandidate)
    res, t, bytes, gctime = @timed _fitness(s.fitstrat, c)
    return t - gctime, res
end


"""
    SizeFitness <: AbstractFitness
    SizeFitness()

Measure fitness as the total number of parameters in the function to be evaluated.
"""
struct SizeFitness <: AbstractFitness end
_fitness(s::SizeFitness, c::AbstractCandidate) = nparams(c)
    

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

_fitness(s::AggFitness, c::AbstractCandidate) = mapfoldl(fs -> _fitness(fs, c), s.aggfun, s.fitnesses)
