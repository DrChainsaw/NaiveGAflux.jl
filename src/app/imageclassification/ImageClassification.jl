module ImageClassification

using ...NaiveGAflux
using ..AutoFlux: fit
import NaiveGAflux: GlobalPool
using Random
import Logging
using Statistics

# To store program state for pause/resume
using Serialization

export ImageClassifier, fit

modelname(c::AbstractCandidate) = modelname(NaiveGAflux.graph(c))
modelname(g::CompGraph) = split(name(g.inputs[]),'.')[1]

include("strategy.jl")
include("archspace.jl")

"""
    ImageClassifier
    ImageClassifier(popsize, seed, newpop)
    ImageClassifier(;popsize=50, seed=1, newpop=false)

Type to make `AutoFlux.fit` train an image classifier using initial population size `popsize` using random seed `seed`.

If `newpop` is `true` the process will start with a new population and existing state in the specified directory will be overwritten.
"""
struct ImageClassifier
    popsize::Int
    seed::Int
    newpop::Bool
end
ImageClassifier(;popsize=50, seed=1, newpop=false) = ImageClassifier(popsize, seed, newpop)

"""
    fit(c::ImageClassifier, x, y; cb, fitnesstrategy, trainstrategy, evolutionstrategy, mdir)

Return a population of image classifiers fitted to the given data.

# Arguments
- `c::ImageClassifier`: Type of models to train. See [`ImageClassifier`](@ref).

- `x`: Input data. Must be a 4D array.

- `y`: Output data. Can either be an 1D array in which case it is assumed that `y` is the raw labes (e.g. `["cat", "dog", "cat", ...]`) or a 2D array in which case it is assumed that `y` is one-hot encoded.

- `cb=identity`: Callback function. After training and evaluating each generation but before evolution `cb(population)` will be called where `population` is the array of candidates. Useful for persistence and plotting.

- `fitnesstrategy::AbstractFitnessStrategy=TrainSplitAccuracy()`: Strategy for fitness. See: [`ImageClassification.AbstractFitnessStrategy`](@ref).

- `trainstrategy::AbstractTrainStrategy=TrainStrategy()`: Strategy for training. See [`ImageClassification.AbstractTrainStrategy`](@ref)

- `evolutionstrategy::AbstractEvolutionStrategy=EliteAndTournamentSelection(popsize=c.popsize)`: Strategy for evolution. See [`ImageClassification.AbstractEvolutionStrategy`](@ref)

- `mdir`: Load models from this directory if present. If persistence is used (e.g. by providing `cb=persist`) candidates will be stored in this directory.

"""
function AutoFlux.fit(c::ImageClassifier, x, y; cb=identity, fitnesstrategy::AbstractFitnessStrategy=TrainSplitAccuracy(), trainstrategy::AbstractTrainStrategy=TrainStrategy(), evolutionstrategy::AbstractEvolutionStrategy=EliteAndTournamentSelection(popsize=c.popsize), mdir)
    ndims(x) == 4 || error("Must use 4D data, got $(ndims(x))D data")

    x, y, fitnessgen = fitnessfun(fitnesstrategy, x, y)
    fit_iter = trainiter(trainstrategy, x, y)
    inshape = size(x)[1:2]
    return fit(c, fit_iter, fitnessgen, evostrategy(evolutionstrategy, inshape); cb=cb, mdir=mdir)
end

"""
    fit(c::ImageClassifier, fit_iter, fitnessgen, evostrategy; cb, mdir)

Return a population of image classifiers fitted to the given data.

Lower level version of `fit` to use when `fit(c::ImageClassifier, x, y)` doesn't cut it.

# Arguments
- `c::ImageClassifier`: Type of models to train. See [`ImageClassifier`](@ref).

- `fit_iter`: Iterator for fitting the models. Expected to produce iterators over some subset of the training data. The produced iterators are in turn expected to produce batches of input-output tuples. See [`RepeatPartitionIterator`](@ref) for an example an iterator which fits the bill.

- `fitnessgen`: Return an `AbstractFitness` when called with no arguments. May or may not produce the same instance depending on whether stateful fitness is used.

- `evostrategy::AbstractEvolution`: Evolution strategy to use. Population `p` will be evolved through `p = evolve!(evostrategy, p)`.

- `cb=identity`: Callback function. After training and evaluating each generation but before evolution `cb(population)` will be called where `population` is the array of candidates. Useful for persistence and plotting.

- `mdir`: Load models from this directory if present. If persistence is used (e.g. by providing `cb=persist`) candidates will be stored in this directory.
"""
function AutoFlux.fit(c::ImageClassifier, fit_iter, fitnessgen, evostrategy::AbstractEvolution; cb = identity, mdir)
    Random.seed!(NaiveGAflux.rng_default, c.seed)
    @info "Start training with baseseed: $(c.seed)"

    insize, outsize = datasize(fit_iter)

    population = initial_models(c.popsize, mdir, c.newpop, fitnessgen, insize, outsize[1])

    # If experiment was resumed we should start by evolving as population is persisted right before evolution
    if all(i -> isfile(NaiveGAflux.filename(population, i)), 1:length(population))
        population = evolve!(evostrategy, population)
    end

    return evolutionloop(population, evostrategy, fit_iter, cb)
end

datasize(itr) = datasize(first(itr))
datasize(t::Tuple) = datasize.(t)
datasize(a::AbstractArray) = size(a)

function evolutionloop(population, evostrategy, trainingiter, cb)
    for (gen, iter) in enumerate(trainingiter)
        @info "Begin generation $gen"

        for (i, cand) in enumerate(population)
            @info "\tTrain candidate $i with $(nv(NaiveGAflux.graph(cand))) vertices"
            Flux.train!(cand, iter)
        end

        # TODO: Bake into evolution? Would anyways like to log selected models...
        for (i, cand) in enumerate(population)
            @info "\tFitness candidate $i: $(fitness(cand))"
        end
        cb(population)

        population = evolve!(evostrategy, population)
    end
    return population
end

function initial_models(nr, mdir, newpop, fitnessgen, insize, outsize)
    if newpop
        rm(mdir, force=true, recursive=true)
    end

    iv(i) = inputvertex(join(["model", i, ".input"]), insize[3], FluxConv{2}())
    as = initial_archspace(insize[1:2], outsize)
    return PersistentArray(mdir, nr, i -> create_model(join(["model", i]), as, iv(i), fitnessgen))
end
create_model(name, as, in, fg) = CacheCandidate(HostCandidate(CandidateModel(CompGraph(in, as(name, in)), newopt(rand() * 0.099 + 0.01), Flux.logitcrossentropy, fg())))

end
