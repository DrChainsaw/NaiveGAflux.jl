module ImageClassification

using ...NaiveGAflux
using ..AutoFlux: fit
import NaiveGAflux: GlobalPool
import NaiveGAflux: shapetrace, squashshapes, fshape, ndimsout
import NaiveGAflux: StatefulGenerationIter
using Random
import Logging
using Statistics
import IterTools: ncycle

# To store program state for pause/resume
using Serialization

export ImageClassifier, fit

modelname(c::AbstractCandidate) = NaiveGAflux.graph(c, modelname)
modelname(g::CompGraph) = split(name(g.inputs[]),'.')[1]

include("strategy.jl")
include("archspace.jl")

"""
    ImageClassifier
    ImageClassifier(popinit, popsize, seed)
    ImageClassifier(;popsize=50, seed=1, newpop=false; insize, outsize)

Type to make `AutoFlux.fit` train an image classifier using initial population size `popsize` using random seed `seed`.

Load models from `mdir` if directory contains models. If persistence is used (e.g. by providing `cb=persist`) candidates will be stored in this directory.

If `newpop` is `true` the process will start with a new population and existing state in the specified directory will be overwritten.
"""
struct ImageClassifier{F}
    popinit::F
    popsize::Int
    seed::Int
end
function ImageClassifier(;popsize=50, seed=1, newpop=false, mdir=defaultdir("ImageClassifier"), insize, outsize)
    popinit = () -> generate_persistent(popsize, newpop, mdir, insize, outsize)
    return ImageClassifier(popinit, popsize, seed)
end

"""
    fit(c::ImageClassifier, x, y; cb, fitnesstrategy, evolutionstrategy)

Return a population of image classifiers fitted to the given data.

# Arguments
- `c::ImageClassifier`: Type of models to train. See [`ImageClassifier`](@ref).

- `x`: Input data. Must be a 4D array.

- `y`: Output data. Can either be an 1D array in which case it is assumed that `y` is the raw labes (e.g. `["cat", "dog", "cat", ...]`) or a 2D array in which case it is assumed that `y` is one-hot encoded.

- `cb=identity`: Callback function. After training and evaluating each generation but before evolution `cb(population)` will be called where `population` is the array of candidates. Useful for persistence and plotting.

- `fitnesstrategy::AbstractFitnessStrategy=TrainSplitAccuracy()`: Strategy for fitness from data. See: [`ImageClassification.AbstractFitnessStrategy`](@ref).

- `evolutionstrategy::AbstractEvolutionStrategy=EliteAndTournamentSelection(popsize=c.popsize)`: Strategy for evolution. See [`ImageClassification.AbstractEvolutionStrategy`](@ref)

"""
function AutoFlux.fit(c::ImageClassifier, x::AbstractArray, y::AbstractArray; cb=identity, fitnesstrategy::AbstractFitnessStrategy=TrainSplitAccuracy(), evolutionstrategy::AbstractEvolutionStrategy=EliteAndTournamentSelection(popsize=c.popsize))
    ndims(x) == 4 || error("Must use 4D data, got $(ndims(x))D data")

    inshape = size(x)[1:2]
    return fit(c, fitnessfun(fitnesstrategy, x, y), evostrategy(evolutionstrategy, inshape); cb)
end

"""
    fit(c::ImageClassifier, fitnesstrategy::AbstractFitness, evostrategy::AbstractEvolution; cb)

Return a population of image classifiers fitted to the given data.

Lower level version of `fit` to use when `fit(c::ImageClassifier, x, y)` doesn't cut it.

# Arguments
- `c::ImageClassifier`: Type of models to train. See [`ImageClassifier`](@ref).

- `fitnessstrategy`: An `AbstractFitness` used to compute the fitness metric for a candidate.

- `evostrategy::AbstractEvolution`: Evolution strategy to use. Population `p` will be evolved through `p = evolve(evostrategy, p)`.

- `cb=identity`: Callback function. After training and evaluating each generation but before evolution `cb(population)` will be called where `population` is the array of candidates. Useful for persistence and plotting.
"""
function AutoFlux.fit(c::ImageClassifier, fitnesstrategy::AbstractFitness, evostrategy::AbstractEvolution; cb = identity)
    Random.seed!(NaiveGAflux.rng_default, c.seed)
    @info "Start training with baseseed: $(c.seed)"

    population = c.popinit()

    # If experiment was resumed we should start by evolving as population is persisted right before evolution
    population = generation(population) > 1 ? evolve(evostrategy, population) : population

    logfitness = LogFitness(;currgen=generation(population), fitnesstrategy)

    return evolutionloop(population, evostrategy, logfitness, pop -> generation(pop) > 100, cb)
end

function evolutionloop(population, evostrategy, fitnesstrategy, stop, cb)
    while true
        @info "Begin generation $(generation(population))"
        
        fittedpopulation = fitness(population, fitnesstrategy)
        cb(fittedpopulation)
        stop(fittedpopulation) && break

        population = evolve(evostrategy, fittedpopulation)
    end
    return fittedpopulation
end

function generate_persistent(nr, newpop, mdir, insize, outsize, cwrap=identity, archspace = initial_archspace(insize[1:2], outsize))
    if newpop
        rm(mdir, force=true, recursive=true)
    end

    iv(i) = inputvertex(join(["model", i, ".input"]), insize[3], FluxConv{2}())
    return Population(PersistentArray(mdir, nr, i -> create_model(join(["model", i]), archspace, iv(i), cwrap)))
end
function create_model(name, as, in, cwrap)
    optselect = optmutation(1.0)
    opt = optselect(Descent(rand() * 0.099 + 0.01))
    # Always cache even if not strictly needed for all fitnessfunctions because consequence is so bad if one forgets
    # it when needed. Users who know what they are doing can unwrap if caching is not wanted.
    cwrap(CandidateOptModel(CompGraph(in, as(name, in)), opt))
end

end
