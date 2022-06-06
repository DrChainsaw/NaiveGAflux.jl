module ImageClassification

using ...NaiveGAflux
using ..AutoFlux: fit
using NaiveGAflux: GlobalPool
using NaiveGAflux: shapetrace, squashshapes, fshape, ndimsout, check_apply
using NaiveGAflux: StatefulGenerationIter
using NaiveNASlib.Advanced, NaiveNASlib.Extend
import Flux
using Flux: Dense, Conv, ConvTranspose, DepthwiseConv, CrossCor, LayerNorm, BatchNorm, InstanceNorm, GroupNorm, 
            MaxPool, MeanPool, Dropout, AlphaDropout, GlobalMaxPool, GlobalMeanPool, cpu, gpu
using Flux: Descent, Momentum, Nesterov, ADAM, NADAM, ADAGrad, WeightDecay
import Functors
using Functors: fmap
using Random
import Logging
using Statistics
import IterTools: ncycle

# To store program state for pause/resume
using Serialization

export ImageClassifier, fit, TrainSplitAccuracy, TrainAccuracyVsSize, AccuracyVsSize, TrainIterConfig, BatchedIterConfig, ShuffleIterConfig, GlobalOptimizerMutation, EliteAndSusSelection, EliteAndTournamentSelection

modelname(c::AbstractCandidate) = NaiveGAflux.model(modelname, c)
modelname(g::CompGraph) = split(name(g.inputs[]),'.')[1]

include("strategy.jl")
include("archspace.jl")

"""
    ImageClassifier
    ImageClassifier(popinit, popsize, seed)
    ImageClassifier(;popsize=50, seed=1, newpop=false, mdir=defaultdir("ImageClassifier"); insize, outsize)

Type to make `AutoFlux.fit` train an image classifier using initial population size `popsize` using random seed `seed`.

Load models from `mdir` if directory contains models. If persistence is used (e.g. by providing `cb=persist` to `fit`) candidates will be stored in this directory.

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
    fit(c::ImageClassifier, x, y; cb, fitnesstrategy, evolutionstrategy, stopcriterion)

Return a population of image classifiers fitted to the given data.

# Arguments
- `c::ImageClassifier`: Type of models to train. See [`ImageClassifier`](@ref).

- `x`: Input data. Must be a 4D array.

- `y`: Output data. Can either be an 1D array in which case it is assumed that `y` is the raw labes (e.g. `["cat", "dog", "cat", ...]`) or a 2D array in which case it is assumed that `y` is one-hot encoded.

- `cb=identity`: Callback function. After training and evaluating each generation but before evolution `cb(population)` will be called where `population` is the array of candidates. Useful for persistence and plotting.

- `fitnesstrategy::AbstractFitnessStrategy=`[`TrainSplitAccuracy()`](@ref): Strategy for fitness from data.

- `evolutionstrategy::AbstractEvolutionStrategy=`[`EliteAndTournamentSelection(popsize=c.popsize)`](@ref): Strategy for evolution.

- `stopcriterion`: Takes the current population and returns true if fitting shall stop. Candidate fitness is available by calling `fitness(c)` where `c` is a member of the population.

"""
function AutoFlux.fit(c::ImageClassifier, x::AbstractArray, y::AbstractArray; 
                            cb=identity, 
                            fitnesstrategy::AbstractFitnessStrategy=TrainSplitAccuracy(), 
                            evolutionstrategy::AbstractEvolutionStrategy=EliteAndTournamentSelection(popsize=c.popsize),
                            stopcriterion = pop -> generation(pop) > 100)
    ndims(x) == 4 || error("Must use 4D data, got $(ndims(x))D data")

    inshape = size(x)[1:2]
    return fit(c, fitnessfun(fitnesstrategy, x, y), evostrategy(evolutionstrategy, inshape), stopcriterion; cb)
end

"""
    fit(c::ImageClassifier, fitnesstrategy::AbstractFitness, evostrategy::AbstractEvolution, stopcriterion; cb)

Return a population of image classifiers fitted to the given data.

Lower level version of `fit` to use when `fit(c::ImageClassifier, x, y)` doesn't cut it.

# Arguments
- `c::ImageClassifier`: Type of models to train. See [`ImageClassifier`](@ref).

- `fitnessstrategy`: An `AbstractFitness` used to compute the fitness metric for a candidate.

- `evostrategy::AbstractEvolution`: Evolution strategy to use. Population `p` will be evolved through `p = evolve(evostrategy, p)`.

- `stopcriterion`: Takes the current population and returns true if fitting shall stop. Candidate fitness is available by calling `fitness(c)` where `c` is a member of the population.

- `cb=identity`: Callback function. After training and evaluating each generation but before evolution `cb(population)` will be called where `population` is the array of candidates. Useful for persistence and plotting.
"""
function AutoFlux.fit(c::ImageClassifier, fitnesstrategy::AbstractFitness, evostrategy::AbstractEvolution, stopcriterion; cb = identity)
    Random.seed!(NaiveGAflux.rng_default, c.seed)
    @info "Start training with baseseed: $(c.seed)"

    population = c.popinit()

    # If experiment was resumed we should start by evolving as population is persisted right before evolution
    population = generation(population) > 1 ? evolve(evostrategy, population) : population

    logfitness = LogFitness(;fitnesstrategy)

    return evolutionloop(population, evostrategy, logfitness, stopcriterion, cb)
end

function evolutionloop(population, evostrategy, fitnesstrategy, stop, cb)
    while true
        @info "Begin generation $(generation(population))"
        
        fittedpopulation = fitness(population, fitnesstrategy)
        cb(fittedpopulation)
        stop(fittedpopulation) && return fittedpopulation

        population = evolve(evostrategy, fittedpopulation)
    end
end

function generate_persistent(nr, newpop, mdir, insize, outsize, cwrap=identity, archspace = initial_archspace(insize[1:2], outsize))
    if newpop
        rm(mdir, force=true, recursive=true)
    end

    iv(i) = conv2dinputvertex(join(["model", i, ".input"]), insize[3])
    return Population(PersistentArray(mdir, nr, i -> create_model(join(["model", i]), archspace, iv(i), cwrap, insize)))
end
function create_model(name, as, in, cwrap, insize)
    optselect = optmutation(1.0)
    opt = optselect(Descent(rand() * 0.099 + 0.01))
    bslimit = batchsizeselection(insize[1:end-1]; alternatives=ntuple(i->2^i, 10))
    imstart = BatchSizeIteratorMap(64, 64, bslimit)
    im = itermapmutation(1.0)(imstart)
    cwrap(CandidateDataIterMap(im, CandidateOptModel(opt, CompGraph(in, as(name, in)))))
end

end
