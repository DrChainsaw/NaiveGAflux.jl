module ImageClassification

using ...NaiveGAflux
using ..AutoFit: fit
import NaiveGAflux: globalpooling2d
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

struct ImageClassifier
    popsize::Int
    seed::Int
    newpop::Bool
end
ImageClassifier(;popsize=50, seed=1, newpop=false) = ImageClassifier(popsize, seed, newpop)

function AutoFit.fit(c::ImageClassifier, x, y; cb=identity, fitnesstrategy=TrainSplitAccuracy(), trainstrategy=TrainStrategy(), evolutionstrategy=EliteAndSusSelection(popsize=c.popsize), mdir)
    ndims(x) == 4 || error("Must use 4D data, got $(ndims(x))D data")

    x, y, fitnessgen = fitnessfun(fitnesstrategy, x, y)
    fit_iter = trainiter(trainstrategy, x, y)
    inshape = size(x)[1:2]
    return fit(c, fit_iter, fitnessgen, evostrategy(evolutionstrategy, inshape); cb=cb, mdir=mdir)
end

function AutoFit.fit(c::ImageClassifier, fit_iter, fitnessgen, evostrategy; cb = identity, mdir)
    Random.seed!(NaiveGAflux.rng_default, c.seed)
    @info "Start experiment with baseseed: $(c.seed)"

    insize, outsize = size(fit_iter)

    population = initial_models(c.popsize, mdir, c.newpop, fitnessgen, insize, outsize[1])

    # If experiment was resumed we should start by evolving as population is persisted right before evolution
    if all(i -> isfile(NaiveGAflux.filename(population, i)), 1:length(population))
        population = evolve!(evostrategy, population)
    end

    return evolutionloop(population, evostrategy, fit_iter, cb)
end

function evolutionloop(population, evostrategy, trainingiter, cb)
    for (gen, iter) in enumerate(trainingiter)
        @info "Begin generation $gen"

        for (i, cand) in enumerate(population)
            @info "\tTrain model $i with $(nv(NaiveGAflux.graph(cand))) vertices"
            Flux.train!(cand, iter)
        end

        # TODO: Bake into evolution? Would anyways like to log selected models...
        for (i, cand) in enumerate(population)
            @info "\tFitness model $i: $(fitness(cand))"
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
create_model(name, as, in, fg) = CacheCandidate(HostCandidate(CandidateModel(CompGraph(in, as(name, in)), newopt(newlr(0.01)), Flux.logitcrossentropy, fg())))

end
