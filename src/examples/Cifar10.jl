module Cifar10

using ..NaiveGAflux
using NaiveGAflux.AutoFlux
using NaiveGAflux.AutoFlux.ImageClassification
using NaiveGAflux.AutoFlux.ImageClassification: TrainStrategy, TrainAccuracyVsSize

export run_experiment

defaultdir(this="CIFAR10") = joinpath(NaiveGAflux.modeldir, this)

function run_experiment((x,y)::Tuple, plt, sctr, seed)

    modeldir = defaultdir()
    ts= TrainStrategy(nepochs=1, seed = seed, dataaug=ShiftIterator ∘ FlipIterator)
    fs= TrainAccuracyVsSize()
    cb = CbAll(persist, MultiPlot(display ∘ plt, PlotFitness(plt, modeldir), ScatterPop(sctr, modeldir), ScatterOpt(sctr, modeldir)))
    c = ImageClassifier(popsize=50, seed=seed)
    fit(c, x, y, fitnesstrategy=fs, trainstrategy=ts, cb=cb, mdir = modeldir)
end

end  # module cifar10
