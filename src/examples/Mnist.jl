module Mnist

using ..NaiveGAflux
using NaiveGAflux.AutoFlux
using NaiveGAflux.AutoFlux.ImageClassification
using NaiveGAflux.AutoFlux.ImageClassification: TrainStrategy, TrainSplitAccuracy, PruneLongRunning

export run_experiment

defaultdir(this="MNIST") = joinpath(NaiveGAflux.modeldir, this)

"""
    run_experiment((x,y)::Tuple; plt=plot, sctr=scatter; seed=1, nepochs=200)

Run experiment for MNIST.

Note that supplying arguments `plot` and `scatter` is only an artifact due to this package not having `Plots` as a  dependency.

#Examples
```julia-repl
julia> using NaiveGAflux, NaiveGAflux.Mnist, MLDatasets, Plots; Plots.scalefontsizes(0.6)

julia> run_experiment(MNIST.traindata(), plot, scatter)
```
"""
function run_experiment((x,y)::Tuple, plt, sctr; seed=1, nepochs=200)

    modeldir = defaultdir()

    ts = TrainStrategy(nepochs=nepochs, seed = seed)
    fs = PruneLongRunning(TrainSplitAccuracy(batchsize=128), 0.075, 0.15)

    cb = CbAll(persist, MultiPlot(display ∘ plt, PlotFitness(plt, modeldir), ScatterPop(sctr, modeldir), ScatterOpt(sctr, modeldir)))

    c = ImageClassifier(popsize=50, seed=seed)

    return fit(c, reshape(x, 28, 28, 1, 60000), y, fitnesstrategy=fs, trainstrategy=ts, cb=cb, mdir = modeldir, gcthreshold = 0.1)
end

end
