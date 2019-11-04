module Cifar10

using ..NaiveGAflux
using NaiveGAflux.AutoFlux
using NaiveGAflux.AutoFlux.ImageClassification
using NaiveGAflux.AutoFlux.ImageClassification: TrainStrategy, TrainSplitAccuracy

export run_experiment

defaultdir(this="CIFAR10") = joinpath(NaiveGAflux.modeldir, this)

"""
    run_experiment((x,y)::Tuple; plt=plot, sctr=scatter; seed=1, nepochs=200)

Run experiment for CIFAR10.

Note that supplying arguments `plot` and `scatter` is only an artifact due to this package not having a dependency on `Plots`.

#Examples
```julia-repl
julia> using NaiveGAflux, NaiveGAflux.Cifar10, MLDatasets, Plots; Plots.scalefontsizes(0.6)

julia> run_experiment(CIFAR10.traindata(), plot, scatter)
```
"""
function run_experiment((x,y)::Tuple, plt, sctr; seed=1, nepochs=200)

    modeldir = defaultdir()

    ts= TrainStrategy(nepochs=nepochs, seed = seed, dataaug=ShiftIterator ∘ FlipIterator)
    fs = TrainSplitAccuracy(nexamples=2048, batchsize=128)

    cb = CbAll(persist, MultiPlot(display ∘ plt, PlotFitness(plt, modeldir), ScatterPop(sctr, modeldir), ScatterOpt(sctr, modeldir)))

    c = ImageClassifier(popsize=50, seed=seed)

    return fit(c, x, y, trainstrategy=ts, cb=cb, mdir = modeldir)
end

end  # module cifar10
