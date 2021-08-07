# AutoFlux

AutoFlux is a neural architecture search application built on top of NaiveGAflux. It is designed to have a single high level API entry point for kicking of the search:

```julia
using NaiveGAflux.AutoFlux
models = fit(data)
```
Where `models` is the whole population of models found after search stopped.

It is possible (and strongly recommended) to supply a callback function which will receive the whole population of models as input after fitness for each generation has been calculated. A few useful functions are provided:

```julia
using NaiveGAflux, Plots
# Persist the whole population in directory models/CIFAR10 so that optimization can be resumed if aborted:
models = fit(CIFAR10.traindata(), cb=persist, mdir="models/CIFAR10")

# Plot best and average fitness for each generation
plotfitness = PlotFitness(plot, "models/CIFAR10");
# Plot data will be serialized in a subdir of "models/CIFAR10" for later postprocessing and for resuming optimization.
models = fit(CIFAR10.traindata(), cb=plotfitness, mdir="models/CIFAR10")


# Scatter plots from examples above:
scatterpop = ScatterPop(scatter, "models/CIFAR10");
scatteropt = ScatterOpt(scatter, "models/CIFAR10");

# Combine multiple plots in one figure:
multiplot = MultiPlot(display ∘ plot, plotfitness, scatterpop, scatteropt)

# Combine multiple callbacks in one function:
callbacks = CbAll(persist, multiplot)

models = fit(CIFAR10.traindata(), cb=callbacks, mdir="models/CIFAR10")
```  

The ambition level is to provide something like a more flexible version of [Large-Scale Evolution of Image Classifiers](https://arxiv.org/abs/1703.01041) with the following main improvements:

* Support for other model types (not done yet).
* Finer search granularity, i.e. about one 1 epoch of training per generation.
* No hardcoded training protocol, it is part of the search.
* Smaller population size feasible to run on a single machine.

Note that the set of modifications of AutoFlux is very different from [Large-Scale Evolution of Image Classifiers](https://arxiv.org/abs/1703.01041), but neither is a strict subset of the other so it is difficult assess which one has the most generic search space. 

In its current state, AutoFlux is perhaps best viewed as a combined example and end-to-end test of NaiveGAflux. 
It is more focused on making use of all components of NaiveGAflux rather than winning benchmarks. In particular,
it makes the very conscious and opinionated choice of not seeding the search space with models which are known
to work well (e.g. resnets etc.) as this is a bit against the point of searching for an architecture. If searching
for a benchmark winner is the goal, one might want to use the lower level APIs to start with a population of 
models which are known to perform well.

#### Performance

I develop this project on my spare time as a hobby and I don't have access to large amounts of computation for 
this. Thus providing benchmarking numbers with any kind of confidence behind them is very time consuming. Rough 
numbers for CIFAR10 is around 90% test accuracy for the best model after about 80 generations with one epoch per
generation. A best half population ensemble gives about 1% improvement. A few other tricks, such increasing the 
number of epochs for training (while decreasing population size) after a couple of generations also seem to 
yield about 1% improvement.