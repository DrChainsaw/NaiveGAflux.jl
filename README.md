# NaiveGAflux

[![Build status](https://github.com/DrChainsaw/NaiveGAflux.jl/workflows/CI/badge.svg?branch=master)](https://github.com/DrChainsaw/NaiveGAflux.jl/actions)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/DrChainsaw/NaiveGAflux.jl?svg=true)](https://ci.appveyor.com/project/DrChainsaw/NaiveGAflux-jl)
[![Codecov](https://codecov.io/gh/DrChainsaw/NaiveGAflux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DrChainsaw/NaiveGAflux.jl)

Neural architecture search for [Flux](https://github.com/FluxML/Flux.jl) models using genetic algorithms.

A marketing person might describe it as "practical proxyless NAS using an unrestricted search space".

The more honest purpose is to serve as a pipe cleaner and example for [NaiveNASflux](https://github.com/DrChainsaw/NaiveNASflux.jl) which is doing most of the heavy lifting.

## Basic Usage

```julia
]add NaiveGAflux
```

The basic idea is to create not just one model, but a population of several candidate models with different hyperparameters. The whole population is then evolved while the models are being trained.

| MNIST                                 | CIFAR10                                |
|:-------------------------------------:|:--------------------------------------:|
| <img src="gif/MNIST.gif" width="500"> | <img src="gif/CIFAR10.gif" width="500">  |

More concretely, this means train each model for a number of iterations, evaluate the fitness of each model, select the ones with highest fitness, apply random mutations (e.g. add/remove neurons/layers) to some of them and repeat until a model with the desired fitness has been produced.

By controlling the number of training iterations before evolving the population, it is possible tune the compromise between fully training each model at the cost of longer time to evolve versus the risk of discarding a model just because it trains slower than the other members.

Like any self-respecting AutoML-type library, NaiveGAflux provides an application with a deceivingly simple API:

```julia
using NaiveGAflux.AutoFlux, MLDatasets

models = fit(CIFAR10.traindata())
```

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

However, most non-toy uses cases will probably require a dedicated application. NaiveGAflux provides the components to make building it easy and fun!

Tired of tuning hyperparameters? Once you've felt the rush from reasoning about hyper-hyperparameters there is no going back!

This package has the following main components:
1. [Search spaces](#search-spaces)
2. [Mutation operations](#mutation)
3. [Crossover operations](#crossover)
4. [Fitness functions](#fitness-functions)
5. [Candidate utilities](#candidate-utilities)
6. [Evolution strategies](#evolution-strategies)
7. [Iterators](#iterators)

Each component is described more in detail below.

Here is a very basic example just to get a feeling for the package:

```julia
using NaiveGAflux, Flux, Random
Random.seed!(NaiveGAflux.rng_default, 0)

nlabels = 3
ninputs = 5

# Step 1: Create initial models
# Search space: 2-4 dense layers of width 3-10
layerspace = VertexSpace(DenseSpace(3:10, [identity, relu, elu, selu]))
initial_hidden = RepeatArchSpace(layerspace, 1:3)
# Output layer has fixed size and is shielded from mutation
outlayer = VertexSpace(Shielded(), DenseSpace(nlabels, identity))
initial_searchspace = ArchSpaceChain(initial_hidden, outlayer)

# Sample 5 models from the initial search space and make an initial population
model(invertex) = CompGraph(invertex, initial_searchspace(invertex))
models = [model(denseinputvertex("input", ninputs)) for _ in 1:5]
@test nvertices.(models) == [4, 3, 4, 5, 3]

population = Population(CandidateModel.(models))
@test generation(population) == 1

# Step 2: Set up fitness function:
# Train model for one epoch using datasettrain, then measure accuracy on datasetvalidate
# Some dummy data just to make stuff run
onehot(y) = Flux.onehotbatch(y, 1:nlabels)
batchsize = 4
datasettrain    = [(randn(ninputs, batchsize), onehot(rand(1:nlabels, batchsize)))]
datasetvalidate = [(randn(ninputs, batchsize), onehot(rand(1:nlabels, batchsize)))]

fitnessfunction = TrainThenFitness(;
    dataiter = datasettrain,
    defaultloss = Flux.logitcrossentropy, # Will be used if not provided by the candidate
    defaultopt = ADAM(), # Same as above. State is wiped after training to prevent memory leaks
    fitstrat = AccuracyFitness(datasetvalidate) # This is what creates our fitness value after training
)

# Step 3: Define how to search for new candidates
# We choose to evolve the existing ones through mutation

# VertexMutation selects valid vertices from the graph to mutate
# MutationProbability applies mutation m with a probability of p
# Lets shorten that a bit:
mp(m, p) = VertexMutation(MutationProbability(m, p))
# Add a layer (40% chance) and/or remove a layer (40% chance)
# You might want to use lower probabilities than this
addlayer = mp(AddVertexMutation(layerspace), 0.4)
remlayer = mp(RemoveVertexMutation(), 0.4)
mutation = MutationChain(remlayer, addlayer)

# Selection:
# The two best models are not changed, the rest are mutated using mutation defined above
elites = EliteSelection(2)
mutate = SusSelection(3, EvolveCandidates(evolvemodel(mutation)))
selection = CombinedEvolution(elites, mutate)

# Step 4: Run evolution
newpopulation = evolve(selection, fitnessfunction, population)
@test newpopulation != population
@test generation(newpopulation) == 2
# Repeat step 4 until a model with the desired fitness is found.
newnewpopulation = evolve(selection, fitnessfunction, newpopulation)
@test newnewpopulation != newpopulation
@test generation(newnewpopulation) == 3
# Maybe in a loop :)
```

### Search Spaces

The search space is a set of possible architectures which the search policy may use to create initial candidates or to extend existing candidates. Search spaces are constructed from simple components which can be combined in multiple ways, giving a lot of flexibility.

Lets start with the most simple search space, a `ParSpace`:

```julia
# Set seed of default random number generator for reproducible results
Random.seed!(NaiveGAflux.rng_default, 1)

ps1d = ParSpace([2,4,6,10])

# Draw from the search space
@test ps1d() == 6
@test ps1d() == 10

# Possible to supply another rng than the default one
@test ps1d(MersenneTwister(0)) == 4

# Can be of any dimension and type
ps2d = ParSpace(["1","2","3"], ["4","5","6","7"])

@test typeof(ps1d) == ParSpace{1, Int}
@test typeof(ps2d) == ParSpace{2, String}

@test ps2d() == ("1", "4")
```

Lets have a look at an example of a search space for convolutional layers:

```julia
Random.seed!(NaiveGAflux.rng_default, 1)

cs = ConvSpace{2}(outsizes=4:32, activations=[relu, elu, selu], kernelsizes=3:9)

inputsize = 16
convlayer = cs(inputsize)

@test string(convlayer) == "Conv((8, 3), 16 => 22, relu, pad=(4, 3, 1, 1))"
```

Lastly, lets look at how to construct a complex search space:

```julia
Random.seed!(NaiveGAflux.rng_default, 0)

# VertexSpace creates a MutableVertex of layers generated by the wrapped search space
cs = VertexSpace(ConvSpace{2}(outsizes=8:256, activations=[identity, relu, elu], kernelsizes=3:5))
bs = VertexSpace(BatchNormSpace([identity, relu]))

# Block of conv->bn and bn->conv respectively.
# Need to make sure there is always at least one SizeAbsorb layer to make fork and res below play nice
csbs = ArchSpaceChain(cs ,bs)
bscs = ArchSpaceChain(bs, cs)

# Randomly generates either conv or conv->bn or bn->conv:
cblock = ArchSpace(ParSpace1D(cs, csbs, bscs))

# Generates between 1 and 5 layers from csbs
rep = RepeatArchSpace(cblock, 1:5)

# Generates between 2 and 4 parallel paths joined by concatenation (inception like-blocks) from rep
fork = ForkArchSpace(rep, 2:4)

# Generates a residual connection around what is generated by rep
res = ResidualArchSpace(rep)

# ... and a residual fork
resfork = ResidualArchSpace(fork)

# Pick one of the above randomly...
repforkres = ArchSpace(ParSpace1D(rep, fork, res, resfork))

# ...1 to 3 times
blocks = RepeatArchSpace(repforkres, 1:3)

# End each block with subsamping through maxpooling
ms = VertexSpace(PoolSpace{2}(windowsizes=2, strides=2, poolfuns=MaxPool))
reduction = ArchSpaceChain(blocks, ms)

# And lets do 2 to 4 reductions
featureextract = RepeatArchSpace(reduction, 2:4)

# Adds 1 to 3 dense layers as outputs
dense = VertexSpace(DenseSpace(16:512, [relu, selu]))
drep = RepeatArchSpace(dense, 0:2)
# Last layer has fixed output size (number of labels)
dout=VertexSpace(Shielded(), DenseSpace(10, identity))
output = ArchSpaceChain(drep, dout)

# Aaaand lets glue it together: Feature extracting conv+bn layers -> global pooling -> dense layers
archspace = ArchSpaceChain(featureextract, GlobalPoolSpace(), output)

# Input is 3 channel image
inputshape = conv2dinputvertex("input", 3)

# Sample one architecture from the search space
graph1 = CompGraph(inputshape, archspace(inputshape))
@test nvertices(graph1) == 79

# And one more...
graph2 = CompGraph(inputshape, archspace(inputshape))
@test nvertices(graph2) == 128
```

### Mutation

Mutation is the way one candidate is transformed to a slightly different candidate. NaiveGAflux supports doing this while preserving parameters and alignment between layers, thus reducing the impact of mutating an already trained candidate.

The following basic mutation operations are currently supported:
1. Change the output size of vertices using `NoutMutation`.
2. Remove vertices using `RemoveVertexMutation`.
3. Add vertices using `AddVertexMutation`.
4. Remove edges between vertices using `RemoveEdgeMutation`.
5. Add edges between vertices using `AddEdgeMutation`.
6. Mutation of kernel size for conv layers using `KernelSizeMutation`.
7. Change of activation function using `ActivationFunctionMutation`.
8. Change the type of optimizer using `OptimizerMutation`.
9. Add an optimizer using `AddOptimizerMutation`.

In addition to the basic mutation operations, there are numerous utilities for adding behaviour and convenience. Here are a few examples:

```julia
Random.seed!(NaiveGAflux.rng_default, 1)

invertex = denseinputvertex("in", 3)
layer1 = fluxvertex(Dense(nout(invertex), 4), invertex)
layer2 = fluxvertex(Dense(nout(layer1), 5), layer1)
graph = CompGraph(invertex, layer2)

mutation = NoutMutation(-0.5, 0.5)

@test nout(layer2) == 5

mutation(layer2)

@test nout(layer2) == 4

# VertexMutation applies the wrapped mutation to all vertices in a CompGraph
mutation = VertexMutation(mutation)

@test nout.(vertices(graph)) == [3,4,4]

mutation(graph)

# Input vertex is never mutated
@test nout.(vertices(graph)) == [3,3,3]

# Use the MutationShield trait to protect vertices from mutation
outlayer = fluxvertex(Dense(nout(layer2), 10), layer2, traitfun = MutationShield)
graph = CompGraph(invertex, outlayer)

mutation(graph)

@test nout.(vertices(graph)) == [3,2,2,10]

# In most cases it makes sense to mutate with a certain probability
mutation = VertexMutation(MutationProbability(NoutMutation(-0.5, 0.5), 0.5))

mutation(graph)

@test nout.(vertices(graph)) == [3,3,2,10]

# Or just chose to either mutate the whole graph or don't do anything
mutation = MutationProbability(VertexMutation(NoutMutation(-0.5, 0.5)), 0.98)

mutation(graph)

@test nout.(vertices(graph)) == [3,4,3,10]
@test size(graph(ones(3,1))) == (10, 1)

# Mutation can also be conditioned:
mutation = VertexMutation(MutationFilter(v -> nout(v) < 4, RemoveVertexMutation()))

mutation(graph)

@test nout.(vertices(graph)) == [3,4,10]

# When adding vertices it is probably a good idea to try to initialize them as identity mappings
addmut = AddVertexMutation(VertexSpace(DenseSpace(5, identity)), IdentityWeightInit())

# Chaining mutations is also useful:
noutmut = NoutMutation(-0.8, 0.8)
mutation = VertexMutation(MutationChain(addmut, noutmut))

mutation(graph)

@test nout.(vertices(graph)) == [3,3,4,10]
```

### Crossover

Crossover is the way two candidates are combined to create new candidates. In NaiveGAflux crossover always maps two candidates into two new candidates. Just as for mutation, NaiveGAflux does this while preserving (to whatever extent possible) the parameters and alignment between layers of the combined models.

Crossover operations might not seem to make much sense when using parameter inheritance (i.e the concept that children retain the parameters of their parents). Randomly combining layers from two very different models will most likely not result in a well performing model. There are however a few potentially redeeming effects:

* Early in the evolution process parameters are not yet well fitted and inheriting parameters is not worse than random initialization
* A mature population on the other hand will consist mostly of models which are close relatives and therefore have somewhat similar weights.

Whether these effects actually make crossover a genuinely useful operation when evolving neural networks is not yet proven though. For now it is perhaps best to view the crossover operations as being provided mostly for the sake of completeness.

The following basic crossover operations are currently supported:
1. Swap segments between two models using `CrossoverSwap`.
2. Swap optimizers between two candidates using `OptimizerCrossover`.
3. Swap learning rate between two candidates using `LearningRateCrossover`.

Most of the mutation utilities also work with crossover operations. Here are a few examples:

```julia
import Functors
Random.seed!(NaiveGAflux.rng_default, 0)

invertex = denseinputvertex("A.in", 3)
layer1 = fluxvertex("A.layer1", Dense(nout(invertex), 4), invertex; layerfun=ActivationContribution)
layer2 = fluxvertex("A.layer2", Dense(nout(layer1), 5), layer1; layerfun=ActivationContribution)
layer3 = fluxvertex("A.layer3", Dense(nout(layer2), 3), layer2; layerfun=ActivationContribution)
layer4 = fluxvertex("A.layer4", Dense(nout(layer3), 2), layer3; layerfun=ActivationContribution)
modelA = CompGraph(invertex, layer4)

# Create an exact copy to show how parameter alignment is preserved
# Prefix names with B so we can show that something actually happened
modelB = Functors.fmap(x -> x isa String ? replace(x, r"^A.\.*" => "B.") : x, modelA)

indata = reshape(collect(Float32, 1:3*2), 3,2)
@test modelA(indata) == modelB(indata)

@test name.(vertices(modelA)) == ["A.in", "A.layer1", "A.layer2", "A.layer3", "A.layer4"]
@test name.(vertices(modelB)) == ["B.in", "B.layer1", "B.layer2", "B.layer3", "B.layer4"]

# CrossoverSwap takes ones vertex from each graph as input and swaps a random segment from each graph
# By default it tries to make segments as similar as possible
swapsame = CrossoverSwap()

swapA = vertices(modelA)[4]
swapB = vertices(modelB)[4]
newA, newB = swapsame((swapA, swapB))

# It returns vertices of a new graph to be compatible with mutation utilities
# Parent models are not modified
@test newA ∉ vertices(modelA)
@test newB ∉ vertices(modelB)

# This is an internal utility which should not be needed in normal use cases.
modelAnew = NaiveGAflux.regraph(newA)
modelBnew = NaiveGAflux.regraph(newB)

@test name.(vertices(modelAnew)) == ["A.in", "A.layer1", "B.layer2", "B.layer3", "A.layer4"] 
@test name.(vertices(modelBnew)) == ["B.in", "B.layer1", "A.layer2", "A.layer3", "B.layer4"]

@test modelA(indata) == modelB(indata) == modelAnew(indata) == modelBnew(indata)

# Deviation parameter will randomly make segments unequal
swapdeviation = CrossoverSwap(0.5)
modelAnew2, modelBnew2 = regraph.(swapdeviation((swapA, swapB)))

@test name.(vertices(modelAnew2)) == ["A.in", "A.layer1", "A.layer2", "B.layer1", "B.layer2", "B.layer3", "A.layer4"] 
@test name.(vertices(modelBnew2)) == ["B.in", "A.layer3", "B.layer4"]

# VertexCrossover applies the wrapped crossover operation to all vertices in a CompGraph
# It in addtion, it selects compatible pairs for us (i.e swapA and swapB).
# It also takes an optional deviation parameter which is used when pairing
crossoverall = VertexCrossover(swapdeviation, 0.5)

modelAnew3, modelBnew3 = crossoverall((modelA, modelB))

# I guess things got swapped back and forth so many times not much changed in the end
@test name.(vertices(modelAnew3)) == ["A.in", "A.layer2", "A.layer4"]
@test name.(vertices(modelBnew3)) ==  ["B.in", "B.layer3", "B.layer1", "B.layer2", "A.layer1", "A.layer3", "B.layer4"] 

# As advertised above, crossovers interop with most mutation utilities, just remember that input is a tuple
# Perform the swapping operation with a 30% probability for each valid vertex pair.
crossoversome = VertexCrossover(MutationProbability(LogMutation(((v1,v2)::Tuple) -> "Swap $(name(v1)) and $(name(v2))", swapdeviation), 0.3))

@test_logs (:info, "Swap A.layer1 and B.layer1") (:info, "Swap A.layer2 and B.layer2") crossoversome((modelA, modelB))
```

### Fitness functions

A handful of ways to compute the fitness of a model are supplied. Apart from the obvious accuracy on some (typically held out) data set, it is also possible to measure fitness as how many (few) parameters a model has and how long it takes to compute the fitness. Fitness metrics can of course be combined to create objectives which balance several factors.

As seen in the very first basic example above, training of a model is just another fitness strategy. This might seem unintuitive at first, but reading it out like "the chosen fitness strategy is to first train the model for N batches, then compute the accuracy on the validation set" makes sense. The practical advantages are that it becomes straight forward to implement fitness strategies which don't involve model training (e.g. using the neural tangent kernel) as well as fitness strategies which measure some aspect of the model training (e.g. time to train for X iterations or training memory consumption). Another useful property is that models which produce `NaN`s or `Inf`s or take very long to train can be assigned a low fitness score.

Examples:

```julia
# Function to compute fitness for does not have to be a CompGraph, or even a neural network
# They must be wrapped in an AbstractCandidate since fitness functions generally need to query the candidate for 
# things which affect the fitness, such as the model but also things like optimizers and loss functions.
candidate1 = CandidateModel(x -> 3:-1:1)
candidate2 = CandidateModel(Dense(ones(Float32, 3,3), collect(Float32, 1:3)))

# Fitness is accuracy on the provided data set
accfitness = AccuracyFitness([(ones(Float32, 3, 1), 1:3)])

@test fitness(accfitness, candidate1) == 0
@test fitness(accfitness, candidate2) == 1

# Measure how long time it takes to evaluate the fitness and add that in addition to the accuracy
let timedfitness = TimeFitness(accfitness)
    c1time, c1acc = fitness(timedfitness, candidate1)
    c2time, c2acc = fitness(timedfitness, candidate2) 
    @test c1acc == 0
    @test c2acc == 1
    @test 0 < c1time 
    @test 0 < c2time 
end

# Use the number of parameters to compute fitness
bigmodelfitness = SizeFitness()
@test fitness(bigmodelfitness, candidate1) == 0
@test fitness(bigmodelfitness, candidate2) == 12

# One typically wants to map high number of params to lower fitness:
smallmodelfitness = MapFitness(bigmodelfitness) do nparameters
    return min(1, 1 / nparameters)
end
@test fitness(smallmodelfitness, candidate1) == 1
@test fitness(smallmodelfitness, candidate2) == 1/12

# Combining fitness is straight forward
combined = AggFitness(+, accfitness, smallmodelfitness, bigmodelfitness)

@test fitness(combined, candidate1) == 1
@test fitness(combined, candidate2) == 13 + 1/12

# GpuFitness moves the candidates to GPU (as selected by Flux.gpu) before computing the wrapped fitness
# Note that any data in the wrapped fitness must also be moved to the same GPU before being fed to the model
gpuaccfitness = GpuFitness(AccuracyFitness(GpuIterator(accfitness.dataset)))

@test fitness(gpuaccfitness, candidate1) == 0
@test fitness(gpuaccfitness, candidate2) == 1

```

### Candidate utilities

As seen above, fitness strategies require an `AbstractCandidate` to compute fitness. To be used by NaiveGAflux, an `AbstractCandidate` needs to 
1. Provide the data needed by the fitness strategy, most commonly the model but also things like lossfunctions and optimizers
2. Be able to create a new version of itself given a function which maps its fields to new fields.

Capability 1. is generally performed through functions of the format `someproperty(candidate; default)` where in general `someproperty(::AbstractCandidate; default=nothing) = default`. The following such functions are currently implemented by NaiveGAflux:
* `graph(c; default)`  : Return a model
* `opt(c; default)`    : Return an optimizer
* `lossfun(c; default)` : Return a lossfunction

All such functions are obviously not used by all fitness strategies and some are used more often than others. Whether an `AbstractCandidate` returns something other than `default` generally depends on whether it is a hyperparameter which is being searched for or not. For example, the very simple `CandidateModel` has only a `model` while `CandidateOptModel` has both a model and an own optimizer which may be mutated/crossedover when evolving.

Capability 2. is what is used then evolving a candidate into a new version of itself. The function to implement for new `AbstractCandidate` types is `newcand(c::MyCandidate, mapfields)` which in most cases has the implementation `newcand(c::MyCandidate, mapfield) = MyCandidate(map(mapfield, getproperty.(c, fieldnames(MyCandidate)))...)`.

Example with a new candidate type and a new fitness strategy for said type:
```julia
struct ExampleCandidate <: AbstractCandidate
    a::Int
    b::Int
end
aval(c::ExampleCandidate; default=nothing) = c.a
bval(c::ExampleCandidate; default=nothing) = c.b

struct ExampleFitness <: AbstractFitness end
NaiveGAflux._fitness(::ExampleFitness, c::AbstractCandidate) = aval(c; default=10) - bval(c; default=5)

# Ok, this is alot of work for quite little in this dummy example
@test fitness(ExampleFitness(), ExampleCandidate(4, 3)) === 1

ctime, examplemetric = fitness(TimeFitness(ExampleFitness()), ExampleCandidate(3,1))
@test examplemetric === 2
@test ctime > 0
```

### Evolution Strategies

Evolution strategies are the functions used to evolve the population in the genetic algorithm from one generation to the next. The following is performed by evolution strategies:

* Select which candidates to use for the next generation
* Produce new candidates, e.g by mutating the selected candidates

Important to note about evolution strategies is that they generally expect candidates which can provide a precomputed fitness value, e.g. `FittedCandidate`s. This is because the fitness value is used by things like sorting where it is not only impractical to recompute it, but is also might lead to undefined behaviour if it is not always the same. Use `Population` to get some help with computing fitness for all candidates before passing them on to evolution.

Note that there is no general requirement on an evolution strategy to return the same population size as it was given. It is also free to create completely new candidates without basing anything on any given candidate.

Examples:

```julia
# For controlled randomness in the examples
struct FakeRng end
Base.rand(::FakeRng) = 0.7

# Dummy candidate for brevity
struct Cand <: AbstractCandidate
    fitness
end
NaiveGAflux.fitness(d::Cand) = d.fitness

# EliteSelection selects the n best candidates
elitesel = EliteSelection(2)
@test evolve(elitesel, Cand.(1:10)) == Cand.([10, 9])

# EvolveCandidates maps candidates to new candidates (e.g. through mutation)
evocands = EvolveCandidates(c -> Cand(fitness(c) + 0.1))
@test evolve(evocands, Cand.(1:10)) == Cand.(1.1:10.1)

# SusSelection selects n random candidates using stochastic uniform sampling
# Selected candidates will be forwarded to the wrapped evolution strategy before returned
sussel = SusSelection(5, evocands, FakeRng())
@test evolve(sussel, Cand.(1:10)) == Cand.([4.1, 6.1, 8.1, 9.1, 10.1])

# CombinedEvolution combines the populations from several evolution strategies
comb = CombinedEvolution(elitesel, sussel)
@test evolve(comb, Cand.(1:10)) == Cand.(Any[10, 9, 4.1, 6.1, 8.1, 9.1, 10.1])
```

### Iterators

While not part of the scope of this package, some simple utilities for iterating over data sets is provided.

The only iterator which is in some sense special for this package is `RepeatPartitionIterator` which produces iterators over a subset of its wrapped iterator. This is useful when one wants to ensure that all models see the same (possibly randomly augmented) data in the same order. Note that this is not certain to be the best strategy for finding good models for a given data set and this package does (intentionally) blur the lines a bit between model training protocol and architecture search.

Examples:

```julia
data = reshape(collect(1:4*5), 4,5)

# mini-batching
biter = BatchIterator(data, 2)
@test size(first(biter)) == (4, 2)

# shuffle data before mini-batching
# Warning: Must use different rng instances with the same seed for features and labels!
siter = ShuffleIterator(data, 2, MersenneTwister(123))
@test size(first(siter)) == size(first(biter))
@test first(siter) != first(biter)

# Apply a function to each batch
miter = MapIterator(x -> 2 .* x, biter)
@test first(miter) == 2 .* first(biter)

# Move data to gpu
giter = GpuIterator(miter)
@test first(giter) == first(miter) |> gpu

labels = collect(0:5)

# Possible to use Flux.onehotbatch for many iterators
biter_labels = Flux.onehotbatch(BatchIterator(labels, 2), 0:5)
@test first(biter_labels) == Flux.onehotbatch(0:1, 0:5)

# This is the only iterator which is "special" for this package:
rpiter = RepeatPartitionIterator(zip(biter, biter_labels), 2)
# It produces iterators over a subset of the wrapped iterator (2 batches in this case)
piter = first(rpiter)
@test length(piter) == 2
# This allows for easily training several models on the same subset of the data
expiter = zip(biter, biter_labels)
for modeli in 1:3
    for ((feature, label), (expf, expl)) in zip(piter, expiter)
        @test feature == expf
        @test label == expl
    end
end

# StatefulGenerationIter is typically used in conjunction with TrainThenFitness to map a generation
# number to an iterator from a RepeatStatefulIterator 
sgiter = StatefulGenerationIter(rpiter)
for (generationnr, topiter) in enumerate(rpiter)
    gendata = collect(NaiveGAflux.itergeneration(sgiter, generationnr))
    expdata = collect(topiter)
    @test gendata == expdata
end

# Timed iterator is useful for preventing that models which take very long time 
# to train slow down the process. We can just stop training them in that case
timediter = TimedIterator(;timelimit=0.1, patience=4, timeoutaction = () -> TimedIteratorStop, accumulate_timeouts=false, base=1:100)

last = 0
for i in timediter
    last = i
    if i > 2
        sleep(0.11)
    end
end
@test last === 6 # Sleep after 2, then 4 patience
```

## Contributing

All contributions are welcome. Please file an issue before creating a PR.
