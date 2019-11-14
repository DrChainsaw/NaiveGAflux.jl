# NaiveGAflux

[![Build Status](https://travis-ci.com/DrChainsaw/NaiveGAflux.jl.svg?branch=master)](https://travis-ci.com/DrChainsaw/NaiveGAflux.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/DrChainsaw/NaiveGAflux.jl?svg=true)](https://ci.appveyor.com/project/DrChainsaw/NaiveGAflux-jl)
[![Codecov](https://codecov.io/gh/DrChainsaw/NaiveGAflux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DrChainsaw/NaiveGAflux.jl)

Neural architecture search for [Flux](https://github.com/FluxML/Flux.jl) models using genetic algorithms.

A marketing person might describe it as "practical proxyless NAS using an unrestricted search space".

The more honest purpose is to serve as a pipe cleaner and example for [NaiveNASflux](https://github.com/DrChainsaw/NaiveNASflux.jl) which is doing most of the heavy lifting.

## Basic Usage

```julia
Pkg.add("https://github.com/DrChainsaw/NaiveGAflux.jl")
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

However, most non-toy uses cases will probably require a dedicated application. NaiveGAflux provides the components to make building it easy and fun!

Tired of tuning hyperparameters? Once you've felt the rush from reasoning about hyper-hyperparameters there is no going back!

This package has the following main components:
1. [Search spaces](#search-spaces)
2. [Mutation operations](#mutation)
3. [Fitness functions](#fitness-functions)
4. [Candidate utilities](#candidate-utilities)
5. [Evolution strategies](#evolution-strategies)
6. [Iterators](#iterators)

Each component is described more in detail below.

Here is a very basic example just to get a feeling for the package:

```julia
using NaiveGAflux, Random
Random.seed!(NaiveGAflux.rng_default, 123)

nlabels = 3
ninputs = 5
inshape = inputvertex("input", ninputs, FluxDense())

# Step 1: Create initial models
layerspace = VertexSpace(DenseSpace(3:10, [identity, relu, elu, selu]))
initial_hidden = RepeatArchSpace(layerspace, 1:3)
# Output layer has fixed size and is shielded from mutation
outlayer = VertexSpace(Shielded(), DenseSpace(nlabels, identity))
initial_searchspace = ListArchSpace(initial_hidden, outlayer)

# Sample 5 models from the initial search space
models = [CompGraph(inshape, initial_searchspace(inshape)) for _ in 1:5]
@test nv.(models) == [3, 5, 3, 4, 5]

# Some dummy data just to make stuff run
batchsize = 4
dataset = (randn(ninputs, batchsize), Flux.onehotbatch(rand(1:nlabels, batchsize), 1:nlabels))

# Not recommended to measure fitness on the training data for real usage.
fitfun = AccuracyFitness([dataset])
opt = Flux.Descent(0.01) # All models use the same optimizer here. Would be a bad idea with a stateful optimizer!
loss = Flux.logitcrossentropy
population = [CandidateModel(model, opt, loss, fitfun) for model in models]

# Step 2: Train the models
for candidate in population
    Flux.train!(candidate, dataset)
end

# Step 3: Evolve the population

# Mutations
mp(m, p) = VertexMutation(MutationProbability(m, p))
# You probably want to use lower probabilities than this
addlayer = mp(AddVertexMutation(layerspace), 0.4)
remlayer = mp(RemoveVertexMutation(), 0.4)

mutation = MutationList(remlayer, addlayer)

# Selection
elites = EliteSelection(2)
mutate = SusSelection(3, EvolveCandidates(evolvemodel(mutation)))
selection = CombinedEvolution(elites, mutate)

# And evolve
newpopulation = evolve!(selection, population)
@test newpopulation != population

# Repeat steps 2 and 3 until a model with the desired fitness is found.
```

### Search Spaces

The search space is a set of possible architectures which the search policy may use to create initial candidates or to extend existing candidates. Search spaces are constructed from simple components which can be combined in multiple ways, giving a lot of flexibility.

Lets start with the most simple search space, a `ParSpace`:

```julia
# Set seed of default random number generator for reproducible results
using NaiveGAflux, Random
Random.seed!(NaiveGAflux.rng_default, 123)

ps1d = ParSpace([2,4,6,10])

# Draw from the search space
@test ps1d() == 2
@test ps1d() == 6

# Possible to supply another rng than the default one
@test ps1d(MersenneTwister(0)) == 2

# Can be of any dimension and type
ps2d = ParSpace(["1","2","3"], ["4","5","6","7"])

@test typeof(ps1d) == ParSpace{1, Int64}
@test typeof(ps2d) == ParSpace{2, String}

@test ps2d() == ("3", "6")
```

Lets have a look at an example of a search space for convolutional layers:

```julia
Random.seed!(NaiveGAflux.rng_default, 1)

outsizes = 4:32
kernelsizes = 3:9
cs = ConvSpace2D(outsizes, [relu, elu, selu], kernelsizes)

@test typeof(cs) == ConvSpace{2}

inputsize = 16
convlayer = cs(inputsize)

@test string(convlayer) == "Conv((5, 4), 16=>18, NNlib.elu)"
```

Lastly, lets look at how to construct a complex search space:

```julia
Random.seed!(NaiveGAflux.rng_default, 666)

# VertexSpace creates a MutableVertex of layers generated by the wrapped search space
cs = VertexSpace(ConvSpace2D(8:256, [identity, relu, elu], 3:5))
bs = VertexSpace(BatchNormSpace([identity, relu]))

# Block of conv->bn and bn->conv respectively.
# Need to make sure there is always at least one SizeAbsorb layer to make fork and res below play nice
csbs = ListArchSpace(cs ,bs)
bscs = ListArchSpace(bs, cs)

# Randomly generates a conv-block:
cblock = ArchSpace(ParSpace1D(cs, csbs, bscs))

# Generates between 1 and 5 layers from cblock
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
ms = VertexSpace(MaxPoolSpace(PoolSpace2D([2])))
reduction = ListArchSpace(blocks, ms)

# And lets do 2 to 4 reductions
featureextract = RepeatArchSpace(reduction, 2:4)

# Adds 1 to 3 dense layers as outputs
dense = VertexSpace(DenseSpace(16:512, [relu, selu]))
drep = RepeatArchSpace(dense, 0:2)
# Last layer has fixed output size (number of labels)
dout=VertexSpace(Shielded(), DenseSpace(10, identity))
output = ListArchSpace(drep, dout)

# Aaaand lets glue it together: Feature extracting conv+bn layers -> global pooling -> dense layers
archspace = ListArchSpace(featureextract, GpVertex2D(), output)

# Input is a 3 channel image
inputshape = inputvertex("input", 3, FluxConv{2}())

# Sample one architecture from the search space
graph1 = CompGraph(inputshape, archspace(inputshape))
@test nv(graph1) == 123

# And one more...
graph2 = CompGraph(inputshape, archspace(inputshape))
@test nv(graph2) == 75
```

### Mutation

Mutation is the way one existing candidate is transformed to a slightly different candidate. NaiveGAflux supports doing this while preserving weights and alignment between layers, thus reducing the impact of mutating an already trained candidate.

The following basic mutation operations are currently supported:
1. Change the output size of vertices using `NoutMutation`.
2. Remove vertices using `RemoveVertexMutation`.
3. Add vertices using `AddVertexMutation`.
4. Remove edges between vertices using `RemoveEdgeMutation`.
5. Add edges between vertices using `AddEdgeMutation`.
6. Mutation of kernel size for conv layers using `KernelSizeMutation`.
7. Change of activation function using `ActivationFunctionMutation`.

It is also possible to implement mutation of learning rate and optimizer using `evolve_candidate`, but a convenient way to do this still TBA, see `AutoFlux.ImageClassification.evolvecandidate` in the meantime.

In addition to the basic mutation operations, there are numerous utilities for adding behaviour and convenience. Here are a few examples:

```julia
using Random
Random.seed!(NaiveGAflux.rng_default, 0)

invertex = inputvertex("in", 3, FluxDense())
layer1 = mutable(Dense(nout(invertex), 4), invertex)
layer2 = mutable(Dense(nout(layer1), 5), layer1)
graph = CompGraph(invertex, layer2)

mutation = NoutMutation(-0.5, 0.5)

@test nout(layer2) == 5

mutation(layer2)

@test nout(layer2) == 6

# VertexMutation applies the wrapped mutation to all vertices in a CompGraph
mutation = VertexMutation(mutation)

@test nout.(vertices(graph)) == [3,4,6]

mutation(graph)

# Input vertex is never mutated
@test nout.(vertices(graph)) == [3,5,4]

# Use the MutationShield trait to protect vertices from mutation
outlayer = mutable(Dense(nout(layer2), 10), layer2, traitfun = MutationShield)
graph = CompGraph(invertex, outlayer)

mutation(graph)

@test nout.(vertices(graph)) == [3,4,3,10]

# In most cases it makes sense to mutate with a certain probability
mutation = VertexMutation(MutationProbability(NoutMutation(-0.5, 0.5), 0.05))

mutation(graph)

@test nout.(vertices(graph)) == [3,4,2,10]

# Or just chose to either mutate the whole graph or don't do anything
mutation = MutationProbability(VertexMutation(NoutMutation(-0.5, 0.5)), 0.05)

mutation(graph)

@test nout.(vertices(graph)) == [3,4,2,10]

# Up until now, size changes have only been kept track of, but not actually applied
@test nout_org.(vertices(graph)) == [3,4,5,10]

Δoutputs(graph, v -> ones(nout_org(v)))
apply_mutation(graph)

@test nout_org.(vertices(graph)) == [3,4,2,10]
@test size(graph(ones(3,1))) == (10, 1)

# NeuronSelectMutation keeps track of changed vertices and performs the above steps when invoked
mutation = VertexMutation(NeuronSelectMutation(NoutMutation(-0.5,0.5)))

mutation(graph)

@test nout.(vertices(graph)) == [3,5,3,10]
@test nout_org.(vertices(graph)) == [3,4,2,10]

select(mutation.m)

@test nout.(vertices(graph)) == nout_org.(vertices(graph)) == [3,4,2,10]
@test size(graph(ones(3,1))) == (10, 1)

# Mutations can also be conditioned:
mutation = VertexMutation(MutationFilter(v -> nout(v) < 4, RemoveVertexMutation()))

mutation(graph)

@test nout.(vertices(graph)) == [3,5,10]

# When adding vertices it is probably a good idea to try to initialize them as identity mappings
addmut = AddVertexMutation(VertexSpace(DenseSpace(5, identity)), IdentityWeightInit())

# Chaining mutations is also useful:
noutmut = NeuronSelectMutation(NoutMutation(-0.8, 0.8))
mutation = VertexMutation(MutationList(addmut, noutmut))

# For deeply composed blobs like this, it can be cumbersome to "dig up" the NeuronSelectMutation.
# NeuronSelect helps finding NeuronSelectMutations in the compositional hierarchy
neuronselect = NeuronSelect()

# PostMutation lets us add actions to perform after a mutation is done
logselect(m, g) = @info "Selecting parameters..."
mutation = PostMutation(mutation, logselect, neuronselect)

@test_logs (:info, "Selecting parameters...") mutation(graph)

@test nout.(vertices(graph)) == nout_org.(vertices(graph)) == [3,6,5,10]
```

### Fitness functions

A handful of ways to compute the fitness of a model are supplied. Apart from the obvious accuracy on some (typically held out) data set, it is also possible to measure fitness as how many (few) parameters a model has and how long it takes on average to perform a forward/backward pass. Fitness metrics can of course be combined to create objectives which balance several factors.

As seen below, some fitness functions are not trivial to use. [Candidate utilities](#candidate-utilities) helps managing this complexity behind a much simpler API.

Examples:

```julia
# Function to compute fitness for does not have to be a CompGraph, or even a neural network
  candidate1 = x -> 3:-1:1
  candidate2 = Dense(param(ones(3,3)), param(1:3))

  # Fitness is accuracy on the provided data set
  accfitness = AccuracyFitness([(ones(3, 1), 1:3)])

  @test fitness(accfitness, candidate1) == 0
  @test fitness(accfitness, candidate2) == 1

  # Measure how long time it takes to train the function
  import NaiveGAflux: Train, Validate
  timetotrain = TimeFitness(Train())

  # No training done yet...
  @test fitness(timetotrain, candidate1) == 0
  @test fitness(timetotrain, candidate2) == 0

  # There is no magic involved here, we need to "instrument" the function to measure
  candidate2_timed = instrument(Train(), timetotrain, candidate2)

  # Instrumented function produces same result as the original function...
  @test candidate2_timed(ones(3,1)) == candidate2((ones(3,1)))
  # ... and TimeFitness measures time elapsed in the background
  @test fitness(timetotrain, candidate2) > 0

  # Just beware that it is not very clever, it just stores the time when a function it instrumented was run...
  @test fitness(timetotrain, sleep(0.2)) == fitness(timetotrain, sleep(0.01))

  # ... and it needs to be reset before being used for another candidate
  # In practice you probably want to create one instance per candidate
  reset!(timetotrain)
  @test fitness(timetotrain, candidate1) == 0

  # One typically wants to map short time to high fitness.
  timefitness = MapFitness(x -> x == 0 ? 0 : 1/(x*1e6), timetotrain)

  # Will see to it so that timetotrain gets to instrument the function
  candidate2_timed = instrument(Train(), timefitness, candidate2)

  @test candidate2_timed(ones(3,1)) == candidate2(ones(3,1))
  @test fitness(timefitness, candidate2) > 0

  # This also propagates ofc
  reset!(timefitness)
  @test fitness(timefitness, candidate2) == 0

  # Use the number of parameters to compute fitness
  nparams = SizeFitness()

  @test fitness(nparams, candidate2) == 12

  # This does not work unfortunately, and it tends to happen when combining fitness functions due to instrumentation
  @test (@test_logs (:warn, "SizeFitness got zero parameters! Check your fitness function!") fitness(nparams, candidate2_timed)) == 0

  # The mitigation for this is to "abuse" the instrumentation API
  instrument(Validate(), nparams, candidate2)
  @test fitness(nparams, candidate2_timed) == 12

  # This however adds state which needs to be reset shall the function be used for something else
  @test fitness(nparams, sum) == 12
  reset!(nparams)
  @test fitness(nparams, param(1:3)) == 3

  # Combining fitness is straight forward
  # Note that one typically wants to map low number of parameters to high fitness (omitted here for brevity)
  combined = AggFitness(+, accfitness, nparams, timefitness)

  @test fitness(combined, candidate2) == 13

  # instrumentation will be aggregated as well
  candidate2_timed = instrument(Train(), combined, candidate2)

  @test candidate2_timed(ones(3,1)) == candidate2(ones(3,1))
  @test fitness(combined, candidate2) > 13

  # Special mention goes to NanGuard.
  # It is hard to ensure that evolution does not produce a model which outputs NaN or Inf.
  # However, Flux typically throws an exception if it sees NaN or Inf.
  # NanGuard keeps the show going and assigns fitness 0 so that the model will not be selected.
  nanguard = NanGuard(combined)

  training_guarded = instrument(Train(), nanguard, candidate2)
  validation_guarded = instrument(Validate(), nanguard, candidate2)

  @test training_guarded(ones(3,1)) == validation_guarded(ones(3,1)) == candidate2(ones(3,1))

  # Now the model gets corrupted somehow...
  candidate2.W.data[1,1] = NaN

  @test any(isnan, candidate2(ones(3,1)))

  @test (@test_logs (:warn, "NaN detected for function with label Train()") training_guarded(ones(3,1))) == zeros(3,1)

  @test (@test_logs (:warn, "NaN detected for function with label Validate()") validation_guarded(ones(3,1))) == zeros(3,1)

  @test fitness(nanguard, candidate2) == 0

  # After a Nan is detected the function will no longer be evaluated until reset
  candidate2.W.data[1,1] = 1

  @test !any(isnan, candidate2(ones(3,1)))
  @test training_guarded(ones(3,1)) == zeros(3,1)
  @test validation_guarded(ones(3,1)) == zeros(3,1)
  @test fitness(nanguard, candidate2) == 0

  reset!(nanguard)
  @test training_guarded(ones(3,1)) == validation_guarded(ones(3,1)) == candidate2(ones(3,1))
```

### Candidate utilities

The main component of a candidate is the model itself of course. There are however a few convenience utilities around candidate handling which may be useful.

As seen in the previous section, some fitness functions are not straight forward to use. By wrapping a model, a fitness function, a loss function and an optimizer in a `CandidateModel`, NaiveGAflux will hide much of the complexity and reduce the API to the following:

* `train!(candidate, data)`
* `fitness(candidate)`

Examples:

```julia
using Random
Random.seed!(NaiveGAflux.rng_default, 0)

archspace = RepeatArchSpace(VertexSpace(DenseSpace(3, elu)), 2)
inpt = inputvertex("in", 3)
dataset = (ones(Float32, 3, 1), Float32[0, 1, 0])

graph = CompGraph(inpt, archspace(inpt))
opt = Flux.ADAM(0.1)
loss = Flux.logitcrossentropy
fitfun = NanGuard(AccuracyFitness([dataset]))

# CandidateModel is the most basic candidate and handles things like fitness instrumentation
candmodel = CandidateModel(graph, opt, loss, fitfun)

Flux.train!(candmodel, Iterators.repeated(dataset, 20))
@test fitness(candmodel) > 0

# HostCandidate moves the model to the GPU when training or evaluating fitness and moves it back afterwards
# Useful for reducing GPU memory consumption (at the cost of longer time to train as cpu<->gpu move takes some time).
# Note, it does not move the data. GpuIterator can provide some assistance here...
dataset_gpu = GpuIterator([dataset])
fitfun_gpu = NanGuard(AccuracyFitness(dataset_gpu))
hostcand = HostCandidate(CandidateModel(graph, Flux.ADAM(0.1), loss, fitfun_gpu))

Flux.train!(hostcand, dataset_gpu)
@test fitness(hostcand) > 0

# CacheCandidate is a practical necessity if using AccuracyFitness.
# It caches the last computed fitness value so it is not recomputed every time fitness is called
cachinghostcand = CacheCandidate(hostcand)

Flux.train!(cachinghostcand, dataset_gpu)
@test fitness(cachinghostcand) > 0
```

### Evolution Strategies

Evolution strategies are the functions used to evolve the population in the genetic algorithm from one generation to the next. The following is performed by evolution strategies:

* Select which candidates to use for the next generation
* Mutate the selected candidates
* Reset/clear state so that the population is prepared for the next generation

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
@test evolve!(elitesel, Cand.(1:10)) == Cand.([10, 9])

# EvolveCandidates maps candidates to new candidates (e.g. through mutation)
evocands = EvolveCandidates(c -> Cand(fitness(c) + 0.1))
@test evolve!(evocands, Cand.(1:10)) == Cand.(1.1:10.1)

# SusSelection selects n random candidates using stochastic uniform sampling
# Selected candidates will be forwarded to the wrapped evolution strategy before returned
sussel = SusSelection(5, evocands, FakeRng())
@test evolve!(sussel, Cand.(1:10)) == Cand.([4.1, 6.1, 8.1, 9.1, 10.1])

# CombinedEvolution combines the populations from several evolution strategies
comb = CombinedEvolution(elitesel, sussel)
@test evolve!(comb, Cand.(1:10)) == Cand.(Any[10, 9, 4.1, 6.1, 8.1, 9.1, 10.1])

# AfterEvolution calls a function after evolution is completed
afterfun(pop) = map(c -> Cand(2fitness(c)), pop)
afterevo = AfterEvolution(comb, afterfun)
@test evolve!(afterevo, Cand.(1:10)) == Cand.(Any[20, 18, 8.2, 12.2, 16.2, 18.2, 20.2])

# Its mainly intended for resetting
ntest = 0
NaiveGAflux.reset!(::Cand) = ntest += 1

resetafter = ResetAfterEvolution(comb)
@test evolve!(resetafter, Cand.(1:10)) == Cand.(Any[10, 9, 4.1, 6.1, 8.1, 9.1, 10.1])
@test ntest == 7
```

### Iterators

While not part of the scope of this package, some simple utilities for iterating over data sets is provided.

The only iterator which is in some sense special for this package is `RepeatPartitionIterator` which produces iterators over a subset of its wrapped iterator. This is useful when there is a non-negligible cost of "switching" models, for example if `HostCandidate` is used as it allows training one model on the whole data subset before moving on to the next model.

Examples:

```julia
data = reshape(collect(1:4*5), 4,5)

# mini-batching
biter = BatchIterator(data, 2)
@test size(first(biter)) == (4, 2)

# shuffle data before mini-batching
# Warning 1: Data will be shuffled inplace!
# Warning 2: Must use different rng instances with the same seed for features and labels!
siter = ShuffleIterator(copy(data), 2, MersenneTwister(123))
@test size(first(siter)) == size(first(biter))
@test first(siter) != first(biter)

# Flip data along a dimension with a certain probability
probability = 1.0
dimension = 1
fiter = FlipIterator(biter, probability, dimension)
@test first(fiter) == reverse(first(biter), dims=dimension)

# Randomly shift the data while cropping and padding to keep the same size
maxshiftdim1 = 2
maxshiftdim2 = 0
siter = ShiftIterator(biter, maxshiftdim1,maxshiftdim2; rng = MersenneTwister(12))
sdata = first(siter)
@test sdata[1:1,:] == zeros(1,2)
@test sdata[2:4,:] == first(biter)[1:3,:]

# Apply a function to each batch
miter = MapIterator(x -> 2 .* x, biter)
@test first(miter) == 2 .* first(biter)

# Move data to gpu
giter = GpuIterator(miter)
@test first(giter) == first(miter)

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

```

## Contributing

All contributions are welcome. Please file an issue before creating a PR.
