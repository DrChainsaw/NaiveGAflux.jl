md"""
# Fitness Functions

A handful of ways to compute the fitness of a model are supplied. Apart from the obvious accuracy on some
(typically held out) data set, it is also possible to measure fitness as how many (few) parameters a model
has and how long it takes to compute the fitness. Fitness metrics can of course be combined to create 
objectives which balance several factors.

As seen in the very first basic example above, training of a model is just another fitness strategy. This 
might seem unintuitive at first, but reading it out like "the chosen fitness strategy is to first train the 
model for N batches, then compute the accuracy on the validation set" makes sense. The practical advantages 
are that it becomes straight forward to implement fitness strategies which don't involve model training 
(e.g. using the neural tangent kernel) as well as fitness strategies which measure some aspect of the model 
training (e.g. time to train for X iterations or training memory consumption). Another useful property is 
that models which produce `NaN`s or `Inf`s or take very long to train can be assigned a low fitness score.

Examples:
"""
@testset "Fitness functions" begin #src
# Function to compute fitness for does not have to be a `CompGraph`, or even a neural network.
# They must be wrapped in an `AbstractCandidate` since fitness functions generally need to query the candidate for 
# things which affect the fitness, such as the model but also things like optimizers and loss functions.
candidate1 = CandidateModel(x -> 3:-1:1)
candidate2 = CandidateModel(Dense(ones(Float32, 3,3), collect(Float32, 1:3)))

# [`AccuracyFitness`](@ref) compuates fitness as accuracy on the provided data set.
accfitness = AccuracyFitness([(ones(Float32, 3, 1), 1:3)])

@test fitness(accfitness, candidate1) == 0
@test fitness(accfitness, candidate2) == 1

# [`TimeFitness`](@ref) measures how long time it takes to evaluate the fitness and add that in addition to the accuracy. 
timedfitness = TimeFitness(accfitness)
c1time, c1acc = fitness(timedfitness, candidate1)
c2time, c2acc = fitness(timedfitness, candidate2) 

@test c1acc == 0
@test c2acc == 1

@test c1time > 0
@test c2time > 0

# [`SizeFitness`](@ref) uses the number of parameters to compute fitness.
bigmodelfitness = SizeFitness()
@test fitness(bigmodelfitness, candidate1) == 0
@test fitness(bigmodelfitness, candidate2) == 12

# One typically wants to map high number of params to lower fitness. [`MapFitness`](@ref) allow use to remap the fitness value from another fitness function.
smallmodelfitness = MapFitness(bigmodelfitness) do nparameters
    return min(1, 1 / nparameters)
end
@test fitness(smallmodelfitness, candidate1) == 1
@test fitness(smallmodelfitness, candidate2) == 1/12

# Combining fitness is straight forward with [`AggFitness`](@ref)
combined = AggFitness(+, accfitness, smallmodelfitness, bigmodelfitness)

@test fitness(combined, candidate1) == 1
@test fitness(combined, candidate2) == 13 + 1/12

# [`GpuFitness`](@ref) moves the candidates to GPU (as selected by `Flux.gpu`) before computing the wrapped fitness.
# Note that any data in the wrapped fitness must also be moved to the same GPU before being fed to the model.
gpuaccfitness = GpuFitness(AccuracyFitness(GpuIterator(accfitness.dataset)))

@test fitness(gpuaccfitness, candidate1) == 0
@test fitness(gpuaccfitness, candidate2) == 1
end #src