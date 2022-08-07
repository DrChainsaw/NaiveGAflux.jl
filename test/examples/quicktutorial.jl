md"""
# Quick Tutorial

Here is a very basic example just to get a feeling for the package. We set up a simple search for 
number of fully connected layers and their widths.
"""

@testset "Basic example" begin #src
using NaiveGAflux, Flux, Random
Random.seed!(NaiveGAflux.rng_default, 0)

nlabels = 3
ninputs = 5

# #### Step 1: Create initial models.
# Search space: 2-4 dense layers of width 3-10.
layerspace = VertexSpace(DenseSpace(3:10, [identity, relu, elu, selu]))
initial_hidden = RepeatArchSpace(layerspace, 1:3)
# Output layer has fixed size and is shielded from mutation.
outlayer = VertexSpace(Shielded(), DenseSpace(nlabels, identity))
initial_searchspace = ArchSpaceChain(initial_hidden, outlayer)

# Sample 5 models from the initial search space and make an initial population.
samplemodel(invertex) = CompGraph(invertex, initial_searchspace(invertex))
models = [samplemodel(denseinputvertex("input", ninputs)) for _ in 1:5]
@test nvertices.(models) == [4, 3, 4, 5, 3]

population = Population(CandidateModel.(models))
@test generation(population) == 1

# #### Step 2: Set up fitness function:
# Train model for one epoch using `datasettrain`, then measure accuracy on `datasetvalidate`.
# We use dummy data here just to make stuff run.
onehot(y) = Flux.onehotbatch(y, 1:nlabels)
batchsize = 4
datasettrain    = [(randn(ninputs, batchsize), onehot(rand(1:nlabels, batchsize)))]
datasetvalidate = [(randn(ninputs, batchsize), onehot(rand(1:nlabels, batchsize)))]

fitnessfunction = TrainThenFitness(;
    dataiter = datasettrain,
    defaultloss = Flux.logitcrossentropy, # Will be used if not provided by the candidate
    defaultopt = Adam(), # Same as above. State is wiped after training to prevent memory leaks
    fitstrat = AccuracyFitness(datasetvalidate) # This is what creates our fitness value after training
)

# #### Step 3: Define how to search for new candidates
# We choose to evolve the existing ones through mutation.

# [`VertexMutation`](@ref) selects valid vertices from the graph to mutate.
# [`MutationProbability`](@ref) applies mutation `m` with a probability of `p`.
# Lets create a short helper for brevity:
mp(m, p) = VertexMutation(MutationProbability(m, p))
# Change size (60% chance) and/or add a layer (40% chance) and/or remove a layer (40% chance).
# You might want to use lower probabilities than this.
changesize = mp(NoutMutation(-0.2, 0.2), 0.6)
addlayer = mp(AddVertexMutation(layerspace), 0.4)
remlayer = mp(RemoveVertexMutation(), 0.4)
mutation = MutationChain(changesize, remlayer, addlayer)


# Selection: The two best models are not changed, then create three new models by 
# applying the mutations above to three of the five models with higher fitness 
# giving higher probability of being selected. 
#
# [`MapCandidate`](@ref) helps with the plumbing of creating new `CandidateModel`s
#  where `mutation` is applied to create a new model. 
elites = EliteSelection(2)
mutate = SusSelection(3, EvolveCandidates(MapCandidate(mutation)))
selection = CombinedEvolution(elites, mutate)

# #### Step 4: Run evolution
newpopulation = evolve(selection, fitnessfunction, population)
@test newpopulation != population
@test generation(newpopulation) == 2
# Repeat until a model with the desired fitness is found.
newnewpopulation = evolve(selection, fitnessfunction, newpopulation)
@test newnewpopulation != newpopulation
@test generation(newnewpopulation) == 3
bestfitness, bestcandnr = findmax(fitness, newnewpopulation)
@test model(newnewpopulation[bestcandnr]) isa CompGraph
# Maybe in a loop :)
end #src