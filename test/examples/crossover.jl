md"""
# Crossover Operations

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
"""

@testset "Crossover examples" begin #src
# Start by creating a model to play with.
Random.seed!(NaiveGAflux.rng_default, 0) #src

invertex = denseinputvertex("A.in", 3)
layer1 = fluxvertex("A.l1", Dense(nout(invertex), 4), invertex; layerfun=ActivationContribution)
layer2 = fluxvertex("A.l2", Dense(nout(layer1), 5), layer1; layerfun=ActivationContribution)
layer3 = fluxvertex("A.l3", Dense(nout(layer2), 3), layer2; layerfun=ActivationContribution)
layer4 = fluxvertex("A.l4", Dense(nout(layer3), 2), layer3; layerfun=ActivationContribution)
modelA = CompGraph(invertex, layer4)

# Create an exact copy to show how parameter alignment is preserved.
# Prefix names with B so we can show that something actually happened.
import Functors
modelB = Functors.fmap(x -> x isa String ? replace(x, r"^A.\.*" => "B.") : x, modelA)

indata = reshape(collect(Float32, 1:3*2), 3,2)
@test modelA(indata) == modelB(indata)

@test name.(vertices(modelA)) == ["A.in", "A.l1", "A.l2", "A.l3", "A.l4"]
@test name.(vertices(modelB)) == ["B.in", "B.l1", "B.l2", "B.l3", "B.l4"]

# `CrossoverSwap` takes ones vertex from each graph as input and swaps a random segment from each graph.
# By default it tries to make segments as similar as possible
swapsame = CrossoverSwap()

swapA = vertices(modelA)[4]
swapB = vertices(modelB)[4]
newA, newB = swapsame((swapA, swapB))

# It returns vertices of a new graph to be compatible with mutation utilities.
# Parent models are not modified.
@test newA ∉ vertices(modelA)
@test newB ∉ vertices(modelB)

# The function `regraph` is an internal utility which should not be needed in normal use cases, but here we use it to make comparison easier.
modelAnew = NaiveGAflux.regraph(newA)
modelBnew = NaiveGAflux.regraph(newB)

@test name.(vertices(modelAnew)) == ["A.in", "A.l1", "B.l2", "B.l3", "A.l4"] 
@test name.(vertices(modelBnew)) == ["B.in", "B.l1", "A.l2", "A.l3", "B.l4"]

@test modelA(indata) == modelB(indata) == modelAnew(indata) == modelBnew(indata)

# Deviation parameter will randomly make segments unequal.
swapdeviation = CrossoverSwap(0.5)
modelAnew2, modelBnew2 = NaiveGAflux.regraph.(swapdeviation((swapA, swapB)))

@test name.(vertices(modelAnew2)) == ["A.in", "A.l1", "A.l2", "B.l1", "B.l2", "B.l3", "A.l4"] 
@test name.(vertices(modelBnew2)) == ["B.in", "A.l3", "B.l4"]

# `VertexCrossover` applies the wrapped crossover operation to all vertices in a `CompGraph` and is the main API for doing crossover.
# It in addtion, it selects compatible pairs for us (i.e `swapA` and `swapB` above).
# It also takes an optional deviation parameter which is used when pairing.
crossoverall = VertexCrossover(swapdeviation, 0.5)

modelAnew3, modelBnew3 = crossoverall((modelA, modelB))

# I guess things got swapped back and forth so many times not much changed in the end.
@test name.(vertices(modelAnew3)) == ["A.in", "A.l2", "A.l4"]
@test name.(vertices(modelBnew3)) ==  ["B.in", "B.l3", "B.l1", "B.l2", "A.l1", "A.l3", "B.l4"] 

# As advertised above, crossovers interop with most mutation utilities, just remember that input is a tuple for things which require a callback.
# Perform the swapping operation with a 30% probability for each valid vertex pair.
crossoversome = VertexCrossover(
                    MutationProbability(
                        LogMutation(
                            ((v1,v2),) -> "Swap $(name(v1)) and $(name(v2))",
                            swapdeviation),
                    0.3)
                )

@test_logs( (:info, "Swap A.l1 and B.l1"),
            (:info, "Swap A.l2 and B.l2"),
            crossoversome((modelA, modelB)))
end #src