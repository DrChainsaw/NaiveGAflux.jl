md"""
# Mutation Operations

Mutation is the way one candidate is transformed to a slightly different candidate. NaiveGAflux supports doing this while 
preserving parameters and alignment between layers, thus reducing the impact of mutating an already trained candidate.

The following basic mutation operations are currently supported:
1. Change the output size of vertices using [`NoutMutation`](@ref).
2. Remove vertices using [`RemoveVertexMutation`](@ref).
3. Add vertices using [`AddVertexMutation`](@ref).
4. Remove edges between vertices using [`RemoveEdgeMutation`](@ref).
5. Add edges between vertices using [`AddEdgeMutation`](@ref).
6. Mutation of kernel size for conv layers using [`KernelSizeMutation`](@ref).
7. Change of activation function using [`ActivationFunctionMutation`](@ref).
8. Change the type of optimizer using [`OptimizerMutation`](@ref).
9. Add an optimizer using [`AddOptimizerMutation`](@ref).

Mutation operations are exported as structs rather than functions since they are designed to be composed with more generic utilities. Here are a few examples:
"""

@testset "Mutation examples" begin #src
Random.seed!(NaiveGAflux.rng_default, 1) #src

# Start with a simple model to mutate.
invertex = denseinputvertex("in", 3)
layer1 = fluxvertex(Dense(nout(invertex), 4), invertex)
layer2 = fluxvertex(Dense(nout(layer1), 5), layer1)
graph = CompGraph(invertex, layer2)

# Create an `NoutMutation` to mutate it.
mutation = NoutMutation(-0.5, 0.5)

@test nout(layer2) == 5

mutation(layer2)

@test nout(layer2) == 4

# `VertexMutation` applies the wrapped mutation to all vertices in a `CompGraph`
mutation = VertexMutation(mutation)

@test nout.(vertices(graph)) == [3,4,4]

mutation(graph)

@test nout.(vertices(graph)) == [3,3,3]

# Input vertex is never mutated, but the other two changed.
# Use the `MutationShield` trait to protect otherwise mutable vertices from mutation.
outlayer = fluxvertex(Dense(nout(layer2), 10), layer2, traitfun = MutationShield)
graph = CompGraph(invertex, outlayer)

mutation(graph)

@test nout.(vertices(graph)) == [3,2,2,10]

# In most cases it makes sense to mutate with a certain probability.
mutation = VertexMutation(MutationProbability(NoutMutation(-0.5, 0.5), 0.5))

mutation(graph)

@test nout.(vertices(graph)) == [3,3,2,10]

# Or just chose to either mutate the whole graph or don't do anything.
mutation = MutationProbability(VertexMutation(NoutMutation(-0.5, 0.5)), 0.98)

mutation(graph)

@test nout.(vertices(graph)) == [3,4,3,10]
@test size(graph(ones(3,1))) == (10, 1)

# Mutation can also be conditioned:
mutation = VertexMutation(MutationFilter(v -> nout(v) < 4, RemoveVertexMutation()))

mutation(graph)

@test nout.(vertices(graph)) == [3,4,10]

# When adding vertices it is probably a good idea to try to initialize them as identity mappings.
addmut = AddVertexMutation(VertexSpace(DenseSpace(5, identity)), IdentityWeightInit())

# Chaining mutations is also useful:
noutmut = NoutMutation(-0.8, 0.8)
mutation = VertexMutation(MutationChain(addmut, noutmut))

mutation(graph)

@test nout.(vertices(graph)) == [3,3,4,10]
end #src