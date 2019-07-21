module NaiveGAflux

using Reexport
@reexport using NaiveNASflux
using Random

# misc types
export Probability, MutationShield

# Vertex selection types
export AbstractVertexSelection, AllVertices, FilterMutationAllowed

# mutation types
export AbstractMutation, MutationProbability, VertexMutation

# functions
export mutate, allow_mutation, select

include("util.jl")
include("mutation.jl")


end # module
