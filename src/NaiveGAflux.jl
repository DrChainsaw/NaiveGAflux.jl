module NaiveGAflux

using Reexport
@reexport using NaiveNASflux
using Random

# misc types
export Probability, MutationShield

# mutation types
export AbstractMutation, MutationProbability, VertexMutation

# functions
export mutate, allow_mutation

include("util.jl")
include("mutation.jl")


end # module
