module NaiveGAflux

using NaiveNASflux
using Random

# types
export AbstractMutation, Probability, VertexMutation

# functions
export mutate

include("util.jl")
include("mutation.jl")


end # module
