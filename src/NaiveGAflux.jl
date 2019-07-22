module NaiveGAflux

using Reexport
@reexport using NaiveNASflux
using Random

const rng_default = Random.GLOBAL_RNG

# misc types
export Probability, MutationShield

# Vertex selection types
export AbstractVertexSelection, AllVertices, FilterMutationAllowed

# mutation types
export AbstractMutation, MutationProbability, RecordMutation, VertexMutation, NoutMutation

# architecture specifications
export AbstractArchSpec

# architecture spec config types
export BaseLayerSpec, AbstractParSpec, FixedNDParSpec, Fixed2DParSpec, ParNDSpec, Par2DSpec, AbstractPadSpec, SamePad, ConvSpec, Conv2DSpec

# functions
export mutate, allow_mutation, select

include("util.jl")
include("mutation.jl")
include("archspec.jl")

end # module
