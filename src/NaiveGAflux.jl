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

# architecture spaces
export AbstractArchSpace, VertexSpace, ArchSpace, RepeatArchSpace

# architecture space config types
export BaseLayerSpace, AbstractParSpace, SingletonParSpace, Singleton2DParSpace, ParSpace, ParSpace1D, ParSpace2D, AbstractPadSpace, SamePad, DenseSpace, ConvSpace, ConvSpace2D, BatchNormSpace, PoolSpace, PoolSpace2D, MaxPoolSpace, VertexConf

# functions
export mutate, allow_mutation, select

include("util.jl")
include("mutation.jl")
include("archspace.jl")

end # module
