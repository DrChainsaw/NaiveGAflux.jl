module NaiveGAflux

using Reexport
@reexport using NaiveNASflux
using Random
using Logging

const rng_default = Random.GLOBAL_RNG

# misc types
export Probability, MutationShield

# Vertex selection types
export AbstractVertexSelection, AllVertices, FilterMutationAllowed

# mutation types
export AbstractMutation, MutationProbability, MutationList, RecordMutation, LogMutation, VertexMutation, NoutMutation, AddVertexMutation, RemoveVertexMutation, NeuronSelectMutation, select

# architecture spaces
export AbstractArchSpace, VertexSpace, ArchSpace, RepeatArchSpace, ListArchSpace, ForkArchSpace, ResidualArchSpace

#  Other search space types
export BaseLayerSpace, AbstractParSpace, SingletonParSpace, Singleton2DParSpace, ParSpace, ParSpace1D, ParSpace2D, CoupledParSpace, AbstractPadSpace, SamePad, NamedLayerSpace, DenseSpace, ConvSpace, ConvSpace2D, BatchNormSpace, PoolSpace, PoolSpace2D, MaxPoolSpace, LayerVertexConf, ConcConf

# functions
export mutate, allow_mutation, select

# Examples
export Cifar10

include("util.jl")
include("archspace.jl")
include("mutation.jl")

include("examples/Cifar10.jl")


end # module
