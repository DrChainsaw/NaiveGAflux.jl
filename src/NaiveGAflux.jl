module NaiveGAflux

using Reexport
@reexport using NaiveNASflux
using Random
using Logging

# For solving pesky entangled neuron select problems. To be moved to NaiveNASlib if things work out
import JuMP
import JuMP: @variable, @constraint, @objective, @expression, MOI, MOI.INFEASIBLE, MOI.FEASIBLE_POINT
using Cbc


const rng_default = Random.GLOBAL_RNG

# misc types
export Probability, MutationShield

# Vertex selection types
export AbstractVertexSelection, AllVertices, FilterMutationAllowed

# mutation types
export AbstractMutation, MutationProbability, MutationList, RecordMutation, LogMutation, MutationFilter, PostMutation, VertexMutation, NoutMutation, AddVertexMutation, RemoveVertexMutation, NeuronSelectMutation, select, PostMutation, NeuronSelect, RemoveZeroNout

# mutation auxillaries
export select, NeuronSelect, RemoveZeroNout

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
include("selection.jl")

include("examples/Cifar10.jl")


end # module
