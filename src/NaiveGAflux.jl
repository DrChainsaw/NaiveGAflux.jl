module NaiveGAflux

using Reexport
@reexport using NaiveNASflux
using Random
using Logging
using Statistics

using Setfield

# For temporary storage of program state for pause/resume type of operations
using Serialization

# For longer term storage of models
using FileIO
using JLD2


if Flux.has_cuarrays()
    using CuArrays
end

const rng_default = Random.GLOBAL_RNG
const modeldir = "models"

# Fitness
export fitness, instrument, reset!, AbstractFitness, AccuracyFitness, TrainAccuracyFitness, MapFitness, TimeFitness, SizeFitness, FitnessCache, NanGuard, AggFitness

# Candidate
export evolvemodel, AbstractCandidate, CandidateModel, HostCandidate, CacheCandidate

# Evolution
export evolve!, AbstractEvolution, NoOpEvolution, AfterEvolution, ResetAfterEvolution, EliteSelection, SusSelection, CombinedEvolution, EvolveCandidates

# misc types
export Probability, MutationShield, ApplyIf, RemoveIfSingleInput, RepeatPartitionIterator, MapIterator, GpuIterator, BatchIterator, FlipIterator, ShiftIterator, ShuffleIterator, PersistentArray

# Persistence
export persist, savemodels

# Vertex selection types
export AbstractVertexSelection, AllVertices, FilterMutationAllowed

# mutation types
export AbstractMutation, MutationProbability, WeightedMutationProbability, HighValueMutationProbability, LowValueMutationProbability, MutationList, RecordMutation, LogMutation, MutationFilter, PostMutation, VertexMutation, NoutMutation, AddVertexMutation, RemoveVertexMutation, AddEdgeMutation, RemoveEdgeMutation, KernelSizeMutation, KernelSizeMutation2D, ActivationFunctionMutation, NeuronSelectMutation, PostMutation, NeuronSelect, RemoveZeroNout

# mutation auxillaries
export NeuronSelect, RemoveZeroNout

# architecture spaces
export AbstractArchSpace, LoggingArchSpace, VertexSpace, ArchSpace, RepeatArchSpace, ListArchSpace, ForkArchSpace, ResidualArchSpace, FunVertex, GpVertex2D

#  Other search space types
export BaseLayerSpace, AbstractParSpace, SingletonParSpace, Singleton2DParSpace, ParSpace, ParSpace1D, ParSpace2D, CoupledParSpace, AbstractPadSpace, SamePad, NamedLayerSpace, LoggingLayerSpace, DenseSpace, ConvSpace, ConvSpace2D, BatchNormSpace, PoolSpace, PoolSpace2D, MaxPoolSpace, LayerVertexConf, Shielded, ConcConf

#weight inits
export AbstractWeightInit, DefaultWeightInit, IdentityWeightInit, PartialIdentityWeightInit, ZeroWeightInit

# functions
export mutate, allow_mutation, select, check_apply, nparams

# Pre-built programs for fitting data
export AutoFlux

# Visulization
export PlotFitness, ScatterPop, ScatterOpt, MultiPlot, CbAll

# Examples
export Cifar10, Mnist

include("util.jl")
include("archspace.jl")
include("mutation.jl")
include("fitness.jl")
include("candidate.jl")
include("evolve.jl")
include("iterators.jl")
include("app/AutoFlux.jl")
include("visualize/callbacks.jl")

include("examples/Cifar10.jl")
include("examples/Mnist.jl")

end # module
