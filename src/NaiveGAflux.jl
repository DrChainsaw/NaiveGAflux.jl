module NaiveGAflux

using Reexport
@reexport using NaiveNASflux
using Random
using Logging
using Statistics
using Serialization

if Flux.has_cuarrays()
    using CuArrays
end

const rng_default = Random.GLOBAL_RNG
const modeldir = "models"

# Fitness
export fitness, instrument, reset!, AbstractFitness, AccuracyFitness, MapFitness, TimeFitness, FitnessCache, NanGuard, AggFitness

# Candidate
export evolvemodel, AbstractCandidate, CandidateModel, HostCandidate, CacheCandidate

# Evolution
export evolve!, AbstractEvolution, NoOpEvolution, AfterEvolution, ResetAfterEvolution, EliteSelection, SusSelection, CombinedEvolution, EvolveCandidates

# misc types
export Probability, MutationShield, ApplyIf, RemoveIfSingleInput, RepeatPartitionIterator, MapIterator, GpuIterator, BatchIterator, FlipIterator, ShiftIterator, PersistentArray, persist

# Vertex selection types
export AbstractVertexSelection, AllVertices, FilterMutationAllowed

# mutation types
export AbstractMutation, MutationProbability, MutationList, RecordMutation, LogMutation, MutationFilter, PostMutation, VertexMutation, NoutMutation, AddVertexMutation, RemoveVertexMutation, NeuronSelectMutation, select, PostMutation, NeuronSelect, RemoveZeroNout

# mutation auxillaries
export select, NeuronSelect, RemoveZeroNout

# architecture spaces
export AbstractArchSpace, LoggingArchSpace, VertexSpace, ArchSpace, RepeatArchSpace, ListArchSpace, ForkArchSpace, ResidualArchSpace, FunVertex, GpVertex2D

#  Other search space types
export BaseLayerSpace, AbstractParSpace, SingletonParSpace, Singleton2DParSpace, ParSpace, ParSpace1D, ParSpace2D, CoupledParSpace, AbstractPadSpace, SamePad, NamedLayerSpace, LoggingLayerSpace, IdSpace, DenseSpace, ConvSpace, ConvSpace2D, BatchNormSpace, PoolSpace, PoolSpace2D, MaxPoolSpace, LayerVertexConf, ConcConf

# functions
export mutate, allow_mutation, select, check_apply

# Examples
export Cifar10

include("util.jl")
include("archspace.jl")
include("mutation.jl")
include("candidate.jl")
include("iterators.jl")
include("examples/Cifar10.jl")

end # module
