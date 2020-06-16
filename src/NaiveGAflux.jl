module NaiveGAflux

using Reexport
@reexport using NaiveNASflux
import NaiveNASflux: nograd
using Random
using Logging
using Statistics
using CuArrays
import MemPool

using Setfield

# For temporary storage of program state for pause/resume type of operations
using Serialization

const rng_default = MersenneTwister(abs(rand(Int)))
const modeldir = "models"

# Fitness
export fitness, instrument, reset!, AbstractFitness, AccuracyFitness, TrainAccuracyFitness, MapFitness, EwmaFitness, TimeFitness, SizeFitness, FitnessCache, NanGuard, AggFitness

# Candidate
export evolvemodel, AbstractCandidate, CandidateModel, HostCandidate, CacheCandidate

# Evolution
export evolve!, AbstractEvolution, NoOpEvolution, AfterEvolution, ResetAfterEvolution, EliteSelection, SusSelection, TournamentSelection, CombinedEvolution, PairCandidates, EvolveCandidates

# Population
export Population, generation

# misc types
export Probability, MutationShield, ApplyIf, RemoveIfSingleInput, RepeatPartitionIterator, SeedIterator, MapIterator, GpuIterator, BatchIterator, FlipIterator, ShiftIterator, ShuffleIterator, PersistentArray, ShieldedOpt

# Persistence
export persist

# Vertex selection types
export AbstractVertexSelection, AllVertices, FilterMutationAllowed

# mutation types
export AbstractMutation, MutationProbability, WeightedMutationProbability, HighValueMutationProbability, LowValueMutationProbability, MutationList, RecordMutation, LogMutation, MutationFilter, PostMutation, VertexMutation, NoutMutation, AddVertexMutation, RemoveVertexMutation, AddEdgeMutation, RemoveEdgeMutation, KernelSizeMutation, KernelSizeMutation2D, ActivationFunctionMutation, NeuronSelectMutation, PostMutation, OptimizerMutation, LearningRateMutation, AddOptimizerMutation

# Crossover types
export AbstractCrossover, VertexCrossover, CrossoverSwap

# mutation auxillaries
export neuronselect, RemoveZeroNout

# architecture spaces
export AbstractArchSpace, LoggingArchSpace, VertexSpace, ArchSpace, RepeatArchSpace, ListArchSpace, ForkArchSpace, ResidualArchSpace, FunctionSpace, GlobalPoolSpace

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

include("util.jl")
include("shape.jl")
include("archspace.jl")
include("mutation.jl")
include("crossover.jl")
include("fitness.jl")
include("candidate.jl")
include("evolve.jl")
include("population.jl")
include("iterators.jl")
include("app/AutoFlux.jl")
include("visualize/callbacks.jl")

end # module
