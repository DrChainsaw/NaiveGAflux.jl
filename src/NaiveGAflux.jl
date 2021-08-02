module NaiveGAflux

using Reexport
@reexport using NaiveNASflux
using NaiveNASflux: FluxDense, FluxConv, FluxConvolutional, FluxNoParLayer, FluxParNorm, FluxRnn, FluxBatchNorm
using NaiveNASflux: nograd, layertype
using NaiveNASlib.Advanced, NaiveNASlib.Extend
import Flux
using Flux: Dense, Conv, ConvTranspose, DepthwiseConv, CrossCor, LayerNorm, BatchNorm, InstanceNorm, GroupNorm, 
            MaxPool, MeanPool, Dropout, AlphaDropout, GlobalMaxPool, GlobalMeanPool, cpu, gpu, WeightDecay,
            SamePad, params
using Random
using Logging
using Statistics
import CUDA
import MemPool
import IterTools
import Functors
using Functors: @functor, functor, fmap
using Printf

using Setfield

# For temporary storage of program state for pause/resume type of operations
using Serialization

const rng_default = MersenneTwister(abs(rand(Int)))
const modeldir = "models"

# Fitness
export fitness, AbstractFitness, LogFitness, GpuFitness, AccuracyFitness, TrainThenFitness, TrainAccuracyFitness, MapFitness, EwmaFitness, TimeFitness, SizeFitness, AggFitness

# Candidate
export evolvemodel, AbstractCandidate, CandidateModel, CandidateOptModel

# Evolution
export evolve, AbstractEvolution, NoOpEvolution, AfterEvolution, EliteSelection, SusSelection, TournamentSelection, CombinedEvolution, EvolutionChain, PairCandidates, ShuffleCandidates, EvolveCandidates

# Population
export Population, generation

# misc types
export Probability, MutationShield, ApplyIf, RemoveIfSingleInput, PersistentArray, ShieldedOpt

# Iterators. These should preferably come from somewhere else, but I haven't found anything which fits the bill w.r.t repeatability over subsets
export RepeatPartitionIterator, SeedIterator, MapIterator, GpuIterator, BatchIterator, ShuffleIterator, TimedIterator, TimedIteratorStop, StatefulGenerationIter

# Persistence
export persist

# Vertex selection types
export AbstractVertexSelection, AllVertices, FilterMutationAllowed

# mutation types
export AbstractMutation, MutationProbability, WeightedMutationProbability, HighValueMutationProbability, LowValueMutationProbability, MutationChain, RecordMutation, LogMutation, MutationFilter, PostMutation, VertexMutation, NoutMutation, AddVertexMutation, RemoveVertexMutation, AddEdgeMutation, RemoveEdgeMutation, KernelSizeMutation, KernelSizeMutation2D, ActivationFunctionMutation, PostMutation, OptimizerMutation, LearningRateMutation, AddOptimizerMutation

# Crossover types
export AbstractCrossover, VertexCrossover, CrossoverSwap, OptimizerCrossover, LearningRateCrossover

# mutation auxillaries
export neuronselect, RemoveZeroNout

# architecture spaces
export AbstractArchSpace, LoggingArchSpace, VertexSpace, NoOpArchSpace, ArchSpace, ConditionalArchSpace, RepeatArchSpace, ArchSpaceChain, ForkArchSpace, ResidualArchSpace, FunctionSpace, GlobalPoolSpace

#  Other search space types
export BaseLayerSpace, AbstractParSpace, SingletonParSpace, Singleton2DParSpace, ParSpace, ParSpace1D, ParSpace2D, CoupledParSpace, NamedLayerSpace, LoggingLayerSpace, DenseSpace, ConvSpace, BatchNormSpace, PoolSpace, LayerVertexConf, Shielded, ConcConf

#weight inits
export AbstractWeightInit, DefaultWeightInit, IdentityWeightInit, PartialIdentityWeightInit, ZeroWeightInit

# functions
export allow_mutation, select, check_apply, nparams

# Pre-built programs for fitting data
export AutoFlux

# Visulization
export PlotFitness, ScatterPop, ScatterOpt, MultiPlot, CbAll

include("util.jl")
include("shape.jl")
include("archspace.jl")
include("mutation.jl")
include("crossover.jl")
include("candidate.jl")
include("fitness.jl")
include("evolve.jl")
include("population.jl")
include("iterators.jl")
include("app/AutoFlux.jl")
include("visualize/callbacks.jl")

end # module
