module NaiveGAflux

using Base: release
using Reexport
@reexport using NaiveNASflux
using NaiveNASlib: name
using NaiveNASflux: FluxDense, FluxConv, FluxConvolutional, FluxNoParLayer, FluxParNorm, FluxRnn, FluxBatchNorm
using NaiveNASflux: nograd, layertype
using NaiveNASlib.Advanced, NaiveNASlib.Extend
import Flux
using Flux: Dense, Conv, ConvTranspose, DepthwiseConv, CrossCor, LayerNorm, BatchNorm, InstanceNorm, GroupNorm, 
            MaxPool, MeanPool, Dropout, AlphaDropout, GlobalMaxPool, GlobalMeanPool, cpu, gpu,
            SamePad
import Optimisers
import Optimisers: WeightDecay
using Random
using Logging
using Statistics
import MemPool
import IterTools
import Functors
using Functors: @functor, functor, fmap
using Printf

using Setfield

# For temporary storage of program state for pause/resume type of operations
using Serialization

const rng_default = MersenneTwister(1)
const modeldir = "models"

# Fitness
export fitness, AbstractFitness, LogFitness, GpuFitness, AccuracyFitness, TrainThenFitness, TrainAccuracyFitness, MapFitness
export EwmaFitness, TimeFitness, SizeFitness, AggFitness

# Candidate
export AbstractCandidate, CandidateModel, CandidateOptModel, CandidateDataIterMap, FittedCandidate, FileCandidate, MapCandidate, model, opt, lossfun

# Evolution
export evolve, AbstractEvolution, NoOpEvolution, AfterEvolution, EliteSelection, SusSelection, TournamentSelection, CombinedEvolution
export EvolutionChain, PairCandidates, ShuffleCandidates, EvolveCandidates

# Population
export Population, generation

# misc types
export Probability, MutationShield, ApplyIf, RemoveIfSingleInput, PersistentArray, ShieldedOpt, ImplicitOpt

# Batch size selection
export BatchSizeSelectionWithDefaultInShape, BatchSizeSelectionScaled, BatchSizeSelectionFromAlternatives, BatchSizeSelectionMaxSize, batchsizeselection

# Iterators. These should preferably come from somewhere else, but I haven't found anything which fits the bill w.r.t repeatability over subsets
export RepeatPartitionIterator, SeedIterator, MapIterator, GpuIterator, BatchIterator, TimedIterator, TimedIteratorStop
export StatefulGenerationIter

# Iterator mapping types for evolving hyperparameters related to datasets, e.g. augmentation and batch size
export BatchSizeIteratorMap, IteratorMaps, ShieldedIteratorMap

# Persistence
export persist

# Vertex selection types
export AbstractVertexSelection, AllVertices, FilterMutationAllowed

# generic mutation types
export AbstractMutation, MutationProbability, WeightedMutationProbability, HighUtilityMutationProbability, LowUtilityMutationProbability 
export MutationChain, RecordMutation, LogMutation, MutationFilter
# graph mutation types
export VertexMutation, NoutMutation, AddVertexMutation, RemoveVertexMutation, AddEdgeMutation, RemoveEdgeMutation, KernelSizeMutation
export KernelSizeMutation2D, ActivationFunctionMutation
# optimiser mutation types
export OptimiserMutation, LearningRateMutation, AddOptimiserMutation
# Iterator wrapping mutation types
export TrainBatchSizeMutation

# Crossover types
export AbstractCrossover, VertexCrossover, CrossoverSwap, OptimiserCrossover, LearningRateCrossover, IteratorMapCrossover

# architecture spaces
export AbstractArchSpace, LoggingArchSpace, VertexSpace, NoOpArchSpace, ArchSpace, ConditionalArchSpace, RepeatArchSpace, ArchSpaceChain
export ForkArchSpace, ResidualArchSpace, FunctionSpace, GlobalPoolSpace

#  Other search space types
export BaseLayerSpace, AbstractParSpace, SingletonParSpace, Singleton2DParSpace, ParSpace, ParSpace1D, ParSpace2D, CoupledParSpace
export NamedLayerSpace, LoggingLayerSpace, DenseSpace, ConvSpace, BatchNormSpace, PoolSpace, LayerVertexConf, Shielded, ConcConf

#weight inits
export AbstractWeightInit, DefaultWeightInit, IdentityWeightInit, PartialIdentityWeightInit, ZeroWeightInit

# functions
export nparams

# Pre-built programs for fitting data
export AutoFlux

# Visulization
export PlotFitness, ScatterPop, ScatterOpt, ScatterBatchSize, MultiPlot, CbAll

# This should come from NaiveNASflux once it matures (or be deleted if turned obsolete) 
include("autooptimiser.jl")
import .AutoOptimiserExperimental
import .AutoOptimiserExperimental: AutoOptimiser, optimisersetup!, mutateoptimiser!
export AutoOptimiser

include("util.jl")
include("shape.jl")
include("batchsize.jl")
include("iteratormaps.jl")
include("archspace.jl")
include("mutation/generic.jl")
include("mutation/graph.jl")
include("mutation/optimiser.jl")
include("mutation/iteratormaps.jl")
include("crossover/generic.jl")
include("crossover/graph.jl")
include("crossover/optimiser.jl")
include("crossover/iteratormaps.jl")
include("candidate.jl")
include("fitness.jl")
include("evolve.jl")
include("population.jl")
include("iterators.jl")
include("app/AutoFlux.jl")
include("visualize/callbacks.jl")


using PackageExtensionCompat
function __init__()
    @require_extensions
end

include("precompile.jl")

end # module
