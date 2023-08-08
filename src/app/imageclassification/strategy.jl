"""
    AbstractFitnessStrategy

Base type for fitness strategies.

See [`fitnessfun`](@ref)
"""
abstract type AbstractFitnessStrategy end

"""
    fitnessfun(s::T, x, y) where T <: AbstractFitnessStrategy

Returns an `AbstractFitness` given the features `x` and labels `y`.

Allows for simple control over e.g. train vs validation split.
"""
fitnessfun(s::T, x, y) where T <: AbstractFitnessStrategy = error("Not implemented for $(T)!")

"""
    AbstractEvolutionStrategy

Base type for strategies on how to perform the evolution.

See [`evostrategy`](@ref)
"""
abstract type AbstractEvolutionStrategy end

"""
    evostrategy(s::AbstractEvolutionStrategy, inshape)

Returns an `AbstractEvolution`.

Argument `inshape` is the size of the input feature maps (i.e. how many pixels images are) and may be used to determine which mutation operations are allowed for example to avoid that feature maps accidentally become 0 sized.
"""
evostrategy(s::AbstractEvolutionStrategy, inshape) = evostrategy_internal(s, inshape)

"""
    struct TrainSplitAccuracy{T} <: AbstractFitnessStrategy
        TrainSplitAccuracy(;split, accuracyconfig, accuracyfitness, trainconfig, trainfitness)

Strategy to train model on a subset of the training data and measure fitness as accuracy on the rest.

Size of subset for accuracy fitness is `ceil(Int, split * nobs)` where `nobs` is the size of along the last dimension of the input.

# Arguments 
- `accuracyconfig` (default [`BatchedIterConfig()`](@ref)) determine how to iterate over the accuracy subset.
- `accuracyfitness` (default [`AccuracyVsSize`](@ref)) determine how to measure fitness based on `accuracyconfig`.
- `trainconfig` (default [`TrainIterConfig()`](@ref)) determines how to iterate over the training subset. 
- `trainfitness` is a function accepting the iterator produced by `trainconfig` and the fitness strategy produced
   by `accuracyfitness` and returns the `AbstractFitness` to be used (default [`TrainThenFitness`](@ref) wrapped in
   [`GpuFitness`](@ref)).
"""
struct TrainSplitAccuracy{S, VC, VF, TC, TF} <: AbstractFitnessStrategy
    split::S
    accconfig::VC
    accfitness::VF
    trainconfig::TC
    trainfitness::TF
end
function TrainSplitAccuracy(;split=0.1, 
            accuracyconfig=BatchedIterConfig(),
            accuracyfitness=AccuracyVsSize,
            trainconfig=TrainIterConfig(),
            trainfitness=(iter, accf) -> GpuFitness(TrainThenFitness(StatefulGenerationIter(iter), Flux.Losses.logitcrossentropy, Adam(), accf, 0.0)))

    return TrainSplitAccuracy(split, accuracyconfig, accuracyfitness, trainconfig, trainfitness)
end
function fitnessfun(s::TrainSplitAccuracy, x, y)
    train_x, acc_x = split_examples(x, ceil(Int, last(size(x)) * s.split))
    train_y, acc_y = split_examples(y, ceil(Int, last(size(y)) * s.split))

    aiter = dataiter(s.accconfig, acc_x, acc_y)
    afitness = s.accfitness(aiter)

    titer = dataiter(s.trainconfig, train_x, train_y)
    return s.trainfitness(titer, afitness)
end

split_examples(a::AbstractArray{T, 1}, splitpoint) where T = a[1:end-splitpoint], a[end-splitpoint+1:end]
split_examples(a::AbstractArray{T, 2}, splitpoint) where T = a[:,1:end-splitpoint], a[:,end-splitpoint+1:end]
split_examples(a::AbstractArray{T, 4}, splitpoint) where T = a[:,:,:,1:end-splitpoint], a[:,:,:,end-splitpoint+1:end]

"""
    AccuracyVsSize(data, accdigits=2, accwrap=identity)

Produces an `AbstractFitness` which measures fitness accuracy on `data` and based on number of parameters.

The two are combined so that a candidate `a` which achieves higher accuracy rounded to the first `accdigits` digits compared to a candidate `b` will always have a better fitness.

If the first `accdigits` of accuracy is the same the candidate with fewer parameters will get higher fitness.

Accuracy part of the fitness is calculated by `accwrap(`[`AccuracyFitness(data)`](@ref)`)`.
"""
AccuracyVsSize(data, accdigits=2; accwrap=identity) = sizevs(accwrap(AccuracyFitness(data)), accdigits)
function sizevs(f::AbstractFitness, accdigits=2)
    truncacc = MapFitness(x -> round(x, digits=accdigits), f)

    size = SizeFitness()
    sizefit = MapFitness(x -> min(10.0^-(accdigits+1), 1 / x), size)

    return AggFitness(+, truncacc, sizefit)
end

"""
    struct TrainAccuracyVsSize <: AbstractFitnessStrategy
    TrainAccuracyVsSize(;trainconfig, trainfitness)

Produces an `AbstractFitness` which measures fitness accuracy on training data and based on number of parameters combined in the same way as is done for [`AccuracyVsSize`](@ref).

# Arguments 
- `trainconfig` (default [`TrainIterConfig()`](@ref)) determines how to iterate over the training subset. 
- `trainfitness` is a function accepting the iterator produced by `trainconfig` and the fitness strategy produced 
   by `accuracyfitness` and returns the `AbstractFitness` to be used (default [`TrainAccuracyFitness`](@ref) wrapped in 
   [`GpuFitness`](@ref)).

Beware that fitness as accuracy on training data will make evolution favour overfitted candidates.
"""
struct TrainAccuracyVsSize{TC, TF} <: AbstractFitnessStrategy
    trainconfig::TC
    trainfitness::TF
end
function TrainAccuracyVsSize(;
                        trainconfig=TrainIterConfig(),
                        trainfitness = dataiter -> sizevs(GpuFitness(TrainAccuracyFitness(
                                                                            dataiter=StatefulGenerationIter(dataiter), 
                                                                            defaultloss=Flux.Losses.logitcrossentropy, defaultopt = Adam())))) 
        return TrainAccuracyVsSize(trainconfig, trainfitness)
end
function fitnessfun(s::TrainAccuracyVsSize, x, y) 
    iter = dataiter(s.trainconfig, x, y)
    return s.trainfitness(iter)
end

"""
    struct BatchedIterConfig{T, V}
    BatchedIterConfig(;batchsize=32, dataaug=identity, iterwrap=identity) 

Configuration for creating batch iterators from array data.

The function `dataiter(s::BatchedIterConfig, x, y)` creates an iterator which returns a tuple of batches from `x` and `y` respectively.

More specifically, the result of `s.iterwrap(zip(s.dataaug(bx), by))` will be returned where `bx` and `by` are `BatchIterator`s.
"""
struct BatchedIterConfig{T, V}
    batchsize::Int
    dataaug::T
    iterwrap::V
end
BatchedIterConfig(;batchsize=1024, dataaug=identity, iterwrap=identity) = BatchedIterConfig(batchsize, dataaug, iterwrap)
dataiter(s::BatchedIterConfig, x, y) = dataiter(x, y, s.batchsize, s.dataaug) |> s.iterwrap

"""
    struct ShuffleIterConfig{T, V}
    ShuffleIterConfig(;batchsize=32, seed=123, dataaug=identity, iterwrap=identity) 

Configuration for creating shuffled batch iterators from array data. Data will be re-shuffled every time the iterator restarts.

The function `dataiter(s::ShuffleIterConfig, x, y)` creates an iterator which returns a tuple of batches from `x` and `y` respectively.

More specifically, the result of `s.iterwrap(Iterators.map(((x,y),) -> (s.dataaug(x), y), iter))` will be returned where `iter` is a `BatchIterator` over `x` and `y` with `shuffle=true`.

Note there there is no upper bound on how many generations are supported as the returned iterator cycles the data indefinitely. Use e.g. `Iterators.take(itr, cld(nepochs * nbatches_per_epoch, nbatches_per_gen))` to limit to `nepochs`
epochs. 
"""
struct ShuffleIterConfig{T, V}
    batchsize::Int
    seed::Int
    dataaug::T
    iterwrap::V
end
ShuffleIterConfig(;batchsize=1024, seed=123, dataaug=identity, iterwrap=identity) = ShuffleIterConfig(batchsize, seed, dataaug, iterwrap)
dataiter(s::ShuffleIterConfig, x, y) = dataiter(x, y, s.batchsize, s.seed, s.dataaug) |> s.iterwrap


"""
    struct TrainIterConfig{T}
    TrainIterConfig(nbatches_per_gen, baseconfig)
    TrainIterConfig(;nbatches_per_gen=400, baseconfig=ShuffleIterConfig())

Standard training strategy for creating a batched iterators from data. Data from `baseconfig` is cycled in partitions of `nbatches_per_gen` each generation using a [`RepeatPartitionIterator`](@ref).

"""
struct TrainIterConfig{T}
    nbatches_per_gen::Int
    baseconfig::T
end
TrainIterConfig(;nbatches_per_gen=400, baseconfig=ShuffleIterConfig()) = TrainIterConfig(nbatches_per_gen,baseconfig)
function dataiter(s::TrainIterConfig, x, y)
    baseiter = dataiter(s.baseconfig, x, y)
    return RepeatPartitionIterator(baseiter, s.nbatches_per_gen)
end

function dataiter(x::AbstractArray,y::AbstractArray{T, 1}, args...) where T 
    _dataiter(x, Flux.onehotbatch(y, sort(unique(y))), args...)
end
dataiter(x::AbstractArray,y::AbstractArray{T, 2}, args...) where T = _dataiter(x,y, args...)

function _dataiter(x::AbstractArray,y::AbstractArray, bs::Integer, seed::Integer, xwrap, ywrap = identity)
    biter = shuffleiter((x,y), bs, seed)

    # iterates feature/label pairs
    # Cycle is pretty important here. It is what makes it so that each generation sees a new shuffle
    # This is also what makes SeedIterators useful with random data augmentation: 
    #   - Even if SeedIterator makes the augmentation itself the same, it will be applied to different images 
    # Something ought to be done about the inconsistency vs BatchedIterConfig through...
    citer = Iterators.map(biter) do (x, y)
        xwrap(x), ywrap(y)
    end |> Iterators.cycle
    # Ensures all models see the exact same examples, even when a RepeatStatefulIterator restarts iteration.
    return SeedIterator(citer;rng=biter.rng)
end
shuffleiter(data, batchsize, seed) = BatchIterator(data, batchsize; shuffle=MersenneTwister(seed))

function _dataiter(x::AbstractArray, y::AbstractArray, bs::Integer, xwrap, ywrap = identity)
    biter = BatchIterator((x,y), bs)
    # iterates feature/label pairs
    return Iterators.map(biter) do (x, y)
        xwrap(x), ywrap(y)
    end
end

"""
    struct GlobalOptimiserMutation{S<:AbstractEvolutionStrategy, F} <: AbstractEvolutionStrategy
    GlobalOptimiserMutation(base::AbstractEvolutionStrategy)
    GlobalOptimiserMutation(base::AbstractEvolutionStrategy, optfun)

Maps the optimiser of each candidate in a population through `optfun` (default `randomlrscale()`).

Basically a thin wrapper for `global_optimiser_mutation`.

Useful for applying the same mutation to every candidate, e.g. global learning rate schedules which all models follow.
"""
struct GlobalOptimiserMutation{S<:AbstractEvolutionStrategy, F} <: AbstractEvolutionStrategy
    base::S
    optfun::F
end
GlobalOptimiserMutation(base::AbstractEvolutionStrategy) = GlobalOptimiserMutation(base, NaiveGAflux.randomlrscale())

function evostrategy_internal(s::GlobalOptimiserMutation, inshape)
    base = evostrategy_internal(s.base, inshape)
    return AfterEvolution(base, pop -> NaiveGAflux.global_optimiser_mutation(pop, s.optfun))
end


"""
    struct EliteAndSusSelection <: AbstractEvolutionStrategy
    EliteAndSusSelection(popsize, nelites, evolve)
    EliteAndSusSelection(;popsize=50, nelites=2, evolve = crossovermutate())

Standard evolution strategy.

Selects `nelites` candidates to move on to the next generation without any mutation.

Also selects `popsize - nelites` candidates out of the whole population using [`SusSelection`](@ref) to evolve by applying random mutation.

Mutation operations are both applied to the model itself (change sizes, add/remove vertices/edges) as well as to the optimiser (change learning rate and optimiser algorithm).

Finally, models are renamed so that the name of each vertex of the model of candidate `i` is prefixed with "model`i`".
"""
struct EliteAndSusSelection{F} <: AbstractEvolutionStrategy
    popsize::Int
    nelites::Int
    evolve::F
end
EliteAndSusSelection(;popsize=50, nelites=2, evolve=crossovermutate()) = EliteAndSusSelection(popsize, nelites, evolve)

function evostrategy_internal(s::EliteAndSusSelection, inshape)
    elite = EliteSelection(s.nelites)

    evolve = SusSelection(s.popsize - s.nelites, EvolutionChain(ShuffleCandidates(), s.evolve(inshape)))

    combine = CombinedEvolution(elite, evolve)
    return AfterEvolution(combine, rename_models ∘ clear_redundant_vertices)
end

"""
    struct EliteAndTournamentSelection <: AbstractEvolutionStrategy
    EliteAndTournamentSelection(popsize, nelites, k, p, evolve)
    EliteAndTournamentSelection(;popsize=50, nelites=2; k=2, p=1.0, evolve = crossovermutate())

Standard evolution strategy.

Selects `nelites` candidates to move on to the next generation without any mutation.

Also selects `popsize - nelites` candidates out of the whole population using [`TournamentSelection`](@ref) to evolve by applying random mutation.

Mutation operations are determined by `evolve` both applied to the model itself (change sizes, add/remove vertices/edges) as well as to the optimiser (change learning rate and optimiser algorithm).

Finally, models are renamed so that the name of each vertex of the model of candidate `i` is prefixed with "model`i`".
"""
struct EliteAndTournamentSelection{R,F} <: AbstractEvolutionStrategy
    popsize::Int
    nelites::Int
    k::Int
    p::R
    evolve::F
end
EliteAndTournamentSelection(;popsize=50, nelites=2, k=2, p=1.0, evolve = crossovermutate()) = EliteAndTournamentSelection(popsize, nelites, k, p, evolve)

function evostrategy_internal(s::EliteAndTournamentSelection, inshape)
    elite = EliteSelection(s.nelites)

    evolve = TournamentSelection(s.popsize - s.nelites, s.k, s.p, s.evolve(inshape))

    combine = CombinedEvolution(elite, evolve)
    return AfterEvolution(combine, rename_models ∘ clear_redundant_vertices)
end

"""
    crossovermutate(;pcrossover=0.3, pmutate=0.9)

Return a function which creates an [`EvolutionChain`](@ref) when called with an inputshape.

Crossover will be applied with a probability of `pcrossover` while mutation will be applied with a probability of `pmutate`. Note that these probabilities apply to models only, not optimisers.

Crossover is done using [`CrossoverSwap`](@ref) for models and [`LearningRateCrossover`](@ref) and [`OptimiserCrossover`](@ref) for optimisers.

Mutation is applied both to the model itself (change sizes, add/remove vertices/edges) as well as to the optimiser (change learning rate and optimiser algorithm).
"""
crossovermutate(;pcrossover=0.3, pmutate=0.8) = function(inshape)
    cross = candidatecrossover(pcrossover)
    crossoverevo = AfterEvolution(PairCandidates(EvolveCandidates(cross)), align_vertex_names)

    mutate = candidatemutation(pmutate, inshape)
    mutationevo = EvolveCandidates(mutate)

    return EvolutionChain(crossoverevo, mutationevo)
end

candidatemutation(p, inshape) = MapCandidate(MutationProbability(graphmutation(inshape), p), optmutation(), itermapmutation())
candidatecrossover(p) = MapCandidate(MutationProbability(graphcrossover(), p), optcrossover(), itermapcrossover())

function clear_redundant_vertices(pop)
    foreach(cand -> NaiveGAflux.model(check_apply, cand), pop)
    return pop
end

function align_vertex_names(pop)
    for i in eachindex(pop)
        m = match(r"model(\d+).\.*", modelname(pop[i]))
        if !isnothing(m)
            pop[i] = rename_model(m.captures[1], pop[i])
        end
    end
    return pop
end

function rename_models(pop)
    for i in eachindex(pop)
        pop[i] = rename_model(i, pop[i])
    end
    return pop
 end

function rename_model(i, cand)
    # No need to copy layers now, so we exclude all AbstractMutableComp and just return them without copying
    walk=Functors.ExcludeWalk(Functors.DefaultWalk(), identity, x -> x isa NaiveNASflux.AbstractMutableComp)
    return fmap(cand; walk) do x
        x isa String ? replace(x, r"^model\d+\.*" => "model$i.") : x
    end
end

itermapcrossover(p= 0.2) = MutationProbability(IteratorMapCrossover(), p) |> IteratorMapCrossover

function itermapmutation(p=0.1)
    m = TrainBatchSizeMutation(-0.2, 0.2, ntuple(i -> 2^(i+2), 8))
    return MutationProbability(m, p)
end

function optcrossover(poptswap=0.3, plrswap=0.4)
    lrc = MutationProbability(LearningRateCrossover(), plrswap) |> OptimiserCrossover
    oc = MutationProbability(OptimiserCrossover(), poptswap) |> OptimiserCrossover
    return MutationChain(lrc, oc)
end

function optmutation(p=0.1)
    lrm = LearningRateMutation()
    om = MutationProbability(OptimiserMutation([Descent, Momentum, Nesterov, Adam, NAdam, AdaGrad]), p)
    return MutationChain(lrm, om)
end

function graphcrossover()
    vertexswap = LogMutation(((v1,v2)::Tuple) -> "Crossover swap between $(name(v1)) and $(name(v2))", CrossoverSwap(0.05; mergefun=NaiveGAflux.default_mergefun(;layerfun=ActivationContributionLow)))
    crossoverop = VertexCrossover(MutationProbability(vertexswap, 0.05), 0.05)
    return LogMutation(((g1,g2)::Tuple) -> "Crossover between $(modelname(g1)) and $(modelname(g2))", crossoverop)
end

function graphmutation(inshape)
    acts = default_actfuns()

    change_nout = NoutMutation(-0.2, 0.2) # Max 20% change in output size
    decrease_nout = NoutMutation(-0.2, 0)
    add_vertex = add_vertex_mutation(acts)
    add_downsampling = AddVertexMutation(ConditionalArchSpace(candownsample(inshape), downsamplingspace(default_layerconf(); outsizes = 2 .^(4:9), activations=acts)))
    rem_vertex = RemoveVertexMutation()
    # [-2, 2] keeps kernel size odd due to CuArrays issue# 356 (odd kernel size => symmetric padding)
    change_kernel = KernelSizeMutation(ParSpace2D([-2, 2]), maxsize=maxkernelsize(inshape))
    decrease_kernel = KernelSizeMutation(ParSpace2D([-2]))
    mutate_act = ActivationFunctionMutation(acts)

    add_edge = AddEdgeMutation(0.1; mergefun=NaiveGAflux.default_mergefun(;layerfun=ActivationContributionLow))
    rem_edge = RemoveEdgeMutation()

    # Create a shorthand alias for MutationProbability
    mpn(m, p) = VertexMutation(MutationProbability(m, p))
    mph(m, p) = VertexMutation(HighUtilityMutationProbability(m, p))
    mpl(m, p) = VertexMutation(LowUtilityMutationProbability(m, p))

    cnout = mpn(LogMutation(v -> "Mutate output size of vertex $(name(v))", change_nout), 0.05)
    dnout = mpl(LogMutation(v -> "Reduce output size of vertex $(name(v))", decrease_nout), 0.05)
    maddv = mph(LogMutation(v -> "Add vertex after $(name(v))", add_vertex), 0.005)
    maddd = mpn(LogMutation(v -> "Add downsampling after $(name(v))", add_downsampling), 0.01)
    mremv = mpl(LogMutation(v -> "Remove vertex $(name(v))", rem_vertex), 0.01)
    ckern = mpn(MutationFilter(allowkernelmutation, LogMutation(v -> "Mutate kernel size of $(name(v))", change_kernel)), 0.02)
    dkern = mpn(MutationFilter(allowkernelmutation, LogMutation(v -> "Decrease kernel size of $(name(v))", decrease_kernel)), 0.02)
    mactf = mpl(LogMutation(v -> "Mutate activation function of $(name(v))", mutate_act), 0.005)
    madde = mph(LogMutation(v -> "Add edge from $(name(v))", add_edge), 0.02)
    mreme = mpn(MutationFilter(v -> length(outputs(v)) > 1, LogMutation(v -> "Remove edge from $(name(v))", rem_edge)), 0.02)

    mremv = MutationFilter(g -> nvertices(g) > 5, mremv)

    # Create two possible mutations: One which is guaranteed to not increase the size:
    dsize = MutationChain(mremv, dnout, dkern, mreme, maddd)
    # ...and another which is not guaranteed to decrease the size
    csize = MutationChain(mremv, cnout, ckern, mreme, madde, maddd, maddv)
    # Add mutation last as new vertices with neuronutility == 0 screws up outputs selection as per https://github.com/DrChainsaw/NaiveNASlib.jl/issues/39

    mgp = VertexMutation(MutationProbability(LogMutation(v -> "Mutate global pool type for $(name(v))", MutateGlobalPool()), 0.1), SelectGlobalPool())

    # If isbig then perform the mutation operation which is guaranteed to not increase the size
    # This is done mostly to avoid OOM and time outs.
    mall = MutationChain(MutationFilter(isbig, dsize), MutationFilter(!isbig, csize), mactf, mgp)

    return LogMutation(g -> "Mutate $(modelname(g))", mall)
end

isbig(g) = nparams(g) > 20e7

candownsample(inshape) = v -> candownsample(v, inshape)
candownsample(v::AbstractVertex, inshape) = is_convtype(v) && !infork(v) && canshrink(v, inshape)

function canshrink(v, inshape)
    # Note assumes stride = 2!
    # Also assumes single output after a global pool and flatten
    allvs = all_in_graph(v)
    gpv = allvs[findfirst(is_globpool, allvs)] |> inputs |> first
    return all(fshape(shapetrace(gpv) |> squashshapes, inshape) .> 1)
end

function infork(v, inputcnt = Dict{AbstractVertex, Int}(inputs(v) .=> 1), seen = Set())
    v in seen && return any(x -> x < 0, values(inputcnt))
    push!(seen, v)

    # How many times do we expect to find v as input to some subsequent vertex if we did not start inside a fork?
    inputcnt[v] = get(inputcnt, v, 0) + length(outputs(v))

    for vi in inputs(v)
        # And now subtract by 1 each time we find it
        inputcnt[vi] = get(inputcnt, vi, 0) - 1
    end

    foreach(vo -> infork(vo, inputcnt, seen), outputs(v))
    # If we have any "unaccounted" for inputs when we hit the output then we started inside a fork
    return any(x -> x < 0, values(inputcnt))
end

maxkernelsize(inshape) = v -> maxkernelsize(v, inshape)
function maxkernelsize(v::AbstractVertex, inshape)
    ks = fshape(shapetrace(v) |> squashshapes, inshape)
    # Kernel sizes must be odd due to CuArrays issue# 356 (odd kernel size => symmetric padding)
    return @. ks - !isodd(ks)
end

allowkernelmutation(v) = allowkernelmutation(NaiveNASflux.layertype(v), v)
allowkernelmutation(l, v) = false
allowkernelmutation(::NaiveNASflux.FluxConvolutional{N}, v) where N = all(isodd, size(NaiveNASflux.weights(layer(v)))[1:N])
    
function add_vertex_mutation(acts)

    function outselect(vs)
        rss = randsubseq(vs, 0.5)
        return isempty(rss) ? [rand(vs)] : rss
    end

    wrapitup(as) = AddVertexMutation(rep_fork_res(as, 1,loglevel=Logging.Info), outselect)

    add_conv = wrapitup(convspace(default_layerconf(),2 .^(4:9), 1:2:5, acts,loglevel=Logging.Info))
    add_dense = wrapitup(LoggingArchSpace(VertexSpace(default_layerconf(), NamedLayerSpace("dense", DenseSpace(2 .^(4:9), acts)));level = Logging.Info))

    return MutationChain(MutationFilter(is_convtype, add_conv), MutationFilter(!is_convtype, add_dense))
end

is_convtype(v::AbstractVertex) = any(is_globpool.(outputs(v))) || any(is_convtype.(outputs(v)))
is_globpool(v::AbstractVertex) = is_globpool(layer(v))
is_globpool(f) = f isa NaiveGAflux.GlobalPool

struct SelectGlobalPool <:AbstractVertexSelection
    s::AbstractVertexSelection
end
SelectGlobalPool() = SelectGlobalPool(AllVertices())
NaiveGAflux.select(s::SelectGlobalPool, g::CompGraph, ms...) = filter(is_globpool, NaiveGAflux.select(s.s, g, ms...))

struct MutateGlobalPool <: AbstractMutation{AbstractVertex} end
function (::MutateGlobalPool)(v::AbstractVertex)
    # Remove the old vertex and replace it with a new one with a different global pool type
    iv = inputs(v)[]
    RemoveVertexMutation()(v)
    v in outputs(iv) && return # failed to remove for some reason

    newname = first(splitext(name(v)))
    insert!(iv, vx -> GlobalPoolSpace(newpool(layer(v)))(newname, vx))
end

newpool(::GlobalPool{MaxPool}) = MeanPool
newpool(::GlobalPool{MeanPool}) = MaxPool
