"""
    AbstractFitnessStrategy

Base type for fitness strategies.

See [`fitnessfun`](@ref)
"""
abstract type AbstractFitnessStrategy end

"""
    fitnessfun(s::T, x, y) where T <: AbstractFitnessStrategy

Returns the tuple `rem_x`, `rem_y`, `fitnessgen` where `rem_x` and `rem_y` are subsets of `x` and `y` and `fitnessgen()` produces an `AbstractFitness`.

Rationale for `rem_x` and `rem_y` is to allow fitness to be calculated on a subset of the training data which the models are not trained on themselves.

Rationale for `fitnessgen` not being an `AbstractFitness` is that some `AbstractFitness` implementations are stateful so that each candidate needs its own instance.
"""
fitnessfun(s::T, x, y) where T <: AbstractFitnessStrategy = error("Not implemented for $(T)!")

"""
    AbstractTrainStrategy

Base type for strategies on how to feed training data.

See [`trainiter`](@ref).
"""
abstract type AbstractTrainStrategy end

"""
    trainiter(s::T, x, y) where T <: AbstractTrainStrategy

Returns an iterator `fit_iter` which in turn iterates over iterators `ss_iter` where each `ss_iter` iterates over a subset of the data in x and y.

See [`RepeatPartitionIterator`](@ref) for an example of an iterator which fits the bill.
"""
trainiter(s::T, x, y) where T <: AbstractTrainStrategy = error("Not implemented for $(T)!")

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
function evostrategy(s::AbstractEvolutionStrategy, inshape)
    evostrat = evostrategy_internal(s, inshape)
    return ResetAfterEvolution(evostrat)
end

"""
    struct TrainSplitAccuracy{T} <: AbstractFitnessStrategy
    TrainSplitAccuracy(nexamples, batchsize, data2fitfun, dataaug)
    TrainSplitAccuracy(;nexamples=2048, batchsize=64, data2fitgen= data -> NanGuard ∘ AccuracyVsSize(data), dataaug=identity)

Strategy to measure fitness on a subset of the training data of size `nexamples`.

Mapping from this subset to a fitness generator is done by `data2fitgen` which takes a data iterator and returns a function (or callable struct) which in turn produces an `AbstractFitness` when called with no arguments.
"""
struct TrainSplitAccuracy{T, V} <: AbstractFitnessStrategy
    nexamples::Int
    batchsize::Int
    data2fitgen::T
    dataaug::V
end
TrainSplitAccuracy(;nexamples=2048, batchsize=64, data2fitgen= data -> NanGuard ∘ AccuracyVsSize(data), dataaug=identity) = TrainSplitAccuracy(nexamples, batchsize, data2fitgen, dataaug)
function fitnessfun(s::TrainSplitAccuracy, x, y)
    rem_x, acc_x = split_examples(x, s.nexamples)
    rem_y, acc_y = split_examples(y, s.nexamples)
    acc_iter = GpuIterator(dataiter(acc_x, acc_y, s.batchsize, 0, s.dataaug))
    return rem_x, rem_y, s.data2fitgen(acc_iter)
end

split_examples(a::AbstractArray{T, 1}, splitpoint) where T = a[1:end-splitpoint], a[end-splitpoint:end]
split_examples(a::AbstractArray{T, 2}, splitpoint) where T = a[:,1:end-splitpoint], a[:,end-splitpoint:end]
split_examples(a::AbstractArray{T, 4}, splitpoint) where T = a[:,:,:,1:end-splitpoint], a[:,:,:,end-splitpoint:end]

"""
    struct AccuracyVsSize{T}
    AccuracyVsSize(data, accdigits=3)

Produces an `AbstractFitness` which measures fitness accuracy on `data` and based on number of parameters.

The two are combined so that a candidate `a` which achieves higher accuracy rounded to the first `accdigits` digits compared to a candidate `b` will always have a better fitness.

Only if the first `accdigits` of accuracy is the same will the number of parameters determine who has higher fitness.

Accuracy part of the fitness is calculated by `accwrap(AccuracyFitness(data))`.
"""
struct AccuracyVsSize{T,V}
    data::T
    accwrap::V
    accdigits::Int
end
AccuracyVsSize(data, accdigits=2; accwrap=identity) = AccuracyVsSize(data, accwrap, accdigits)
(f::AccuracyVsSize)() = sizevs(f.accwrap(AccuracyFitness(f.data)), f.accdigits)

function sizevs(f::AbstractFitness, accdigits)
    truncacc = MapFitness(x -> round(x, digits=accdigits), f)

    size = SizeFitness()
    sizefit = MapFitness(x -> min(10.0^-(accdigits+1), 1 / x), size)

    return AggFitness(+, truncacc, sizefit)
end

"""
    struct TrainAccuracyVsSize <: AbstractFitnessStrategy
    TrainAccuracyVsSize(;accdigits=3, accwrap=identity)

Produces an `AbstractFitness` which measures fitness accuracy on training data and based on number of parameters.

The two are combined so that a candidate `a` which achieves higher accuracy rounded to the first `accdigits` digits compared to a candidate `b` will always have a better fitness.

Only if the first `accdigits` of accuracy is the same will the number of parameters determine who has higher fitness.

Accuracy part of the fitness is calculated by `accwrap(TrainAccuracyFitness())`.

Beware that fitness as accuracy on training data will make evolution favour overfitted candidates.
"""
struct TrainAccuracyVsSize{T} <: AbstractFitnessStrategy
    accwrap::T
    accdigits::Int
end
TrainAccuracyVsSize(;accdigits=3, accwrap=identity) = TrainAccuracyVsSize(accwrap, accdigits)
fitnessfun(s::TrainAccuracyVsSize, x, y) = x, y, () -> NanGuard(sizevs(s.accwrap(TrainAccuracyFitness()), s.accdigits))

"""
    struct PruneLongRunning{T <: AbstractFitnessStrategy, D <: Real} <: AbstractFitnessStrategy
    PruneLongRunning(s::AbstractFitnessStrategy, t1, t2)

Produces an `AbstractFitness` generator which multiplies the fitness produced by `s` with a factor `f < 1` if training time takes longer than `t1`. If training time takes longer than t2, fitness will be zero.

As the name suggests, the purpose is to get rid of models which take too long to train.
"""
struct PruneLongRunning{T <: AbstractFitnessStrategy, D <: Real} <: AbstractFitnessStrategy
    s::T
    t1::D
    t2::D
end
function fitnessfun(s::PruneLongRunning, x, y)
    x, y, fitgen = fitnessfun(s.s, x, y)
    return x, y, () -> prunelongrunning(fitgen(), s.t1, s.t2)
end

function prunelongrunning(s::AbstractFitness, t1, t2)
    mapping(t) = 1 - (t - t1) / (t2 - t1)
    scaled = MapFitness(t -> clamp(mapping(t), 0, 1), TimeFitness(NaiveGAflux.Train(), 1))
    return AggFitness(*, scaled, s)
end

"""
    struct TrainStrategy{T} <: AbstractTrainStrategy
    TrainStrategy(nepochs, batchsize, nbatches_per_gen, seed, dataaug)
    TrainStrategy(;nepochs=200, batchsize=32, nbatches_per_gen=400, seed=123, dataaug=identity)

Standard training strategy. Data is cycled `nepochs` times in partitions of `nbatches_per_gen` and batchsize of `batchsize` each generation using a [`RepeatPartitionIterator`](@ref).

Data can be augmented using `dataaug`.
"""
struct TrainStrategy{T} <: AbstractTrainStrategy
    nepochs::Int
    batchsize::Int
    nbatches_per_gen::Int
    seed::Int
    dataaug::T
end
TrainStrategy(;nepochs=200, batchsize=32, nbatches_per_gen=400, seed=123, dataaug=identity) = TrainStrategy(nepochs, batchsize, nbatches_per_gen, seed, dataaug)
function trainiter(s::TrainStrategy, x, y)
    baseiter = dataiter(x, y, s.batchsize, s.seed, s.dataaug)
    partiter = RepeatPartitionIterator(GpuIterator(baseiter), s.nbatches_per_gen)
    return Iterators.cycle(partiter, s.nepochs)
end

batch(x, batchsize, seed) = ShuffleIterator(NaiveGAflux.Singleton(x), batchsize, MersenneTwister(seed))
dataiter(x,y::AbstractArray{T, 1}, bs, s, wrap) where T = zip(wrap(batch(x, bs, s)), Flux.onehotbatch(batch(y, bs, s), sort(unique(y))))
dataiter(x,y::AbstractArray{T, 2}, bs, s, wrap) where T = zip(wrap(batch(x, bs, s)), batch(y, bs, s))

"""
    struct GlobalOptimizerMutation{S<:AbstractEvolutionStrategy, F} <: AbstractEvolutionStrategy
    GlobalOptimizerMutation(base::AbstractEvolutionStrategy)
    GlobalOptimizerMutation(base::AbstractEvolutionStrategy, optfun)

Maps the optimizer of each candidate in a population through `optfun` (default `randomlrscale()`).

Basically a thin wrapper for [`NaiveGAflux.global_optimizer_mutation`](@ref).

Useful for applying the same mutation to every candidate, e.g. global learning rate schedules which all models follow.
"""
struct GlobalOptimizerMutation{S<:AbstractEvolutionStrategy, F} <: AbstractEvolutionStrategy
    base::S
    optfun::F
end
GlobalOptimizerMutation(base::AbstractEvolutionStrategy) = GlobalOptimizerMutation(base, NaiveGAflux.randomlrscale())

function evostrategy_internal(s::GlobalOptimizerMutation, inshape)
    base = evostrategy_internal(s.base, inshape)
    return AfterEvolution(base, pop -> NaiveGAflux.global_optimizer_mutation(pop, s.optfun))
end


"""
    struct EliteAndSusSelection <: AbstractEvolutionStrategy
    EliteAndSusSelection(popsize, nelites, evolve)
    EliteAndSusSelection(;popsize=50, nelites=2, evolve = crossovermutate())

Standard evolution strategy.

Selects `nelites` candidates to move on to the next generation without any mutation.

Also selects `popsize - nelites` candidates out of the whole population using [`SusSelection`](@ref) to evolve by applying random mutation.

Mutation operations are both applied to the model itself (change sizes, add/remove vertices/edges) as well as to the optimizer (change learning rate and optimizer algorithm).

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

Mutation operations are determined by `evolve` both applied to the model itself (change sizes, add/remove vertices/edges) as well as to the optimizer (change learning rate and optimizer algorithm).

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

Crossover will be applied with a probability of `pcrossover` while mutation will be applied with a probability of `pmutate`. Note that these probabilities apply to models only, not optimizers.

Crossover is done using [`CrossoverSwap`](@ref) for models and [`LearningRateCrossover`](@ref) and [`OptimizerCrossover`](@ref) for optimizers.

Mutation is applied both to the model itself (change sizes, add/remove vertices/edges) as well as to the optimizer (change learning rate and optimizer algorithm).
"""
crossovermutate(;pcrossover=0.3, pmutate=0.9) = function(inshape)
    cross = candidatecrossover(pcrossover)
    crossoverevo = AfterEvolution(PairCandidates(EvolveCandidates(cross)), align_vertex_names)

    mutate = candidatemutation(pmutate, inshape)
    mutationevo = EvolveCandidates(mutate)

    return EvolutionChain(crossoverevo, mutationevo)
end

candidatemutation(p, inshape) = evolvemodel(MutationProbability(graphmutation(inshape), p), optmutation())
candidatecrossover(p) = evolvemodel(MutationProbability(graphcrossover(), p), optcrossover())

function clear_redundant_vertices(pop)
    foreach(cand -> NaiveGAflux.graph(cand, check_apply), pop)
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
    rename_model(str::String; cf) = replace(str, r"^model\d+\.*" => "model$i.")
    rename_model(x...;cf) = clone(x...; cf=cf)
    rename_model(m::AbstractMutableComp; cf) = m # No need to copy below this level
    return NaiveGAflux.mapcandidate(g -> copy(g, rename_model))(cand)
end

function optcrossover(poptswap=0.3, plrswap=0.4)
    lrc = MutationProbability(LearningRateCrossover(), plrswap) |> OptimizerCrossover
    oc = MutationProbability(OptimizerCrossover(), poptswap) |> OptimizerCrossover
    return MutationChain(lrc, oc)
end

function optmutation(p=0.05)
    lrm = LearningRateMutation()
    om = MutationProbability(OptimizerMutation([Descent, Momentum, Nesterov, ADAM, NADAM, ADAGrad]), p)
    return MutationChain(lrm, om)
end

function graphcrossover()
    vertexswap = LogMutation(((v1,v2)::Tuple) -> "Crossover swap between $(name(v1)) and $(name(v2))", CrossoverSwap(0.05))
    crossoverop = VertexCrossover(MutationProbability(vertexswap, 0.05), 0.05)
    return LogMutation(((g1,g2)::Tuple) -> "Crossover between $(modelname(g1)) and $(modelname(g2))", crossoverop)
end

function graphmutation(inshape)
    acts = [identity, relu, elu, selu]

    increase_nout = NeuronSelectMutation(NoutMutation(0, 0.1)) # Max 10% change in output size
    decrease_nout = NeuronSelectMutation(NoutMutation(-0.1, 0))
    add_vertex = add_vertex_mutation(acts)
    add_maxpool = AddVertexMutation(VertexSpace(default_layerconf(), NamedLayerSpace("maxpool", MaxPoolSpace(PoolSpace2D([2])))))
    rem_vertex = RemoveVertexMutation()
    # [-2, 2] keeps kernel size odd due to CuArrays issue# 356 (odd kernel size => symmetric padding)
    increase_kernel = KernelSizeMutation(ParSpace2D([ 2]), maxsize=maxkernelsize(inshape))
    decrease_kernel = KernelSizeMutation(ParSpace2D([-2]))
    mutate_act = ActivationFunctionMutation(acts)

    add_edge = AddEdgeMutation(0.1)
    rem_edge = RemoveEdgeMutation()

    # Create a shorthand alias for MutationProbability
    mpn(m, p) = VertexMutation(MutationProbability(m, p))
    mph(m, p) = VertexMutation(HighValueMutationProbability(m, p))
    mpl(m, p) = VertexMutation(LowValueMutationProbability(m, p))

    inout = mph(LogMutation(v -> "Increase size of vertex $(name(v))", increase_nout), 0.05)
    dnout = mpl(LogMutation(v -> "Reduce size of vertex $(name(v))", decrease_nout), 0.05)
    maddv = mph(LogMutation(v -> "Add vertex after $(name(v))", add_vertex), 0.005)
    maddm = mpn(MutationFilter(canaddmaxpool(inshape), LogMutation(v -> "Add maxpool after $(name(v))", add_maxpool)), 0.01)
    mremv = mpl(LogMutation(v -> "Remove vertex $(name(v))", rem_vertex), 0.01)
    ikern = mpl(LogMutation(v -> "Mutate kernel size of $(name(v))", increase_kernel), 0.005)
    dkern = mpl(LogMutation(v -> "Decrease kernel size of $(name(v))", decrease_kernel), 0.005)
    mactf = mpl(LogMutation(v -> "Mutate activation function of $(name(v))", mutate_act), 0.005)
    madde = mph(LogMutation(v -> "Add edge from $(name(v))", add_edge), 0.02)
    mreme = mpn(MutationFilter(v -> length(outputs(v)) > 1, LogMutation(v -> "Remove edge from $(name(v))", rem_edge)), 0.02)

    mremv = MutationFilter(g -> nv(g) > 5, mremv)

    # Create two possible mutations: One which is guaranteed to not increase the size:
    dsize = MutationChain(mremv, PostMutation(dnout, neuronselect), dkern, mreme, maddm)
    # ...and another which can (and typically does) increase the size:
    isize = MutationChain(PostMutation(inout, neuronselect), ikern, madde, maddm, maddv)
    # Add mutation last as new vertices with neuron_value == 0 screws up outputs selection as per https://github.com/DrChainsaw/NaiveNASlib.jl/issues/39

    mgp = VertexMutation(MutationProbability(LogMutation(v -> "Mutate global pool type for $(name(v))", MutateGlobalPool()), 0.1), SelectGlobalPool())

    # If isbig then perform the mutation operation which is guaranteed to not increase the size
    # This is done mostly to avoid OOM and time outs.
    # If not big then randomly decide whether to increase the model complexity or decrease it
    decr = nothing
    function decrease_size(g)
        if !isnothing(decr)
            rv = decr
            decr = nothing
            return rv
        end
        decr = isbig(g) || rand() > 0.5 # Perhaps this can be determined somehow...
        return decr
    end
    mall = MutationChain(MutationFilter(decrease_size, dsize), MutationFilter(!decrease_size, isize), mactf, mgp)

    return LogMutation(g -> "Mutate $(modelname(g))", mall)
end

isbig(g) = nparams(g) > 20e7

canaddmaxpool(inshape) = v -> canaddmaxpool(v, inshape)
canaddmaxpool(v::AbstractVertex, inshape) = is_convtype(v) && !infork(v) && nmaxpool(all_in_graph(v)) < log2(minimum(inshape))

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

nmaxpool(vs) = sum(endswith.(name.(vs), "maxpool"))

maxkernelsize(inshape) = v -> maxkernelsize(v, inshape)
function maxkernelsize(v::AbstractVertex, inshape)
    ks = inshape .÷ 2^nmaxpool(NaiveNASlib.flatten(v))
    # Kernel sizes must be odd due to CuArrays issue# 356 (odd kernel size => symmetric padding)
    return @. ks - !isodd(ks)
 end

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
is_globpool(v::AbstractVertex) = is_globpool(base(v))
is_globpool(v::InputVertex) = false
is_globpool(v::CompVertex) = is_globpool(v.computation)
is_globpool(l::AbstractMutableComp) = is_globpool(NaiveNASflux.wrapped(l))
is_globpool(f) = f isa NaiveGAflux.GlobalPool

struct SelectGlobalPool <:AbstractVertexSelection
    s::AbstractVertexSelection
end
SelectGlobalPool() = SelectGlobalPool(AllVertices())
NaiveGAflux.select(s::SelectGlobalPool, g::CompGraph) = filter(is_globpool, NaiveGAflux.select(s.s, g))

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
