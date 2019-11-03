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
    evostrategy(s::T, inshape) where T <: AbstractEvolutionStrategy

Returns an `AbstractEvolution`.

Argument `inshape` is the size of the input feature maps (i.e. how many pixels images are) and may be used to determine which mutation operations are allowed for example to avoid that feature maps accidentally become 0 sized.
"""
evostrategy(s::T, inshape) where T <: AbstractEvolutionStrategy = error("Not implemented for $(T)!")

"""
    struct TrainSplitAccuracy{T} <: AbstractFitnessStrategy
    TrainSplitAccuracy(nexamples, batchsize, data2fitfun)
    TrainSplitAccuracy(;nexamples=2048, batchsize=64, data2fitgen= data -> NanGuard ∘ AccuracyVsSize(data))

Strategy to measure fitness on a subset of the training data of size `nexamples`.

Mapping from this subset to a fitness generator is done by `data2fitgen` which takes a data iterator and returns a function (or callable struct) which in turn produces an `AbstractFitness` when called with no arguments.
"""
struct TrainSplitAccuracy{T} <: AbstractFitnessStrategy
    nexamples::Int
    batchsize::Int
    data2fitgen::T
end
TrainSplitAccuracy(;nexamples=2048, batchsize=64, data2fitgen= data -> NanGuard ∘ AccuracyVsSize(data)) = TrainSplitAccuracy(nexamples, batchsize, data2fitgen)
function fitnessfun(s::TrainSplitAccuracy, x, y)
    rem_x, acc_x = split_examples(x, s.nexamples)
    rem_y, acc_y = split_examples(y, s.nexamples)
    acc_iter = GpuIterator(dataiter(acc_x, acc_y, s.batchsize, 0, identity))
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
"""
struct AccuracyVsSize{T}
    data::T
    accdigits::Int
end
AccuracyVsSize(data, accdigits=3) = AccuracyVsSize(data, accdigits)
function (f::AccuracyVsSize)()
    acc = AccuracyFitness(f.data)
    truncacc = MapFitness(x -> round(x, digits=f.accdigits), acc)

    size = SizeFitness()
    sizefit = MapFitness(x -> min(10.0^-f.accdigits, 1 / x), size)

    return AggFitness(+, truncacc, sizefit)
end

"""
    struct TrainStrategy{T} <: AbstractTrainStrategy
    TrainStrategy(nepochs, batchsize, nbatches_per_gen, seed, dataaug)
    TrainStrategy(;nepochs=200, batchsize=64, nbatches_per_gen=400, seed=123, dataaug=identity)

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
TrainStrategy(;nepochs=200, batchsize=64, nbatches_per_gen=400, seed=123, dataaug=identity) = TrainStrategy(nepochs, batchsize, nbatches_per_gen, seed, dataaug)
function trainiter(s::TrainStrategy, x, y)
    baseiter = dataiter(x, y, s.batchsize, s.seed, s.dataaug)
    epochiter = Iterators.cycle(baseiter, s.nepochs)
    return RepeatPartitionIterator(GpuIterator(epochiter), s.nbatches_per_gen)
end

batch(x, batchsize, seed) = ShuffleIterator(x, batchsize, MersenneTwister(seed))
dataiter(x,y::AbstractArray{T, 1}, bs, s, wrap) where T = zip(wrap(batch(x, bs, s)), Flux.onehotbatch(batch(y, bs, s), unique(y)))
dataiter(x,y::AbstractArray{T, 2}, bs, s, wrap) where T = zip(wrap(batch(x, bs, s)), batch(y, bs, s))

"""
    struct EliteAndSusSelection <: AbstractEvolutionStrategy
    EliteAndSusSelection(popsize, nelites)
    EliteAndSusSelection(;popsize=50, nelites=2)

Standard evolution strategy.

Selects `nelites` candidates to move on to the next generation without any mutation.

Also selects `popsize - nelites` candidates out of the whole population using [`SusSelection`](@ref) to evolve by applying random mutation.

Mutation operations are both applied to the model itself (change sizes, add/remove vertices/edges) as well as to the optimizer (change learning rate and optimizer algorithm).

Finally, models are renamed so that the name of each vertex of the model of candidate `i` is prefixed with "model`i`".
"""
struct EliteAndSusSelection <: AbstractEvolutionStrategy
    popsize::Int
    nelites::Int
end
EliteAndSusSelection(;popsize=50, nelites=2) = EliteAndSusSelection(popsize, nelites)

function evostrategy(s::EliteAndSusSelection, inshape)
    elite = EliteSelection(s.nelites)

    mutate = EvolveCandidates(evolvecandidate(inshape))
    evolve = SusSelection(s.popsize - s.nelites, mutate)

    combine = CombinedEvolution(elite, evolve)
    reset = ResetAfterEvolution(combine)
    return AfterEvolution(reset, rename_models ∘ clear_redundant_vertices)
end

function clear_redundant_vertices(pop)
    foreach(cand -> check_apply(NaiveGAflux.graph(cand)), pop)
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

function evolvecandidate(inshape)
    mutate_opt(opt::Flux.Optimise.Optimiser) = newopt(opt)
    mutate_opt(x) = deepcopy(x)
    return evolvemodel(mutation(inshape), mutate_opt)
end

newlr(o::Flux.Optimise.Optimiser) = newlr(o.os[].eta)
newlr(lr::Number) = clamp(lr + (rand() - 0.5) * lr, 1e-6, 0.3) +  (NaiveGAflux.apply(Probability(0.05)) ? 0.2*rand() : 0)

newopt(lr::Number) = Flux.Optimise.Optimiser([rand([Descent, Momentum, Nesterov, ADAM, NADAM])(lr)])
newopt(opt::Flux.Optimise.Optimiser) = NaiveGAflux.apply(Probability(0.05)) ? newopt(newlr(opt)) : sameopt(opt.os[], newlr(opt))
sameopt(::T, lr) where T = Flux.Optimise.Optimiser([T(lr)])

function mutation(inshape)
    acts = [identity, relu, elu, selu]

    increase_nout = NeuronSelectMutation(NoutMutation(0, 0.05)) # Max 5% change in output size
    decrease_nout = NeuronSelectMutation(NoutMutation(-0.05, 0))
    add_vertex = add_vertex_mutation(acts)
    add_maxpool = AddVertexMutation(VertexSpace(default_layerconf(), NamedLayerSpace("maxpool", MaxPoolSpace(PoolSpace2D([2])))))
    rem_vertex = RemoveVertexMutation()
    # [-2, 2] keeps kernel size odd due to CuArrays issue# 356 (odd kernel size => symmetric padding)
    mutate_kernel = KernelSizeMutation(ParSpace2D([-2, 2]), maxsize=maxkernelsize(inshape))
    decrease_kernel = KernelSizeMutation(ParSpace2D([-2]))
    mutate_act = ActivationFunctionMutation(acts)

    add_edge = AddEdgeMutation(0.1)
    rem_edge = RemoveEdgeMutation()

    # Create a shorthand alias for MutationProbability
    mpn(m, p) = VertexMutation(MutationProbability(m, p))
    mph(m, p) = VertexMutation(HighValueMutationProbability(m, p))
    mpl(m, p) = VertexMutation(LowValueMutationProbability(m, p))

    inout = mph(LogMutation(v -> "\tIncrease size of vertex $(name(v))", increase_nout), 0.025)
    dnout = mpl(LogMutation(v -> "\tReduce size of vertex $(name(v))", decrease_nout), 0.025)
    maddv = mph(LogMutation(v -> "\tAdd vertex after $(name(v))", add_vertex), 0.005)
    maddm = mpn(MutationFilter(canaddmaxpool(inshape), LogMutation(v -> "\tAdd maxpool after $(name(v))", add_maxpool)), 0.0005)
    mremv = mpl(LogMutation(v -> "\tRemove vertex $(name(v))", rem_vertex), 0.005)
    mkern = mpl(LogMutation(v -> "\tMutate kernel size of $(name(v))", mutate_kernel), 0.01)
    dkern = mpl(LogMutation(v -> "\tDecrease kernel size of $(name(v))", decrease_kernel), 0.005)
    mactf = mpl(LogMutation(v -> "\tMutate activation function of $(name(v))", mutate_act), 0.005)
    madde = mph(LogMutation(v -> "\tAdd edge from $(name(v))", add_edge), 0.01)
    mreme = mpn(MutationFilter(v -> length(outputs(v)) > 1, LogMutation(v -> "\tRemove edge from $(name(v))", rem_edge)), 0.01)

    mremv = MutationFilter(g -> nv(g) > 5, mremv)

    # Create two possible mutations: One which is guaranteed to not increase the size:
    dsize = MutationList(mremv, PostMutation(dnout, NeuronSelect()), dkern, maddm)
    # ...and another which can either decrease or increase the size:
    msize = MutationList(mremv, PostMutation(inout, NeuronSelect()), PostMutation(dnout, NeuronSelect()), mkern, madde, mreme, maddm, maddv)
    # Add mutation last as new vertices with neuron_value == 0 screws up outputs selection as per https://github.com/DrChainsaw/NaiveNASlib.jl/issues/39

    # If isbig then perform the mutation operation which is guaranteed to not increase the size
    # Otherwise perform the mutation which might decrease or increase the size
    # This is done mostly to avoid OOM and time outs. Doesn't hurt that it also speeds things up
    mall = MutationList(MutationFilter(isbig, dsize), MutationFilter(!isbig, msize), mactf)

    return LogMutation(g -> "Mutate model $(modelname(g))", mall)
end

isbig(g) = nparams(g) > 20e7

canaddmaxpool(inshape) = v -> canaddmaxpool(v, inshape)
canaddmaxpool(v::AbstractVertex, inshape) = is_convtype(v) && !infork(v) && nmaxpool(all_in_graph(v)) < log2(minimum(inshape))

function infork(v, forkcnt = 0)
    forkcnt < 0 && return true
    isempty(outputs(v)) && return false
    cnt = length(outputs(v)) - length(inputs(v))
    return any(infork.(outputs(v), forkcnt + cnt))
end

nmaxpool(vs) = sum(endswith.(name.(vs), "maxpool"))

maxkernelsize(inshape) = v -> maxkernelsize(v, inshape)
maxkernelsize(v::AbstractVertex, inshape) = @. inshape / 2^nmaxpool(flatten(v)) + 1

Flux.mapchildren(f, aa::AbstractArray{<:Integer, 1}) = aa

function add_vertex_mutation(acts)

    function outselect(vs)
        rss = randsubseq(vs, 0.5)
        return isempty(rss) ? [rand(vs)] : rss
    end

    wrapitup(as) = AddVertexMutation(rep_fork_res(as, 1,loglevel=Logging.Info), outselect)

    add_conv = wrapitup(convspace(default_layerconf(), 8:128, 1:2:5, acts,loglevel=Logging.Info))
    add_dense = wrapitup(LoggingArchSpace(Logging.Info, VertexSpace(default_layerconf(), NamedLayerSpace("dense", DenseSpace(16:512, acts)))))

    return MutationList(MutationFilter(is_convtype, add_conv), MutationFilter(!is_convtype, add_dense))
end

is_convtype(v::AbstractVertex) = any(is_globpool.(outputs(v))) || any(is_convtype.(outputs(v)))
is_globpool(v::AbstractVertex) = is_globpool(base(v))
is_globpool(v::InputVertex) = false
is_globpool(v::CompVertex) = is_globpool(v.computation)
is_globpool(l::ActivationContribution) = is_globpool(NaiveNASflux.wrapped(l))
is_globpool(f) = f == globalpooling2d
