module ImageClassification

using ...NaiveGAflux
using ..AutoFit: fit
import NaiveGAflux: globalpooling2d
using Random
import Logging
using Statistics

# To store program state for pause/resume
using Serialization

export ImageClassifier, fit

struct ImageClassifier
    popsize::Int
    batchsize::Int
    nepochs::Int
    seed::Int
end
ImageClassifier(;popsize=50, batchsize=64, nepochs=200, seed=abs(rand(Int))) = ImageClassifier(popsize,batchsize, nepochs, seed)

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

    tot = AggFitness(+, truncacc, sizefit)

    cache = FitnessCache(tot)
    return NanGuard(cache)
end


function iterators(train_x, train_y; nepochs=200, batchsize=64, fitnessize=2048, nbatches_per_gen=300, seed=123)
    fit_x, evo_x = split_examples(train_x, fitnessize)
    fit_y, evo_y = split_examples(train_y, fitnessize)

    fit_iter = RepeatPartitionIterator(GpuIterator(Iterators.cycle(dataiter(fit_x, fit_y, batchsize, seed), nepochs)), nbatches_per_gen)
    evo_iter = GpuIterator(dataiter(evo_x, evo_y, batchsize, seed, identity))

    return fit_iter, evo_iter
end
split_examples(a::AbstractArray{T, 1}, splitpoint) where T = a[1:end-splitpoint], a[end-splitpoint:end]
split_examples(a::AbstractArray{T, 2}, splitpoint) where T = a[:,1:end-splitpoint], a[:,end-splitpoint:end]
split_examples(a::AbstractArray{T, 4}, splitpoint) where T = a[:,:,:,1:end-splitpoint], a[:,:,:,end-splitpoint:end]

batch(data, batchsize, seed) = ShuffleIterator(data, batchsize, MersenneTwister(seed))
dataiter(x,y::AbstractArray{T, 1}, bs, s, wrap = FlipIterator ∘ ShiftIterator) where T = zip(wrap(batch(x, bs, s)), Flux.onehotbatch(batch(y, bs, s), unique(y)))
dataiter(x,y::AbstractArray{T, 2}, bs, s, wrap = FlipIterator ∘ ShiftIterator) where T = zip(wrap(batch(x, bs, s)), batch(y, bs, s))


function AutoFit.fit(c::ImageClassifier, x, y; cb=identity, mdir)
    ndims(x) == 4 || error("Must use 4D data, got $(ndims(x))D data")
    nexamples = size(x, 4)
    fitnessize = min(2048, ceil(Int,0.01 * nexamples))
    fit_iter, evo_iter = iterators(x, y, nepochs=c.nepochs, batchsize=c.batchsize, fitnessize=fitnessize, seed=c.seed)

    fitnesstrat = AccuracyVsSize(evo_iter)
    inshape = first(size(fit_iter))[1:2]
    return fit(c, fit_iter, fitnesstrat, evolutionstrategy(c.popsize, inshape); cb=cb, mdir=mdir)
end

"""
    run_experiment(popsize, fit_iter, evo_iter; nelites = 2, baseseed=666, cb = identity, mdir = defaultdir(), newpop = false)

Runs the Cifar10 experiment. This command also adds persistence and plotting:
run_experiment(50, iterators(CIFAR10.traindata())...; baseseed=abs(rand(Int)), cb=CbAll(persist, MultiPlot(display ∘ plot, PlotFitness(plot), ScatterPop(scatter), ScatterOpt(scatter))))

"""
function AutoFit.fit(c::ImageClassifier, fit_iter, fitnesstrat, evostrategy; cb = identity, mdir, newpop = false)
    Random.seed!(NaiveGAflux.rng_default, c.seed)
    @info "Start experiment with baseseed: $(c.seed)"

    insize, outsize = size(fit_iter)

    population = initial_models(c.popsize, mdir, newpop, fitnesstrat, insize, outsize[1])

    # If experiment was resumed we should start by evolving as population is persisted right before evoluation
    if all(i -> isfile(NaiveGAflux.filename(population, i)), 1:length(population))
        population = evolve!(evostrategy, population)
    end

    return evolutionloop(population, evostrategy, fit_iter, cb)
end

function evolutionloop(population, evostrategy, trainingiter, cb)
    for (gen, iter) in enumerate(trainingiter)
        @info "Begin generation $gen"

        for (i, cand) in enumerate(population)
            @info "\tTrain model $i with $(nv(NaiveGAflux.graph(cand))) vertices"
            Flux.train!(cand, iter)
        end

        # TODO: Bake into evolution? Would anyways like to log selected models...
        for (i, cand) in enumerate(population)
            @info "\tFitness model $i: $(fitness(cand))"
        end
        cb(population)

        population = evolve!(evostrategy, population)
    end
    return population
end


function evolutionstrategy(popsize, inshape, nelites=2)
    elite = EliteSelection(nelites)

    mutate = EvolveCandidates(evolvecandidate(inshape))
    evolve = SusSelection(popsize - nelites, mutate)

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
    function mutate_opt(opt::Flux.Optimise.Optimiser)
        return newopt(opt)
    end
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

nparams(c::AbstractCandidate) = nparams(NaiveGAflux.graph(c))
nparams(g::CompGraph) = mapreduce(prod ∘ size, +, params(g).order)
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

function initial_models(nr, mdir, newpop, fitnessgen, insize, outsize)
    if newpop
        rm(mdir, force=true, recursive=true)
    end

    iv(i) = inputvertex(join(["model", i, ".input"]), insize[3], FluxConv{2}())
    as = initial_archspace(insize[1:2], outsize)
    return PersistentArray(mdir, nr, i -> create_model(join(["model", i]), as, iv(i), fitnessgen))
end
create_model(name, as, in, fg) = CacheCandidate(HostCandidate(CandidateModel(CompGraph(in, as(name, in)), newopt(newlr(0.01)), Flux.logitcrossentropy, fg())))

modelname(c::AbstractCandidate) = modelname(NaiveGAflux.graph(c))
modelname(g::CompGraph) = split(name(g.inputs[]),'.')[1]

default_layerconf() = LayerVertexConf(ActivationContribution ∘ LazyMutable, NaiveGAflux.default_logging())
function initial_archspace(inshape, outsize)

    layerconf = default_layerconf()
    outconf = LayerVertexConf(layerconf.layerfun, MutationShield ∘ layerconf.traitfun)

    acts = [identity, relu, elu, selu]

    # Only use odd kernel sizes due to CuArrays issue# 356
    # Bias selection towards smaller number of large kernels in the beginning...
    conv1 = convspace(layerconf, 4:64, 3:2:9, acts)
    # Then larger number of small kernels
    conv2 = convspace(layerconf, 32:512, 1:2:5, acts)

    # Convblocks are repeated, forked or put in residual connections...
    # ...and the procedure is repeated for the output space.
    # Makes for some crazy architectures
    rfr1 = rep_fork_res(conv1,2)
    rfr2 = rep_fork_res(conv2,2)

    # Each "block" is finished with a maxpool to downsample
    maxpoolvertex = VertexSpace(layerconf, NamedLayerSpace("maxpool", MaxPoolSpace(PoolSpace2D([2]))))
    red1 = ListArchSpace(rfr1, maxpoolvertex)
    red2 = ListArchSpace(rfr2, maxpoolvertex)

    # How many times can shape be reduced by a factor 2
    maxreps = Int.(max(6, log2(minimum(inshape))))
    @show maxreps
    # Block 1 (large kernels and small sizes) repeated up to 2 times
    block1 = RepeatArchSpace(red1, 1:maxreps ÷ 2)
    # And the same for block type 2
    block2 = RepeatArchSpace(red2, 1:maxreps ÷ 2)

    # Ok, lets work on the output layers.
    # Two main options:

    # Option 1: Just a global pooling layer
    # For this to work we need to ensure that the layer before the global pool has exactly 10 outputs, that is what this is all about (or else we could just have allowed 0 dense layers in the search space for option 2).
    convout = convspace(outconf, outsize, 1:2:5, identity)
    blockcout = ListArchSpace(convout, GpVertex2D())

    # Option 2: 1-3 Dense layers after the global pool
    dense = VertexSpace(layerconf, NamedLayerSpace("dense", DenseSpace(16:512, acts)))
    drep = RepeatArchSpace(dense, 0:2)
    dout=VertexSpace(outconf, NamedLayerSpace("dense", DenseSpace(outsize, identity)))
    blockdout = ListArchSpace(GpVertex2D(), drep, dout)

    blockout = ArchSpace(ParSpace([blockdout, blockcout]))

    # Remember that each "block" here is a random and pretty big search space.
    # Basically the only constraint is to not randomly run out of GPU memory...
    return ListArchSpace(block1, block2, blockout)
end

function rep_fork_res(s, n, min_rp=1;loglevel=Logging.Debug)
    n == 0 && return s

    resconf = VertexConf(outwrap = ActivationContribution, traitdecoration = MutationShield ∘ NaiveGAflux.default_logging())
    concconf = ConcConf(ActivationContribution,  MutationShield ∘ NaiveGAflux.default_logging())

    msgfun(v) = "\tCreated $(name(v)), nin: $(nin(v)), nout: $(nout(v))"

    rep = RepeatArchSpace(s, min_rp:2)
    fork = LoggingArchSpace(loglevel, msgfun, ForkArchSpace(rep, min_rp:3, conf=concconf))
    res = LoggingArchSpace(loglevel, msgfun, ResidualArchSpace(rep, resconf))
    rep = LoggingArchSpace(loglevel, msgfun, rep)
    return rep_fork_res(ArchSpace(ParSpace([rep, fork, res])), n-1, 0, loglevel=loglevel)
end

function convspace(conf, outsizes, kernelsizes, acts; loglevel=Logging.Debug)
    # CoupledParSpace due to CuArrays issue# 356
    msgfun(v) = "\tCreated $(name(v)), nin: $(nin(v)), nout: $(nout(v))"
    conv2d = LoggingArchSpace(loglevel, msgfun, VertexSpace(conf, NamedLayerSpace("conv2d", ConvSpace2D(outsizes, acts, kernelsizes))))
    bn = LoggingArchSpace(loglevel, msgfun, VertexSpace(conf, NamedLayerSpace("batchnorm", BatchNormSpace(acts))))

    # Make sure that each alternative has the option to change output size
    # This is important to make fork and res play nice together
    convbn = ListArchSpace(conv2d, bn)
    bnconv = ListArchSpace(bn, conv2d)

    return ArchSpace(ParSpace([conv2d, convbn, bnconv]))
end

end
