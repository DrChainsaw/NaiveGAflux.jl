module Cifar10

using ..NaiveGAflux
import NaiveGAflux:globalpooling2d
using Random
import Logging
using Statistics
using Serialization

export run_experiment, iterators

export PlotFitness, ScatterPop, ScatterOpt, MultiPlot, CbAll

defaultdir(this="CIFAR10") = joinpath(NaiveGAflux.modeldir, this)

function iterators((train_x,train_y)::Tuple; nepochs=200, batchsize=32, fitnessize=2048, nbatches_per_gen=400, seed=123)
    batch(data) = ShuffleIterator(data, batchsize, MersenneTwister(seed))
    dataiter(x,y, wrap = FlipIterator ∘ ShiftIterator) = zip(wrap(batch(x)), Flux.onehotbatch(batch(y), 0:9))

    fit_x, fit_y = train_x[:,:,:,1:end-fitnessize], train_y[1:end-fitnessize]
    evo_x, evo_y = train_x[:,:,:,end-fitnessize:end], train_y[end-fitnessize:end]

    fit_iter = RepeatPartitionIterator(GpuIterator(Iterators.cycle(dataiter(fit_x, fit_y), nepochs)), nbatches_per_gen)
    evo_iter = GpuIterator(dataiter(evo_x, evo_y, identity))

    return fit_iter, evo_iter
end
"""
    run_experiment(popsize, fit_iter, evo_iter; nelites = 2, baseseed=666, cb = identity, mdir = defaultdir(), newpop = false)

Runs the Cifar10 experiment. This command also adds persistence and plotting:
run_experiment(50, iterators(CIFAR10.traindata())...; baseseed=abs(rand(Int)), cb=CbAll(persist, MultiPlot(display ∘ plot, PlotFitness(plot), ScatterPop(scatter), ScatterOpt(scatter))))

"""
function run_experiment(popsize, fit_iter, evo_iter; nelites = 2, baseseed=666, cb = identity, mdir = defaultdir(), newpop = false)
    Random.seed!(NaiveGAflux.rng_default, baseseed)
    @info "Start experiment with baseseed: $baseseed."

    population = initial_models(popsize, mdir, newpop, () -> fitnessfun(evo_iter))
    evostrategy = evolutionstrategy(popsize, nelites)

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
        return population
    end
    return population
end


function evolutionstrategy(popsize, nelites=2)
    elite = EliteSelection(nelites)

    mutate = EvolveCandidates(evolvecandidate())
    evolve = SusSelection(popsize - nelites, mutate)

    combine = CombinedEvolution(elite, evolve)
    reset = ResetAfterEvolution(combine)
    return AfterEvolution(reset, rename_models)
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

function evolvecandidate()
    function mutate_opt(opt::Flux.Optimise.Optimiser)
        return newopt(opt)
    end
    mutate_opt(x) = deepcopy(x)
    return evolvemodel(mutation(), mutate_opt)
end

newlr(o::Flux.Optimise.Optimiser) = newlr(o.os[].eta)
newlr(lr::Number) = clamp(lr + (rand() - 0.5) * lr, 1e-6, 0.3) +  (NaiveGAflux.apply(Probability(0.05)) ? 0.2*rand() : 0)

newopt(lr::Number) = Flux.Optimise.Optimiser([rand([Descent, Momentum, Nesterov, ADAM, NADAM])(lr)])
newopt(opt::Flux.Optimise.Optimiser) = NaiveGAflux.apply(Probability(0.05)) ? newopt(newlr(opt)) : sameopt(opt.os[], newlr(opt))
sameopt(::T, lr) where T = Flux.Optimise.Optimiser([T(lr)])

function mutation()
    acts = [identity, relu, elu, selu]

    increase_nout = NeuronSelectMutation(NoutMutation(0, 0.05)) # Max 5% change in output size
    decrease_nout = NeuronSelectMutation(NoutMutation(-0.05, 0))
    add_vertex = add_vertex_mutation(acts)
    add_maxpool = AddVertexMutation(VertexSpace(default_layerconf(), NamedLayerSpace("maxpool", MaxPoolSpace(PoolSpace2D([2])))))
    rem_vertex = RemoveVertexMutation()
    # [-2, 2] keeps kernel size odd due to CuArrays issue# 356 (odd kernel size => symmetric padding)
    mutate_kernel = KernelSizeMutation(ParSpace2D([-2, 2]), maxsize=maxkernelsize)
    decrease_kernel = KernelSizeMutation(ParSpace2D([-2]))
    mutate_act = ActivationFunctionMutation(acts)

    add_edge = AddEdgeMutation(0.1)

    # Create a shorthand alias for MutationProbability
    mpn(m, p) = VertexMutation(MutationProbability(m, p))
    mph(m, p) = VertexMutation(HighValueMutationProbability(m, p))
    mpl(m, p) = VertexMutation(LowValueMutationProbability(m, p))

    inout = mph(LogMutation(v -> "\tIncrease size of vertex $(name(v))", increase_nout), 0.025)
    dnout = mpl(LogMutation(v -> "\tReduce size of vertex $(name(v))", decrease_nout), 0.025)
    maddv = mph(LogMutation(v -> "\tAdd vertex after $(name(v))", add_vertex), 0.005)
    maddm = mpn(MutationFilter(canaddmaxpool, LogMutation(v -> "\tAdd maxpool after $(name(v))", add_maxpool)), 0.0005)
    mremv = mpl(LogMutation(v -> "\tRemove vertex $(name(v))", rem_vertex), 0.005)
    mkern = mpl(LogMutation(v -> "\tMutate kernel size of $(name(v))", mutate_kernel), 0.01)
    dkern = mpl(LogMutation(v -> "\tDecrease kernel size of $(name(v))", decrease_kernel), 0.005)
    mactf = mpl(LogMutation(v -> "\tMutate activation function of $(name(v))", mutate_act), 0.005)
    madde = mph(LogMutation(v -> "\tAdd edge from $(name(v))", add_edge), 0.5)

    mremv = MutationFilter(g -> nv(g) > 5, mremv)

    # Create two possible mutations: One which is guaranteed to not increase the size:
    dsize = MutationList(mremv, PostMutation(dnout, NeuronSelect()), dkern, maddm)
    # ...and another which can either decrease or increase the size:
    msize = MutationList(mremv, PostMutation(inout, NeuronSelect()), PostMutation(dnout, NeuronSelect()), mkern, madde, maddm, maddv)
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

canaddmaxpool(v::AbstractVertex) = is_convtype(v) && !occursin.(r"(path|res|maxpool)", name(v)) && nmaxpool(all_in_graph(v)) < 4

nmaxpool(vs) = sum(endswith.(name.(vs), "maxpool"))

maxkernelsize(v::AbstractVertex, insize=(32,32)) = @. insize / 2^nmaxpool(flatten(v)) + 1

Flux.mapchildren(f, aa::AbstractArray{<:Integer, 1}) = aa

function add_vertex_mutation(acts)

    function outselect(vs)
        rss = randsubseq(vs, 0.5)
        isempty(rss) && return [rand(vs)]
        return rss
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

function initial_models(nr, mdir, newpop, fitnessgen)
    if newpop
        rm(mdir, force=true, recursive=true)
    end

    iv(i) = inputvertex(join(["model", i, ".input"]), 3, FluxConv{2}())
    as = initial_archspace()
    return PersistentArray(mdir, nr, i -> create_model(join(["model", i]), as, iv(i), fitnessgen))
end
create_model(name, as, in, fg) = CacheCandidate(HostCandidate(CandidateModel(CompGraph(in, as(name, in)), newopt(newlr(0.01)), Flux.logitcrossentropy, fg())))

modelname(c::AbstractCandidate) = modelname(NaiveGAflux.graph(c))
modelname(g::CompGraph) = split(name(g.inputs[]),'.')[1]

function fitnessfun(dataset, accdigits=3)
    acc = AccuracyFitness(dataset)
    truncacc = MapFitness(x -> round(x, digits=accdigits), acc)

    size = SizeFitness()
    sizefit = MapFitness(x -> min(10.0^-accdigits, 1 / x), size)

    tot = AggFitness(+, truncacc, sizefit)

    cache = FitnessCache(tot)
    return NanGuard(cache)
end


default_layerconf() = LayerVertexConf(ActivationContribution ∘ LazyMutable, NaiveGAflux.default_logging())
function initial_archspace()

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

    # Block 1 (large kernels and small sizes) repeated up to 2 times
    block1 = RepeatArchSpace(red1, 1:2)
    # And the same for block type 2
    block2 = RepeatArchSpace(red2, 1:2)

    # Ok, lets work on the output layers.
    # Two main options:

    # Option 1: Just a global pooling layer
    # For this to work we need to ensure that the layer before the global pool has exactly 10 outputs, that is what this is all about (or else we could just have allowed 0 dense layers in the search space for option 2).
    convout = convspace(outconf, 10, 1:2:5, identity)
    blockcout = ListArchSpace(convout, GpVertex2D())

    # Option 2: 1-3 Dense layers after the global pool
    dense = VertexSpace(layerconf, NamedLayerSpace("dense", DenseSpace(16:512, acts)))
    drep = RepeatArchSpace(dense, 0:2)
    dout=VertexSpace(outconf, NamedLayerSpace("dense", DenseSpace(10, identity)))
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


## Plotting stuff. Maybe move to own file if reusable...

loadifpresent(filename, default=Float32[]) = isfile(filename) ? deserialize(filename) : default

"""
    PlotFitness(plotfun, basedir=joinpath(defaultdir(), "PlotFitness"))

Plots best and average fitness for each generation.

Also serializes data so that plotting can be resumed if evolution is aborted.

# Examples
```julia-repl
julia> using NaiveGAflux, NaiveGAflux.Cifar10, MLDatasets, Plots

julia> gr();

julia> run_experiment(50, iterators(CIFAR10.traindata())...; cb=PlotFitness(plot));
```
"""
struct PlotFitness
    best::Vector{Float32}
    avg::Vector{Float32}
    plt
    basedir
end

function PlotFitness(plotfun, basedir=joinpath(defaultdir(), "PlotFitness"))
    best = loadifpresent(joinpath(basedir, "best.jls"))
    avg = loadifpresent(joinpath(basedir, "avg.jls"))
    plt = plotfun(hcat(best,avg), label=["Best", "Avg"], xlabel="Generation", ylabel="Fitness", m=[:circle, :circle], legend=:bottomright)
    return PlotFitness(best, avg, plt, basedir)
end

function plotfitness(p::PlotFitness, population)
    fits = fitness.(population)
    best = maximum(fits)
    avg = mean(fits)
    push!(p.best, best)
    push!(p.avg, avg)
    push!(p.plt, length(p.best), [best, avg])
end

function (p::PlotFitness)(population)
    plotfitness(p, population)
    mkpath(p.basedir)
    serialize(joinpath(p.basedir, "best.jls"), p.best)
    serialize(joinpath(p.basedir, "avg.jls"), p.avg)
    return p.plt
end

"""
    ScatterPop(plotfun, basedir=joinpath(defaultdir(), "ScatterPop"))

Scatter plot of number of vertices in model vs fitness vs number of parameters for each candidate.

Also serializes data so that plotting can be resumed if evolution is aborted.

# Examples
```julia-repl
julia> using NaiveGAflux, NaiveGAflux.Cifar10, MLDatasets, Plots

julia> gr();

julia> run_experiment(50, iterators(CIFAR10.traindata())...; cb=ScatterPop(scatter));
```
"""
struct ScatterPop
    plotfun
    data::Vector{Array{Float32, 2}}
    basedir
end

function ScatterPop(plotfun, basedir=joinpath(defaultdir(), "ScatterPop"))
    data = loadifpresent(joinpath(basedir, "nvfitnp.jls"), [zeros(Float32,0,0)])
    return ScatterPop(plotfun, data, basedir)
end

function plotfitness(p::ScatterPop, population)
    fits = fitness.(population)
    nverts = nv.(NaiveGAflux.graph.(population))
    npars = nparams.(population)
    push!(p.data, hcat(nverts, fits, npars))
    return p.plotfun(nverts, fits, zcolor=npars/1e6, m=(:heat, 0.8), xlabel="Number of vertices", ylabel="Fitness", colorbar_title="Number of parameters (1e6)", label="")
end

function(p::ScatterPop)(population)
    plt = plotfitness(p, population)
    mkpath(p.basedir)
    serialize(joinpath(p.basedir, "nvfitnp.jls"), p.data)
    return plt
end

"""
    ScatterOpt(plotfun, basedir=joinpath(defaultdir(), "ScatterOpt"))

Scatter plot of learning rate vs fitness vs optimizer type for each candidate.

Also serializes data so that plotting can be resumed if evolution is aborted.

# Examples
```julia-repl
julia> using NaiveGAflux, NaiveGAflux.Cifar10, MLDatasets, Plots

julia> gr();

julia> run_experiment(50, iterators(CIFAR10.traindata())...; cb=ScatterOpt(scatter));
```
"""
struct ScatterOpt
    plotfun
    data::Vector{Array{Any, 2}}
    basedir
end

function ScatterOpt(plotfun, basedir=joinpath(defaultdir(), "ScatterOpt"))
    data = loadifpresent(joinpath(basedir, "fitlropt.jls"), [zeros(0,0)])
    return ScatterOpt(plotfun, data, basedir)
end

function plotfitness(p::ScatterOpt, population)

    opt(c::AbstractCandidate) = opt(c.c)
    opt(c::CandidateModel) = c.opt

    fits = fitness.(population)
    opts = opt.(population)
    lrs = map(o -> o.os[].eta, opts)
    ots = map(o -> typeof(o.os[]), opts)

    push!(p.data, hcat(fits, lrs, ots))

    uots = unique(ots)
    inds = map(o -> o .== ots, uots)

    fitso = map(indv -> fits[indv], inds)
    lrso = map(indv -> lrs[indv], inds)

    return p.plotfun(lrso, fitso, xlabel="Learning rate", ylabel="Fitness", label=string.(uots), legend=:outerright, legendfontsize=5)
end

function(p::ScatterOpt)(population)
    plt = plotfitness(p, population)
    mkpath(p.basedir)
    serialize(joinpath(p.basedir, "fitlropt.jls"), p.data)
    return plt
end

"""
    MultiPlot(plotfun, plts...)

Multiple plots in the same figure.

# Examples
```julia-repl
julia> using NaiveGAflux, NaiveGAflux.Cifar10, MLDatasets, Plots

julia> gr();

julia> run_experiment(50, iterators(CIFAR10.traindata())...; cb=MultiPlot(display ∘ plot, PlotFitness(plot), ScatterPop(scatter), ScatterOpt(scatter)));
```
"""
struct MultiPlot
    plotfun
    plts
end
MultiPlot(plotfun, plts...) = MultiPlot(plotfun, plts)

(p::MultiPlot)(population) = p.plotfun(map(pp -> pp(population), p.plts)...)

struct CbAll
    cbs
end
CbAll(cbs...) = CbAll(cbs)

(cba::CbAll)(population) = foreach(cb -> cb(population), cba.cbs)


end  # module cifar10
