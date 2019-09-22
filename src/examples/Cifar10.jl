module Cifar10

using ..NaiveGAflux
import NaiveGAflux:globalpooling2d
using Random
import Logging

export run_experiment, initial_models

# TODO: Need to handle this somehow...
NaiveNASlib.minΔninfactor(::ActivationContribution) = 1
NaiveNASlib.minΔnoutfactor(::ActivationContribution) = 1


function run_experiment(popsize, datatrain::RepeatPartitionIterator, datafitness; nelites = 2, baseseed=666, cb = () -> nothing)
    Random.seed!(NaiveGAflux.rng_default, baseseed)

    population = initial_models(popsize, () -> fitnessfun(datafitness))
    evostrategy = evolutionstrategy(popsize, nelites)

    for (gen, iter) in enumerate(datatrain)
        @info "Begin generation $gen"

        for (i, cand) in enumerate(population)
            @info "\tTrain model $i"
            for (x,y) in iter
                Flux.train!(cand, [(x, onehot(y)) |> gpu])
            end
        end

        # TODO: Bake into evolution? Would anyways like to log selected models...
        for (i, cand) in enumerate(population)
            @info "\tFitness model $i: $(fitness(cand))"
        end

        evolve!(evostrategy, population)
        cb()
    end

    return population
end

# Workaround as losses fail with Flux.OneHotMatrix on Appveyor x86 (works everywhere else)
onehot(y) = Float32.(Flux.onehotbatch(y, 0:9))


function evolutionstrategy(popsize, nelites=2)
    elite = EliteSelection(nelites)

    # Looks like a complete mess because mutation is stateful -> we need to create a new instance each time we mutate
    mutate = EvolveCandidates(c -> evolvemodel(mutation())(c))
    evolve = SusSelection(popsize - nelites, mutate)

    combine = CombinedEvolution(elite, evolve)
    return ResetAfterEvolution(combine)
end

function mutation()
    mutate_nout = NeuronSelectMutation(NoutMutation(-0.1, 0.05)) # Max 10% change in output size
    add_vertex = add_vertex_mutation()
    rem_vertex = RemoveVertexMutation()

    # Create a shorthand alias for MutationProbability
    mp(m,p) = VertexMutation(MutationProbability(m, Probability(p)))

    mnout = mp(LogMutation(v -> "\tChange size of vertex $(name(v))", mutate_nout), 0.05)
    maddv = mp(LogMutation(v -> "\tAdd vertex after $(name(v))", add_vertex), 0.01)
    mremv = mp(LogMutation(v -> "\tRemove vertex $(name(v))", rem_vertex), 0.04)

    # Add mutation last as new vertices with neuron_value == 0 screws up outputs selection as per https://github.com/DrChainsaw/NaiveNASlib.jl/issues/39
    return LogMutation(g -> "Mutate model $(modelname(g))", MutationList(MutationFilter(g -> nv(g) > 5, mremv), PostMutation(mnout, NeuronSelect()), MutationFilter(g -> nv(g) < 100, maddv)))
end

struct MapArchSpace <: AbstractArchSpace
    f::Function
    s::AbstractArchSpace
end
(s::MapArchSpace)(in::AbstractVertex, rng=NaiveGAflux.rng_default; outsize=missing) = s.f(s.s(in, rng, outsize=outsize))
(s::MapArchSpace)(name::String, in::AbstractVertex, rng=NaiveGAflux.rng_default; outsize=missing) = s.f(s.s(name, in, rng, outsize=outsize))

Flux.mapchildren(f, aa::AbstractArray{<:Integer, 1}) = aa

function add_vertex_mutation()
    acts = [identity, relu, elu, selu]

    wrapitup(as) = AddVertexMutation(MapArchSpace(gpu, rep_fork_res(as, 1,loglevel=Logging.Info)))

    # TODO: New layers to have identity mapping
    add_conv = wrapitup(convspace(default_layerconf(), 8:128, 1:2:7, acts,loglevel=Logging.Info))
    add_dense = wrapitup(LoggingArchSpace(Logging.Info, VertexSpace(default_layerconf(), NamedLayerSpace("dense", DenseSpace(BaseLayerSpace(16:512, acts))))))

    return MutationList(MutationFilter(is_convtype, add_conv), MutationFilter(!is_convtype, add_dense))
end

is_convtype(v::AbstractVertex) = any(is_globpool.(outputs(v))) || any(is_convtype.(outputs(v)))
is_globpool(v::AbstractVertex) = is_globpool(base(v))
is_globpool(v::InputVertex) = false
is_globpool(v::CompVertex) = is_globpool(v.computation)
is_globpool(l::ActivationContribution) = is_globpool(NaiveNASflux.wrapped(l))
is_globpool(f) = f == globalpooling2d

function initial_models(nr, fitnessgen)
    iv(i) = inputvertex(join(["model", i, ".input"]), 3, FluxConv{2}())
    as = initial_archspace()
    return map(i -> create_model(join(["model", i]), as, iv(i), fitnessgen), 1:nr)
end
create_model(name, as, in, fg) = CandidateModel(CompGraph(in, as(name, in)) |> gpu, Descent(0.01), Flux.logitcrossentropy, fg())
modelname(g) = split(name(g.inputs[]),'.')[1]

function fitnessfun(dataset, accdigits=3)
    acc = AccuracyFitness(dataset)
    truncacc = MapFitness(x -> round(x, digits=accdigits), acc)

    # TODO: Enable turn off for testing stability
    time = TimeFitness(NaiveGAflux.Validate())
    timefit = MapFitness(x -> min(10.0^-accdigits, 1/(x * 10.0^(5+accdigits))), time)

    tot = AggFitness(sum, truncacc, timefit)

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
    # TODO: Maxpools can currently be removed, but are never added. Care needs to be taken to not insert too many so shape is subsampled below 1. MutationFilter and then count number of maxpools in graph?
    maxpoolvertex = VertexSpace(layerconf, NamedLayerSpace("maxpool", MaxPoolSpace(PoolSpace2D([2]))))
    red1 = ListArchSpace(rfr1, maxpoolvertex)
    red2 = ListArchSpace(rfr2, maxpoolvertex)

    # Block 1 (large kernels and small sizes) repeated up to 3 times
    block1 = RepeatArchSpace(red1, 1:3)
    # Lets keep number of type 2 blocks to just one
    block2 = red2

    # Ok, lets work on the output layers.
    # Two main options:

    # Option 1: Just a global pooling layer
    # For this to work we need to ensure that the layer before the global pool has exactly 10 outputs, that is what this is all about (or else we could just have allowed 0 dense layers in the search space for option 2).
    convout = convspace(outconf, 10, 1:2:5, identity)
    blockcout = ListArchSpace(convout, GpVertex2D())

    # Option 2: 1-3 Dense layers after the global pool
    dense = VertexSpace(layerconf, NamedLayerSpace("dense", DenseSpace(BaseLayerSpace(16:512, acts))))
    drep = RepeatArchSpace(dense, 0:2)
    dout=VertexSpace(outconf, NamedLayerSpace("dense", DenseSpace(BaseLayerSpace(10, identity))))
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

    msgfun(v) = "Created $(name(v)), nin: $(nin(v)), nout: $(nout(v))"

    rep = RepeatArchSpace(s, min_rp:3)
    fork = LoggingArchSpace(loglevel, msgfun, ForkArchSpace(rep, min_rp:3, conf=concconf))
    res = LoggingArchSpace(loglevel, msgfun, ResidualArchSpace(rep, resconf))
    rep = LoggingArchSpace(loglevel, msgfun, rep)
    return rep_fork_res(ArchSpace(ParSpace([rep, fork, res])), n-1, 0, loglevel=loglevel)
end

function convspace(conf, outsizes, kernelsizes, acts; loglevel=Logging.Debug)
    # CoupledParSpace due to CuArrays issue# 356
    msgfun(v) = "Created $(name(v)), nin: $(nin(v)), nout: $(nout(v))"
    conv2d = LoggingArchSpace(loglevel, msgfun, VertexSpace(conf, NamedLayerSpace("conv2d", ConvSpace(BaseLayerSpace(outsizes, acts), CoupledParSpace(kernelsizes, 2)))))
    bn = LoggingArchSpace(loglevel, msgfun, VertexSpace(conf, NamedLayerSpace("batchnorm", BatchNormSpace(acts))))

    # Make sure that each alternative has the option to change output size
    # This is important to make fork and res play nice together
    convbn = ListArchSpace(conv2d, bn)
    bnconv = ListArchSpace(bn, conv2d)

    return ArchSpace(ParSpace([conv2d, convbn, bnconv]))
end


 # TODO Debugging utils. To be removed
inout_info(g) = [name.(vertices(g)) nin.(vertices(g)) nout.(vertices(g))]

function sizecheck(g)
    for v in vertices(g)
        nins1 = nin(v)
        nins2 = nout.(inputs(v))
        if nins1 != nins2
            @warn "Size fail for $(name(v))! $nins1 vs $nins2"
        end
    end
end

changed(g) = filter(v -> nin(v) != nin_org(v) || nout(v) != nout_org(v), vertices(g))

end  # module cifar10
