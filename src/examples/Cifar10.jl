module Cifar10

using ..NaiveGAflux
using Random

export run_experiment, initial_models

# TODO: Need to handle this somehow...
NaiveNASlib.minΔninfactor(::ActivationContribution) = 1
NaiveNASlib.minΔnoutfactor(::ActivationContribution) = 1

# Stop-gap solution until a real candidate/entity type is created
struct Model
    g::CompGraph
    opt
end
(m::Model)(x) = m.g(x)
Flux.@treelike Model

function run_experiment(popsize, niters, data; nevolve=100, baseseed=666)
    Random.seed!(NaiveGAflux.rng_default, baseseed)

    population = initial_models(popsize)

    for i in 1:niters
        @info "Begin iteration $i"
        x_train,y_train = data()
        trainmodels!(population, (x_train, onehot(y_train)))

        if i % nevolve == 0
            evolvemodels!(population)
        end
    end
    return population
end

# Workaround as losses fail with Flux.OneHotMatrix on Appveyor x86 (works everywhere else)
onehot(y) = Float32.(Flux.onehotbatch(y, 0:9))

function trainmodels!(population, data)
    data = [data |> gpu]
    nptot = 0
    for model in population
        np = sum(map(ppp -> prod(size(ppp)), params(model.g).order))
        nptot += np
        @info "Train model $(name(model.g.inputs[])) nv: $(nv(model.g)) np: $np"
        loss(x,y) = Flux.logitcrossentropy(model(x), y)
        Flux.train!(loss, params(model), data, model.opt)
    end
    @info "nptot: $nptot"
    sleep(1)
end

function evolvemodels!(population)
    @info "\tEvolve population!"

    for model in population
        mutation = create_mutation()
        mutation(model.g)
    end
    @info "\tDone evolving!"
end

struct MapArchSpace <: AbstractArchSpace
    f::Function
    s::AbstractArchSpace
end
(s::MapArchSpace)(in::AbstractVertex, rng=NaiveGAflux.rng_default; outsize=missing) = s.f(s.s(in, rng, outsize=outsize))
(s::MapArchSpace)(name::String, in::AbstractVertex, rng=rng_default; outsize=missing) = s.f(s.s(name, in, rng, outsize=outsize))

function create_mutation()

    function rankfun(vsel, vvals)
        @info "\t\tSelect params for $(name(vsel))"
        return NaiveGAflux.default_neuronselect(vsel, vvals)
    end

    mutate_nout = NeuronSelectMutation(rankfun, NoutMutation(-0.1, 0.05)) # Max 10% change in output size
    add_vertex = add_vertex_mutation()
    rem_vertex = NeuronSelectMutation(rankfun, RemoveVertexMutation())

    # Create a shorthand alias for MutationProbability
    mp(m,p) = MutationProbability(m, Probability(p))

    mnout = mp(LogMutation(v -> "\tChange size of vertex $(name(v))", mutate_nout), 0.05)
    maddv = mp(LogMutation(v -> "\tAdd vertex after $(name(v))", add_vertex), 0.01)
    mremv = mp(LogMutation(v -> "\tRemove vertex $(name(v))", rem_vertex), 0.01)

    # Remove mutation last to not attempt to mutate a removed vertex as this most likely results in an error
    return LogMutation(g -> "Mutate model $(modelname(g))", PostMutation(VertexMutation(MutationList(maddv, mnout, mremv)), NeuronSelect(), RemoveZeroNout(), (m,g) -> apply_mutation(g)))
end

function add_vertex_mutation()
    acts = [identity, relu, elu, selu]

    wrapitup(as) = AddVertexMutation(MapArchSpace(gpu, rep_fork_res(as, 1)))

    # TODO: New layers to have identity mapping
    add_conv = wrapitup(convspace(default_layerconf(), 8:128, 1:2:7, acts))
    add_dense = wrapitup(VertexSpace(default_layerconf(), NamedLayerSpace("dense", DenseSpace(BaseLayerSpace(16:512, acts)))))

    return MutationList(MutationFilter(is_convtype, add_conv), MutationFilter(!is_convtype, add_dense))
end

is_convtype(v::AbstractVertex) = any(is_globpool.(outputs(v))) || any(is_convtype.(outputs(v)))
is_globpool(v::AbstractVertex) = is_globpool(base(v))
is_globpool(v::InputVertex) = false
is_globpool(v::CompVertex) = is_globpool(v.computation)
is_globpool(l::ActivationContribution) = is_globpool(NaiveNASflux.wrapped(l))
is_globpool(f) = f == globalpooling2d

function initial_models(nr)
    iv(i) = inputvertex(join(["model", i, ".input"]), 3, FluxConv{2}())
    as = initial_archspace()
    return map(i -> create_model(join(["model", i]), as, iv(i)), 1:nr)
end
modelname(g) = split(name(g.inputs[]),'.')[1]
create_model(name, as, in) = Model(CompGraph(in, as(name, in)) |> gpu, Descent(0.01))

struct GpVertex <: AbstractArchSpace end
(s::GpVertex)(in::AbstractVertex, rng=nothing; outsize=nothing) = funvertex(globalpooling2d, in)
(s::GpVertex)(name::String, in::AbstractVertex, rng=nothing; outsize=nothing) = funvertex(join([name,".globpool"]), globalpooling2d, in)

funvertex(fun, in::AbstractVertex) = invariantvertex(ActivationContribution(fun), in, mutation=IoChange, traitdecoration = MutationShield ∘ NaiveGAflux.default_logging() ∘ validated())

funvertex(name::String, fun, in::AbstractVertex) =
invariantvertex(ActivationContribution(fun), in, mutation=IoChange, traitdecoration = MutationShield ∘ NaiveGAflux.default_logging() ∘ validated() ∘ named(name))

default_layerconf() = LayerVertexConf(ActivationContribution ∘ LazyMutable, NaiveGAflux.default_logging() ∘ validated())


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

    # Block 1 (large kernels and small sizes) repeated up to 3 times
    block1 = RepeatArchSpace(red1, 1:3)
    # Lets keep number of type 2 blocks to just one
    block2 = red2

    # Ok, lets work on the output layers.
    # Two main options:

    # Option 1: Just a global pooling layer
    # For this to work we need to ensure that the layer before the global pool has exactly 10 outputs, that is what this is all about (or else we could just have allowed 0 dense layers in the search space for option 2).
    convout = convspace(outconf, 10, 1:2:5, identity)
    blockcout = ListArchSpace(convout, GpVertex())

    # Option 2: 1-3 Dense layers after the global pool
    dense = VertexSpace(layerconf, NamedLayerSpace("dense", DenseSpace(BaseLayerSpace(16:512, acts))))
    drep = RepeatArchSpace(dense, 0:2)
    dout=VertexSpace(outconf, NamedLayerSpace("dense", DenseSpace(BaseLayerSpace(10, identity))))
    blockdout = ListArchSpace(GpVertex(), drep, dout)

    blockout = ArchSpace(ParSpace([blockdout, blockcout]))

    # Remember that each "block" here is a random and pretty big search space.
    # Basically the only constraint is to not randomly run out of GPU memory...
    return ListArchSpace(block1, block2, blockout)
end

function rep_fork_res(s, n, min_rp=1)
    n == 0 && return s

    resconf = VertexConf(outwrap = ActivationContribution, traitdecoration = MutationShield ∘ NaiveGAflux.default_logging() ∘ validated())
    concconf = ConcConf(ActivationContribution,  MutationShield ∘ NaiveGAflux.default_logging() ∘ validated())

    rep = RepeatArchSpace(s, min_rp:3)
    fork = ForkArchSpace(rep, min_rp:3, conf=concconf)
    res = ResidualArchSpace(rep, resconf)
    return rep_fork_res(ArchSpace(ParSpace([rep, fork, res])), n-1, 0)
end

# About 50% faster on GPU to create a MeanPool and use it compared to dropdims(mean(x, dims=[1:2]), dims=(1,2)). CBA to figure out why...
globalpooling2d(x) = dropdims(MeanPool(size(x)[1:2])(x),dims=(1,2))

function convspace(conf, outsizes, kernelsizes, acts)
    # CoupledParSpace due to CuArrays issue# 356
    conv2d = VertexSpace(conf, NamedLayerSpace("conv2d", ConvSpace(BaseLayerSpace(outsizes, acts), CoupledParSpace(kernelsizes, 2))))
    bn = VertexSpace(conf, NamedLayerSpace("batchnorm", BatchNormSpace(acts)))

    # Make sure that each alternative has the option to change output size
    # This is important to make fork and res play nice together
    convbn = ListArchSpace(conv2d, bn)
    bnconv = ListArchSpace(bn, conv2d)

    return ArchSpace(ParSpace([conv2d, convbn, bnconv]))
end

end  # module cifar10
