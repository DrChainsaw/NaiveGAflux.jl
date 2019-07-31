module Cifar10

using ..NaiveGAflux
using Random

export run_experiment, initial_models

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
end

# Workaround as losses fail with Flix.OneHotMatrix on Appveyor x86 (works everywhere else)
onehot(y) = Float32.(Flux.onehotbatch(y, 0:9))

function trainmodels!(population, data)
    data = [data |> gpu]
    for model in population
        loss(x,y) = Flux.logitcrossentropy(model(x), y)
        Flux.train!(loss, params(model), data, model.opt)
    end
end

function evolvemodels!(population)
    @info "\tEvolve population!"
    for model in population
        m, record = mutation()
        m(model.g)
        select_pars(record)
        apply_mutation(model.g)
    end
end

function select_pars(record)
    for v in record.mutated
        @info "\t\t Select params for $(name(v))"
        ranked = sortperm(neuron_value(v))
        Δout = nout(v) - nout_org(op(v))

        if Δout < 0
            inds = sort(ranked[-Δout:end])
            Δnout(v, inds)
        end

        if Δout > 0
            inds = vcat(collect(1:nout_org(op(v))), repeat([-1], Δout))
            Δnout(v, inds)
        end
    end
end

function mutation()
    mutate_nout = RecordMutation(NoutMutation(-0.1, 0.1)) # Max 10% change in output size

    # Create a shorthand alias for MutationProbability
    mp(m,p) = MutationProbability(m, Probability(p))

    return VertexMutation(mp(mutate_nout, 0.05)), mutate_nout
end


function initial_models(nr)
    iv(i) = inputvertex(join(["model", i, ".input"]), 3, FluxConv{2}())
    as = initial_archspace()
    return map(i -> create_model(join(["model", i]), as, iv(i)), 1:nr)
end

create_model(name, as, in) = Model(CompGraph(in, as(name, in)) |> gpu, ADAM(0.01))

struct GpVertex <: AbstractArchSpace end
(s::GpVertex)(in::AbstractVertex, rng=nothing; outsize=nothing) = funvertex(globalpooling2d, in)
(s::GpVertex)(name::String, in::AbstractVertex, rng=nothing; outsize=nothing) = funvertex(join([name,".globpool"]), globalpooling2d, in)

funvertex(fun, in::AbstractVertex) = invariantvertex(fun(s), in, mutation=IoChange, traitdecoration = MutationShield ∘ NaiveGAflux.default_logging() ∘ validated())

funvertex(name::String, fun, in::AbstractVertex) =
invariantvertex(fun, in, mutation=IoChange, traitdecoration = MutationShield ∘ NaiveGAflux.default_logging() ∘ validated() ∘ named(name))


function initial_archspace()

    layerconf = LayerVertexConf(ActivationContribution ∘ LazyMutable, NaiveGAflux.default_logging() ∘ validated())
    outconf = LayerVertexConf(ActivationContribution ∘ LazyMutable, MutationShield ∘ NaiveGAflux.default_logging() ∘ validated())

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

    # TODO: Wrap elem addition in ActivationContribution so it becomes possible to select params
    # API does not currently allow this...
    resconf = VertexConf(IoChange, MutationShield ∘ NaiveGAflux.default_logging() ∘ validated())
    concconf = ConcConf(MutationShield ∘ NaiveGAflux.default_logging() ∘ validated())

    rep = RepeatArchSpace(s, min_rp:3)
    fork = ForkArchSpace(rep, min_rp:3, conf=concconf)
    res = ResidualArchSpace(rep,resconf)
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
