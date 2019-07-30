

# TODO: Create module

## Stuff to fix in NaiveNASflux

function Flux.mapchildren(f, a::AbstractVector{<:AbstractVertex})
    f.(a)
    return a
end
Flux.children(v::AbstractVertex) = (base(v),)
function Flux.mapchildren(f, v::AbstractVertex)
    f.(Flux.children(v))
    return v
end
Flux.children(g::CompGraph) = Tuple(vertices(g))
function Flux.mapchildren(f, g::CompGraph)
    f.(Flux.children(g))
    return g
end
#Flux.isleaf(::AbstractVertex) = false

function Flux.mapchildren(f, m::MutableLayer)
    m.layer = f(m.layer)
    return m
end

## End stuff to fix in NaiveNASflux

struct Model
    g::CompGraph
    opt
end
(m::Model)(x) = m.g(x)
Flux.@treelike Model

function run_experiment(popsize, niters, data)
    population = initial_models(popsize)
    for i in 1:niters
        @info "Begin iteration $i"
        x_train,y_train = data()
        trainmodels!(population, (x_train, Flux.onehotbatch(y_train, 0:9)))
    end
end

function trainmodels!(population, data)
    data = [data |> gpu]
    for model in population
        loss(x,y) = Flux.logitcrossentropy(model(x), y)
        Flux.train!(loss, params(model), data, model.opt)
    end
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

struct SoftMaxVertex <: AbstractArchSpace end
(s::SoftMaxVertex)(in::AbstractVertex, rng=nothing; outsize=nothing) = funvertex(softmax, in)
(s::SoftMaxVertex)(name::String, in::AbstractVertex, rng=nothing; outsize=nothing) = funvertex(join([name,".softmax"]), softmax, in)

funvertex(fun, in::AbstractVertex) = invariantvertex(fun(s), in, mutation=IoChange, traitdecoration = MutationShield ∘ validated() ∘ NaiveGAflux.default_logging())

funvertex(name::String, fun, in::AbstractVertex) =
invariantvertex(fun, in, mutation=IoChange, traitdecoration = MutationShield ∘ validated() ∘ NaiveGAflux.default_logging() ∘ named(name))


function initial_archspace()

    # TODO: identity and shield is a Workaround for NaiveNASflux issue #9
    vc_nopar = LayerVertexConf(identity, MutationShield ∘ validated() ∘ NaiveGAflux.default_logging())

    layerconf = LayerVertexConf(
    #ActivationContribution ∘ Fails on gpu :(
    LazyMutable, validated() ∘ NaiveGAflux.default_logging())
    outconf = LayerVertexConf(
    #ActivationContribution ∘
    LazyMutable, MutationShield ∘ validated() ∘ NaiveGAflux.default_logging())

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

    rep = RepeatArchSpace(s, min_rp:3)
    fork = ForkArchSpace(rep, min_rp:3)
    res = ResidualArchSpace(rep)
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
