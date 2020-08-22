

ActivationContributionLow(l) = ActivationContribution(l, NeuronValueEvery(20))

default_layerconf() = LayerVertexConf(ActivationContributionLow ∘ LazyMutable, NaiveGAflux.default_logging())
# Global pool has its own config because it happens to not be compatible with LazyMutable. Should be fixed someday
default_globpoolconf() = LayerVertexConf(ActivationContributionLow, MutationShield ∘ NaiveGAflux.default_logging())

default_actfuns() = [identity, relu, elu, selu]
function initial_archspace(inshape, outsize)

    layerconf = default_layerconf()
    gpconf = default_globpoolconf()
    outconf = let OutShield(t) = MutationShield(t, KernelSizeMutation, ActivationFunctionMutation)
        LayerVertexConf(layerconf.layerfun, OutShield ∘ layerconf.traitfun)
    end

    acts = default_actfuns()

    # Only use odd kernel sizes due to CuArrays issue# 356
    # Bias selection towards smaller number of large kernels in the beginning...
    conv1 = convspace(layerconf, 2 .^(3:8), 1:2:7, acts)
    # Then larger number of small kernels
    conv2 = convspace(layerconf, 2 .^(5:9), 1:2:5, acts)

    # Convblocks are repeated, forked or put in residual connections.
    rfr1 = rep_fork_res(conv1, 1)
    rfr2 = rep_fork_res(conv2, 1)

    # Each "block" is finished with downsampling
    red1 = ArchSpaceChain(rfr1, downsamplingspace(layerconf, outsizes=2 .^(3:8), activations=acts))
    red2 = ArchSpaceChain(rfr2, downsamplingspace(layerconf, outsizes=2 .^(5:9), activations=acts))

    # How many times can shape be reduced by a factor 2
    maxreps = min(6, floor(Int, log2(minimum(inshape))))
    # Block 1 (large kernels and small sizes) repeated up to 2 times
    block1 = RepeatArchSpace(red1, 1:maxreps ÷ 2)
    # And the same for block type 2
    block2 = RepeatArchSpace(red2, 1:maxreps ÷ 2)

    # Ok, lets work on the output layers.
    # Two main options:

    # Option 1: Just a global pooling layer
    # For this to work we need to ensure that the layer before the global pool has exactly 10 outputs, that is what this is all about (or else we could just have allowed 0 dense layers in the search space for option 2).
    convout = convspace(outconf, outsize, 1:2:5, identity)
    blockcout = ArchSpaceChain(convout, GlobalPoolSpace(gpconf))

    # Option 2: 1-3 Dense layers after the global pool
    dense = VertexSpace(layerconf, NamedLayerSpace("dense", DenseSpace(2 .^(8:12), acts)))
    drep = RepeatArchSpace(dense, 0:2)
    dout=VertexSpace(outconf, NamedLayerSpace("dense", DenseSpace(outsize, identity)))
    blockdout = ArchSpaceChain(GlobalPoolSpace(gpconf), drep, dout)

    blockout = ArchSpace(blockdout, blockcout)

    # Remember that each "block" here is a random and pretty big search space.
    # Basically the only constraint is to not randomly run out of GPU memory...
    return ArchSpaceChain(block1, block2, blockout)
end

function rep_fork_res(s, n, min_rp=1;loglevel=Logging.Debug)
    n == 0 && return s

    resconf = VertexConf(outwrap = ActivationContributionLow, traitdecoration = MutationShield ∘ NaiveGAflux.default_logging())
    concconf = ConcConf(ActivationContributionLow,  MutationShield ∘ NaiveGAflux.default_logging())

    msgfun(v) = "Created $(name(v)), nin: $(nin(v)), nout: $(nout(v))"

    rep = RepeatArchSpace(s, min_rp:2) # Does not need logging as layers log themselves
    fork = LoggingArchSpace(msgfun, ForkArchSpace(rep, min_rp:3, conf=concconf); level = loglevel)
    res = LoggingArchSpace(msgfun, ResidualArchSpace(rep, resconf); level = loglevel)
    # Manical cackling laughter...
    return rep_fork_res(ArchSpace(rep, fork, res), n-1, 0, loglevel=loglevel)
end

function convspace(conf, outsizes, kernelsizes, acts; loglevel=Logging.Debug)
    msgfun(v) = "Created $(name(v)), nin: $(nin(v)), nout: $(nout(v))"
    aswrap(vname, as) = LoggingArchSpace(msgfun, VertexSpace(conf, NamedLayerSpace(vname, as)); level=loglevel)
    
    conv2d = aswrap("conv2d", ConvSpace{2}(outsizes=outsizes, activations=acts, kernelsizes=kernelsizes))
    bn = aswrap("batchnorm", BatchNormSpace(acts))

    # Make sure that each alternative has the option to change output size
    # This is important to make fork and res play nice together
    convbn = ArchSpaceChain(conv2d, bn)
    bnconv = ArchSpaceChain(bn, conv2d)

    return ArchSpace(ParSpace([conv2d, convbn, bnconv]))
end

function downsamplingspace(conf; outsizes, sizes=2, strides=2, activations=identity)
    maxpoolspace = NamedLayerSpace("maxpool", PoolSpace{2}(windowsizes=sizes, strides=strides, poolfuns=MaxPool))
    meanpoolspace = NamedLayerSpace("meanpool", PoolSpace{2}(windowsizes=sizes, strides=strides, poolfuns=MeanPool))
    convspace = NamedLayerSpace("stridedconv", ConvSpace{2}(outsizes=outsizes, kernelsizes=sizes, activations=activations, strides=strides, paddings=0))
    return ArchSpace(maxpoolspace, meanpoolspace, convspace; conf=conf)
end
