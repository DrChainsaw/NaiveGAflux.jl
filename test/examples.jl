@testset "Basic example" begin
    using NaiveGAflux, Random
    Random.seed!(NaiveGAflux.rng_default, 0)

    nlabels = 3
    ninputs = 5

    # Step 1: Create initial models
    # Search space: 2-4 dense layers of width 3-10
    layerspace = VertexSpace(DenseSpace(3:10, [identity, relu, elu, selu]))
    initial_hidden = RepeatArchSpace(layerspace, 1:3)
    # Output layer has fixed size and is shielded from mutation
    outlayer = VertexSpace(Shielded(), DenseSpace(nlabels, identity))
    initial_searchspace = ArchSpaceChain(initial_hidden, outlayer)

    # Sample 5 models from the initial search space and make an initial population
    model(invertex) = CompGraph(invertex, initial_searchspace(invertex))
    models = [model(inputvertex("input", ninputs, FluxDense())) for _ in 1:5]
    @test nv.(models) == [4, 3, 4, 5, 3]

    population = Population(CandidateModel.(models))
    @test generation(population) == 1

    # Step 2: Set up fitness function:
    # Train model for one epoch using datasettrain, then measure accuracy on datasetvalidate
    # Some dummy data just to make stuff run
    onehot(y) = Flux.onehotbatch(y, 1:nlabels)
    batchsize = 4
    datasettrain    = [(randn(ninputs, batchsize), onehot(rand(1:nlabels, batchsize)))]
    datasetvalidate = [(randn(ninputs, batchsize), onehot(rand(1:nlabels, batchsize)))]
    
    fitnessfunction = TrainThenFitness(;
        dataiter = datasettrain,
        defaultloss = Flux.logitcrossentropy, # Will be used if not provided by the candidate
        defaultopt = ADAM(), # Same as above. State is wiped after training to prevent memory leaks
        fitstrat = AccuracyFitness(datasetvalidate) # This is what creates our fitness value after training
    )

    # Step 3: Define how to search for new candidates
    # We choose to evolve the existing ones through mutation

    # VertexMutation selects valid vertices from the graph to mutate
    # MutationProbability applies mutation m with a probability of p
    # Lets shorten that a bit:
    mp(m, p) = VertexMutation(MutationProbability(m, p))
    # Add a layer (40% chance) and/or remove a layer (40% chance)
    # You might want to use lower probabilities than this
    addlayer = mp(AddVertexMutation(layerspace), 0.4)
    remlayer = mp(RemoveVertexMutation(), 0.4)
    mutation = MutationChain(remlayer, addlayer)

    # Selection:
    # The two best models are not changed, the rest are mutated using mutation defined above
    elites = EliteSelection(2)
    mutate = SusSelection(3, EvolveCandidates(evolvemodel(mutation)))
    selection = CombinedEvolution(elites, mutate)

    # Step 4: Run evolution
    newpopulation = evolve(selection, fitnessfunction, population)
    @test newpopulation != population
    @test generation(newpopulation) == 2
    # Repeat step 4 until a model with the desired fitness is found.
    newnewpopulation = evolve(selection, fitnessfunction, newpopulation)
    @test newnewpopulation != newpopulation
    @test generation(newnewpopulation) == 3
    # Maybe in a loop :)
end

@testset "ParSpace example" begin
    # Set seed of default random number generator for reproducible results
    using NaiveGAflux, Random
    Random.seed!(NaiveGAflux.rng_default, 1)

    ps1d = ParSpace([2,4,6,10])

    # Draw from the search space
    @test ps1d() == 6
    @test ps1d() == 10

    # Possible to supply another rng than the default one
    @test ps1d(MersenneTwister(0)) == 4

    # Can be of any dimension and type
    ps2d = ParSpace(["1","2","3"], ["4","5","6","7"])

    @test typeof(ps1d) == ParSpace{1, Int}
    @test typeof(ps2d) == ParSpace{2, String}

    @test ps2d() == ("1", "4")
end

@testset "ConvSpace example" begin
    Random.seed!(NaiveGAflux.rng_default, 1)

    cs = ConvSpace{2}(outsizes=4:32, activations=[relu, elu, selu], kernelsizes=3:9)

    inputsize = 16
    convlayer = cs(inputsize)

    @test string(convlayer) == "Conv((8, 3), 16=>22, relu)"
end

@testset "ArchSpace example" begin
    Random.seed!(NaiveGAflux.rng_default, 0)

    # VertexSpace creates a MutableVertex of layers generated by the wrapped search space
    cs = VertexSpace(ConvSpace{2}(outsizes=8:256, activations=[identity, relu, elu], kernelsizes=3:5))
    bs = VertexSpace(BatchNormSpace([identity, relu]))

    # Block of conv->bn and bn->conv respectively.
    # Need to make sure there is always at least one SizeAbsorb layer to make fork and res below play nice
    csbs = ArchSpaceChain(cs ,bs)
    bscs = ArchSpaceChain(bs, cs)

    # Randomly generates either conv or conv->bn or bn->conv:
    cblock = ArchSpace(ParSpace1D(cs, csbs, bscs))

    # Generates between 1 and 5 layers from csbs
    rep = RepeatArchSpace(cblock, 1:5)

    # Generates between 2 and 4 parallel paths joined by concatenation (inception like-blocks) from rep
    fork = ForkArchSpace(rep, 2:4)

    # Generates a residual connection around what is generated by rep
    res = ResidualArchSpace(rep)

    # ... and a residual fork
    resfork = ResidualArchSpace(fork)

    # Pick one of the above randomly...
    repforkres = ArchSpace(ParSpace1D(rep, fork, res, resfork))

    # ...1 to 3 times
    blocks = RepeatArchSpace(repforkres, 1:3)

    # End each block with subsamping through maxpooling
    ms = VertexSpace(PoolSpace{2}(windowsizes=2, strides=2, poolfuns=MaxPool))
    reduction = ArchSpaceChain(blocks, ms)

    # And lets do 2 to 4 reductions
    featureextract = RepeatArchSpace(reduction, 2:4)

    # Adds 1 to 3 dense layers as outputs
    dense = VertexSpace(DenseSpace(16:512, [relu, selu]))
    drep = RepeatArchSpace(dense, 0:2)
    # Last layer has fixed output size (number of labels)
    dout=VertexSpace(Shielded(), DenseSpace(10, identity))
    output = ArchSpaceChain(drep, dout)

    # Aaaand lets glue it together: Feature extracting conv+bn layers -> global pooling -> dense layers
    archspace = ArchSpaceChain(featureextract, GlobalPoolSpace(), output)

    # Input is 3 channel image
    inputshape = inputvertex("input", 3, FluxConv{2}())

    # Sample one architecture from the search space
    graph1 = CompGraph(inputshape, archspace(inputshape))
    @test nv(graph1) == 79

    # And one more...
    graph2 = CompGraph(inputshape, archspace(inputshape))
    @test nv(graph2) == 128
end

@testset "Mutation examples" begin
    using NaiveGAflux, Random
    Random.seed!(NaiveGAflux.rng_default, 0)

    invertex = inputvertex("in", 3, FluxDense())
    layer1 = mutable(Dense(nout(invertex), 4), invertex)
    layer2 = mutable(Dense(nout(layer1), 5), layer1)
    graph = CompGraph(invertex, layer2)

    mutation = NoutMutation(-0.5, 0.5)

    @test nout(layer2) == 5

    mutation(layer2)

    @test nout(layer2) == 6

    # VertexMutation applies the wrapped mutation to all vertices in a CompGraph
    mutation = VertexMutation(mutation)

    @test nout.(vertices(graph)) == [3,4,6]

    mutation(graph)

    # Input vertex is never mutated
    @test nout.(vertices(graph)) == [3,5,4]

    # Use the MutationShield trait to protect vertices from mutation
    outlayer = mutable(Dense(nout(layer2), 10), layer2, traitfun = MutationShield)
    graph = CompGraph(invertex, outlayer)

    mutation(graph)

    @test nout.(vertices(graph)) == [3,4,3,10]

    # In most cases it makes sense to mutate with a certain probability
    mutation = VertexMutation(MutationProbability(NoutMutation(-0.5, 0.5), 0.5))

    mutation(graph)

    @test nout.(vertices(graph)) == [3,3,2,10]

    # Or just chose to either mutate the whole graph or don't do anything
    mutation = MutationProbability(VertexMutation(NoutMutation(-0.5, 0.5)), 0.98)

    mutation(graph)

    @test nout.(vertices(graph)) == [3,4,3,10]
    @test size(graph(ones(3,1))) == (10, 1)

    # Mutation can also be conditioned:
    mutation = VertexMutation(MutationFilter(v -> nout(v) < 4, RemoveVertexMutation()))

    mutation(graph)

    @test nout.(vertices(graph)) == [3,4,10]

    # When adding vertices it is probably a good idea to try to initialize them as identity mappings
    addmut = AddVertexMutation(VertexSpace(DenseSpace(5, identity)), IdentityWeightInit())

    # Chaining mutations is also useful:
    noutmut = NoutMutation(-0.8, 0.8)
    mutation = VertexMutation(MutationChain(addmut, noutmut))
    
    mutation(graph)

    @test nout.(vertices(graph)) == [3,3,4,10]
end

@testset "Crossover examples" begin
    using NaiveGAflux, Random
    import NaiveGAflux: regraph
    Random.seed!(NaiveGAflux.rng_default, 0)

    invertex = inputvertex("A.in", 3, FluxDense())
    layer1 = mutable("A.layer1", Dense(nout(invertex), 4), invertex; layerfun=ActivationContribution)
    layer2 = mutable("A.layer2", Dense(nout(layer1), 5), layer1; layerfun=ActivationContribution)
    layer3 = mutable("A.layer3", Dense(nout(layer2), 3), layer2; layerfun=ActivationContribution)
    layer4 = mutable("A.layer4", Dense(nout(layer3), 2), layer3; layerfun=ActivationContribution)
    modelA = CompGraph(invertex, layer4)

    # Create an exact copy to show how parameter alignment is preserved
    # Prefix names with B so we can show that something actually happened
    changeprefix(str::String; cf) = replace(str, r"^A.\.*" => "B.")
    changeprefix(x...;cf=clone) = clone(x...; cf=cf)
    modelB = copy(modelA, changeprefix)

    indata = reshape(collect(Float32, 1:3*2), 3,2)
    @test modelA(indata) == modelB(indata)

    @test name.(vertices(modelA)) == ["A.in", "A.layer1", "A.layer2", "A.layer3", "A.layer4"]
    @test name.(vertices(modelB)) == ["B.in", "B.layer1", "B.layer2", "B.layer3", "B.layer4"]

    # CrossoverSwap takes ones vertex from each graph as input and swaps a random segment from each graph
    # By default it tries to make segments as similar as possible
    swapsame = CrossoverSwap()

    swapA = vertices(modelA)[4]
    swapB = vertices(modelB)[4]
    newA, newB = swapsame((swapA, swapB))

    # It returns vertices of a new graph to be compatible with mutation utilities
    # Parent models are not modified
    @test newA ∉ vertices(modelA)
    @test newB ∉ vertices(modelB)

    # This is an internal utility which should not be needed in normal use cases.
    modelAnew = regraph(newA)
    modelBnew = regraph(newB)

    @test name.(vertices(modelAnew)) == ["A.in", "A.layer1", "B.layer2", "B.layer3", "A.layer4"] 
    @test name.(vertices(modelBnew)) == ["B.in", "B.layer1", "A.layer2", "A.layer3", "B.layer4"]

    @test modelA(indata) == modelB(indata) == modelAnew(indata) == modelBnew(indata)

    # Deviation parameter will randomly make segments unequal
    swapdeviation = CrossoverSwap(0.5)
    modelAnew2, modelBnew2 = regraph.(swapdeviation((swapA, swapB)))

    @test name.(vertices(modelAnew2)) == ["A.in", "A.layer1", "A.layer2", "B.layer1", "B.layer2", "B.layer3", "A.layer4"] 
    @test name.(vertices(modelBnew2)) == ["B.in", "A.layer3", "B.layer4"]

    # VertexCrossover applies the wrapped crossover operation to all vertices in a CompGraph
    # It in addtion, it selects compatible pairs for us (i.e swapA and swapB).
    # It also takes an optional deviation parameter which is used when pairing
    crossoverall = VertexCrossover(swapdeviation, 0.5)

    modelAnew3, modelBnew3 = crossoverall((modelA, modelB))

    # I guess things got swapped back and forth so many times not much changed in the end
    @test name.(vertices(modelAnew3)) == ["A.in", "A.layer2", "A.layer4"]
    @test name.(vertices(modelBnew3)) ==  ["B.in", "B.layer3", "B.layer1", "B.layer2", "A.layer1", "A.layer3", "B.layer4"] 

    # As advertised above, crossovers interop with most mutation utilities, just remember that input is a tuple
    # Perform the swapping operation with a 30% probability for each valid vertex pair.
    crossoversome = VertexCrossover(MutationProbability(LogMutation(((v1,v2)::Tuple) -> "Swap $(name(v1)) and $(name(v2))", swapdeviation), 0.3))

    @test_logs (:info, "Swap A.layer1 and B.layer1") (:info, "Swap A.layer2 and B.layer2") crossoversome((modelA, modelB))
end

@testset "Fitness functions" begin
    # Function to compute fitness for does not have to be a CompGraph, or even a neural network
    # They must be wrapped in an AbstractCandidate since fitness functions generally need to query the candidate for 
    # things which affect the fitness, such as the model but also things like optimizers and loss functions.
    candidate1 = CandidateModel(x -> 3:-1:1)
    candidate2 = CandidateModel(Dense(ones(Float32, 3,3), collect(Float32, 1:3)))

    # Fitness is accuracy on the provided data set
    accfitness = AccuracyFitness([(ones(Float32, 3, 1), 1:3)])
    
    @test fitness(accfitness, candidate1) == 0
    @test fitness(accfitness, candidate2) == 1

    # Measure how long time it takes to evaluate the fitness and add that in addition to the accuracy
    let timedfitness = TimeFitness(accfitness)
        c1time, c1acc = fitness(timedfitness, candidate1)
        c2time, c2acc = fitness(timedfitness, candidate2) 
        @test c1acc == 0
        @test c2acc == 1
        @test 0 < c1time 
        @test 0 < c2time 
    end

    # Use the number of parameters to compute fitness
    bigmodelfitness = SizeFitness()
    @test fitness(bigmodelfitness, candidate1) == 0
    @test fitness(bigmodelfitness, candidate2) == 12

    # One typically wants to map high number of params to lower fitness:
    smallmodelfitness = MapFitness(bigmodelfitness) do nparameters
        return min(1, 1 / nparameters)
    end
    @test fitness(smallmodelfitness, candidate1) == 1
    @test fitness(smallmodelfitness, candidate2) == 1/12

    # Combining fitness is straight forward
    combined = AggFitness(+, accfitness, smallmodelfitness, bigmodelfitness)

    @test fitness(combined, candidate1) == 1
    @test fitness(combined, candidate2) == 13 + 1/12

    # GpuFitness moves the candidates to GPU (as selected by Flux.gpu) before computing the wrapped fitness
    # Note that any data in the wrapped fitness must also be moved to the same GPU before being fed to the model
    gpuaccfitness = GpuFitness(AccuracyFitness(GpuIterator(accfitness.dataset)))
    
    @test fitness(gpuaccfitness, candidate1) == 0
    @test fitness(gpuaccfitness, candidate2) == 1
end

@testset "Candidate handling" begin

    struct ExampleCandidate <: AbstractCandidate
        a::Int
        b::Int
    end
    aval(c::ExampleCandidate; default=nothing) = c.a
    bval(c::ExampleCandidate; default=nothing) = c.b
  
    struct ExampleFitness <: AbstractFitness end
    NaiveGAflux._fitness(::ExampleFitness, c::AbstractCandidate) = aval(c; default=10) - bval(c; default=5)

    # Ok, this is alot of work for quite little in this dummy example
    @test fitness(ExampleFitness(), ExampleCandidate(4, 3)) === 1

    ctime, examplemetric = fitness(TimeFitness(ExampleFitness()), ExampleCandidate(3,1))
    @test examplemetric === 2
    @test ctime > 0
end

@testset "Evolution strategies" begin
    # For controlled randomness in the examples
    struct FakeRng end
    Base.rand(::FakeRng) = 0.7

    # Dummy candidate for brevity
    struct Cand <: AbstractCandidate
        fitness
    end
    NaiveGAflux.fitness(d::Cand) = d.fitness

    # EliteSelection selects the n best candidates
    elitesel = EliteSelection(2)
    @test evolve(elitesel, Cand.(1:10)) == Cand.([10, 9])

    # EvolveCandidates maps candidates to new candidates (e.g. through mutation)
    evocands = EvolveCandidates(c -> Cand(fitness(c) + 0.1))
    @test evolve(evocands, Cand.(1:10)) == Cand.(1.1:10.1)

    # SusSelection selects n random candidates using stochastic uniform sampling
    # Selected candidates will be forwarded to the wrapped evolution strategy before returned
    sussel = SusSelection(5, evocands, FakeRng())
    @test evolve(sussel, Cand.(1:10)) == Cand.([4.1, 6.1, 8.1, 9.1, 10.1])

    # CombinedEvolution combines the populations from several evolution strategies
    comb = CombinedEvolution(elitesel, sussel)
    @test evolve(comb, Cand.(1:10)) == Cand.(Any[10, 9, 4.1, 6.1, 8.1, 9.1, 10.1])
end

@testset "Iterators" begin
    data = reshape(collect(1:4*5), 4,5)

    # mini-batching
    biter = BatchIterator(data, 2)
    @test size(first(biter)) == (4, 2)

    # shuffle data before mini-batching
    # Warning: Must use different rng instances with the same seed for features and labels!
    siter = ShuffleIterator(data, 2, MersenneTwister(123))
    @test size(first(siter)) == size(first(biter))
    @test first(siter) != first(biter)

    # Apply a function to each batch
    miter = MapIterator(x -> 2 .* x, biter)
    @test first(miter) == 2 .* first(biter)

    # Move data to gpu
    giter = GpuIterator(miter)
    @test first(giter) == first(miter) |> gpu

    labels = collect(0:5)

    # Possible to use Flux.onehotbatch for many iterators
    biter_labels = Flux.onehotbatch(BatchIterator(labels, 2), 0:5)
    @test first(biter_labels) == Flux.onehotbatch(0:1, 0:5)

    # This is the only iterator which is "special" for this package:
    rpiter = RepeatPartitionIterator(zip(biter, biter_labels), 2)
    # It produces iterators over a subset of the wrapped iterator (2 batches in this case)
    piter = first(rpiter)
    @test length(piter) == 2
    # This allows for easily training several models on the same subset of the data
    expiter = zip(biter, biter_labels)
    for modeli in 1:3
        for ((feature, label), (expf, expl)) in zip(piter, expiter)
            @test feature == expf
            @test label == expl
        end
    end

    # StatefulGenerationIter is typically used in conjunction with TrainThenFitness to map a generation
    # number to an iterator from a RepeatStatefulIterator 
    sgiter = StatefulGenerationIter(rpiter)
    for (generationnr, topiter) in enumerate(rpiter)
        gendata = collect(NaiveGAflux.itergeneration(sgiter, generationnr))
        expdata = collect(topiter)
        @test gendata == expdata
    end

    # Timed iterator is useful for preventing that models which take very long time to train/validate slow down the process
    timediter = TimedIterator(;timelimit=0.1, patience=4, timeoutaction = () -> TimedIteratorStop, accumulate_timeouts=false, base=1:100)

    last = 0
    for i in timediter
        last = i
        if i > 2
            sleep(0.11) # Does not matter here if overloaded CI VM takes longer than this to get back to us
        end
    end
    @test last === 6 # Sleep after 2, then 4 patience
end
