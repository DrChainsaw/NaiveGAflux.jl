@testset "Candidate" begin

    struct DummyFitness <: AbstractFitness end
    import NaiveGAflux: FileCandidate
    import MemPool
    @testset "CandidateModel $wrp" for wrp in (
        identity,
        HostCandidate,
        CacheCandidate,
        FileCandidate,
        CacheCandidate ∘ HostCandidate,
        CacheCandidate ∘ FileCandidate ∘ HostCandidate)


        import NaiveGAflux: AbstractFunLabel, Train, TrainLoss, Validate

        invertex = inputvertex("in", 3, FluxDense())
        hlayer = mutable("hlayer", Dense(3,4), invertex)
        outlayer = mutable("outlayer", Dense(4, 2), hlayer)
        graph = CompGraph(invertex, outlayer)

        try

            cand = wrp(CandidateModel(graph, Descent(0.01), (x,y) -> sum(x .- y), DummyFitness()))

            labs = []
            function NaiveGAflux.instrument(l::AbstractFunLabel, s::DummyFitness, f)
                push!(labs, l)
                return f
            end

            data(c::CandidateModel) = (ones(Float32, 3, 2), ones(Float32, 2,2))
            data(c::HostCandidate) = gpu.(data(c.c))
            data(c::FileCandidate) = NaiveGAflux.callcand(data, c)
            data(c::AbstractCandidate) = data(c.c)

            Flux.train!(cand, data(cand))

            @test labs == [Train(), TrainLoss()]

            NaiveGAflux.fitness(::DummyFitness, f) = 17
            @test fitness(cand) == 17

            @test labs == [Train(), TrainLoss(), Validate()]

            wasreset = false
            NaiveGAflux.reset!(::DummyFitness) = wasreset = true

            reset!(cand)

            @test wasreset


            graphmutation = VertexMutation(MutationFilter(v -> name(v)=="hlayer", AddVertexMutation(ArchSpace(DenseSpace([1], [relu])))))
            optmutation = OptimizerMutation((Momentum, Nesterov, ADAM))
            evofun = evolvemodel(graphmutation, optmutation)
            newcand = evofun(cand)

            @test NaiveGAflux.graph(newcand, nv) == 4
            @test NaiveGAflux.graph(cand, nv) == 3

            optimizer(c::AbstractCandidate) = optimizer(c.c)
            optimizer(c::CandidateModel) = typeof(c.opt)
            optimizer(c::FileCandidate) = NaiveGAflux.callcand(optimizer, c)

            @test optimizer(newcand) != optimizer(cand)

            Flux.train!(cand, data(cand))
            Flux.train!(newcand, data(newcand))

            teststrat() = NaiveGAflux.default_crossoverswap_strategy(v -> ones(nout_org(v)))
            graphcrossover = VertexCrossover(CrossoverSwap(;pairgen = (v1,v2) -> (1,1), strategy=teststrat); pairgen = (v1,v2;ind1) -> ind1==1 ? (2,3) : nothing)
            optcrossover = OptimizerCrossover()
            crossfun = evolvemodel(graphcrossover, optcrossover)

            newcand1, newcand2 = crossfun((cand, newcand))

            @test optimizer(newcand1) == optimizer(newcand)
            @test optimizer(newcand2) == optimizer(cand)

            @test NaiveGAflux.graph(newcand1, nv) == 4
            @test NaiveGAflux.graph(newcand2, nv) == 3

            Flux.train!(newcand1, data(newcand1))
            Flux.train!(newcand2, data(newcand2))

        finally
            MemPool.cleanup()
        end
    end

    @testset "FileCandidate" begin
        try
            @testset "FileCandidate cleanup" begin 
                fc = FileCandidate([1,2,3], 0.01, wait)
                # TODO: This can't be right?!
                wait(@async wait(fc.movetimer))            

                fname = MemPool.default_path(fc.c)

                @test isfile(fname)

                finalize(fc)
                @test !isfile(fname)
            end

            @testset "Functor" begin
                v1 = mutable("v1", Dense(3,3;initW=idmapping), inputvertex("in", 3, FluxDense()))
                cand1 = FileCandidate(CandidateModel(CompGraph(inputs(v1)[], v1), Descent(0.01), (x,y) -> sum(x .- y), DummyFitness()))

                mul(x::AbstractArray) = 2 .* x
                mul(x) = x
                cand2 = Flux.fmap(mul, cand1)

                indata = collect(Float32, reshape(1:6,3,2))
                @test NaiveGAflux.graph(cand2)(indata) == 2 .* indata
            end

            @testset "Serialization" begin
                using Serialization


                v = mutable("v1", Dense(3,3), inputvertex("in", 3, FluxDense()))
                secand = CacheCandidate(FileCandidate(CandidateModel(CompGraph(inputs(v)[], v), Descent(0.01), (x,y) -> sum(x .- y), DummyFitness())))

                indata = randn(3, 2)
                expected = v(indata)

                io = PipeBuffer()
                serialize(io, secand)

                MemPool.cleanup()

                # Make sure the data is gone
                @test_throws KeyError NaiveGAflux.graph(secand)

                decand = deserialize(io)

                @test NaiveGAflux.graph(decand)(indata) == expected

            end
        finally
            MemPool.cleanup()
        end
    end

    @testset "Global optimizer mutation" begin
        import NaiveGAflux.Flux.Optimise: Optimiser
        import NaiveGAflux: sameopt, learningrate, BoundedRandomWalk, global_optimizer_mutation, randomlrscale

        @testset "Random learning rate scale" begin
            using Random
            so = ShieldedOpt

            omf = randomlrscale();

            om1 = omf()
            @test learningrate(om1(Descent(0.1))) ≈ learningrate(om1(Momentum(0.1)))

            opt = Optimiser(so(Descent(0.1)), Momentum(0.1), so(Descent(1.0)), ADAM(1.0), Descent(1.0))
            @test length(om1(opt).os) == 4
            @test learningrate(om1(opt)) ≈ learningrate(om1(Descent(0.01)))

            om2 = omf()
            @test learningrate(om2(Descent(0.1))) ≈ learningrate(om2(Momentum(0.1)))

            # Differnt iterations shall yield different results ofc
            @test learningrate(om1(Descent(0.1))) != learningrate(om2(Momentum(0.1)))

            # Make sure learning rate stays within bounds when using BoundedRandomWalk
            rng = MersenneTwister(0);
            brw = BoundedRandomWalk(-1.0, 1.0, () -> randn(rng))
            @test collect(extrema(cumprod([10^brw() for i in 1:10000]))) ≈ [0.1, 10] atol = 1e-10
        end

        @testset "Global learning rate scaling" begin
            v1 = inputvertex("in", 3, FluxDense())
            pop = CandidateModel.(Ref(CompGraph(v1, v1)), Descent.(0.1:0.1:1.0), Flux.mse, Ref(DummyFitness()))

            lr(c) = c.opt.eta
            @test lr.(pop) == 0.1:0.1:1.0

            popscal = global_optimizer_mutation(pop, pp -> OptimizerMutation(o -> sameopt(o, 10learningrate(o))))

            @test lr.(popscal) == 1:10
        end
    end
end
