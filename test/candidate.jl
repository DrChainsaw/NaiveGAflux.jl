@testset "Candidate" begin

    struct DummyFitness <: AbstractFitness end
    import NaiveGAflux: FileCandidate
    import MemPool
    @testset "CandidateModel $wrp" for wrp in (identity, HostCandidate, CacheCandidate, FileCandidate)
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

            data(wrp) = (ones(Float32, 3, 2), ones(Float32, 2,2))
            data(c::HostCandidate) = gpu.(data(c.c))

            Flux.train!(cand, data(cand))

            @test labs == [Train(), TrainLoss()]

            NaiveGAflux.fitness(::DummyFitness, f) = 17
            @test fitness(cand) == 17

            @test labs == [Train(), TrainLoss(), Validate()]

            wasreset = false
            NaiveGAflux.reset!(::DummyFitness) = wasreset = true

            reset!(cand)

            @test wasreset

            graphmutation = VertexMutation(MutationFilter(v -> name(v)=="hlayer", RemoveVertexMutation()))
            optmutation = OptimizerMutation((Momentum, Nesterov, ADAM))
            evofun = evolvemodel(graphmutation, optmutation)
            newcand = evofun(cand)

            @test nv(NaiveGAflux.graph(newcand)) == 2
            @test nv(NaiveGAflux.graph(cand)) == 3

            optimizer(c::AbstractCandidate) = optimizer(c.c)
            optimizer(c::CandidateModel) = typeof(c.opt)
            optimizer(c::FileCandidate) = NaiveGAflux.callcand(optimizer, c)

            @test optimizer(newcand) != optimizer(cand)

            Flux.train!(cand, data(cand))
            Flux.train!(newcand, data(newcand))

        finally
            MemPool.cleanup()
        end
    end

    @testset "FileCandidate cleanup" begin
        try
            fc = FileCandidate([1,2,3], t -> wait(t))

            fname = MemPool.default_path(fc.c[])

            @test isfile(fname)
            finalize(fc.c)

            @test !isfile(fname)
        finally
            MemPool.cleanup()
        end
    end

    @testset "eagermutation" begin
        invertex = inputvertex("in", 3, FluxDense())
        hlayer = mutable("hlayer", Dense(3,4), invertex)
        outlayer = mutable("outlayer", Dense(4, 2), hlayer)
        graph = CompGraph(invertex, outlayer)

        Δnout(hlayer, 2)
        Δoutputs(hlayer, v -> ones(nout_org(v)))
        apply_mutation(graph)

        @test nout(layer(hlayer)) == 4
        @test nin(layer(outlayer)) == 4

        NaiveGAflux.eagermutation(graph)

        @test nout(layer(hlayer)) == 6
        @test nin(layer(outlayer)) == 6
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
