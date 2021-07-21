@testset "Candidate" begin

    struct DummyFitness <: AbstractFitness end
    NaiveGAflux._fitness(::DummyFitness, f::AbstractCandidate) = 17
    import NaiveGAflux: FileCandidate, AbstractWrappingCandidate, opt, FittedCandidate
    import MemPool
    @testset "$ctype" for (ctype, candfun) in (
        (CandidateModel, CandidateModel),
        (CandidateOptModel, g -> CandidateOptModel(Descent(0.01), g))
    )
    
        @testset " $lbl" for (lbl, wrp) in (
            ("", identity),
            (FileCandidate, FileCandidate),
            (FittedCandidate, c -> FittedCandidate(3, 0.34, c)),
            ("FittedCandidate ∘ FileCandidate", c -> FittedCandidate(1, 0.78, FileCandidate(c)))
            )

            invertex = denseinputvertex("in", 3)
            hlayer = fluxvertex("hlayer", Dense(3,4), invertex)
            outlayer = fluxvertex("outlayer", Dense(4, 2), hlayer)
            graph = CompGraph(invertex, outlayer)

            try

                cand = wrp(candfun(graph))

                @test fitness(DummyFitness(), cand) == 17
                # Just to make sure we can access the graph
                @test fitness(SizeFitness(), cand) == nparams(graph)

                graphmutation = VertexMutation(MutationFilter(v -> name(v)=="hlayer", AddVertexMutation(ArchSpace(DenseSpace([1], [relu])))))
                optmutation = OptimizerMutation((Momentum, Nesterov, ADAM))
                evofun = evolvemodel(graphmutation, optmutation)
                newcand = evofun(cand)

                @test NaiveGAflux.graph(newcand, nv) == 4
                @test NaiveGAflux.graph(cand, nv) == 3

                optimizer(c) = typeof(opt(c)) 

                if ctype === CandidateModel
                    @test optimizer(newcand) === optimizer(cand) === Nothing
                else
                    @test optimizer(newcand) !== optimizer(cand) !== Nothing
                end

                teststrat() = NaiveGAflux.default_crossoverswap_strategy(v -> 1)
                graphcrossover = VertexCrossover(CrossoverSwap(;pairgen = (v1,v2) -> (1,1), strategy=teststrat); pairgen = (v1,v2;ind1) -> ind1==1 ? (2,3) : nothing)
                optcrossover = OptimizerCrossover()
                crossfun = evolvemodel(graphcrossover, optcrossover)

                newcand1, newcand2 = crossfun((cand, newcand))

                @test optimizer(newcand1) === optimizer(newcand)
                @test optimizer(newcand2) === optimizer(cand)

                @test NaiveGAflux.graph(newcand1, nv) == 4
                @test NaiveGAflux.graph(newcand2, nv) == 3
            finally
                MemPool.cleanup()
            end
        end
    end

    @testset "FileCandidate" begin
        import NaiveGAflux: graph
        try
           @testset "FileCandidate cleanup" begin
                fc = FileCandidate([1,2,3], 1.0)
                # Waiting for timer to end does not seem to be reliable, so we'll just stop the timer and call the timeout function manually
                close(fc.movetimer)
                NaiveGAflux.candtodisk(fc.c, fc.writelock)

                fname = MemPool.default_path(fc.c)

                @test isfile(fname)

                finalize(fc)
                @test !isfile(fname)
            end

            @testset "Functor" begin
                v1 = fluxvertex("v1", Dense(3,3;init=Flux.identity_init), denseinputvertex("in", 3))
                cand1 = FileCandidate(CandidateModel(CompGraph(inputs(v1)[], v1)))

                mul(x::AbstractArray) = 2 .* x
                mul(x) = x
                cand2 = Flux.fmap(mul, cand1)

                indata = collect(Float32, reshape(1:6,3,2))
                @test graph(cand2)(indata) == 2 .* indata
            end

            @testset "Serialization" begin
                using Serialization

                v = fluxvertex("v1", Dense(3,3), denseinputvertex("in", 3))
                secand = FileCandidate(CandidateModel(CompGraph(inputs(v)[], v)))

                indata = randn(3, 2)
                expected = v(indata)

                io = PipeBuffer()
                serialize(io, secand)

                MemPool.cleanup()

                # Make sure the data is gone
                @test_throws KeyError NaiveGAflux.graph(secand)

                decand = deserialize(io)

                @test graph(decand)(indata) == expected
            end

            @testset "Hold in mem" begin
                import NaiveGAflux: wrappedcand, callcand, candinmem
                struct BoolCand <: AbstractCandidate
                    x::Ref{Bool}
                end
                testref(c::BoolCand, f=identity) = f(c.x)
                testref(c::AbstractWrappingCandidate) = testref(wrappedcand(c))
                testref(c::FileCandidate, f) =  callcand(testref, c, f)             

                fc = FileCandidate(BoolCand(Ref(true)), 0.1)
                
                #1 test the testcase: Change true to false after move to disk
                x = testref(fc, identity) # And neither should this now
                @test isopen(fc.movetimer)

                t0 = time()
                while candinmem(fc) && time() - t0 < 5
                    sleep(0.01)
                end
                @test time() - t0 < 5 # Or else we timed out
                x[] = false

                @test false == x[] != testref(fc)[] # Note graph(fc) should not start movetimer
                @test !isopen(fc.movetimer)
                
                x = testref(fc, identity) # And neither should this now
                @test !isopen(fc.movetimer)
                NaiveGAflux.release!(fc)
                @test isopen(fc.movetimer)

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
            v1 = denseinputvertex("in", 3)
            pop = CandidateOptModel.(Descent.(0.1:0.1:1.0), Ref(CompGraph(v1, v1)))

            lr(c) = c.opt.eta
            @test lr.(pop) == 0.1:0.1:1.0

            popscal = global_optimizer_mutation(pop, pp -> OptimizerMutation(o -> sameopt(o, 10learningrate(o))))

            @test lr.(popscal) == 1:10
        end
    end
end
