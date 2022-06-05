@testset "MapType" begin
    import NaiveGAflux: MapType

    @testset "Basic" begin
        mt = MapType{Integer}(x -> 2x, identity)

        @test mt(2) == 4
        @test mt(UInt16(3)) == 6
        @test mt('c') == 'c'
        @test mt(2.0) == 2.0
    end

    @testset "Crossover" begin
        import NaiveGAflux: FluxOptimizer

        struct MapTypeTestCrossover{T} <: AbstractCrossover{T} end
        (::MapTypeTestCrossover)((c1, c2)) = c2,c1


        c1 = CandidateOptModel(Descent(), CompGraph(inputvertex("c1", 1), AbstractVertex[]))
        c2 = CandidateOptModel(Momentum(), CompGraph(inputvertex("c2", 1), AbstractVertex[]))
        
        mt1, mt2 = MapType(MapTypeTestCrossover{CompGraph}(), (c1,c2), (identity, identity))
        @test name.(inputs(mt1(model(c1)))) == ["c2"]
        @test name.(inputs(mt2(model(c2)))) == ["c1"]
        @test mt1(3) == 3
        @test mt2('c') == 'c'

        mt1, mt2 = MapType(MapTypeTestCrossover{FluxOptimizer}(), (c1,c2), (identity, identity))
        @test typeof(mt1(opt(c1))) == Momentum
        @test typeof(mt2(opt(c2))) == Descent 
        @test mt1(3) == 3
        @test mt2('c') == 'c'
    end
end

@testset "MapCandidate" begin
    import NaiveGAflux: MapCandidate
    
    struct CollectMutation{T} <: AbstractMutation{T}
        seen::Vector{T}
    end
    CollectMutation{T}() where T = CollectMutation{T}(T[])
    # This way CollectMutation is type stable. Mutation often isn't though, but now we can test that 
    # MapCandidate does not add any extra type instability
    (m::CollectMutation{T1})(x::T2) where {T1, T2<:T1} = push!(m.seen, x)[end]::T2

    function testcand(name, opt=Descent())
        v0 = inputvertex(name * "_v0", 2)
        v1 = fluxvertex(name * "_v1", Dense(nout(v0) => 3), v0)
        CandidateOptModel(opt, CompGraph(v0, v1))
    end
    @testset "Mutation" begin

        @testset "CompGraph" begin
            graphmutation = CollectMutation{CompGraph}()
            cnew = @inferred MapCandidate(graphmutation, deepcopy)(testcand("c"))
            @test length(graphmutation.seen) == 1 
            @test graphmutation.seen[] === model(cnew)       
        end

        @testset "Optimiser" begin
            optmutation = CollectMutation{FluxOptimizer}()
            cnew = MapCandidate(optmutation, deepcopy)(testcand("c"))
            @test length(optmutation.seen) == 1
            @test optmutation.seen[] === opt(cnew)
        end

        @testset "CompGraph + Optimiser" begin  
            graphmutation = CollectMutation{CompGraph}()
            optmutation = CollectMutation{FluxOptimizer}()
            cnew = @inferred MapCandidate((graphmutation, optmutation), deepcopy)(testcand("c"))
            
            @test length(graphmutation.seen) == 1 
            @test graphmutation.seen[] === model(cnew)   
            
            @test length(optmutation.seen) == 1
            @test optmutation.seen[] === opt(cnew)
        end
    end

    @testset "Crossover" begin
        
        @testset "CompGraph" begin
            graphcrossover = CollectMutation{Tuple{CompGraph, CompGraph}}()
            c1,c2 = testcand("c1"), testcand("c2")
            cnew1, cnew2 = @inferred MapCandidate(graphcrossover, deepcopy)((c1, c2))

            @test length(graphcrossover.seen) == 1 
            @test graphcrossover.seen[] === (model(cnew1), model(cnew2))  
        end

        @testset "Optimiser" begin
            optcrossover = CollectMutation{Tuple{FluxOptimizer, FluxOptimizer}}()
            c1,c2 = testcand("c1", Descent()), testcand("c2", Momentum())
            cnew1, cnew2 = @inferred MapCandidate(optcrossover , deepcopy)((c1, c2))

            @test length(optcrossover.seen) == 1 
            @test optcrossover.seen[] === (opt(cnew1), opt(cnew2))  
        end

        @testset "CompGraph + Optimiser" begin
            graphcrossover = CollectMutation{Tuple{CompGraph, CompGraph}}()
            optcrossover = CollectMutation{Tuple{FluxOptimizer, FluxOptimizer}}()
            c1,c2 = testcand("c1", Descent()), testcand("c2", Momentum())
            cnew1, cnew2 = @inferred MapCandidate((graphcrossover, optcrossover), deepcopy)((c1, c2))

            @test length(graphcrossover.seen) == 1 
            @test graphcrossover.seen[] === (model(cnew1), model(cnew2))  
            @test length(optcrossover.seen) == 1 
            @test optcrossover.seen[] === (opt(cnew1), opt(cnew2))  
        end
    end
end

@testset "Candidate" begin

    struct DummyFitness <: AbstractFitness end
    NaiveGAflux._fitness(::DummyFitness, f::AbstractCandidate) = 17
    using NaiveGAflux: FileCandidate, AbstractWrappingCandidate, FittedCandidate, trainiterator, validationiterator, _evolvemodel
    using Functors: fmap
    import MemPool

    CandidateBatchIterMap(g) = CandidateDataIterMap(BatchSizeIteratorMap(16,32, batchsizeselection((3,)), g), CandidateModel(g))

    @testset "$ctype" for (ctype, candfun) in (
        (CandidateModel, CandidateModel),
        (CandidateOptModel, g -> CandidateOptModel(Descent(0.01), g)),
        (CandidateBatchIterMap, CandidateBatchIterMap)
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
                evofun = _evolvemodel(graphmutation, optmutation)
                newcand = evofun(cand)

                @test NaiveGAflux.model(nvertices, newcand) == 4
                @test NaiveGAflux.model(nvertices, cand) == 3

                opttype(c) = typeof(opt(c)) 

                if ctype === CandidateOptModel
                    @test opttype(newcand) !== opttype(cand) !== Nothing
                    fmapped = fmap(identity, newcand)
                    @test opt(fmapped) !== opt(newcand)
                    @test opttype(fmapped) === opttype(newcand)
                else
                    @test opttype(newcand) === opttype(cand) === Nothing
                end

                if ctype == CandidateBatchIterMap
                    @test length(first(trainiterator(cand; default=(1:100,)))) == 16  
                    @test length(first(validationiterator(cand; default=(1:100,)))) == 32  
                    # TODO Add mutation
                    @test length(first(trainiterator(newcand; default=(1:100,)))) == 16  
                    @test length(first(validationiterator(newcand; default=(1:100,)))) == 32  
                else
                    @test length(first(trainiterator(cand; default=(1:100,)))) == 100  
                    @test length(first(validationiterator(cand; default=(1:100,)))) == 100
                end

                teststrat() = NaiveGAflux.default_crossoverswap_strategy(v -> 1)
                graphcrossover = VertexCrossover(CrossoverSwap(;pairgen = (v1,v2) -> (1,1), strategy=teststrat); pairgen = (v1,v2;ind1) -> ind1==1 ? (2,3) : nothing)
                optcrossover = OptimizerCrossover()
                crossfun = _evolvemodel(graphcrossover, optcrossover)

                newcand1, newcand2 = crossfun((cand, newcand))

                @test opttype(newcand1) === opttype(newcand)
                @test opttype(newcand2) === opttype(cand)

                @test NaiveGAflux.model(nvertices, newcand1) == 4
                @test NaiveGAflux.model(nvertices, newcand2) == 3
            finally
                MemPool.cleanup()
            end
        end
    end

    @testset "FileCandidate" begin
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
                @test cand2 isa FileCandidate

                @test cand1.hold == cand2.hold == false

                indata = collect(Float32, reshape(1:6,3,2))
                @test model(cand2)(indata) == 2 .* indata
                @test cand2.hold == true

                # We accessed wrappedcand when calling graph, so now we hold the model in memory until release!:d 
                # Make sure this is the case for fmapped candidates too
                cand3 = Flux.fmap(identity, cand2)
                @test cand3.hold == cand2.hold == true
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
                @test_throws KeyError NaiveGAflux.model(secand)

                decand = deserialize(io)

                @test model(decand)(indata) == expected
            end

            @testset "Hold in mem" begin
                import NaiveGAflux: wrappedcand, callcand, candinmem
                struct BoolCand <: AbstractCandidate
                    x::Base.RefValue{Bool}
                end
                testref(c::BoolCand, f=identity) = f(c.x)
                testref(c::AbstractWrappingCandidate) = testref(wrappedcand(c))
                testref(c::FileCandidate, f) =  callcand(testref, c, f)             

                fc = FileCandidate(BoolCand(Ref(true)), 0.1)
                
                #1 test the testcase: Change true to false after move to disk
                x = testref(fc, identity) # And neither should this now
                @test isopen(fc.movetimer) == true

                t0 = time()
                while candinmem(fc) && time() - t0 < 5
                    sleep(0.01)
                end
                @test time() - t0 < 5 # Or else we timed out
                x[] = false

                @test false == x[] != testref(fc)[] # Note model(fc) should not start movetimer
                @test !isopen(fc.movetimer)
                
                x = testref(fc, identity) # And neither should this now
                @test !isopen(fc.movetimer)
                NaiveGAflux.release!(fc)
                @test isopen(fc.movetimer)
            end

            @testset "Move to disk collision" begin
                using NaiveGAflux: AbstractCandidate, wrappedcand, release!
                using Serialization

                struct TakesLongToSerialize <: AbstractCandidate end

                finish_serialize = Condition()
                function Serialization.serialize(s::AbstractSerializer, ::TakesLongToSerialize)      
                    wait(finish_serialize)
                    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
                    serialize(s, TakesLongToSerialize)
                end

                fc = FileCandidate(TakesLongToSerialize(), 10.0)
                # Waiting for timer to end does not seem to be reliable, so we'll just stop the timer and call the timeout function manually
                close(fc.movetimer)
                # Async as the serialization will block
                movetask = @async NaiveGAflux.candtodisk(fc.c, fc.writelock)

                # Also blocked due to locking
                gettask = @async @test_logs (:warn, r"Try to access FileCandidate which is being moved to disk") wrappedcand(fc)

                # Hammer the condition to improve robustness towards deadlocks
                cnt = 0
                while !istaskdone(gettask)
                    sleep(0.05)
                    notify(finish_serialize)
                    cnt += 1
                    if cnt == 20
                        @warn "Task did not complete after $cnt attempts. Testcase might be deadlocked!"
                    end
                end

                @test fetch(gettask) == TakesLongToSerialize()
                # Calling wrappedcand causes FileCandidate to hold the candidate in memory until released 
                @test isopen(fc.movetimer) == false

                release!(fc)

                @test isopen(fc.movetimer) == true
                close(fc.movetimer) # So we don't create garbage after fc.movedelay seconds
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
