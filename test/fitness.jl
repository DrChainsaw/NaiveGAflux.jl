@testset "Fitness" begin

    struct MockFitness <: AbstractFitness f end
    NaiveGAflux._fitness(s::MockFitness, c) = s.f

    struct MockCandidate{F} <: AbstractCandidate 
        f::F
    end
    IdCand() = MockCandidate(identity)
    NaiveGAflux.graph(c::MockCandidate, f=identity) = f(c.f)

    @testset "LogFitness" begin
        import NaiveGAflux: FittedCandidate
        iv = inputvertex("in", 2)
        c = CompGraph(iv, mutable(Dense(2,3), iv)) |> CandidateModel
        lf = LogFitness(MockFitness(3))
        @test @test_logs (:info, " Candidate:   1\tvertices:   2\tparams:  0.01k\tfitness: 3") fitness(lf, c) == 3
        @test @test_logs (:info, " Candidate:   2\tvertices:   2\tparams:  0.01k\tfitness: 3") fitness(lf, c) == 3
        fc = FittedCandidate(1, 3, c)
        @test @test_logs (:info, " Candidate:   1\tvertices:   2\tparams:  0.01k\tfitness: 3") fitness(lf, fc) == 3
        @test @test_logs (:info, " Candidate:   2\tvertices:   2\tparams:  0.01k\tfitness: 3") fitness(lf, fc) == 3
    end

    @testset "GpuFitness" begin
        label = Val(:GpuTest)
        wasgpumapped = false
        function Flux.gpu(x::MockCandidate{Val{:GpuTest}}) 
            wasgpumapped = true
            return x
        end
        
        wascpumapped = false
        function Flux.cpu(x::MockCandidate{Val{:GpuTest}}) 
            wascpumapped = true
            return x
        end

        @test fitness(GpuFitness(MockFitness(4)), MockCandidate(label)) == 4
        @test wasgpumapped
        @test wascpumapped
    end

    @testset "AccuracyFitness" begin
        struct DummyIter end
        Base.iterate(::DummyIter) = (([0 1 0 0; 1 0 0 1; 0 0 1 0], [1 0 0 0; 1 0 0 1; 0 0 1 0]), 1)
        Base.iterate(::DummyIter, state) = state == 1 ? (([1 0; 0 0; 0 1], [0 0; 1 1; 0 0]), 2) : nothing

        @test fitness(AccuracyFitness(DummyIter()), IdCand()) == 0.5
    end

    @testset "TrainThenFitness" begin
        iv = inputvertex("in", 2)
        ov = mutable(Dense(2, 2), iv)
        cand = CandidateModel(CompGraph(iv, ov))

        ttf = TrainThenFitness(
            dataiter = [(ones(Float32, 2, 1), ones(Float32, 2, 1))],
            defaultloss=Flux.mse,
            defaultopt = Descent(0.0001),
            fitstrat = MockFitness(17),
            invalidfitness = 123.456)

        @test fitness(ttf, cand) === 17

        struct NaNCandidateModel <: AbstractCandidate end
        NaiveGAflux.ninputs(::NaNCandidateModel) = 1
        NaiveGAflux.graph(f::NaNCandidateModel) = f 
        (::NaNCandidateModel)(x) = x
        NaiveGAflux.lossfun(::NaNCandidateModel; default) = (args...) -> NaN32

        @test @test_logs (:warn, "NaN loss detected when training!") fitness(ttf, NaNCandidateModel()) == 123.456
    end

    @testset "TrainAccuracyFitness" begin
        dataiter  = [([0 1 0; 1 0 0; 0 0 1], [0 0 0; 1 1 0; 0 0 1])]
        iv = inputvertex("in", 3)
        idgraph = CompGraph(iv, iv)

        taf = TrainAccuracyFitness(;drop=0.5,dataiter, defaultloss = Flux.Losses.mse, defaultopt = Descent(0.001))

        # Note: First example will be dropped
        @test fitness(taf, CandidateModel(idgraph))  == 0.5
    end

    @testset "MapFitness" begin
        @test fitness(MapFitness(f -> 2f, MockFitness(3)), IdCand()) == 6
    end

    @testset "EwmaFitness" begin
        import NaiveGAflux: FittedCandidate

        @test fitness(EwmaFitness(MockFitness(10)), IdCand()) == 10
        @test fitness(EwmaFitness(MockFitness(20)), FittedCandidate(0, 10, IdCand())) == 15
        @test fitness(EwmaFitness(MockFitness(20)), FittedCandidate(0, 15, IdCand())) == 17.5
    end

    @testset "TimeFitness" begin

        function busysleep(t)
            t0 = time()
            # Busy wait to avoid yielding since this causes sporadic failures in CI
            while time() - t0 < t
                1+1
            end
        end
        tf = TimeFitness(MockFitness(13))
        NaiveGAflux._fitness(s::MockFitness, c::MockCandidate{typeof(busysleep)}) = (busysleep(0.1); s.f)
        ftime, f = fitness(tf, MockCandidate(busysleep))
        @test ftime ≈ 0.1 atol=0.01
        @test f == 13
    end

    @testset "SizeFitness" begin
        @test fitness(SizeFitness(), MockCandidate(Dense(2,3))) == 9
    end

    @testset "AggFitness" begin
        af = AggFitness(+, MockFitness(3), MockFitness(2))

        @test fitness(af, IdCand()) == 5
        @test_throws ErrorException AggFitness(+)
    end
end
