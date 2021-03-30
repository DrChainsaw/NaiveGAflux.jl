@testset "Fitness" begin

    struct MockFitness <: AbstractFitness f end
    NaiveGAflux.fitness(s::MockFitness, c, gen) = s.f

    struct MockCandidate{F} <: AbstractCandidate 
        f::F
    end
    IdCand() = MockCandidate(identity)
    NaiveGAflux.graph(c::MockCandidate, f=identity) = f(c.f)
    NaiveGAflux.fitness(f::AbstractFitness, c::MockCandidate) = fitness(f, c, 0)

    @testset "LogFitness" begin
        iv = inputvertex("in", 2)
        c = CompGraph(iv, mutable(Dense(2,3), iv)) |> MockCandidate
        @test @test_logs (:info, "  Candidate: 1\tvertices: 2\tparams:  0.01k\tfitness: 3.000000") fitness(LogFitness(MockFitness(3)), c) == 3
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

    @testset "TrainAccuracyFitness" begin
        import NaiveGAflux: Train, TrainLoss
        x = nothing
        f(x) = [0 1 0; 1 0 0; 0 0 1]
        y =    [0 0 0; 1 1 0; 0 0 1]

        taf = TrainAccuracyFitness()
        @test_throws AssertionError fitness(taf, (args...) -> error("shall not be called"))

        fi = instrument(Train(), taf, f)
        li = instrument(TrainLoss(), taf, Flux.Losses.mse)

        @test li(fi(x), y) ≈ 2/9
        @test fitness(taf, (args...) -> error("shall not be called")) == 0.5

        reset!(taf)
        @test_throws AssertionError fitness(taf, (args...) -> error("shall not be called"))
    end

    @testset "MapFitness" begin
        @test fitness( MapFitness(f -> 2f, MockFitness(3)), IdCand()) == 6
    end

    @testset "EwmaFitness" begin
        cnt = 0
        fvals = [10.0, 20.0, 20.0]
        function stepfun(f)
            cnt = cnt+1
            return fvals[cnt] * f
        end


        ef = EwmaFitness(MapFitness(stepfun, MockFitness(1)))
        efc = deepcopy(ef)

        @test fitness(ef, IdCand()) == fitness(efc, IdCand()) == 10
        @test fitness(ef, IdCand()) == fitness(efc, IdCand()) == 15
        @test fitness(ef, IdCand()) == fitness(efc, IdCand()) == 17.5
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
        NaiveGAflux.fitness(s::MockFitness, c::MockCandidate{typeof(busysleep)}, gen) = (busysleep(0.1); s.f)
        ftime, f = fitness(tf, MockCandidate(busysleep))
        @test ftime ≈ 0.1 atol=0.01
        @test f == 13
    end

    @testset "SizeFitness" begin
        import NaiveGAflux: Validate

        sf = SizeFitness()
        c = MockCandidate(Dense(2,3))

        @test fitness(sf, c) == 9
    end

    @testset "AggFitness" begin
        af = AggFitness(+, MockFitness(3), MockFitness(2))

        @test fitness(af, IdCand()) == 5
        @test_throws ErrorException AggFitness(+)
    end
end
