@testset "Fitness" begin

    struct MockFitness <: AbstractFitness f end
    NaiveGAflux.fitness(s::MockFitness, f) = s.f

    @testset "AccuracyFitness" begin
        struct DummyIter end
        Base.iterate(::DummyIter) = (([0 1 0; 1 0 0; 0 0 1], [1 0 0; 0 1 0; 0 0 1]), 1)
        Base.iterate(::DummyIter, state) = state == 1 ? (([0 1 0; 1 0 0; 0 0 1], [0 1 0; 1 0 1; 0 0 0]), 2) : nothing

        @test fitness(AccuracyFitness(DummyIter()), identity) == 0.5
    end

    @testset "TrainAccuracyFitness" begin
        import NaiveGAflux: Train, TrainLoss
        x = nothing
        f(x) = [0 1 0; 1 0 0; 0 0 1]
        y =    [0 0 0; 1 1 0; 0 0 1]

        taf = TrainAccuracyFitness()
        @test_throws AssertionError fitness(taf, (args...) -> error("shall not be called"))

        fi = instrument(Train(), taf, f)
        li = instrument(TrainLoss(), taf, Flux.mse)

        @test li(fi(x), y) == 2//9
        @test fitness(taf, (args...) -> error("shall not be called")) == 0.5

        reset!(taf)
        @test_throws AssertionError fitness(taf, (args...) -> error("shall not be called"))
    end

    @testset "MapFitness" begin
        NaiveGAflux.instrument(::NaiveGAflux.AbstractFunLabel, s::MockFitness, f) = x -> s.f*f(x)

        @test fitness( MapFitness(f -> 2f, MockFitness(3)), identity) == 6
        @test instrument(NaiveGAflux.Train(), MapFitness(identity, MockFitness(2)), x -> 3x)(5) == 2*3*5

        wasreset = false
        NaiveGAflux.reset!(s::MockFitness) = wasreset=true

        reset!(MapFitness(identity, MockFitness(2)))

        @test wasreset
    end

    @testset "TimeFitness" begin
        import NaiveGAflux: Train, Validate
        tf = TimeFitness(Train(), 1)
        function sleepret(t)
            sleep(t)
            return t
        end

        @test fitness(tf, identity) == 0

        # First call doesn't count
        @test instrument(Train(), tf, sleepret)(0.5) == 0.5
        @test instrument(Train(), tf, sleepret)(0.02) == 0.02
        @test instrument(Validate(), tf, sleepret)(0.5) == 0.5
        @test instrument(Train(), tf, sleepret)(0.04) == 0.04

        @test fitness(tf, identity) < 0.5 / 3

        reset!(tf)

        @test fitness(tf, identity) == 0
    end

    @testset "SizeFitness" begin
        import NaiveGAflux: Validate

        sf = SizeFitness()
        l = Dense(2,3)

        @test fitness(sf, l) == 9

        @test_logs (:warn, "SizeFitness got zero parameters! Check your fitness function!") fitness(sf, identity)

        @test instrument(Validate(), sf, l) == l

        @test fitness(sf, identity) == 9

        sf = NanGuard(Validate(), SizeFitness())

        @test instrument(Validate(), sf, l) != l

        @test fitness(sf, identity) == 9
    end

    @testset "FitnessCache" begin
        struct RandomFitness <: AbstractFitness end
        NaiveGAflux.fitness(s::RandomFitness, f) = rand()

        cf = FitnessCache(RandomFitness())
        @test fitness(cf, identity) == fitness(cf, identity)

        val = fitness(cf, identity)
        reset!(cf)
        @test fitness(cf, identity) != val
    end

    @testset "NanGuard" begin

        import NaiveGAflux: Train, TrainLoss, Validate

        badfun(x::Real, val=NaN) = val
        function badfun(x::AbstractArray, val=NaN)
            x[1] = val
            return x
        end
        badfun(x::AbstractArray{<:Integer}, val=NaN) = badfun(Float64.(x), val)

        @testset "NanGuard $val" for val in (NaN, Inf)

            ng = NanGuard(Train(), MockFitness(1))

            okfun = instrument(Train(), ng, identity)
            nokfun = instrument(Train(), ng, x -> badfun(x, val))

            @test okfun(5) == 5
            @test fitness(ng, identity) == 1

            @test okfun([3,4,5]) == [3,4,5]
            @test fitness(ng, identity) == 1

            # Overwritten by NanGuard
            @test (@test_logs (:warn, Regex("$val detected")) nokfun(3)) == 0
            @test nokfun(3) == 0
            @test fitness(ng, identity) == 0

            @test okfun(3) == 0
            @test fitness(ng, identity) == 0

            wasreset = false
            NaiveGAflux.reset!(::MockFitness) = wasreset = true
            reset!(ng)

            @test wasreset

            @test okfun(5) == 5
            @test fitness(ng, identity) == 1

            @test (@test_logs (:warn, Regex("$val detected")) nokfun([1,2,3])) == [0,0,0]
            @test nokfun([1,2,3]) == [0,0,0]

            @test fitness(ng, identity) == 0
        end

        @testset "NanGuard all" begin
            ng = NanGuard(MockFitness(1))

            okfun(t) = instrument(t, ng, identity)
            nokfun(t) = instrument(t, ng, badfun)

            @test okfun(Train())(3) == 3
            @test okfun(TrainLoss())([1,2,3]) == [1,2,3]
            @test okfun(Validate())([1,2,3]) == [1,2,3]

            @test fitness(ng, identity) == 1

            @test (@test_logs (:warn, r"NaN detected") nokfun(Train())(3)) == 0
            @test okfun(TrainLoss())([1,2,3]) == [1,2,3]
            @test okfun(Validate())([1,2,3]) == [1,2,3]

            @test fitness(ng, identity) == 0

            @test okfun(Train())(3) == 0
            @test (@test_logs (:warn, r"NaN detected") nokfun(TrainLoss())(Float32[1,2,3])) == [0,0,0]
            @test okfun(Validate())([1,2,3]) == [1,2,3]

            @test fitness(ng, identity) == 0

            @test okfun(Train())(3) == 0
            @test okfun(TrainLoss())([1,2,3]) == [0,0,0]
            @test (@test_logs (:warn, r"NaN detected") nokfun(Validate())([1,2,3])) == [0,0,0]

            @test fitness(ng, identity) == 0

            @test okfun(Train())(3) == 0
            @test okfun(TrainLoss())([1,2,3]) == [0,0,0]
            @test okfun(Validate())([1,2,3]) == [0,0,0]

            @test fitness(ng, identity) == 0

            reset!(ng)

            @test okfun(Train())(3) == 3
            @test okfun(TrainLoss())([1,2,3]) == [1,2,3]
            @test okfun(Validate())([1,2,3]) == [1,2,3]

            @test fitness(ng, identity) == 1
        end
    end

    @testset "AggFitness" begin
        af = AggFitness(+, MockFitness(3), MockFitness(2))

        @test fitness(af, identity) == 5

        NaiveGAflux.instrument(::NaiveGAflux.AbstractFunLabel, s::MockFitness, f) = x -> s.f*f(x)
        @test instrument(NaiveGAflux.Train(), af, x -> 5x)(7) == 2*3*5*7

        nreset = 0
        NaiveGAflux.reset!(::MockFitness) = nreset += 1
        reset!(af)

        @test nreset == 2

        @test_throws ErrorException AggFitness(+)
    end
end
