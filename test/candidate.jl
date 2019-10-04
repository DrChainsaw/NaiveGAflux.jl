@testset "Fitness" begin

    struct MockFitness <: AbstractFitness f end
    NaiveGAflux.fitness(s::MockFitness, f) = s.f

    @testset "AccuracyFitness" begin
        struct DummyIter end
        Base.iterate(::DummyIter) = (([0 1 0; 1 0 0; 0 0 1], [1 0 0; 0 1 0; 0 0 1]), 1)
        Base.iterate(::DummyIter, state) = state == 1 ? (([0 1 0; 1 0 0; 0 0 1], [0 1 0; 1 0 1; 0 0 0]), 2) : nothing

        @test fitness(AccuracyFitness(DummyIter()), identity) == 0.5
    end

    @testset "MapFitness" begin
        NaiveGAflux.instrument(::NaiveGAflux.AbstractFunLabel, s::MockFitness, f::Function) = x -> s.f*f(x)

        @test fitness( MapFitness(f -> 2f, MockFitness(3)), identity) == 6
        @test instrument(NaiveGAflux.Train(), MapFitness(identity, MockFitness(2)), x -> 3x)(5) == 2*3*5

        wasreset = false
        NaiveGAflux.reset!(s::MockFitness) = wasreset=true

        reset!(MapFitness(identity, MockFitness(2)))

        @test wasreset
    end

    @testset "TimeFitness" begin
        import NaiveGAflux: Train, Validate
        tf = TimeFitness(Train())
        function sleepret(t)
            sleep(t)
            return t
        end

        @test fitness(tf, identity) == 0

        @test instrument(Train(), tf, sleepret)(0.02) == 0.02
        @test instrument(Validate(), tf, sleepret)(0.5) == 0.5
        @test instrument(Train(), tf, sleepret)(0.04) == 0.04

        @test fitness(tf, identity) < 0.5 / 3

        reset!(tf)

        @test fitness(tf, identity) == 0
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
        function badfun(x::TrackedArray, val=NaN)
            badfun(x.data, val)
            return x
        end

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

            @test (@test_logs (:warn, Regex("$val detected")) nokfun(param([1,2,3]))) == [0,0,0]
            @test nokfun(param([1,2,3])) == [0,0,0]

            @test fitness(ng, identity) == 0
        end

        @testset "NanGuard all" begin
            ng = NanGuard(MockFitness(1))

            okfun(t) = instrument(t, ng, identity)
            nokfun(t) = instrument(t, ng, badfun)

            @test okfun(Train())(3) == 3
            @test okfun(TrainLoss())([1,2,3]) == [1,2,3]
            @test okfun(Validate())(param([1,2,3])) == [1,2,3]

            @test fitness(ng, identity) == 1

            @test (@test_logs (:warn, r"NaN detected") nokfun(Train())(3)) == 0
            @test okfun(TrainLoss())([1,2,3]) == [1,2,3]
            @test okfun(Validate())(param([1,2,3])) == [1,2,3]

            @test fitness(ng, identity) == 0

            @test okfun(Train())(3) == 0
            @test (@test_logs (:warn, r"NaN detected") nokfun(TrainLoss())(Float32[1,2,3])) == [0,0,0]
            @test okfun(Validate())(param([1,2,3])) == [1,2,3]

            @test fitness(ng, identity) == 0

            @test okfun(Train())(3) == 0
            @test okfun(TrainLoss())([1,2,3]) == [0,0,0]
            @test (@test_logs (:warn, r"NaN detected") nokfun(Validate())(param([1,2,3]))) == [0,0,0]

            @test fitness(ng, identity) == 0

            @test okfun(Train())(3) == 0
            @test okfun(TrainLoss())([1,2,3]) == [0,0,0]
            @test okfun(Validate())(param([1,2,3])) == [0,0,0]

            @test fitness(ng, identity) == 0

            reset!(ng)

            @test okfun(Train())(3) == 3
            @test okfun(TrainLoss())([1,2,3]) == [1,2,3]
            @test okfun(Validate())(param([1,2,3])) == [1,2,3]

            @test fitness(ng, identity) == 1
        end
    end

    @testset "AggFitness" begin
        af = AggFitness(sum, MockFitness(3), MockFitness(2))

        @test fitness(af, identity) == 5

        NaiveGAflux.instrument(::NaiveGAflux.AbstractFunLabel, s::MockFitness, f::Function) = x -> s.f*f(x)
        @test instrument(NaiveGAflux.Train(), af, x -> 5x)(7) == 2*3*5*7

        nreset = 0
        NaiveGAflux.reset!(::MockFitness) = nreset += 1
        reset!(af)

        @test nreset == 2
    end
end

@testset "Candidate" begin

    @testset "CandidateModel $wrp" for wrp in (identity, HostCandidate, CacheCandidate)
        import NaiveGAflux: AbstractFunLabel, Train, TrainLoss, Validate
        struct DummyFitness <: AbstractFitness end

        invertex = inputvertex("in", 3, FluxDense())
        hlayer = mutable("hlayer", Dense(3,4), invertex)
        outlayer = mutable("outlayer", Dense(4, 2), hlayer)
        graph = CompGraph(invertex, outlayer)

        cand = wrp(CandidateModel(graph, Flux.Descent(0.01), (x,y) -> sum(x .- y), DummyFitness()))

        labs = []
        function NaiveGAflux.instrument(l::AbstractFunLabel, s::DummyFitness, f::Function)
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

        evofun = evolvemodel(VertexMutation(MutationFilter(v -> name(v)=="hlayer", RemoveVertexMutation())))
        newcand = evofun(cand)

        @test nv(NaiveGAflux.graph(newcand)) == 2
        @test nv(NaiveGAflux.graph(cand)) == 3

        Flux.train!(cand, data(cand))
        Flux.train!(newcand, data(newcand))
    end

end

@testset "Evolution" begin

    struct MockCand <: AbstractCandidate
        val::Real
    end
    NaiveGAflux.fitness(c::MockCand) = c.val

    struct DummyCand <: AbstractCandidate end


    @testset "EliteSelection" begin
        pop = MockCand.([3, 7, 4, 5, 9, 0])
        @test fitness.(evolve!(EliteSelection(3), pop)) == [9, 7, 5]
    end

    @testset "ResetAfterSelection" begin
        nreset = 0
        NaiveGAflux.reset!(::DummyCand) = nreset += 1

        pop = [DummyCand() for i in 1:5]
        @test evolve!(ResetAfterEvolution(NoOpEvolution()), pop) == pop

        @test nreset == length(pop)
    end

    @testset "SusSelection" begin
        pop = MockCand.(0.1:0.1:0.8)
        @test fitness.(evolve!(SusSelection(4, NoOpEvolution(), MockRng([0.05])), pop)) == [0.1, 0.4, 0.6, 0.7]
    end

    @testset "CombinedEvolution" begin
        pop = [DummyCand() for i in 1:5]
        @test evolve!(CombinedEvolution(NoOpEvolution(), NoOpEvolution()), pop) == vcat(pop,pop)
    end

    @testset "EvolveCandidates" begin
        pop = MockCand.([2,3,5,9])
        @test fitness.(evolve!(EvolveCandidates(mc -> MockCand(2mc.val)), pop)) == 2fitness.(pop)
    end

end
