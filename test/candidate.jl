@testset "Fitness" begin

    @testset "AccuracyFitness" begin
        struct DummyIter end
        Base.iterate(::DummyIter) = (([0 1 0; 1 0 0; 0 0 1], [1 0 0; 0 1 0; 0 0 1]), 1)
        Base.iterate(::DummyIter, state) = state == 1 ? (([0 1 0; 1 0 0; 0 0 1], [0 1 0; 1 0 1; 0 0 0]), 2) : nothing

        @test fitness(AccuracyFitness(DummyIter()), identity) == 0.5
    end

    @testset "MapFitness" begin
        struct MockFitness <: AbstractFitness f end
        NaiveGAflux.fitness(s::MockFitness, f) = s.f
        NaiveGAflux.instrument(::NaiveGAflux.AbstractFunLabel, s::MockFitness, f::Function) = x -> s.f*f(x)

        @test fitness( MapFitness(f -> 2f, MockFitness(3)), identity) == 6
        @test instrument(NaiveGAflux.Train(), MapFitness(identity, MockFitness(2)), x -> 3x)(5) == 2*3*5
    end

    @testset "TimeFitness" begin
        import NaiveGAflux: Train, Validate
        tf = TimeFitness(Train())
        function sleepret(t)
            sleep(t)
            return t
        end

        @test instrument(Train(), tf, sleepret)(0.02) == 0.02
        @test instrument(Validate(), tf, sleepret)(0.1) == 0.1
        @test instrument(Train(), tf, sleepret)(0.04) == 0.04

        @test fitness(tf, identity) ≈ 0.03 rtol=0.1
    end

    @testset "FitnessCache" begin
        struct RandomFitness <: AbstractFitness end
        NaiveGAflux.fitness(s::RandomFitness, f) = rand()

        cf = FitnessCache(RandomFitness())
        @test fitness(cf, identity) == fitness(cf, identity)
    end
end


@testset "Evolution" begin

    @testset "EliteSelection" begin
        struct MockCand <: AbstractCandidate
            val::Real
        end
        NaiveGAflux.fitness(c::MockCand) = c.val

        pop = MockCand.([3, 7, 4, 5, 9, 0])
        @test fitness.(evolve(EliteSelection(3), pop)) == [9, 7, 5]
    end

end
