
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
