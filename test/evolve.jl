
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

    @testset "TournamentSelection" begin
        pop = MockCand.(1:10)
        Random.shuffle(::MockRng, a::AbstractVector) = reverse(a)
        Random.rand(r::MockRng, n::Int) = map(i -> rand(r), 1:n)

        @test fitness.(evolve!(TournamentSelection(4, 2, 1, NoOpEvolution(), MockRng([0.05])), pop)) == [10, 8, 6, 4]
        @test fitness.(evolve!(TournamentSelection(8, 3, 0.7, NoOpEvolution(), MockRng([0.01, 0.9, 0.01])), pop)) == [9, 9, 7, 4, 2, 8, 5, 2]

        @test fitness.(evolve!(TournamentSelection(20, 2, 1, NoOpEvolution(), MockRng([0.01])), pop)) == repeat(10:-2:2, 4)

        # Edge case, nselect*k % popsize == 0
        @test fitness.(evolve!(TournamentSelection(5, 2, 1, NoOpEvolution(), MockRng([0.01])), pop)) == 10:-2:2
        # Edge case, nselect*k % popsize == 1
        @test fitness.(evolve!(TournamentSelection(2, 2, 1, NoOpEvolution(), MockRng([0.01])), MockCand.(1:3))) == [3, 2]
    end

    @testset "CombinedEvolution" begin
        pop = [DummyCand() for i in 1:5]
        @test evolve!(CombinedEvolution(NoOpEvolution(), NoOpEvolution()), pop) == vcat(pop,pop)
    end

    @testset "EvolutionChain" begin
        dfitness = EvolveCandidates(mc -> MockCand(2 * fitness(mc)))
        sqrfitness = EvolveCandidates(mc -> MockCand(fitness(mc)^2))
        @test evolve!(EvolutionChain(), MockCand.(1:5)) == MockCand.(1:5)
        @test evolve!(EvolutionChain(dfitness, dfitness, sqrfitness), MockCand.(1:10)) == MockCand.((4:4:40).^2)
    end

    @testset "PairCandidates" begin
        struct PairConsumer <: AbstractEvolution end
        nseen = 0
        function NaiveGAflux._evolve!(::PairConsumer, pop)
            nseen = length(pop)
            @test typeof(pop) == Vector{Tuple{MockCand, MockCand}}
            return pop
        end

        @test evolve!(PairCandidates(NoOpEvolution()), []) == []
        pop = MockCand.(1:9)
        @test evolve!(PairCandidates(PairConsumer()), pop) == pop
        @test nseen == 5
        pop = MockCand.(1:4)
        @test evolve!(PairCandidates(PairConsumer()), pop) == pop
        @test nseen == 2
    end

    @testset "EvolveCandidates" begin
        pop = MockCand.([2,3,5,9])
        @test fitness.(evolve!(EvolveCandidates(mc -> MockCand(2mc.val)), pop)) == 2fitness.(pop)
    end

end
