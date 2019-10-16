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
