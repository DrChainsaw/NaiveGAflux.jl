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
        @test instrument(Validate(), tf, sleepret)(0.1) == 0.1
        @test instrument(Train(), tf, sleepret)(0.04) == 0.04

        @test fitness(tf, identity) ≈ 0.03 rtol=0.1

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
end

@testset "Candidate" begin

    @testset "CandidateModel" begin
        import NaiveGAflux: AbstractFunLabel, Train, Validate
        struct DummyFitness <: AbstractFitness end

        invertex = inputvertex("in", 3, FluxDense())
        hlayer = mutable("hlayer", Dense(3,4), invertex)
        outlayer = mutable("outlayer", Dense(4, 2), hlayer)
        graph = CompGraph(invertex, outlayer)

        cand = CandidateModel(graph, Flux.Descent(0.01), Flux.mse, DummyFitness())

        labs = []
        function NaiveGAflux.instrument(l::AbstractFunLabel, s::DummyFitness, f::Function)
            push!(labs, l)
            return f
        end

        Flux.train!(cand, [(ones(Float32, 3, 2), ones(Float32, 2,2))])

        @test labs == [Train()]

        NaiveGAflux.fitness(::DummyFitness, f) = 17
        @test fitness(cand) == 17

        @test labs == [Train(), Validate()]

        wasreset = false
        NaiveGAflux.reset!(::DummyFitness) = wasreset = true

        reset!(cand)

        @test wasreset

        evofun = evolvemodel(VertexMutation(MutationFilter(v -> name(v)=="hlayer", RemoveVertexMutation())))
        newcand = evofun(cand)

        @test nv(newcand.graph) == 2
        @test nv(cand.graph) == 3

        Flux.train!(cand, [(ones(Float32, 3, 2), ones(Float32, 2,2))])
        Flux.train!(newcand, [(ones(Float32, 3, 2), ones(Float32, 2,2))])
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
