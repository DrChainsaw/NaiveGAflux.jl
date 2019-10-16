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
