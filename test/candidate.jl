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

    @testset "eagermutation" begin
        invertex = inputvertex("in", 3, FluxDense())
        hlayer = mutable("hlayer", Dense(3,4), invertex)
        outlayer = mutable("outlayer", Dense(4, 2), hlayer)
        graph = CompGraph(invertex, outlayer)

        Δnout(hlayer, 2)
        Δoutputs(hlayer, v -> ones(nout_org(v)))
        apply_mutation(graph)

        @test nout(layer(hlayer)) == 4
        @test nin(layer(outlayer)) == 4

        NaiveGAflux.eagermutation(graph)

        @test nout(layer(hlayer)) == 6
        @test nin(layer(outlayer)) == 6
    end

    @testset "Save models" begin
        testdir = "test_savemodels"

        # Weird name to avoid collisions (e.g. MockCand is defined in another testset and therefore unusable)
        struct SaveModelsCand <: AbstractCandidate
            i
        end
        NaiveGAflux.graph(c::SaveModelsCand) = c.i

        ndir = joinpath(testdir,"normal")
        pdir = joinpath(testdir,"persistent")
        cdir = joinpath(testdir,"curried")

        normal = SaveModelsCand.(1:10)
        persistent = PersistentArray(pdir, length(normal), i -> normal[i])

        curried = savemodels(cdir)

        filenames = map(i -> "$i.jld2", 1:length(normal))

        try
            savemodels(normal, ndir)
            @test all(isfile.(joinpath.(ndir, filenames)))
            rm(ndir, force=true, recursive=true)

            savemodels(persistent)
            @test all(isfile.(joinpath.(pdir, "models", filenames)))
            rm(pdir, force=true, recursive=true)

            curried(normal)
            @test all(isfile.(joinpath.(cdir, filenames)))
            rm(cdir, force=true, recursive=true)
        finally
            rm(testdir, force=true, recursive=true)
        end

    end

end
