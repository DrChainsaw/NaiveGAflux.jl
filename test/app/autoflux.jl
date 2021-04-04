@testset "AutoFlux" begin

    @testset "ImageClassifier smoketest" begin
        using NaiveGAflux.AutoFlux
        import NaiveGAflux.AutoFlux.ImageClassification: TrainSplitAccuracy, TrainIterConfig, BatchedIterConfig, ShuffleIterConfig, AccuracyVsSize, TrainAccuracyVsSize, EliteAndTournamentSelection, EliteAndSusSelection, GlobalOptimizerMutation, modelname
        using Random

        rng = MersenneTwister(123)
        x = randn(rng, Float32, 32,32,2,4)
        y = Float32.(Flux.onehotbatch(rand(rng, 0:5,4), 0:5))

        dummydir = joinpath(NaiveGAflux.modeldir, "ImageClassifier_smoketest")
        c = ImageClassifier(popsize = 5, seed=1, mdir=dummydir, insize=size(x), outsize=size(y,1))

        f = TrainSplitAccuracy(;split=0.25, 
            accuracyconfig=BatchedIterConfig(batchsize=1),
            accuracyfitness=data -> AccuracyVsSize(data, 3;accwrap=EwmaFitness),
            trainconfig=TrainIterConfig(nbatches_per_gen=1, baseconfig=ShuffleIterConfig(batchsize=1))) 


        # Logs are mainly to prevent CI timeouts
        @info "\tSmoke test with TrainSplitAccuracy and EliteAndSusSelection"
        pop = @test_logs (:info, "Begin generation 1") (:info, "Begin generation 2") (:info, "Begin generation 3") (:info, r"Mutate model") match_mode=:any fit(c, x, y, fitnesstrategy=f, evolutionstrategy = GlobalOptimizerMutation(EliteAndSusSelection(popsize=c.popsize, nelites=1)), stopcriterion = pop -> generation(pop) == 3)

        @test length(pop) == c.popsize
        @test modelname.(pop) == ["model$i" for i in 1:length(pop)]

        globallearningrate(c::AbstractCandidate) = globallearningrate(c.c)
        globallearningrate(c::CandidateOptModel) = globallearningrate(c.opt)
        globallearningrate(o::Flux.Optimiser) = prod(globallearningrate.(o.os))
        globallearningrate(o) = 1
        globallearningrate(o::ShieldedOpt{Descent}) = o.opt.eta

        @test unique(globallearningrate.(pop)) != [1]
        @test length(unique(globallearningrate.(pop))) == 1

        # Now try TrainAccuracyVsSize and EliteAndTournamentSelection
        @info "\tSmoke test with TrainAccuracyVsSize and EliteAndTournamentSelection"
        pop = @test_logs (:info, "Begin generation 1") (:info, "Begin generation 2") (:info, "Begin generation 3") (:info, r"Mutate model") match_mode=:any fit(c, x, y, fitnesstrategy=TrainAccuracyVsSize(), evolutionstrategy = GlobalOptimizerMutation(EliteAndTournamentSelection(popsize=c.popsize, nelites=1, k=2)), stopcriterion = pop -> generation(pop) == 3)

        @test length(pop) == c.popsize
        @test modelname.(pop) == ["model$i" for i in 1:length(pop)]

        @test unique(globallearningrate.(pop)) != [1]
        @test length(unique(globallearningrate.(pop))) == 1
    end

    @testset "sizevs" begin
        import NaiveGAflux.AutoFlux.ImageClassification: sizevs
        struct SizeVsTestFitness <: AbstractFitness
            fitness
        end
        NaiveGAflux._fitness(s::SizeVsTestFitness, c) = s.fitness

        struct SizeVsTestFakeModel <: AbstractCandidate
            nparams::Int
        end
        NaiveGAflux.nparams(s::SizeVsTestFakeModel) = s.nparams

        basefitness = SizeVsTestFitness(0.12345678)

        testfitness = sizevs(basefitness, 2)
        @test fitness(testfitness, SizeVsTestFakeModel(1)) == 0.121
        @test fitness(testfitness, SizeVsTestFakeModel(100_000)) ≈ 0.12001

        testfitness = sizevs(basefitness, 3)
        @test fitness(testfitness, SizeVsTestFakeModel(1)) == 0.1231
        @test fitness(testfitness, SizeVsTestFakeModel(100_000)) ≈ 0.12301

        testfitness = sizevs(basefitness, 4)
        @test fitness(testfitness, SizeVsTestFakeModel(1)) == 0.12351
        @test fitness(testfitness, SizeVsTestFakeModel(100_000)) ≈ 0.12351
    end

    @testset "TrainIterConfig" begin
        import NaiveGAflux.AutoFlux.ImageClassification: TrainIterConfig, ShuffleIterConfig, dataiter
        nexamples = 10
        x = randn(Float32, 4,4,3, nexamples)
        y = rand(0:7, nexamples)

        @testset "Test $ne epochs and $nbpg batches per generation" for ne in (1, 2, 10), nbpg in (2, 10)
            bs = 3
            s = TrainIterConfig(nbatches_per_gen=nbpg, baseconfig=ShuffleIterConfig(batchsize=bs))
            itr =  Iterators.take(dataiter(s, x, y), cld(ne * nexamples, nbpg))

            totnrofexamples = ne * nexamples
            @test mapreduce(length ∘ collect, +, itr) == totnrofexamples
            @test length(itr |> first) == min(nbpg, totnrofexamples)
            @test size(itr |> first |> first |> first, 4) == bs

            # All models shall see the same examples
            for iitr in itr
                @test collect(iitr) == collect(iitr)
            end
        end
    end
end
