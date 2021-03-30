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
        pop = @test_logs (:info, "Begin generation 1") (:info, "Begin generation 2") (:info, "Begin generation 3") (:info, r"Mutate model") match_mode=:any fit(c, x, y, fitnesstrategy=TrainAccuracyVsSize(), trainstrategy=t, evolutionstrategy = GlobalOptimizerMutation(EliteAndTournamentSelection(popsize=c.popsize, nelites=1, k=2)), stopcriterion = pop -> generation(pop) == 3)

        @test length(pop) == c.popsize
        @test modelname.(pop) == ["model$i" for i in 1:length(pop)]

        @test unique(globallearningrate.(pop)) != [1]
        @test length(unique(globallearningrate.(pop))) == 1
    end

    @testset "PruneLongRunning" begin
        import NaiveGAflux.AutoFlux.ImageClassification: PruneLongRunning, TrainSplitAccuracy, fitnessfun

        x = ones(Float32, 5,5,3,4)
        y = [1 1 1 1; 0 0 0 0]

        fs = PruneLongRunning(TrainSplitAccuracy(nexamples=2, batchsize=2), 0.1, 0.3)

        xx, yy, fg = fitnessfun(fs, x, y)

        @test size(xx) == (5,5,3,2)
        @test size(yy) == (2, 2)

        function sleepret(t)
            t0 = time()
            # Busy wait to avoid yielding since this causes sporadic failures in CI
            while time() - t0 < t
                1+1
            end
            return t
        end

        ff = fg()
        sleepreti = instrument(NaiveGAflux.Train(), ff, sleepret)
        instrument(NaiveGAflux.Validate(), ff, Dense(1,1))

        # a little warmup to hopefully remove any compiler delays
        @test sleepreti(0.1) == 0.1
        @test sleepreti(0.7) == 0.7
        @test fitness(ff, x -> [1 0; 0 1]) == 0

        reset!(ff)

        @test sleepreti(0.01) == 0.01
        @test sleepreti(0.02) == 0.02
        fitness(ff, x -> [1 0; 0 1]) # Avoid compiler delays?
        @test fitness(ff, x -> [1 0; 0 1]) == 0.501 #SizeFitness gives 0.001 extra

        sleepreti(0.4)
        @test fitness(ff, x -> [1 0; 0 1]) < 0.501  #SizeFitness gives 0.001 extra

        sleepreti(0.7)
        @test fitness(ff, x -> [1 0; 0 1]) == 0
    end

    @testset "sizevs" begin
        import NaiveGAflux.AutoFlux.ImageClassification: sizevs
        struct SizeVsTestFitness <: AbstractFitness
            fitness
        end
        NaiveGAflux.fitness(s::SizeVsTestFitness, f) = s.fitness

        struct SizeVsTestFakeModel
            nparams::Int
        end
        Flux.params(s::SizeVsTestFakeModel) = (order=1:s.nparams,)

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

    @testset "TrainIterStrategy" begin
        import NaiveGAflux.AutoFlux.ImageClassification: TrainIterStrategy, trainiter
        nexamples = 10
        x = randn(Float32, 4,4,3, nexamples)
        y = rand(0:7, nexamples)

        @testset "Test $ne epochs and $nbpg batches per generation" for ne in (1, 2, 10), nbpg in (2, 10)
            bs = 3
            s = TrainIterStrategy(nepochs=ne, batchsize=bs, nbatches_per_gen=nbpg)
            itr = trainiter(s, x, y)

            totsize = ne * ceil(nexamples / bs)
            @test mapreduce(length ∘ collect, +, itr) == totsize
            @test length(itr |> first) == min(nbpg, totsize)
            @test size(itr |> first |> first |> first, 4) == bs

            # All models shall see the same examples
            for iitr in itr
                @test collect(iitr) == collect(iitr)
            end
        end
    end
end
