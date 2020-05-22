@testset "AutoFlux" begin

    @testset "ImageClassifier smoketest" begin
        using NaiveGAflux.AutoFlux
        import NaiveGAflux.AutoFlux.ImageClassification: TrainSplitAccuracy, TrainStrategy, TrainAccuracyVsSize, EliteAndTournamentSelection, EliteAndSusSelection, GlobalOptimizerMutation
        using Random

        # Use Float64 instead of Float32 due to https://github.com/FluxML/Flux.jl/issues/979

        # Workaround as losses fail with Flux.OneHotMatrix on Appveyor x86 (works everywhere else)
        onehot(y) = Float64.(Flux.onehotbatch(y, 0:5))

        rng = MersenneTwister(123)
        x = randn(rng, Float64, 32,32,2,4)
        y = onehot(rand(rng, 0:5,4))

        c = ImageClassifier(popsize = 5, seed=1)
        f = TrainSplitAccuracy(nexamples=1, batchsize=1)
        t = TrainStrategy(nepochs=1, batchsize=1, nbatches_per_gen=1)

        dummydir = joinpath(NaiveGAflux.modeldir, "ImageClassifier_smoketest")

        # Logs are mainly to prevent CI timeouts
        @info "\tSmoke test with TrainSplitAccuracy and EliteAndSusSelection"
        pop = @test_logs (:info, "Begin generation 1") (:info, "Begin generation 2") (:info, "Begin generation 3") (:info, r"Mutate model") match_mode=:any fit(c, x, y, fitnesstrategy=f, trainstrategy=t, evolutionstrategy = GlobalOptimizerMutation(EliteAndSusSelection(popsize=c.popsize, nelites=1)), mdir = dummydir)

        @test length(pop) == c.popsize

        # Now try TrainAccuracyVsSize and EliteAndTournamentSelection
        @info "\tSmoke test with TrainAccuracyVsSize and EliteAndTournamentSelection"
        pop = @test_logs (:info, "Begin generation 1") (:info, "Begin generation 2") (:info, "Begin generation 3") (:info, r"Mutate model") match_mode=:any fit(c, x, y, fitnesstrategy=TrainAccuracyVsSize(), trainstrategy=t, evolutionstrategy = GlobalOptimizerMutation(EliteAndTournamentSelection(popsize=c.popsize, nelites=1, k=2)), mdir = dummydir)

        @test length(pop) == c.popsize
    end

    @testset "PruneLongRunning" begin
        import NaiveGAflux.AutoFlux.ImageClassification: PruneLongRunning,  TrainSplitAccuracy, fitnessfun

        x = ones(Float32, 5,5,3,4)
        y = [1 1 1 1; 0 0 0 0]

        fs = PruneLongRunning(TrainSplitAccuracy(nexamples=2, batchsize=2), 0.01, 0.03)

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
        @test sleepreti(0.01) == 0.01
        @test sleepreti(0.07) == 0.07
        @test fitness(ff, x -> [1 0; 0 1]) == 0

        reset!(ff)

        @test sleepreti(0.001) == 0.001
        @test sleepreti(0.002) == 0.002
        fitness(ff, x -> [1 0; 0 1]) # Avoid compiler delays?
        @test fitness(ff, x -> [1 0; 0 1]) == 0.501 #SizeFitness gives 0.001 extra

        sleepreti(0.04)
        @test fitness(ff, x -> [1 0; 0 1]) < 0.501  #SizeFitness gives 0.001 extra

        sleepreti(0.07)
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
end
