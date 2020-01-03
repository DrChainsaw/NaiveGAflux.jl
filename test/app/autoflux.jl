@testset "AutoFlux" begin

    @testset "ImageClassifier smoketest" begin
        using NaiveGAflux.AutoFlux
        import NaiveGAflux.AutoFlux.ImageClassification: TrainSplitAccuracy, TrainStrategy, TrainAccuracyVsSize
        using Random

        # Use Float64 instead of Float32 due to https://github.com/FluxML/Flux.jl/issues/979

        # Workaround as losses fail with Flux.OneHotMatrix on Appveyor x86 (works everywhere else)
        onehot(y) = Float64.(Flux.onehotbatch(y, 0:5))

        rng = MersenneTwister(123)
        x = randn(rng, Float64, 32,32,2,4)
        y = onehot(rand(rng, 0:5,4))

        c = ImageClassifier(popsize = 3, seed=12345)
        f = TrainSplitAccuracy(nexamples=1, batchsize=1)
        t = TrainStrategy(nepochs=1, batchsize=1, nbatches_per_gen=1)

        dummydir = joinpath(NaiveGAflux.modeldir, "ImageClassifier_smoketest")

        pop = @test_logs (:info, "Begin generation 1") (:info, "Begin generation 2") (:info, "Begin generation 3") (:info, r"Mutate model") match_mode=:any fit(c, x, y, fitnesstrategy=f, trainstrategy=t, mdir = dummydir)

        @test length(pop) == c.popsize

        # Now try TrainAccuracyVsSize
        pop = @test_logs (:info, "Begin generation 1") (:info, "Begin generation 2") (:info, "Begin generation 3") (:info, r"Mutate model") match_mode=:any fit(c, x, y, fitnesstrategy=TrainAccuracyVsSize(), trainstrategy=t, mdir = dummydir)

        @test length(pop) == c.popsize

    end

    @testset "PruneLongRunning" begin
        import NaiveGAflux.AutoFlux.ImageClassification: PruneLongRunning,  TrainSplitAccuracy, fitnessfun

        x = ones(Float32, 5,5,3,4)
        y = [1 1 1 1; 0 0 0 0]

        fs = PruneLongRunning(TrainSplitAccuracy(nexamples=2, batchsize=2), 0.1, 0.3)

        xx, yy, fg = fitnessfun(fs, x, y)

        @test size(xx) == (5,5,3,2)
        @test size(yy) == (2, 2)

        function sleepret(t)
            sleep(t)
            return t
        end

        ff = fg()

        sleepreti = instrument(NaiveGAflux.Train(), ff, sleepret)
        instrument(NaiveGAflux.Validate(), ff, Dense(1,1))

        @test sleepreti(0.01) == 0.01
        @test sleepreti(0.02) == 0.02
        @test fitness(ff, x -> [1 0; 0 1]) == 0.51 #SizeFitness gives 0.01 extra

        sleepreti(0.4)
        @test fitness(ff, x -> [1 0; 0 1]) < 0.51  #SizeFitness gives 0.01 extra

        sleepreti(0.7)
        @test fitness(ff, x -> [1 0; 0 1]) == 0

    end
end
