@testset "AutoFlux" begin

    @testset "ImageClassifier smoketest" begin
        using NaiveGAflux.AutoFlux
        import NaiveGAflux.AutoFlux.ImageClassification: TrainSplitAccuracy, TrainStrategy
        using Random

        # Workaround as losses fail with Flux.OneHotMatrix on Appveyor x86 (works everywhere else)
        onehot(y) = Float32.(Flux.onehotbatch(y, 0:5))

        rng = MersenneTwister(123)
        x = randn(rng, Float32, 32,32,2,4)
        y = onehot(rand(rng, 0:5,4))

        c = ImageClassifier(popsize = 3, seed=12345)
        f = TrainSplitAccuracy(nexamples=1, batchsize=1)
        t = TrainStrategy(nepochs=1, batchsize=1, nbatches_per_gen=1)

        dummydir = joinpath(NaiveGAflux.modeldir, "ImageClassifier_smoketest")

        pop = @test_logs (:info, "Begin generation 1") (:info, "Begin generation 2") (:info, "Begin generation 3") (:info, r"Mutate model") match_mode=:any fit(c, x, y, fitnesstrategy=f, trainstrategy=t, mdir = dummydir)

        @test length(pop) == c.popsize

    end
end
