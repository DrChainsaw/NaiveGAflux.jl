@testset "AutoFit" begin

    @testset "ImageClassifier smoketest" begin
        using NaiveGAflux.AutoFit
        using Random

        # Workaround as losses fail with Flux.OneHotMatrix on Appveyor x86 (works everywhere else)
        onehot(y) = Float32.(Flux.onehotbatch(y, 0:5))

        rng = MersenneTwister(123)
        x = randn(rng, Float32, 32,32,3,10)
        y = onehot(rand(rng, 0:5,10))

        c = ImageClassifier(popsize = 3, batchsize = 2, nepochs = 3, seed=12345)

        @test_logs (:info, "Begin generation 1") (:info, "Begin generation 2") (:info, "Begin generation 3") match_mode=:any fit(c, x, y, mdir = joinpath(NaiveGAflux.modeldir, "ImageClassifier_smoketest"))

    end
end
