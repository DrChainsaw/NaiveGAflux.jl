
# Temp test to debug appveyor issue
@testset "crossentropytest" begin
    @test Flux.crossentropy(Float32[0.1 0.2; 0.3 0.4; 0.5 0.6], Float32[0 1; 1 0; 0 0]) ≈ 1.4067054 rtol = 1e-5
    @test Flux.crossentropy(Float32[0.1 0.2; 0.3 0.4; 0.5 0.6], Flux.onehotbatch([2,1], 1:3)) ≈ 1.4067054 rtol = 1e-5
    @test Flux.logitcrossentropy(Float32[0.1 0.2; 0.3 0.4; 0.5 0.6], Float32[0 1; 1 0; 0 0]) ≈ 1.2119015 rtol = 1e-5
    @test Flux.logitcrossentropy(Float32[0.1 0.2; 0.3 0.4; 0.5 0.6], Flux.onehotbatch([2,1], 1:3)) ≈ 1.2119015 rtol = 1e-5
end


@testset "Smoketest" begin
    using NaiveGAflux.Cifar10

    Random.seed!(NaiveGAflux.rng_default, 12345)

    struct DummyData end
    (::DummyData)() = randn(Float32, 32,32,3,2), rand(0:9,2)

    @test_logs (:info, "Begin iteration 1") (:info, "Begin iteration 2") (:info, "Begin iteration 3") run_experiment(2, 3, DummyData())
end
