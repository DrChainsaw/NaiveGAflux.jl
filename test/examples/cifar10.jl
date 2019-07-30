
@testset "Smoketest" begin
    import NaiveGAflux:run_experiment

    Random.seed!(NaiveGAflux.rng_default, 12345)

    struct DummyData end
    (::DummyData)() = randn(Float32, 32,32,3,2), rand(0:9,2)

    @test_logs (:info, "Begin iteration 1") (:info, "Begin iteration 2") (:info, "Begin iteration 3") run_experiment(2, 3, DummyData())
end
