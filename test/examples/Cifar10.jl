
@testset "CIFAR10 Smoketest" begin
    using NaiveGAflux.Cifar10

    struct DummyData end
    (::DummyData)() = randn(Float32, 32,32,3,2), rand(0:9,2)

    @test_logs (:info, "Begin iteration 1") (:info, "Begin iteration 2") (:info, "Begin iteration 3") match_mode=:any run_experiment(2, 3, DummyData(), nevolve=1, baseseed=12345)
end
