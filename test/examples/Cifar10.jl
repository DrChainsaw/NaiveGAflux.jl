
@testset "CIFAR10 Smoketest" begin
    using NaiveGAflux.Cifar10
    using Random

    # Workaround as losses fail with Flux.OneHotMatrix on Appveyor x86 (works everywhere else)
    onehot(y) = Float32.(Flux.onehotbatch(y, 0:9))

    rng = MersenneTwister(123)
    struct DummyDataIter
        n::Int
    end
    Base.iterate(d::DummyDataIter, s=0) = s==d.n ? nothing : ((randn(rng, Float32, 32,32,3,2), onehot(rand(rng, 0:9,2))), s+1)
    Base.length(d::DummyDataIter) = d.n

    trainiter = RepeatPartitionIterator(GpuIterator(Iterators.cycle(DummyDataIter(1), 3)), 1)
    valiter = GpuIterator(DummyDataIter(1))

    @test_logs (:info, "Begin generation 1") (:info, "Begin generation 2") (:info, "Begin generation 3") match_mode=:any run_experiment(2, trainiter, valiter, nelites = 0, baseseed=12345, mdir = joinpath(NaiveGAflux.modeldir, "Cifar10_smoketest"))
end
