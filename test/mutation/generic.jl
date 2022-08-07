

@testset "Generic mutation" begin

    struct NoOpMutation{T} <:AbstractMutation{T} end
    (m::NoOpMutation{T})(t::T) where T = t
    ProbeMutation(T) = RecordMutation(NoOpMutation{T}())

    @testset "MutationProbability" begin
        probe = ProbeMutation(Int)
        m = MutationProbability(probe, Probability(0.3, MockRng([0.2,0.5,0.1])))

        @test m(1) == 1
        @test m(2) == 2
        @test m(3) == 3
        @test m(4) == 4
        @test probe.mutated == [1,3,4]
    end

    @testset "MutationProbability vector" begin
        probe = ProbeMutation(Int)
        m = MutationProbability(probe, Probability(0.3, MockRng([0.2,0.5,0.1])))

        @test m(1:4) == 1:4
        @test probe.mutated == [1,3,4]
    end

    @testset "WeightedMutationProbability" begin
        probe = ProbeMutation(Real)
        rng = MockRng([0.5])
        m = WeightedMutationProbability(probe, p -> Probability(p, rng))

        @test m(0.1) == 0.1
        @test m(0.6) == 0.6
        @test m(0.4) == 0.4
        @test m(0.9) == 0.9
        @test probe.mutated == [0.6,0.9]
    end

    @testset "WeightedMutationProbability vector" begin
        probe = ProbeMutation(Real)
        rng = MockRng([0.5])
        m = WeightedMutationProbability(probe, p -> Probability(p, rng))

        @test m([0.1, 0.6, 0.4, 0.9]) == [0.1, 0.6, 0.4, 0.9]
        @test probe.mutated == [0.6,0.9]
    end

    @testset "Neuron utlity weighted mutation" begin
        using Statistics
        import NaiveNASflux: AbstractMutableComp, neuronutility, wrapped
        struct DummyValue{T, W<:AbstractMutableComp} <: AbstractMutableComp
            utlity::T
            w::W
        end
        NaiveNASflux.neuronutility(d::DummyValue) = d.utlity
        NaiveNASflux.wrapped(d::DummyValue) = d.w

        l(in, outsize, utlity) = fluxvertex(Dense(nout(in), outsize), in, layerfun = l -> DummyValue(utlity, l))

        v0 = inputvertex("in", 3)
        v1 = l(v0, 4, 1:4)
        v2 = l(v1, 3, 100:300)
        v3 = l(v2, 5, 0.1:0.1:0.5)

        @testset "weighted_neuronutility_high pbase $pbase" for pbase in (0.05, 0.1, 0.3, 0.7, 0.9, 0.95)
            import NaiveGAflux: weighted_neuronutility_high
            wnv = weighted_neuronutility_high(pbase, spread=0.5)
            wp = map(p -> p.p, wnv.([v1,v2,v3]))
            @test wp[2] > wp[1] > wp[3]
            @test mean(wp) ≈ pbase rtol = 0.1
        end

        @testset "HighUtilityMutationProbability" begin

            probe = ProbeMutation(MutationVertex)
            m = HighUtilityMutationProbability(probe, 0.1, MockRng([0.15]))

            m(v1)
            m(v2)
            m(v3)
            @test probe.mutated == [v2]
        end

        @testset "weighted_neuronutility_low pbase $pbase" for pbase in (0.05, 0.1, 0.3, 0.7, 0.9, 0.95)
            import NaiveGAflux: weighted_neuronutility_low
            wnv = weighted_neuronutility_low(pbase,spread=0.8)
            wp = map(p -> p.p, wnv.([v1,v2,v3]))
            @test wp[2] < wp[1] < wp[3]
            @test mean(wp) ≈ pbase rtol = 0.1
        end

        @testset "LowUtilityMutationProbability" begin
            probe = ProbeMutation(MutationVertex)
            m = LowUtilityMutationProbability(probe, 0.1, MockRng([0.15]))

            m(v1)
            m(v2)
            m(v3)
            @test probe.mutated == [v1, v3]
        end
    end

    @testset "MutationChain" begin
        probes = ProbeMutation.(repeat([Int], 3))
        m = MutationChain(probes...)
        @test m(1) == 1
        @test getfield.(probes, :mutated) == [[1],[1],[1]]
    end

    @testset "MutationChain vector" begin
        probes = ProbeMutation.(repeat([Int], 3))
        m = MutationChain(probes...)
        @test m(1:2) == 1:2
        @test getfield.(probes, :mutated) == [[1,2],[1,2],[1,2]]
    end

    @testset "LogMutation" begin
        probe = ProbeMutation(Int)
        m = LogMutation(i -> "Mutate $i", probe)

        @test @test_logs (:info, "Mutate 17") m(17) == 17
        @test probe.mutated == [17]
    end

    @testset "LogMutation vector" begin
        probe = ProbeMutation(Int)
        m = LogMutation(i -> "Mutate $i", probe)

        @test @test_logs (:info, "Mutate 17") (:info, "Mutate 21") m([17, 21]) == [17, 21]
        @test probe.mutated == [17, 21]
    end

    @testset "MutationFilter" begin
        probe = ProbeMutation(Int)
        m = MutationFilter(i -> i > 3, probe)

        @test m(1) == 1
        @test probe.mutated == []

        @test m(4) == 4
        @test probe.mutated == [4]
    end

    @testset "MutationFilter vector" begin
        probe = ProbeMutation(Int)
        m = MutationFilter(i -> i > 3, probe)

        @test m(1:5) == 1:5
        @test probe.mutated == [4,5]
    end
end
