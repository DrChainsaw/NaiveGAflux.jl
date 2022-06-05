@testset "Optimizer crossover" begin
    using NaiveGAflux.Flux.Optimise

    prts(o) = typeof(o)
    prts(o::Optimiser) = "$(typeof(o))$(prts.(Tuple(o.os)))"

    @testset "Swap optimizers $(prts(o1)) and $(prts(o2))" for (o1, o2) in (
        (ADAM(), Momentum()),
        (Optimiser(Descent(), WeightDecay()), Optimiser(Momentum(), Nesterov())),
        )
        oc = OptimizerCrossover()
        ooc = OptimizerCrossover(oc)
        @test prts.(oc((o1,o2))) == prts.(ooc((o1,o2))) == prts.((o2, o1))
        @test prts.(oc((o2,o1))) == prts.(ooc((o2,o1))) == prts.((o1, o2))
    end

    @testset "Don't swap shielded" begin
        o1 = ShieldedOpt(Descent())
        o2 = ShieldedOpt(Momentum())
        @test OptimizerCrossover()((o1,o2)) == (o1,o2)
    end

    @testset "Cardinality difference" begin

        @testset "Single opt vs Optimiser" begin
            oc = OptimizerCrossover()
            @test prts.(oc((Descent(), Optimiser(Momentum(), WeightDecay())))) == prts.((Momentum(), Optimiser(Descent(), WeightDecay())))
        end

        @testset "Different size Optimisers" begin
            oc = OptimizerCrossover()
            o1 = Optimiser(Descent(), WeightDecay(), Momentum())
            o2 = Optimiser(ADAM(), ADAMW(), NADAM(), RADAM())

            o1n,o2n = oc((o1,o2))

            @test prts(o1n) == prts(Optimiser(ADAM(), ADAMW(), NADAM()))
            @test prts(o2n) == prts(Optimiser(Descent(), WeightDecay(), Momentum(), RADAM()))
        end
    end

    @testset "LogMutation and MutationProbability" begin
        mplm(c) = MutationProbability(LogMutation(((o1,o2)::Tuple) -> "Crossover between $(prts(o1)) and $(prts(o2))", c), Probability(0.2, MockRng([0.3, 0.1, 0.3])))
        oc = OptimizerCrossover() |> mplm |> OptimizerCrossover

        o1 = Optimiser(Descent(), WeightDecay(), Momentum())
        o2 = Optimiser(ADAM(), ADAGrad(), AdaMax())

        o1n,o2n = @test_logs (:info, "Crossover between WeightDecay and ADAGrad") oc((o1,o2))

        @test typeof.(o1n.os) == [Descent, ADAGrad, Momentum]
        @test typeof.(o2n.os) == [ADAM, WeightDecay, AdaMax]
    end

    @testset "Learningrate crossover" begin
        import NaiveGAflux: learningrate
        @testset "Single opt" begin
            oc = LearningRateCrossover()
            o1,o2 = oc((Descent(0.1), Momentum(0.2)))

            @test typeof(o1) == Descent
            @test o1.eta == 0.2

            @test typeof(o2) == Momentum
            @test o2.eta == 0.1
        end

        @testset "Shielded opt" begin
            oc = LearningRateCrossover()
            o1,o2 = oc((ShieldedOpt(Descent(0.1)), Momentum(0.2)))

            @test typeof(o1) == ShieldedOpt{Descent}
            @test o1.opt.eta == 0.1

            @test typeof(o2) == Momentum
            @test o2.eta == 0.2
        end

        @testset "Optimiser" begin
            oc = LearningRateCrossover()
            o1 = Optimiser(Descent(0.1), Momentum(0.2), WeightDecay(0.1))
            o2 = Optimiser(ADAM(0.3), RADAM(0.4), NADAM(0.5), Nesterov(0.6))

            o1n,o2n = oc((o1,o2))

            @test prts(o1n) == prts(o1)
            @test prts(o2n) == prts(o2)

            @test learningrate.(o1n.os[1:end-1]) == [0.3, 0.4]
            @test learningrate.(o2n.os) == [0.1, 0.2, 0.5, 0.6]

        end
    end
end