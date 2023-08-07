@testset "Optimiser crossover" begin
    import NaiveGAflux: ImplicitOpt
    import Optimisers
    import Optimisers: OptimiserChain, Descent, Momentum, Nesterov, Adam, WeightDecay

    prts(o) = typeof(o)

    @testset "Swap optimisers $(prts(o1)) and $(prts(o2))" for (o1, o2) in (
        (Adam(), Momentum()),
        (OptimiserChain(Descent(), WeightDecay()), OptimiserChain(Momentum(), Nesterov())),
        (ImplicitOpt(OptimiserChain(Descent(), WeightDecay())), ImplicitOpt(OptimiserChain(Momentum(), Nesterov()))),
        )
        oc = OptimiserCrossover()
        ooc = OptimiserCrossover(oc)
        @test prts.(oc((o1,o2))) == prts.(ooc((o1,o2))) == prts.((o2, o1))
        @test prts.(oc((o2,o1))) == prts.(ooc((o2,o1))) == prts.((o1, o2))
    end

    @testset "ShieldedOpt" begin

        oc = OptimiserCrossover()
        
        @testset "$baseo1 and $baseo2" for (baseo1, baseo2) in (
            (Descent(), Momentum()),
            (Descent(),OptimiserChain(Momentum())),
            (OptimiserChain(Descent()),OptimiserChain(Momentum()))
        )
            @testset "With Shielding $w1 and $w2" for (w1, w2) in (
                (identity, ShieldedOpt),
                (ShieldedOpt, identity)
            )
                o1 = w1(baseo1)
                o2 = w2(baseo2)

                @test oc((o1, o2)) == (o1, o2)
                @test oc((o2, o1)) == (o2, o1)
            end
        end

        @testset "Inner shielding$(wrap == identity ? "" : wrap)" for wrap in (identity, OptimiserChain)
            o1 = OptimiserChain(ShieldedOpt(Descent()))
            o2 = wrap(Momentum())

            @test prts.(oc((o1, o2))) == prts.((o1, o2))
            @test prts.(oc((o2, o1))) == prts.((o2, o1))
        end
    end

    @testset "Cardinality difference" begin
        import Optimisers
        import Optimisers: Momentum, WeightDecay, OptimiserChain, Descent, AdamW, NAdam, RAdam

        @testset "Single opt vs Optimiser" begin
            oc = OptimiserCrossover()
            @test prts.(oc((Descent(), OptimiserChain(Momentum(), WeightDecay())))) == prts.((Momentum(), OptimiserChain(Descent(), WeightDecay())))
        end

        @testset "Different size Optimisers" begin
            oc = OptimiserCrossover()
            o1 = OptimiserChain(Descent(), WeightDecay(), Momentum())
            o2 = OptimiserChain(Adam(), AdamW(), NAdam(), RAdam())

            o1n,o2n = oc((o1,o2))

            @test prts(o1n) == prts(OptimiserChain(Adam(), AdamW(), NAdam()))
            @test prts(o2n) == prts(OptimiserChain(Descent(), WeightDecay(), Momentum(), RAdam()))
        end
    end

    @testset "LogMutation and MutationProbability" begin
        import Optimisers
        import Optimisers: Descent, WeightDecay, Momentum, Adam, AdaGrad, AdaMax

        mplm(c) = MutationProbability(LogMutation(((o1,o2)::Tuple) -> "Crossover between $(prts(o1)) and $(prts(o2))", c), Probability(0.2, MockRng([0.3, 0.1, 0.3])))
        oc = OptimiserCrossover() |> mplm |> OptimiserCrossover

        o1 = OptimiserChain(Descent(), WeightDecay(), Momentum())
        o2 = OptimiserChain(Adam(), AdaGrad(), AdaMax())

        o1n,o2n = @test_logs (:info, "Crossover between WeightDecay{Float32} and AdaGrad{Float32}") oc((o1,o2))

        @test typeof.(o1n.opts) == (Descent{Float32}, AdaGrad{Float32}, Momentum{Float32})
        @test typeof.(o2n.opts) == (Adam{Float32}, WeightDecay{Float32}, AdaMax{Float32})
    end

    @testset "Learningrate crossover" begin
        import NaiveGAflux: learningrate
        @testset "Single opt" begin
            oc = LearningRateCrossover()
            o1,o2 = oc((Descent(0.1f0), Momentum(0.2f0)))

            @test typeof(o1) == Descent{Float32}
            @test o1.eta == 0.2f0

            @test typeof(o2) == Momentum{Float32}
            @test o2.eta == 0.1f0
        end

        @testset "Shielded opt" begin
            oc = LearningRateCrossover()
            o1,o2 = oc((ShieldedOpt(Descent(0.1f0)), Momentum(0.2f0)))

            @test typeof(o1) == ShieldedOpt{Descent{Float32}}
            @test o1.rule.eta == 0.1f0

            @test typeof(o2) == Momentum{Float32}
            @test o2.eta == 0.2f0
        end

        @testset "OptimiserChain" begin
            oc = LearningRateCrossover()
            o1 = OptimiserChain(Descent(0.1f0), Momentum(0.2f0), WeightDecay(0.1f0))
            o2 = OptimiserChain(Adam(0.3f0), RAdam(0.4f0), NAdam(0.5f0), Nesterov(0.6f0))

            o1n,o2n = oc((o1,o2))

            @test prts(o1n) == prts(o1)
            @test prts(o2n) == prts(o2)

            @test learningrate.(o1n.opts[1:end-1]) == (0.3f0, 0.4f0)
            @test learningrate.(o2n.opts) == (0.1f0, 0.2f0, 0.5f0, 0.6f0)

        end
    end
end