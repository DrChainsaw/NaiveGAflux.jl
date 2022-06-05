@testset "Optimizer mutation" begin
    import NaiveGAflux: sameopt, learningrate
    import NaiveGAflux.Flux.Optimise: Optimiser

    @testset "Mutate learning rate" begin
        m = OptimizerMutation(o -> sameopt(o, 10 * learningrate(o)))

        @test learningrate(m(Descent(0.1))) == 1.0
        @test learningrate(m(ShieldedOpt(Momentum(0.1)))) == 0.1
        @test learningrate(m(Optimiser(Nesterov(0.1), ShieldedOpt(ADAM(0.1))))) == 0.1

        @test learningrate(LearningRateMutation(MockRng([0.0]))(Descent(0.1))) == 0.085
    end

    @testset "Mutate optimizer type" begin
        m = OptimizerMutation((Momentum, ))

        @test typeof(m(Descent())) == Momentum
        @test typeof(m(ShieldedOpt(Descent()))) == ShieldedOpt{Descent}
        @test typeof.(m(Optimiser(Nesterov(), ShieldedOpt(ADAM()))).os) == [Momentum, ShieldedOpt{ADAM}]
    end

    @testset "Add optimizer" begin
        m = AddOptimizerMutation(o -> Descent(0.1))

        @test typeof.(m(Descent(0.2)).os) == [Descent]
        @test typeof.(m(Momentum(0.2)).os) == [Momentum, Descent]
        @test typeof.(m(Flux.Optimiser(Nesterov(), Descent(), ShieldedOpt(Descent()))).os) == [Nesterov, ShieldedOpt{Descent}, Descent]
    end

    @testset "MutationChain and LogMutation" begin
        m = MutationChain(LogMutation(o -> "First", OptimizerMutation((Momentum, ))), LogMutation(o -> "Second", AddOptimizerMutation(o -> Descent())))

        @test_logs (:info, "First") (:info, "Second") typeof.(m(Nesterov()).os) == [Momentum, Descent]
        @test_logs (:info, "First") (:info, "First") (:info, "Second") (:info, "Second") m([Nesterov(), ADAM()])
    end
end