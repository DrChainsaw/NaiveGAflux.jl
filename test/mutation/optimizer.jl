@testset "Optimizer mutation" begin
    import NaiveGAflux: setlearningrate, learningrate
    import NaiveGAflux.Optimisers: OptimiserChain, Descent, Momentum, Nesterov, Adam

    @testset "Mutate learning rate" begin
        m = OptimizerMutation(o -> setlearningrate(o, 10 * learningrate(o)))

        @test learningrate(m(Descent(0.1))) == 1.0
        @test learningrate(m(ShieldedOpt(Momentum(0.1)))) == 0.1
        @test learningrate(m(OptimiserChain(Nesterov(0.1), ShieldedOpt(Adam(0.1))))) == 0.1

        @test learningrate(LearningRateMutation(MockRng([0.0]))(Descent(0.1))) == 0.085
    end

    @testset "Mutate optimizer type" begin
        m = OptimizerMutation((Momentum, ))

        @test typeof(m(Descent())) == Momentum{Float32}
        @test typeof(m(ShieldedOpt(Descent()))) == ShieldedOpt{Descent{Float32}}
        @test typeof.(m(OptimiserChain(Nesterov(), ShieldedOpt(Adam()))).opts) == (Momentum{Float32}, ShieldedOpt{Adam{Float32}})
    end

    @testset "Add optimizer" begin
        m = AddOptimizerMutation(o -> Descent(0.1f0))

        @test typeof.(m(Descent(0.2f0)).opts) == tuple(Descent{Float32})
        @test typeof.(m(Momentum(0.2f0)).opts) == (Momentum{Float32}, Descent{Float32})
        @test typeof(m(ShieldedOpt(Descent()))) == ShieldedOpt{Descent{Float32}}
        @test typeof.(m(OptimiserChain(Nesterov(), Descent(), ShieldedOpt(Descent()))).opts) == (Nesterov{Float32}, ShieldedOpt{Descent{Float32}}, Descent{Float32})
    end

    @testset "MutationChain and LogMutation" begin
        m = MutationChain(LogMutation(o -> "First", OptimizerMutation((Momentum, ))), LogMutation(o -> "Second", AddOptimizerMutation(o -> Descent())))

        @test_logs (:info, "First") (:info, "Second") typeof.(m(Nesterov()).opts) == (Momentum{Float32}, Descent{Float32})
        @test_logs (:info, "First") (:info, "First") (:info, "Second") (:info, "Second") m([Nesterov(), Adam()])
    end
end