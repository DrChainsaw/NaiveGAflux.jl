@testset "Optimiser mutation" begin
    import NaiveGAflux: setlearningrate, learningrate, ImplicitOpt
    import Optimisers
    import Optimisers: OptimiserChain, Descent, Momentum, Nesterov, Adam

    @testset "Mutate learning rate" begin
        m = OptimiserMutation(o -> setlearningrate(o, 10 * learningrate(o)))

        @test learningrate(m(Descent(0.1))) == 1.0
        @test learningrate(m(ShieldedOpt(Momentum(0.1)))) == 0.1
        @test learningrate(m(OptimiserChain(Nesterov(0.1), ShieldedOpt(Adam(0.1))))) == 0.1

        @test learningrate(LearningRateMutation(MockRng([0.0]))(Descent(0.1))) == 0.085
    end

    @testset "Mutate optimiser type" begin
        m = OptimiserMutation((Momentum, ))

        @test typeof(m(Descent())) == Momentum{Float32}
        @test typeof(m(ShieldedOpt(Descent()))) == ShieldedOpt{Descent{Float32}}
        @test typeof(m(ImplicitOpt(Nesterov()))) == ImplicitOpt{Momentum{Float32}}
        @test typeof(m(OptimiserChain(Nesterov(), ShieldedOpt(Adam())))) == OptimiserChain{Tuple{Momentum{Float32}, ShieldedOpt{Adam{Float32}}}}
        @test typeof(m(ImplicitOpt(OptimiserChain(Nesterov(), Adam())))) == ImplicitOpt{OptimiserChain{Tuple{Momentum{Float32}, Momentum{Float32}}}}
    end

    @testset "Add optimiser" begin
        m = AddOptimiserMutation(o -> Descent(0.1f0))

        @test typeof(m(Descent(0.2f0))) == OptimiserChain{Tuple{Descent{Float32}}}
        @test typeof(m(Momentum(0.2f0))) == OptimiserChain{Tuple{Momentum{Float32}, Descent{Float32}}}
        @test typeof(m(ShieldedOpt(Descent()))) == ShieldedOpt{Descent{Float32}}
        @test typeof(m(ImplicitOpt(Nesterov()))) == ImplicitOpt{OptimiserChain{Tuple{Nesterov{Float32}, Descent{Float32}}}}
        @test typeof(m(OptimiserChain(Nesterov(), Descent(), ShieldedOpt(Descent())))) == OptimiserChain{Tuple{Nesterov{Float32}, ShieldedOpt{Descent{Float32}}, Descent{Float32}}}
        @test typeof(m(ImplicitOpt(OptimiserChain(Nesterov(), Descent(), ShieldedOpt(Descent()))))) == ImplicitOpt{OptimiserChain{Tuple{Nesterov{Float32}, ShieldedOpt{Descent{Float32}}, Descent{Float32}}}}
    end

    @testset "MutationChain and LogMutation" begin
        m = MutationChain(LogMutation(o -> "First", OptimiserMutation((Momentum, ))), LogMutation(o -> "Second", AddOptimiserMutation(o -> Descent())))

        @test_logs (:info, "First") (:info, "Second") typeof.(m(Nesterov()).opts) == (Momentum{Float32}, Descent{Float32})
        @test_logs (:info, "First") (:info, "First") (:info, "Second") (:info, "Second") m([Nesterov(), Adam()])
    end
end