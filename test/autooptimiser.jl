@testset "AutoOptimiser" begin
    import Flux
    import Flux: Dense, gradient
    import Optimisers
    import Optimisers: Descent

    import NaiveGAflux.AutoOptimiserExperimental: optimisersetup!

    @testset "Gradients with layerfun=$layerfun" for (layerfun, isinit) in (
        (AutoOptimiser, false),
        (AutoOptimiser(Descent(0.1)), true),
        (LazyMutable ∘ AutoOptimiser, false),
        (ActivationContribution ∘ LazyMutable ∘ AutoOptimiser, false),
        (LazyMutable ∘ ActivationContribution ∘ AutoOptimiser, false),
        (AutoOptimiser ∘ LazyMutable ∘ ActivationContribution, false),
    ) 
        iv = denseinputvertex("iv", 1)
        v1 = fluxvertex("v1", Flux.Dense(nout(iv) => 1; init=ones), iv; layerfun)             
        g = CompGraph(iv, v1)
        x = ones(Float32, 1, 1)
        
        if !isinit
            @test_throws ArgumentError("AutoOptimiser without optimiser state invoked. Forgot to call optimisersetup!?") gradient(() -> sum(g(x)))
        else
            gradient(() -> sum(g(x))) 
            @test NaiveNASflux.weights(layer(v1)) != ones(1,1)
            # Again with opposite sign to reset change so test below succeeds
            gradient(() -> -sum(g(x))) 
        end

        optimisersetup!(Descent(0.5f0), g)
        gradient(() -> sum(g(x)))

        @test NaiveNASflux.weights(layer(v1)) == fill(0.5, 1, 1)
        @test NaiveNASflux.bias(layer(v1)) == fill(-0.5, 1)
    end

    @testset "optimisersetup!" begin
        iv = denseinputvertex("iv", 1)
        v1 = fluxvertex("v1", Flux.Dense(nout(iv) => 1; init=ones), iv)             
        g = CompGraph(iv, v1)

        @test_logs (:warn, "No implict optimiser found! Call is a noop") optimisersetup!(Descent(), g)
    end

end