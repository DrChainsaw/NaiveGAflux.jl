@testset "Optimizer state mutation" begin
    import NaiveGAflux: StateAlign

    #allopts = (Descent, Momentum, Nesterov, ADAM, RMSProp, RADAM, AdaMax, ADAGrad, ADADelta, AMSGrad, NADAM, ADAMW, InvDecay, ExpDecay, WeightDecay)
    allopts = (ADAM,)

    function test_opt_mutation(g, v, opt, x = randn(Float32, nout(g.inputs[]), 2))
        loss(x,y) = Flux.mse(g(x), y)
        Flux.train!(loss, params(g), [(x, g(x) .* 0.7)], opt)
        @test keys(optstate(opt)) == Set(params(g).order)

        Δnout(v, -2)
        apply_mutation(g)
        @test keys(optstate(opt)) == Set(params(g).order)

        Flux.train!(loss, params(g), [(x, g(x) .* 0.7)], opt)
        @test keys(optstate(opt)) == Set(params(g).order)
    end

    optstate(opt) = opt.state

    dense(name, vin, outsize, lf) =  mutable(name, Dense(nout(vin), outsize), vin; layerfun=lf)

    @testset "StateAlign only with optimizer $optfun" for optfun in allopts
        opt = optfun()
        statealign = StateAlign(optstate(opt))
        v0 = inputvertex("in", 3, FluxDense())
        v1 = dense("v1", v0, 5, statealign)
        v2 = dense("v2", v1, 3, statealign)

        g = CompGraph(v0, v2)
        test_opt_mutation(g, v2, opt)
    end


end
