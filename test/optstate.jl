@testset "Optimizer state mutation" begin
    import NaiveGAflux: StateAlign, Stateless, Stateful, withopt, optstate,opttype

    # SOTA on CIFAR10 :) 
    MultiOpt() = Flux.Optimise.Optimiser(ADAM(), ExpDecay(), Descent(), Nesterov())

    allopts = (Descent, Momentum, Nesterov, ADAM, RMSProp, RADAM, AdaMax, ADAGrad, ADADelta, AMSGrad, NADAM, ADAMW, InvDecay, ExpDecay, WeightDecay, MultiOpt)

    function test_opt_mutation(g, v, opt, x = randn(Float32, nout(g.inputs[]), 2))
        loss(x,y) = Flux.mse(g(x), y)
        Flux.train!(loss, params(g), [(x, g(x) .* 0.7)], opt)
        test_keys(opt, g)

        Δnout(v, -2)
        apply_mutation(g)
        test_keys(opt, g)

        Flux.train!(loss, params(g), [(x, g(x) .* 0.7)], opt)
        test_keys(opt, g)
    end

    test_keys(opt::Flux.Optimise.Optimiser, g) = test_keys.(opt.os, [g])
    test_keys(opt, g) = test_keys(opttype(opt), opt, g)
    test_keys(::Stateless, opt, g) = @test all(isempty, optstate(opt))
    function test_keys(::Stateful, opt, g)
        pars = Set(params(g).order)
        for state in optstate(opt)
            @test keys(state) == pars
        end
    end

    dense(name, vin, outsize, lf) =  mutable(name, Dense(nout(vin), outsize), vin; layerfun=lf)

    @testset "StateAlign only with optimizer $optfun" for optfun in allopts
        opt = optfun()
        v0 = inputvertex("in", 3, FluxDense())
        v1 = dense("v1", v0, 5, withopt(opt))
        v2 = dense("v2", v1, 3, withopt(opt))

        g = CompGraph(v0, v2)
        test_opt_mutation(g, v2, opt)
    end
end
