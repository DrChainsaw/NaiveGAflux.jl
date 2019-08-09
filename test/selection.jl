
@testset "Selection" begin

    #In order to be easily portable to NaiveNASlib
    iv(size, name="in") = inputvertex(name, size, FluxDense())

    av(in, outsize, name) = mutable(name, Dense(nout(in), outsize), in, traitfun = NaiveGAflux.default_logging())


    cc(ins...; name) = concat(ins...; traitdecoration=named(name) ∘ NaiveGAflux.default_logging())
    nc(name) = traitconf(named(name))

    @testset "Absorb 2 Absorb" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = av(v1, 4, "v2")

        Δnout(v1, -2)
        NaiveGAflux.select_outputs(v1, 1:nout_org(op(v1)))

        @test out_inds(op(v1)) == in_inds(op(v2))[] == [3,4,5]

        g = CompGraph(inpt, v2)
        apply_mutation(g)

        @test size(g(ones(3, 1))) == (nout(v2), 1)
    end

    @testset "Absorb 2 Absorb revert" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = av(v1, 4, "v2")

        Δnout(v1, -2)
        NaiveGAflux.select_outputs(NaiveGAflux.NoutRevert(), v1, 1:nout_org(op(v1)))

        @test out_inds(op(v1)) == in_inds(op(v2))[] == [1,2,3,4,5]
        g = CompGraph(inpt, v2)
        apply_mutation(g)

        @test size(g(ones(3, 1))) == (nout(v2), 1)
    end

    @testset "Absorb 2 Absorb fail" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = av(v1, 4, "v2")

        Δnout(v1, -2)
        @test_throws ErrorException NaiveGAflux.select_outputs(NaiveGAflux.SelectionFail(), v1, 1:nout_org(op(v1)))
    end

    @testset "SizeStack duplicate" begin
        inpt = iv(3)
        v1 = av(inpt, 7, "v1")
        v2 = av(inpt, 4, "v2")
        v3 = cc(v2, v1, name="v3")
        v4 = cc(v3, v2, name="v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(3,1))) == (nout(v4), 1)

        @test minΔnoutfactor(v4) == 2
        Δnout(v4, -4)

        @test nout(v1) == 5
        @test nout(v2) == 3

        NaiveGAflux.select_outputs(v4, 1:nout_org(op(v4)))
        apply_mutation(g)

        @test nout(v1) == 5
        @test nout(v2) == 3

        @test size(g(ones(3,1))) == (nout(v4), 1)
    end

    @testset "SizeInvariant duplicate" begin
        inpt = iv(3)
        v1 = av(inpt, 7, "v1")
        v2 = av(inpt, 4, "v2")
        v3 = av(inpt, 3, "v3")

        v4 = cc(v1, v2, name="v4")
        v5 = cc(v2, v3, v2, name="v5")

        v6 = nc("v6") >> v4 + v5

        g = CompGraph(inpt, v6)
        @test size(g(ones(3,1))) == (nout(v6), 1)

        @test minΔnoutfactor(v6) == 2
        Δnout(v6, -4)

        @test nout(v1) == 4
        @test nout(v2) == 3
        @test nout(v3) == 1

        NaiveGAflux.select_outputs(v6, 1:nout_org(op(v6)))
        apply_mutation(g)

        @test nout(v1) == 4
        @test nout(v2) == 3
        @test nout(v3) == 1

        @test size(g(ones(3,1))) == (nout(v6), 1)
    end

    @testset "SizeInvariant exact infeasible" begin
        inpt = iv(3)
        v1 = av(inpt, 10, "v1")
        v2 = av(inpt, 6, "v2")
        v3 = av(inpt, 10, "v3")
        v4 = av(inpt, 4, "v4")

        v5 = cc(v1, v2, v3, name="v5")
        v6 = cc(v2, v1, v2, v4, name="v6")

        v7 = nc("v7") >> v5 + v6

        g = CompGraph(inpt, v7)
        @test size(g(ones(3,1))) == (nout(v7), 1)

        @test minΔnoutfactor(v7) == 2
        Δnout(v7, -4)

        @test nout(v1) == 9
        @test nout(v2) == 5
        @test nout(v3) == 8
        @test nout(v4) == 3

        @test_logs (:warn, "Selection for vertex v7 failed! Relaxing size constraint...")  match_mode=:any NaiveGAflux.select_outputs(v7, 1:nout_org(op(v7)))
        apply_mutation(g)

        @test nout(v1) == 5
        @test nout(v2) == 3
        @test nout(v3) == 6
        @test nout(v4) == 3

        @test size(g(ones(3,1))) == (nout(v7), 1)
    end


end
