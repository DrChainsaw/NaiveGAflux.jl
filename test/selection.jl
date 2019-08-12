
@testset "Selection" begin

    #In order to be easily portable to NaiveNASlib
    iv(size, name="in") = inputvertex(name, size, FluxDense())

    av(in, outsize, name) = mutable(name, Dense(nout(in), outsize), in)

    cc(ins...; name) = concat(ins...; traitdecoration=named(name))
    nc(name) = traitconf(named(name))

    select_outputs_and_change(v, values) = select_outputs_and_change(NaiveGAflux.NoutExact(), v, values)
    function select_outputs_and_change(s, v, values)
        execute, selected = NaiveGAflux.select_outputs(s, v, values)
        if execute
            Δnout(v, selected)
        end
    end

    @testset "Absorb 2 Absorb" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = av(v1, 4, "v2")

        g = CompGraph(inpt, v2)

        Δnout(v1, -2)
        select_outputs_and_change(v1, 1:nout_org(op(v1)))

        @test out_inds(op(v1)) == in_inds(op(v2))[] == [3,4,5]
        apply_mutation(g)

        @test size(g(ones(3, 1))) == (nout(v2), 1)

        Δnout(v1, 3)
        select_outputs_and_change(v1, 1:nout_org(op(v1)))

        @test out_inds(op(v1)) == in_inds(op(v2))[] == [1,2,3,-1,-1,-1]
        apply_mutation(g)

        @test size(g(ones(3, 1))) == (nout(v2), 1)
    end

    @testset "Absorb 2 Absorb revert" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = av(v1, 4, "v2")

        g = CompGraph(inpt, v2)

        Δnout(v1, -2)
        select_outputs_and_change(NaiveGAflux.NoutRevert(), v1, 1:nout_org(op(v1)))

        @test out_inds(op(v1)) == in_inds(op(v2))[] == [1,2,3,4,5]
        apply_mutation(g)

        @test size(g(ones(3, 1))) == (nout(v2), 1)

        Δnout(v1, +3)

        select_outputs_and_change(NaiveGAflux.NoutRevert(), v1, 1:nout_org(op(v1)))

        @test out_inds(op(v1)) == in_inds(op(v2))[] == [1,2,3,4,5]
        apply_mutation(g)

        @test size(g(ones(3, 1))) == (nout(v2), 1)
    end

    @testset "Absorb 2 Absorb fail" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = av(v1, 4, "v2")

        Δnout(v1, -2)
        @test_throws ErrorException select_outputs_and_change(NaiveGAflux.SelectionFail(), v1, 1:nout_org(op(v1)))
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

        select_outputs_and_change(v4, 1:nout_org(op(v4)))
        apply_mutation(g)

        @test nout(v1) == 5
        @test nout(v2) == 3

        @test size(g(ones(3,1))) == (nout(v4), 1)

        Δnout(v4, 6)

        @test nout(v1) == 9
        @test nout(v2) == 4

        select_outputs_and_change(v4, 1:nout_org(op(v4)))
        apply_mutation(g)

        @test nout(v1) == 9
        @test nout(v2) == 4

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

        select_outputs_and_change(v6, 1:nout_org(op(v6)))
        apply_mutation(g)

        @test nout(v1) == 4
        @test nout(v2) == 3
        @test nout(v3) == 1

        @test size(g(ones(3,1))) == (nout(v6), 1)

        Δnout(v6, 6)

        @test nout(v1) == 8
        @test nout(v2) == 5
        @test nout(v3) == 3

        select_outputs_and_change(v6, 1:nout_org(op(v6)))
        apply_mutation(g)

        @test nout(v1) == 8
        @test nout(v2) == 5
        @test nout(v3) == 3

        @test size(g(ones(3,1))) == (nout(v6), 1)
    end

    @testset "SizeStack one immutable" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = cc(inpt, v1, name="v2")

        g = CompGraph(inpt, v2)
        @test size(g(ones(3,1))) == (nout(v2), 1)

        Δnout(v1, -3)

        @test nin(v2) == [nout(inpt), nout(v1)] == [3, 2]
        @test nout(v2) == 5

        # "Tempt" optimizer to not select inputs from inpt
        select_outputs_and_change(NaiveGAflux.NoutRelaxSize(0.5, 1), v2, -nout(inpt):nout_org(op(v1))-1)
        apply_mutation(g)

        @test nin(v2) == [nout(inpt), nout(v1)] == [3, 2]
        @test nout(v2) == 5

        @test size(g(ones(3,1))) == (nout(v2), 1)
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

        @test_logs (:warn, "Selection for vertex v7 failed! Relaxing size constraint...")  match_mode=:any select_outputs_and_change(v7, 1:nout_org(op(v7)))
        apply_mutation(g)

        @test nout(v1) == 5
        @test nout(v2) == 3
        @test nout(v3) == 6
        @test nout(v4) == 3

        @test size(g(ones(3,1))) == (nout(v7), 1)

        Δnout(v7, 14)

        @test nout(v1) == 10
        @test nout(v2) == 6
        @test nout(v3) == 12
        @test nout(v4) == 6

        # Works on the first try this time around
        select_outputs_and_change(v7, 1:nout_org(op(v7)))
        apply_mutation(g)

        @test nout(v1) == 10
        @test nout(v2) == 6
        @test nout(v3) == 12
        @test nout(v4) == 6

        @test size(g(ones(3,1))) == (nout(v7), 1)
    end

    @testset "SizeInvariant increase exact infeasible" begin
        inpt = iv(3)
        v1 = av(inpt, 3, "v1")
        v2 = av(inpt, 4, "v2")
        v3 = av(inpt, 4, "v3")
        v4 = av(inpt, 5, "v4")

        v5 = cc(v1, v2, v4, name="v5")
        v6 = cc(v3, v4, v1, name="v6")

        v7 = nc("v7") >> v5 + v6

        g = CompGraph(inpt, v7)
        @test size(g(ones(3,1))) == (nout(v7), 1)

        @test minΔnoutfactor(v7) == 1
        Δnout(v7, 5)

        @test nout(v1) == 4
        @test nout(v2) == 5
        @test nout(v3) == 5
        @test nout(v4) == 8

        @test_logs (:warn, "Selection for vertex v7 failed! Relaxing size constraint...")  match_mode=:any select_outputs_and_change(v7, 1:nout_org(op(v7)))
        apply_mutation(g)

        # Sizes can't change when increasing, even if problem is relaxed :(
        @test nout(v1) == 4
        @test nout(v2) == 5
        @test nout(v3) == 5
        @test nout(v4) == 8

        @test size(g(ones(3,1))) == (nout(v7), 1)
    end
end
