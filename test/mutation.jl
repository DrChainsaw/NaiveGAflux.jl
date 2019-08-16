

@testset "Mutation" begin

    struct NoOpMutation{T} <:AbstractMutation{T} end
    function (m::NoOpMutation)(t) end
    ProbeMutation(T) = RecordMutation(NoOpMutation{T}())

    @testset "MutationProbability" begin
        probe = ProbeMutation(Int)
        m = MutationProbability(probe, Probability(0.3, MockRng([0.2,0.5,0.1])))

        m(1)
        m(2)
        m(3)
        m(4)
        @test probe.mutated == [1,3,4]
    end

    @testset "MutationList" begin
        probes = ProbeMutation.(repeat([Int], 3))
        m = MutationList(probes...)
        m(1)
        @test getfield.(probes, :mutated) == [[1],[1],[1]]
    end

    @testset "LogMutation" begin
        probe = ProbeMutation(Int)
        m = LogMutation(i -> "Mutate $i", probe)

        @test_logs (:info, "Mutate 17") m(17)
        @test probe.mutated == [17]
    end

    @testset "MutationFilter" begin
        probe = ProbeMutation(Int)
        m = MutationFilter(i -> i > 3, probe)

        m(1)
        @test probe.mutated == []

        m(4)
        @test probe.mutated == [4]
    end

    @testset "PostMutation" begin
        probe = ProbeMutation(Int)

        expect_m = nothing
        expect_e = nothing
        function action(m,e)
            expect_m = m
            expect_e = e
        end

        m = PostMutation(action, probe)
        m(11)

        @test probe.mutated == [11]
        @test expect_m == m
        @test expect_e == 11
    end

    dense(in, outsize;layerfun = LazyMutable, name="dense") = mutable(name, Dense(nout(in), outsize), in, layerfun=layerfun)
    dense(in, outsizes...;layerfun = LazyMutable, name="dense") = foldl((next,i) -> dense(next, outsizes[i], name=join([name, i]), layerfun=layerfun), 1:length(outsizes), init=in)

    @testset "VertexMutation" begin
        inpt = inputvertex("in", 4, FluxDense())
        outpt = dense(inpt, 3,4,5)
        graph = CompGraph(inpt, outpt)

        probe = ProbeMutation(AbstractVertex)
        m = VertexMutation(probe)
        m(graph)
        # Vertex 1 (inpt) is immutable, all others are selected
        @test probe.mutated == vertices(graph)[2:end]
    end

    @testset "NoutMutation" begin
        inpt = inputvertex("in", 3, FluxDense())

        # Can't mutate, don't do anything
        NoutMutation(0.4)(inpt)
        @test nout(inpt) == 3

        # Can't mutate due to too small size
        v = dense(inpt, 1)
        NoutMutation(-0.8, -1.0)(v)
        @test nout(v) == 1

        rng = MockRng([0.5])
        v = dense(inpt, 11)

        NoutMutation(0.4, rng)(v)
        @test nout(v) == 13

        NoutMutation(-0.4, rng)(v)
        @test nout(v) == 11

        NoutMutation(-0.001, rng)(v)
        @test nout(v) == 10

        NoutMutation(0.001, rng)(v)
        @test nout(v) == 11

        NoutMutation(-0.1, 0.3, rng)(v)
        @test nout(v) == 12

        # "Hidden" size 1 vertex
        v0 = dense(inpt,1, name="v0")
        v1 = dense(inpt,1, name="v1")
        v2 = concat(v0, v1, traitfun=named("v2"))

        NoutMutation(-1, rng)(v2)
        @test nout(v2) == 2
        @test nin(v2) == [nout(v0), nout(v1)] == [1, 1]
    end

    @testset "AddVertexMutation" begin
        inpt = inputvertex("in", 3, FluxDense())
        v1 = dense(inpt, 5)

        @test inputs(v1) == [inpt]

        space = ArchSpace(DenseSpace(BaseLayerSpace(4, relu)))

        AddVertexMutation(space)(inpt)

        @test inputs(v1) != [inpt]
        @test nin(v1) == [3]

        v2 = dense(v1, 2)
        v3 = dense(v1, 1)

        AddVertexMutation(space, outs -> view(outs, 2))(v1)

        @test inputs(v2) == [v1]
        @test inputs(v3) != [v1]
    end

    @testset "RemoveVertexMutation" begin
        inpt = inputvertex("in", 3, FluxDense())
        v1 = dense(inpt, 5)
        v2 = dense(v1, 3)

        RemoveVertexMutation()(v1)

        @test inputs(v2) == [inpt]
    end

    @testset "NeuronSelectMutation" begin

        # Dummy neuron selection function just to mix things up in a predicable way
        oddfirst(v, vvals) = true, vcat(sort(vcat(1:2:nout_org(op(v)), 2:2:nout_org(op(v)))[1:min(nout(v),end)]), -ones(Int, max(0, nout(v) - nout_org(op(v)))))
        batchnorm(inpt; name="bn") = mutable(name, BatchNorm(nout(inpt)), inpt)

        @testset "NeuronSelectMutation NoutMutation" begin
            inpt = inputvertex("in", 3, FluxDense())
            v1 = dense(inpt, 5)
            v2 = mutable(BatchNorm(nout(v1)), v1)
            v3 = dense(v2, 6)

            m = NeuronSelectMutation(oddfirst, NoutMutation(-0.5, MockRng([0])))
            m(v2)
            select(m)

            @test out_inds(op(v2)) == [1,3,5]
            @test in_inds(op(v3)) == [[1,3,5]]
            @test out_inds(op(v1)) == [1,3,5]
        end

        @testset "NeuronSelectMutation RemoveVertexMutation" begin
            m = NeuronSelectMutation(oddfirst, RemoveVertexMutation())

            # Increase the input size
            inpt = inputvertex("in", 2, FluxDense())
            v3 = dense(inpt, 7,5,3, layerfun=identity)

            m(inputs(v3)[])
            select(m)
            @test in_inds(op(v3)) == [[1,2,3,4,5,-1,-1]]
            @test out_inds(op(inputs(v3)[])) == 1:7
            apply_mutation(CompGraph(inpt, v3))

            # Increase the output size
            v3 = dense(inpt, 3,5,7, layerfun=identity)

            m(inputs(v3)[])
            select(m)
            @test in_inds(op(v3)) == [[1,2,3,4,5]]
            @test out_inds(op(inputs(v3)[])) == [1,2,3,-1,-1]
            apply_mutation(CompGraph(inpt, v3))

            # Decrease the output size
            m = NeuronSelectMutation(oddfirst, RemoveVertexMutation(RemoveStrategy(DecreaseBigger())))
            v3 = dense(inpt, 7,5,3, layerfun=identity)

            m(inputs(v3)[])
            select(m)
            @test in_inds(op(v3)) == [[1,2,3,4,5]]
            @test out_inds(op(inputs(v3)[])) == [1,2,3,5,7]
            apply_mutation(CompGraph(inpt, v3))

            # Decrease the input size
            v3 = dense(inpt, 3,5,7, layerfun=identity)

            m(inputs(v3)[])
            select(m)
            @test in_inds(op(v3)) == [[1,3,5]]
            @test out_inds(op(inputs(v3)[])) == 1:3
            apply_mutation(CompGraph(inpt, v3))

            # Decrease outsize of input and increase insize of output
            m = NeuronSelectMutation(oddfirst, RemoveVertexMutation(RemoveStrategy(AlignSizeBoth())))
            v3 = dense(inpt, 7,5,3, layerfun=identity)

            m(inputs(v3)[])
            select(m)
            @test in_inds(op(v3)) == [[1,2,3,4,5,-1]]
            @test out_inds(op(inputs(v3)[])) == [1,2,3,4,5,7]
            apply_mutation(CompGraph(inpt, v3))
        end

        @testset "NeuronSelect" begin
            m1 = MutationProbability(NeuronSelectMutation(oddfirst, NoutMutation(0.5, MockRng([1]))), Probability(0.2, MockRng([0.1, 0.3])))
            m2 = MutationProbability(NeuronSelectMutation(oddfirst, NoutMutation(-0.5, MockRng([0]))), Probability(0.2, MockRng([0.3, 0.1])))
            m = VertexMutation(MutationList(m1, m2))

            inpt = inputvertex("in", 2, FluxDense())
            v3 = dense(inpt, 3,5,7)
            g = CompGraph(inpt, v3)

            m(g)
            NeuronSelect()(m, g)
            vs = vertices(g)[2:end]

            @test [out_inds(op(vs[1]))] == in_inds(op(vs[2])) == [[1,2,3,-1]]
            @test [out_inds(op(vs[2]))] == in_inds(op(vs[3])) == [[1,3,5]]
            @test out_inds(op(vs[3])) == [1,2,3,4,5,6,7,-1,-1,-1]
        end

        @testset "NeuronSelectMutation deep transparent" begin

            inpt = inputvertex("in", 3, FluxDense())
            v1 = dense(inpt, 8, name="v1")
            v2 = dense(inpt, 4, name="v2")
            v3 = concat(v1,v2, traitfun=named("v3"))
            pa1 = batchnorm(v3, name="pa1")
            pb1 = batchnorm(v3, name="pb1")
            pc1 = batchnorm(v3, name="pc1")
            pd1 = dense(v3, 5, name="pd1")
            pa1pa1 = batchnorm(pa1, name="pa1pa1")
            pa1pb1 = batchnorm(pa1, name="pa1pb1")
            pa2 = concat(pa1pa1, pa1pb1, traitfun=named("pa2"))
            v4 = concat(pa2, pb1, pc1, pd1, traitfun=named("v4"))

            rankfun(v, vvals) = NaiveGAflux.select_outputs(v, 1:nout_org(op(vvals)))
            m = NeuronSelectMutation(rankfun , NoutMutation(0.5))
            push!(m.m.mutated, v4)

            g = CompGraph(inpt, v4)
            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)

            @test minΔnoutfactor(v4) == 4
            Δnout(v4, -8)
            noutv4 = nout(v4)

            @test nout(v1) == 7
            @test nout(v2) == 4
            @test nout(pd1) == 1

            select(m)
            apply_mutation(g)

            @test nout(v1) == 7
            @test nout(v2) == 4
            @test nout(pd1) == 1

            @test size(g(ones(Float32, 3,2))) == (noutv4, 2)

            Δnout(v4, +24)
            Δnout(v1, -5)
            noutv4 = nout(v4)

            @test nout(v1) == 6
            @test nout(v2) == 5
            @test nout(pd1) == 5

            select(m)

            @test nout(v1) == 6
            @test nout(v2) == 5
            @test nout(pd1) == 5

            # This is a sucky almost-reimplement-the-alg-in-a-different-way kinda test
            # I just can't think of a better test since there are many ways to get this wrong without messing up the sizes and the only way to spot it is worse performance when mutating a trained model
            indsv4 = out_inds(op(v4))
            indsv1 = out_inds(op(v1))
            indsv2 = out_inds(op(v2))
            indspd1 = out_inds(op(pd1))

            shift_nonneg(v, n, o) = map(x -> x < 0 ? x : x + n * length(v)+n+o, v)

            reps = minΔnoutfactor(v4)
            offs = 0
            # Concatentated activations from v1 and v2 are interleaved in the graph like this:
            # [a1, a2, a1, a2, a1, a2] where a1 is activation from v1 and a2 is activation from v2
            for r in 1:reps
                inds = (1:nout(v1)) .+ offs
                @test indsv4[inds] == shift_nonneg(indsv1, r-1, (r-1) * (nout(v1) - 2))
                offs = inds[end]

                inds = (1:nout(v2)) .+ offs
                @test indsv4[inds] == shift_nonneg(indsv2, r-1, (r-1) * (nout_org(op(v2)) + 1) + nout_org(op(v1)))
                offs = inds[end]
            end

            inds = (1:nout(pd1)) .+ offs
            @test indsv4[inds] == [45, -1, -1, -1, -1]

            apply_mutation(g)

            @test size(g(ones(Float32, 3,2))) == (noutv4, 2)
        end

        @testset "NeuronSelectMutation residual" begin

            inpt = inputvertex("in", 3, FluxDense())
            v1 = dense(inpt, 8, name="v1")
            v2 = dense(inpt, 4, name="v2")
            v3 = concat(v1,v2, traitfun=named("v3"))
            v4 = dense(v3, nout(v3), name="v4")
            v5 = "v5" >> v3 + v4
            v6 = dense(v5, 2, name="v6")

            rankfun(v, vvals) = NaiveGAflux.select_outputs(v, 1:nout_org(op(vvals)))
            m = NeuronSelectMutation(rankfun , NoutMutation(0.5))
            push!(m.m.mutated, v4)

            g = CompGraph(inpt, v5)
            @test size(g(ones(Float32, 3,2))) == (nout(v5), 2)

            Δnout(v4, -6)

            @test nout(v1) == 4
            @test nout(v2) == 2

            select(m)
            apply_mutation(g)

            @test nout(v1) == 4
            @test nout(v2) == 2

            @test size(g(ones(Float32, 3,2))) == (nout(v5), 2)
        end

        @testset "NeuronSelectMutation entangled SizeStack" begin
            inpt = inputvertex("in", 3, FluxDense())
            v1 = dense(inpt, 5, name="v1")
            v2 = dense(inpt, 4, name="v2")
            v3 = dense(inpt, 3, name="v3")
            v4 = dense(inpt, 6, name="v4")

            v5 = concat(v1, v2, traitfun=named("v5"))
            v6 = concat(v2, v3, traitfun=named("v6"))
            v7 = concat(v3, v4, traitfun=named("v7"))

            v8 = concat(v5, v6, traitfun=named("v8"))
            v9 = concat(v6, v7, traitfun=named("v9"))

            v10 = dense(inpt, nout(v9), name="v10")
            add = traitconf(named("add")) >> v8 + v9# + v10

            rankfun(v, vvals) = NaiveGAflux.select_outputs(v, 1:nout_org(op(vvals)))
            m = NeuronSelectMutation(rankfun , NoutMutation(0.5))
            push!(m.m.mutated, v10)

            g = CompGraph(inpt, add)
            @test size(g(ones(Float32, 3,2))) == (nout(add), 2)

            @test minΔnoutfactor(add) == 2
            Δnout(add, -4)

            @test nout(add) == 12

            select(m)
            apply_mutation(g)

            @test nout(add) == 12

            @test size(g(ones(Float32, 3,2))) == (nout(add), 2)
        end

        @testset "NeuronSelectMutation remove SizeAbsorb with SizeInvariant input decrease bigger output" begin
            inpt = inputvertex("in", 3, FluxDense())
            v1a = dense(inpt, 2, name="v1a")
            # Add a vertex which must be considered even though it is not input to v3
            v1b = dense(inpt, 2, name="v1b")
            vadd = "vadd" >> v1a + v1b
            v2 = batchnorm(v1a, name="v2")
            v3 = dense(v2, 4, name="v3")
            v4 = dense(v3, 3, name="v4")

            g = CompGraph(inpt, v4)
            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)

            rankfun(v, vvals) = NaiveGAflux.select_outputs(v, 1:nout_org(op(vvals)))
            m = NeuronSelectMutation(rankfun , RemoveVertexMutation(RemoveStrategy(DecreaseBigger())))

            m(v3)

            select(m)
            @test in_inds(op(v4))[] == [3, 4]
            @test out_inds(op(v1a)) == out_inds(op(v1b)) == out_inds(op(vadd)) == out_inds(op(v2)) == [1, 2]

            apply_mutation(g)

            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)
        end

        @testset "NeuronSelectMutation remove SizeStack input increase" begin
            inpt = inputvertex("in", 3, FluxDense())
            v0 = dense(inpt, 4, name="v0")
            v1 = dense(v0, 2, name="v1")
            v2 = dense(v0, 3, name="v2")
            v3 = dense(v0, 4, name="v3")

            v4 = concat("v4", v1,v2,v1,v3)
            v5 = dense(v4, 3, name="v5")

            g = CompGraph(inpt, v4)
            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)

            rankfun(v, vvals) = NaiveGAflux.select_outputs(v, 1:nout_org(op(vvals)))
            m = NeuronSelectMutation(rankfun , RemoveVertexMutation())

            Δnout(v1, 1)

            m(v1)

            @test nout(v4) == nin(v5)[] == 15

            select(m)
            @test out_inds(op(v4)) == [1, 2, -1, -1, 3, 4, 5, 6, 7, -1, -1, 8, 9, 10, 11]
            @test out_inds(op(v0)) == [1,2,3,4]

            apply_mutation(g)

            @test nout(v4) == nin(v5)[] == 15

            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)
        end

        @testset "NeuronSelectMutation remove SizeStack input decrease" begin
            inpt = inputvertex("in", 3, FluxDense())
            v0 = dense(inpt, 4, name="v0")
            v1 = dense(v0, 2, name="v1")
            v2 = dense(v0, 3, name="v2")
            v3 = dense(v0, 4, name="v3")

            v4 = concat("v4", v1,v2,v1,v3)
            v5 = dense(v4, 3, name="v5")

            g = CompGraph(inpt, v4)
            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)

            rankfun(v, vvals) = NaiveGAflux.select_outputs(v, 1:nout_org(op(vvals)))
            m = NeuronSelectMutation(rankfun , RemoveVertexMutation(RemoveStrategy(DecreaseBigger())))

            Δnout(v1, 1)
            m(v1)

            @test nout(v4) == nin(v5)[] == 13

            select(m)
            @test out_inds(op(v4)) == [1, 2, -1, 3, 4, 5, 6, 7, -1, 8, 9, 10, 11]
            @test out_inds(op(v0)) == [2,3,4]

            apply_mutation(g)

            @test nout(v4) == nin(v5)[] == 13

            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)
        end

        @testset "NeuronSelectMutation remove SizeStack input change both" begin
            inpt = inputvertex("in", 3, FluxDense())
            v0 = dense(inpt, 4, name="v0")
            v1 = dense(v0, 2, name="v1")
            v2 = dense(v0, 3, name="v2")
            v3 = dense(v0, 4, name="v3")

            v4 = concat("v4", v1,v2,v1,v3)
            v5 = dense(v4, 3, name="v5")

            g = CompGraph(inpt, v4)
            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)

            rankfun(v, vvals) = NaiveGAflux.select_outputs(v, 1:nout_org(op(vvals)))
            m = NeuronSelectMutation(rankfun , RemoveVertexMutation(RemoveStrategy(AlignSizeBoth())))

            Δnout(v1, 4)
            m(v1)

            @test nout(v4) == nin(v5)[] == 17

            select(m)
            @test out_inds(op(v4)) == [1, 2, -1, -1, -1, 3, 4, 5, 6, 7, -1, -1, -1, 8, 9, 10, 11]
            @test out_inds(op(v0)) == [1,2,3,4,-1]

            apply_mutation(g)

            @test nout(v4) == nin(v5)[] == 17

            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)
        end

        @testset "NeuronSelectMutation remove after SizeInvariant SizeStack input change both" begin
            inpt = inputvertex("in", 3, FluxDense())
            v0 = dense(inpt, 4, name="v0")
            vb = batchnorm(v0, name="vb")
            v1 = dense(vb, 2, name="v1")
            v2 = dense(v0, 3, name="v2")
            v3 = dense(v0, 4, name="v3")

            v4 = concat("v4", v1,v2,v1,v3)
            v5 = dense(v4, 3, name="v5")

            g = CompGraph(inpt, v4)
            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)

            rankfun(v, vvals) = NaiveGAflux.select_outputs(v, 1:nout_org(op(vvals)))
            m = NeuronSelectMutation(rankfun , RemoveVertexMutation(RemoveStrategy(AlignSizeBoth())))

            Δnout(v1, 4)
            m(v1)

            @test nout(v4) == nin(v5)[] == 17

            select(m)
            @test out_inds(op(v4)) == [1, 2, -1, -1, -1, 3, 4, 5, 6, 7, -6, -6, -6, 8, 9, 10, 11]
            @test out_inds(op(v0)) == [1,2,3,4,-1]

            apply_mutation(g)

            @test nout(v4) == nin(v5)[] == 17

            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)
        end

        @testset "NeuronSelectMutation remove SizeInvariant SizeStack input no change" begin
            inpt = inputvertex("in", 3, FluxDense())
            v0 = dense(inpt, 4, name="v0")
            v1 = dense(v0, 2, name="v1")
            v2 = dense(v0, 3, name="v2")
            v3 = dense(v0, 4, name="v3")

            vb = batchnorm(v1, name="vb")
            v4 = concat("v4", vb,v2,vb,v3)
            v5 = dense(v4, 3, name="v5")

            g = CompGraph(inpt, v4)
            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)

            rankfun(v, vvals) = NaiveGAflux.select_outputs(v, 1:nout_org(op(vvals)))
            m = NeuronSelectMutation(rankfun , RemoveVertexMutation())

            Δnout(v1, 1)
            m(vb)

            @test nout(v4) == nin(v5)[] == 13

            select(m)
            @test out_inds(op(v4)) == [1, 2, -1, 3, 4, 5, 6, 7, -1, 8, 9, 10, 11]
            @test out_inds(op(v1)) == [1,2,-1]

            apply_mutation(g)

            @test nout(v4) == nin(v5)[] == 13

            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)
        end

        @testset "NeuronSelectMutation remove SizeInvariant SizeStack output no change" begin
            inpt = inputvertex("in", 3, FluxDense())
            v0 = dense(inpt, 2, name="v0")
            v1 = dense(v0, 4, name="v1")

            v3 = concat("v3", v0, v1)
            v4 = batchnorm(v3, name="v4")
            v5 = dense(v4, 3, name="v5")

            g = CompGraph(inpt, v4)
            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)


            rankfun(v, vvals) = NaiveGAflux.select_outputs(v, vvals == v3 ? (1:nout_org(vvals)) : (nout_org(vvals):-1:1))

            m = NeuronSelectMutation(rankfun , RemoveVertexMutation(RemoveStrategy(AlignSizeBoth())))

            Δnout(v1, -1)
            m(v4)

            # Situation: both nout and nin appear to have changed as a result of the removal, but there was another reason
            # The risk here is that we don't propagate selected indices from v3 to v5 and vice versa
            select(m)
            @test out_inds(op(v3)) == in_inds(op(v5))[] == [1, 2, 4, 5, 6]
        end
    end

    @testset "RemoveZeroNout" begin
        inpt = inputvertex("in", 4, FluxDense())
        v0 = mutable("v0", Dense(nout(inpt), 5), inpt)
        v1 = mutable("v1", Dense(nout(v0), 5), v0)
        v2 = mutable("v2", Dense(nout(v1), 5), v1)

        function path(bname, add=nothing)
            p1 = mutable("$(bname)1", Dense(nout(v2), 6), v2)
            p2pa1 = mutable("$(bname)2pa1", Dense(nout(p1), 2), p1)
            p2pa2 = mutable("$(bname)2pa2", Dense(nout(p2pa1), 2), p2pa1)
            p2pb1 = mutable("$(bname)2pb1", Dense(nout(p1), 1), p1)
            p2pb2 = mutable("$(bname)2pb2", Dense(nout(p2pb1), 5), p2pb1)
            if !isnothing(add) p2pb2 = traitconf(named("$(bname)2add")) >> p2pb2 + add end
            p3 = concat(p2pa2,p2pb2, traitfun=named("$(bname)3"))
            return mutable("$(bname)4", Dense(nout(p3), 4), p3)
        end

        vert(want::String, graph::CompGraph) = vertices(graph)[name.(vertices(graph)) .== want][]
        vert(want::String, v::AbstractVertex) = flatten(v)[name.(flatten(v)) .== want][]

        for to_rm in ["pa4", "pa2pa2"]
            for vconc_out in [nothing, "pa1", "pa2pa2", "pa3", "pa4"]
                for vadd_in in [nothing, v0, v1, v2]

                    #Path to remove
                    pa = path("pa",vadd_in)
                    #Other path
                    pb = path("pb")

                    v3 = concat(pa,pb, traitfun = named("v3"))
                    v4 = mutable("v4", Dense(nout(v3), 6), v3)
                    if !isnothing(vconc_out)
                        v4 = concat(v4, vert(vconc_out, v4), traitfun = named("v4$(vconc_out)_conc"))
                    end
                    v5 = mutable("v5", Dense(nout(v4), 5), v4)

                    g = copy(CompGraph(inpt, v5))
                    @test size(g(ones(4,2))) == (5,2)

                    to_remove = vert(to_rm, g)
                    Δnout(to_remove, -nout(to_remove))

                    RemoveZeroNout()(g)
                    apply_mutation(g)
                    @test !in(to_remove, vertices(g))
                    @test size(g(ones(4,2))) == (5,2)
                end
            end
        end
    end

end
