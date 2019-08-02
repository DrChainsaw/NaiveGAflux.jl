

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
        using Logging
        probe = ProbeMutation(Int)
        m = LogMutation(i -> "Mutate $i", probe)

        @test_logs (:info, "Mutate 17") m(17)
        @test probe.mutated == [17]
    end

    dense(in, outsizes...;layerfun = LazyMutable) = foldl((next,size) -> mutable(Dense(nout(next), size), next, layerfun=layerfun), outsizes, init=in)

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

        oddfirst(v) = reverse(vcat(1:2:nout_org(op(v)), 2:2:nout_org(op(v))))

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
            @test in_inds(op(v3)) == [[1,2,3]]
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

    end
end
