

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

    dense(in, outsizes...) = foldl((next,size) -> mutable(Dense(nout(next), size), next), outsizes, init=in)

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
end
