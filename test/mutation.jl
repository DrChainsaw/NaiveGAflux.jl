

@testset "Mutation" begin

    @testset "Unsupported fallback" begin
        struct Dummy <:AbstractMutation{Any} end
        @test_throws ArgumentError mutate(Dummy(), "Test")
    end

    struct NoOpMutation{T} <:AbstractMutation{T} end
    function NaiveGAflux.mutate(m::NoOpMutation, t) end
    ProbeMutation(T) = RecordMutation(NoOpMutation{T}())

    @testset "MutationProbability" begin
        probe = ProbeMutation(Int)
        m = MutationProbability(probe, Probability(0.3, MockRng([0.2,0.5,0.1])))

        mutate(m, 1)
        mutate(m, 2)
        mutate(m, 3)
        mutate(m, 4)
        @test probe.mutated == [1,3,4]
    end

    dense(in, outsizes...) = foldl((next,size) -> mutable(Dense(nout(next), size), next), outsizes, init=in)

    @testset "VertexMutation" begin
        inpt = inputvertex("in", 4, FluxDense())
        outpt = dense(inpt, 3,4,5)
        graph = CompGraph(inpt, outpt)

        probe = ProbeMutation(AbstractVertex)
        m = VertexMutation(probe)
        mutate(m, graph)
        # Vertex 1 (inpt) is immutable, all others are selected
        @test probe.mutated == vertices(graph)[2:end]
    end

    @testset "NoutMutation" begin
        inpt = inputvertex("in", 3, FluxDense())

        # Can't mutate, don't do anything
        mutate(NoutMutation(0.4), inpt)
        @test nout(inpt) == 3

        rng = MockRng([0.5])
        v = dense(inpt, 11)

        mutate(NoutMutation(0.4, rng), v)
        @test nout(v) == 13

        mutate(NoutMutation(-0.4, rng), v)
        @test nout(v) == 11

        mutate(NoutMutation(-0.001, rng), v)
        @test nout(v) == 10

        mutate(NoutMutation(0.001, rng), v)
        @test nout(v) == 11
    end
end
