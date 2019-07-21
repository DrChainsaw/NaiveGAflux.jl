

@testset "Mutation" begin

    @testset "Unsupported fallback" begin
        struct Dummy <:AbstractMutation{Any} end
        @test_throws ArgumentError mutate(Dummy(), "Test")
    end

    struct ProbeMutation{T} <:AbstractMutation{T}
        seen::AbstractVector
        ProbeMutation{T}() where T = new(T[])
    end
    NaiveGAflux.mutate(m::ProbeMutation{T}, t::T) where T= push!(m.seen, t)

    @testset "MutationProbability" begin
        probe = ProbeMutation{Int}()
        m = MutationProbability(probe, Probability(0.3, MockRng([0.2,0.5,0.1])))

        mutate(m, 1)
        mutate(m, 2)
        mutate(m, 3)
        mutate(m, 4)
        @test probe.seen == [1,3,4]
    end

    dense(in, outsizes...) = foldl((next,size) -> mutable(Dense(nout(next), size), next), outsizes, init=in)

    @testset "VertexMutation" begin
        inpt = inputvertex("in", 4, FluxDense())
        outpt = dense(inpt, 3,4,5)
        graph = CompGraph(inpt, outpt)

        probe = ProbeMutation{AbstractVertex}()
        m = VertexMutation(probe)
        mutate(m, graph)
        # Vertex 1 (inpt) is immutable, all others are selected
        @test probe.seen == vertices(graph)[2:end]
    end
end
