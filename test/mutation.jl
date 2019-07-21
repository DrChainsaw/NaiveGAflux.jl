

@testset "Mutation" begin

    @testset "Unsupported fallback" begin
        struct Dummy <:AbstractMutation{Any} end
        @test_throws ArgumentError mutate(Dummy(), "Test")
    end

    struct ProbeMutation <:AbstractMutation{AbstractVertex}
        seen::AbstractVector
        ProbeMutation() = new(AbstractVertex[])
    end
    NaiveGAflux.mutate(m::ProbeMutation, t) = push!(m.seen, t)

    dense(in, outsizes...) = foldl((next,size) -> mutable(Dense(nout(next), size), next), outsizes, init=in)

    @testset "VertexMutation" begin
        inpt = inputvertex("in", 4, FluxDense())
        outpt = dense(inpt, 3,4,5)
        graph = CompGraph(inpt, outpt)

        probe = ProbeMutation()
        m = VertexMutation(probe, Probability(0.3, MockRng([0.2,0.5,0.1])))
        mutate(m, graph)
        @test probe.seen == vertices(graph)[[1,3,4]]
    end
end
