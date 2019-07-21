

@testset "Mutation" begin
    using NaiveNASflux
    using Random

    @testset "Unsupported fallback" begin
        struct Dummy <:AbstractMutation end
        @test_throws ArgumentError mutate(Dummy(), "Test")
    end

    mutable struct MockRng <:AbstractRNG
        seq::AbstractVector
        ind::Integer
        MockRng(seq) = new(seq, 0)
    end

    function Random.rand(rng::MockRng)
        rng.ind = rng.ind % length(rng.seq) + 1
        return rng.seq[rng.ind]
    end


    @testset "Probability" begin
        import NaiveGAflux: apply
        @test_throws AssertionError Probability(-1)
        @test_throws AssertionError Probability(1.1)

        p = Probability(0.3, MockRng([0.4, 0.3, 0.2]))
        @test !apply(p)
        @test !apply(p)
        @test apply(p)

        cnt = 0
        ff() = cnt += 1
        apply(ff, p)
        @test cnt == 0
        apply(ff, p)
        @test cnt == 0
        apply(ff, p)
        @test cnt == 1

        p = Probability(0.3, MockRng(0:0.1:0.9))
        cnt = 0
        foreach(_ -> apply(ff, p), 1:10)
        @test cnt == 3
    end

    struct ProbeMutation <:AbstractMutation
        seen::AbstractVector
        ProbeMutation() = new(Any[])
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
