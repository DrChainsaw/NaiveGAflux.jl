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

@testset "MutationShield" begin
    v1 = inputvertex("v1", 3)
    v2 = mutable("v2", Dense(nout(v1), 5), v1)
    v3 = mutable("v3", Dense(nout(v2), 4), v2, traitfun = MutationShield)
    v4 = mutable("v4", Dense(nout(v3), 2), v3, traitfun = validated() ∘ MutationShield)

    @test !allow_mutation(v1)
    @test allow_mutation(v2)
    @test !allow_mutation(v3)
    @test !allow_mutation(v4)
end
