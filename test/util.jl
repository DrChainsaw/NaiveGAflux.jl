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
    apply(ff, p, () -> cnt -= 2)
    @test cnt == -1


    p = Probability(0.3, MockRng(0:0.1:0.9))
    cnt = 0
    foreach(_ -> apply(ff, p), 1:10)
    @test cnt == 3

    p = Probability(0.5, MockRng([0.1, 0.9]))
    @test apply(() -> :a, p, () -> :b) == :a
    @test apply(() -> :a, p, () -> :b) == :b
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

    t = MutationShield(SizeAbsorb())
    ct(::SizeAbsorb;cf) = SizeInvariant()
    ct(x...;cf=ct) = clone(x...,cf=cf)
    tn = ct(t)
    @test base(tn) == SizeInvariant()
end

@testset "VertexSelection" begin

    v1 = inputvertex("v1", 3)
    v2 = mutable("v2", Dense(nout(v1), 5), v1)
    v3 = mutable("v3", Dense(nout(v2), 4), v2, traitfun = MutationShield)
    v4 = mutable("v4", Dense(nout(v2), 2), v3)
    g1 = CompGraph(v1, v4)

    @testset "AllVertices" begin
        @test select(AllVertices(), g1) == [v1,v2,v3,v4]
    end

    @testset "FilterMutationAllowed" begin
        @test select(FilterMutationAllowed(), g1) == [v2,v4]
    end

end

@testset "remove_redundant_vertices" begin
    v1 = inputvertex("in", 3)
    v2 = mutable("v2", Dense(nout(v1), 5), v1)
    v3 = mutable("V3", Dense(nout(v1), 5), v1)
    v4 = traitconf(t -> RemoveIfSingleInput(NamedTrait(t, "v4"))) >>  v2 + v3
    v5 = mutable("v5", BatchNorm(nout(v4)), v4)
    v6 = concat("v6", v3, v5, traitfun = t -> RemoveIfSingleInput(t))
    v7 = mutable("v7", Dense(nout(v6), 2), v6)

    g = CompGraph(v1, v7)

    nv_pre = nv(g)

    check_apply(g)
    # Nothing is redundant
    @test nv_pre == nv(g)

    remove_edge!(v3, v4)
    check_apply(g)

    @test nv_pre == nv(g)+1

    # Note, v3 also disappears from the graph as it is no longer used to compute the output
    remove_edge!(v3, v6)
    check_apply(g)

    @test nv_pre == nv(g) + 3
end

@testset "Clone ApplyIf" begin
    t = ApplyIf(x -> true, identity, SizeAbsorb())
    ct(::SizeAbsorb;cf) = SizeInvariant()
    ct(x...;cf=ct) = clone(x...,cf=cf)
    tn = ct(t)
    @test base(tn) == SizeInvariant()
end

@testset "PersistentArray" begin
    testdir = "testPersistentArray"
    pa = PersistentArray(testdir, 5, identity)

    @test pa == 1:5
    @test map(identity, pa) == pa
    @test identity.(pa) == pa
    @test mapfoldl(identity, vcat, pa) == pa
    pa[1] = 3
    @test pa == [3,2,3,4,5]

    @test identity.(pa) == pa

    try
        persist(pa)
        @test PersistentArray(testdir, 7, x -> 2x) == [3,2,3,4,5,12,14]

        rm(pa, 2)
        @test PersistentArray(testdir, 3, x -> 17) == [3,17,3]

        rm(pa)
        @test PersistentArray(testdir, 3, x -> 11) == [11,11,11]
    finally
        rm(testdir, force=true, recursive=true)
    end
end

@testset "Bounded Random Walk" begin
    import NaiveGAflux: BoundedRandomWalk
    using Random

    driftup = BoundedRandomWalk(-1.0, 1.0, i -> 0.2i^2)
    @test cumsum([driftup(i) for i in 1:5]) == [0.2, 0.8, 1.0, 1.0, 1.0]

    driftdown = BoundedRandomWalk(-1.0, 1.0, i -> -0.2i^2)
    @test cumsum([driftdown(i) for i in 1:5]) == -[0.2, 0.8, 1.0, 1.0, 1.0]

    rng = MersenneTwister(0);
    brw = BoundedRandomWalk(-2.34, 3.45, () -> randn(rng))
    @test collect(extrema(cumsum([brw() for i in 1:10000]))) ≈ [-2.34, 3.45] atol = 1e-10
end

@testset "Mergeopts" begin
    import NaiveGAflux: mergeopts, learningrate

    dm = mergeopts(Descent(0.1), Descent(2), Descent(0.4))
    @test learningrate(dm) ≈ 0.08

    wd = mergeopts(WeightDecay(0.1), WeightDecay(2), WeightDecay(0.4))
    @test wd.wd ≈ 0.08

    dd = mergeopts(Momentum, Descent(0.1), Descent(2), Descent(0.4))
    @test typeof.(dd) == [Descent, Descent, Descent]

    mm = mergeopts(Momentum, Descent(0.1), Momentum(0.2), Momentum(0.3), Descent(0.2))
    @test typeof.(mm) == [Descent, Descent, Momentum]
end

@testset "Optimizer trait" begin
    import NaiveGAflux: opttype, optmap, FluxOptimizer

    @test opttype("Not an optimizer") == nothing
    @test opttype(Descent()) == FluxOptimizer()
    @test opttype(Flux.Optimiser(Descent(), ADAM())) == FluxOptimizer()

    isopt = "Is an optimizer!"
    noopt = "Is not an optimizer!"
    om = optmap(o -> isopt, o -> noopt)

    @test om(1234) == noopt
    @test om(Momentum()) == isopt
    @test om(NaiveGAflux.ShieldedOpt(Momentum())) == isopt
    @test om(Flux.Optimiser(Momentum(), Nesterov())) == isopt
end
