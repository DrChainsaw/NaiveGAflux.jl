

@testset "Mutation" begin

    struct NoOpMutation{T} <:AbstractMutation{T} end
    (m::NoOpMutation{T})(t::T) where T = t
    ProbeMutation(T) = RecordMutation(NoOpMutation{T}())

    @testset "MutationProbability" begin
        probe = ProbeMutation(Int)
        m = MutationProbability(probe, Probability(0.3, MockRng([0.2,0.5,0.1])))

        @test m(1) == 1
        @test m(2) == 2
        @test m(3) == 3
        @test m(4) == 4
        @test probe.mutated == [1,3,4]
    end

    @testset "MutationProbability vector" begin
        probe = ProbeMutation(Int)
        m = MutationProbability(probe, Probability(0.3, MockRng([0.2,0.5,0.1])))

        @test m(1:4) == 1:4
        @test probe.mutated == [1,3,4]
    end

    @testset "WeightedMutationProbability" begin
        probe = ProbeMutation(Real)
        rng = MockRng([0.5])
        m = WeightedMutationProbability(probe, p -> Probability(p, rng))

        @test m(0.1) == 0.1
        @test m(0.6) == 0.6
        @test m(0.4) == 0.4
        @test m(0.9) == 0.9
        @test probe.mutated == [0.6,0.9]
    end

    @testset "WeightedMutationProbability vector" begin
        probe = ProbeMutation(Real)
        rng = MockRng([0.5])
        m = WeightedMutationProbability(probe, p -> Probability(p, rng))

        @test m([0.1, 0.6, 0.4, 0.9]) == [0.1, 0.6, 0.4, 0.9]
        @test probe.mutated == [0.6,0.9]
    end

    @testset "Neuron value weighted mutation" begin
        using Statistics
        struct DummyValue{T, W<:AbstractMutableComp} <: AbstractMutableComp
            values::T
            w::W
        end
        NaiveNASflux.neuron_value(d::DummyValue) = d.values
        NaiveNASflux.wrapped(d::DummyValue) = d.w

        l(in, outsize, value) = mutable(Dense(nout(in), outsize), in, layerfun = l -> DummyValue(value, l))

        v0 = inputvertex("in", 3)
        v1 = l(v0, 4, 1:4)
        v2 = l(v1, 3, 100:300)
        v3 = l(v2, 5, 0.1:0.1:0.5)

        @testset "weighted_neuron_value_high pbase $pbase" for pbase in (0.05, 0.1, 0.3, 0.7, 0.9, 0.95)
            import NaiveGAflux: weighted_neuron_value_high
            wnv = weighted_neuron_value_high(pbase, spread=0.5)
            wp = map(p -> p.p, wnv.([v1,v2,v3]))
            @test wp[2] > wp[1] > wp[3]
            @test mean(wp) ≈ pbase rtol = 0.1
        end

        @testset "HighValueMutationProbability" begin

            probe = ProbeMutation(MutationVertex)
            m = HighValueMutationProbability(probe, 0.1, MockRng([0.15]))

            m(v1)
            m(v2)
            m(v3)
            @test probe.mutated == [v2]
        end

        @testset "weighted_neuron_value_low pbase $pbase" for pbase in (0.05, 0.1, 0.3, 0.7, 0.9, 0.95)
            import NaiveGAflux: weighted_neuron_value_low
            wnv = weighted_neuron_value_low(pbase,spread=0.8)
            wp = map(p -> p.p, wnv.([v1,v2,v3]))
            @test wp[2] < wp[1] < wp[3]
            @test mean(wp) ≈ pbase rtol = 0.1
        end

        @testset "LowValueMutationProbability" begin
            probe = ProbeMutation(MutationVertex)
            m = LowValueMutationProbability(probe, 0.1, MockRng([0.15]))

            m(v1)
            m(v2)
            m(v3)
            @test probe.mutated == [v1, v3]
        end
    end

    @testset "MutationChain" begin
        probes = ProbeMutation.(repeat([Int], 3))
        m = MutationChain(probes...)
        @test m(1) == 1
        @test getfield.(probes, :mutated) == [[1],[1],[1]]
    end

    @testset "MutationChain vector" begin
        probes = ProbeMutation.(repeat([Int], 3))
        m = MutationChain(probes...)
        @test m(1:2) == 1:2
        @test getfield.(probes, :mutated) == [[1,2],[1,2],[1,2]]
    end

    @testset "LogMutation" begin
        probe = ProbeMutation(Int)
        m = LogMutation(i -> "Mutate $i", probe)

        @test @test_logs (:info, "Mutate 17") m(17) == 17
        @test probe.mutated == [17]
    end

    @testset "LogMutation vector" begin
        probe = ProbeMutation(Int)
        m = LogMutation(i -> "Mutate $i", probe)

        @test @test_logs (:info, "Mutate 17") (:info, "Mutate 21") m([17, 21]) == [17, 21]
        @test probe.mutated == [17, 21]
    end

    @testset "MutationFilter" begin
        probe = ProbeMutation(Int)
        m = MutationFilter(i -> i > 3, probe)

        @test m(1) == 1
        @test probe.mutated == []

        @test m(4) == 4
        @test probe.mutated == [4]
    end

    @testset "MutationFilter vector" begin
        probe = ProbeMutation(Int)
        m = MutationFilter(i -> i > 3, probe)

        @test m(1:5) == 1:5
        @test probe.mutated == [4,5]
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
        @test m(11) == 11

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
        @test m(graph) == graph
        # Vertex 1 (inpt) is immutable, all others are selected
        @test probe.mutated == vertices(graph)[2:end]
    end

    @testset "NoutMutation" begin
        inpt = inputvertex("in", 3, FluxDense())

        # Can't mutate, don't do anything
        @test NoutMutation(0.4)(inpt) == inpt
        @test nout(inpt) == 3

        # Can't mutate due to too small size
        v = dense(inpt, 1)
        @test NoutMutation(-0.8, -1.0)(v) == v
        @test nout(v) == 1

        rng = MockRng([0.5])
        v = dense(inpt, 11)

        @test NoutMutation(0.4, rng)(v) == v
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

    @testset "NoutMutation vector" begin
        inpt = inputvertex("in", 3, FluxDense())
        v1 = dense(inpt, 4; name="v1")
        v2 = dense(v1, 5; name="v2")
        v3 = dense(v2, 6; name="v3")
        v4 = dense(v3, 7; name="v4")

        rng = MockRng([0.5])
        @test NoutMutation(0.8, rng)([inpt,v2,v3]) == [inpt,v2,v3]

        @test nout(v1) == 4
        @test nout(v2) == 7
        @test nout(v3) == 8
        @test nout(v4) == 7
    end

    @testset "AddVertexMutation" begin
        inpt = inputvertex("in", 3, FluxDense())
        v1 = dense(inpt, 5)

        @test inputs(v1) == [inpt]

        space = ArchSpace(DenseSpace(4, relu))

        @test AddVertexMutation(space)(inpt) == inpt

        @test inputs(v1) != [inpt]
        @test nin(v1) == [3]
        @test inputs(v1)[](1:nout(inpt)) == 1:nout(inpt)

        v2 = dense(v1, 2)
        v3 = dense(v1, 1)

        @test AddVertexMutation(space, outs -> [outs[2]])(v1) == v1

        @test inputs(v2) == [v1]
        @test inputs(v3) != [v1]
    end

    @testset "RemoveVertexMutation" begin
        @testset "Simple" begin
            inpt = inputvertex("in", 3, FluxDense())
            v1 = dense(inpt, 5)
            v2 = dense(v1, 3)

            @test RemoveVertexMutation()(v1) == v1

            @test inputs(v2) == [inpt]
            @test [nout(inpt)] == nin(v2) == [3]
        end

        @testset "Simple aligned" begin
            inpt = inputvertex("in", 3, FluxDense())
            v1 = dense(inpt, 3)
            v2 = dense(v1, 3)

            @test RemoveVertexMutation()(v1) == v1

            @test inputs(v2) == [inpt]
            @test [nout(inpt)] == nin(v2) == [3]
        end

        @testset "Size cycle" begin
            inpt = inputvertex("in", 3, FluxDense())
            v1 = dense(inpt, 4)
            v2a = dense(v1, 2; name="v2a")
            v2b = dense(v1, 2; name="v2b")
            v3 = concat("v3", v2a, v2b)
            v4 = "v4" >> v3 + v1

            @test_logs (:warn, r"Size cycle detected!") RemoveVertexMutation()(v2b)
        end
    end

    @testset "KernelSizeMutation $convtype" for convtype in (Conv, ConvTranspose, DepthwiseConv)
        v = mutable(convtype((3,5), 2=>2, pad=(1,1,2,2)), inputvertex("in", 2))
        indata = ones(Float32, 7,7,2,2)
        @test size(v(indata)) == size(indata)

        rng = SeqRng()
        @test KernelSizeMutation2D(4,rng=rng)(v) == v

        @test size(v(indata)) == size(indata)
        @test size(NaiveNASflux.weights(layer(v)))[1:2] == (1, 2)

        # Test with maxvalue
        @test KernelSizeMutation(Singleton2DParSpace(4), maxsize=v->(4, 10))(v) == v
        @test size(v(indata)) == size(indata)
        @test size(NaiveNASflux.weights(layer(v)))[1:2] == (4, 6)
    end

    @testset "ActivationFunctionMutation Dense" begin
        v = mutable(Dense(2,3), inputvertex("in", 2))
        @test ActivationFunctionMutation(elu)(v) == v
        @test layer(v).σ == elu
    end

    @testset "ActivationFunctionMutation RNN" begin
        v = mutable(RNN(2,3), inputvertex("in", 2))
        @test ActivationFunctionMutation(elu)(v) == v
        @test layer(v).cell.σ == elu
    end

    @testset "ActivationFunctionMutation $convtype" for convtype in (Conv, ConvTranspose, DepthwiseConv)
        v = mutable(convtype((3,5), 2=>2), inputvertex("in", 2))
        @test ActivationFunctionMutation(elu)(v) == v
        @test layer(v).σ == elu
    end

    Flux.GroupNorm(n) = GroupNorm(n,n)
    @testset "ActivationFunctionMutation $normtype" for normtype in (BatchNorm, InstanceNorm, GroupNorm)
        v = mutable(normtype(2), inputvertex("in", 2))
        @test ActivationFunctionMutation(elu)(v) == v
        @test layer(v).λ == elu
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
        vert(want::String, v::AbstractVertex) = NaiveNASlib.flatten(v)[name.(NaiveNASlib.flatten(v)) .== want][]

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
                    Δsize!(NaiveNASlib.NeuronIndices(), NaiveNASlib.OnlyFor(), to_remove, [missing],  Int[])

                    RemoveZeroNout()(g)
                    Δsize!(AlignNinToNout(), vertices(g))

                    @test !in(to_remove, vertices(g))
                    @test size(g(ones(4,2))) == (5,2)
                end
            end
        end
    end

    @testset "AddEdgeMutation" begin
        import NaiveGAflux: default_mergefun
        cl(name, in, outsize; kwargs...) = mutable(name, Conv((1,1), nout(in)=>outsize; kwargs...), in)
        dl(name, in, outsize) = mutable(name, Dense(nout(in), outsize), in)
 
        @testset "No shapechange" begin
            import NaiveGAflux: no_shapechange

            @testset "Test size changing ops" begin
                v0 = inputvertex("in", 3, FluxConv{2}())
                v1 = cl("v1", v0, 4)
                v2 = cl("v2", v1, 5)
                v3 = mutable("v3", MaxPool((2,2)), v2)
                v4 = cl("v4", v3, 4)
                v5 = cl("v5", v4, 3)
                v6 = cl("v6", v5, 2; stride=2)
                v7 = cl("v7", v6, 3)
                v8 = cl("v8", v7, 4)
                v9 = cl("v9", v8, 2; pad=1)
                v10 = cl("v10", v9, 3)

                @test name.(no_shapechange(v0)) == name.([v2,v3])
                @test name.(no_shapechange(v1)) == name.([v3])
                @test name.(no_shapechange(v2)) == []

                @test name.(no_shapechange(v3)) == name.([v5, v6])
                @test name.(no_shapechange(v4)) == name.([v6])
                @test name.(no_shapechange(v5)) == []

                @test name.(no_shapechange(v6)) == name.([v8, v9])
                @test name.(no_shapechange(v7)) == name.([v9])
                @test name.(no_shapechange(v8)) == []
            end

            @testset "Branchy graph" begin
                v0 = inputvertex("in", 3, FluxConv{2}())
                v1 = cl("v1", v0, 4)

                v1a1 = cl("v1a1", v1, 5)
                v1a2 = cl("v1a2", v1a1, 3)

                v1b1 = cl("v1b1", v1, 4)

                v2 = concat("v2", v1a2, v1b1)

                v2a1 = cl("v2a1", v2, 2;pad=1)
                v2a2 = cl("v2a2", v2a1, 3)
                v2a3 = cl("v2a3", v2a2, 4)

                v2b1 = cl("v2b1", v2, 4)
                v2b2 = cl("v2b2", v2b1, 3; pad=1)
                v2b3 = cl("v2b3", v2b2, 2)

                v2c1 = cl("v2c1", v2, 5)
                v2c2 = cl("v2c2", v2c1, 3)
                v2c3 = cl("v2c3", v2c2, 2;pad=1)

                v3 = concat("v3", v2a3, v2b3, v2c3)
                v4 = cl("v4", v3, 3)

                @test name.(no_shapechange(v0)) == name.([v1a1, v1a2, v1b1, v2, v2a1, v2b1, v2b2, v2c1, v2c2, v2c3])
                @test name.(no_shapechange(v1)) == name.([v1a2, v2, v2a1, v2b1, v2b2, v2c1, v2c2, v2c3])

                @test name.(no_shapechange(v1a1)) == name.([v2, v2a1, v2b1, v2b2, v2c1, v2c2, v2c3])
                @test name.(no_shapechange(v1b1)) == name.([v1a2, v2a1, v2b1, v2b2, v2c1, v2c2, v2c3])

                @test name.(no_shapechange(v2a1)) == name.([v2a3, v2b3, v3, v4])
                @test name.(no_shapechange(v2b2)) == name.([v2a2, v2a3, v3, v4])
                @test name.(no_shapechange(v2c3)) == name.([v2a2, v2a3, v2b3, v4])
            end

            @testset "With global pool and flatten" begin
                v0 = inputvertex("in", 3, FluxConv{2}())
                v1 = cl("v1", v0, 2)
                v2 = cl("v2", v1, 3)
                v3 = cl("v3", v2, 4)
                v4 = invariantvertex(NaiveGAflux.GlobalPool{MaxPool}(), v3; traitdecoration=named("v4"))
                v5 = dl("v5", v4, 3)
                v6 = dl("v6", v5, 4)
                v7 = dl("v7", v6, 2)

                @test name.(no_shapechange(v0)) == name.([v2,v3,v4])
                @test name.(no_shapechange(v1)) == name.([v3, v4])
                @test name.(no_shapechange(v2)) == name.([v4])
                @test name.(no_shapechange(v3)) == []

                @test name.(no_shapechange(v4)) == name.([v6,v7])
                @test name.(no_shapechange(v5)) == name.([v7])
            end
        end

        @testset "AddEdgeMutation pconc=$pconc" for pconc in (0, 1)

            v0 = inputvertex("in", 3, FluxConv{2}())
            v1 = cl("v1", v0, 4)
            v2 = cl("v2", v1, 5)
            v3 = cl("v3", v2, 3)
            v4 = mutable("v4", MaxPool((2,2)), v3)
            v5 = cl("v5", v4, 4)
            v6 = cl("v6", v5, 3)
            v7 = cl("v7", v6, 2)

            g = CompGraph(v0, v7)

            indata = ones(Float32,5,4,3,2)
            @test size(g(indata)) == (2,2,2,2)

            m = AddEdgeMutation(1.0, mergefun = default_mergefun(pconc), valuefun = v -> 1:nout(v))

            # edge to v2 not possible as v1 is input to it already
            @test m(v1) == v1

            vm1 = outputs(v1)[end]
            @test outputs(v1) == [v2, vm1]
            @test outputs(vm1) == [v3]
            @test inputs(vm1) == [v2, v1]
            @test size(g(indata)) == (2,2,2,2)

            # Will select vm1 again which already has v1 as input
            @test m(v1) == v1
            @test outputs(v1) == [v2, vm1]
            @test outputs(vm1) == [v3]
            @test inputs(vm1) == [v2, v1]
            @test size(g(indata)) == (2,2,2,2)

            @test m(v4) == v4
            vm4 = outputs(v4)[end]
            @test outputs(v4) == [v5, vm4]
            @test outputs(vm4) == [v6]
            @test inputs(vm4) == [v5, v4]
            @test size(g(indata)) == (2,2,2,2)
        end

        @testset "AddEdgeMutation first in path from multi-input capable" begin
            v0 = inputvertex("in", 3, FluxConv{2}())
            v1 = cl("v1", v0, 3)
            v2 = "v2" >> v0 + v1
            v2a1 = cl("v2a1", v2, 3)
            v2b1 = cl("v2b1", v2, 3)
            v3 = "v3" >> v2a1 + v2b1
  
            m = AddEdgeMutation(1.0,  mergefun = default_mergefun(0.0), valuefun = v -> 1:nout(v))
            
            m(v2a1)
            # Synopsis: v2b1 is selected as a suitable candidate
            # Since it is single input only, its inputs (v2) will be considered
            # However, v2 is also input to v2a1 meaning it can not be used despite being multi input
            @test name.(outputs(v2a1)) == [name(v3)]
        end

        @testset "AddEdgeMutation fail" begin
            m = AddEdgeMutation(1.0, mergefun=default_mergefun(1), valuefun=v -> 1:nout(v))

            v0 = inputvertex("in", 4, FluxConv{2}())
            v1 = cl("v1", v0, 4)

            # No suitable vertex as v0 is already output to v1
            @test m(v0) == v0
            @test all_in_graph(v0) == [v0, v1]
            @test outputs(v0) == [v1]

            v2 = cl("v2", v1, nout(v0) ÷ 2)

            # No suitable vertx as v1 is already output to v2
            @test m(v1) == v1
            @test all_in_graph(v0) == [v0, v1, v2]
            @test outputs(v1) == [v2]

            v3 = concat("v3", v2, v2)
            v4 = "v4" >> v0 + v3

            # Will try to add v1 as input to v3, but v4 can not change size as v0 is immutable
            @test_logs (:warn, "Could not align sizes of v2 and v4!") m(v2)
            @test Set(all_in_graph(v0)) == Set([v0, v1, v2, v3, v4])
            @test outputs(v2) == [v3, v3]
            @test inputs(v4) == [v0, v3]
        end
    end

    @testset "RemoveEdgeMutation" begin
        dl(name, in, outsize) = mutable(name, Dense(nout(in), outsize), in)

        @testset "RemoveEdgeMutation SizeStack" begin
            m = RemoveEdgeMutation(valuefun=v->1:nout(v))

            v0 = inputvertex("in", 3, FluxDense())
            v1 = dl("v1", v0, 4)
            v2 = dl("v2", v1, 5)

            # No action
            @test m(v1) == v1
            @test outputs(v1) == [v2]

            v3 = dl("v3", v0, 6)
            v4 = concat("v4", v1, v3)

            @test outputs(v1) == [v2, v4]
            @test inputs(v4) == [v1, v3]
            # v4 fulfils the criteria for removal
            @test m(v1) == v1
            @test outputs(v1) == [v2]
            @test inputs(v4) == [v3]

            g = CompGraph(v0, v4)
            indata = ones(Float32, nout(v0), 3)

            @test size(g(indata)) == (nout(v4), 3)
        end

        @testset "RemoveEdgeMutation SizeInvariant" begin
            m = RemoveEdgeMutation(valuefun=v->1:nout(v))

            v0 = inputvertex("in", 3, FluxDense())
            v1 = dl("v1", v0, nout(v0))
            v2 = dl("v2", v1, 5)
            v3 = dl("v3", v0, nout(v0))
            v4 = "v4" >> v1 + v3

            @test outputs(v1) == [v2, v4]
            @test inputs(v4) == [v1, v3]
            # v4 fulfils the criteria for removal
            @test m(v1) == v1
            @test outputs(v1) == [v2]
            @test inputs(v4) == [v3]

            g = CompGraph(v0, v4)
            indata = ones(Float32, nout(v0), 3)

            @test size(g(indata)) == (nout(v4), 3)
        end
    end

    @testset "OptimizerMutation" begin
        import NaiveGAflux: sameopt, learningrate
        import NaiveGAflux.Flux.Optimise: Optimiser

        @testset "Mutate learning rate" begin
            m = OptimizerMutation(o -> sameopt(o, 10 * learningrate(o)))

            @test learningrate(m(Descent(0.1))) == 1.0
            @test learningrate(m(ShieldedOpt(Momentum(0.1)))) == 0.1
            @test learningrate(m(Optimiser(Nesterov(0.1), ShieldedOpt(ADAM(0.1))))) == 0.1

            @test learningrate(LearningRateMutation(MockRng([0.0]))(Descent(0.1))) == 0.085
        end

        @testset "Mutate optimizer type" begin
            m = OptimizerMutation((Momentum, ))

            @test typeof(m(Descent())) == Momentum
            @test typeof(m(ShieldedOpt(Descent()))) == ShieldedOpt{Descent}
            @test typeof.(m(Optimiser(Nesterov(), ShieldedOpt(ADAM()))).os) == [Momentum, ShieldedOpt{ADAM}]
        end

        @testset "Add optimizer" begin
            m = AddOptimizerMutation(o -> Descent(0.1))

            @test typeof.(m(Descent(0.2)).os) == [Descent]
            @test typeof.(m(Momentum(0.2)).os) == [Momentum, Descent]
            @test typeof.(m(Flux.Optimiser(Nesterov(), Descent(), ShieldedOpt(Descent()))).os) == [Nesterov, ShieldedOpt{Descent}, Descent]
        end

        @testset "MutationChain and LogMutation" begin
            m = MutationChain(LogMutation(o -> "First", OptimizerMutation((Momentum, ))), LogMutation(o -> "Second", AddOptimizerMutation(o -> Descent())))

            @test_logs (:info, "First") (:info, "Second") typeof.(m(Nesterov()).os) == [Momentum, Descent]
            @test_logs (:info, "First") (:info, "First") (:info, "Second") (:info, "Second") m([Nesterov(), ADAM()])
        end
    end

end
