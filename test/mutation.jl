

@testset "Mutation" begin

    struct NoOpMutation{T} <:AbstractMutation{T} end
    (m::NoOpMutation)(t) = t
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

    @testset "Neuron value weighted mutation" begin
        using Statistics
        struct DummyValue <: AbstractMutableComp
            values
        end
        NaiveNASflux.neuron_value(d::DummyValue) = d.values

        l(in, outsize, value) = mutable(Dense(nout(in), outsize), in, layerfun = l -> DummyValue(value))

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

    @testset "MutationList" begin
        probes = ProbeMutation.(repeat([Int], 3))
        m = MutationList(probes...)
        @test m(1) == 1
        @test getfield.(probes, :mutated) == [[1],[1],[1]]
    end

    @testset "LogMutation" begin
        probe = ProbeMutation(Int)
        m = LogMutation(i -> "Mutate $i", probe)

        @test @test_logs (:info, "Mutate 17") m(17) == 17
        @test probe.mutated == [17]
    end

    @testset "MutationFilter" begin
        probe = ProbeMutation(Int)
        m = MutationFilter(i -> i > 3, probe)

        @test m(1) == 1
        @test probe.mutated == []

        @test m(4) == 4
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
        inpt = inputvertex("in", 3, FluxDense())
        v1 = dense(inpt, 5)
        v2 = dense(v1, 3)

        @test RemoveVertexMutation()(v1) == v1

        @test inputs(v2) == [inpt]
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

    @testset "NeuronSelectMutation" begin

        # Dummy neuron selection function just to mix things up in a predicable way
        function oddfirst(v)
            values = zeros(nout_org(v))
            nvals = length(values)
            values[1:2:nvals] = nvals:-1:(nvals ÷ 2 + 1)
            values[2:2:nvals] = (nvals ÷ 2):-1:1
            return values
        end
        batchnorm(inpt; name="bn") = mutable(name, BatchNorm(nout(inpt)), inpt)

        noutselect = NaiveGAflux.Nout(OutSelectExact())

        @testset "NeuronSelectMutation NoutMutation" begin
            inpt = inputvertex("in", 3, FluxDense())
            v1 = dense(inpt, 5)
            v2 = mutable(BatchNorm(nout(v1)), v1)
            v3 = dense(v2, 6)

            m = NeuronSelectMutation(oddfirst, noutselect, NoutMutation(-0.5, MockRng([0])))
            @test m(v2) == v2
            select(m)

            @test out_inds(op(v2)) == [1,3,5]
            @test in_inds(op(v3)) == [[1,3,5]]
            @test out_inds(op(v1)) == [1,3,5]
        end


        @testset "NeuronSelect" begin
            m1 = MutationProbability(NeuronSelectMutation(oddfirst, noutselect, NoutMutation(0.5, MockRng([1]))), Probability(0.2, MockRng([0.1, 0.3])))
            m2 = MutationProbability(NeuronSelectMutation(oddfirst, noutselect, NoutMutation(-0.5, MockRng([0]))), Probability(0.2, MockRng([0.3, 0.1])))
            m = VertexMutation(MutationList(m1, m2))

            inpt = inputvertex("in", 2, FluxDense())
            v3 = dense(inpt, 3,5,7)
            g = CompGraph(inpt, v3)

            @test m(g) == g
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

            rankfun(v) = 1:nout_org(v)
            m = NeuronSelectMutation(rankfun , NoutMutation(0.5))
            push!(m.m.mutated, v4)

            g = CompGraph(inpt, v4)
            @test size(g(ones(Float32, 3,2))) == (nout(v4), 2)

            @test minΔnoutfactor(v4) == 4
            Δnout(v4, -8)
            noutv4 = nout(v4)

            @test nout(v1) == 6
            @test nout(v2) == 4
            @test nout(pd1) == 5

            select(m)
            apply_mutation(g)

            @test nout(v1) == 6
            @test nout(v2) == 4
            @test nout(pd1) == 5

            @test size(g(ones(Float32, 3,2))) == (noutv4, 2)

            Δnout(v4, +8)
            Δnout(v1, -5)
            noutv4 = nout(v4)

            @test nout(v1) == 2
            @test nout(v2) == 8
            @test nout(pd1) == 5

            select(m)
            apply_mutation(g)

            @test nout(v1) == 2
            @test nout(v2) == 8
            @test nout(pd1) == 5

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

            rankfun(v) = 1:nout_org(v)
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

            rankfun(v) = 1:nout_org(v)
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
                    Δ = -nout(to_remove)
                    Δnout(NaiveNASlib.OnlyFor(), to_remove, Δ)

                    RemoveZeroNout()(g)
                    Δsize(AlignNinToNout(), vertices(g))
                    apply_mutation(g)

                    @test !in(to_remove, vertices(g))
                    @test size(g(ones(4,2))) == (5,2)
                end
            end
        end
    end

    @testset "AddEdgeMutation" begin
        import NaiveGAflux: default_mergefun
        cl(name, in, outsize) = mutable(name, Conv((1,1), nout(in)=>outsize), in)

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

            m = AddEdgeMutation(1.0, mergefun = default_mergefun(pconc), valuefun = v -> 1:nout_org(v))

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

        @testset "AddEdgeMutation fail" begin
            m = AddEdgeMutation(1.0, mergefun=default_mergefun(1), valuefun=v -> 1:nout_org(v))

            v0 = inputvertex("in", 3, FluxConv{2}())
            v1 = cl("v1", v0, 4)

            # No suitable vertx as v0 is already output to v1
            @test m(v0) == v0
            @test all_in_graph(v0) == [v0, v1]
            @test outputs(v0) == [v1]

            v2 = cl("v2", v1, 3)

            # No suitable vertx as v1 is already output to v2
            @test m(v1) == v1
            @test all_in_graph(v0) == [v0, v1, v2]
            @test outputs(v1) == [v2]

            v3 = concat("v3", v2)
            v4 = "v4" >> v0 + v3

            # Will try to add v1 as input to v3, but v4 can not change size as v0 is immutable
            @test_logs (:warn, "Selection for vertex v1 failed! Reverting...") m(v1)
            @test Set(all_in_graph(v0)) == Set([v0, v1, v2, v3, v4])
        end
    end

    @testset "RemoveEdgeMutation" begin
        dl(name, in, outsize) = mutable(name, Dense(nout(in), outsize), in)

        @testset "RemoveEdgeMutation SizeStack" begin
            m = RemoveEdgeMutation(valuefun=v->1:nout_org(v))

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
            m = RemoveEdgeMutation(valuefun=v->1:nout_org(v))

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
    end

end
