

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
        batchnorm(inpt) = mutable(BatchNorm(nout(inpt)), inpt)

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

        @testset "NeuronSelect" begin
            m1 = MutationProbability(NeuronSelectMutation(oddfirst, NoutMutation(0.5, MockRng([1]))), Probability(0.2, MockRng([0.1, 0.3])))
            m2 = MutationProbability(NeuronSelectMutation(oddfirst, NoutMutation(-0.5, MockRng([0]))), Probability(0.2, MockRng([0.3, 0.1])))
            m = VertexMutation(MutationList(m1, m2))

            inpt = inputvertex("in", 2, FluxDense())
            v3 = dense(inpt, 3,5,7)
            g = CompGraph(inpt, v3)

            m(g)
            NeuronSelect()(m, g)
            vs = vertices(g)[2:end]

            @test [out_inds(op(vs[1]))] == in_inds(op(vs[2])) == [[1,2,3,-1]]
            @test [out_inds(op(vs[2]))] == in_inds(op(vs[3])) == [[1,3,5]]
            @test out_inds(op(vs[3])) == [1,2,3,4,5,6,7,-1,-1,-1]
        end

        @testset "NeuronSelectMutation deep transparent" begin

            sizeinfo(v::AbstractVertex, offs=0, dd=Dict()) = sizeinfo(trait(v), v, offs, dd)
            sizeinfo(t::DecoratingTrait, v, offs, dd) = sizeinfo(base(t), v, offs, dd)
            function sizeinfo(::SizeStack, v, offs, dd)
                for vin in inputs(v)
                    sizeinfo(vin, offs, dd)
                    offs += nout_org(op(vin))
                end
                return dd
            end
            function sizeinfo(::SizeInvariant, v, offs, dd)
                foreach(vin -> sizeinfo(vin, offs, dd), inputs(v))
                return dd
            end
            function sizeinfo(::SizeAbsorb, v, offs, dd)
                orgsize = nout_org(op(v))
                selectfrom = hcat(get(() -> zeros(Int, orgsize, 0), dd, v), (1:orgsize) .+ offs)
                dd[v] = selectfrom
            end

            inpt = inputvertex("in", 3, FluxDense())
            v1 = dense(inpt, 8)
            v2 = dense(inpt, 4)
            v3 = concat(v1,v2)
            pa1 = batchnorm(v3)
            pb1 = batchnorm(v3)
            pc1 = batchnorm(v3)
            pd1 = dense(v3, 5)
            pa1pa1 = batchnorm(pa1)
            pa1pb1 = batchnorm(pa1)
            pa2 = concat(pa1pa1, pa1pb1)
            v4 = concat(pa2, pb1, pc1, pd1)

            g = CompGraph(inpt, v4)
            @test size(g(ones(3,2))) == (nout(v4), 2)

            @test minΔnoutfactor(v4) == 4
            Δnout(v4, -8)
            noutv4 = nout(v4)

            @test nout(v1) == 7
            @test nout(v2) == 4
            @test nout(pd1) == 1

            #m = NeuronSelectMutation(v -> collect(1:nout(v)), NoutMutation(0.5))
            #push!(m.m.mutated, v4)
            #select(m)
            dd = sizeinfo(v4)
            sel = sort(vcat(vec(dd[v1][1:7,:]), vec(dd[v2][1:nout(v2),:]), vec(dd[pd1][nout(pd1),:])))
            Δnout(v4, sel)

            apply_mutation(g)

            @test nout(v1) == 7
            @test nout(v2) == 4
            @test nout(pd1) == 1

            @test size(g(ones(3,2))) == (noutv4, 2)

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
            p3 = concat(p2pa2,p2pb2, traitdecoration=named("$(bname)3"))
            return mutable("$(bname)4", Dense(nout(p3), 4), p3)
        end

        vert(want::String, graph::CompGraph) = vertices(graph)[name.(vertices(graph)) .== want][]
        vert(want::String, v::AbstractVertex) = flatten(v)[name.(flatten(v)) .== want][]

        for to_rm in ["pa4", "pa2pa2"]
            for vconc_out in [nothing, "pa1", "pa2pa2", "pa3", "pa4"]
                for vadd_in in [nothing, v0, v1, v2]

                    #Path to remove
                    pa = path("pa",vadd_in)
                    #Other path
                    pb = path("pb")

                    v3 = concat(pa,pb, traitdecoration = named("v3"))
                    v4 = mutable("v4", Dense(nout(v3), 6), v3)
                    if !isnothing(vconc_out)
                        v4 = concat(v4, vert(vconc_out, v4), traitdecoration = named("v4$(vconc_out)_conc"))
                    end
                    v5 = mutable("v5", Dense(nout(v4), 5), v4)

                    g = copy(CompGraph(inpt, v5))
                    @test size(g(ones(4,2))) == (5,2)

                    to_remove = vert(to_rm, g)
                    Δnout(to_remove, -nout(to_remove))

                    RemoveZeroNout()(g)
                    apply_mutation(g)
                    @test !in(to_remove, vertices(g))
                    @test size(g(ones(4,2))) == (5,2)
                end
            end
        end
    end

end
