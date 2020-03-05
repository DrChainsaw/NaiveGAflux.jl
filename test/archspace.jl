@testset "Architecture Spaces" begin

    @testset "BasicLayerSpace" begin
        import NaiveGAflux: outsize, activation
        rng = SeqRng()
        space = BaseLayerSpace([1,2,3], [relu,σ,identity])
        @test outsize(space, rng) == 1
        @test activation(space, rng) == σ
        @test outsize(space, rng) == 3
        @test activation(space, rng) == relu
    end

    @testset "SingletonParSpace" begin
        @test SingletonParSpace(666)() == 666
        for d in 2:4
            @test SingletonParSpace(1:d...)() == Tuple(1:d)
        end
        @test Singleton2DParSpace(7)() == (7,7)
    end

    @testset "ParSpace" begin
        rng = SeqRng()
        space = ParSpace(1:3)
        @test [space(rng) for _ in 1:4]== [1,2,3,1]

        rng.ind = 0
        space = ParSpace2D(2:4)
        @test space(rng) == (2, 3)
        @test space(rng) == (4, 2)

        rng.ind = 0
        space = ParSpace(1:4, 5:7, 8:9)
        @test space(rng) == (1, 6, 8)
        @test space(rng) == (2, 7, 9)
    end

    @testset "SamePad" begin
        @test SamePad()(2, 1) == (1, 0)
        @test SamePad()(3, 1) == (1, 1)
        @test SamePad()(5, 2) == (4, 4)
        @test SamePad()((4,6), 2) == (3, 3, 5, 5)
        @test SamePad()((3,7), (2,4)) == (2,2,12,12)
    end

    @testset "NamedLayerSpace" begin
        rng = SeqRng()
        s1 = DenseSpace(2, identity)
        s2 = NamedLayerSpace("test", s1)

        @test name(s1) == ""
        @test name(s2) == "test"

        l1 = s1(3, rng)
        l2 = s2(3, rng)

        @test l1.σ == l2.σ
        @test size(l1.W) == size(l2.W)

        l1 = s1(3, rng, outsize = 4)
        l2 = s2(3, rng, outsize = 4)
        @test size(l1.W) == size(l2.W)
    end

    @testset "LoggingLayerSpace" begin
        rng = SeqRng()
        s1 = DenseSpace(2, identity)
        s2 = LoggingLayerSpace(NamedLayerSpace("test", s1))

        @test name(s2) == "test"

        l1 = s1(3, rng)
        l2 = (@test_logs (:debug, "Create Dense(3, 2) from test") min_level=Logging.Debug s2(3,rng))

        @test l1.σ == l2.σ
        @test size(l1.W) == size(l2.W)

        l1 = s1(3, rng, outsize = 4)
        l2 = (@test_logs s2(3,rng, outsize = 4))

        @test size(l1.W) == size(l2.W)
    end

    @testset "DenseSpace" begin
        rng = SeqRng()
        space = DenseSpace(3, σ)
        l = space(2, rng)
        @test l.σ == σ
        @test size(l.W) == (3,2)
        l = space(3, rng,outsize=2)
        @test l.σ == σ
        @test size(l.W) == (2,3)
    end

    @testset "ConvSpace" begin
        rng = SeqRng()
        space = ConvSpace2D(5, relu, 2:5)
        l = space(4, rng)
        @test size(l.weight) == (2,3,4,5)
        @test size(l(ones(5,5,4,1))) == (5,5,5,1)

        rng.ind = 0
        space = ConvSpace(4, elu, 2:5)
        l = space(3, rng)
        @test size(l.weight) == (2,3,4)
        @test size(l(ones(5,3,1))) == (5,4,1)

        l = space(4, rng, outsize=3)
        @test size(l.weight) == (3,4,3)
        @test size(l(ones(5,4,1))) == (5,3,1)
    end

    @testset "BatchNormSpace" begin
        space = BatchNormSpace(relu)
        @test space(2).λ == relu
        @test space(3).λ == relu

        rng = SeqRng()
        space = BatchNormSpace(relu,elu,identity)
        @test space(2,rng).λ == relu
        @test space(3,rng).λ == elu
        @test space(4,rng).λ == identity

        rng = SeqRng()
        space = BatchNormSpace([relu,elu,identity])
        @test space(2,rng).λ == relu
        @test space(3,rng).λ == elu
        @test space(4,rng).λ == identity
    end

    @testset "MaxPoolSpace" begin
        rng = SeqRng()
        space = MaxPoolSpace(PoolSpace(1:3))
        @test space(2, rng).k == (1,)
        @test space(2, rng).k == (3,)
        @test space(2, rng).k == (2,)

        space = MaxPoolSpace(PoolSpace2D(1:3))
        @test space(2, rng).k == (1,2)
        @test space(2, rng).k == (2,3)
    end

    @testset "VertexSpace" begin
        space = VertexSpace(DenseSpace(3,relu))
        inpt = inputvertex("in", 2)
        v = space(inpt)
        @test nin(v) == [2]
        @test nout(v) == 3

        v = space("v", inpt)
        @test name(v) == "v"

        space = VertexSpace(NamedLayerSpace("dense", DenseSpace(3,relu)))
        v = space("v", inpt)
        @test name(v) == "v.dense"
    end

    @testset "LoggingArchSpace" begin
        space = LoggingArchSpace(VertexSpace(NamedLayerSpace("dense", DenseSpace(3, identity))))
        inpt = inputvertex("in", 2)

        v = (@test_logs (:debug, "Created test.dense") min_level=Logging.Debug space("test", inpt))
        @test nin(v) == [2]
        @test nout(v) == 3

        v = (@test_logs space(inpt))
        @test nin(v) == [2]
        @test nout(v) == 3
    end

    @testset "ArchSpace" begin
        bs = BaseLayerSpace(3, elu)
        space = ArchSpace(DenseSpace(bs))
        inpt = inputvertex("in", 2)
        v = space(inpt)
        @test nin(v) == [2]
        @test nout(v) == 3

        v = space(inpt, outsize = 4)
        @test nin(v) == [2]
        @test nout(v) == 4

        v = space("v", inpt, outsize = 4)
        @test name(v) == "v"
        @test nin(v) == [2]
        @test nout(v) == 4

        rng = SeqRng()
        space = ArchSpace(ConvSpace2D(bs, 2:5), BatchNormSpace(relu), MaxPoolSpace(PoolSpace2D([2])))

        v = space("conv", inpt, rng)
        @test layertype(v) == FluxConv{2}()
        @test nin(v) == [2]
        @test nout(v) == 3
        @test name(v) == "conv"

        rng.ind = 1
        v = space("bn", inpt, rng)
        @test layertype(v) == FluxBatchNorm()
        @test nin(v) == [2]
        @test nout(v) == 2
        @test name(v) == "bn"
    end

    @testset "RepeatArchSpace" begin
        space = RepeatArchSpace(VertexSpace(BatchNormSpace(relu)), 3)
        inpt = inputvertex("in", 3)
        v = space(inpt)
        @test nv(CompGraph(inpt, v)) == 4

        v = space("test", inpt)
        @test name.(flatten(v)) == ["in", "test.1", "test.2", "test.3"]

        space = RepeatArchSpace(VertexSpace(BatchNormSpace(relu)), [2,5])
        rng = SeqRng()
        v = space(inpt, rng)
        @test nv(CompGraph(inpt, v)) == 3

        v = space(inpt, rng)
        @test nv(CompGraph(inpt, v)) == 6

        space = RepeatArchSpace(VertexSpace(DenseSpace(3, relu)), 2)
        v = space(inpt, outsize=4)
        @test nout(v) == 4
        @test nin(v) == [4]

        v = space("test", inpt, outsize=4)
        @test name.(flatten(v)) == ["in", "test.1", "test.2"]
        @test nout(v) == 4
        @test nin(v) == [4]

        space = RepeatArchSpace(VertexSpace(BatchNormSpace(relu)), 0)
        @test space(inpt) == inpt
    end

    @testset "ListArchSpace" begin
        space = ListArchSpace(VertexSpace.(DenseSpace.((2,3), relu))...)
        inpt = inputvertex("in", 3)

        v = space(inpt)
        @test nv(CompGraph(inpt, v)) == 3
        @test nout(v) == 3
        @test nin(v) == [2]

        v = space("v", inpt)
        @test name.(flatten(v)) == ["in", "v.1", "v.2"]
        @test nv(CompGraph(inpt, v)) == 3
        @test nout(v) == 3
        @test nin(v) == [2]

        v = space(inpt, outsize=4)
        @test nv(CompGraph(inpt, v)) == 3
        @test nout(v) == 4
        @test nin(v) == [4]

        v = space("v", inpt, outsize=4)
        @test name.(flatten(v)) == ["in", "v.1", "v.2"]
        @test nv(CompGraph(inpt, v)) == 3
        @test nout(v) == 4
        @test nin(v) == [4]
    end

    @testset "ForkArchSpace" begin
        # No concatenation when only one path is rolled
        space = ForkArchSpace(VertexSpace(BatchNormSpace(relu)), 1)
        inpt = inputvertex("in", 3, FluxDense())
        v = space(inpt)
        @test inputs(v) == [inpt]

        space = ForkArchSpace(VertexSpace(BatchNormSpace(relu)), 3)
        v = space(inpt)
        @test length(inputs(v)) == 3

        v = space("test", inpt)
        @test name.(flatten(v)) == ["in", "test.path1", "test.path2", "test.path3", "test.cat"]

        space = ForkArchSpace(VertexSpace(BatchNormSpace(relu)), [2,5])
        rng = SeqRng()

        v = space(inpt, rng)
        @test length(inputs(v)) == 2

        v = space(inpt, rng)
        @test length(inputs(v)) == 5

        space = ForkArchSpace(VertexSpace(DenseSpace(3, relu)), 3)
        v = space(inpt, outsize=13)
        @test nout(v) == 13
        @test nin(v) == [4, 4, 5]

        v = space("v", inpt, outsize=13)
        @test name(v) == "v.cat"
        @test nout(v) == 13
        @test nin(v) == [4, 4, 5]

        # Edge case: Smaller outsize than number of paths
        v = space(inpt, outsize=2)
        @test nout(v) == 2
        @test nin(v) == [1, 1]

        v = space("v", inpt, outsize=2)
        @test nout(v) == 2
        @test nin(v) == [1, 1]

        space = ForkArchSpace(VertexSpace(DenseSpace(3, relu)), 0)
        @test space(inpt) == inpt
    end

    @testset "ResidualArchSpace" begin
        space = ResidualArchSpace(DenseSpace(3, relu))
        inpt = inputvertex("in", 4)

        v = space(inpt)
        @test nin(v) == [4, 4]
        @test nout(v) == 4
        @test layertype(inputs(v)[2]) == FluxDense()

        v = space("v", inpt)
        @test name.(flatten(v)) == ["in", "v.res", "v.add"]
        @test nin(v) == [4, 4]
        @test nout(v) == 4
        @test layertype(inputs(v)[2]) == FluxDense()

    end

    @testset "FunctionSpace" begin
        space = FunctionSpace(x -> 2 .* x; namesuff=".fun")
        inpt = inputvertex("in", 3)

        v = space(inpt)
        @test nin(v) == [3]
        @test nout(v) == 3
        @test v([1 2 3]) == [2 4 6]

        v = space("v", inpt)
        @test name(v) == "v.fun"
        @test nin(v) == [3]
        @test nout(v) == 3
        @test v([1 2 3]) == [2 4 6]

        space = GlobalPoolSpace2D()
        v = space("v", inpt)

        @test size(v(ones(Float32, 3,4,5,6))) == (5,6)
    end

    @testset "WeightInit" begin

        @testset "DenseSpace" begin
            space = DenseSpace(3, identity)
            indata = reshape(1:2*4, :, 2)
            insize = size(indata, 1)

            @testset "IdentityWeightInit" begin
                v = space(insize, outsize=insize, wi=IdentityWeightInit())
                @test v(indata) == indata
            end

            @testset "ZeroWeightInit" begin
                v = space(insize, wi=ZeroWeightInit())
                @test v(indata) == zeros(nout(v), size(indata,2))
            end
        end

        @testset "ConvSpace2D" begin
            space = ConvSpace2D(3, identity, [3])
            indata = reshape(1:2*4*5*6,4,5,6,2)
            insize = size(indata, 3)

            @testset "IdentityWeightInit" begin
                v = space(insize, outsize=insize, wi=IdentityWeightInit())
                @test v(indata) == indata
            end

            @testset "ZeroWeightInit" begin
                v = space(insize, wi=ZeroWeightInit())
                @test v(indata) == zeros(size(indata,1), size(indata,2), nout(v), size(indata,4))
            end
        end

        function test_identity_dense(space, inpt = inputvertex("in", 3))
            v = space(inpt, outsize=nout(inpt), wi=IdentityWeightInit())
            g = CompGraph(inpt, v)

            indata = reshape(1:2*nout(inpt), :, 2)
            @test g(indata) == indata
        end

        function test_identity_conv(space, inpt = inputvertex("in", 3))
            v = space(inpt, outsize=nout(inpt), wi=IdentityWeightInit())
            g = CompGraph(inpt, v)

            indata = reshape(1:nout(inpt)*4*5, 5,4,nout(inpt), 1)
            @test g(indata) == indata
        end

        @testset "RepeatArchSpace identity" begin
            test_identity_dense(RepeatArchSpace(VertexSpace(DenseSpace(3, identity)), 3))
        end

        @testset "ListArchSpace identity" begin
            test_identity_dense(ListArchSpace(VertexSpace.(DenseSpace.((2,3), relu))...))
        end

        @testset "ForkArchSpace identity $npaths paths" for npaths in (1, 2, 3)
            test_identity_dense(ForkArchSpace(VertexSpace(DenseSpace(3, identity)), npaths), inputvertex("in", npaths*2+1))
            test_identity_conv(ForkArchSpace(VertexSpace(ConvSpace2D(3, identity, [3])), npaths), inputvertex("in", npaths*2+1))
        end

        @testset "ForkArchSpace RepeatArchSpace identity $nreps repetitions" for nreps in (1,2,3)
            test_identity_dense(ForkArchSpace(RepeatArchSpace(VertexSpace(DenseSpace(3, identity)), nreps), 3), inputvertex("in", 3*2+1))
            test_identity_conv(ForkArchSpace(RepeatArchSpace(VertexSpace(ConvSpace2D(3, identity, [3])), nreps), 3), inputvertex("in", 3*2+1))
        end

        @testset "ForkArchSpace ListArchSpace identity $sizes sizes" for sizes in ((3,), (2,3), (2,3,4))
            test_identity_dense(ForkArchSpace(ListArchSpace(VertexSpace.(DenseSpace.(sizes, identity))...), 3), inputvertex("in", 3*2+1))
            test_identity_conv(ForkArchSpace(ListArchSpace(VertexSpace.((MaxPoolSpace(PoolSpace2D([1])), ConvSpace2D.(sizes, identity, ([3],))...))...), 3), inputvertex("in", 3*2+1))
        end

        @testset "ResidualArchSpace identity" begin
            test_identity_dense(ResidualArchSpace(DenseSpace(3, identity)))
            test_identity_conv(ResidualArchSpace(ConvSpace2D(3, identity, [3])))
        end

        @testset "ResidualArchSpace ForkArchSpace identity $npaths paths" for npaths in (1,2,3)
            test_identity_dense(ResidualArchSpace(ForkArchSpace(VertexSpace(DenseSpace(3, identity)), npaths)))
            test_identity_conv(ResidualArchSpace(ForkArchSpace(VertexSpace(ConvSpace2D(3, identity, [3])), npaths)))
        end

        @testset "ForkArchSpace ResidualArchSpace identity" begin
            # ID mapping only possible with one path
            test_identity_dense(ForkArchSpace(ResidualArchSpace(VertexSpace(DenseSpace(3, identity))), 1))
            test_identity_conv(ForkArchSpace(ResidualArchSpace(VertexSpace(ConvSpace2D(3, identity, [3]))), 1))
        end

    end

end
