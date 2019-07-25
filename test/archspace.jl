@testset "Architecture Spaces" begin

    mutable struct SeqRng
        ind
        SeqRng(ind) = new(ind)
        SeqRng() = new(0)
    end
    function Random.rand(rng::SeqRng, vec)
        rng.ind = rng.ind % length(vec) + 1
        return vec[rng.ind]
    end

    @testset "BasicLayerSpace" begin
        rng = SeqRng()
        space = BaseLayerSpace([1,2,3], [relu,σ,identity])
        @test space(rng) == (1, σ)
        @test space(rng) == (3, relu)
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

    @testset "DenseSpace" begin
        rng = SeqRng()
        space = DenseSpace(BaseLayerSpace(3, σ))
        l = space(2, rng)
        @test l.σ == σ
        @test size(l.W) == (3,2)
    end

    @testset "ConvSpace" begin
        rng = SeqRng()
        space = ConvSpace2D(BaseLayerSpace(5, relu), 2:5)
        l = space(4, rng)
        @test size(l.weight) == (2,3,4,5)
        @test size(l(ones(5,5,4,1))) == (5,5,5,1)

        rng.ind = 0
        space = ConvSpace(BaseLayerSpace(4, elu), 2:5)
        l = space(3, rng)
        @test size(l.weight) == (2,3,4)
        @test size(l(ones(5,3,1))) == (5,4,1)

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
        space = VertexSpace(DenseSpace(BaseLayerSpace(3,relu)))
        inpt = inputvertex("in", 2)
        v = space(inpt)
        @test nin(v) == [2]
        @test nout(v) == 3

        v = space("v", inpt)
        @test name(v) == "v"
    end

    @testset "ArchSpace" begin
        bs = BaseLayerSpace(3, elu)
        space = ArchSpace(DenseSpace(bs))
        inpt = inputvertex("in", 2)
        v = space(inpt)
        @test nin(v) == [2]
        @test nout(v) == 3

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
        @test name.(flatten(v)) == ["in", "test.path1", "test.path2", "test.path3", "test"]

        space = ForkArchSpace(VertexSpace(BatchNormSpace(relu)), [2,5])
        rng = SeqRng()

        v = space(inpt, rng)
        @test length(inputs(v)) == 2

        v = space(inpt, rng)
        @test length(inputs(v)) == 5
    end

end
