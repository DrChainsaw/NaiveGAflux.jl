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
        import NaiveGAflux: gen_nout, gen_act
        rng = SeqRng()
        space = BaseLayerSpace([1,2,3], [relu,σ,identity])
        @test [gen_nout(space, rng) for _ in 1:6] == [1,2,3,1,2,3]
        @test [gen_act(space, rng) for _ in 1:4] == [relu,σ,identity,relu]
    end

    @testset "FixedNDParSpace" begin
        @test FixedNDParSpace(666)() == 666
        for d in 2:4
            @test FixedNDParSpace(1:d...)() == Tuple(1:d)
        end
        @test Fixed2DParSpace(7)() == (7,7)
    end

    @testset "ParNDSpace" begin
        rng = SeqRng()
        space = ParNDSpace(1:3)
        @test [space(rng) for _ in 1:4]== [1,2,3,1]

        rng.ind = 0
        space = Par2DSpace(2:4)
        @test space(rng) == (2, 3)
        @test space(rng) == (4, 2)

        rng.ind = 0
        space = ParNDSpace(1:4, 5:7, 8:9)
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

    @testset "ConvSpace" begin
        rng = SeqRng()
        space = Conv2DSpace(BaseLayerSpace([5], [relu]), 2:5)
        l = space(4, rng)
        @test size(l.weight) == (2,3,4,5)
        @test size(l(ones(5,5,4,1))) == (5,5,5,1)
    end

end
