@testset "TrainBatchSizeMutation" begin
    import NaiveGAflux: batchsize

    @testset "Quantize to Int" begin

        @testset "Forced to 10" begin
            bsim = BatchSizeIteratorMap(100, 200, batchsizeselection((3,)))
            m = TrainBatchSizeMutation(0.1, 0.1)
            @test batchsize(m(bsim).tbs) == 110
        end

        @testset "Forced to -10" begin
            bsim = BatchSizeIteratorMap(100, 200, batchsizeselection((3,)))
            m = TrainBatchSizeMutation(-0.1, -0.1)
            @test batchsize(m(bsim).tbs) == 90
        end

        @testset "Larger than 0" begin
            bsim = BatchSizeIteratorMap(1, 200, batchsizeselection((3,)))
            m = TrainBatchSizeMutation(-0.9, -0.9)
            @test batchsize(m(bsim).tbs) == 1
        end

        @testset "Random" begin
            bsim = BatchSizeIteratorMap(100, 200, batchsizeselection((3,)))
            rng = MockRng([0.5])
            m = TrainBatchSizeMutation(0.0, 1.0, rng)
            @test batchsize(m(bsim).tbs) == 150
        end
    end

    @testset "Quantize to set of numbers" begin  
        @testset "Force one step up" begin
            bsim = BatchSizeIteratorMap(3, 200, batchsizeselection((3,)))
            m = TrainBatchSizeMutation(0.1, 0.1, 1:10)
            @test batchsize(m(bsim).tbs) == 4
        end

        @testset "Force one step down" begin
            bsim = BatchSizeIteratorMap(3, 200, batchsizeselection((3,)))
            m = TrainBatchSizeMutation(-0.1, -0.1, 1:10)
            @test batchsize(m(bsim).tbs) == 2
        end

        @testset "Force two steps up" begin
            bsim = BatchSizeIteratorMap(3, 200, batchsizeselection((3,)))
            m = TrainBatchSizeMutation(0.2, 0.2, 1:10)
            @test batchsize(m(bsim).tbs) == 5
        end

        @testset "Random" begin
            bsim = BatchSizeIteratorMap(128, 200, batchsizeselection((3,)))
            rng = MockRng([0.5])
            m = TrainBatchSizeMutation(0.0, -1.0, ntuple(i -> 2^i, 10), rng)
            @test batchsize(m(bsim).tbs) == 4        
        end
    end

    @testset "Shielded" begin
        sim = ShieldedIteratorMap(BatchSizeIteratorMap(100, 200, batchsizeselection((3,))))
        m = TrainBatchSizeMutation(0.1, 0.1)
        @test batchsize(m(sim).map.tbs) == 100
    end

    @testset "IteratorMaps" begin
        im = IteratorMaps(BatchSizeIteratorMap(100, 200, batchsizeselection((3,))), BatchSizeIteratorMap(100, 200, batchsizeselection((3,))))
        m = TrainBatchSizeMutation(0.1, 0.1)
        @test batchsize(m(im).maps[1].tbs) == 110
        @test batchsize(m(im).maps[2].tbs) == 110
    end
end