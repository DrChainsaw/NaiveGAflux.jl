@testset "RepeatPartitionIterator" begin

    @testset "RepeatPartitionIterator basic" begin

        bitr = RepeatPartitionIterator(1:20, 5)

        for (itr, exp) in zip(bitr, [1:5, 6:10, 11:15, 16:20])
            @test collect(itr) == exp
            @test collect(itr) == exp
            @test collect(itr) == exp
        end
    end

    @testset "RepeatPartitionIterator partition" begin

        bitr = RepeatPartitionIterator(Iterators.partition(1:20, 5), 2)

        for (itr, exp) in zip(bitr, [[1:5, 6:10], [11:15, 16:20]])
            @test collect(itr) == exp
            @test collect(itr) == exp
            @test collect(itr) == exp
        end
    end

    @testset "RepeatPartitionIterator repeated partition" begin
        bitr = RepeatPartitionIterator(Iterators.cycle(Iterators.partition(1:20, 5), 3), 2)

        cnt = 0;
        for (itr, exp) in zip(bitr, [[1:5, 6:10], [11:15, 16:20],[1:5, 6:10], [11:15, 16:20],[1:5, 6:10], [11:15, 16:20]])
            @test collect(itr) == exp
            @test collect(itr) == exp
            @test collect(itr) == exp
            cnt += 1
        end
    end
end

@testset "MapIterator" begin
    itr = MapIterator(x -> 2x, [1,2,3,4,5])
    @test collect(itr) == [2,4,6,8,10]
end

@testset "GpuIterator" begin
    dorg = 1:100;
    itr = GpuIterator(zip([view(dorg,1:10)], [dorg]))
    d1,d2 = first(itr) |> cpu
    @test !isa(d1, SubArray)
    @test d1 == dorg[1:10]
    @test d2 == dorg
end

@testset "BatchIterator" begin
    itr = BatchIterator(collect(reshape(1:2*3*4*5,2,3,4,5)), 2)
    for (i, batch) in enumerate(itr)
        @test size(batch) == (2,3,4,i==3 ? 1 : 2)
    end

    @test "biter: $itr" == "biter: BatchIterator(size=(2, 3, 4, 5), batchsize=2)"
end

@testset "FlipIterator" begin
    itr = FlipIterator([[1 2 3 4; 5 6 7 8], [1 2; 3 4]], 1.0, 2)
    for (act, exp) in zip(itr, [[4 3 2 1; 8 7 6 5], [2 1; 4 3]])
        @test act == exp
    end

    itr = FlipIterator([[1 2 3 4; 5 6 7 8]], 0.0, 2)
    @test first(itr) == [1 2 3 4; 5 6 7 8]
end

@testset "ShiftIterator" begin
    itr = ShiftIterator([[1 2 3 4; 5 6 7 8], [5 6 7 8; 1 2 3 4]], 0, 2, rng=SeqRng(0))

    for (act, exp) in zip(itr, [[0 1 2 3; 0 5 6 7], [0 5 6 7; 0 1 2 3]])
        @test act == exp
    end
end
