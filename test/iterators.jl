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

@testset "SeedIterator" begin
    rng = MersenneTwister(123)
    testitr = SeedIterator(MapIterator(x -> x * rand(rng, Int), ones(10)); rng=rng, seed=12)
    @test collect(testitr) == collect(testitr)

    rng = MersenneTwister(1234)
    nesteditr = SeedIterator(MapIterator(x -> x * rand(rng, Int), testitr); rng=rng, seed=1)
    @test collect(nesteditr) == collect(nesteditr)
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

@testset "BatchIterator singleton" begin
    itr = BatchIterator(Singleton([1,3,5,7,9,11]), 2)
    for (i, b) in enumerate(itr)
        @test b == [1,3] .+ 4(i-1)
    end
end

@testset "ShuffleIterator ndims $(length(dims))" for dims in ((5), (3,4), (2,3,4), (2,3,4,5), (2,3,4,5,6), (2,3,4,5,6,7))
    sitr = ShuffleIterator(collect(reshape(1:prod(dims),dims...)), 2, MersenneTwister(123))
    bitr = BatchIterator(collect(reshape(1:prod(dims),dims...)), 2)
    sall, nall = Set{Int}(), Set{Int}()
    for (sb, nb) in zip(sitr, bitr)
        @test sb != nb
        @test size(sb) == size(nb)
        push!(sall, sb...)
        push!(nall, nb...)
    end
    @test sall == nall
end

@testset "ShuffleIterator singleton" begin
    itr = ShuffleIterator(Singleton([1,3,5,7,9,11]), 2, MersenneTwister(123))
    for b in itr
        @test length(b) == 2
    end
    @test sort(vcat(collect(itr)...)) == [1,3,5,7,9,11]
end
