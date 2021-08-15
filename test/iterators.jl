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
        import IterTools: ncycle
        bitr = RepeatPartitionIterator(ncycle(Iterators.partition(1:20, 5), 3), 2)

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
    testitr = SeedIterator(Iterators.map(x -> x * rand(rng, Int), ones(10)); rng=rng, seed=12)
    @test collect(testitr) == collect(testitr)

    rng = MersenneTwister(1234)
    nesteditr = SeedIterator(Iterators.map(x -> x * rand(rng, Int), testitr); rng=rng, seed=1)
    @test collect(nesteditr) == collect(nesteditr)
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

    @testset "Single array" begin
        itr = BatchIterator(collect(reshape(1:2*3*4*5,2,3,4,5)), 2)
        for (i, batch) in enumerate(itr)
            @test size(batch) == (2,3,4,i==3 ? 1 : 2)
        end

        @test "biter: $itr" == "biter: BatchIterator(size=(2, 3, 4, 5), batchsize=2, shuffle=false)"
    end

    @testset "Tuple data shuffle=$shuffle" for shuffle in (true, false)
        itr = BatchIterator((collect([1:10 21:30]'), 110:10:200), 3; shuffle)
        for (i, (x, y)) in enumerate(itr)
            expsize = i == 4 ? 1 : 3
            @test size(x) == (2, expsize)
            @test size(y) == (expsize,)
        end
    end

    @testset "BatchIterator singleton" begin
        itr = BatchIterator(Singleton([1,3,5,7,9,11]), 2)
        for (i, b) in enumerate(itr)
            @test b == [1,3] .+ 4(i-1)
        end
    end

    @testset "BatchIterator shuffle basic" begin
        @test reduce(vcat, BatchIterator(1:20, 3; shuffle=true)) |> sort == 1:20

        itr = BatchIterator(ones(2,3,4), 4; shuffle=MersenneTwister(2))
        @test "siter: $itr" == "siter: BatchIterator(size=(2, 3, 4), batchsize=4, shuffle=true)"
    end

    @testset "BatchIterator shuffle ndims $(length(dims))" for dims in ((5), (3,4), (2,3,4), (2,3,4,5), (2,3,4,5,6), (2,3,4,5,6,7))
        sitr = BatchIterator(collect(reshape(1:prod(dims),dims...)), 2;shuffle=MersenneTwister(12))
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
end

@testset "RepeatPartitionIterator and ShuffleIterator" begin
    import IterTools: ncycle

    @testset "Single epoch small" begin
        ritr = RepeatPartitionIterator(BatchIterator(1:20, 3; shuffle=MersenneTwister(123)), 4)
        for itr in ritr
            @test collect(itr) == collect(itr)
        end
    end

    @testset "Multi epoch small" begin
        sitr = BatchIterator(1:20, 3;shuffle=MersenneTwister(123))
        citr = ncycle(sitr, 2)
        ritr = RepeatPartitionIterator(SeedIterator(citr; rng=sitr.rng), 4)
        for itr in ritr
            @test collect(itr) == collect(itr)
        end
    end

    @testset "Multi epoch big" begin
        sitr = BatchIterator(1:20, 3;shuffle= MersenneTwister(123))
        citr = ncycle(sitr, 4)
        ritr = RepeatPartitionIterator(SeedIterator(citr; rng=sitr.rng), 10)
        for (i, itr) in enumerate(ritr)
            @test collect(itr) == collect(itr)
        end
    end
end

@testset "StatefulGenerationIter" begin
    import NaiveGAflux: itergeneration, StatefulGenerationIter
    ritr = RepeatPartitionIterator(BatchIterator(1:20, 3), 4)
    sitr = RepeatPartitionIterator(BatchIterator(1:20, 3), 4) |> StatefulGenerationIter
    for (i, itr) in enumerate(ritr)
        @test collect(itr) == collect(itergeneration(sitr, i))
    end
end

@testset "TimedIterator" begin

    @testset "No stopping accumulate = $acc" for (acc, exp) in (
        (true, 7),
        (false, 0)
    )
        timeoutcnt = 0

        titer = TimedIterator(;timelimit=0.1, patience=2, timeoutaction = () -> timeoutcnt += 1, accumulate_timeouts=acc, base=1:10)

        @test collect(titer) == 1:10
        @test timeoutcnt === 0 # Or else we'll have flakey tests...

        for i in titer
            if iseven(i)
                sleep(0.11) # Does not matter here if overloaded CI VM takes longer than this to get back to us
            end
        end
        # When accumulating timeouts: after 1,2,3,4 our patience is up, call timeoutaction for 4,5,6,7,8,9,10
        # When not accumulating: We never reach patience level 
        @test timeoutcnt == exp 
    end

    @testset "Stop iteration at timeout" begin
        # also test that we really timeout when not accumulating here
        titer = TimedIterator(;timelimit=0.1, patience=4, timeoutaction = () -> TimedIteratorStop, accumulate_timeouts=false, base=1:10)

        last = 0
        for i in titer
            last = i
            if i > 2
                sleep(0.11) # Does not matter here if overloaded CI VM takes longer than this to get back to us
            end
        end
        @test last === 6 # Sleep after 2, then 4 patience
    end

end
