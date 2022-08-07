@testset "Iterator maps" begin
    import NaiveGAflux: maptrain, mapvalidation, limit_maxbatchsize

    @testset "BatchSizeIteratorMap" begin
        function testgraph(insize)
            v0 = denseinputvertex("v0", insize)
            v1 = fluxvertex("v1", Dense(nout(v0) => 5), v0)
            v2 = fluxvertex("v2", Dense(nout(v1) => 2), v1)
            v3 = concat("v3", v1, v2)
            CompGraph(v0, "v4" >> v3 + v3)
        end

        bsim = BatchSizeIteratorMap(2, 4, batchsizeselection((5,); batchsizefun=(bs, newbs; scale, kws...) -> newbs * scale))

        @testset "Single array" begin
            @test collect(maptrain(bsim, (1:20,))) == [a:a+1 for a in 1:2:20]    
            @test collect(mapvalidation(bsim, (1:20,))) == [a:a+3 for a in 1:4:20]   
            
            bsimnew = limit_maxbatchsize(bsim, 2; scale=5)
            @test collect(maptrain(bsimnew, (1:20,))) == [a:a+9 for a in 1:10:20] 
            @test collect(mapvalidation(bsimnew, (1:20,))) == [a:a+9 for a in 1:10:20] 
        end

        @testset "BatchIterator" begin
            itr = maptrain(bsim, BatchIterator((1:6, 7:12), 4))
            @testset "Iteration $i" for (i, (res, exp)) in enumerate(zip(itr, (([a, a+1], [a+6,a+7]) for a in 1:2:6)))
                @test res == exp
            end
        end
    end

    @testset "IteratorMaps" begin
        NaiveGAflux.maptrain(::Val{:TestDummy1}, itr) = Iterators.map(x -> 2x, itr)
        NaiveGAflux.maptrain(::Val{:TestDummy2}, itr) = Iterators.map(x -> 3x, itr)

        NaiveGAflux.mapvalidation(::Val{:TestDummy1}, itr) = Iterators.map(x -> 5x, itr)
        NaiveGAflux.mapvalidation(::Val{:TestDummy2}, itr) = Iterators.map(x -> 7x, itr)
        
        td1 = Val(:TestDummy1)
        td2 = Val(:TestDummy2)

        @test collect(maptrain(IteratorMaps(td1), 1:3)) == 2:2:6
        @test collect(mapvalidation(IteratorMaps(td1), 1:3)) == 5:5:15

        @test collect(maptrain(IteratorMaps(td1, td2), 1:3)) == 6:6:18
        @test collect(mapvalidation(IteratorMaps(td1, td2), 1:3)) == 35:35:105
    end

    @testset "ShieldedIteratorMap" begin
        NaiveGAflux.maptrain(::Val{:TestDummy1}, itr) = Iterators.map(x -> 2x, itr)
        NaiveGAflux.mapvalidation(::Val{:TestDummy1}, itr) = Iterators.map(x -> 5x, itr)
        NaiveGAflux.limit_maxbatchsize(::Val{:TestDummy1}, args...; kwargs...) = Val(:TestDummy2)

        sim = ShieldedIteratorMap(Val(:TestDummy1))

        @test collect(maptrain(sim, 1:3)) == 2:2:6
        @test collect(mapvalidation(sim, 1:3)) == 5:5:15
        @test limit_maxbatchsize(sim, 13; blah=14) == ShieldedIteratorMap(Val(:TestDummy2))
    end
end