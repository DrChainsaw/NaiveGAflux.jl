@testset "BatchSizeSelection" begin
    
    @testset "BatchSizeSelectionWithDefaultInShape" begin
        testfun = function(x; inshape_nobatch)
            return x => inshape_nobatch
        end

        @test BatchSizeSelectionWithDefaultInShape((2,3,4), testfun)(13) == (13 => (2,3,4))

        @test BatchSizeSelectionWithDefaultInShape((2,3,4), testfun)(13; inshape_nobatch=(3,)) == (13 => (3,))
    end

    @testset "BatchSizeSelectionScaled" begin
        testfun = function(x; availablebytes=1)
            return x => availablebytes
        end
        @test BatchSizeSelectionScaled(0.5, testfun)(4; availablebytes=6) == (4 => 3)
    end

    @testset "BatchSizeSelectionFromAlternatives" begin
        bs = BatchSizeSelectionFromAlternatives([2, 3, 7], identity)
        @test bs(0) === 0
        @test bs(1) === 0
        @test bs(2) === 2
        @test bs(3) === 3
        @test bs(4) === 3
        @test bs(5) === 3
        @test bs(6) === 3
        @test bs(7) === 7
        @test bs(8) === 7
    end

    @testset "BatchSizeSelectionMaxSize" begin

        BatchSizeSelectionMaxSize(10, Pair)(1, 13) == 10 => 13
    end

    @testset "availablebytes" begin
        # Just a smoketest so that we e.g don't crash if CUDA.functional() is false
        @test NaiveGAflux._availablebytes() > 0
    end

    function testgraph(insize)
        v0 = denseinputvertex("v0", insize)
        v1 = fluxvertex("v1", Dense(nout(v0) => 5), v0)
        v2 = fluxvertex("v2", Dense(nout(v1) => 2), v1)
        v3 = concat("v3", v1, v2)
        CompGraph(v0, "v4" >> v3 + v3)
    end

    @testset "activationsizes" begin
        graph = testgraph(3)
        @test NaiveGAflux.activationsizes(graph, (3,)) == sum(nout, vertices(graph)) * 4
    end

    @testset "Max batch size" begin
        import NaiveGAflux: maxtrainbatchsize, maxvalidationbatchsize
        graph = testgraph(5)

        @test maxtrainbatchsize(graph, (5,), 1000) == 2
        @test maxtrainbatchsize(graph, (5,), 2000) == 4

        @test maxvalidationbatchsize(graph, (5,), 1000) == 8
        @test maxvalidationbatchsize(graph, (5,), 2000) == 17
    end

    @testset "limit_maxbatchsize" begin
        import NaiveGAflux: limit_maxbatchsize, TrainBatchSize, ValidationBatchSize
        graph = testgraph(5)

        @test limit_maxbatchsize(graph, TrainBatchSize(1); inshape_nobatch=(5,), availablebytes=1000) == 1 
        @test limit_maxbatchsize(graph, TrainBatchSize(2); inshape_nobatch=(5,), availablebytes=1000) == 2
        @test limit_maxbatchsize(graph, TrainBatchSize(3); inshape_nobatch=(5,), availablebytes=1000) == 2
        
        @test limit_maxbatchsize(graph, ValidationBatchSize(6); inshape_nobatch=(5,), availablebytes=1000) == 6
        @test limit_maxbatchsize(graph, ValidationBatchSize(8); inshape_nobatch=(5,), availablebytes=1000) == 8
        @test limit_maxbatchsize(graph, ValidationBatchSize(10); inshape_nobatch=(5,), availablebytes=1000) == 8
    end

    @testset "batchsizeselection" begin
        import NaiveGAflux: limit_maxbatchsize, TrainBatchSize, ValidationBatchSize
        # Pretty much the integration tests as it uses all the above components
        graph = testgraph(4)
        bs = batchsizeselection((4,))
        
        @test bs(graph, TrainBatchSize(31), availablebytes=10000) == 19
        @test bs(graph, ValidationBatchSize(31), availablebytes=10000) == 31

        bs = batchsizeselection((4,); maxmemutil=0.1)
        @test bs(graph, TrainBatchSize(31), availablebytes=10000) == 2
        @test bs(graph, ValidationBatchSize(31), availablebytes=10000) == 8

        bs = batchsizeselection((4,); uppersize=64)
        @test bs(graph, TrainBatchSize(31), availablebytes=10000) == 19
        @test bs(graph, ValidationBatchSize(31), availablebytes=10000) == 64

        bs = batchsizeselection((4,); alternatives=2 .^ (0:10))
        @test bs(graph, TrainBatchSize(33), availablebytes=10000) == 16
        @test bs(graph, ValidationBatchSize(33), availablebytes=10000) == 32

        bs = batchsizeselection((4,); uppersize=65, alternatives=2 .^ (0:10))
        @test bs(graph, TrainBatchSize(31), availablebytes=10000) == 16
        @test bs(graph, ValidationBatchSize(31), availablebytes=10000) == 64

    end
end