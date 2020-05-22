@testset "visualization" begin

    @testset "Peristent plots" begin
        # All tests use the same dir to also catch if one plot overwrites another by default
        testdir = "test_persistent_plots"

        struct DummyGraph
            nv::Int
            np::Int
        end
        NaiveNASlib.nv(g::DummyGraph) = g.nv
        NaiveGAflux.nparams(g::DummyGraph) = g.np
        struct PlotTestCand <: AbstractCandidate
            fitness::Real
            graph::DummyGraph
            PlotTestCand(fitness, nv, np) = new(fitness, DummyGraph(nv, np))
        end
        NaiveGAflux.graph(c::PlotTestCand) = c.graph
        NaiveGAflux.fitness(c::PlotTestCand) = c.fitness

        try
            @testset "PlotFitness" begin

                p = PlotFitness((args...;kwargs...) -> [], testdir)
                @test !isdir(p.basedir)

                p(PlotTestCand.(1:3, [10, 20, 30], [100, 200, 300]))

                @test p.plt == [1, [3, 2]]
                @test p.best == [3]
                @test p.avg == [2]

                p(PlotTestCand.(2:4, [10, 20, 30], [100, 200, 300]))
                @test p.plt == [1, [3, 2], 2, [4, 3]]
                @test p.best == [3, 4]
                @test p.avg == [2, 3]

                p2 = PlotFitness((args...;kwargs...) -> [args...], testdir)
                @test p2.plt == [[1,2], [3 2; 4 3]]
                @test p2.best == p.best
                @test p2.avg == p.avg
            end

            @testset "ScatterPop" begin

                p = ScatterPop((args...;kwargs...) -> true, testdir)
                @test !isdir(p.basedir)

                @test p(PlotTestCand.(1:3, [10, 20, 30], [100, 200, 300]))
                @test p.data ==  [[10 1 100; 20 2 200; 30 3 300]]

                @test p(PlotTestCand.(2:4, [20, 30, 40], [200, 300, 400]))
                @test p.data ==  [[10 1 100; 20 2 200; 30 3 300], [20 2 200; 30 3 300; 40 4 400]]

                p2 = ScatterPop((args...;kwargs...) -> true, testdir)
                @test p2.data == p.data

                @test p2(PlotTestCand.(3:5, [30, 40, 50], [300, 400, 500]))
                @test p2.data ==  [[10 1 100; 20 2 200; 30 3 300], [20 2 200; 30 3 300; 40 4 400], [30 3 300; 40 4 400; 50 5 500]]
            end

            @testset "ScatterOpt" begin
                NaiveGAflux.opt(c::PlotTestCand) = fitness(c) > 2 ? ADAM(nparams(c) - fitness(c)) : Flux.Optimiser([Descent(nparams(c) - fitness(c))])

                p = ScatterOpt((args...;kwargs...) -> true, testdir)
                @test !isdir(p.basedir)

                @test p(PlotTestCand.(1:3, [10, 20, 30], [100, 200, 300]))
                @test p.data ==  [[1 99.0 Descent; 2 198.0 Descent; 3 297.0 ADAM]]

                @test p(PlotTestCand.(2:4, [20, 30, 40], [200, 300, 400]))
                @test p.data ==  [[1 99 Descent; 2 198 Descent; 3 297 ADAM], [2 198 Descent; 3 297 ADAM; 4 396 ADAM]]

                p2 = ScatterOpt((args...;kwargs...) -> true, testdir)
                @test p2.data == p.data

                @test p2(PlotTestCand.(3:5, [30, 40, 50], [300, 400, 500]))
                @test p2.data ==  [[1 99 Descent; 2 198 Descent; 3 297 ADAM], [2 198 Descent; 3 297 ADAM; 4 396 ADAM],[3 297 ADAM; 4 396 ADAM; 5 495 ADAM]]
            end

        finally
            rm(testdir, force=true, recursive=true)
        end

        @testset "MultiPlot" begin
            struct MultiPlotTestMock
                nr::Int
                seen::Vector{Int}
                MultiPlotTestMock(nr) = new(nr, Int[])
            end
            NaiveGAflux.plotgen(p::MultiPlotTestMock, gen=length(p.seen)) = gen==0 ? p.nr : p.nr * p.seen[gen]
            function (p::MultiPlotTestMock)(pop)
                push!(p.seen, pop)
                return NaiveGAflux.plotgen(p)
            end

            pdata = []
            spy(x...) = push!(pdata, x...)
            p = MultiPlot(spy, MultiPlotTestMock.(2:4)...)

            @test pdata == 2:4

            @test p(3) == pdata ==[2,3,4,6,9,12]
            @test mapfoldl(pp -> pp.seen, vcat, p.plts) == [3,3,3]
        end

        @testset "CbAll" begin
            cb1 = false
            cb2 = false

            CbAll(pop -> cb1 = pop, pop -> cb2 = pop)(123)

            @test cb1 == 123
            @test cb2 == 123
        end
    end
end
