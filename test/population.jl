@testset "Population" begin

    function test_interface(p, members)
        test_iterator(p, members)

        ep = evolve!(NoOpEvolution(), p)
        test_iterator(ep, p)

        @test generation(ep) == generation(p) + 1
    end

    function test_iterator(p1, p2)
        @test length(p1) == length(p2)
        @test size(p1) == size(p2)
        @test collect(p1) == collect(p2)
    end

    test_interface(Population(1:10), 1:10)
end
