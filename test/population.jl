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

    @testset "Persistent pop" begin
        testdir = "testPersistentPopulation"
        mems = PersistentArray(testdir, 4, identity)
        pop1 = Population(mems)
        @test generation(pop1) == 1
        pop2 = evolve!(NoOpEvolution(), pop1)
        @test generation(pop2) == 2

        try

            persist(pop2)
            pop2 = Population(mems)

            @test generation(pop2) == 2

            pop3 = evolve!(NoOpEvolution(), pop2)
            @test generation(pop3) == 3

            persist(pop3)
            pop3 = Population(mems)

            @test generation(pop3) == 3
        finally
            rm(testdir, force=true, recursive=true)
        end
    end
end
