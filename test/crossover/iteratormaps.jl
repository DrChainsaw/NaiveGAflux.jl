@testset "IteratorMapCrossover" begin
    import NaiveGAflux: AbstractIteratorMap
    struct ImcTestDummy1 <: AbstractIteratorMap end
    struct ImcTestDummy2 <: AbstractIteratorMap end
    struct ImcTestDummy3 <: AbstractIteratorMap end

    @testset "Simple" begin
        @test IteratorMapCrossover()((ImcTestDummy1(), ImcTestDummy2())) ==  (ImcTestDummy2(), ImcTestDummy1())     
    end

    @testset "IteratorMaps" begin
        ims1 = IteratorMaps(ImcTestDummy1(), ImcTestDummy2())
        ims2 = IteratorMaps(ImcTestDummy2(), ImcTestDummy1())

        imc = IteratorMapCrossover()

        @test imc((ims1, ims2)) == (ims2, ims1)
        @test imc((ims1, ImcTestDummy2())) == (IteratorMaps(ImcTestDummy2(), ImcTestDummy2()), ImcTestDummy1())
    end

    @testset "LogMutation and MutationProbability" begin
        mplm(c) = MutationProbability(LogMutation(((im1,im2)::Tuple) -> "Crossover between $(im1) and $(im2)", c), Probability(0.2, MockRng([0.3, 0.1, 0.3])))
        imc = IteratorMapCrossover() |> mplm |> IteratorMapCrossover

        ims1 = IteratorMaps(ImcTestDummy1(), ImcTestDummy2(), ImcTestDummy3())
        ims2 = IteratorMaps(ImcTestDummy3(), ImcTestDummy1(), ImcTestDummy2())

        ims1n,ims2n = @test_logs (:info, "Crossover between ImcTestDummy2() and ImcTestDummy1()") imc((ims1,ims2))

        @test ims1n == IteratorMaps(ImcTestDummy1(), ImcTestDummy1(), ImcTestDummy3())
        @test ims2n == IteratorMaps(ImcTestDummy3(), ImcTestDummy2(), ImcTestDummy2())
    end
end