@testset "Crossover" begin

    @testset "CrossoverSwap" begin
        import NaiveGAflux: crossoverswap

        iv(np) = inputvertex("$np.in", 3, FluxDense())
        dv(in, outsize, name) = mutable(name, Dense(nout(in), outsize), in)

        @testset "Simple Dense swap" begin

            function g(sizes, np)
                vi = iv(np)
                vo = foldl(enumerate(sizes); init=vi) do vin, (i,s)::Tuple
                    dv(vin, s, "$np.dv$i")
                end
                return CompGraph(vi, vo)
            end

            @testset "Single vertex swap" begin
                ga = g(4:6, "a")
                gb = g(1:3, "b")

                crossoverswap(vertices(ga)[3], vertices(gb)[3])
                apply_mutation(ga)
                apply_mutation(gb)

                @test nout.(vertices(ga)) == [3,4,2,6]
                @test nout.(vertices(gb)) == [3,1,5,3]

                @test size(ga(ones(3, 2))) == (6, 2)
                @test size(gb(ones(3, 2))) == (3, 2)
            end

            @testset "Path swap" begin
                ga = g(4:8, "a")
                gb = g(1:5, "b")

                crossoverswap(vertices(ga)[2], vertices(ga)[5], vertices(gb)[3], vertices(gb)[4])
                apply_mutation(ga)
                apply_mutation(gb)

                @test nout.(vertices(ga)) == [3,2,3,8]
                @test nout.(vertices(gb)) == [3,1,4,5,6,7,4,5]

                @test size(ga(ones(3, 2))) == (8, 2)
                @test size(gb(ones(3, 2))) == (5, 2)
            end
        end

        @testset "Swap add and conc" begin
            function g(sizes, np, cfun)
                vi = iv(np)
                vs = map((i, s) -> dv(vi, s, "$np.dv$i"), eachindex(sizes[1:end-1]), sizes[1:end-1])
                vm = cfun(vs...)
                return CompGraph(vi, dv(vm, sizes[end], "$np.dv$(length(sizes))"))
            end

            ga = g(3:7, "a", (vs...) ->concat("a.merge", vs...))
            gb = g(3 .* ones(Int, 4), "b", (vs...) -> +("b.merge" >> vs[1], vs[2:end]...))

            crossoverswap(vertices(ga)[end-1], vertices(gb)[end-1])

            apply_mutation(ga)
            apply_mutation(gb)

            @test nout.(vertices(ga)) == [3, 4, 4, 4, 4, 4, 7]
            @test nout.(vertices(gb)) == [3, 3, 3, 3, 9, 3]

            @test size(ga(ones(3, 2))) == (7, 2)
            @test size(gb(ones(3, 2))) == (3, 2)
        end
    end
end
