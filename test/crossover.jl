@testset "Crossover" begin

    @testset "CrossoverSwap" begin

        function crossoverswap(v1::AbstractVertex, v2::AbstractVertex)
            vc1, i1, o1 = stripvertex(v1)
            vc2, i2, o2 = stripvertex(v2)

            foreach(iv -> create_edge!(iv, vc2), i1)
            foreach(ov -> create_edge!(vc2, ov), o1)

            foreach(iv -> create_edge!(iv, vc1), i2)
            foreach(ov -> create_edge!(vc1, ov), o2)
            return vc1, vc2
        end

        function stripvertex(v)
            i, o = copy(inputs(v)), copy(outputs(v))
            foreach(iv -> remove_edge!(iv, v; strategy = NoSizeChange()), i)
            foreach(ov -> remove_edge!(v, ov, strategy = NoSizeChange()), o)
            vc = clone(v)
            return vc, i, o
        end

        @testset "Simple Dense swap" begin
            iv(np) = inputvertex("$np.in", 3, FluxDense())
            dv(in, outsize, name) = mutable(name, Dense(nout(in), outsize), in)

            function g(sizes, np, vfun)
                vi = iv(np)
                vo = foldl(enumerate(sizes); init=vi) do vin, (i,s)::Tuple
                    vfun(vin, s, "$np.dv$i")
                end
                return CompGraph(vi, vo)
            end

            ga = g(4:6, "a", dv)
            gb = g(1:3, "b", dv)

            crossoverswap(vertices(ga)[3], vertices(gb)[3])

            @test nout.(vertices(ga)) == [3,4,2,6]
            @test nout.(vertices(gb)) == [3,1,5,3]
        end
    end
end
