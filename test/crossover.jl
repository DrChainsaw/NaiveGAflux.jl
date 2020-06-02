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

        import NaiveGAflux: stripedges, stripinputs, stripoutputs
        # TODO Rewrite in a non-destrucive manner. LightGraphs?
        function swappablefrom(v)
            myinds = map(vo -> indexin([v], inputs(vo)), outputs(v))
            o = stripoutputs(v)
            swappable = swappablefrom(v, AbstractVertex[v])
            foreach((ov, pos) -> create_edge!(v, ov; pos=pos[], strategy = NoSizeChange()), o, myinds)
            return swappable
        end

        function swappablefrom(v, seen)
            i = stripinputs(v)
            ok = all(vv -> vv in seen, all_in_graph(v))
            foreach(iv -> create_edge!(iv, v; strategy=NoSizeChange()), i)
            push!(seen, v)
            swappable = mapreduce(vi -> swappablefrom(vi, seen), vcat, inputs(v), init=[])
            return ok ? vcat(v, swappable) : swappable
        end

        @testset "Find swappable path" begin
            function g(np, bconnect = false)
                vi = iv(np)
                dv1 = dv(vi, 4, "$np.dv1")

                dva1 = dv(dv1, 3, "$np.dva1")
                dvaa1 = dv(dva1, 5, "$np.dvaa1")
                dvaa2 = dv(dvaa1, 4, "$np.dvaa2")
                dvab1 = dv(dva1, 2, "$np.dvab1")
                dvab2 = dv(dvab1, 4, "$np.dvab2")
                add_aa_bb = "$np.add_aa_bb" >> dvaa2 + dvab2

                dvb1 = dv(dv1, 5, "$np.dvb1")
                vba1 = bconnect ? concat("$np.conc_dvb1_dvab2", dvb1, dvab2) : dv(dvb1, nout(dvb1) + nout(dvab2), "$np.dvba1")
                dvbb1 = dv(dv1, 4, "$np.dvbb1")
                conc_ba_bb = concat("$np.conc_ba_bb", vba1, dvbb1)

                conc_a_b = concat("$np.conc_a_b", add_aa_bb, conc_ba_bb)
                return CompGraph(vi, dv(conc_a_b, 4, "$np.out"))
            end

            v4n(graph::CompGraph, want) = v4n(vertices(graph), want)
            v4n(vs, want) = vs[findfirst(v -> want == name(v), vs)]

            ga = g("a")
            vsa = vertices(ga)
            aswap = swappablefrom(v4n(ga, "a.add_aa_bb"))
            @test name.(vsa) == name.(vertices(ga))
            @test name.(aswap) == ["a.add_aa_bb", "a.dva1"]

            gb = g("b", true)
            vsb = vertices(gb)
            @test name.(swappablefrom(v4n(gb, "b.add_aa_bb"))) == ["b.add_aa_bb"]
            @test name.(vsb) == name.(vertices(gb))

            bswap = swappablefrom(v4n(gb, "b.dvbb1"))
            @test name.(vsb) == name.(vertices(gb))
            @test name.(bswap) == ["b.dvbb1"]


            crossoverswap(aswap[end], aswap[1], bswap[end], bswap[1])
            apply_mutation(ga)
            apply_mutation(gb)

            @test name.(vertices(ga)) == ["a.in", "a.dv1", "a.dvb1", "a.dvba1", "a.dvbb1", "a.conc_ba_bb", "b.dvbb1", "a.conc_a_b", "a.out"]
            @test size(ga(ones(3,2))) == (4, 2)

            @test name.(vertices(gb)) == ["b.in", "b.dv1", "b.dva1", "b.dvaa1", "b.dvaa2", "b.dvab1", "b.dvab2", "b.add_aa_bb", "b.dvb1", "b.conc_dvb1_dvab2", "a.dva1", "a.dvaa1", "a.dvaa2", "a.dvab1", "a.dvab2", "a.add_aa_bb", "b.conc_ba_bb", "b.conc_a_b", "b.out"]
            @test size(gb(ones(3,2))) == (4, 2)
        end
    end
end
