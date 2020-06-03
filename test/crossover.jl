@testset "Crossover" begin

    @testset "CrossoverSwap" begin
        import NaiveGAflux: crossoverswap, separablefrom

        iv(np) = inputvertex("$np.in", 3, FluxDense())
        dv(in, outsize, name) = mutable(name, Dense(nout(in), outsize), in)

        v4n(graph::CompGraph, want) = v4n(vertices(graph), want)
        v4n(vs, want) = vs[findfirst(v -> want == name(v), vs)]

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

            ga = g("a")
            vsa = vertices(ga)
            aswap = separablefrom(v4n(ga, "a.add_aa_bb"))
            @test name.(vsa) == name.(vertices(ga))
            @test name.(aswap) == ["a.add_aa_bb", "a.dva1"]

            gb = g("b", true)
            vsb = vertices(gb)
            @test name.(separablefrom(v4n(gb, "b.add_aa_bb"))) == ["b.add_aa_bb"]
            @test name.(vsb) == name.(vertices(gb))

            bswap = separablefrom(v4n(gb, "b.dvbb1"))
            @test name.(vsb) == name.(vertices(gb))
            @test name.(bswap) == ["b.dvbb1"]


            crossoverswap(aswap[end], aswap[1], bswap[end], bswap[1])
            apply_mutation(ga)
            apply_mutation(gb)

            @test name.(vertices(ga)) == ["a.in", "a.dv1", "b.dvbb1", "a.dvb1", "a.dvba1", "a.dvbb1", "a.conc_ba_bb", "a.conc_a_b", "a.out"]
            @test size(ga(ones(3,2))) == (4, 2)

            @test name.(vertices(gb)) == ["b.in", "b.dv1", "b.dva1", "b.dvaa1", "b.dvaa2", "b.dvab1", "b.dvab2", "b.add_aa_bb", "b.dvb1", "b.conc_dvb1_dvab2", "a.dva1", "a.dvaa1", "a.dvaa2", "a.dvab1", "a.dvab2", "a.add_aa_bb", "b.conc_ba_bb", "b.conc_a_b", "b.out"]
            @test size(gb(ones(3,2))) == (4, 2)
        end

        @testset "Swapping preserves edge order" begin
            import NaiveGAflux: stripoutedges!, stripinedges!, addoutedges!, addinedges!
            using Random
            function g(np)
                vi = iv(np)
                dv1 = dv(vi, 2, "$np.dv1")
                dv2 = dv(vi, 3, "$np.dv2")
                dv3 = dv(vi, 4, "$np.dv3")
                ca1 = concat("$np.ca1", dv1, dv2, dv1, dv3, dv1)
                dva1 = dv(ca1, 4, "$np.dva1")
                dvb1 = dv(dv2, 3, "$np.dvb1")
                c2 = concat("$np.c2", dvb1, dva1, dvb1, dva1)
                c3 = concat("$np.c3", dva1, dva1, dvb1)
                c4 = concat("$np.c4", c2,c3)
                return CompGraph(vi, c4)
            end
            @testset "Strip and add out edges" begin
                gg = g("a")
                dva1 = v4n(gg, "a.dva1")

                indata = randn(MersenneTwister(0), 3,2)
                graphout_before = gg(indata)

                expected1 = name.(outputs(dva1))
                expected2 = mapreduce(vo -> name.(inputs(vo)), vcat, unique(outputs(dva1)))

                o, oi = stripoutedges!(dva1)
                @test outputs(dva1) == []
                addoutedges!(dva1, o, oi, NoSizeChange())
                actual1 = name.(outputs(dva1))
                actual2 = mapreduce(vo -> name.(inputs(vo)), vcat, unique(outputs(dva1)))

                @test actual1 == expected1
                @test actual2 == expected2

                @test gg(indata) == graphout_before
            end

            @testset "Strip and add in edges" begin
                gg = g("a")
                ca1 = v4n(gg, "a.ca1")

                indata = randn(MersenneTwister(0), 3,2)
                graphout_before = gg(indata)

                expected = name.(inputs(ca1))

                i, ii = stripinedges!(ca1)
                @test inputs(ca1) == []
                addinedges!(ca1, i, ii, NoSizeChange())
                actual = name.(inputs(ca1))

                @test actual == expected

                @test gg(indata) == graphout_before
            end

            g_org = g("a")
            g_new = copy(g_org)

            indata = randn(MersenneTwister(0), 3, 2)
            out_org = g_org(indata)

            swappable_new = separablefrom(v4n(g_new, "a.dva1"))
            @test name.(swappable_new) == ["a.dva1", "a.ca1"]
            @test name.(vertices(g_org)) == name.(vertices(g_new))

            @test g_org(indata) == g_new(indata) == out_org

            vs_org = vertices(g_org)

            swappable_org = separablefrom(v4n(g_org, "a.dva1"))
            crossoverswap(swappable_org[end], swappable_org[1], swappable_new[end], swappable_new[1])

            @test name.(vertices(g_org)) == name.(vertices(g_new)) == name.(vs_org)
            @test g_org(indata) == g_new(indata) == out_org

        end

        @testset "Revert after failed size align" begin
            idv(in, outsize, name) = mutable(name, Dense(nout(in), outsize), in; traitfun=t -> NamedTrait(Immutable(), name))

            function g(np, mergesize, mergeop)
                vi = iv(np)
                dv1 = dv(vi, mergesize, "$np.dv1")
                dv2 = idv(dv1, mergesize, "$np.dv2")
                m1 = mergeop("$np.m1", dv1, vi, dv2, dv1)
                dv3 = dv(m1, 5, "$np.dv3")
                return CompGraph(vi, dv3)
            end

            ga = g("a", 3, (vname, vs...) -> +(vname >> vs[1], vs[2:end]...))
            gb = g("b", 5, concat)

            indata = randn(MersenneTwister(0), 3, 2)
            outa = ga(indata)
            outb = gb(indata)

            vsa = vertices(ga)
            vsb = vertices(gb)

            noutsa = nout.(vsa)
            ninsa = nin.(vsa)

            noutsb = nout.(vsb)
            ninsb = nin.(vsb)

            @test_logs (:warn, "Failed to align sizes when adding edge between b.dv1 and a.m1 for crossover. Reverting...") crossoverswap(vertices(ga)[end-1], vertices(gb)[end-1])

            @test name.(vsa) == name.(vertices(ga))
            @test name.(vsb) == name.(vertices(gb))

            @test noutsa == nout.(vertices(ga))
            @test ninsa == nin.(vertices(ga))

            @test noutsb == nout.(vertices(gb))
            @test ninsb == nin.(vertices(gb))

            @test outa == ga(indata)
            @test outb == gb(indata)
        end
    end
end
