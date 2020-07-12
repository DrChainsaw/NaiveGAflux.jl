@testset "Crossover" begin

    v4n(graph::CompGraph, want) = v4n(vertices(graph), want)
    v4n(vs, want) = vs[findfirst(v -> want == name(v), vs)]

    teststrat() = NaiveGAflux.default_crossoverswap_strategy(v -> 1:nout_org(v))

    @testset "CrossoverSwap" begin
        import NaiveGAflux: crossoverswap!, separablefrom

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

                @test crossoverswap!(vertices(ga)[3], vertices(gb)[3]; strategy=teststrat) == (true,true)

                @test name.(vertices(ga)) == ["a.in", "a.dv1", "b.dv2", "a.dv3"]
                @test name.(vertices(gb)) == ["b.in", "b.dv1", "a.dv2", "b.dv3"]

                @test nout.(vertices(ga)) == [3,1,3,6]
                @test nout.(vertices(gb)) == [3,1,3,3]

                @test size(ga(ones(3, 2))) == (6, 2)
                @test size(gb(ones(3, 2))) == (3, 2)
            end

            @testset "Path swap" begin
                ga = g(4:8, "a")
                gb = g(1:5, "b")

                @test crossoverswap!(vertices(ga)[2], vertices(ga)[5], vertices(gb)[3], vertices(gb)[4]; strategy=teststrat) == (true,true)

                @test nout.(vertices(ga)) == [3,2,4,8]
                @test nout.(vertices(gb)) == [3,1,4,5,6,4,4,5]

                @test size(ga(ones(3, 2))) == (8, 2)
                @test size(gb(ones(3, 2))) == (5, 2)
            end

            @testset "Swap same graph is noop" begin
                ga = g(4:7, "a")

                ren(str::String;cf=ren) = "b" * str[2:end]
                ren(x...;cf=ren) = clone(x...;cf=cf)
                gb = copy(ga, ren)

                indata = reshape(collect(Float32, 1:3*2),3,:)
                orgout = ga(indata)
                @test orgout == gb(indata)

                @test name.(vertices(ga)) == ["a.in", "a.dv1", "a.dv2", "a.dv3", "a.dv4"]
                @test name.(vertices(gb)) == ["b.in", "b.dv1", "b.dv2", "b.dv3", "b.dv4"]

                vsa = vertices(ga)
                vsb = vertices(gb)
                @test crossoverswap!(vsa[2], vsa[4], vsb[2], vsb[4]) == (true,true)

                @test orgout == ga(indata) == gb(indata)
            end
        end

        @testset "Swap add and conc with conc larger" begin
            function g(sizes, np, cfun)
                vi = iv(np)
                vs = map((i, s) -> dv(vi, s, "$np.dv$i"), eachindex(sizes[1:end-1]), sizes[1:end-1])
                vm = cfun("$np.merge", vs...)
                return CompGraph(vi, dv(vm, sizes[end], "$np.dv$(length(sizes))"))
            end

            ga = g(3:7, "a", concat)
            gb = g(3 .* ones(Int, 4), "b", (name, vs...) -> +(name >> vs[1], vs[2:end]...))

            @test crossoverswap!(vertices(ga)[end-1], vertices(gb)[end-1]; strategy=teststrat) == (true, true)

            @test nout.(vertices(ga)) == [3, 4, 4, 4, 4, 4, 7]
            @test nout.(vertices(gb)) == [3, 1, 2, 2, 5, 3]

            @test size(ga(ones(3, 2))) == (7, 2)
            @test size(gb(ones(3, 2))) == (3, 2)
        end

        @testset "Swap add and conc with add larger" begin
            function g(sizes, np, cfun)
                vi = iv(np)
                vs = map((i, s) -> dv(vi, s, "$np.dv$i"), eachindex(sizes[1:end-1]), sizes[1:end-1])
                vm = cfun("$np.merge", vs...)
                return CompGraph(vi, dv(vm, sizes[end], "$np.dv$(length(sizes))"))
            end

            ga = g(3:6, "a", concat)
            gb = g(3 .* ones(Int, 6), "b", (name, vs...) -> +(name >> vs[1], vs[2:end]...))

            @test crossoverswap!(vertices(ga)[end-1], vertices(gb)[end-1]; strategy=teststrat) == (true, true)

            @test nout.(vertices(ga)) == [3, 3, 3, 3, 3, 6]
            @test nout.(vertices(gb)) == [3, 1, 1, 1, 1, 1, 5, 3]

            @test size(ga(ones(3, 2))) == (6, 2)
            @test size(gb(ones(3, 2))) == (3, 2)
        end

        @testset "Swap conc and dense" begin
            function g(sizes, np, cfun)
                vi = iv(np)
                vs = map((i, s) -> dv(vi, s, "$np.dv$i"), eachindex(sizes[1:end]), sizes[1:end])
                vm = cfun("$np.merge", vs...)
                dvn = dv(vm, 2, "$np.dvn")
                dvo = dv(dvn, 4, "$np.dvo")
                return CompGraph(vi, dvo)
            end

            ga = g(2:4, "a", concat)
            gb = g(2:4, "b", concat)

            crossoverswap!(vertices(ga)[end-1], vertices(gb)[end-2]; mergefun = v -> concat("extramerge", v), strategy=teststrat)

            @test name.(vertices(ga)) == ["a.in", "a.dv1", "a.dv2", "a.dv3", "a.merge", "b.merge", "a.dvo"]
            @test name.(vertices(gb)) == ["b.in", "b.dv1", "b.dv2", "b.dv3", "extramerge", "a.dvn", "b.dvn", "b.dvo"]

            @test nout.(vertices(ga)) == [3, 1, 1, 1, 3, 3, 4]
            @test nout.(vertices(gb)) == [3, 3, 3, 4, 10, 3, 2, 4]

            @test size(ga(ones(3, 2))) == (4, 2)
            @test size(gb(ones(3, 2))) == (4, 2)
        end

        @testset "Swap residual" begin
            function g(sizes, np, withinv)
                vi = iv(np)
                dv0 = dv(vi, 3, "$np.dv0")
                dv1 = dv(dv0, sizes[1], "$np.dv1")
                vo = foldl(enumerate(sizes[2:end]); init=dv1) do vin, (i,s)::Tuple
                    dv(vin, s, "$np.dv$i")
                end
                dvn = dv(vo, sizes[1], "$np.dvn")
                vn = withinv ? invariantvertex(identity, dvn; traitdecoration= named("$np.inv")) : dvn
                add = "$np.add" >> dv1 + vn
                dvo = dv(add, 4, "$np.dvo")
                return CompGraph(vi, dvo)
            end

            @testset "Invariant before add" begin
                ga = g(4:6, "a", true)
                gb = g(5:6, "b", true)

                vsa = vertices(ga)
                vsb = vertices(gb)

                @test crossoverswap!(vsa[4], vsa[end-2], vsb[4], vsb[end-2]; strategy=teststrat) == (true, true)
                @test size(ga(ones(Float32, 3, 2))) == (4,2)
                @test size(gb(ones(Float32, 3, 2))) == (4,2)
            end
        end

        @testset "Find swappable path" begin

            @testset "Residual on input" begin
                function g(np)
                    vi = iv(np)
                    dv1 = dv(vi, 4, "$np.dv1")
                    dv2 = dv(dv1, nout(vi), "$np.dv2")
                    add1 = "$np.add1" >> vi + dv2
                    dv3 = dv(add1, 4, "$np.dv3")
                    dv4 = dv(dv3, 3, "$np.dv4")
                    return CompGraph(vi, dv4)
                end

                ga = g("a")

                indata = randn(MersenneTwister(1), 3, 2)
                outa = ga(indata)

                vsa = vertices(ga)
                aswap = separablefrom(v4n(ga, "a.dv3"), ga.inputs)
                @test name.(aswap) == ["a.dv3", "a.add1"]

            end

            @testset "Dual branched graph with optional extra connection" begin
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

                indata = randn(MersenneTwister(1), 3, 2)
                outa = ga(indata)

                vsa = vertices(ga)
                aswap = separablefrom(v4n(ga, "a.add_aa_bb"), ga.inputs)
                @test name.(vsa) == name.(vertices(ga))
                @test name.(aswap) == ["a.add_aa_bb", "a.dva1"]

                @test ga(indata) == outa

                gb = g("b", true)
                outb = gb(indata)
                vsb = vertices(gb)
                @test name.(separablefrom(v4n(gb, "b.add_aa_bb"), gb.inputs)) == ["b.add_aa_bb"]
                @test name.(vsb) == name.(vertices(gb))

                @test gb(indata) == outb

                bswap = separablefrom(v4n(gb, "b.dvbb1"), gb.inputs)
                @test name.(vsb) == name.(vertices(gb))
                @test name.(bswap) == ["b.dvbb1"]

                @test gb(indata) == outb

                @test crossoverswap!(aswap[end], aswap[1], bswap[end], bswap[1]; strategy=teststrat) == (true, true)

                @test name.(vertices(ga)) == ["a.in", "a.dv1", "b.dvbb1", "a.dvb1", "a.dvba1", "a.dvbb1", "a.conc_ba_bb", "a.conc_a_b", "a.out"]
                @test size(ga(ones(3,2))) == (4, 2)

                @test name.(vertices(gb)) == ["b.in", "b.dv1", "b.dva1", "b.dvaa1", "b.dvaa2", "b.dvab1", "b.dvab2", "b.add_aa_bb", "b.dvb1", "b.conc_dvb1_dvab2", "a.dva1", "a.dvaa1", "a.dvaa2", "a.dvab1", "a.dvab2", "a.add_aa_bb", "b.conc_ba_bb", "b.conc_a_b", "b.out"]
                @test size(gb(ones(3,2))) == (4, 2)
            end
        end

        @testset "Swapping preserves edge order" begin
            import NaiveGAflux: stripoutedges!, stripinedges!, addoutedges!, addinedges!
            function g(np)
                vi = iv(np)
                dv1 = dv(vi, 2, "$np.dv1")
                dv2 = dv(vi, 3, "$np.dv2")
                dv3 = dv(vi, 4, "$np.dv3")
                ca1 = concat("$np.ca1", dv1, dv2, dv1, dv3, dv1)
                dva1 = dv(ca1, 4, "$np.dva1")
                dva2 = dv(dva1, 3, "$np.dva2")
                dvb1 = dv(dv2, 3, "$np.dvb1")
                c2 = concat("$np.c2", dvb1, dva1, dvb1, dva1)
                c3 = concat("$np.c3", dva1, dva1, dvb1)
                c4 = concat("$np.c4", c2,c3,dva2)
                out = dv(c4, 4, "$np.out")
                return CompGraph(vi, out)
            end
            @testset "Strip and add out edges" begin
                gg = g("a")
                dva1 = v4n(gg, "a.dva1")

                indata = randn(MersenneTwister(0), 3,2)
                graphout_before = gg(indata)

                expected1 = name.(outputs(dva1))
                expected2 = mapreduce(vo -> name.(inputs(vo)), vcat, unique(outputs(dva1)))

                dummy = stripoutedges!(dva1)
                @test outputs(dva1) == []
                addoutedges!(dva1, dummy, teststrat)
                apply_mutation(gg) # just to verify nothing has changed

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

                ins = stripinedges!(ca1)
                @test mapreduce(inputs, vcat, inputs(ca1)) == []
                addinedges!(ca1, ins, teststrat)
                apply_mutation(gg) #just to verify nothing has changed

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
            nouts_org = nout.(vs_org)

            swappable_org = separablefrom(v4n(g_org, "a.dva1"))
            @test crossoverswap!(swappable_org[end], swappable_org[1], swappable_new[end], swappable_new[1]; strategy=teststrat) == (true, true)

            @test name.(vertices(g_org)) == name.(vertices(g_new)) == name.(vs_org)
            @test nout.(vertices(g_org)) == nout.(vertices(g_new)) == nouts_org
            @test g_org(indata) == g_new(indata) == out_org
        end

        @testset "Partial success" begin
            idv(in, outsize, name) = mutable(name, Dense(nout(in), outsize), in; traitfun=t -> NamedTrait(Immutable(), name))

            function g(np, mergesize, mergeop)
                vi = iv(np)
                dv1 = dv(vi, mergesize, "$np.dv1")
                dv2 = idv(dv1, mergesize, "$np.dv2")
                m1 = mergeop("$np.m1", dv1, vi, dv2)
                dv3 = dv(m1, 5, "$np.dv3")
                return CompGraph(vi, dv3)
            end

            @testset "Fail first" begin
                ga = g("a", 3, (vname, vs...) -> +(vname >> vs[1], vs[2:end]...))
                gb = g("b", 5, concat)

                @test @test_logs (:warn, "Failed to align sizes for vertices b.dv2 and a.dv2.dummy for crossover. Reverting...") crossoverswap!(vertices(ga)[end-1], vertices(gb)[end-1]; strategy=teststrat) == (true, false)

                @test name.(vertices(ga)) == ["a.in", "a.dv1", "a.dv2", "b.m1", "a.dv3"]
                @test size(ga(ones(3,2))) == (5,2)
            end

            @testset "Fail second" begin
                ga = g("a", 5, concat)
                gb = g("b", 3, (vname, vs...) -> +(vname >> vs[1], vs[2:end]...))

                @test @test_logs (:warn, "Failed to align sizes for vertices a.dv2 and b.dv2.dummy for crossover. Reverting...") crossoverswap!(vertices(ga)[end-1], vertices(gb)[end-1]; strategy=teststrat) == (false, true)

                @test name.(vertices(gb)) == ["b.in", "b.dv1", "b.dv2", "a.m1", "b.dv3"]
                @test size(gb(ones(3,2))) == (5,2)
            end
        end

        @testset "Swap invariant layers" begin
            bv(in, name) = mutable(name, BatchNorm(nout(in)), in)
            pv(in, name) = mutable(name, MaxPool((3,3); pad=(1,1)), in)
            cv(in, outsize, name) = mutable(name, Conv((3,3), nout(in) => outsize; pad=(1,1)), in)

            @testset "Linear graph" begin
                function g(np)
                    vi = iv(np)
                    cv1 = cv(vi, 4, "$np.cv1")
                    bv1 = bv(cv1, "$np.bv1")
                    cv2 = cv(bv1, 5, "$np.cv2")
                    pv1 = pv(cv2, "$np.pv1")
                    cv3 = cv(pv1, 2, "$np.cv3")
                    return CompGraph(vi, cv3)
                end

                @testset "Swap $v1 to $v2" for (v1,v2) in (
                    ("bv1", "bv1"),
                    ("pv1", "pv1"),
                    ("bv1", "pv1")
                    )

                    vs(g, np) = v4n(g, "$np.$v1"), v4n(g, "$np.$v2")

                    ga = g("a")
                    gb = g("b")

                    @test crossoverswap!(vs(ga, "a")..., vs(gb, "b")...; strategy=teststrat) == (true, true)

                    @test "a.$v1" ∉ name.(vertices(ga))
                    @test "b.$v1" ∈ name.(vertices(ga))

                    @test "a.$v2" ∉ name.(vertices(ga))
                    @test "b.$v2" ∈ name.(vertices(ga))

                    @test "b.$v1" ∉ name.(vertices(gb))
                    @test "a.$v1" ∈ name.(vertices(gb))

                    @test "b.$v2" ∉ name.(vertices(gb))
                    @test "a.$v2" ∈ name.(vertices(gb))

                    @test size(ga(ones(Float32, 4,4,3,2))) == (2,2,2,2)
                    @test size(gb(ones(Float32, 4,4,3,2))) == (2,2,2,2)
                end

                @testset "Self swap is noop" begin
                    g_org = g("a")

                    bn = layer(v4n(g_org, "a.bv1"))
                    bn.γ .= randn(Float32, nout(bn))

                    indata = randn(Float32, 4,4,3,2)
                    out_org = g_org(indata)

                    g_new = copy(g_org)

                    @test crossoverswap!(v4n(g_org, "a.bv1"), v4n(g_new, "a.bv1"); strategy=teststrat) == (true, true)

                    @test g_org(indata) == out_org
                    @test g_new(indata) == out_org
                end
            end

            @testset "Swap double transparent parlayer" begin

                function g(sizes, np)
                    vi = iv(np)
                    cv1 = cv(vi, sizes[1], "$np.cv1")
                    bv1 = bv(cv1, "$np.bv1")
                    bv2 = bv(bv1, "$np.bv2")
                    cv2 = cv(bv2, sizes[2], "$np.cv2")
                    cv3 = cv(cv2, 4, "$np.cv3")
                    return CompGraph(vi, cv3)
                end

                ga = g((3,2), "a")
                gb = g((6,2), "b")

                strat = () -> NaiveGAflux.default_crossoverswap_strategy(v -> 1:nout_org(v))
                @test crossoverswap!(vertices(ga)[3], vertices(ga)[4], vertices(gb)[3], vertices(gb)[4]; strategy = strat) == (true,true)

                @test name.(vertices(ga)) == ["a.in", "a.cv1", "b.bv1", "b.bv2", "a.cv2", "a.cv3"]
                @test name.(vertices(gb)) == ["b.in", "b.cv1", "a.bv1", "a.bv2", "b.cv2", "b.cv3"]

                @test size(ga(ones(Float32, 4,4,3,1))) == (4,4,4,1)
                @test size(gb(ones(Float32, 4,4,3,1))) == (4,4,4,1)
            end
        end
    end

    @testset "VertexCrossover" begin
        import NaiveGAflux: GlobalPool, sameactdims, default_pairgen

        iv(np) = inputvertex("$np.in", 3, FluxDense())
        dv(in, outsize, name) = mutable(name, Dense(nout(in), outsize), in)
        gv(in, name) = invariantvertex(GlobalPool(MaxPool), in; traitdecoration = named(name))
        cv(in, outsize, name) = mutable(name, Conv((3,3), nout(in) => outsize; pad=(1,1)), in)

        @testset "Linear graph" begin
            function g(np, cvs, dvs)
                vi = iv(np)
                cvn = foldl(enumerate(cvs); init=vi) do vin, (i,s)::Tuple
                    cv(vin, s, "$np.cv$i")
                end
                pv1 = gv(cvn, "$np.pv1")
                dvn = foldl(enumerate(dvs); init=pv1) do vin, (i,s)::Tuple
                    dv(vin, s, "$np.dv$i")
                end
                return CompGraph(vi, dvn)
            end

            @testset "regraph" begin
                import NaiveGAflux: regraph

                g_org = g("test", 3:4, 5:6)

                v = vertices(g_org)[4]

                g_new = regraph(v, 1, 1)
                @test name.(vertices(g_org)) == name.(vertices(g_new))

                @test_throws AssertionError regraph(v, 2 ,1)
                @test_throws AssertionError regraph(v, 1, 2)
            end

            @testset "Generate pairs" begin
                ga = g("a", (4,5), (3,2))

                vsa = vertices(ga)[2:end]
                @test sameactdims.(vsa[end], vsa) == sameactdims.(vsa[end-1], vsa) == Bool[0, 0, 0, 1, 1]
                @test sameactdims.(vsa[1], vsa) == sameactdims.(vsa[2], vsa) == Bool[1, 1, 0, 0, 0]
                @test sameactdims.(vsa[3], vsa) == Bool[0,0,1,0,0]

                @testset "Match same size" begin
                    gb = g("b", (4,5), (3,2))
                    vsb = vertices(gb)[2:end]

                    @test default_pairgen(vsa, vsb; ind1=1) == (1, 1)
                    @test default_pairgen(vsa, vsb; ind1=3) == (3, 3)
                    @test default_pairgen(vsa, vsb; ind1=length(vsa)+1) == nothing
                end

                @testset "Match larger" begin
                    gb = g("b", 2:7, (8,))
                    vsb = vertices(gb)[2:end]

                    @test default_pairgen(vsa, vsb; ind1=1) == (1, 2)
                    @test default_pairgen(vsa, vsb; ind1=3) == (3, 7) # Only global pool matches with global pool
                    @test default_pairgen(vsa, vsb; ind1=5) == (5, 8)
                end

                @testset "Match smaller" begin
                    gb = g("b", (2,), (3,))
                    vsb = vertices(gb)[2:end]

                    @test default_pairgen(vsa, vsb; ind1=1) == (1, 1)
                    @test default_pairgen(vsa, vsb; ind1=2) == (2, 1)
                    @test default_pairgen(vsa, vsb; ind1=3) == (3, 2) # Only global pool matches with global pool
                    @test default_pairgen(vsa, vsb; ind1=4) == (4, 3)
                    @test default_pairgen(vsa, vsb; ind1=5) == (5, 3)
                end
            end

            @testset "Test swap single" begin
                import NaiveGAflux: crossover, default_pairgen, crossoverswap
                ga = g("a", 3:4, 5:6)
                gb = g("b", 2:4, 5:7)

                pairgen_outer(vs1,vs2; ind1=1) = ind1 > 3 ? nothing : default_pairgen(vs1, vs2; ind1=min(2*ind1, length(vs1)))
                pairgen_inner(vs1,vs2) = default_pairgen(vs1, vs2; ind1 = length(vs1)-1)

                crossfun(args...) = crossoverswap(args...;pairgen = pairgen_inner, strategy=teststrat)

                anames = name.(vertices(ga))
                bnames = name.(vertices(gb))

                ga_new, gb_new = crossover(ga, gb; pairgen=pairgen_outer, crossoverfun=crossfun)
                @test nout.(vertices(ga_new)) == nout_org.(vertices(ga_new))
                @test nout.(vertices(gb_new)) == nout_org.(vertices(gb_new))

                # Originals not impacted
                @test anames == name.(vertices(ga))
                @test bnames == name.(vertices(gb))

                @test name.(vertices(ga_new)) == ["a.in", "b.cv2", "b.cv3", "a.pv1", "b.dv2", "b.dv3"]
                @test name.(vertices(gb_new)) == ["b.in", "b.cv1", "a.cv1", "a.cv2", "b.pv1", "b.dv1", "a.dv1", "a.dv2"]

                @test size(ga_new(ones(Float32, 4,4,3,2))) == (6, 2)
                @test size(gb_new(ones(Float32, 4,4,3,2))) == (6, 2)
            end

            @testset "VertexCrossover" begin
                import NaiveGAflux: default_pairgen

                pairgen_outer(vs1,vs2; ind1=1) = default_pairgen(vs1, vs2; ind1=2ind1)
                pairgen_inner(vs1, vs2) = default_pairgen(vs1, vs2; ind1=length(vs1))

                c = VertexCrossover(CrossoverSwap(;pairgen=pairgen_inner, strategy=teststrat); pairgen=pairgen_outer)

                ga = g("a", 3:6, 7:8)
                gb = g("b", 2:3, 4:10)

                anames = name.(vertices(ga))
                bnames = name.(vertices(gb))

                ga_new, gb_new = c((ga,gb))

                @test nout.(vertices(ga_new)) == nout_org.(vertices(ga_new))
                @test nout.(vertices(gb_new)) == nout_org.(vertices(gb_new))

                # Originals not impacted
                @test anames == name.(vertices(ga))
                @test bnames == name.(vertices(gb))

                @test name.(vertices(ga_new)) == ["a.in", "a.cv1", "b.cv2", "a.cv3", "a.cv4", "a.pv1", "b.dv6", "a.dv2"]
                @test name.(vertices(gb_new)) == ["b.in", "b.cv1", "a.cv2", "b.pv1", "b.dv1", "b.dv2", "b.dv3", "b.dv4", "b.dv5", "a.dv1", "b.dv7"]

                @test nout.(vertices(ga_new)) == nout_org.(vertices(ga_new))
                @test mapreduce(nin, vcat, vertices(ga_new)) == mapreduce(nin_org, vcat, vertices(ga_new))

                @test nout.(vertices(gb_new)) == nout_org.(vertices(gb_new))
                @test mapreduce(nin, vcat, vertices(gb_new)) == mapreduce(nin_org, vcat, vertices(gb_new))

                NaiveNASflux.forcemutation(ga_new)
                NaiveNASflux.forcemutation(gb_new)

                @test nout(v4n(ga_new, "b.cv2")) == nout(layer(v4n(ga_new, "b.cv2")))
                @test nin(v4n(ga_new, "a.cv3")) == [nin(layer(v4n(ga_new, "a.cv3")))]

                @test nout(v4n(ga_new, "b.dv6")) == nout(layer(v4n(ga_new, "b.dv6")))
                @test nin(v4n(ga_new, "a.dv2")) == [nin(layer(v4n(ga_new, "a.dv2")))]

                @test nout(v4n(gb_new, "a.cv2")) == nout(layer(v4n(gb_new, "a.cv2")))
                @test nin(v4n(gb_new, "b.dv1")) == [nin(layer(v4n(gb_new, "b.dv1")))]

                @test nout(v4n(gb_new, "a.dv1")) == nout(layer(v4n(gb_new, "a.dv1")))
                @test nin(v4n(gb_new, "b.dv7")) == [nin(layer(v4n(gb_new, "b.dv7")))]

                @test size(ga_new(ones(Float32, 4,4,3,2))) == (8,2)
                @test size(gb_new(ones(Float32, 4,4,3,2))) == (10,2)
            end
        end

        @testset "Branchy graph" begin
            function forks(vbase, vf, vs, np)
                fin = map(enumerate(vs)) do (i, s)
                    vf(vbase, s, "$np$i")
                end

                return map(enumerate(fin)) do (i, fstart)
                    foldl(enumerate(vs[1:end-i]); init=fstart) do vin, (i,s)::Tuple
                        vf(vin, s, "$(name(fstart)).$i")
                    end
                end
            end

            function g(np, cvs, dvs)
                vi = iv(np)
                cvbase = cv(vi, 3, "$np.cvbbase")

                cvfa = forks(cvbase, cv, cvs, "$np.cvfa")
                m1 = concat("$np.m1", cvfa...)

                cvfb = map(enumerate(forks(m1, cv, cvs, "$np.cvfb"))) do (i, vin)
                    cv(vin, minimum(cvs), "$(name(vin)).out")
                end
                m2 = +("$np.m2" >> cvfb[1], cvfb[2:end]...)

                pv1 = gv(m2, "$np.pv1")
                dvn = foldl(enumerate(dvs); init=pv1) do vin, (i,s)::Tuple
                    dv(vin, s, "$np.dv$i")
                end
                return CompGraph(vi, dvn)
            end

            @testset "Crossover with MutationProbability and NeuronSelectMutation" begin
                import NaiveGAflux: default_pairgen

                pairgen_inner(vs1, vs2) = default_pairgen(vs1, vs2; ind1=length(vs1))

                # NeuronSelectMutation doesn't really do anything. It is just to test the code path until a use case for it materializes. I should probably just have deleted the methods instead when no longer needed...
                c = VertexCrossover(MutationProbability(PostMutation(NeuronSelectMutation(CrossoverSwap(;pairgen=pairgen_inner, strategy=teststrat)), neuronselect), Probability(0.2, MockRng([0.3, 0.1, 0.3]))))

                ga = g("a", 3:6, 7:8)
                gb = g("b", 2:3, 4:10)

                anames = name.(vertices(ga))
                bnames = name.(vertices(gb))

                ga_new, gb_new = c((ga,gb))

                # Originals not impacted
                @test anames == name.(vertices(ga))
                @test bnames == name.(vertices(gb))

                @test unique(first.(split.(name.(vertices(ga_new)), '.'))) == ["a" ,"b"]
                @test unique(first.(split.(name.(vertices(gb_new)), '.'))) == ["b" ,"a"]

                @test nout.(vertices(ga_new)) == nout_org.(vertices(ga_new))
                @test mapreduce(nin, vcat, vertices(ga_new)) == mapreduce(nin_org, vcat, vertices(ga_new))

                @test nout.(vertices(gb_new)) == nout_org.(vertices(gb_new))
                @test mapreduce(nin, vcat, vertices(gb_new)) == mapreduce(nin_org, vcat, vertices(gb_new))

                @test size(ga_new(ones(Float32, 4,4,3,2))) == (8,2)
                @test size(gb_new(ones(Float32, 4,4,3,2))) == (10,2)
            end
        end

        @testset "Crossover conv size mismatch" begin
            pv(in, name) = mutable(name, MaxPool((2,2); stride=(2,2)), in)
            function g(np)
                vi = iv(np)
                cv1 = cv(vi, 4, "$np.cv1")
                pv1 = pv(cv1, "$np.pv1")
                cv2 = cv(pv1, 2, "$np.cv2")

                cva1 = cv(cv2, 3, "$np.cva1")
                cva2 = cv(cva1, 2, "$np.cva2")

                cvb1 = cv(cv2, 4, "$np.cvb1")
                cvb2 = cv(cvb1, 2, "$np.cvb2")

                mv = "$np.mv" >> cva2 + cvb2
                cv3 = cv(mv, 3, "$np.cv3")
                pv2 = pv(cv3, "$np.pv2")
                cv4 = cv(pv2, 4, "$np.cv4")
                cv5 = cv(cv4, 3, "$np.cv5")

                return CompGraph(vi, cv5)
            end

            @testset "Swap branch for graph" begin

                ga = g("a")
                gb = g("b")


                pairgen_outer(vs1, vs2; ind1=1) = ind1 == 1 ? (5, 9) : nothing

                # Anonymous type to avoid name collisions with other tests (e.g. structs named DummyRng, DummyRng1, etc..)
                selectfirst = nr -> 1:nr
                Random.randn(rng::typeof(selectfirst), nr) = selectfirst(nr)
                function pairgen_inner(vs1, vs2)
                    # Test that the testcase is working :)
                    @test name.(vs1) == ["a.cva1", "a.cva2"]
                    @test name(first(vs2)) == "b.cv1"
                    @test name(last(vs2)) == "b.cv3"
                    return NaiveGAflux.default_inputs_pairgen(vs1, vs2, 10; ind1=2, rng=selectfirst)
                end

                c = VertexCrossover(CrossoverSwap(;pairgen=pairgen_inner, strategy=teststrat); pairgen=pairgen_outer)

                ga_new, gb_new = c((ga,gb))

                @test name.(vertices(ga_new)) == ["a.in", "a.cv1", "a.pv1", "a.cv2", "a.cva1", "b.cv2", "b.cva1", "b.cva2", "b.cvb1", "b.cvb2", "b.mv", "b.cv3", "a.cvb1", "a.cvb2", "a.mv", "a.cv3", "a.pv2", "a.cv4", "a.cv5"]
                @test name.(vertices(gb_new)) ==  ["b.in", "b.cv1", "b.pv1", "a.cva2", "b.pv2", "b.cv4", "b.cv5"]

                @test size(ga(ones(Float32, 10,10,3,1))) == size(ga_new(ones(Float32, 10,10,3,1)))
                @test size(gb(ones(Float32, 10,10,3,1))) == size(gb_new(ones(Float32, 10,10,3,1)))
            end

            @testset "Swap branch with single vertex" begin

                ga = g("a")
                gb = g("b")

                pairgen_outer(vs1, vs2; ind1=1) = ind1 == 1 ? (5, 11) : nothing

                # Anonymous type to avoid name collisions with other tests (e.g. structs named DummyRng, DummyRng1, etc..)
                selectfirst = nr -> 1:nr
                Random.randn(rng::typeof(selectfirst), nr) = selectfirst(nr)
                function pairgen_inner(vs1, vs2)
                    # Test that the testcase is working :)
                    @test name.(vs1) == ["a.cva1", "a.cva2"]
                    @test name(first(vs2)) == "b.cv1"
                    @test name(last(vs2)) == "b.cv4"
                    return NaiveGAflux.default_inputs_pairgen(vs1, vs2, 10; ind1=2, rng=selectfirst)
                end
                c = VertexCrossover(CrossoverSwap(;pairgen=pairgen_inner, strategy=teststrat); pairgen=pairgen_outer)

                ga_new, gb_new = c((ga,gb))

                @test name.(vertices(ga_new)) == ["a.in", "a.cv1", "a.pv1", "a.cv2", "a.cva1", "b.cv4", "a.cvb1", "a.cvb2", "a.mv", "a.cv3", "a.pv2", "a.cv4", "a.cv5"]
                @test name.(vertices(gb_new)) == ["b.in", "b.cv1", "b.pv1", "b.cv2", "b.cva1", "b.cva2", "b.cvb1", "b.cvb2", "b.mv", "b.cv3", "b.pv2", "a.cva2", "b.cv5"]

                @test size(ga(ones(Float32, 10,10,3,1))) == size(ga_new(ones(Float32, 10,10,3,1)))
                @test size(gb(ones(Float32, 10,10,3,1))) == size(gb_new(ones(Float32, 10,10,3,1)))
            end
        end
    end

    @testset "OptimizerCrossover" begin
        using NaiveGAflux.Flux.Optimise

        prts(o) = typeof(o)
        prts(o::Optimiser) = "$(typeof(o))$(prts.(Tuple(o.os)))"

        @testset "Swap optimizers $(prts(o1)) and $(prts(o2))" for (o1, o2) in (
            (ADAM(), Momentum()),
            (Optimiser(Descent(), WeightDecay()), Optimiser(Momentum(), Nesterov())),
            )
            oc = OptimizerCrossover()
            ooc = OptimizerCrossover(oc)
            @test prts.(oc((o1,o2))) == prts.(ooc((o1,o2))) == prts.((o2, o1))
            @test prts.(oc((o2,o1))) == prts.(ooc((o2,o1))) == prts.((o1, o2))
        end

        @testset "Don't swap shielded" begin
            o1 = ShieldedOpt(Descent())
            o2 = ShieldedOpt(Momentum())
            @test OptimizerCrossover()((o1,o2)) == (o1,o2)
        end

        @testset "Cardinality difference" begin

            @testset "Single opt vs Optimiser" begin
                oc = OptimizerCrossover()
                @test prts.(oc((Descent(), Optimiser(Momentum(), WeightDecay())))) == prts.((Momentum(), Optimiser(Descent(), WeightDecay())))
            end

            @testset "Different size Optimisers" begin
                oc = OptimizerCrossover()
                o1 = Optimiser(Descent(), WeightDecay(), Momentum())
                o2 = Optimiser(ADAM(), ADAMW(), NADAM(), RADAM())

                o1n,o2n = oc((o1,o2))

                @test prts(o1n) == prts(Optimiser(ADAM(), ADAMW(), NADAM()))
                @test prts(o2n) == prts(Optimiser(Descent(), WeightDecay(), Momentum(), RADAM()))
            end
        end

        @testset "LogMutation and MutationProbability" begin
            mplm(c) = MutationProbability(LogMutation(((o1,o2)::Tuple) -> "Crossover between $(prts(o1)) and $(prts(o2))", c), Probability(0.2, MockRng([0.3, 0.1, 0.3])))
            oc = OptimizerCrossover() |> mplm |> OptimizerCrossover

            o1 = Optimiser(Descent(), WeightDecay(), Momentum())
            o2 = Optimiser(ADAM(), ADAGrad(), AdaMax())

            o1n,o2n = @test_logs (:info, "Crossover between WeightDecay and ADAGrad") oc((o1,o2))

            @test typeof.(o1n.os) == [Descent, ADAGrad, Momentum]
            @test typeof.(o2n.os) == [ADAM, WeightDecay, AdaMax]
        end

        @testset "Learningrate crossover" begin
            import NaiveGAflux: learningrate
            @testset "Single opt" begin
                oc = LearningRateCrossover()
                o1,o2 = oc((Descent(0.1), Momentum(0.2)))

                @test typeof(o1) == Descent
                @test o1.eta == 0.2

                @test typeof(o2) == Momentum
                @test o2.eta == 0.1
            end

            @testset "Shielded opt" begin
                oc = LearningRateCrossover()
                o1,o2 = oc((ShieldedOpt(Descent(0.1)), Momentum(0.2)))

                @test typeof(o1) == ShieldedOpt{Descent}
                @test o1.opt.eta == 0.1

                @test typeof(o2) == Momentum
                @test o2.eta == 0.2
            end

            @testset "Optimiser" begin
                oc = LearningRateCrossover()
                o1 = Optimiser(Descent(0.1), Momentum(0.2), WeightDecay(0.1))
                o2 = Optimiser(ADAM(0.3), RADAM(0.4), NADAM(0.5), Nesterov(0.6))

                o1n,o2n = oc((o1,o2))

                @test prts(o1n) == prts(o1)
                @test prts(o2n) == prts(o2)

                @test learningrate.(o1n.os[1:end-1]) == [0.3, 0.4]
                @test learningrate.(o2n.os) == [0.1, 0.2, 0.5, 0.6]

            end
        end
    end
end
