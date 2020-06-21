@testset "Shape" begin
    import NaiveGAflux: ΔShape, ShapeAdd, ShapeMul, ShapeDiv, fshape, revert, combine, combine, filter_noops, ShapeTrace, shapetrace, Δshapes, squashshapes, orderΔshapes

    @testset "ΔShapes" begin

        @testset "ShapeAdd" begin
            @test fshape(ShapeAdd((1,2)), (3,4)) == (4,6)
            @test fshape(combine(ShapeAdd((1,2)), ShapeAdd((-2,-1))), (3,4)) == (2,5)
            @test combine(ShapeAdd((1,2)), ShapeAdd((3,4))) == tuple(ShapeAdd((4,6)))
            @test revert(ShapeAdd((1,2))) == ShapeAdd((-1,-2))
            @test filter_noops(ShapeAdd((0,1,2,3))) == tuple(ShapeAdd((0,1,2,3)))
            @test filter_noops(ShapeAdd((0,0))) == tuple()
        end

        @testset "ShapeMul" begin
            @test fshape(ShapeMul((1,2)), (3,4)) == (3,8)
            @test fshape(combine(ShapeMul((1,2)), ShapeMul((2,3))), (3,4)) == (6,24)
            @test combine(ShapeMul((1,2)), ShapeMul((3,4))) == tuple(ShapeMul((3,8)))
            @test revert(ShapeMul((1,2))) == ShapeDiv((1,2))
            @test filter_noops(ShapeMul((0,1,2,3))) == tuple(ShapeMul((0,1,2,3)))
            @test filter_noops(ShapeMul((1,1))) == tuple()
        end

        @testset "ShapeDiv" begin
            @test fshape(ShapeDiv((1,2)), (3,4)) == (3,2)
            @test fshape(combine(ShapeDiv((1,2)), ShapeDiv((2,3))), (3,4)) == (2,1)
            @test combine(ShapeDiv((1,2)), ShapeDiv((3,4))) == tuple(ShapeDiv((3,8)))
            @test revert(ShapeDiv((1,2))) == ShapeMul((1,2))
            @test filter_noops(ShapeDiv((0,1,2,3))) == tuple(ShapeDiv((0,1,2,3)))
            @test filter_noops(ShapeDiv((1,1))) == tuple()
        end

        @testset "Combine $s1 and $s2" for (s1,s2,exp) in (
            (ShapeDiv(2,2), ShapeMul(2,2), (ShapeDiv(2,2), ShapeMul(2,2))),
            (ShapeMul(2,2), ShapeDiv(2,2), tuple(ShapeMul(1,1))),
            (ShapeDiv(3,3), ShapeMul(2,2), (ShapeDiv(3,3), ShapeMul(2,2))),
            (ShapeMul(3,3), ShapeDiv(2,2), (ShapeMul(3,3), ShapeDiv(2,2))),
            (ShapeDiv(2,2), ShapeMul(3,3), (ShapeDiv(2,2), ShapeMul(3,3))),
            (ShapeMul(2,2), ShapeDiv(3,3), (ShapeMul(2,2), ShapeDiv(3,3))),
            (ShapeDiv(4,4), ShapeMul(2,2), (ShapeDiv(4,4), ShapeMul(2,2))),
            (ShapeMul(4,4), ShapeDiv(2,2), tuple(ShapeMul(2,2))),
            (ShapeDiv(2,2), ShapeMul(4,4), tuple(ShapeDiv(2,2), ShapeMul(4,4))),
            (ShapeMul(2,2), ShapeDiv(4,4), tuple(ShapeMul(2,2), ShapeDiv(4,4)))
            )
            act = combine(s1,s2)
            @test act == exp
            @testset "Same fshape with insize $insize" for insize in 1:5:100
                is = (insize,insize+1)
                @test fshape((s1,s2), is) == fshape(act, is) == fshape(exp, is)
            end
        end

        @testset "Swap shapes" begin
            import NaiveGAflux:  swapΔshape
            @test swapΔshape(ShapeAdd(1,2), ShapeAdd(2,3)) == (ShapeAdd(2,3), ShapeAdd(1,2))

            sasm = ShapeAdd(1,2), ShapeMul(2,3)
            @test swapΔshape(sasm...) == (ShapeMul(2,3), ShapeAdd(2,6))
            @test fshape(swapΔshape(sasm...), (11,12)) == fshape(sasm, (11,12)) == (24,42)
            @testset "swap $(typeof.(sasm)) size $s" for s in 1:10
                @test fshape(swapΔshape(sasm...), (s,s)) == fshape(sasm, (s,s))
            end

            @test swapΔshape(ShapeMul(2,3), ShapeAdd(1,2)) == (ShapeMul(2,3), ShapeAdd(1,2))
            @test swapΔshape(ShapeMul(2,3), ShapeAdd(4,12)) == (ShapeAdd(2, 4), ShapeMul(2,3))

            sdsa = ShapeDiv(2,3), ShapeAdd(1,2)
            @test swapΔshape(sdsa...) == (ShapeAdd(2,6), ShapeDiv(2,3))
            @test fshape(swapΔshape(sdsa...), (13,14)) == fshape(sdsa, (13,14)) == (8,7)
            @testset "swap $(typeof.(sdsa)) size $s" for s in 1:10
                @test fshape(swapΔshape(sdsa...), (s,s)) == fshape(sdsa, (s,s))
            end

            @test swapΔshape(ShapeAdd(1,2), ShapeDiv(2,3)) == (ShapeAdd(1,2), ShapeDiv(2,3))
            @test swapΔshape(ShapeAdd(4,12), ShapeDiv(2,3)) == (ShapeDiv(2,3), ShapeAdd(2, 4))

            @test swapΔshape(ShapeMul(2,4), ShapeDiv(4,8)) == (ShapeMul(2,4), ShapeDiv(4,8))
        end

        @testset "Order shapes" begin
            @test orderΔshapes(tuple(ShapeAdd(1,2))) == tuple(ShapeAdd(1,2))
            @test orderΔshapes((ShapeAdd(1,2), ShapeMul(2,3), ShapeAdd(4,6))) == (ShapeAdd(2,2), ShapeAdd(1,2), ShapeMul(2,3))

            @testset "Mixed ShapeAdds" begin
                s = (ShapeAdd(1,2), ShapeDiv(2,3), ShapeMul(2,2), ShapeAdd(4,4), ShapeAdd(3,5), ShapeAdd(6,8), ShapeMul(2,3), ShapeMul(1,1), ShapeAdd(3,7))
                os =  orderΔshapes(s)
                @test os == (ShapeAdd{2}((6, 12)), ShapeAdd{2}((1, 2)), ShapeAdd{2}((4, 6)), ShapeDiv{2}((2, 3)), ShapeMul{2}((2, 2)), ShapeAdd{2}((3, 5)), ShapeMul{2}((2, 3)), ShapeMul{2}((1, 1)), ShapeAdd{2}((3, 7)))

                @testset "same shape with input size $insize" for insize in 1:10
                    @test fshape(s, (insize,insize)) == fshape(os, (insize,insize))
                end
            end

            @testset "Single ShapeAdd with ShapeMul and ShapeDiv" begin
                s = (ShapeMul(2,2), ShapeDiv(12,12), ShapeAdd(2,3), ShapeMul(2,2), ShapeMul(3,3))
                os = orderΔshapes(s)
                @test os == (ShapeMul(2,2), ShapeDiv(12,12), ShapeMul(3,3), ShapeMul(2,2), ShapeAdd(12,18))
                @testset "same shape with input size $insize" for insize in 1:5:100
                    @test fshape(s, (insize,insize)) == fshape(os, (insize,insize))
                end
            end
        end

        @testset "squash shapes" begin
            @test squashshapes(ShapeAdd(1,2)) == tuple(ShapeAdd(1,2))
            @test squashshapes(ShapeAdd(1,2), ShapeAdd(3,4)) == tuple(ShapeAdd(4,6))
            @test squashshapes(ShapeAdd(1,2), ShapeAdd(3,4), ShapeAdd(5,6)) == tuple(ShapeAdd(9,12))
            @test squashshapes(ShapeAdd(1,2), ShapeAdd(3,4), ShapeAdd(-4,-6)) == tuple()
            @test squashshapes(ShapeAdd(1,2), ShapeMul(3,4)) == (ShapeAdd(1,2), ShapeMul(3,4))
            as = (ShapeAdd(1,2), ShapeMul(3,4))
            sa = revert(as)
            @test squashshapes((as..., sa...)) == tuple()
            @test squashshapes(ShapeAdd(4,2), ShapeDiv(2,2), ShapeMul(2,2), ShapeAdd(-4,-2)) == (ShapeDiv(2,2), ShapeMul(2,2))

            @testset "Squash mix add first" begin
                s = (ShapeAdd(2,3), ShapeMul(2,2), ShapeMul(2,2), ShapeMul(3,3), ShapeDiv(12,12))
                sq = squashshapes(s)
                @test sq == tuple(ShapeAdd(2,3))
                @testset "squashed shape with input size $insize" for insize in 1:5:100
                    @test fshape(s, (insize, insize)) == fshape(sq, (insize,insize))
                end
            end

            @testset "Squash mix add last" begin
                s = (ShapeMul(2,2), ShapeMul(2,2), ShapeMul(3,3), ShapeDiv(12,12), ShapeAdd(2,3))
                sq = squashshapes(s)
                @test sq == tuple(ShapeAdd(2,3))
                @testset "squashed shape with input size $insize" for insize in 1:5:100
                    @test fshape(s, (insize, insize)) == fshape(sq, (insize,insize))
                end
            end

            @testset "Squash mix add mid" begin
                s = (ShapeMul(2,2), ShapeMul(2,2), ShapeAdd(2,3), ShapeMul(3,3), ShapeDiv(12,12))
                sq = squashshapes(s)
                @test sq == (ShapeMul(12, 12), ShapeAdd(6, 9), ShapeDiv(12, 12))
                @testset "squashed shape with input size $insize" for insize in 1:5:100
                    @test fshape(s, (insize, insize)) == fshape(sq, (insize,insize))
                end
            end

            @testset "Squash mix add and div mid" begin
                s = (ShapeMul(2,2), ShapeMul(2,2), ShapeDiv(12,12), ShapeAdd(2,3), ShapeMul(3,3))
                sq = squashshapes(s)
                @test sq == (ShapeMul(4, 4), ShapeDiv(12,12), ShapeMul(3,3), ShapeAdd(6, 9))
                @testset "squashed shape with input size $insize" for insize in 1:5:100
                    @test fshape(s, (insize, insize)) == fshape(sq, (insize,insize))
                end
            end
        end

        @testset "Diff shapes" begin
            import NaiveGAflux: Δshapediff

            @testset "Primitives" begin
                @test Δshapediff(ShapeAdd(2,2), ShapeAdd(2,2)) == tuple()
                @test Δshapediff(ShapeAdd(2,3), ShapeAdd(3,4)) == tuple(ShapeAdd(-1,-1))

                @test Δshapediff(ShapeMul(2,2), ShapeMul(2,2)) == tuple()
                @test Δshapediff(ShapeMul(2,3), ShapeMul(3,4)) == (ShapeDiv(3,4), ShapeMul(2,3))

                @test Δshapediff(ShapeDiv(2,2), ShapeDiv(2,2)) == tuple()
                @test Δshapediff(ShapeDiv(2,3), ShapeDiv(3,4)) == (ShapeMul(3,4), ShapeDiv(2,3))
                @test Δshapediff(ShapeDiv(2,3), ShapeDiv(4,6)) == tuple(ShapeMul(2,2))
            end

            @testset "Tuples" begin
                @testset "Basic" begin
                    s1 = (ShapeAdd(2,2), ShapeDiv(2,2), ShapeMul(4,2))
                    @test Δshapediff(s1, s1) == tuple()

                    @test Δshapediff(s1, s1[1:2]) == tuple(ShapeMul(4,2))
                    @test Δshapediff(s1, s1[2:3]) == tuple(ShapeAdd(4,2))
                    @test Δshapediff(s1, s1[[1,3]]) == (ShapeDiv(8, 4), ShapeMul(4, 2))

                    @test Δshapediff(s1[1:2], s1) == tuple(ShapeDiv(4,2))
                    @test Δshapediff(s1[2:3], s1) == tuple(ShapeAdd(-4,-2))
                    @test Δshapediff(s1[[1,3]], s1) == (ShapeDiv(4, 2), ShapeMul(8, 4))

                    s2 = (ShapeDiv(4,4), ShapeAdd(3,5), ShapeMul(8,4), ShapeDiv(2,2))
                    @test Δshapediff(s1,s2) == (ShapeDiv(4, 2), ShapeMul(8, 4), ShapeAdd(-20, -18))
                end
            end
        end
    end

    @testset "ShapeTrace" begin
        iv(N=2) = inputvertex("in", 1, FluxConv{N}())
        bv(in, name) = mutable(name, BatchNorm(nout(in)), in)
        pv(in, name; ks=(3,3), stride=ntuple(i->1, length(ks)), kwargs...) = mutable(name, MaxPool(ks; stride=stride, kwargs...), in)
        cv(in, name; ks=(3,3), kwargs...) = mutable(name, Conv(ks, nout(in) => nout(in); kwargs...), in)

        @testset "Trace $vf" for vf in (cv,pv)
            vi = iv()
            v1 = vf(vi, "v1"; ks=(1,1))
            v2 = vf(v1, "v2"; ks=(3,3))
            v3 = vf(v2, "v3"; ks=(2,4), pad=(5,7))
            v4 = vf(v3, "v4"; ks=(3,3), stride=(1, 2))
            v5 = vf(v4, "v5"; ks=(4,5), pad=(2,3,4,5), stride=(3,4))

            @test filter_noops(Δshapes(v1)) == tuple()
            @testset "shape for $(name(vn))" for vn in (v2,v3,v4,v5)
                s = ShapeTrace(vn).trace
                @test fshape(s, (10,9)) == size(vn(ones(Float32, 10,9, nout(vi), 1)))[1:2]
            end

            @testset "shapetrace graph" begin
                g = CompGraph(vi, v5)
                sg = g(ShapeTrace(vi)).trace
                sv = shapetrace(v5; trfun = v -> ShapeTrace(v)).trace
                @test fshape(sg, (30,31)) == fshape(sv, (30,31))== size(g(ones(Float32, 30,31, nout(vi), 1)))[1:2]
            end
        end

        @testset "Trace $vf merge" for vf in (cv, pv)
            vi = iv()
            v1 = vf(vi, "v1"; ks=(2,5))
            va1 = vf(v1, "va1"; ks=(1,1), stride=(2,2))
            va2 = vf(va1, "va2"; ks=(2,2))
            vb1 = vf(v1, "vb1"; ks=(3,3))
            vb2 = vf(vb1, "vb2";ks=(1,1), stride=(2,2))
            mv = concat("concat", va2,vb2)
            v2 = vf(mv, "v2"; ks=(3,3))

            tr = shapetrace(v2)

            @test fshape(squashshapes(tr), (13,17)) == size(CompGraph(vi,v2)(ones(Float32, 13,17,1,1)))[1:2]
        end

        @testset "Conv dilation" begin
            vi = iv()
            v1 = cv(vi, "v1"; ks=(1,1), dilation=2)
            v2 = cv(v1, "v2"; ks=(2,2), dilation=2)
            v3 = cv(v2, "v3"; ks=(3,4), dilation=(4,5), pad=(1,2,3,4))
            v4 = cv(v3, "v4"; ks=(1,2), dilation=(2,3), pad=(4,5), stride=(6,7))

            @test filter_noops(Δshapes(v1)) == tuple()
            @testset "shape for $(name(vn))" for vn in (v2,v3,v4)
                s = ShapeTrace(vn).trace
                @test fshape(s, (20,19)) == size(vn(ones(Float32, 20,19, nout(vi), 1)))[1:2]
            end

            @testset "shapetrace graph" begin
                g = CompGraph(vi, v4)
                sg = g(ShapeTrace(vi)).trace
                sv = shapetrace(v4; trfun = v -> ShapeTrace(v)).trace
                @test fshape(sg, (30,31)) == fshape(sv, (30,31))== size(g(ones(Float32, 30,31, nout(vi), 1)))[1:2]
            end
        end

        @testset "1D  $vf" for vf in (cv,pv)
            vi = iv(1)
            v1 = vf(vi, "v1"; ks=(1,), pad=1)
            v2 = vf(v1, "v2", ks=(3,), stride=(2,), pad=(1,1))

            @testset "shape for $(name(vn))" for vn in (v1, v2)
                s = ShapeTrace(vn).trace
                @test fshape(s, (7,)) == size(vn(ones(Float32, 7, nout(vi), 1)))[1:1]
            end
        end

        @testset "3D  $vf" for vf in (cv,pv)
            vi = iv(3)
            v1 = vf(vi, "v1"; ks=(1,2,3), pad=1)
            v2 = vf(v1, "v2", ks=(3,3,3), stride=(2,1,3), pad=(1,2,3,4,5,6))

            @testset "shape for $(name(vn))" for vn in (v1, v2)
                s = ShapeTrace(vn).trace
                @test fshape(s, (17,11,20)) == size(vn(ones(Float32, 17, 11, 20, nout(vi), 1)))[1:3]
            end
        end

        @testset "Squash misaligned branches" begin
            vi = iv()
            v1 = pv(vi, "v1"; ks=(2,2))
            va1 = pv(v1, "va1"; ks=(1,1), stride=(2,2))
            va2 = pv(va1, "va2"; ks=(3,3))
            vb1 = pv(v1, "vb1"; ks=(3,3), pad=(2,2))
            vb2 = pv(vb1, "vb2"; ks=(9,9))
            vb3 = pv(vb2, "vb3"; ks=(3,3))
            av = "add" >> va2 + vb3
            v2 = pv(av, "v2"; ks=(2,3))

            g = CompGraph(vi ,v2)
            @test size(g(ones(Float32, 13,14, nout(vi), 1)))[1:2] == (3,3)

            trv2 = shapetrace(v2)
            sv2 = squashshapes(trv2)

            trva2 = shapetrace(va2)
            sva2 = squashshapes(trva2)

            trvb3 = shapetrace(vb3)
            svb3 = squashshapes(trvb3)

            @test sv2[1][1] == sva2
            @test sv2[1][2] == svb3
            @test sv2[2] == filter_noops(Δshapes(v2))

            @test fshape((sva2...,sv2[2]...), (13,14)) == (3,3)
            @test fshape((svb3...,sv2[2]...), (13,14)) == (3,3)

            @test fshape((sva2...,sv2[2]...), (16,17)) == (5,4)
            @test fshape((svb3...,sv2[2]...), (16,17)) == (6,6)
        end

        @testset "Squash nested branches" begin

            # The fact that I could use the shapetrace to align this monstrosity is perhaps a sign that it is not completely useless...
            vi = iv()
            v1 = pv(vi, "v1"; ks=(3,3))

            va1 = pv(v1, "va1"; ks=(2,2), stride=(2,2))
            va2 = pv(va1, "va2"; ks=(5,5))

            vaa1 = pv(va2, "vaa1"; ks=(1,1), stride=(3,3))
            vaa2 = pv(vaa1, "vaa2"; ks=(3,3))
            vab1 = pv(va2, "vab1"; ks=(7,7), stride=(3,3))

            mva1 = "mva1" >> vaa2 + vab1
            va3 = pv(mva1, "va3"; ks=(5,5))

            vb1 = pv(v1, "vb1"; ks=(4,4), stride=(3,3))
            vb2 = pv(vb1, "vb2"; ks=(3,3), stride=(2,2))
            vb3 = pv(vb2, "vb3"; ks=(2,2))
            vb4 = pv(vb3, "vb4"; ks=(6,6))

            mv1 = "mv1" >> va3 + vb4
            v2 = pv(mv1, "v2"; ks=(3,3))

            @test squashshapes(shapetrace(v2)) == squashshapes(shapetrace(v2, v1)) == (ShapeAdd(-59, -59), ShapeDiv(6, 6))
        end

        @testset "Squash with different start types" begin
            import Setfield: @set
            import NaiveGAflux: allΔshapetypes
            vi = iv()

            va1 = pv(vi, "va1"; ks=(3,3), stride=(2,2), pad=(1,1))
            va2 = pv(va1, "va2"; ks = (3,3))

            vb1 = pv(vi, "vb1"; ks=(5,5))
            vb2 = pv(vb1,"vb1"; ks=(3,3), stride=(2,2), pad=(1,1))
            mv = "add" >> va2 + vb2

            st = shapetrace(mv);
            # All vertices insert ΔShapes in the same order, so we need to do this to make them have different order
            st = Setfield.@set st.trace[1][1].trace = filter_noops(st.trace[1][1].trace)

            @test allΔshapetypes(st) == [ShapeDiv{2}, ShapeAdd{2}]

            @test squashshapes(st) == (ShapeDiv(2, 2), ShapeAdd(-2,-2))
        end
    end
end
