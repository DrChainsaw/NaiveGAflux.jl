@testset "Shape" begin
    import NaiveGAflux: ΔShape, ShapeAdd, ShapeMul, ShapeDiv, AggΔShape, fshape, revert, combine, combine, filter_noops, ShapeTraceV0, shapetrace, shapequery, squashshapes, orderΔshapes

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
    end

    @testset "ShapeTrace" begin
        iv(N=2) = inputvertex("in", 1, FluxConv{N}())
        bv(in, name) = mutable(name, BatchNorm(nout(in)), in)
        pv(in, name; ks = (3,3), kwargs...) = mutable(name, MaxPool(ks; kwargs...), in)
        cv(in, name; ks=(3,3), kwargs...) = mutable(name, Conv(ks, nout(in) => nout(in); kwargs...), in)

        @testset "Trace conv" begin
            vi = iv()
            cv1 = cv(vi, "cv1"; ks=(1,1))
            cv2 = cv(cv1, "cv2"; ks=(3,3))
            cv3 = cv(cv2, "cv3"; ks=(2,4), pad=(5,7))
            cv4 = cv(cv3, "cv4"; ks=(3,3), stride=(1, 2))
            cv5 = cv(cv4, "cv5"; ks=(4,5), pad=(2,3,4,5), stride=(3,4))

            @test shapequery(trait(cv1), cv1, ShapeTraceV0(cv1)).trace == tuple()
            @testset "shape for $(name(cvn))" for cvn in (cv2,cv3,cv4,cv5)
                s = cvn(ShapeTraceV0(cvn)).trace
                @test fshape(s, (10,9)) == size(cvn(ones(Float32, 10,9, nout(vi), 1)))[1:2]
            end

            @testset "shapetrace" begin
                g = CompGraph(vi, cv5)
                sg = g(ShapeTraceV0(vi)).trace
                sv = shapetrace(cv5; trfun = v -> ShapeTraceV0(v)).trace
                @test fshape(sg, (30,31)) == fshape(sv, (30,31))== size(g(ones(Float32, 30,31, nout(vi), 1)))[1:2]
            end
        end

        @testset "Trace conv merge" begin
            vi = iv()
            cv1 = cv(vi, "cv1"; ks=(2,5))
            cva1 = cv(cv1, "cva1"; ks=(1,1), stride=(2,2))
            cva2 = cv(cva1, "cva2"; ks=(2,2))
            cvb1 = cv(cv1, "cvb1"; ks=(3,3))
            cvb2 = cv(cvb1, "cvb2";ks=(1,1), stride=(2,2))
            mv = concat("concat", cva2,cvb2)
            cv2 = cv(mv, "cv2"; ks=(3,3))

            tr = shapetrace(cv2)
            # @show name(tr.origin)
            # for tt in  tr.trace
            #     @show tt
            # end

        end
    end
end
