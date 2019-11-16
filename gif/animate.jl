


function animatepopopt(dir::String)
  Plots.scalefontsizes()
  sctrpoplim = (args...;kwargs...) -> scatter(args...;kwargs...,xlims=(1, 100), ylims=(0,1), clims=(4, 9))
  sctroptlim = (args...;kwargs...) -> scatter(args...;kwargs...,xlims=(-6, 0), ylims=(0,1))

  scp = ScatterPop(sctrpoplim, dir);
  sco = ScatterOpt(sctroptlim, dir);
  l = @layout [a b]
  mpl = MultiPlot((ps...) -> plot(ps...; layout=(2,1), size=(600,800)), scp, sco; init=false)

  anim = @animate for i in eachindex(scp.data)
    pl = NaiveGAflux.plotgen(mpl, i)
    pl.subplots[1].attr[:title] = "Generation $i"
  end
  gif(anim, joinpath(dir, "evo.gif"), fps = 2)
end
