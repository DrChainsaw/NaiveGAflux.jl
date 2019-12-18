


loadifpresent(filename, default=Float32[]) = isfile(filename) ? deserialize(filename) : default

"""
    PlotFitness(plotfun, rootdir, subdir="PlotFitness")

Plots best and average fitness for each generation.

Also serializes data so that plotting can be resumed if evolution is aborted.

# Examples
```julia-repl
julia> using NaiveGAflux, Plots

julia> gr();

julia> cb=PlotFitness(plot, "models/test");
```
"""
struct PlotFitness
    best::Vector{Float32}
    avg::Vector{Float32}
    plt
    basedir
end

function PlotFitness(plotfun, rootdir, subdir="PlotFitness")
    basedir = joinpath(rootdir, subdir)
    best = loadifpresent(joinpath(basedir, "best.jls"))
    avg = loadifpresent(joinpath(basedir, "avg.jls"))
    plt = plotfun(hcat(best,avg), label=["Best" "Avg"], xlabel="Generation", ylabel="Fitness", m=[:circle, :circle], legend=:bottomright)
    return PlotFitness(best, avg, plt, basedir)
end

function plotfitness(p::PlotFitness, population)
    fits = fitness.(population)
    best = maximum(fits)
    avg = mean(fits)
    push!(p.best, best)
    push!(p.avg, avg)
    push!(p.plt, length(p.best), [best, avg])
end

plotgen(p::PlotFitness, gen=length(p.best)) = p.plt # Plot already loaded with data...

function (p::PlotFitness)(population)
    plotfitness(p, population)
    mkpath(p.basedir)
    serialize(joinpath(p.basedir, "best.jls"), p.best)
    serialize(joinpath(p.basedir, "avg.jls"), p.avg)
    return p.plt
end

"""
    ScatterPop(plotfun, rootdir, subdir="ScatterPop")

Scatter plot of number of vertices in model vs fitness vs number of parameters for each candidate.

Also serializes data so that plotting can be resumed if evolution is aborted.

# Examples
```julia-repl
julia> using NaiveGAflux, Plots

julia> cb=ScatterPop(scatter, "models/test");
```
"""
struct ScatterPop
    plotfun
    data::Vector{Array{Float32, 2}}
    basedir
end

function ScatterPop(plotfun, rootdir::AbstractString, subdir="ScatterPop")
    basedir = joinpath(rootdir, subdir)
    data = loadifpresent(joinpath(basedir, "nvfitnp.jls"), Vector{Array{Float32, 2}}(undef, 0))
    return ScatterPop(plotfun, data, basedir)
end

function plotfitness(p::ScatterPop, population)
    fits = fitness.(population)
    nverts = nv.(graph.(population))
    npars = nparams.(population)

    push!(p.data, hcat(nverts, fits, npars))
    plotgen(p)
end

function plotgen(p::ScatterPop, gen=length(p.data))
    gen == 0 && return p.plotfun()
    data = p.data[gen]
    nverts = data[:,1]
    fits = data[:,2]
    npars = data[:,3]
    return p.plotfun(nverts, fits, zcolor=log10.(npars), m=(:heat, 0.8), xlabel="Number of vertices", ylabel="Fitness", colorbar_title="Number of parameters (log10)", label="")
end

function(p::ScatterPop)(population)
    plt = plotfitness(p, population)
    mkpath(p.basedir)
    serialize(joinpath(p.basedir, "nvfitnp.jls"), p.data)
    return plt
end

"""
    ScatterOpt(plotfun, rootdir, subdir="ScatterOpt")

Scatter plot of learning rate vs fitness vs optimizer type for each candidate.

Also serializes data so that plotting can be resumed if evolution is aborted.

# Examples
```julia-repl
julia> using NaiveGAflux, Plots

julia> cb=ScatterOpt(scatter, "models/test");
```
"""
struct ScatterOpt
    plotfun
    data::Vector{Array{Any, 2}}
    basedir
end

function ScatterOpt(plotfun, rootdir::AbstractString, subdir="ScatterOpt")
    basedir = joinpath(rootdir, subdir)
    data = loadifpresent(joinpath(basedir, "fitlropt.jls"), Vector{Array{Any, 2}}(undef, 0))
    return ScatterOpt(plotfun, data, basedir)
end

function plotfitness(p::ScatterOpt, population)
    fits = fitness.(population)
    opts = opt.(population)
    lrs = map(o -> lr(o), opts)
    ots = map(o -> ot(o), opts)

    push!(p.data, hcat(fits, lrs, ots))
    plotgen(p)
end

opt(c::AbstractCandidate) = opt(c.c)
opt(c::CandidateModel) = c.opt
lr(o::Flux.Optimise.Optimiser) = mean(map(oo -> lr(oo), o.os))
lr(o) = o.eta
ot(o::Flux.Optimise.Optimiser) = ot(o.os[1])
ot(o) = typeof(o)


function plotgen(p::ScatterOpt, gen = length(p.data))
    gen == 0 && return p.plotfun()
    data = p.data[gen]
    fits = data[:,1]
    lrs = data[:,2]
    ots = map(ostr -> last(split(ostr, ".")), string.(data[:,3]))

    uots = sort(unique(ots))
    inds = map(o -> o .== ots, uots)

    fitso = map(indv -> fits[indv], inds)
    lrso = map(indv -> log10.(lrs[indv]), inds)

    return p.plotfun(lrso, fitso, xlabel="Learning rate (log10)", ylabel="Fitness", label=reshape(uots,1,:), legend=:outerright, legendfontsize=5)
end

function(p::ScatterOpt)(population)
    plt = plotfitness(p, population)
    mkpath(p.basedir)
    serialize(joinpath(p.basedir, "fitlropt.jls"), p.data)
    return plt
end

"""
    MultiPlot(plotfun, plts...)

Multiple plots in the same figure.

# Examples
```julia-repl
julia> using NaiveGAflux, Plots

julia> pdir = "models/test"

julia> cb = MultiPlot(display ∘ plot, PlotFitness(plot, pdir), ScatterPop(scatter, pdir), ScatterOpt(scatter, pdir)));
```
"""
struct MultiPlot
    plotfun
    plts
end
function MultiPlot(plotfun, plts...;init=true)
    mp = MultiPlot(plotfun, plts)
    if init
        plotgen(mp)
    end
    return mp
end

(p::MultiPlot)(population) = p.plotfun(map(pp -> pp(population), p.plts)...)

plotgen(p::MultiPlot) = p.plotfun(map(pp -> plotgen(pp), p.plts)...)
plotgen(p::MultiPlot, gen) = p.plotfun(map(pp -> plotgen(pp, gen), p.plts)...)

"""
    struct CbAll
    CbAll(cbs...)

Aggregates several callbacks into one struct.

# Examples
```julia-repl
julia> using NaiveGAflux, Plots

julia> pdir = "models/test"

julia> cb = CbAll(persist, MultiPlot(display ∘ plot, PlotFitness(plot, pdir), ScatterPop(scatter, pdir), ScatterOpt(scatter, pdir)))
```
"""
struct CbAll
    cbs
end
CbAll(cbs...) = CbAll(cbs)

(cba::CbAll)(population) = foreach(cb -> cb(population), cba.cbs)
