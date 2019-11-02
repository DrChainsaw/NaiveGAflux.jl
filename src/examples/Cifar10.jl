module Cifar10

using ..NaiveGAflux

 # For longer term storage of models
using FileIO
using JLD2

export run_experiment, iterators

export PlotFitness, ScatterPop, ScatterOpt, MultiPlot, CbAll

defaultdir(this="CIFAR10") = joinpath(NaiveGAflux.modeldir, this)

function savemodels(pop::AbstractArray{<:AbstractCandidate}, dir=joinpath(defaultdir(), "models"))
    mkpath(dir)
    for (i, cand) in enumerate(pop)
        model = NaiveGAflux.graph(cand)
        FileIO.save(joinpath(dir, "$i.jld2"), "model$i", cand |> cpu)
    end
end
# Curried version of the above for other dirs than the default
savemodels(dir::AbstractString) = pop -> savemodels(pop,dir)

## Plotting stuff. Maybe move to own file if reusable...

loadifpresent(filename, default=Float32[]) = isfile(filename) ? deserialize(filename) : default

"""
    PlotFitness(plotfun, basedir=joinpath(defaultdir(), "PlotFitness"))

Plots best and average fitness for each generation.

Also serializes data so that plotting can be resumed if evolution is aborted.

# Examples
```julia-repl
julia> using NaiveGAflux, NaiveGAflux.Cifar10, MLDatasets, Plots

julia> gr();

julia> run_experiment(50, iterators(CIFAR10.traindata())...; cb=PlotFitness(plot));
```
"""
struct PlotFitness
    best::Vector{Float32}
    avg::Vector{Float32}
    plt
    basedir
end

function PlotFitness(plotfun, basedir=joinpath(defaultdir(), "PlotFitness"))
    best = loadifpresent(joinpath(basedir, "best.jls"))
    avg = loadifpresent(joinpath(basedir, "avg.jls"))
    plt = plotfun(hcat(best,avg), label=["Best", "Avg"], xlabel="Generation", ylabel="Fitness", m=[:circle, :circle], legend=:bottomright)
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
    ScatterPop(plotfun, basedir=joinpath(defaultdir(), "ScatterPop"))

Scatter plot of number of vertices in model vs fitness vs number of parameters for each candidate.

Also serializes data so that plotting can be resumed if evolution is aborted.

# Examples
```julia-repl
julia> using NaiveGAflux, NaiveGAflux.Cifar10, MLDatasets, Plots

julia> gr();

julia> run_experiment(50, iterators(CIFAR10.traindata())...; cb=ScatterPop(scatter));
```
"""
struct ScatterPop
    plotfun
    data::Vector{Array{Float32, 2}}
    basedir
end

function ScatterPop(plotfun, basedir=joinpath(defaultdir(), "ScatterPop"))
    data = loadifpresent(joinpath(basedir, "nvfitnp.jls"), [zeros(Float32,0,0)])
    return ScatterPop(plotfun, data, basedir)
end

function plotfitness(p::ScatterPop, population)
    fits = fitness.(population)
    nverts = nv.(NaiveGAflux.graph.(population))
    npars = nparams.(population)
    push!(p.data, hcat(nverts, fits, npars))
    plotgen(p)
end

function plotgen(p::ScatterPop, gen=length(p.data))
    gen == 1 && return p.plotfun()
    data = p.data[gen]
    nverts = data[:,1]
    fits = data[:,2]
    npars = data[:,3]
    return p.plotfun(nverts, fits, zcolor=npars/1e6, m=(:heat, 0.8), xlabel="Number of vertices", ylabel="Fitness", colorbar_title="Number of parameters (1e6)", label="")
end

function(p::ScatterPop)(population)
    plt = plotfitness(p, population)
    mkpath(p.basedir)
    serialize(joinpath(p.basedir, "nvfitnp.jls"), p.data)
    return plt
end

"""
    ScatterOpt(plotfun, basedir=joinpath(defaultdir(), "ScatterOpt"))

Scatter plot of learning rate vs fitness vs optimizer type for each candidate.

Also serializes data so that plotting can be resumed if evolution is aborted.

# Examples
```julia-repl
julia> using NaiveGAflux, NaiveGAflux.Cifar10, MLDatasets, Plots

julia> gr();

julia> run_experiment(50, iterators(CIFAR10.traindata())...; cb=ScatterOpt(scatter));
```
"""
struct ScatterOpt
    plotfun
    data::Vector{Array{Any, 2}}
    basedir
end

function ScatterOpt(plotfun, basedir=joinpath(defaultdir(), "ScatterOpt"))
    data = loadifpresent(joinpath(basedir, "fitlropt.jls"), [zeros(0,0)])
    return ScatterOpt(plotfun, data, basedir)
end

function plotfitness(p::ScatterOpt, population)

    opt(c::AbstractCandidate) = opt(c.c)
    opt(c::CandidateModel) = c.opt

    fits = fitness.(population)
    opts = opt.(population)
    lrs = map(o -> o.os[].eta, opts)
    ots = map(o -> typeof(o.os[]), opts)

    push!(p.data, hcat(fits, lrs, ots))
    plotgen(p)
end

function plotgen(p::ScatterOpt, gen = length(p.data))
    gen == 1 && return p.plotfun()
    data = p.data[gen]
    fits = data[:,1]
    lrs = data[:,2]
    ots = data[:,3]

    uots = unique(ots)
    inds = map(o -> o .== ots, uots)

    fitso = map(indv -> fits[indv], inds)
    lrso = map(indv -> log10.(lrs[indv]), inds)

    return p.plotfun(lrso, fitso, xlabel="Learning rate (log10)", ylabel="Fitness", label=string.(uots), legend=:outerright, legendfontsize=5)
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
julia> using NaiveGAflux, NaiveGAflux.Cifar10, MLDatasets, Plots

julia> gr();

julia> run_experiment(50, iterators(CIFAR10.traindata())...; cb=MultiPlot(display ∘ plot, PlotFitness(plot), ScatterPop(scatter), ScatterOpt(scatter)));
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

struct CbAll
    cbs
end
CbAll(cbs...) = CbAll(cbs)

(cba::CbAll)(population) = foreach(cb -> cb(population), cba.cbs)


end  # module cifar10
