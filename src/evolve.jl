"""
    AbstractEvolution

Abstract base type for strategies for how to evolve a population into a new population.
"""
abstract type AbstractEvolution end

"""
    evolve!(e::AbstractEvolution, population::AbstractArray{<:AbstractCandidate})

Evolve `population` into a new population. New population may or may not contain same individuals as before.
"""
function evolve! end

"""
    NoOpEvolution <: AbstractEvolution
    NoOpEvolution()

Does not evolve the given population.
"""
struct NoOpEvolution <: AbstractEvolution end
evolve!(::NoOpEvolution, pop) = pop

"""
    AfterEvolution <: AbstractEvolution
    AfterEvolution(evo::AbstractEvolution, fun::Function)

Return `fun(newpop)` where `newpop = evolve!(e.evo, pop)` where `pop` is original population to evolve.
"""
struct AfterEvolution <: AbstractEvolution
    evo::AbstractEvolution
    fun::Function
end

function evolve!(e::AfterEvolution, pop)
    newpop = evolve!(e.evo, pop)
    return e.fun(newpop)
end

"""
    ResetAfterEvolution(evo::AbstractEvolution)

Alias for `AfterEvolution` with `fun == reset!`.
"""
ResetAfterEvolution(evo::AbstractEvolution) = AfterEvolution(evo, resetandreturn)
function resetandreturn(pop)
    foreach(reset!, pop)
    return pop
end

"""
    EliteSelection <: AbstractEvolution
    EliteSelection(nselect::Integer)

Selects the only the `nselect` highest fitness candidates.
"""
struct EliteSelection <: AbstractEvolution
    nselect::Integer
end
evolve!(e::EliteSelection, pop::AbstractArray{<:AbstractCandidate}) = partialsort(pop, 1:e.nselect, by=fitness, rev=true)

"""
    SusSelection <: AbstractEvolution
    SusSelection(nselect::Integer, evo::AbstractEvolution, rng=rng_default)

Selects candidates for further evolution using stochastic universal sampling.
"""
struct SusSelection <: AbstractEvolution
    nselect::Integer
    evo::AbstractEvolution
    rng
    SusSelection(nselect, evo, rng=rng_default) = new(nselect, evo, rng)
end

function evolve!(e::SusSelection, pop::AbstractArray{<:AbstractCandidate})
    csfitness = cumsum(fitness.(pop))

    gridspace = csfitness[end] / e.nselect
    start = rand(e.rng)*gridspace

    selected = similar(pop, e.nselect)
    for i in eachindex(selected)
        candind = findfirst(x -> x >= start + (i-1)*gridspace, csfitness)
        selected[i] = pop[candind]
    end
    return evolve!(e.evo, selected)
end

"""
    CombinedEvolution <: AbstractEvolution
    CombinedEvolution(evos::AbstractArray{<:AbstractEvolution})
    CombinedEvolution(evos::AbstractEvolution...)

Combines the evolved populations from several `AbstractEvolutions` into one population.
"""
struct CombinedEvolution <: AbstractEvolution
    evos::AbstractArray{<:AbstractEvolution}
end
CombinedEvolution(evos::AbstractEvolution...) = CombinedEvolution(collect(evos))
evolve!(e::CombinedEvolution, pop) = mapfoldl(evo -> evolve!(evo, pop), vcat, e.evos)

"""
    EvolveCandidates <: AbstractEvolution
    EvolveCandidates(fun::Function)

Applies `fun` for each candidate in a given population.

Useful with [`evolvemodel`](@ref).
"""
struct EvolveCandidates <: AbstractEvolution
    fun::Function
end
evolve!(e::EvolveCandidates, pop) = map(e.fun, pop)
