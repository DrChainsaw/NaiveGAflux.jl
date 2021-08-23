"""
    AbstractEvolution

Abstract base type for strategies for how to evolve a population into a new population.
"""
abstract type AbstractEvolution end

"""
    evolve(e::AbstractEvolution, population)

Return a new population evolved by `e`.

New population may or may not contain same individuals as before.
"""
evolve(e::AbstractEvolution, population) = _evolve(e, population)

"""
    NoOpEvolution <: AbstractEvolution
    NoOpEvolution()

Does not evolve the given population.
"""
struct NoOpEvolution <: AbstractEvolution end
_evolve(::NoOpEvolution, pop) = pop

"""
    AfterEvolution <: AbstractEvolution
    AfterEvolution(evo, fun)

Return `fun(newpop)` where `newpop = evolve(e.evo, pop)` where `pop` is original population to evolve.
"""
struct AfterEvolution{F, E} <: AbstractEvolution
    evo::E
    fun::F
end

function _evolve(e::AfterEvolution, pop)
    newpop = evolve(e.evo, pop)
    return e.fun(newpop)
end

"""
    EliteSelection <: AbstractEvolution
    EliteSelection(nselect::Integer, evo=NoOpEvolution())

Selects the `nselect` highest fitness candidates to be passed on to `evo`.
"""
struct EliteSelection{N, E} <: AbstractEvolution
    nselect::N
    evo::E
end
EliteSelection(n::Integer) = EliteSelection(n, NoOpEvolution())
_evolve(e::EliteSelection, pop::AbstractArray{<:AbstractCandidate}) = evolve(e.evo, partialsort(pop, 1:e.nselect, by=fitness, rev=true))

"""
    SusSelection <: AbstractEvolution
    SusSelection(nselect, evo, rng=rng_default)

Selects candidates for further evolution using stochastic universal sampling.
"""
struct SusSelection{N, R, E} <: AbstractEvolution
    nselect::N
    evo::E
    rng::R
end
SusSelection(nselect, evo) = SusSelection(nselect, evo, rng_default)

function _evolve(e::SusSelection, pop::AbstractArray{<:AbstractCandidate})
    csfitness = cumsum(fitness.(pop))

    gridspace = csfitness[end] / e.nselect
    start = rand(e.rng)*gridspace

    selected = similar(pop, e.nselect)
    for i in eachindex(selected)
        candind = findfirst(x -> x >= start + (i-1)*gridspace, csfitness)
        selected[i] = pop[candind]
    end
    return evolve(e.evo, selected)
end

"""
    TournamentSelection <: AbstractEvolution
    TournamentSelection(nselect, k, p::Real, evo, rng=rng_default)

Selects candidates for further evolution using tournament selection.

Holds `nselect` tournaments with one winner each where each tournament has `k`
random candidates from the given population.

Winner of a tournament is selected as the candidate with highest fitness with a
probability `p`, second highest fitness with a probability `p(p-1)`, third highest
fitness with a probability of `p((p-1)^2)` and so on.
"""
struct TournamentSelection{N,P,E,R} <: AbstractEvolution
    nselect::N
    k::N
    p::P
    evo::E
    rng::R
    function TournamentSelection(nselect::N, k::N, p::P, evo::E, rng::R=rng_default) where {N, P, E, R}
        @assert 0 <= p <= 1 "0 <= p <= 1 not fulfilled! p = $p"
        return new{N, Vector{P}, E, R}(nselect, k, p .* ((1 - p) .^ collect(0:k-1)), evo, rng)
    end
end

function _evolve(e::TournamentSelection, pop::AbstractArray{<:AbstractCandidate})
    # Step 1: Create a random tournament order with as little repetition as possible so that we can select
    # e.nselect candidates out of e.nselect tournaments with e.k random candidates in each tournament
    n = e.nselect * e.k
    nrem = mod(n, length(pop))
    nrep = max(0, n ÷ length(pop))
    torder = mapfoldl(i -> shuffle(e.rng, pop), vcat, 1:nrep, init = shuffle(e.rng, pop)[1:nrem])

    selected = similar(pop, e.nselect)
    for (i, cands) in enumerate(Iterators.partition(torder, e.k))
        score = rand(e.rng, e.k) .* e.p
        selected[i] = partialsort(cands, argmax(score), by=fitness, rev=true)
    end
    return evolve(e.evo, selected)
end

"""
    CombinedEvolution <: AbstractEvolution
    CombinedEvolution(evos::AbstractArray)
    CombinedEvolution(evos...)

Combines the evolved populations from several `AbstractEvolution`s into one population.
"""
struct CombinedEvolution{E<:AbstractArray} <: AbstractEvolution
    evos::E
end
CombinedEvolution(evos...) = CombinedEvolution(collect(evos))
_evolve(e::CombinedEvolution, pop) = mapfoldl(evo -> evolve(evo, pop), vcat, e.evos)

"""
    EvolutionChain <: AbstractEvolution
    EvolutionChain(evos::AbstractArray)
    EvolutionChain(evos...)


Chains multiple `AbstractEvolution`s in a sequence so that output from the first is input to the next and so on.
"""
struct EvolutionChain{E<:AbstractArray} <: AbstractEvolution
    evos::E
end
EvolutionChain(evos...) = EvolutionChain(collect(evos))
_evolve(e::EvolutionChain, pop) = foldr(evolve, reverse(e.evos); init=pop)

"""
    PairCandidates <: AbstractEvolution
    PairCandidates(evo::AbstractEvolution)

Creates pairs of candidates in a population and calls `evolve(evo, pairs)` where `pairs` is the array of pairs.
"""
struct PairCandidates{E<:AbstractEvolution} <: AbstractEvolution
    evo::E
end
function _evolve(e::PairCandidates, pop)
    padpop = iseven(length(pop)) ? pop : vcat(pop, pop[1])
    pairs = collect(zip(padpop[1:2:end], padpop[2:2:end]))
    return foldl(evolve(e.evo, pairs), init=[]) do popout, (c1,c2)::Tuple
        length(popout) < length(pop) - 1 ? vcat(popout, c1, c2) : vcat(popout, c1)
    end
end

"""
    ShuffleCandidates
    ShuffleCandidates()
    ShuffleCandidates(rng)

Shuffles the population using `rng`.

Useful with [`PairCandidates`](@ref) in case the prior selection does not shuffle the population.
"""
struct ShuffleCandidates{R} <: AbstractEvolution
    rng::R
end
ShuffleCandidates() = ShuffleCandidates(rng_default)
_evolve(e::ShuffleCandidates, pop) = shuffle(e.rng, pop)

"""
    EvolveCandidates <: AbstractEvolution
    EvolveCandidates(fun)

Applies `fun(c)` for each candidate `c` in a given population.

Useful with [`evolvemodel`](@ref).
"""
struct EvolveCandidates{F} <: AbstractEvolution
    fun::F
end
_evolve(e::EvolveCandidates, pop) = map(e.fun, pop)
