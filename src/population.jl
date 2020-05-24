
"""
    AbstractPopulation

Abstract base type for population.

Keeps track of which generation it is and allows for iterating over its members.
"""
abstract type AbstractPopulation end

generation(p::AbstractPopulation) = generation(b.base)

Base.iterate(p::AbstractPopulation, s...) = Base.iterate(wrappedpop(p), s...)

Base.length(p::AbstractPopulation) = length(wrappedpop(p))
Base.eltype(p::AbstractPopulation) = eltype(wrappedpop(p))
Base.size(p::AbstractPopulation) = size(wrappedpop(p))
Base.IteratorSize(p::AbstractPopulation) = Base.IteratorSize(wrappedpop(p))
Base.IteratorEltype(p::AbstractPopulation) = Base.IteratorEltype(wrappedpop(p))


"""
    struct Population{N, P}
    Population(gen, members)
    Population(members)

Basic population type which just adds generation counting to its members.

Evolving the population returns a new `Population` with the generation counter incremented.
"""
struct Population{N, P} <: AbstractPopulation
    gen::N
    members::P
end
Population(members) = Population(1, members)
wrappedpop(p::Population) = p.members

generation(p::Population) = p.gen

evolve!(e::AbstractEvolution, p::Population) = Population(p.gen + 1, evolve!(e, wrappedpop(p)))
