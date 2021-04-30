
"""
    AbstractPopulation

Abstract base type for population.

Keeps track of which generation it is and allows for iterating over its members.
"""
abstract type AbstractPopulation end

generation(p::AbstractPopulation) = generation(p.base)

Base.iterate(p::AbstractPopulation, s...) = Base.iterate(wrappedpop(p), s...)

Base.length(p::AbstractPopulation) = length(wrappedpop(p))
Base.eltype(p::AbstractPopulation) = eltype(wrappedpop(p))
Base.size(p::AbstractPopulation) = size(wrappedpop(p))
Base.IteratorSize(p::AbstractPopulation) = Base.IteratorSize(wrappedpop(p))
Base.IteratorEltype(p::AbstractPopulation) = Base.IteratorEltype(wrappedpop(p))

generation_filename(dir) = joinpath(dir, "generation.txt")

"""
    struct Population{N, P}
    Population(gen, members)
    Population(members)

Basic population type which just adds generation counting to its members.

Evolving the population returns a new `Population` with the generation counter incremented.
"""
struct Population{N<:Integer, P} <: AbstractPopulation
    gen::N
    members::P
end
Population(members) = Population(1, members)
Population(members::PersistentArray) = Population(members.savedir, members)
function Population(savedir::AbstractString, members)
    gfile = generation_filename(savedir)
    gen = !isfile(gfile) ? 1 : parse(Int, readline(gfile))
    return Population(gen, members)
end

wrappedpop(p::Population) = p.members

generation(p::Population) = p.gen

fitness(p::Population, f::AbstractFitness) = Population(p.gen, map(wrappedpop(p)) do cand
    FittedCandidate(cand, f, p.gen)
end)
evolve(e::AbstractEvolution, p::Population) = Population(p.gen + 1, evolve(e, wrappedpop(p)))
evolve(e::AbstractEvolution, f::AbstractFitness, p::Population) = evolve(e, fitness(p, f))


function persist(p::Population{N, <:PersistentArray}) where N
    persist(wrappedpop(p))
    open(io -> write(io, string(p.gen)), generation_filename(wrappedpop(p).savedir); write=true)
end
