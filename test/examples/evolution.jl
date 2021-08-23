md"""
# Evolution Strategies

Evolution strategies are the functions used to evolve the population in the genetic algorithm from one generation to the next. The following is performed by evolution strategies:

* Select which candidates to use for the next generation
* Produce new candidates, e.g by mutating the selected candidates

Important to note about evolution strategies is that they generally expect candidates which can provide a precomputed fitness value,
e.g. [`FittedCandidate`](@ref)s. This is because the fitness value is used by things like sorting where it is not only impractical
to recompute it, but is also might lead to undefined behaviour if it is not always the same. Use [`Population`](@ref) to get some 
help with computing fitness for all candidates before passing them on to evolution.

Note that there is no general requirement on an evolution strategy to return the same population size as it was given. It is also 
free to create completely new candidates without basing anything on any given candidate.
"""

@testset "Evolution strategies" begin #src
# For controlled randomness in the examples.
struct FakeRng end
Base.rand(::FakeRng) = 0.7

# Dummy candidate for brevity.
struct Cand <: AbstractCandidate
    fitness
end
NaiveGAflux.fitness(d::Cand) = d.fitness

# [`EliteSelection`](@ref) selects the n best candidates.
elitesel = EliteSelection(2)
@test evolve(elitesel, Cand.(1:10)) == Cand.([10, 9])

# [`EvolveCandidates`](@ref) maps candidates to new candidates (e.g. through mutation).
evocands = EvolveCandidates(c -> Cand(fitness(c) + 0.1))
@test evolve(evocands, Cand.(1:10)) == Cand.(1.1:10.1)

# [`SusSelection`](@ref) selects candidates randomly using stochastic uniform sampling.
# Selected candidates will be forwarded to the wrapped evolution strategy before returned.
sussel = SusSelection(5, evocands, FakeRng())
@test evolve(sussel, Cand.(1:10)) == Cand.([4.1, 6.1, 8.1, 9.1, 10.1])

# [`CombinedEvolution`](@ref) combines the populations from several evolution strategies.
comb = CombinedEvolution(elitesel, sussel)
@test evolve(comb, Cand.(1:10)) == Cand.(Any[10, 9, 4.1, 6.1, 8.1, 9.1, 10.1])
end #src
