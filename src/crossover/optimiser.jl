"""
    OptimiserCrossover{C} <: AbstractCrossover{Optimisers.AbstractRule}
    OptimiserCrossover()
    OptimiserCrossover(crossover)

Apply crossover between optimisers.

Type of crossover is determined by `crossover` (default `optimiserswap`) which when given a a tuple of two optimisers will return the result of the crossover operation as a tuple of optimisers.

Designed to be composable with most utility `AbstractMutation`s as well as with itself. For instance, the following seemingly odd construct will swap components of a `Optimisers.OptimiserChain` with a probability of `0.2` per component:

`OptimiserCrossover(MutationProbability(OptimiserCrossover(), 0.2))`

Compare with the following which either swaps all components or none:

`MutationProbability(OptimiserCrossover(), 0.2)`
"""
struct OptimiserCrossover{C} <: AbstractCrossover{Optimisers.AbstractRule}
    crossover::C
end
OptimiserCrossover() = OptimiserCrossover(optimiserswap)

"""
    LearningRateCrossover()

Return an `OptimiserCrossover` which will swap learning rates between optimisers but not change anything else.

Does not do anything if any of the optimisers don't have a learning rate (e.g. WeightDecay).
"""
LearningRateCrossover() = OptimiserCrossover(learningrateswap)

(oc::OptimiserCrossover)(os) = oc.crossover(os)
(oc::OptimiserCrossover)(os::EitherIs{ShieldedOpt}) = os
(oc::OptimiserCrossover)(os::MixTuple{ShieldedOpt, Optimisers.OptimiserChain}) = os
(oc::OptimiserCrossover)(os::EitherIs{Optimisers.OptimiserChain}) = zipcrossover(reoptiter, os, oc.crossover)
(oc::OptimiserCrossover)((o1, o2)::NTuple{2, ImplicitOpt}) = ImplicitOpt.(oc((o1.rule, o2.rule)))


reoptiter(o) = (o,), identity
reoptiter(o::Optimisers.OptimiserChain) = o.opts, Optimisers.OptimiserChain

optimiserswap((o1, o2)) = o2,o1
optimiserswap(os::EitherIs{ShieldedOpt}) = os

learningrateswap((o1,o2)::Tuple) = (setlearningrate(o1, learningrate(o2)) , setlearningrate(o2, learningrate(o1)))
learningrateswap(os::EitherIs{ShieldedOpt}) = os
learningrateswap(os::EitherIs{WeightDecay}) = os
