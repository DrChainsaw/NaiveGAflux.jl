"""
    OptimizerCrossover{C} <: AbstractCrossover{Optimisers.AbstractRule}
    OptimizerCrossover()
    OptimizerCrossover(crossover)

Apply crossover between optimizers.

Type of crossover is determined by `crossover` (default `optimizerswap`) which when given a a tuple of two optimizers will return the result of the crossover operation as a tuple of optimizers.

Designed to be composable with most utility `AbstractMutation`s as well as with itself. For instance, the following seemingly odd construct will swap components of a `Optimisers.OptimiserChain` with a probability of `0.2` per component:

`OptimizerCrossover(MutationProbability(OptimizerCrossover(), 0.2))`

Compare with the following which either swaps all components or none:

`MutationProbability(OptimizerCrossover(), 0.2)`
"""
struct OptimizerCrossover{C} <: AbstractCrossover{Optimisers.AbstractRule}
    crossover::C
end
OptimizerCrossover() = OptimizerCrossover(optimizerswap)

"""
    LearningRateCrossover()

Return an `OptimizerCrossover` which will swap learning rates between optimizers but not change anything else.

Does not do anything if any of the optimizers don't have a learning rate (e.g. WeightDecay).
"""
LearningRateCrossover() = OptimizerCrossover(learningrateswap)

(oc::OptimizerCrossover)(os) = oc.crossover(os)
(oc::OptimizerCrossover)(os::EitherIs{ShieldedOpt}) = os
(oc::OptimizerCrossover)(os::MixTuple{ShieldedOpt, Optimisers.OptimiserChain}) = os
(oc::OptimizerCrossover)(os::EitherIs{Optimisers.OptimiserChain}) = zipcrossover(reoptiter, os, oc.crossover)
(oc::OptimizerCrossover)((o1, o2)::NTuple{2, ImplicitOpt}) = ImplicitOpt.(oc((o1.rule, o2.rule)))


reoptiter(o) = (o,), identity
reoptiter(o::Optimisers.OptimiserChain) = o.opts, Optimisers.OptimiserChain

optimizerswap((o1, o2)) = o2,o1
optimizerswap(os::EitherIs{ShieldedOpt}) = os

learningrateswap((o1,o2)::Tuple) = (setlearningrate(o1, learningrate(o2)) , setlearningrate(o2, learningrate(o1)))
learningrateswap(os::EitherIs{ShieldedOpt}) = os
learningrateswap(os::EitherIs{WeightDecay}) = os
