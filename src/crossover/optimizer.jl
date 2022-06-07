"""
    OptimizerCrossover{C} <: AbstractCrossover{FluxOptimizer}
    OptimizerCrossover()
    OptimizerCrossover(crossover)

Apply crossover between optimizers.

Type of crossover is determined by `crossover` (default `optimizerswap`) which when given a a tuple of two optimizers will return the result of the crossover operation as a tuple of optimizers.

Designed to be composable with most utility `AbstractMutation`s as well as with itself. For instance, the following seemingly odd construct will swap components of a `Flux.Optimiser` with a probability of `0.2` per component:

`OptimizerCrossover(MutationProbability(OptimizerCrossover(), 0.2))`

Compare with the following which either swaps all components or none:

`MutationProbability(OptimizerCrossover(), 0.2)`
"""
struct OptimizerCrossover{C} <: AbstractCrossover{FluxOptimizer}
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
(oc::OptimizerCrossover)(os::MixTuple{ShieldedOpt, Flux.Optimiser}) = os
(oc::OptimizerCrossover)(os::EitherIs{Flux.Optimiser}) = zipcrossover(reoptiter, os, oc.crossover)


reoptiter(o) = (o,), identity
reoptiter(o::Flux.Optimiser) = Tuple(o.os), Flux.Optimiser

optimizerswap((o1, o2)) = o2,o1
optimizerswap(os::EitherIs{ShieldedOpt}) = os

learningrateswap((o1,o2)::Tuple) = (@set o1.eta = learningrate(o2)) , (@set o2.eta = learningrate(o1))
learningrateswap(os::EitherIs{ShieldedOpt}) = os
learningrateswap(os::EitherIs{WeightDecay}) = os
