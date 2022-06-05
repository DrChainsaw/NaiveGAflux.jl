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

EitherIs{T} = Union{Tuple{T, Any}, Tuple{Any,T}}

(oc::OptimizerCrossover)(os) = oc.crossover(os)
(oc::OptimizerCrossover)(os::EitherIs{ShieldedOpt}) = os
function (oc::OptimizerCrossover)((o1,o2)::EitherIs{Flux.Optimiser})
    os1,o1re = optiter(o1)
    os2,o2re = optiter(o2)
    res = oc.crossover.(zip(os1,os2))
    os1n = (t[1] for t in res)
    os2n = (t[2] for t in res)
    return o1re(os1n..., os1[length(os2)+1:end]...), o2re(os2n..., os2[length(os1)+1:end]...)
end

optiter(o) = (o,), (os...) -> os[1]
optiter(o::Flux.Optimiser) = Tuple(o.os), (os...) -> Flux.Optimiser(os...)

optimizerswap((o1, o2)::Tuple) = o2,o1

learningrateswap((o1,o2)::Tuple) = (@set o1.eta = learningrate(o2)) , (@set o2.eta = learningrate(o1))
learningrateswap(os::EitherIs{ShieldedOpt}) = os
learningrateswap(os::EitherIs{WeightDecay}) = os
