

"""
    IteratorMapCrossover{C} <: AbstractCrossover{AbstractIteratorMap}
    IteratorMapCrossover() 
    IteratorMapCrossover(crossover)

Apply crossover between `AbstractIteratorMap`s.

Type of crossover is determined by `crossover` (default `iteratormapswap`) which when given a a tuple of two `AbstractIteratorMap`s will return the result of the crossover operation as a tuple of `AbstractIteratorMap`s.

Designed to be composable with most utility `AbstractMutation`s as well as with itself. For instance, the following seemingly odd construct will swap components of two `IteratorMaps` with a probability of `0.2` per component:

`IteratorMapCrossover(MutationProbability(IteratorMapCrossover(), 0.2))`

Compare with the following which either swaps all components or none:

`MutationProbability(IteratorMapCrossover(), 0.2)`
"""
struct IteratorMapCrossover{C} <: AbstractCrossover{AbstractIteratorMap}
    crossover::C
end
IteratorMapCrossover() = IteratorMapCrossover(iteratormapswap)

(ic::IteratorMapCrossover)(ims) = ic.crossover(ims)
(ic::IteratorMapCrossover)(ims::EitherIs{ShieldedIteratorMap}) = ims
(ic::IteratorMapCrossover)(ims::MixTuple{ShieldedIteratorMap, IteratorMaps}) = ims
(ic::IteratorMapCrossover)(ims::EitherIs{IteratorMaps}) = zipcrossover(reimiter, ims, ic.crossover)

reimiter(im) = (im,), identity
reimiter(im::IteratorMaps) = im.maps, IteratorMaps

iteratormapswap((im1, im2)::Tuple, args...) = im2, im1
iteratormapswap(ims::EitherIs{ShieldedIteratorMap}, args...) = ims