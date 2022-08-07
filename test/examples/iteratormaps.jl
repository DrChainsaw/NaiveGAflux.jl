md"""
# Iterator Maps

Iterator maps is the name chosen (in lack of a better name) for mapping an iterator to a new iterator. The main use 
cases for this are:

1. Limiting the batch size of a candidate to prevent out of memory errors (see [Batch Size Utilities](@ref BatchSizeUtilsAPI)).
2. Enabling search for the best training batch size (using e.g. [`TrainBatchSizeMutation`](@ref) and/or [`IteratorMapCrossover`](@ref)).
3. Enabling search for the best data augmentation setting (not part of this package as of yet).

Iterator maps are inteded to be used with [`CandidateDataIterMap`](@ref) and must extend [`AbstractIteratorMap`](@ref).
See [`API`](@ref IteratorMapInterfaceFunctionsAPI) documentation for functions related to iterator maps.

In an attempt to hit two birds with one stone, here is an example of a custom iterator map which logs the sizes
of what a wrapped iterator returns. This allows us to see the effects of [`BatchSizeIteratorMap`](@ref) without
digging too much into the internals.
"""

@testset "Spy on the size" begin #src

using NaiveGAflux, Flux
import NaiveGAflux: AbstractIteratorMap

struct SizeSpyingIteratorMap <: AbstractIteratorMap end

NaiveGAflux.maptrain(::SizeSpyingIteratorMap, iter) = Iterators.map(iter) do val
    @info "The sizes are $(size.(val))"
    return val
end

# Create the iterator map we want to use. Last argument to [`BatchSizeIteratorMap`](@ref) is 
# normally created through [`batchsizeselection`](@ref), but here we will use a dummy model
# for which the maximum batch size computation is not defined.
iteratormap = IteratorMaps(SizeSpyingIteratorMap(), BatchSizeIteratorMap(8, 16, (bs, _) -> bs)) 

# Create a candidate with the above mentioned dummy model.
cand = CandidateDataIterMap(iteratormap, CandidateModel(sum))

# Data set has `20` examples, and here we provide it "raw" without any batching for brevity. 
# Other arguments are not important for this example.
fitstrat = TrainThenFitness(
                            dataiter = ((randn(32, 32, 3, 20), randn(1, 20)),),
                            defaultloss = (x, y) -> sum(x .+ y),
                            defaultopt = Flux.Optimise.Descent(),
                            fitstrat = SizeFitness()
                            )

# When the model is trained it will wrap the iterator accoring to our `iteratormap`.
@test_logs((:info, "The sizes are ((32, 32, 3, 8), (1, 8))"),
           (:info, "The sizes are ((32, 32, 3, 8), (1, 8))"),
           (:info, "The sizes are ((32, 32, 3, 4), (1, 4))"), 
           fitness(fitstrat, cand))

# Lets mutate the candidate with a new batch size (`SizeSpyingIteratorMap` does not have any properties to mutate).
# Here we set `l1 == l2` to prevent that randomness breaks the testcase, but you might want to use something like 
# `TrainBatchSizeMutation(-0.1, 0.1, ntuple(i -> 2^i))`. The last argument is to make sure we select a power of two
# as the new batch size.
batchsizemutation = TrainBatchSizeMutation(0.1, 0.1, ntuple(i -> 2^i, 10))

# MapCandidate creates new candidates from a set of mutations or crossovers.
newcand = cand |> MapCandidate(batchsizemutation)

@test_logs((:info, "The sizes are ((32, 32, 3, 16), (1, 16))"),
           (:info, "The sizes are ((32, 32, 3, 4), (1, 4))"), 
           fitness(fitstrat, newcand))

end #src