md"""
# Iterators

While not part of the scope of this package, some simple utilities for iterating over data sets is provided.

The only iterator which is in some sense special for this package is `RepeatPartitionIterator` which produces
iterators over a subset of its wrapped iterator. This is useful when one wants to ensure that all models see 
the same (possibly randomly augmented) data in the same order. Note that this is not certain to be the best 
strategy for finding good models for a given data set and this package does (intentionally) blur the lines a 
bit between model training protocol and architecture search.
"""

@testset "Iterators" begin #src
data = reshape(collect(1:4*5), 4,5)

# Batching is done by [`BatchIterator`](@ref)
biter = BatchIterator(data, 2)
@test size(first(biter)) == (4, 2)

# Shuffle data before batching with [`ShuffleIterator`](@ref).
# Warning: Must use different rng instances with the same seed for features and labels!
siter = ShuffleIterator(data, 2, MersenneTwister(123))
@test size(first(siter)) == size(first(biter))
@test first(siter) != first(biter)

# Apply a function to each batch.
miter = MapIterator(x -> 2 .* x, biter)
@test first(miter) == 2 .* first(biter)

# Move data to gpu.
giter = GpuIterator(miter)
@test first(giter) == first(miter) |> gpu

labels = collect(0:5)

# Possible to use `Flux.onehotbatch` for many iterators.
biter_labels = Flux.onehotbatch(BatchIterator(labels, 2), 0:5)
@test first(biter_labels) == Flux.onehotbatch(0:1, 0:5)

# This is the only iterator which is "special" for this package:
rpiter = RepeatPartitionIterator(zip(biter, biter_labels), 2)
# It produces iterators over a subset of the wrapped iterator (2 batches in this case).
piter = first(rpiter)
@test length(piter) == 2
# This allows for easily training several models on the same subset of the data.
expiter = zip(biter, biter_labels)
for modeli in 1:3
    for ((feature, label), (expf, expl)) in zip(piter, expiter)
        @test feature == expf
        @test label == expl
    end
end

# [`StatefulGenerationIter`](@ref) is typically used in conjunction with [`TrainThenFitness`](@ref) to map a generation.
# number to an iterator from a [`RepeatStatefulIterator`](@ref).
sgiter = StatefulGenerationIter(rpiter)
for (generationnr, topiter) in enumerate(rpiter)
    gendata = collect(NaiveGAflux.itergeneration(sgiter, generationnr))
    expdata = collect(topiter)
    @test gendata == expdata
end

# [`TimedIterator`](@ref) is useful for preventing that models which take very long time to train/validate slow down the process.
timediter = TimedIterator(;
                    timelimit=0.1, 
                    patience=4, 
                    timeoutaction = () -> TimedIteratorStop,
                    accumulate_timeouts=false, 
                    base=1:100)

last = 0
for i in timediter
    last = i
    if i > 2
        # Does not matter here if overloaded CI VM takes longer than this to get back to us. #src
        sleep(0.11)
    end
end
@test last === 6 # Sleep after 2, then 4 patience.
end #src
