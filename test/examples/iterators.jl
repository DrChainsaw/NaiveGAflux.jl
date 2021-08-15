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
labels = collect(1:5)

# Simple and fast batching of in-memory data is done by [`BatchIterator`](@ref)
biter = BatchIterator((data, labels), 2)
@test size.(first(biter)) == ((4, 2), (2,))

# Move data to gpu.
giter = GpuIterator(biter)
@test first(giter) == first(biter) |> gpu


# This is the only iterator which is "special" for this package:
rpiter = RepeatPartitionIterator(biter, 2)
# It produces iterators over a subset of the wrapped iterator (2 batches in this case).
piter = first(rpiter)
@test length(piter) == 2
# This allows for easily training several models on the same subset of the data.
for modeli in 1:3
    for ((feature, label), (expf, expl)) in zip(piter, biter)
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
