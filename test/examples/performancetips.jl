md"""
# Performance Tips

This section contains a couple of things one can do to speed up the program. These things are either experimental or unorthodox which is why they are not enabled by default.

## Faster Training with AutoOptimiser and ImplicitOpt

By default the models will use the normal `Flux` training flow of 
1) Compute the initial optimiser state from the optimiser rule and model
2) Compute the gradient of the entire model 
3) Update the model and optimiser state using the gradient

The main performance problem is that NaiveGAflux can create quite messy nested models with few repeating motifs. This can cause compile times to explode since the model gradient is a nested `NamedTuple` which mirrors the model structure. Since each unique model architecture will have its own unique gradient type the compilation will happen for each evaluated model, even after mutation (since the model architecture might be different then).

To mitigate this issue, the [`AutoOptimiser`](@ref) updates parameters and optimiser state of a single layer (e.g. a `Flux.Dense`) during the backwards pass. This is obviously a rather surprising side effect from what is normally just a pure calculation which is one has to actively opt in for it.

The way it works is simply that it wraps both a specific layer (e.g. a `Flux.Dense`) and the associated optimiser state for the wrapped layer and intercepts the gradient during the pullback.

Use a [`LayerVertexConf`](@ref) to make layers created by NaiveGAflux use an [`AutoOptimiser`](@ref).

!!! warning 
    Be careful to ensure that all `VertexSpace`s (which produce layers with trainable parameters) have a `LayerVertexConf` which adds an `AutoOptimiser` since a mix of auto optimising and non-auto optimising layers is not supported. 

    NaiveGAflux will throw an error if an inconsistent model is found, but it can be difficult to find out the origin of the error since it is not caught right away.

A secondary benefit is that parameter gradients are consumed right after they are computed and can be garbage collected immediately. By default, [`AutoOptimiser`](@ref) will not return the gradient of the wrapped layer just to enable this. This behaviour is configurable by supplying a different `gradfun` when creating the [`AutoOptimiser`](@ref).

When using [`AutoOptimiser`](@ref)s, one must also wrap optimiser rules in an [`ImplicitOpt`](@ref). This tells NaiveGAflux that step 3) above shall be skipped since the model is already updated after gradient calculation. 

!!! info
    `AutoFlux` uses `AutoOptimiser` and `ImplicitOpt` so there is no need to configure them there.

Here is the example from the [Quick Tutorial](@ref) modified to use [`AutoOptimiser`](@ref).
"""

@testset "Quick Tutorial with AutoOptimiser" begin #src
using NaiveGAflux, Random
import Flux, Optimisers
import Flux: relu, elu, selu
import Optimisers: Descent, Momentum, Nesterov, Adam
Random.seed!(NaiveGAflux.rng_default, 0)
rng = Xoshiro(0)

nlabels = 3
ninputs = 5
popsize = 5

# #### Step 1: Create initial models
# Our LayerVertexConf which wraps layers in `AutoOptimiser` (and `ActivationContribution`).
layerconf = LayerVertexConf(layerfun=ActivationContribution ∘ AutoOptimiser)
# Search space: 2-4 dense layers of width 3-10.
layerspace = VertexSpace(layerconf, DenseSpace(3:10, [identity, relu, elu, selu]))
initial_hidden = RepeatArchSpace(layerspace, 1:3)
# Output layer has fixed size and is shielded from mutation. Don't forget to add the `layerconf` here as well!
outlayer = VertexSpace(Shielded(layerconf), DenseSpace(nlabels, identity))
initial_searchspace = ArchSpaceChain(initial_hidden, outlayer)

# Sample 5 models from the initial search space and make an initial population.
samplemodel(invertex) = CompGraph(invertex, initial_searchspace(invertex))
initial_models = [samplemodel(denseinputvertex("input", ninputs)) for _ in 1:popsize]
@test nvertices.(initial_models) == [4, 3, 4, 5, 3]

# Lets add optimisers into the search space this time just to show how `ImplicitOpt` is used then.
optalts = (Descent, Momentum, Nesterov, Adam)
initial_learningrates = 10f0.^rand(rng, -3:-1, popsize)
initial_optrules = ImplicitOpt.(initial_learningrates .|> rand(rng, optalts, popsize))

population = Population(CandidateOptModel.(initial_optrules, initial_models))
@test generation(population) == 1

# #### Step 2: Set up fitness function:
# Basically the same as in the quick tutorial except we wrap the default optimiser in an `ImplicitOpt`. Note that this does not matter here since each candidate now has its own optimiser.
onehot(y) = Flux.onehotbatch(y, 1:nlabels)
batchsize = 4
datasettrain    = [(randn(Float32, ninputs, batchsize), onehot(rand(1:nlabels, batchsize)))]
datasetvalidate = [(randn(Float32, ninputs, batchsize), onehot(rand(1:nlabels, batchsize)))]

fitnessfunction = TrainThenFitness(;
    dataiter = datasettrain,
    defaultloss = Flux.logitcrossentropy, # Will be used if not provided by the candidate
    defaultopt = ImplicitOpt(Adam()), # Same as above. Note that this will not be used
    fitstrat = AccuracyFitness(datasetvalidate) # This is what creates our fitness value after training
)

# #### Step 3: Define how to search for new candidates
# We choose to evolve the existing ones through mutation.

# [`VertexMutation`](@ref) selects valid vertices from the graph to mutate.
# [`MutationProbability`](@ref) applies mutation `m` with a probability of `p`.
# Lets create a short helper for brevity:
mp(m, p) = VertexMutation(MutationProbability(m, p))
# This time around we also add an optimiser mutation in the mix
changesize = mp(NoutMutation(-0.2, 0.2), 0.6)
addlayer = mp(AddVertexMutation(layerspace), 0.4)
remlayer = mp(RemoveVertexMutation(), 0.4)
modelmutation = MutationChain(changesize, remlayer, addlayer)

changelr = MutationProbability(LearningRateMutation(), 0.5)
changeoptrule = MutationProbability(OptimiserMutation((optalts)), 0.1)
optmutation = MutationChain(changelr, changeoptrule)

# These steps are the same as in [Quick Tutorial](@ref) except we add the optmutation to [`MapCandidate`](@ref) 
elites = EliteSelection(2)
mutate = SusSelection(3, EvolveCandidates(MapCandidate(modelmutation, optmutation)))
selection = CombinedEvolution(elites, mutate)

# #### Step 4: Run evolution
newpopulation = evolve(selection, fitnessfunction, population)
@test newpopulation != population
@test generation(newpopulation) == 2 

end #src

md"""
## Garbage Collect Between Batches

One should never need to call garbage collection manually from inside a program. However, there is an [CUDA.jl issue](https://github.com/JuliaGPU/CUDA.jl/issues/1540) which causes the program to halt for very long times when using alot of memory. For one reason or the other this is alleviated by garbage collection.

The following iterator wrapper keeps the issue at bay until a more permanent fix is issued:
"""

@testset "GpuGcIterator" begin #src
using NaiveGAflux
function cudareclaim()
    ## Uncomment the line below!
    ## CUDA.reclaim()
end

struct GpuGcIterator{I}
    base::I
end

function Base.iterate(itr::GpuGcIterator) 
    valstate = iterate(itr.base)
    valstate === nothing && return nothing
    val, state = valstate
    gctask = Threads.@spawn Timer(0.0)
    return val, (2, gctask, state)
end

function Base.iterate(itr::GpuGcIterator, (cnt, gctask, state)) 
    close(fetch(gctask))

    if cnt > 5
        ## This is quite fast and enough to keep the problem away in 99% of cases
        GC.gc(false)
        cnt = 0
    end

    gctask = Threads.@spawn begin
        ## This can often get us unstuck if the above doesn't help in time
        reclaimcnt = Ref(0)
        Timer(0.1; interval=0.1) do t
            GC.gc(false)
            reclaimcnt[] += 1
            if reclaimcnt[] > 5
                GC.gc()
                cudareclaim()
            end
            if reclaimcnt[] > 40
                ## Stop the task eventually, e.g if program crashes
                close(t)
            end
        end
    end
    
    valstate = iterate(itr.base, state)
    if valstate === nothing
        close(fetch(gctask))
        return nothing
    end
    val, state = valstate
    return val, (cnt+1, gctask, state)
end


Base.IteratorSize(::Type{GpuGcIterator{I}}) where I = Base.IteratorSize(I)
Base.IteratorEltype(::Type{GpuGcIterator{I}}) where I = Base.IteratorEltype(I)

Base.length(itr::GpuGcIterator) = length(itr.base)
Base.size(itr::GpuGcIterator) = size(itr.base)
Base.eltype(::Type{GpuGcIterator{I}}) where I = eltype(I)

# This makes it work even when wrapping a [`StatefulGenerationIter`](@ref):
NaiveGAflux.itergeneration(itr::GpuGcIterator, gen) = GpuGcIterator(NaiveGAflux.itergeneration(itr.base, gen))

# Just wrap your iterator in the `GpuGcIterator` (don't forget to wrap the validation iterator)

trainiter = GpuGcIterator(BatchIterator(randn(10, 128), 16; shuffle=true))
valiter = GpuGcIterator(BatchIterator(randn(10, 64), 32))

@test size.(collect(trainiter)) == repeat([(10, 16)], 8)
@test size.(collect(valiter)) == repeat([(10, 32)], 2)

@test size.(collect(NaiveGAflux.itergeneration(trainiter, 1))) == repeat([(10, 16)], 8)

end #src
