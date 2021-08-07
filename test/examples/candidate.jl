md"""
# Candidate Utilities

As seen in [Fitness Functions](@ref), fitness strategies require an `AbstractCandidate` to compute fitness. To be used by NaiveGAflux,
an `AbstractCandidate` needs to 
1. Provide the data needed by the fitness strategy, most commonly the model but also things like lossfunctions and optimizers
2. Be able to create a new version of itself given a function which maps its fields to new fields.

Capability 1. is generally performed through functions of the format `someproperty(candidate; default)` where in general 
`someproperty(::AbstractCandidate; default=nothing) = default`. The following such functions are currently implemented by NaiveGAflux:

* `model(c; default)`  : Return a model
* `opt(c; default)`    : Return an optimizer
* `lossfun(c; default)` : Return a lossfunction

All such functions are obviously not used by all fitness strategies and some are used more often than others. Whether an 
`AbstractCandidate` returns something other than `default` generally depends on whether it is a hyperparameter which is 
being searched for or not. For example, the very simple `CandidateModel` has only a `model` while `CandidateOptModel` 
has both a model and an own optimizer which may be mutated/crossedover when evolving.

Capability 2. is what is used then evolving a candidate into a new version of itself. The function to implement for new
`AbstractCandidate` types is `newcand(c::MyCandidate, mapfields)` which in most cases has the implementation 
`newcand(c::MyCandidate, mapfield) = MyCandidate(map(mapfield, getproperty.(c, fieldnames(MyCandidate)))...)`.
Furthermore, candidates must also be functors from [Functors.jl](https://github.com/FluxML/Functors.jl) to
support things like GPU<->CPU movement.

Example with a new candidate type and a new fitness strategy for said type:
"""

@testset "Candidate handling" begin #src
import Functors
struct ExampleCandidate <: AbstractCandidate
    a::Int
    b::Int
end
aval(c::ExampleCandidate; default=nothing) = c.a
bval(c::ExampleCandidate; default=nothing) = c.b

Functors.@functor ExampleCandidate

struct ExampleFitness <: AbstractFitness end
NaiveGAflux._fitness(::ExampleFitness, c::AbstractCandidate) = aval(c; default=10) - bval(c; default=5)

# Ok, this is alot of work for quite little in this dummy example.
@test fitness(ExampleFitness(), ExampleCandidate(4, 3)) === 1

ctime, examplemetric = fitness(TimeFitness(ExampleFitness()), ExampleCandidate(3,1))
@test examplemetric === 2
@test ctime > 0
end #src
