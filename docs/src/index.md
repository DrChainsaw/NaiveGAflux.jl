# Introduction

NaiveGAflux is a toolbox for doing neural architecture search for Flux models using genetic algorithms. 
It is primarily designed for searching by making modifications to well performing models. This is typically 
done in a train-validate-select-evolve loop where the validation metric serves as the fitness in selection.

There is however absolutely no enforcement of this structure and the parts are designed to work standalone
and in a composable manner to support a wide variety of search strategies.

It is also not limited to model architecture related hyperparameters. Support for inclusion of optimizers,
learningrates and batchsizes into the search space is built in and the framework supports adding any 
hyperparameter (e.g data augmentation strategies and loss functions) through simple interfaces.

## Readers Guideline

The [Quick Tutorial](@ref) serves as a starting point to get an idea of the syntax and type of capabilities of NaiveGAflux.

NaiveGAflux comes bundled with a neural architecture search application called [AutoFlux](@ref) which glues most of NaiveGAfluxs components together. 
This is a good next read if you are interested in quickly getting started with model search using a standard train-validate-select-evolve loop. 
It is however a bit disconnected from NaiveGAflux and many readers will find it better to just continue to the next sections.

There is a short set of examples for each main component types. Each component is designed to work well in isolation and examples are largely 
self-contained, allowing you to pick and choose the ones you like when building an own application.

1. [Search Spaces](@ref)
2. [Mutation Operations](@ref)
3. [Crossover Operations](@ref)
4. [Fitness Functions](@ref)
5. [Candidate Utilities](@ref)
6. [Evolution Strategies](@ref)
7. [Iterator Maps](@ref) 
8. [Iterators](@ref)





