# NaiveGAflux

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://DrChainsaw.github.io/NaiveGAflux.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://DrChainsaw.github.io/NaiveGAflux.jl/dev)
[![Build status](https://github.com/DrChainsaw/NaiveGAflux.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/DrChainsaw/NaiveGAflux.jl/actions)
[![Codecov](https://codecov.io/gh/DrChainsaw/NaiveGAflux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DrChainsaw/NaiveGAflux.jl)

Neural architecture search for [Flux](https://github.com/FluxML/Flux.jl) models using genetic algorithms.

Tired of tuning hyperparameters? Once you've felt the rush from reasoning about hyper-hyperparameters there is no going back!

## Basic Usage

```julia
]add NaiveGAflux
```

The basic idea is to create not just one model, but a population of several candidate models with different hyperparameters. The whole population is then evolved while the models are being trained.

| MNIST                                 | CIFAR10                                |
|:-------------------------------------:|:--------------------------------------:|
| <img src="gif/MNIST.gif" width="500"> | <img src="gif/CIFAR10.gif" width="500">  |

More concretely, this means train each model for a number of iterations, evaluate the fitness of each model, select the ones with highest fitness, apply random mutations (e.g. add/remove neurons/layers) to some of them and repeat until a model with the desired fitness has been produced.

By controlling the number of training iterations before evolving the population, it is possible tune the compromise between fully training each model at the cost of longer time to evolve versus the risk of discarding a model just because it trains slower than the other members.

Like any self-respecting AutoML-type library, NaiveGAflux provides an application with a deceivingly simple API:

```julia
using NaiveGAflux.AutoFlux, MLDatasets

models = fit(CIFAR10.traindata())
```

However, most non-toy uses cases will probably require a dedicated application. NaiveGAflux provides the components to make building it easy and fun! 
Head over to the [documentation](https://DrChainsaw.github.io/NaiveGAflux.jl/stable) to get started.

## Contributing

All contributions are welcome. Please file an issue before creating a PR.
