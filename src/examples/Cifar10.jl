module Cifar10

using ..NaiveGAflux
using AutoFit

export run_experiment

defaultdir(this="CIFAR10") = joinpath(NaiveGAflux.modeldir, this)


end  # module cifar10
