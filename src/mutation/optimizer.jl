"""
    struct OptimizerMutation{F} <: AbstractMutation{Optimisers.AbstractRule}
    OptimizerMutation(optfun)
    OptimizerMutation(os::Union{Tuple, <:AbstractArray})

Mutatates optimizers not wrapped in `ShieldedOpt` through `optfun`.

Invoked recursively for `Optimisers.OptimiserChain`s.
"""
struct OptimizerMutation{F} <: AbstractMutation{Optimisers.AbstractRule}
    optfun::F
end
OptimizerMutation(os::Union{Tuple, <:AbstractArray}, rng=rng_default) = OptimizerMutation(o -> rand(rng, os)(learningrate(o)))

"""
    LearningRateMutation(rng=rng_default)

Return an `OptimizerMutation` which mutates the learning rate of optimizers.
"""
LearningRateMutation(rng=rng_default) = OptimizerMutation(o -> nudgelr(o, rng))

(m::OptimizerMutation)(opt::Optimisers.OptimiserChain) = Optimisers.OptimiserChain(m.(opt.opts))
(m::OptimizerMutation)(o::ShieldedOpt) = o;
(m::OptimizerMutation)(o::Optimisers.AbstractRule) = m.optfun(o)


nudgelr(o, rng=rng_default) = setlearningrate(o, nudgelr(learningrate(o), rng))
nudgelr(lr::Number, rng=rng_default) = clamp(lr + (rand(rng) - 0.5) * lr * 0.3, 1e-6, 1.0)

learningrate(o::Optimisers.OptimiserChain) = prod(learningrate.(o.opts))
learningrate(o::ShieldedOpt) = learningrate(o.opt)
learningrate(o) = o.eta

newlr(o, lrf = nudgelr) = setlearningrate(o, lrf(learningrate(o)))
setlearningrate(o, lr) = @set o.eta = lr

"""
    AddOptimizerMutation{F} <: AbstractMutation{Optimisers.AbstractRule}

Adds optimizer generated by `optgen(os)` to the set of optimizers where `os` is the existing set.

An attempt to merge optimizers of the same type is made using `mergeopts`.
"""
struct AddOptimizerMutation{F} <: AbstractMutation{Optimisers.AbstractRule}
    optgen::F
end
(m::AddOptimizerMutation)(o::ShieldedOpt) = o;
(m::AddOptimizerMutation)(o::Optimisers.AbstractRule) = m(Optimisers.OptimiserChain(o))
function (m::AddOptimizerMutation)(opt::Optimisers.OptimiserChain)
    newopt = m.optgen(opt)
    return Optimisers.OptimiserChain(mergeopts(typeof(newopt), newopt, opt.opts...))
end
