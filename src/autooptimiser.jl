module AutoOptimiserExperimental

# This file should either be be moved to NaiveNASflux once it is mature enough or deleted when obsolete

# Criteria for maturity: 
# 1. Implementation of size mutation so that iterator state can persist after a model is mutated
# 2. Implementation of optimizer per vertex as a search space parameter in NaiveGAflux

# Criteria for obsoletion
# 1. Gradients of deeply nested models does not cause explosive compile times when updating parameters through Optimisers.update!
# 2. Mutation of optimiser state can be done on the optimiser state itself

import NaiveNASflux, Flux, Optimisers
import NaiveNASflux.ChainRulesCore
import NaiveNASflux.ChainRulesCore: HasReverseMode, RuleConfig, rrule_via_ad, Tangent, NoTangent
import NaiveNASflux: AbstractMutableComp, wrapped, LazyMutable, NoParams, ResetLazyMutable, MutationTriggered
using NaiveNASlib, Functors
using NaiveNASlib.Extend

"""
    AutoOptimiser{L}
    AutoOptimiser(gradfun, layer, opt)
    AutoOptimiser(gradfun, opt)
    AutoOptimiser(opt)

Updates parameters for `layer` whenever gradients are computed during the backwards pass.

This bending of the (r)rules serves the following purposes:
1) It allows for gradients to be gc:ed/freed eagerly, reducing the memory pressure.
2) It prevents explosive compile times when updating parameters for large and nested models.
3) It allows optimiser state to be mutated and transformed (e.g. `gpu`/`cpu`) together with the model (not implemented yet).

Designed to be used as `layerfun` argument to [`fluxvertex`](@ref). In particular, `AutoOptimiser(o)` and
`AutoOptimiser(gradfun, o)` where `o` is an `Optimiser.AbstractRule` returns a function which expects the
`layer` to enable constructs like `layerfun = $LazyMutable ∘ $AutoOptimiser(Descent())`.
    
Parameters are updated through `gradfun(optstate, layer, grad)` which is expected to return a tuple
with the new optimiser state and the gradient for layer (typically either `grad` or `NoTangent()`).
Note that this is different compared to what `Optimiser.update!` returns, meaning that `Optimisers.update!`
by itself can't be given as `gradfun` (although `gradfun` typically calls `Optimiser.update!`).

"""
mutable struct AutoOptimiser{F,L} <: AbstractMutableComp
    gradfun::F
    layer::L
    optstate # We want to be able to change type of this for now
end
const AnyOptimiser = Union{Flux.Optimise.AbstractOptimiser, Optimisers.AbstractRule}
AutoOptimiser(o::AnyOptimiser) = AutoOptimiser(MaskGradient(), o)
AutoOptimiser(gradfun, o::AnyOptimiser) = l -> AutoOptimiser(gradfun, l, Optimisers.setup(o, l))
AutoOptimiser(l) = AutoOptimiser(MaskGradient(), l, nothing)

AutoOptimiser(m::NoParams) = m
AutoOptimiser(gradfun, m::NoParams) = m
AutoOptimiser(gradfun, m::NoParams, ::AnyOptimiser) = m

@functor AutoOptimiser

wrapped(a::AutoOptimiser) = a.layer

# We should probably return empty/nothing here, but for now lets keep this so Flux.params works
Flux.trainable(a::AutoOptimiser) = (;layer=Flux.trainable(a.layer)) 

(a::AutoOptimiser)(args...) = wrapped(a)(args...)

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, a::AutoOptimiser, args...)
    res, back = rrule_via_ad(config, a.layer, args...)
    function AutoOptimiser_back(Δ)
        δs = back(Δ)
        a.optstate, layergrad = a.gradfun(a.optstate, a.layer, δs[1])
        _makegrad(AutoOptimiser, layergrad), δs[2:end]...
    end
    return res, AutoOptimiser_back
end
_makegrad(::Type{AutoOptimiser}, grad) = Tangent{AutoOptimiser}(;layer=grad)  
_makegrad(::Type{AutoOptimiser}, ::Union{Nothing, NoTangent}) = NoTangent()

optimisersetup!(rule::Optimisers.AbstractRule, x) = mutateoptimiser!(rule, x)
function mutateoptimiser!(f, g::CompGraph) 
    ok = recursive_optimisersetup!(f, g.outputs)
    if !ok
        @warn "No implict optimiser found! Call is a noop"
    end
    ok
end
mutateoptimiser!(f, v::AbstractVertex) = mutateoptimiser!(f, base(v))
mutateoptimiser!(f, ::InputVertex) = false
mutateoptimiser!(f, v::CompVertex) = mutateoptimiser!(f, v.computation)
mutateoptimiser!(f, m::AbstractMutableComp) = mutateoptimiser!(f, wrapped(m))
mutateoptimiser!(f, r::ResetLazyMutable) = mutateoptimiser!(f, r.wrapped)
mutateoptimiser!(f, m::MutationTriggered) = mutateoptimiser!(f, m.wrapped)

function mutateoptimiser!(f::Optimisers.AbstractRule , a::AutoOptimiser)
    a.optstate = Optimisers.setup(f, a.layer)
    mutateoptimiser!(f, wrapped(a)) || true
end
function mutateoptimiser!(f , a::AutoOptimiser)
    a.optstate = f(a)
    mutateoptimiser!(f, wrapped(a)) || true
end
mutateoptimiser!(f, x) = false

recursive_optimisersetup!(f, vs::Union{Tuple, AbstractArray}) = mapreduce(v -> recursive_optimisersetup!(f, v), |, vs; init=false)
function recursive_optimisersetup!(f, v::AbstractVertex)
    ok = mutateoptimiser!(f, v)
    recursive_optimisersetup!(f, inputs(v)) || ok
end

_updateparams_safe!(::Nothing, args...) = throw(ArgumentError("AutoOptimiser without optimiser state invoked. Forgot to call optimisersetup!?"))
function _updateparams_safe!(optstate, model, grads)
    deepany(grads) do g
        g isa Number || return false
        isnan(g) || isinf(g)      
    end && return optstate
    first(Optimisers.update!(optstate, model, grads)), grads
end

deepany(f, x::Union{Tuple, NamedTuple}) = any(e -> deepany(f, e), x)
deepany(f, x::Nothing) = f(x)
deepany(f, x) = any(f, x)

struct MaskGradient{F}
    gradfun::F
end
MaskGradient() = MaskGradient(_updateparams_safe!)
(m::MaskGradient)(args...) = first(m.gradfun(args...)), NoTangent()

end