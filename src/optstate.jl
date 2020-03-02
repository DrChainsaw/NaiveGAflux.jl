# TODO: Move to NaiveNASflux one mature enough

struct Stateful end
struct Stateless end

opttype(x) = @error "No optimizer type defined for $x"

macro stateful(OT)
    statetype(OT, Stateful)
end
macro stateless(OT)
    statetype(OT, Stateless)
end

statetype(OT, T) =  :(statetype(@__MODULE__, $(esc(OT)), $(esc(T))))
statetype(m::Module, OT, T) = @eval m opttype(::$OT) = $(T())

for opt in (Momentum, Nesterov, ADAM, RMSProp, RADAM, AdaMax, ADAGrad, ADADelta, AMSGrad, NADAM, InvDecay, ExpDecay)
    @stateful opt
end

for opt in (Descent, WeightDecay)
    @stateless opt
end


optstate(opt::Flux.Optimise.Optimiser) = mapreduce(o -> optstate(o), vcat, opt.os)
optstate(opt) = optstate(opttype(opt), opt)
optstate(::Stateful, opt::T) where T = mapreduce(fn -> getfield(opt, fn), addstate, fieldnames(T), init=[])
optstate(::Stateless, opt) = []

addstate(a, x) = a
addstate(a, d::AbstractDict) = vcat(a, d)


"""
    StateAlign{T<:AbstractDict, F} <: AbstractMutableComp
    StateAlign{T}(state::T, m::F)

Aligns state with parameters so that when parameters of wrapped `AbstractMutableComp` are mutated, state is updated accordingly.

Indended use is optimizers with state (e.g. Nesterov and ADAM).
"""
struct StateAlign{T<:AbstractDict} <: AbstractMutableComp
    state::T
    m::AbstractMutableComp
end
StateAlign(state::T) where T<:AbstractDict = m -> StateAlign{T}(state, m)

NaiveNASflux.@mutable_functor StateAlign

withopt(opt::Flux.Optimise.Optimiser) = mapreduce(withopt, ∘, opt.os)
withopt(opt) = withopt(opttype(opt), opt)
withopt(::Stateless, opt) = identity
withopt(::Stateful, opt) = StateAlign(first(optstate(opt)))

(m::StateAlign)(x...) = m.m(x...)
NaiveNASflux.wrapped(m::StateAlign) = m.m
NaiveNASflux.layer(m::StateAlign) = layer(NaiveNASflux.wrapped(m))

NaiveNASlib.mutate_inputs(m::StateAlign, inputs::AbstractArray{<:Integer,1}...) = NaiveNASflux.mutate(m; inputs=inputs[1], outputs=1:nout(m))
NaiveNASlib.mutate_outputs(m::StateAlign, outputs) = NaiveNASflux.mutate(m; inputs=Base.OneTo.(nin(m)), outputs=outputs)

function NaiveNASflux.mutate(m::StateAlign; inputs, outputs, otherkwargs...)
    #ops = Original parameters
    #nps = New parameters
    ops = params(m)
    NaiveNASflux.mutate(m.m; inputs=inputs, outputs=outputs, otherkwargs...)
    nps = params(m)

    for (op,np) in zip(ops, nps)
        op in keys(m.state) || continue

        os = m.state[op]
        ns = select_state(os, op, layer(m), inputs, outputs)

        m.state[np] = ns
        delete!(m.state, op)
    end
end

select_state(s, op, l, ins, outs) = s
select_state(s::Tuple, op, l, ins, outs) = map(si -> select_state(si, op, l, ins, outs), s)
select_state(s::AbstractVector{<:Number}, op::AbstractVector{<:Number}, l, ins, outs) = NaiveNASflux.select(s, 1 => outs)
select_state(s::AbstractArray{<:Number, N}, op::AbstractArray{<:Number, N}, l, ins, outs) where N = NaiveNASflux.select(s, outdim(l) => outs, indim(l) => ins)
