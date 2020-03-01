# TODO: Move to NaiveNASflux one mature enough

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
StateAlign(state::T) where T = m -> StateAlign{T}(state, m)
(m::StateAlign)(x...) = m.m(x...)
wrapped(m::StateAlign) = m.m
layer(m::StateAlign) = layer(wrapped(m))

NaiveNASlib.mutate_inputs(m::StateAlign, inputs::AbstractArray{<:Integer,1}...) = NaiveNASflux.mutate(m; inputs=inputs[1], outputs=1:nout(m))
NaiveNASlib.mutate_outputs(m::StateAlign, outputs) = NaiveNASflux.mutate(m; inputs=Base.OneTo.(nin(m)), outputs=outputs)

function NaiveNASflux.mutate(m::StateAlign; inputs, outputs)
    ops = params(m)
    NaiveNASflux.mutate(m.m, inputs=inputs, outputs=outputs)
    nps = params(m)

    for (op,np) in zip(ops, nps)
        os in keys(m.state) || continue

        os = m.state[op]
        ns = select_state(osi, op, layer(m), inputs, outputs, os)

        m.state[np] = ns
        delete!(m.state, os)
    end
end

select_state(s, op, l, ins, outs) = s
select_state(s::Tuple, op, l, ins, outs) = map(si -> select_state(si, op, l, ins, outs))
select_state(s::AbstractVector{<:Number}, op::AbstractVector{<:Number}, l, ins, outs) = NaiveNASflux.select(s, 1 => outs)
select_state(s::AbstractArray{<:Number, N}, op::AbstractArray{<:Number, N}, l, ins, outs) where N = NaiveNASflux.select(s, outdim(l) => outs, indim(l) => ins)
