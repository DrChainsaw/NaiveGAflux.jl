# TODO: This file shall go to NaiveNASlib once things work somewhat well

# Methods to help select or add a number of outputs given a new size as this problem apparently belongs to the class of FU-complete problems. And yes, I curse the day I conceived the idea for this project right now...

"""
    AbstractSelectionStrategy

Base type for how to select the exact inputs/outputs indices from a vertex given a size change.
"""
abstract type AbstractSelectionStrategy end

"""
    LogSelection <: AbstractSelectionStrategy
    LogSelection(msgfun::Function, andthen)
    LogSelection(level, msgfun::Function, andthen)

Logs output from function `msgfun` at LogLevel `level` and executes `AbstractSelectionStrategy andthen`.
"""
struct LogSelection <: AbstractSelectionStrategy
    level::Logging.LogLevel
    msgfun
    andthen::AbstractSelectionStrategy
end
LogSelection(msgfun::Function, andthen) = LogSelection(Logging.Info, msgfun, andthen)
LogSelectionFallback(nextstr, andthen; level=Logging.Warn) = LogSelection(level, v -> "Selection for vertex $(name(v)) failed! $nextstr", andthen)

"""
    SelectionFail <: AbstractSelectionStrategy

Throws an error.
"""
struct SelectionFail <: AbstractSelectionStrategy end

"""
    NoutRevert <: AbstractSelectionStrategy
    NoutRevert()

Reverts output size change for a vertex.
"""
struct NoutRevert <: AbstractSelectionStrategy end

"""
    AbstractJuMPSelectionStrategy

Base type for how to select the exact inputs/outputs indices from a vertex given a size change using JuMP to handle the constraints.
"""
abstract type AbstractJuMPSelectionStrategy <: AbstractSelectionStrategy end

fallback(::AbstractJuMPSelectionStrategy) = SelectionFail()

"""
    NoutExact <: AbstractSelectionStrategy
    NoutExact()
    NoutExact(fallbackstrategy)

Selects output indices from a vertex with the constraint that `nout(v)` for all `v` which needs to change as a result of the selection are unchanged.

Possible to set a fallbackstrategy should it be impossible to select indices according to the strategy.
"""
struct NoutExact <: AbstractJuMPSelectionStrategy
    fallback::AbstractSelectionStrategy
end
NoutExact() = NoutExact(LogSelectionFallback("Relaxing size constraint...", NoutRelaxSize()))
fallback(s::NoutExact) = s.fallback

"""
    NoutRelaxSize <: AbstractSelectionStrategy
    NoutRelaxSize()
    NoutRelaxSize(lower, upper)
    NoutRelaxSize(fallbackstrategy)
    NoutRelaxSize(lower, upper, fallbackstrategy)

Selects output indices from a vertex with the constraint that `nout(v)` for all vertices `v` which needs to change as a result of the selection is in the range `max(1, lower * noutv) <= nout(v) <= upper * noutv` where `noutv` is the result of `nout(v)` before selecting indices.

Possible to set a fallbackstrategy should it be impossible to select indices according to the strategy.
"""
struct NoutRelaxSize <: AbstractJuMPSelectionStrategy
    lower::Real
    upper::Real
    fallback::AbstractSelectionStrategy
end
NoutRelaxSize(fallback=LogSelectionFallback("Reverting...", NoutRevert())) = NoutRelaxSize(0.7, 1, NoutRelaxSize(0.5, 1, NoutRelaxSize(0.3, 1.5, NoutRelaxSize(0.2, 2, fallback))))
NoutRelaxSize(lower::Real, upper::Real) = NoutRelaxSize(lower, upper, NoutRevert())
fallback(s::NoutRelaxSize) = s.fallback


"""
    validouts(v::AbstractVertex)

Return a `Dict` mapping vertices `vi` seen from `v`s output direction to matrices `mi.current` and `mi.after` of output indices as seen from `v`.

For output selection to be consistent, either all or none of the indices in a row in `mi.current` must be selected.

For output insertion to be consistent, either all or none of the indices in a row in `mi.after` must be chosen.

 Matrix `mi` has more than one column if activations from `vi` is input (possibly through an arbitrary number of size transparent layers such as BatchNorm) to `v` more than once.

 Furthermore, in the presense of `SizeInvariant` vertices, an indices may be present in more than one `mi`, making selection of indices non-trivial at best in the general case.

 # Examples
 ```julia-repl
julia> iv = inputvertex("in", 2, FluxDense());

julia> v1 = mutable("v1", Dense(2, 3), iv);

julia> v2 = mutable("v2", Dense(2, 5), iv);

julia> v = concat(v1,v2,v1,v2);

julia> cdict = validouts(v);

julia> nout(v)
16

julia> for (vi, mi) in cdict
       @show name(vi)
       display(mi.current)
       end
name(vi) = "v2"
5×2 Array{Int64,2}:
 4  12
 5  13
 6  14
 7  15
 8  16
name(vi) = "v1"
3×2 Array{Int64,2}:
 1   9
 2  10
 3  11
```

"""
validouts(v::AbstractVertex, offs=(0, 0), dd=Dict(), visited = [], out=true) = validouts(trait(v), v, offs, dd, visited, out)
validouts(t::DecoratingTrait, v, offs, dd, visited, out) = validouts(base(t), v, offs, dd, visited, out)
function validouts(::SizeStack, v, offs, dd, visited, out)
    !out && return dd
    has_visited!(visited,(offs, v)) && return dd

    for vin in inputs(v)
        validouts(vin, offs, dd, visited, true)
        offs = offs .+ (nout_org(op(vin)), nout(vin))
    end
    return dd
end

function validouts(::SizeInvariant, v, offs, dd, visited, out)
    has_visited!(visited,(offs, v)) && return dd

    foreach(vin -> validouts(vin, offs, dd, visited, true), inputs(v))
    foreach(vout -> validouts(vout, offs, dd, visited, false), outputs(v))
    return dd
end

function validouts(::SizeAbsorb, v, offs, dd, visited, out)
    !out && return dd
    has_visited!(visited, (offs, v)) && return dd

    initial = length(visited) == 1

    # length(visited) > 1 is only false if the first vertex we call validouts for is of type SizeAbsorb
    # If it is, we need to propagate the call instead of adding indices as the outputs of v might take us to a SizeInvariant vertex which in turn might take us to a SizeStack vertex
    if !initial
        orgsize = nout_org(op(v))
        newsize = nout(v)
        s,n = get(() -> (zeros(Int, orgsize, 0), zeros(Int, newsize, 0) ), dd, v)

        selectfrom = hcat(s, (1:orgsize) .+ offs[1])
        afterselection = hcat(n, (1:newsize) .+ offs[2])
        dd[v] = (current=selectfrom, after=afterselection)
    end
    foreach(vout -> validouts(vout, offs, dd, visited, false), outputs(v))

    # This is true if all outputs of v also are (or lead to) size absorb types and we shall indeed populate dd with the indices of this vertex
    if initial && isempty(dd)
        dd[v] = (current=(1:nout_org(op(v))) .+ offs[1] , after=(1:nout(op(v))) .+ offs[2])
    end

    return dd
end

function has_visited!(visited, x)
    x in visited && return true
    push!(visited, x)
    return false
end

# Step 1: Select which outputs to use given possible constraints described by validouts. Since this might be infeasible to do (in special cases) we get the execute flag which is true if we shall proceed with the selection
"""
    select_outputs(v::AbstractVertex, values)
    select_outputs(s::AbstractSelectionStrategy, v, values)

Returns a tuple `(success, result)` where `result` is a vector so that `Δnout(v, result)` selects outputs for `v` in a way which is consistent with (or as close as possible to) current output size for all vertices visible in the out direction of `v`.

The function generally tries to maximize the `sum(values[selected])` where `selected` is all elements in `results` larger than 0 (negative values in `result` indicates a new output shall be inserted at that position). This however is up to the implementation of the `AbstractSelectionStrategy s`.

Since selection of outputs is not guaranteed to work in all cases, a flag `success` is also returned. If `success` is `false` then calling `Δnout(v, result)` might fail.

See [`validouts`](@ref) for a description of the constraints which may cause the selection to fail.

# Examples
```julia-repl
julia> iv = inputvertex("in", 2, FluxDense());

julia> v1 = mutable("v1", Dense(2, 3), iv);

julia> v2 = mutable("v2", Dense(2, 5), iv);

julia> v = concat(v1,v2,v1,v2);

julia> Δnout(v, -2);

julia> nout(v1)
3

julia> nout(v2)
4

julia> NaiveGAflux.select_outputs(v, 1:nout_org(op(v))) # Dummy values, prefer the higher indices
(true, [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16])

julia> Δnout(v1, 3);

julia> NaiveGAflux.select_outputs(v, 1:nout_org(op(v)))
(true, [1, 2, 3, -1, -1, -1, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, 13, 14, 15, 16])
```

"""
select_outputs(v::AbstractVertex, values) = select_outputs(NoutExact(), v, values)
select_outputs(s::AbstractSelectionStrategy, v, values) = select_outputs(s, v, values, validouts(v))

function select_outputs(s::LogSelection, v, values, cdict)
    @logmsg s.level s.msgfun(v)
    return select_outputs(s.andthen, v, values, cdict)
end

select_outputs(s::SelectionFail, v, values, cdict) = error("Selection failed for vertex $(name(v))")

function select_outputs(s::NoutRevert, v, values, cdict)
    Δ = nout_org(op(v)) - nout(v)
    Δnout(v, Δ)
    return false, 1:nout(v)
end

function select_outputs(s::AbstractJuMPSelectionStrategy, v, values, cdict)
    model = optmodel(s, v, values)

    # variable for selecting a subset of the existing outputs.
    selectvar = @variable(model, selectvar[1:length(values)], Bin)
    # Variable for deciding at what positions to insert new outputs.
    insertvar = @variable(model, insertvar[1:nout(v)], Bin)

    # Will be added to objective to try to insert new neurons last
    # Should not really matter whether it does that or not, but makes things a bit easier to debug
    insertlast = @expression(model, 0)

    for (vi, mi) in cdict
        select_i = rowconstraint(s, model, selectvar, mi.current)
        sizeconstraint(s, vi, model, select_i)
        Δsizeexp = @expression(model, length(select_i) - sum(select_i))

        # Check if size shall be increased
        if length(insertvar) > length(selectvar)
            # This is a bit unfortunate as it basically makes it impossible to relax
            # Root issue is that mi.after is calculated under an the assumption that current nout(vi) will be the size after selection as well. Maybe some other formulation does not have this restricion, but I can't come up with one which is guaranteed to either work or know it has failed.
            insert_i = rowconstraint(s, model, insertvar, mi.after)
            Δsizeexp = @expression(model, Δsizeexp + sum(insert_i))

            @constraint(model, insert_i[1] == 0) # Or else it won't be possible to know where to split
            # This will make sure that we are consistent to what mi.after prescribes
            # Note that this basically prevents relaxation of size constraint, but this is needed because mi.after is calculated assuming nout(vi) is the result after selection.
            # It does offer the flexibility to trade an existing output for a new one should that help resolving something.
            @constraint(model, sum(insert_i) == length(insert_i) - sum(select_i))

            last_i = min(length(select_i), length(insert_i))
            insertlast = @expression(model, insertlast + sum(insert_i[1:last_i]))
        end

        Δfactorconstraint(s, model, minΔnoutfactor(vi), Δsizeexp)

    end

    @objective(model, Max, values' * selectvar - insertlast)

    JuMP.optimize!(model)

    !accept(s, model) && return select_outputs(fallback(s), v, values, cdict)

    # insertvar is 1.0 at indices where a new output shall be added and 0.0 where an existing one shall be selected
    result = -round.(Int, JuMP.value.(insertvar))
    selected = findall(xi -> xi > 0, JuMP.value.(selectvar))

    # TODO: Needs investigation
    sum(result) == 0 && return true, selected

    j = 1
    for i in eachindex(result)
        if result[i] == 0
            result[i] = selected[j]
            j += 1
        end
    end

    return true, result
end

accept(::AbstractJuMPSelectionStrategy, model::JuMP.Model) = JuMP.termination_status(model) != MOI.INFEASIBLE && JuMP.primal_status(model) == MOI.FEASIBLE_POINT # Beware: primal_status seems unreliable for Cbc. See MathOptInterface issue #822

optmodel(::AbstractJuMPSelectionStrategy, v, values) = JuMP.Model(JuMP.with_optimizer(Cbc.Optimizer, loglevel=0))

function rowconstraint(::AbstractJuMPSelectionStrategy, model, x, indmat)
    var = @variable(model, [1:size(indmat,1)], Bin)
    @constraint(model, size(indmat,2) .* var .- sum(x[indmat], dims=2) .== 0)
    return var
end

nselect_out(v) = min(nout(v), nout_org(op(v)))
limits(s::NoutRelaxSize, n) =  (max(1, s.lower * n), s.upper * n)

sizeconstraint(::NoutExact, v, model, var) = @constraint(model, sum(var) == nselect_out(v))
function sizeconstraint(s::NoutRelaxSize, v, model, var)
    # Wouldn't mind being able to relax the size constraint like this:
    #@objective(model, Min, 10*(sum(selectvar) - nout(v))^2)
    # but that requires MISOCP and only commercial solvers seem to do that
    nmin, nmax = limits(s, nselect_out(v))
    @constraint(model, nmin <= sum(var) <= nmax)
end

function Δfactorconstraint(::AbstractJuMPSelectionStrategy, model, f, Δsizeexp)
    # Δfactor constraint:
    #  - Constraint that answer shall result in an integer multiple of f being not selected
    fv = @variable(model, integer=true)
    @constraint(model, f * fv == Δsizeexp)
end
