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


function selectvalidouts(v::AbstractVertex, scorefun::Function)
    score = scorefun(v)
    valouts = validouts(v)

    selected = nothing
    for (vi, mi) in valouts
        Δout = nout(vi) - size(mi, 1)
        if Δout < 0
            # Case 1: Size has decreased, we need to select a subset of the outputs based on the score

            # Step 1: The constraint is that we must pick all values in a row of mi for neuron selection to be consistent. Matrix mi has more than one column if activations from vi is input (possibly through an arbitrary number of size transparent layers such as BatchNorm) to v more than once.
            bestrows = sortperm(vec(sum(score[mi], dims=2)), lt = >)
            # Step 2: Due to negative values being used to indicate insertion of neurons, we must do this:
            # Add each column from mi to selected as a separate array (selected is an array of arrays)
            # Each columns array in selected individually sorted here to keep unnecessary shuffling of neurons to a minimum.
            toadd = [sort(vec(mi[bestrows[1:nout(vi)], i])) for i in 1:size(mi, 2)]
            isnothing(selected) ? selected = toadd : append!(selected, toadd)
        else
            # Case 2: Size has not decreased,
            # We want to insert -1s to show where new columns are to be added

            # This is the reason for selected being an array of arrays: Activations from multiple repeated inputs might be interleaved and we must retain the original neuron ordering without moving the negative values around.
            toadd =  [vec(vcat(mi[:,i], -ones(Int,Δout, 1))) for i in 1:size(mi, 2)]
            isnothing(selected) ? selected = toadd : append!(selected, toadd)
        end
    end

     # Now concatenate all column arrays in selected. Sort them by their first element which due to the above is always the smallest. Note that due to how validouts works, we can always assume that all nonnegative numbers in a column array in selected are either strictly greater than or strictly less than all nonnegative numbers of all other column arrays.
     return foldl(vcat, sort(selected, by = v -> v[1]))
end

validouts(v::AbstractVertex, offs=0, dd=Dict(), visited = [], out=true) = validouts(trait(v), v, offs, dd, visited, out)
validouts(t::DecoratingTrait, v, offs, dd, visited, out) = validouts(base(t), v, offs, dd, visited, out)
function validouts(::SizeStack, v, offs, dd, visited, out)
    !out && return dd
    has_visited!(visited,(offs, v)) && return dd

    for vin in inputs(v)
        validouts(vin, offs, dd, visited, true)
        offs += nout_org(op(vin))
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
        selectfrom = hcat(get(() -> zeros(Int, orgsize, 0), dd, v), (1:orgsize) .+ offs)
        dd[v] = selectfrom
    end
    foreach(vout -> validouts(vout, offs, dd, visited, false), outputs(v))

    # This is true if all outputs of v also are (or lead to) size absorb types and we shall indeed populate dd with the indices of this vertex
    if initial && isempty(dd)
        dd[v] = (1:nout_org(op(v))) .+ offs
    end

    return dd
end

function has_visited!(visited, x)
    x in visited && return true
    push!(visited, x)
    return false
end


select_outputs(v::AbstractVertex, values) = select_outputs(NoutExact(), v, values)

function select_outputs(s::AbstractSelectionStrategy, v, values)
    execute, selected = select_outputs(s, v, values, validouts(v))

    if execute
        #TODO: Need to also handle neuron insertion, similar to how it is done in select_neurons above
        Δnout(v, selected)
    end
end

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
    model, mainvar = mainmodel(s, values)

    # Wouldn't mind being able to relax the size constraint, but that requires MISOCP and only commercial solvers seem to do that
    #@objective(model, Min, 10*(sum(mainvar) - nout(vselect))^2)
    sizeconstraint(s, v, model, mainvar)

    for (vi, mi) in cdict
        select_i = rowconstraint(s, model, mainvar, mi)
        sizeconstraint(s, vi, model, select_i)
    end

    JuMP.optimize!(model)

    !accept(s, model) && return select_outputs(fallback(s), v, values, cdict)
    return true, findall(xi -> xi > 0, JuMP.value.(mainvar))
end

accept(::AbstractJuMPSelectionStrategy, model::JuMP.Model) = JuMP.termination_status(model) != MOI.INFEASIBLE && JuMP.primal_status(model) == MOI.FEASIBLE_POINT # Beware: primal_status seems unreliable for Cbc. See MathOptInterface issue #822

function mainmodel(::AbstractJuMPSelectionStrategy, values)

    model = JuMP.Model(JuMP.with_optimizer(Cbc.Optimizer, loglevel=0))

    x = @variable(model, x[1:length(values)], Bin)
    @objective(model, Max, values' * x)

    return model, x
end

function rowconstraint(::AbstractJuMPSelectionStrategy, model, x, indmat)
    var = @variable(model, [1:size(indmat,1)], Bin)
    @constraint(model, size(indmat,2) .* var .- sum(x[indmat], dims=2) .== 0)
    return var
end

nselect_out(v) = min(nout(v), nout_org(op(v)))
sizeconstraint(::NoutExact, v, model, var) = @constraint(model, sum(var) == nselect_out(v))
function sizeconstraint(s::NoutRelaxSize, v, model, var)
    f = minΔnoutfactor(v)
    nmin = max(1, s.lower * nselect_out(v))
    nmax =  s.upper * nselect_out(v)
    return sizeconstraint(model, var, f, nmin, nmax)
end

function sizeconstraint(model, var, f, nmin, nmax)
    @constraint(model, nmin <= sum(var) <= nmax)
    # minΔfactor constraint:
    #  - Constraint that answer shall result in an integer multiple of f being not selected
    fv = @variable(model, integer=true)
    @constraint(model, f * fv == sum(var) - length(var))
end
