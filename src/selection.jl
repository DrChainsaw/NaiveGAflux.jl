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


# TODO: Remove and replace calls with select_outputs
selectvalidouts(v::AbstractVertex, scorefun::Function) = select_outputs(v, scorefun(v))


"""
    validouts(v::AbstractVertex)

Return a `Dict` mapping vertices `vi` seen from `v`s output direction to matrices `mi` of output indices as seen from `v`.

For output selection to be consistent, either all or none of the indices in a row in `mi` must be selected.

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
       display(mi)
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

function process_selected_outputs(v, cdict::Dict, selected::AbstractVector)
    select_insert = insert_new_outputs(cdict, selected)
    Δnout(v, select_insert)
end

function insert_new_outputs(cdict, selected::AbstractVector{T}) where T<:Integer
    # Due to negative values being used to indicate insertion of neurons, we might end up in situations like this:
    #   v = concat(v1, v2, v1)
    #   v1 shall increase by 2 and v2 shall increase by 3
    #   To make this happen, we need to call Δnout(v, a) where
    #   a = [1,2,...,nout(v1), -1,-1, nout(v1)+1, ..., nout(v1) + nout(v2), -1, -1, -1, nout(v1) + nout(v2) +1, ...,nout(v), -1, -1].
    # Furthermore, for some involved vertices the size has decreased and we might need to select a subset of the outputs. Uhg...

    # Here is how we'll try to tackle it:
    # Add each column from mi to sel_ins as a separate array (selected is an array of arrays).
    # We only pick indices which are part of selected as it represents the solution to the much harder (?) problem to select a subset of indices for those vertices for which the size shall decrease.

    sel_ins = Array{T,1}[]
    for (vi, mi) in cdict
        Δinsert = max(0, nout(vi) - size(mi[1], 1))

        # This is the reason for sel_ins being an array of arrays: Repeated inputs might be interleaved and we must retain the original ordering without moving the negative values around.
        toadd =  [vec(vcat(filter(mii -> mii in selected, mi[1][:,i]), -ones(T,Δinsert))) for i in 1:size(mi[1], 2)]
        @show toadd
        append!(sel_ins, toadd)
    end

     # Now concatenate all column arrays in selected. Sort them by their first element which due to the above is always the smallest. Note that due to how validouts works, we can always assume that all nonnegative numbers in a column array in selected are either strictly greater than or strictly less than all nonnegative numbers of all other column arrays.
     @show sel_ins
     @show selected
     return foldl(vcat, sort(filter(!isempty, sel_ins), by = v -> v[1]), init=T[])

end

# Step 1: Select which outputs to use given possible constraints described by validouts. Since this might be infeasible to do (in special cases) we get the execute flag which is true if we shall proceed with the selection
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

    # Check if size shall be increased
    if length(insertvar) > length(selectvar)
        #insertvar needs to be tied to selectvar for cases when size is relaxed
        @constraint(model, sum(insertvar) == length(insertvar) - sum(selectvar))
    end

    # Will be added to objective to try to insert new neurons last
    # Should not really matter whether it does that or not, but makes things a bit easier to debug
    insertlast = @expression(model, 0)

    for (vi, mi) in cdict
        select_i = rowconstraint(s, model, selectvar, mi.current)
        sizeconstraint(s, vi, model, select_i)

        if length(insertvar) > length(selectvar)
            insmat = mi.after

            insert_i = rowconstraint(s, model, insertvar, insmat)
            @constraint(model, size(insmat, 2) * sum(insert_i) == sum(insertvar[insmat]))

            @constraint(model, insert_i[1] == 0) # Or else it won't be possible to know where to split

            #Isn't this needed? Might be redundant due to constraint on sum and rowconstraint on insertvar
            #@constraint(model, sum(insert_i) == length(insert_i) - sum(select_i))

            last_i = min(length(select_i), length(insert_i))
            insertlast = @expression(model, insertlast + sum(insert_i[1:last_i]))
        end
    end

    @objective(model, Max, values' * selectvar - insertlast)

    JuMP.optimize!(model)

    !accept(s, model) && return select_outputs(fallback(s), v, values, cdict)

    # insertvar is 1.0 at indices where a new output shall be added and 0.0 where an existing one shall be selected
    result = -Int.(JuMP.value.(insertvar))
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
sizeconstraint(::NoutExact, v, model, var) = @constraint(model, sum(var) == nselect_out(v))
function sizeconstraint(s::NoutRelaxSize, v, model, var)
    f = minΔnoutfactor(v)
    nmin = max(1, s.lower * nselect_out(v))
    nmax =  s.upper * nselect_out(v)
    return sizeconstraint(model, var, f, nmin, nmax)
end

# Wouldn't mind being able to relax the size constraint like this:
#@objective(model, Min, 10*(sum(selectvar) - nout(v))^2)
# but that requires MISOCP and only commercial solvers seem to do that
function sizeconstraint(model, var, f, nmin, nmax)
    @constraint(model, nmin <= sum(var) <= nmax)
    # minΔfactor constraint:
    #  - Constraint that answer shall result in an integer multiple of f being not selected
    fv = @variable(model, integer=true)
    @constraint(model, f * fv == sum(var) - length(var))
end
