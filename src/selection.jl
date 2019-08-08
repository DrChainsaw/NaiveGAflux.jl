# TODO: This file shall go to NaiveNASlib once things work somewhat well

# Methods to help select or add a number of outputs given a new size as this problem apparently belongs to the class of FU-complete problems. And yes, I curse the day I conceived the idea for this project right now...

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


function select_outputs(v, values)
    valouts = validouts(v)
    noutsfun(v) = min(nout(v), nout_org(op(v)))
    model, selected = select_outputs(v, values, valouts, noutsfun)

    if (JuMP.termination_status(model) == MOI.INFEASIBLE) || (JuMP.primal_status(model) != MOI.FEASIBLE_POINT)
        # Relax size changes
        sizefun(v) = minΔnoutfactor(v), max(1, noutsfun(v) ÷ 2), noutsfun(v)
        model, selected = select_outputs(v, values, valouts, sizefun)
    end

    Δnout(v, selected)
end

function select_outputs(vselect, values, cdict, sizefun)
    model, mainvar = mainmodel(values)

    # Wouldn't mind being able to relax the size constraint, but that requires MISOCP and only commercial solvers seem to do that
    #@objective(model, Min, 10*(sum(mainvar) - nout(vselect))^2)
    sizeconstraint(model, mainvar, sizefun(vselect)...)

    for (vi, mi) in cdict
        select_i = rowconstraint(model, mainvar, mi)
        sizeconstraint(model, select_i, sizefun(vi)...)
    end

    JuMP.optimize!(model)

    return model, findall(xi -> xi > 0, JuMP.value.(mainvar))
end

function mainmodel(values)

    model = JuMP.Model(JuMP.with_optimizer(Cbc.Optimizer, loglevel=0))

    x = @variable(model, x[1:length(values)], Bin)
    @objective(model, Max, values' * x)

    return model, x
end

function rowconstraint(model, x, indmat)
    var = @variable(model, [1:size(indmat,1)], Bin)
    @constraint(model, size(indmat,2) .* var .- sum(x[indmat], dims=2) .== 0)
    return var
end

sizeconstraint(model, var, nselect) = @constraint(model, sum(var) == nselect)
function sizeconstraint(model, var, f, nmin, nmax)
    @constraint(model, nmin <= sum(var) <= nmax)
    # minΔfactor constraint:
    #  - Constraint that answer shall result in an integer multiple of f being not selected
    fv = @variable(model, integer=true)
    @constraint(model, f * fv == sum(var) - length(var))
end
