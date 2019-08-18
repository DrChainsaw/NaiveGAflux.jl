# TODO: This file shall go to NaiveNASlib once things work somewhat well

# Methods to help select or add a number of outputs given a new size as this problem apparently belongs to the class of FU-complete problems. And yes, I curse the day I conceived the idea for this project right now...

NaiveNASlib.Δnout(::Immutable, v::AbstractVertex, Δ::T; s) where T<:AbstractArray{<:Integer} = !NaiveNASlib.has_visited_out(s, v) && Δ != 1:nout(v) && error("Tried to change nout of immutable $v to $Δ")

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


struct ValidOutsInfo{I <:Integer, T <: MutationTrait}
    current::Matrix{I}
    after::Matrix{I}
    trait::T
end
ValidOutsInfo(currsize::I, aftersize::I, trait::T) where {T<:MutationTrait,I<:Integer} = ValidOutsInfo(zeros(I, currsize, 0), zeros(I, aftersize, 0), trait)
addinds(v::ValidOutsInfo{I, T}, c::Integer, a::Integer) where {I,T} = ValidOutsInfo([v.current range(c, length=size(v.current,1))], [v.after range(a, length=size(v.after,1))], v.trait)


struct NoutMainVar <: AbstractJuMPSelectionStrategy
    main::AbstractJuMPSelectionStrategy
    child::AbstractJuMPSelectionStrategy
end
NoutMainVar() = NoutMainVar(NoutExact(), NoutRelaxSize())
NoutMainVar(m::LogSelection, c) = LogSelection(m.level, m.msgfun, NoutMainVar(m.andthen, s))
NoutMainVar(m::AbstractSelectionStrategy, c) = m
fallback(s::NoutMainVar) = NoutMainVar(fallback(s.main), fallback(s.child))


"""
    validouts(v::AbstractVertex)

Return a `Dict` mapping vertices `vi` seen from `v`s output direction to `ValidOutsInfo mi`.

For output selection to be consistent, either all or none of the indices in a row in `mi.current` must be selected.

For output insertion to be consistent, either all or none of the indices in a row in `mi.after` must be chosen.

 Matrices in `mi` have more than one column if activations from `vi` is input (possibly through an arbitrary number of size transparent layers such as BatchNorm) to `v` more than once.

 Furthermore, in the presense of `SizeInvariant` vertices, some indices may be present in more than one `mi`, making selection of indices non-trivial at best in the general case.

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
validouts(v::AbstractVertex, skipin::Set, skipout::Set, out::Bool=true) = validouts(v, out, v, Dict(), (), (1, 1), maskinit(v), Set(), skipin, skipout)
function validouts(v::AbstractVertex, out::Bool=true, vfrom::AbstractVertex=v, dd=Dict(), path=(), offs=(1, 1), mask=maskinit(v), visited = Set(), skipin::Set=Set(), skipout::Set=Set())
    has_visited!(visited, v) && return dd
    out && v in skipin && return dd
    !out && v in skipout && return dd
    validouts(v, Val(out), vfrom, dd, path, offs, mask, visited, skipin, skipout)
    delete!(visited, v)
    return dd
end
maskinit(v) = (trues(nout_org(v)), trues(nout(v)))
validouts(v::AbstractVertex, args...) = validouts(trait(v), v, args...)
validouts(t::DecoratingTrait, args...) = validouts(base(t), args...)

function validouts(::SizeStack, v, ::Val{true}, vfrom, dd, path, offs, mask, args...)
    cnts = (1, 1)
    for vin in inputs(v)
        next = cnts .+ (nout_org(vin), nout(vin)) .- 1
        newmask = map(mt -> mt[1][mt[2]:mt[3]], zip(mask, cnts, next))
        cnts = cnts .+ (nout_org(vin), nout(vin))
        validouts(vin, true, v, dd, path, offs, newmask, args...)
        offs = offs .+ sum.(newmask)
    end
    return dd
end

function validouts(::SizeStack, v, ::Val{false}, vfrom, dd, path, offs, mask, args...)
    newmask = (BitVector(), BitVector())
    for vin in inputs(v)
        if vin == vfrom
            append!.(newmask, mask)
        else
            append!.(newmask, map(mv -> .!mv, maskinit(vin)))
        end
    end

    for (p, vout) in enumerate(outputs(v))
        newpath = (path..., p)
        validouts(vout, false, v, dd, newpath, offs, newmask, args...)
    end
    return dd
end

function validouts(::SizeInvariant, v, out, vfrom, dd, path, args...)
    foreach(vin -> validouts(vin, true, v, dd, path, args...), inputs(v))
    for (p, vout) in enumerate(outputs(v))
        newpath = (path..., p)
        validouts(vout, false, v, dd, newpath, args...)
    end
    return dd
end

validouts(t::SizeAbsorb, args...) = addvalidouts(t, args...)
validouts(t::Immutable, args...) = addvalidouts(t, args...)

function addvalidouts(t::MutationTrait, v, ::Val{true}, vfrom, dd, path, offs, mask, visited, args...)
    initial = length(visited) == 1

    # length(visited) > 1 is only false if the first vertex we call validouts for is of type SizeAbsorb
    # If it is, we need to propagate the call instead of adding indices as the outputs of v might take us to a SizeInvariant vertex which in turn might take us to a SizeStack vertex
    orgsize = sum(mask[1])
    newsize = sum(mask[2])
    if !initial
        info = get(() -> ValidOutsInfo(orgsize, newsize, t), dd, v)
        dd[v] = addinds(info, offs...)
    end
    for (p, vout) in enumerate(outputs(v))
        newpath = (path..., p)
        validouts(vout, false, v, dd, newpath, offs, mask, visited, args...)
    end

    # This is true if all outputs of v also are (or lead to) size absorb types and we shall indeed populate dd with the indices of this vertex
    if initial && isempty(dd)
        dd[v] = addinds(ValidOutsInfo(orgsize, newsize, t), offs...)
    end

    return dd
end
function addvalidouts(t::MutationTrait, v, ::Val{false}, vfrom, dd, args...)
    foreach(vin -> validouts(vin, true, v, dd, args...), inputs(v))
    return dd
end

function has_visited!(visited, x)
    x in visited && return true
    push!(visited, x)
    return false
end

NaiveNASlib.nout_org(v::AbstractVertex) = nout_org(trait(v), v)
NaiveNASlib.nout_org(t::DecoratingTrait, v) = nout_org(base(t), v)
NaiveNASlib.nout_org(::MutationSizeTrait, v::MutationVertex) = nout_org(op(v))
NaiveNASlib.nout_org(::Immutable, v) = nout(v)

NaiveNASlib.nin_org(v::AbstractVertex) = nin_org(trait(v), v)
NaiveNASlib.nin_org(t::DecoratingTrait, v) = nin_org(base(t), v)
NaiveNASlib.nin_org(::MutationSizeTrait, v::MutationVertex) = nin_org(op(v))
NaiveNASlib.nin_org(::Immutable, v) = nout(v)

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
select_outputs(v::AbstractVertex, values, skipin=[], skipout=[]) = select_outputs(NoutExact(), v, values, skipin, skipout)
select_outputs(s::AbstractSelectionStrategy, v, values, skipin=[], skipout=[])= select_outputs(s, v, values, validouts(v, Set(skipin), Set(skipout)))

function select_outputs(s::LogSelection, v, values, cdict)
    @logmsg s.level s.msgfun(v)
    return select_outputs(s.andthen, v, values, cdict)
end

select_outputs(s::SelectionFail, v, values, cdict) = error("Selection failed for vertex $(name(v))")

function select_outputs(s::NoutRevert, v, values, cdict)
    if !ismissing(minΔnoutfactor(v))
        Δ = nout_org(v) - nout(v)
        Δnout(v, Δ)
    end
    return false, 1:nout(v)
end

function select_outputs(s::AbstractJuMPSelectionStrategy, v, values, cdict)
    model = optmodel(s, v, values)

    # variable for selecting a subset of the existing outputs.
    selectvar = @variable(model, selectvar[1:length(values)], Bin)
    # Variable for deciding at what positions to insert new outputs.
    insertvar = @variable(model, insertvar[1:nout(v)], Bin)

    # insertlast will be added to objective to try to insert new neurons last
    # Should not really matter whether it does that or not, but makes things a bit easier to debug
    insertlast = sizeconstraintmainvar(s, SizeAbsorb(), v, model, selectvar, insertvar)

    for (vi, mi) in cdict
        # TODO: Don't add to cdict if this happens?
        isempty(mi.current) && continue
        select_i = rowconstraint(s, model, selectvar, mi.current)
        sizeconstraint(s, mi.trait, vi, model, select_i)
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
    # Valid rows don't include all rows when vertex to select from has not_org < nout_org of a vertex in cdict.
    # This typically happens when resizing vertex to select from due to removal of its output vertex
    validrows = [i for i in 1:size(indmat, 1) if all(indmat[i,:] .<= length(x))]
    var = @variable(model, [1:length(validrows)], Bin)
    @constraint(model, size(indmat,2) .* var .- sum(x[indmat[validrows,:]], dims=2) .== 0)
    return var
end


sizeconstraintmainvar(::AbstractJuMPSelectionStrategy, t, v, model, selvar, insvar) = @expression(model, 0)
function sizeconstraintmainvar(s::NoutMainVar, t, v, model, selvar, insvar)
    toselect = min(length(selvar), length(insvar))
    sizeconstraint(s.main, toselect, model, selvar)
    if length(insvar) > length(selvar)
        @constraint(model, sum(insvar) == length(insvar) - sum(selvar))
    end
    return @expression(model, sum(insvar[1:toselect]))
end

function sizeconstraint(s::NoutMainVar, t, v, model, var)
    #sizeconstraint(s.child, t, v, model, var)
end

nselect_out(v) = min(nout(v), nout_org(v))
limits(s::NoutRelaxSize, n) =  (max(1, s.lower * n), s.upper * n)

sizeconstraint(s::AbstractJuMPSelectionStrategy, t, v, model, var) = sizeconstraint(s, min(length(var), nselect_out(v)), model, var)
sizeconstraint(s::NoutRelaxSize, t::Immutable, v, model, var) = sizeconstraint(NoutExact(), t, v, model, var)

sizeconstraint(::NoutExact, size::Integer, model, var) = @constraint(model, sum(var) == size)
function sizeconstraint(s::NoutRelaxSize, sizetarget::Integer, model, var)
    # Wouldn't mind being able to relax the size constraint like this:
    #@objective(model, Min, 10*(sum(selectvar) - sizetarget^2))
    # but that requires MISOCP and only commercial solvers seem to do that
    nmin, nmax = limits(s, sizetarget)
    @constraint(model, nmin <= sum(var) <= nmax)
end

Δfactorconstraint(s::NoutMainVar, model, f, Δsizeexp) = Δfactorconstraint(s.child, model, f, Δsizeexp)
function Δfactorconstraint(::NoutExact, model, f, Δsizeexp) end
function Δfactorconstraint(::NoutRelaxSize, model, ::Missing, Δsizeexp) end
function Δfactorconstraint(::NoutRelaxSize, model, f, Δsizeexp)
    # Δfactor constraint:
    #  - Constraint that answer shall result in an integer multiple of f being not selected
    fv = @variable(model, integer=true)
    @constraint(model, f * fv == Δsizeexp)
end
