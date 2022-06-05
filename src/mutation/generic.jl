"""
    AbstractMutation{T}

Abstract type defining a mutation operation on entities of type `T`.

Implementations are expected to be callable using an entity of type `T` as only input.

May also implement a callable accepting a `AbstractVector{<:T}` if it is useful to work on
all items to mutate at once.
"""
abstract type AbstractMutation{T} end

(m::AbstractMutation{T})(es::AbstractVector{<:T}) where T = m.(es)

"""
    AbstractCrossover{T}

Type alias for `AbstractMutation{Tuple{T,T}}` defining a crossover of two entities of type `T`.

Implementations are expected to be callable using a tuple of two type `T` as only input.
"""
const AbstractCrossover{T} = AbstractMutation{Tuple{T,T}}

"""
    DecoratingMutation{T}

Abstract type indicating that the type itself does not perform any mutation but wraps a type which might do.

Must either implement callable method for `AbstractVector{<:T}` or accept keyword arguments `next=wrapped(m)` and
`noop=identity` along with a single `T`.
"""
abstract type DecoratingMutation{T} <: AbstractMutation{T} end
wrapped(m::DecoratingMutation) = m.m

mutationleaves(m::DecoratingMutation) = (mutationleaves(wrapped(m))...,)
mutationleaves(tm::Tuple) = mapreduce(mutationleaves, (t1,t2) -> (t1...,t2...), tm)
mutationleaves(m) = tuple(m)

# Apart from being overengineered this helps protecting against forgetting to handle arrays in a DecoratingMutation
# The next/noop happens to work with most existing DecoratingMutations, but it is a bit arbitrary and in some cases 
# one must implement both the per-element and the vector of elements versions.
function (m::DecoratingMutation{T})(es::AbstractVector{<:T}) where T 
    cnt = Ref(1)
    fornext = Int[]
    next = function(e) 
        push!(fornext, cnt[])
        cnt[] += 1
        e
    end
    noop = function(e)
        cnt[] += 1
        e
    end

    allres = m.(es; next, noop)
    mres = wrapped(m)(es[fornext])

    # Mutation might accidentally widen the type compared to allres and then we can't insert mres into allres.
    # Lets fix that if it happens
    RT = typejoin(eltype(allres), eltype(mres))
    res = RT === eltype(allres) ? allres :  convert(Vector{RT}, allres)

    res[fornext] = mres
    return res
end

"""
    MutationProbability{T} <: DecoratingMutation{T}
    MutationProbability(m::AbstractMutation{T}, p::Probability)
    MutationProbability(m::AbstractMutation{T}, p::Number)

Applies `m` with probability `p`.
"""
struct MutationProbability{T, P<:Probability} <: DecoratingMutation{T}
    m::AbstractMutation{T}
    p::P
end
MutationProbability(m::AbstractMutation{T}, p::Number) where T = MutationProbability(m, Probability(p))
(m::MutationProbability{T})(e::T; next=m.m, noop=identity) where T = apply(() -> next(e), m.p, () -> noop(e))

"""
    WeightedMutationProbability{T,F} <: DecoratingMutation{T}
    WeightedMutationProbability(m::AbstractMutation::T, pfun::F)

Applies `m` to an entity `e` with a probability `pfun(e)`.
"""
struct WeightedMutationProbability{T,F} <: DecoratingMutation{T}
    m::AbstractMutation{T}
    pfun::F
end
(m::WeightedMutationProbability{T})(e::T; next=m.m, noop=identity) where T = apply(() -> next(e), m.pfun(e), () -> noop(e))

"""
    HighUtilityMutationProbability(m::AbstractMutation{T}, pbase::Real, rng=rng_default; spread=0.5)

Return a `WeightedMutationProbability` which applies `m` to vertices with an (approximately) average probability of `pbase` and where high `neuronutility` compared to other vertices in same graph means higher probability.

Parameter `spread` can be used to control how much the difference in probability is between high and low utlity. High spread means high difference while low spread means low difference.
"""
HighUtilityMutationProbability(m::AbstractMutation{T}, pbase::Real, rng=rng_default;spread=0.5) where T <: AbstractVertex = WeightedMutationProbability(m, weighted_neuronutility_high(pbase, rng,spread=spread))

"""
    LowUtilityMutationProbability(m::AbstractMutation{T}, pbase::Real, rng=rng_default; spread=2)

Return a `WeightedMutationProbability` which applies `m` to vertices with an (approximately) average probability of `pbase` and where low `neuronutility` compared to other vertices in same graph means higher probability.

Parameter `spread` can be used to control how much the difference in probability is between high and low utlity. High spread means high difference while low spread means low difference.
"""
LowUtilityMutationProbability(m::AbstractMutation{T}, pbase::Real, rng=rng_default;spread=2) where T <: AbstractVertex = WeightedMutationProbability(m, weighted_neuronutility_low(pbase, rng, spread=spread))


weighted_neuronutility_high(pbase, rng=rng_default; spread=0.5) = function(v::AbstractVertex)
    ismissing(NaiveNASflux.neuronutility(v)) && return pbase
    return Probability(fixnan(pbase ^ normexp(v, spread), pbase), rng)
end

weighted_neuronutility_low(pbase, rng=rng_default;spread=2) = function(v::AbstractVertex)
    ismissing(NaiveNASflux.neuronutility(v)) && return pbase
    return Probability(fixnan(pbase ^ (1/normexp(v, 1/spread)), pbase), rng)
end

fixnan(x, rep) = isnan(x) ? rep : clamp(x, 0.0, 1.0)

# This is pretty hacky and arbitrary. Change to something better
function normexp(v::AbstractVertex, s)
    allvertices = filter(allow_mutation, all_in_graph(v))
    allvalues = map(vi -> NaiveNASflux.neuronutility(vi), allvertices)
    meanvalues = map(mean, skipmissing(allvalues))
    meanvalue = mean(meanvalues)
    maxvalue = maximum(meanvalues)
    utlity = mean(NaiveNASflux.neuronutility(v))
    # Basic idea: maxvalue - utlity means the (to be) exponent is <= 0 while the division seems to normalize so that 
    # average of pbase ^ normexp across allvertices is near pbase (no proof!). The factor 2 is just to prevent 
    # probability of vertex with maxvalue to be 1.
    return (2maxvalue^s - utlity^s) / (2maxvalue^s - meanvalue^s)
end


"""
    MutationChain{T} <: DecoratingMutation{T}
    MutationChain(m::AbstractMutation{T}...)

Chains multiple `AbstractMutation{T}`s after each other.

Input entities will be mutated by the first `AbstractMutation{T}` in the chain and the output will be fed into the next `AbstractMutation{T}` in the chain and so on. The output from the last `AbstractMutation{T}` is returned.
"""
struct MutationChain{T} <: DecoratingMutation{T}
    m::Tuple{Vararg{AbstractMutation{T}}}
end
MutationChain(m::AbstractMutation{T}...) where T = MutationChain(m)
# Identical, but can't use Union due to ambiguity
(m::MutationChain{T})(es::AbstractVector{<:T}) where T = foldl((ei, mi) -> mi(ei), m.m; init=es)
(m::MutationChain{T})(e::T) where T = foldl((ei, mi) -> mi(ei), m.m; init=e)

"""
    RecordMutation{T} <: DecoratingMutation{T}
    RecordMutation(m::AbstractMutation{T})

Records all mutated entities.

Intended use case is to be able to do parameter selection on mutated vertices.
"""
struct RecordMutation{T} <: DecoratingMutation{T}
    m::AbstractMutation{T}
    mutated::Vector{T}
end
RecordMutation(m::AbstractMutation{T}) where T = RecordMutation(m, T[])
function (m::RecordMutation{T})(e::T; next=m.m, noop=identity) where T
    em = next(e)
    push!(m.mutated, em)
    return em
end
function fetchmutated!(m::RecordMutation)
    mutated = copy(m.mutated)
    deleteat!(m.mutated, eachindex(m.mutated))
    return mutated
end

"""
    LogMutation{T} < :DecoratingMutation{T}
    LogMutation(strfun, m::AbstractMutation{T}; level = Logging.Info, nextlogfun=e -> PrefixLogger("   "))
    LogMutation(strfun, level::LogLevel, nextlogfun, m::AbstractMutation{T})

Logs all mutation operations.

Argument `strfun` maps the mutated entity to the logged string.

Calling `nextlogfun(e)` where `e` is the entity to mutate produces an `AbstractLogger` which will be used when applying `m(e)`.

By default, this is used to add a level of indentation to subsequent logging calls which makes logs of hierarchical mutations (e.g. mutate a CompGraph by applying mutations to some of its vertices) easier to read. Set `nextlogfun = e -> current_logger()` to remove this behaviour.
"""
struct LogMutation{T,F,L<:LogLevel,LF} <: DecoratingMutation{T}
    strfun::F
    level::L
    nextlogfun::LF
    m::AbstractMutation{T}
end
LogMutation(strfun, m::AbstractMutation{T}; level = Logging.Info, nextlogfun=e -> PrefixLogger("   ")) where T = LogMutation(strfun, level, nextlogfun, m)
function (m::LogMutation{T})(e::T; next=m.m, noop=identity) where T
    @logmsg m.level m.strfun(e)
    return with_logger(() -> next(e), m.nextlogfun(e))
end

"""
    MutationFilter{T} <: DecoratingMutation{T}
    MutationFilter(predicate, m)

Applies mutation `m` only for entities `e` for which `predicate(e)` returns true.
"""
struct MutationFilter{T,P} <: DecoratingMutation{T}
    predicate::P
    m::AbstractMutation{T}
end
function (m::MutationFilter{T})(e::T; next=m.m, noop=identity) where T
    m.predicate(e) && return next(e)
    return noop(e)
end

"""
    PostMutation{T} <: DecoratingMutation{T}
    PostMutation(actions, m::AbstractMutation{T})
    PostMutation(m::AbstractMutation{T}, actions...)

Performs a set of actions after a wrapped `AbstractMutation` is applied.

Actions will be invoked with arguments (m::PostMutation{T}, e::T) where m is the enclosing `PostMutation` and `e` is the mutated entity of type `T`.
"""
struct PostMutation{T,A} <: DecoratingMutation{T}
    actions::A
    m::AbstractMutation{T}
end
PostMutation(m::AbstractMutation{T}, actions...) where T = PostMutation(actions, m)
PostMutation(action::Function, m::AbstractMutation{T}) where T = PostMutation(m, action)
function (m::PostMutation{T})(e::T; next=m.m, noop=identity) where T
    eout = next(e)
    foreach(a -> a(m, eout), m.actions)
    return eout
end

