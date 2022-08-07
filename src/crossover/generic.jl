
# Useful for dispatch of type crossover(pair::EitherIs{Shielded}) = pair
const MixTuple{T1, T2} = Union{Tuple{T1, T2}, Tuple{T2, T1}}
const EitherIs{T} = MixTuple{T, Any}

# Useful for doing crossover between candiates which wraps a collection of candidates, 
# e.g. Flux.Optimiser and IteratorMaps
function zipcrossover(reiterfun, (c1,c2), crossoverfun)
    cs1,c1re = reiterfun(c1)
    cs2,c2re = reiterfun(c2)
    res = crossoverfun.(zip(cs1,cs2))
    cs1n = (t[1] for t in res)
    cs2n = (t[2] for t in res)
    return c1re(cs1n..., cs1[length(cs2)+1:end]...), c2re(cs2n..., cs2[length(cs1)+1:end]...)
end
