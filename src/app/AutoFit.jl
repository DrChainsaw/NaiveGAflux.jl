module AutoFit

export fit
export ImageClassifier

"""
    fit(x, y; cb)
    fit(data::Tuple; cb)

Return a population of models fitted to the given data.

The type of model will depend on the shape of `x`.

The following model types are currently supported
- 4D data -> ImageClassifier

Keyword `cb` cen be used to supply a callback function which will be called each generation with the current population as input.
"""
function fit(x, y; cb=identity, mdir=missing)
    ndims(x) == 4 && return fit(ImageClassifier(), x, y;cb=identity, mdir=ismissing(mdir) ? defaultdir("ImageClassifier") : mdir)
    error("No model for $(ndims(x))D data")
end
fit((x,y)::Tuple; cb=identity) = fit(x, y; cb=identity, mdir=missing)

defaultdir(subdir, basedir = NaiveGAflux.modeldir) = joinpath(basedir, subdir)

include("ImageClassification.jl")
using .ImageClassification


end
