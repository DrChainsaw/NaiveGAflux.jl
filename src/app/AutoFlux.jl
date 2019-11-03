module AutoFlux

export fit
export ImageClassification, ImageClassifier


"""
    fit(x, y; cb)
    fit(data::Tuple; cb)

Return a population of models fitted to the given data.

The type of model will depend on the shape of `x`.

The following model types are currently supported
- 4D data -> ImageClassifier

Keyword `cb` can be used to supply a callback function which will be called each generation with the current population as input.
"""
function fit(x, y; cb=identity, mdir=missing)
    ndims(x) == 4 && return fit(ImageClassifier(), x, y;cb=identity, mdir=modeldir(mdir, "ImageClassifier"))
    error("No model for $(ndims(x))D data")
end
fit((x,y)::Tuple; cb=identity) = fit(x, y; cb=identity, mdir=missing)

modeldir(::Missing, subdir) = defaultdir(subdir)
modeldir(d, subdir) = d
defaultdir(subdir, basedir = NaiveGAflux.modeldir) = joinpath(basedir, subdir)

include("imageclassification/ImageClassification.jl")
using .ImageClassification


end
