using Documenter, Literate, NaiveGAflux

const nndir = joinpath(dirname(pathof(NaiveGAflux)), "..")

function literate_example(sourcefile; rootdir=nndir, sourcedir = "test/examples", destdir="docs/src/examples")
    fullpath = Literate.markdown(joinpath(rootdir, sourcedir, sourcefile), joinpath(rootdir, destdir); flavor=Literate.DocumenterFlavor(), mdstrings=true, codefence="````julia" => "````")
    dirs = splitpath(fullpath)
    srcind = findfirst(==("src"), dirs)
    joinpath(dirs[srcind+1:end]...)
end

quicktutorial = literate_example("quicktutorial.jl")
searchspace = literate_example("searchspace.jl")
mutation = literate_example("mutation.jl")
crossover = literate_example("crossover.jl")
fitness = literate_example("fitness.jl")
candidate = literate_example("candidate.jl")
evolution = literate_example("evolution.jl")
iterators = literate_example("iterators.jl")

makedocs(   sitename="NaiveGAflux",
            root = joinpath(nndir, "docs"), 
            format = Documenter.HTML(
                prettyurls = get(ENV, "CI", nothing) == "true"
            ),
            pages = [
                "index.md",
                quicktutorial,
                "autoflux/index.md",
                "Components" => [
                    searchspace,
                    mutation,
                    crossover,
                    fitness,
                    candidate,
                    evolution,
                    iterators
                ],
                "API Reference" => [
                    "reference/searchspace.md",
                    "reference/mutation.md",
                    "reference/crossover.md",
                    "reference/fitness.md",
                    "reference/candidate.md",
                    "reference/evolution.md",
                    "reference/iterators.md"
                ]
            ],
            modules = [NaiveGAflux],
        )

function touchfile(filename, rootdir=nndir, destdir="test/examples")
    filepath = joinpath(rootdir, destdir, filename)
    isfile(filepath) && return
    write(filepath, """
    md\"\"\"
    # Markdown header
    \"\"\"
    """)
end

if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/DrChainsaw/NaiveGAflux.jl.git"
    )
end