using Documenter, Literate, NaiveGAflux, NaiveGAflux.AutoFlux, NaiveGAflux.AutoFlux.ImageClassification

const nndir = joinpath(dirname(pathof(NaiveGAflux)), "..")

function literate_example(sourcefile; rootdir=nndir, sourcedir = "test/examples", destdir="docs/src/examples")
    fullpath = Literate.markdown(joinpath(rootdir, sourcedir, sourcefile), joinpath(rootdir, destdir); flavor=Literate.DocumenterFlavor(), mdstrings=true, codefence="````julia" => "````")
    dirs = splitpath(fullpath)
    srcind = findfirst(==("src"), dirs)
    joinpath(dirs[srcind+1:end]...)
end

quicktutorial = literate_example("quicktutorial.jl")
searchspace_ex = literate_example("searchspace.jl")
mutation_ex = literate_example("mutation.jl")
crossover_ex = literate_example("crossover.jl")
fitness_ex = literate_example("fitness.jl")
candidate_ex = literate_example("candidate.jl")
evolution_ex = literate_example("evolution.jl")
iterators_ex = literate_example("iterators.jl")

makedocs(   sitename="NaiveGAflux",
            root = joinpath(nndir, "docs"), 
            format = Documenter.HTML(
                prettyurls = get(ENV, "CI", nothing) == "true"
            ),
            pages = [
                "index.md",
                quicktutorial,
                "AutoFlux" => [
                    "autoflux/index.md",
                    "autoflux/reference/reference.md"
                ],
                "Components" => [
                    searchspace_ex,
                    mutation_ex,
                    crossover_ex,
                    fitness_ex,
                    candidate_ex,
                    evolution_ex,
                    iterators_ex
                ],
                "API Reference" => [
                    "reference/searchspace.md",
                    "reference/mutation.md",
                    "reference/crossover.md",
                    "reference/fitness.md",
                    "reference/candidate.md",
                    "reference/evolution.md",
                    "reference/batchsize.md",
                    "reference/iterators.md",
                    "reference/utils.md",
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
        repo = "github.com/DrChainsaw/NaiveGAflux.jl.git",
        push_preview=true
    )
end