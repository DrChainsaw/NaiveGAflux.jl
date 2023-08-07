@testset "Examples" begin
    using Markdown: @md_str
    include("examples/quicktutorial.jl")
    include("examples/performancetips.jl")
    include("examples/searchspace.jl")
    include("examples/mutation.jl")
    include("examples/crossover.jl")
    include("examples/fitness.jl")
    include("examples/candidate.jl")
    include("examples/evolution.jl")
    include("examples/iteratormaps.jl")
    include("examples/iterators.jl")
end
