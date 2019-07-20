
@testset "Unsupported fallback" begin
    struct Dummy <:AbstractMutation end
    @test_throws ArgumentError mutate(Dummy(), "Test")
end
