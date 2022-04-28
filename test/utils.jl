using Test
using DECAES
using DECAES: GrowableCache, CachedFunction

@testset "GrowableCache/CachedFunction" begin
    c = GrowableCache(Float64[], Float64[], Ref(0))
    push!(c, (1.0, 2.0))
    @test keys(c) == [1.0] && values(c) == [2.0]
    push!(c, (3.0, 4.0))
    @test keys(c) == [1.0, 3.0] && values(c) == [2.0, 4.0]

    c = GrowableCache(Float64[], Float64[], Ref(0))
    pushfirst!(c, (1.0, 2.0))
    @test keys(c) == [1.0] && values(c) == [2.0]
    pushfirst!(c, (3.0, 4.0))
    @test keys(c) == [3.0, 1.0] && values(c) == [4.0, 2.0]
    pushfirst!(c, (5.0, 6.0))
    @test keys(c) == [5.0, 3.0, 1.0] && values(c) == [6.0, 4.0, 2.0]

    for (i, (k, v)) in enumerate(c)
        @test k == [5.0, 3.0, 1.0][i]
        @test v == [6.0, 4.0, 2.0][i]
    end

    c = GrowableCache(Float64[], Float64[], Ref(0))
    c[1.5] = 2.0
    @test c[1.5] == 2.0
    c[1.5] = 3.0
    @test c[1.5] == 3.0

    count = Ref(0)
    f_inner = x -> (count[] += 1; x^2)
    c = GrowableCache(Float64[], Float64[], Ref(0))
    f = CachedFunction(f_inner, c)

    v = f(1.5)
    @test count[] == length(keys(c)) == length(values(c)) == 1
    @test keys(c)[1] == 1.5 && v == values(c)[1] == 2.25

    v = f(1.5)
    @test count[] == length(keys(c)) == length(values(c)) == 1
    @test v == values(c)[1] == 2.25

    v = f(2.0)
    @test count[] == length(keys(c)) == length(values(c)) == 2
    @test keys(c)[1] == 1.5 && values(c)[1] == 2.25
    @test keys(c)[2] == 2.0 && v == values(c)[2] == 4.0

    empty!(c)
    v = f(2.0)
    @test count[] == 3
    @test length(keys(c)) == length(values(c)) == 2
    @test keys(c)[1] == 2.0 && v == values(c)[2] == 4.0
end
