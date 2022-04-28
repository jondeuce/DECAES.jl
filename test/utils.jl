@testset "GrowableCache" begin
    c = GrowableCache{Float64, Float64}()
    push!(c, (1.0, 2.0))
    @test c.keys == [1.0] && c.values == [2.0]
    push!(c, (3.0, 4.0))
    @test c.keys == [1.0, 3.0] && c.values == [2.0, 4.0]

    c = GrowableCache{Float64, Float64}()
    pushfirst!(c, (1.0, 2.0))
    @test c.keys == [1.0] && c.values == [2.0]
    pushfirst!(c, (3.0, 4.0))
    @test c.keys == [3.0, 1.0] && c.values == [4.0, 2.0]
    pushfirst!(c, (5.0, 6.0))
    @test c.keys == [5.0, 3.0, 1.0] && c.values == [6.0, 4.0, 2.0]

    for (i, (k, v)) in enumerate(c)
        @test k == [5.0, 3.0, 1.0][i]
        @test v == [6.0, 4.0, 2.0][i]
    end

    c = GrowableCache{Float64, Float64}()
    c[1.5] = 2.0
    @test c[1.5] == 2.0
    c[1.5] = 3.0
    @test c[1.5] == 3.0
end

@testset "CachedFunction" begin
    count = Ref(0)
    f_inner(x) = (count[] += 1; x^2)
    c = GrowableCache{Float64, Float64}()
    f = CachedFunction(f_inner, c)

    v = f(1.5)
    @test count[] == length(c.keys) == length(c.values) == 1
    @test c.keys[1] == 1.5 && v == c.values[1] == 2.25

    v = f(1.5)
    @test count[] == length(c.keys) == length(c.values) == 1
    @test v == c.values[1] == 2.25

    v = f(2.0)
    @test count[] == length(c.keys) == length(c.values) == 2
    @test c.keys[1] == 1.5 && c.values[1] == 2.25
    @test c.keys[2] == 2.0 && v == c.values[2] == 4.0

    empty!(c)
    v = f(2.0)
    @test count[] == 3
    @test length(c.keys) == length(c.values) == 2
    @test c.keys[1] == 2.0 && v == c.values[2] == 4.0
end
