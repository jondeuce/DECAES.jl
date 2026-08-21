@testset "bisect_root" begin
    f = sin
    x, fx = DECAES.bisect_root(f, 3.0, 4.5; xatol = 1e-6, xrtol = 0.0, ftol = 0.0)
    @test abs(x - π) <= 1e-6

    x, fx = DECAES.bisect_root(f, 2.5, 4.0; xatol = 0.0, xrtol = 1e-7, ftol = 0.0)
    @test abs(x - π) <= 1e-7 * π

    x, fx = DECAES.bisect_root(f, 2.0, 3.5; xatol = 0.0, xrtol = 0.0, ftol = 1e-8)
    @test abs(fx) <= 1e-8
end

@testset "brent_root" begin
    f = sin
    x, fx = DECAES.brent_root(f, 3.0, 4.5; xatol = 1e-6, xrtol = 0.0, ftol = 0.0)
    @test abs(x - π) <= 1e-6

    x, fx = DECAES.brent_root(f, 2.5, 4.0; xatol = 0.0, xrtol = 1e-7, ftol = 0.0)
    @test abs(x - π) <= 1e-7 * π

    x, fx = DECAES.brent_root(f, 2.0, 3.5; xatol = 0.0, xrtol = 0.0, ftol = 1e-8)
    @test abs(fx) <= 1e-8
end

@testset "newton_bisect_root" begin
    f_df = sincos
    x, fx = DECAES.newton_bisect_root(f_df, 3.7, 3.0, 4.5; xatol = 1e-6, xrtol = 0.0, ftol = 0.0)
    @test abs(x - π) <= 1e-6

    x, fx = DECAES.newton_bisect_root(f_df, 3.4, 2.5, 4.0; xatol = 0.0, xrtol = 1e-7, ftol = 0.0)
    @test abs(x - π) <= 1e-7 * π

    x, fx = DECAES.newton_bisect_root(f_df, 2.7, 2.0, 3.5; xatol = 0.0, xrtol = 0.0, ftol = 1e-8)
    @test abs(fx) <= 1e-8
end

@testset "brent_minimize" begin
    f = abs2 ∘ sin
    x, fx = DECAES.brent_minimize(f, 3.0, 4.5; xatol = 1e-6, xrtol = 0.0)
    @test abs(x - π) <= 1e-6

    x, fx = DECAES.brent_minimize(f, 2.5, 4.0; xatol = 0.0, xrtol = 1e-7)
    @test abs(x - π) <= 1e-7 * π
end

@testset "bracket_minimum" begin
    f = abs2 ∘ sin

    # A minimum already straddled by the first three points is returned without stepping
    @test DECAES.bracket_minimum(f, 3.1, 0.5) == (2.6, 3.6)

    # Otherwise the walk descends in whichever direction is downhill, from either side, and dilation only lengthens its reach
    for dilate in (1.0, 1.5, 2.0), a in (-1.0, 0.7, 6.0, 9.0)
        x₁, x₂ = DECAES.bracket_minimum(f, a, 0.1; dilate)
        x, _ = DECAES.brent_minimize(f, x₁, x₂; xatol = 1e-8, xrtol = 0.0)
        @test x₁ < x < x₂
        @test f(x) <= 1e-14
    end

    # A function that descends without turning has no bracket, and the walk gives up rather than running away
    @test all(isnan, DECAES.bracket_minimum(exp, 0.0, 1.0; dilate = 1.5, maxiters = 12))
    @test all(isnan, DECAES.bracket_minimum(x -> -atan(x), 0.0, 1.0; dilate = 2.0, maxiters = 20))

    # A minimum further out than `maxiters` steps of dilation will be missed
    g = x -> abs2(x - 1e6)
    @test all(isnan, DECAES.bracket_minimum(g, 0.0, 1.0; dilate = 1.5, maxiters = 12))
    @test !any(isnan, DECAES.bracket_minimum(g, 0.0, 1.0; dilate = 1.5, maxiters = 40))
end

@testset "brent_newton_minimize" begin
    f_df = sincos
    x, fx = DECAES.brent_newton_minimize(f_df, 4.0, 5.8, 5.1; xatol = 1e-6, xrtol = 0.0)
    @test abs(x - 3π / 2) <= 1e-6

    x, fx = DECAES.brent_newton_minimize(f_df, 4.2, 6.1, 5.0; xatol = 0.0, xrtol = 1e-7)
    @test abs(x - 3π / 2) <= 1e-7 * 3π / 2

    f_df = x -> (exp(x) - x / 2, exp(x) - 0.5)
    x, fx = DECAES.brent_newton_minimize(f_df, -2.0, 1.0, -0.1; xatol = 1e-8, xrtol = 0.0)
    @test abs(x - log(0.5)) <= 1e-8

    x, fx = DECAES.brent_newton_minimize(f_df, -2.2, 0.8, -0.3; xatol = 0.0, xrtol = 1e-9)
    @test abs(x - log(0.5)) <= 1e-9 * abs(log(0.5))
end

# Every solver returns `(x, y, bracket)`, and `bisect_root` also returns the bracket values. The bracket is in coordinate order, and its values must belong to those endpoints whichever order the caller supplied them in.
@testset "solver bracket API" begin
    f(x) = x^2 - 2
    for (a, b) in ((0.0, 2.0), (2.0, 0.0))
        x, y, (lo, hi), (ylo, yhi) = DECAES.bisect_root(f, a, b, f(a), f(b); xatol = 1e-8)
        @test lo <= hi && lo <= x <= hi
        @test ylo == f(lo) && yhi == f(hi)
        @test x ≈ √2 atol = 1e-6
    end

    # An unbracketed interval takes the early return, which must pair its endpoints the same way
    x, y, (lo, hi), (ylo, yhi) = DECAES.bisect_root(f, 3.0, 2.0, f(3.0), f(2.0); xatol = 1e-8)
    @test lo == 2.0 && hi == 3.0
    @test ylo == f(2.0) && yhi == f(3.0)

    for (a, b) in ((0.0, 2.0), (2.0, 0.0))
        x, y, (lo, hi) = DECAES.brent_root(f, a, b; xatol = 1e-10)
        @test lo <= x <= hi
        @test x ≈ √2 atol = 1e-8
    end

    # `brent_minimize` bounds its converged bracket by 4·xatol
    g(x) = (x - 0.3)^2
    x, y, (lo, hi) = DECAES.brent_minimize(g, -1.0, 1.0; xatol = 1e-6, xrtol = 0.0, maxiters = 100)
    @test lo <= x <= hi
    @test hi - lo <= 4e-6
    @test x ≈ 0.3 atol = 1e-5

    # The returned value belongs to the returned point, whether the solver stopped on the value, on the step size, or on the iteration cap
    f_df = sincos
    for maxiters in (1, 2, 4, 100), (x0, a, b) in ((3.7, 3.0, 4.5), (2.7, 2.0, 3.5))
        x, y = DECAES.newton_bisect_root(f_df, x0, a, b; xatol = 1e-6, xrtol = 0.0, ftol = 0.0, maxiters)
        @test y == f_df(x)[1]
    end

    for (a, b) in ((0.0, 2.0), (2.0, 0.0))
        x, y = DECAES.bisect_root(f, a, b, f(a), f(b); xatol = 1e-8)
        @test y == f(x)
        x, y = DECAES.brent_root(f, a, b; xatol = 1e-10)
        @test y == f(x)
    end

    x, y = DECAES.brent_minimize(g, -1.0, 1.0; xatol = 1e-6, xrtol = 0.0, maxiters = 100)
    @test y == g(x)
end
