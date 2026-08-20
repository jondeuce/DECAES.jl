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
end
