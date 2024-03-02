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

@testset "brents_method" begin
    f = abs2 ∘ sin
    x, fx = DECAES.brents_method(f, 3.0, 4.5; xatol = 1e-6, xrtol = 0.0)
    @test abs(x - π) <= 1e-6

    x, fx = DECAES.brents_method(f, 2.5, 4.0; xatol = 0.0, xrtol = 1e-7)
    @test abs(x - π) <= 1e-7 * π
end
