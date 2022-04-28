function verify_solution(m, n)
    A = randn(m, n)
    b = randn(m)
    work = NNLSWorkspace(A, b)
    nnls!(work)

    GC.@preserve work begin
        x  = NNLS.solution(work)
        w  = NNLS.dual(work)
        i₊ = NNLS.components(work)
        m₊ = NNLS.ncomponents(work)
        m₀ = m - m₊
        i₀ = setdiff(1:length(x), i₊)
        x₊, x₀ = x[i₊], x[i₀]
        w₀, w₋ = w[i₊], w[i₀]
        A₊, A₀ = A[:,i₊], A[:,i₀]
        L  = NNLS.choleskyfactor(work, Val(:L))
        U  = NNLS.choleskyfactor(work, Val(:U))

        # Internals
        @test isperm(work.idx)

        # Solution
        @test all(>(0), x₊)
        @test all(==(0), x₀)
        @test x₊ ≈ (A₊'A₊) \ (A₊'b)
        @test x₊ ≈ A₊ \ b

        # Dual (i.e. gradient)
        if m₊ < m
            # Less components than rows; in general solution is not exact and gradient has negative components
            @test all(<(0), w₋)
            @test all(==(0), w₀)
            @test w₋ ≈ A₀'b - A₀'A₊ * x₊
        else
            # At least as many components as rows; solution is exact, gradient is zero
            @test all(==(0), w)
        end

        # Cholesky factors
        @test A₊'A₊ ≈ L'L
        @test A₊'A₊ ≈ U*U'
    end
    work
end

@testset "NNLS" begin
    for m in [1,2,5,8,16,25,32,50], n in [1,2,5,8,16,25,32,50]
        verify_solution(m, n)
    end
end

function build_lcurve_corner_cached_fun(::Type{T} = Float64) where {T}
    f = CachedFunction(logμ -> SA[exp(logμ), exp(-logμ)], GrowableCache{T, SVector{2,T}}())
    f = LCurveCornerCachedFunction(f, GrowableCache{T, LCurveCornerPoint{T}}(), GrowableCache{T, LCurveCornerState{T}}())
    return f
end

function run_lcurve_corner(f)
    return lcurve_corner(f, log(0.1), log(10.0); xtol = 1e-6, Ptol = 1e-6, Ctol = 0)
end

@testset "lsqnonneg_lcurve" begin
    # Test allocations
    f = build_lcurve_corner_cached_fun()
    empty!(f)
    @test @allocated(run_lcurve_corner(f)) > 0 # caches will be populated with first call
    empty!(f)
    @test @allocated(run_lcurve_corner(f)) == 0 # caches should be reused with second call

    # Maximum curvature point for the graph (x(μ), y(μ)) = (μ, 1/μ) occurs at μ=1, i.e. logμ=0
    @test run_lcurve_corner(f) ≈ 0 atol = 1e-3
end
