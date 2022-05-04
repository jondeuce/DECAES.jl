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

function mock_nnls_data(m, n)
    opts = DECAES.mock_t2map_opts(; MatrixSize = (1, 1, 1), nTE = m, nT2 = n, SetFlipAngle = 180.0, RefConAngle = 180.0)
    prob = DECAES.mock_surrogate_search_problem(Val(1), Val(m), opts)
    (; A = prob.As[:,:,end], b = prob.b)
end

∇finitediff(f, x, dx = 1e-4) = (f(x + dx) - f(x - dx)) ./ (2 * dx)

@testset "Least Squares Gradients" begin
    for (m, n) in [(4,5), (5,4)]
        A, b = rand(m, n), rand(m)
        C_μ = μ -> [A; μ * LinearAlgebra.I]
        B_μ = μ -> (C = C_μ(μ); LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(C'C)))
        x_μ = μ -> C_μ(μ) \ [b; zeros(n)]

        μ = 0.1
        B = B_μ(μ)
        x = x_μ(μ)

        # Derivative of solution x w.r.t. μ
        f = _μ -> x_μ(_μ)
        dx_dμ = -2*μ*(B\(B\(A'b)))
        @test dx_dμ ≈ ∇finitediff(f, μ) rtol = 1e-4

        # Derivative of solution l2-norm ||x||^2 w.r.t. μ
        f = _μ -> sum(abs2, x_μ(_μ))
        dx²_dμ = 2 * x'dx_dμ
        @test dx²_dμ ≈ (-4*μ)*b'*(A*(B\(B\(B\(A'b))))) rtol = 1e-4
        @test dx²_dμ ≈ ∇finitediff(f, μ) rtol = 1e-4

        # Derivative of A*x (or equivalently, A*x-b) w.r.t. μ
        f = _μ -> A*x_μ(_μ)
        dAx_dμ = A*dx_dμ
        @test dAx_dμ ≈ -2*μ*(A*(B\(B\(A'b)))) rtol = 1e-4
        @test dAx_dμ ≈ ∇finitediff(f, μ) rtol = 1e-4

        # Derivative of residual l2-norm ||A*x-b||^2 w.r.t. μ
        f = _μ -> sum(abs2, A*x_μ(_μ) - b)
        dAxb_dμ = 2*(A*x-b)'dAx_dμ
        @test dAxb_dμ ≈ -4*μ*((A*x-b)'*(A*(B\(B\(A'b))))) rtol = 1e-4
        @test dAxb_dμ ≈ ∇finitediff(f, μ) rtol = 1e-4
    end
end

@testset "NNLSTikhonovRegProblem" begin
    for (m, n) in [(4,5), (5,4)]
        A, b = mock_nnls_data(m, n)
        work = NNLSTikhonovRegProblem(A, b)

        f = μ -> (DECAES.solve!(work, μ); DECAES.chi2(work))
        μ = 0.1
        DECAES.solve!(work, μ)
        ∇μ = DECAES.∇chi2(work)
        @test ∇μ ≈ ∇finitediff(f, μ) rtol = 1e-4

        @test @allocated(DECAES.solve!(work, μ)) == 0
        @test_broken @allocated(DECAES.∇chi2(work)) == 0
    end
end
