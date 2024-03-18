const NNLS_SIZES = vec(collect(Iterators.product([1, 2, 5, 8, 16, 32], [1, 2, 5, 8, 16, 32])))

function rand_NNLS_data(m, n)
    # A strictly positive, unconstrained x has negative entries
    x = rand(MersenneTwister(0), n) .* ifelse.(isodd.(1:n), -1, 1)
    A = rand(MersenneTwister(0), m, n)
    b = A * x
    return A, b
end

function verify_NNLS(m, n)
    A, b = rand_NNLS_data(m, n)
    work = NNLS.NNLSWorkspace(A, b)
    NNLS.nnls!(work)

    GC.@preserve work begin
        x  = NNLS.solution(work)
        w  = NNLS.dual(work)
        n₊ = NNLS.ncomponents(work)
        U  = NNLS.choleskyfactor(work, Val(:U))
        L  = NNLS.choleskyfactor(work, Val(:L))

        # Solution partitioning
        i₊ = work.idx[1:n₊]
        i₀ = work.idx[n₊+1:end]
        x₊, x₀ = x[i₊], x[i₀]
        w₀, w₋ = w[i₊], w[i₀]
        A₊, A₀ = A[:, i₊], A[:, i₀]

        @test isperm(work.idx)
        @test NNLS.components(work) == i₊
        @test setdiff(1:n, NNLS.components(work)) == sort(i₀)

        # Solution
        @test all(>(0), x₊)
        @test all(==(0), x₀)
        @test x₊ ≈ (A₊'A₊) \ (A₊'b)
        @test x₊ ≈ A₊ \ b

        # Dual (i.e. gradient)
        if NNLS.residualnorm(work) > 0
            # Solution is not exact and gradient has negative components
            @test all(<(0), w₋)
            @test all(==(0), w₀)
            @test w₋ ≈ -A₀' * (A₊ * x₊ - b) rtol = 1e-12 atol = 1e-8
            @test w ≈ -A' * (A * x - b) rtol = 1e-12 atol = 1e-8
        else
            # Solution is exact, gradient is zero
            @test all(==(0), w)
        end

        # KKT conditions
        @test all(>=(0), x) # primal feasibility
        @test all(<=(0), w) # dual feasibility
        @test all(==(0), x .* w) # complementary slackness

        # Internals
        @test U == work.A[1:n₊, i₊]
        @test L == work.A[1:n₊, i₊]'
        @test U * x₊ ≈ work.b[1:n₊]

        @test work.zz[1:n₊] == x₊
        @test work.A[n₊+1:end, i₀]' * work.zz[n₊+1:end] ≈ w₋
        @test work.zz[n₊+1:end] == work.b[n₊+1:end]
        @test NNLS.residualnorm(work) ≈ norm(A * x - b) rtol = 1e-12 atol = 1e-12

        if n₊ > 0
            b₊ = rand(n₊)
            U⁻¹b₊ = copy(b₊)
            NNLS.solve_triangular_system!(U⁻¹b₊, U, 1:n₊, n₊, Val(false))
            @test U⁻¹b₊ ≈ U \ b₊

            L⁻¹b₊ = copy(b₊)
            NNLS.solve_triangular_system!(L⁻¹b₊, U, 1:n₊, n₊, Val(true))
            @test L⁻¹b₊ ≈ L \ b₊

            U⁻¹b₊ = copy(b₊)
            NNLS.solve_triangular_system!(U⁻¹b₊, work.A, work.idx, n₊, Val(false))
            @test U⁻¹b₊ ≈ U \ b₊

            L⁻¹b₊ = copy(b₊)
            NNLS.solve_triangular_system!(L⁻¹b₊, work.A, work.idx, n₊, Val(true))
            @test L⁻¹b₊ ≈ L \ b₊
        end

        # Cholesky factors
        @test A₊'A₊ ≈ U'U
        @test A₊'A₊ ≈ L * L'

        F = cholesky!(NNLS.NormalEquation(), work)
        if n₊ > 0
            x′ = rand(MersenneTwister(0), n₊)
            x′′ = copy(x′)
            b′ = rand(MersenneTwister(0), n₊)
            b′′ = copy(b′)
            ldiv!(x′, F, b′)
            @test x′ ≈ (A₊'A₊) \ b′
            @test b′ ≈ b′′ && !(x′ ≈ x′′)
            @test @allocated(ldiv!(x′, cholesky!(NNLS.NormalEquation(), work), b′)) == 0
        end
    end # GC.@preserve

    return work
end

@testset "NNLS" begin
    for (m, n) in NNLS_SIZES
        verify_NNLS(m, n)
    end
end

function build_lcurve_corner_cached_fun(::Type{T} = Float64) where {T}
    # Mock lcurve function with (ξ(μ), η(μ)) = (μ, 1/μ)
    f = CachedFunction(logμ -> SA[exp(logμ), exp(-logμ)], GrowableCache{T, SVector{2, T}}())
    f = LCurveCornerCachedFunction(f, GrowableCache{T, LCurveCornerPoint{T}}(), GrowableCache{T, LCurveCornerState{T}}())
    return f
end

function run_lcurve_corner(f)
    return lcurve_corner(f, log(0.1), log(10.0); xtol = 1e-6, Ptol = 1e-6, Ctol = 0)
end

@testset "lcurve_corner" begin
    # Test allocations
    f = build_lcurve_corner_cached_fun()
    empty!(f)
    @test @allocated(run_lcurve_corner(f)) > 0 # caches will be populated with first call
    empty!(f)
    @test @allocated(run_lcurve_corner(f)) == 0 # caches should be reused with second call

    # Maximum curvature point for the graph (x(μ), y(μ)) = (μ, 1/μ) occurs at μ=1, i.e. logμ=0
    @test run_lcurve_corner(f) ≈ 0 atol = 1e-3
end

@testset "lsqnonneg_lcurve" begin
    for (m, n) in NNLS_SIZES
        A, b = rand_NNLS_data(m, n)
        work = DECAES.lsqnonneg_lcurve_work(A, b)
        @test @allocated(DECAES.lsqnonneg_lcurve!(work)) == 0 # caches should be initialized to be sufficiently large that normally they don't need to grow
    end
end

@testset "lsqnonneg_chi2" begin
    for (m, n) in NNLS_SIZES
        A, b = rand_NNLS_data(m, n)
        work = DECAES.lsqnonneg_chi2_work(A, b)

        # Test solver
        x_unreg = DECAES.solve!(work.nnls_prob)
        chi2_min = DECAES.chi2(work.nnls_prob)
        chi2_max = sum(abs2, b) # lim_{μ -> ∞} ||A*x(μ) - b||² = ||b||², since lim_{μ -> ∞} x(μ) = 0
        chi2_target = √(chi2_min * chi2_max)
        chi2factor_target = chi2_target / chi2_min #TODO: add some randomness?

        for (method, atol) in [
            :bisect => 0.01 * (chi2factor_target - 1),
            :brent => 0.005 * (chi2factor_target - 1),
        ]
            (; x, mu, chi2factor) = DECAES.lsqnonneg_chi2!(work, chi2factor_target; method)

            if chi2_min <= 0
                # Unregularized solution should be returned
                @test x === x_unreg
                @test isfinite(mu)
                @test mu == 0
                @test chi2factor == 1
            elseif sum(abs2, x_unreg) == 0
                # Unreg solution is x = 0, and regularization can only reduce ||x||.
                # Since ||x|| = 0 is already minimized, μ is undefined; we enforce that μ = 0 is returned which is consistent with chi2factor = 1.
                @test x == x_unreg
                @test isfinite(mu)
                @test mu == 0 # any value of μ should result in x == x_unreg and chi2factor == 1
                @test chi2factor == 1
            else
                @test mu > 0
                @test chi2factor ≈ chi2factor_target atol = atol
            end

            @test @allocated(DECAES.lsqnonneg_chi2!(work, chi2factor_target; method)) == 0 # caches should be initialized to be sufficiently large that normally they don't need to grow
        end
    end
end

@testset "lsqnonneg_gcv" begin
    for (m, n) in NNLS_SIZES
        A, b = rand_NNLS_data(m, n)
        work = DECAES.NNLSGCVRegProblem(A, b)

        # Test loss function
        logμ = randn()
        gcv = DECAES.gcv!(work, logμ)
        χ² = DECAES.chi2(DECAES.get_cache(work.nnls_prob_smooth_cache))
        @test gcv ≈ χ² / DECAES.gcv_tr(A, exp(logμ))^2
        @test gcv ≈ χ² / DECAES.gcv_tr_brute(A, exp(logμ))^2
        @test @allocated(DECAES.gcv!(work, logμ)) == 0

        # Test solver
        @test @allocated(DECAES.lsqnonneg_gcv!(work)) == 0 # caches should be initialized to be sufficiently large that normally they don't need to grow
    end
end

function mock_nnls_data(m, n)
    opts = DECAES.mock_t2map_opts(; MatrixSize = (1, 1, 1), nTE = m, nT2 = n)
    prob = DECAES.mock_surrogate_search_problem(Val(1), Val(m), opts)
    return (; A = prob.As[:, :, end], b = prob.b)
end

∇finitediff(f, x, h = √eps(one(x))) = (f(x .+ h) .- f(x .- h)) ./ 2h
∇²finitediff(f, x, h = ∛eps(one(x))) = (f(x .+ h) .- 2 .* f(x) .+ f(x .- h)) ./ h^2

∇logfinitediff(f, logx, h = √eps(one(logx))) = ∇finitediff(f, logx, h) ./ exp(logx)
∇²logfinitediff(f, logx, h = ∛eps(one(logx))) = (∇²finitediff(f, logx, h) - ∇finitediff(f, logx, h)) ./ exp(2 * logx)

@testset "Least Squares Gradients" begin
    #=
    Gradient notes:

    d/dμ x(μ):
      ∇x = -2 * μ * (B\(B\(A'b)))
         = -2 * μ * (B\x)             <-- x = (A'A)\(A'b) = B\(A'b)

    d/dμ ||A*x(μ)-b||^2:
      ∇μ = 2 * ((A*x-b)' * (A*∇x))
         = 2 * (A'*(A*x-b))' * ∇x
         = 2 * ((-μ^2*x)' * ∇x)       <-- A'*(A*x-b) = -μ^2*x, as 0 = w = [A; μI]' * ([A; μI]*x - [b;0]) = [A; μI]' * [A*x-b; μ*x] = A'*(A*x-b) + μ^2*x
         = 4μ^3 * x' * (B\x)

    d/dμ ||x(μ)||^2:
      ∇μ = -4μ * b' * (A*(B\(B\(B\(A'b)))))
         = -4μ * b' * (A*(B\(B\x)))   <-- x = (A'A)\(A'b) = B\(A'b)
         = -4μ * (B\(A'b))'* (B\x)    <-- B = B'
         = -4μ * x' * (B\x)
    =#

    for (m, n) in NNLS_SIZES
        A, b = rand_NNLS_data(m, n)
        A_μ = μ -> [A; μ * LinearAlgebra.I]
        B_μ = μ -> LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(A'A + μ^2 * I))
        x_μ = μ -> A_μ(μ) \ [b; zeros(n)]

        μ = 0.99
        B = B_μ(μ)
        x = x_μ(μ)

        # Derivative of solution x w.r.t. μ
        f = _μ -> x_μ(_μ)
        dx_dμ = -2 * μ * (B \ x)
        @test dx_dμ ≈ -2 * μ * (B \ (B \ (A'b)))
        @test dx_dμ ≈ ∇finitediff(f, μ) rtol = 1e-4

        # Derivative of A*x (or equivalently, A*x-b) w.r.t. μ
        f = _μ -> A * x_μ(_μ)
        dAx_dμ = A * dx_dμ
        @test dAx_dμ ≈ -2 * μ * (A * (B \ (B \ (A'b))))
        @test dAx_dμ ≈ ∇finitediff(f, μ) rtol = 1e-4

        # Derivative of solution l2-norm ||x||^2 w.r.t. μ
        f = _μ -> sum(abs2, x_μ(_μ))
        dx²_dμ = 2 * x'dx_dμ
        @test dx²_dμ ≈ (-4 * μ) * b' * (A * (B \ (B \ (B \ (A'b)))))
        @test dx²_dμ ≈ ∇finitediff(f, μ) rtol = 1e-4

        # Derivative of residual l2-norm ||A*x-b||^2 w.r.t. μ
        f = _μ -> sum(abs2, A * x_μ(_μ) - b)
        dAxb_dμ = -2μ^2 * x'dx_dμ
        @test dAxb_dμ ≈ 2 * (A * x - b)'dAx_dμ
        @test dAxb_dμ ≈ -4 * μ * ((A * x - b)' * (A * (B \ (B \ (A'b)))))
        @test dAxb_dμ ≈ ∇finitediff(f, μ) rtol = 1e-4
    end
end

function verify_NNLSTikhonovRegProblem(m, n)
    T = Double64 # need higher precision for finite differences
    A, b = rand_NNLS_data(m, n)
    A, b = T.(A), T.(b)
    work = NNLSTikhonovRegProblem(A, b)
    withsolve(f, μ) = (DECAES.solve!(work, μ); return f(work))

    GC.@preserve work for (i, μ) in enumerate([0.01, 0.05, 0.25, 0.99])
        @test xor(i > 1, isnan(DECAES.mu(work))) # μ should be initialized to NaN
        @test DECAES.mu!(work, μ) == μ
        @test DECAES.mu(work) == μ

        x = DECAES.solve!(work, μ)
        @test all(>=(0), x)
        @test withsolve(DECAES.reg, μ) ≈ μ^2 * sum(abs2, x)
        @test withsolve(DECAES.chi2, μ) ≈ sum(abs2, A * x - b)
        @test withsolve(DECAES.resnorm_sq, μ) ≈ sum(abs2, A * x - b)
        @test withsolve(DECAES.seminorm_sq, μ) ≈ sum(abs2, x)

        @test @allocated(DECAES.solve!(work, μ)) == 0
        @test @allocated(DECAES.mu(work)) == 0

        ∇μ = DECAES.∇reg(work)
        @test ∇μ ≈ ∇logfinitediff(logμ -> withsolve(DECAES.reg, exp(logμ)), log(μ), T(1e-6)) rtol = 1e-4
        @test @allocated(DECAES.∇reg(work)) == 0
        @test @inferred(DECAES.∇reg(work)) isa T

        ∇μ = DECAES.∇chi2(work)
        @test ∇μ ≈ ∇logfinitediff(logμ -> withsolve(DECAES.chi2, exp(logμ)), log(μ), T(1e-6)) rtol = 1e-4
        @test @allocated(DECAES.∇chi2(work)) == 0
        @test @inferred(DECAES.∇chi2(work)) isa T

        ∇μ = DECAES.∇resnorm_sq(work)
        @test ∇μ ≈ ∇logfinitediff(logμ -> withsolve(DECAES.resnorm_sq, exp(logμ)), log(μ), T(1e-6)) rtol = 1e-4
        @test @allocated(DECAES.∇resnorm_sq(work)) == 0
        @test @inferred(DECAES.∇resnorm_sq(work)) isa T

        ∇μ = DECAES.∇seminorm_sq(work)
        @test ∇μ ≈ ∇logfinitediff(logμ -> withsolve(DECAES.seminorm_sq, exp(logμ)), log(μ), T(1e-6)) rtol = 1e-4
        @test @allocated(DECAES.∇seminorm_sq(work)) == 0
        @test @inferred(DECAES.∇seminorm_sq(work)) isa T

        ∇μ = DECAES.solution_gradnorm(work)
        @test ∇μ ≈ norm(∇logfinitediff(logμ -> withsolve(copy ∘ DECAES.solution, exp(logμ)), log(μ), T(1e-6))) rtol = 1e-4
        @test @allocated(DECAES.solution_gradnorm(work)) == 0
        @test @inferred(DECAES.solution_gradnorm(work)) isa T

        ∇²μ = DECAES.∇²resnorm_sq(work)
        @test ∇²μ ≈ ∇²logfinitediff(logμ -> withsolve(DECAES.resnorm_sq, exp(logμ)), log(μ), T(1e-6)) rtol = 1e-3 atol = 1e-4
        @test @allocated(DECAES.∇²resnorm_sq(work)) == 0
        @test @inferred(DECAES.∇²resnorm_sq(work)) isa T

        ∇²μ = DECAES.∇²seminorm_sq(work)
        @test ∇²μ ≈ ∇²logfinitediff(logμ -> withsolve(DECAES.seminorm_sq, exp(logμ)), log(μ), T(1e-6)) rtol = 1e-3 atol = 1e-4
        @test @allocated(DECAES.∇²seminorm_sq(work)) == 0
        @test @inferred(DECAES.∇²seminorm_sq(work)) isa T

        # Curvature computation
        DECAES.solve!(work, μ)
        ξ = DECAES.resnorm_sq(work)
        η = DECAES.seminorm_sq(work)
        if η > 0
            ξ_fun = logμ -> (DECAES.solve!(work, exp(logμ)); DECAES.resnorm_sq(work))
            η_fun = logμ -> (DECAES.solve!(work, exp(logμ)); DECAES.seminorm_sq(work))
            C_fun = logμ -> (DECAES.solve!(work, exp(logμ)); DECAES.curvature(identity, work))
            C_menger = DECAES.menger(ξ_fun, η_fun; h = T(1e-4))

            ξ′ = DECAES.∇resnorm_sq(work)
            η′ = DECAES.∇seminorm_sq(work)
            ξ′′ = DECAES.∇²resnorm_sq(work)
            η′′ = DECAES.∇²seminorm_sq(work)
            C = (ξ′ * η′′ - η′ * ξ′′) / √((ξ′^2 + η′^2)^3)

            @test C_fun(log(μ)) ≈ C
            @test C_fun(log(μ)) ≈ C_menger(log(μ)) rtol = 1e-4

            C̄_fun = logμ -> (DECAES.solve!(work, exp(logμ)); DECAES.curvature(log, work))
            C̄_menger = DECAES.menger(log ∘ ξ_fun, log ∘ η_fun; h = T(1e-4))

            _ξ′ = ξ′ / ξ # d/dlogμ ξ(μ) = ξ'(μ) / ξ(μ)
            _η′ = η′ / η
            _ξ′′ = ξ′′ / ξ - _ξ′^2 # d²/d(logμ)² ξ(μ) = ξ''(μ) / ξ(μ) - (ξ'(μ) / ξ(μ))^2
            _η′′ = η′′ / η - _η′^2
            C̄ = (_ξ′ * _η′′ - _η′ * _ξ′′) / √((_ξ′^2 + _η′^2)^3)

            @test C̄_fun(log(μ)) ≈ C̄
            @test C̄_fun(log(μ)) ≈ C̄_menger(log(μ)) rtol = 1e-4
        end
    end # GC.@preserve
end

@testset "NNLSTikhonovRegProblem" begin
    for (m, n) in NNLS_SIZES
        verify_NNLSTikhonovRegProblem(m, n)
    end
end
