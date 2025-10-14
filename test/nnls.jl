const NNLS_SIZES = vec(collect(Iterators.product([1, 2, 5, 8, 13, 16, 25, 32], [1, 2, 5, 8, 13, 16, 25, 32])))

function rand_NNLS_data(m, n)
    x = rand(n)
    x[rand(1:2):2:end] .*= -1 # flip even (or odd) signs
    A = rand(m, n) # A strictly positive
    b = A * x # b corresponds to unconstrained x with negative entries
    return A, b
end

function maybe_pad_NNLS_data(A0, b0, μ)
    n = size(A0, 2)
    if μ > 0
        # Tikhonov-padded problem: min_{x ≥ 0} ||Ax-b||² + μ²||x||²
        A = [A0; μ * LinearAlgebra.I(n)]
        b = [b0; zeros(n)]
        return A, b
    else
        # Unregularized problem: min_{x ≥ 0} ||Ax-b||²
        return copy(A0), copy(b0)
    end
end

∇finitediff(f, x, h = √eps(one(x))) = (f(x .+ h) .- f(x .- h)) ./ 2h
∇²finitediff(f, x, h = ∛eps(one(x))) = (f(x .+ h) .- 2 .* f(x) .+ f(x .- h)) ./ h^2

∇logfinitediff(f, logx, h = √eps(one(logx))) = ∇finitediff(f, logx, h) ./ exp(logx)
∇²logfinitediff(f, logx, h = ∛eps(one(logx))) = (∇²finitediff(f, logx, h) - ∇finitediff(f, logx, h)) ./ exp(2 * logx)

function verify_NNLS(m₀, n, μ = 0.0)
    @assert μ >= 0 "μ must be non-negative"
    D64 = Double64
    m = μ > 0 ? m₀ + n : m₀
    A0, b0 = rand_NNLS_data(m₀, n)
    A, b = maybe_pad_NNLS_data(A0, b0, μ)
    work = NNLS.NNLSWorkspace(A, b)

    NNLS.load!(work, A, b)
    @test work.A == A
    @test work.b == b

    # Fill workspace with junk values
    work.A .= randn(m, n)
    work.b .= randn(m)
    work.x .= randn(n)
    work.w .= randn(n)
    work.zz .= randn(m)
    work.idx .= rand(Int, n)
    work.diag .= rand(Bool, n)
    work.rnorm[] = rand()
    work.mode[] = rand(1:100)
    work.nsetp[] = rand(0:min(m, n))

    @inferred NNLS.nnls!(work, A, b)
    @test work.mode[] == 0 # success

    GC.@preserve work begin
        x = NNLS.solution(work)
        w = NNLS.dual(work)
        n₊ = NNLS.ncomponents(work)
        U = NNLS.choleskyfactor(work, Val(:U))
        L = NNLS.choleskyfactor(work, Val(:L))

        # Solution partitioning
        idx = work.idx
        invidx = work.invidx
        @test isperm(idx)
        @test isperm(invidx)
        @test invperm(idx) == invidx

        i₊ = idx[1:n₊]
        i₀ = idx[n₊+1:end]
        x₊, x₀ = x[i₊], x[i₀]
        w₀, w₋ = w[i₊], w[i₀]
        A₊, A₀ = A[:, i₊], A[:, i₀]

        @test NNLS.components(work) == i₊
        @test setdiff(1:n, NNLS.components(work)) == sort(i₀)

        # Solution
        @test all(>(0), x₊)
        @test all(==(0), x₀)
        # @test x₊ ≈ (A₊' * A₊) \ (A₊' * b)
        @test x₊ ≈ (D64.(A₊)' * D64.(A₊)) \ (D64.(A₊)' * D64.(b))
        @test x₊ ≈ A₊ \ b

        # Dual (i.e. gradient)
        maxn₊ = min(m, n)
        if n₊ < maxn₊
            # Solution is not full rank and gradient has negative components
            @test NNLS.residualnorm(work) > 10 * eps()

            @test count(<(0), w) == n - n₊
            @test count(==(0), w) == n₊
            @test all(<(0), w₋)
            @test all(==(0), w₀)

            @test w₋ ≈ -A₀' * (A₊ * x₊ - b) rtol = 1e-8 atol = 1e-12 * norm(A'b)
            @test w ≈ -A' * (A * x - b) rtol = 1e-8 atol = 1e-12 * norm(A'b)

            # Gradient of positive components is A0₊'(A0₊ * x₊ - b0) + μ^2 * x₊ = 0
            @test A0[:, i₊]' * (A0[:, i₊] * x₊ - b0) ≈ -μ^2 * x₊ rtol = 1e-8 atol = 1e-12 * norm(A0'b0)
        else
            # Solution is full rank, gradient is zero
            @test all(==(0), w)
            if μ == 0
                # Should be exactly zero since b ∈ range(A) by construction, but allow for floating point error.
                @test NNLS.residualnorm(work) <= 10 * eps()
            else
                # Should be strictly positive, since b ∉ range(A) in general due to zero padding
                @test NNLS.residualnorm(work) >= 10 * eps()
            end
        end

        # KKT conditions
        @test all(>=(0), x) # primal feasibility
        @test all(<=(0), w) # dual feasibility
        @test all(==(0), x .* w) # complementary slackness

        # Internals
        @test U == work.A[1:n₊, 1:n₊]
        @test L == work.A[1:n₊, 1:n₊]'
        @test U * x₊ ≈ work.b[1:n₊]

        @test work.zz[1:n₊] == x₊
        # @test work.A[n₊+1:end, i₀]' * work.zz[n₊+1:end] ≈ w₋ #TODO?
        @test work.zz[n₊+1:end] == work.b[n₊+1:end]
        @test NNLS.residualnorm(work) ≈ norm(A * x - b) rtol = 1e-12 atol = 1e-12

        if n₊ > 0
            b₊ = rand(n₊)
            U⁻¹b₊ = copy(b₊)
            @inferred NNLS.solve_triangular_system!(U⁻¹b₊, U, n₊, Val(false))
            @test U⁻¹b₊ ≈ U \ b₊

            L⁻¹b₊ = copy(b₊)
            @inferred NNLS.solve_triangular_system!(L⁻¹b₊, U, n₊, Val(true))
            @test L⁻¹b₊ ≈ L \ b₊

            U⁻¹b₊ = copy(b₊)
            @inferred NNLS.solve_triangular_system!(U⁻¹b₊, work.A, n₊, Val(false))
            @test U⁻¹b₊ ≈ U \ b₊

            L⁻¹b₊ = copy(b₊)
            @inferred NNLS.solve_triangular_system!(L⁻¹b₊, work.A, n₊, Val(true))
            @test L⁻¹b₊ ≈ L \ b₊
        end

        # Cholesky factors
        @test A₊' * A₊ ≈ U' * U
        @test A₊' * A₊ ≈ L * L'

        F = cholesky!(NNLS.NormalEquation(), work)
        if n₊ > 0
            x′, b′ = rand(n₊), rand(n₊)
            x′′, b′′ = copy(x′), copy(b′)
            ldiv!(x′, F, b′)
            @test b′ ≈ b′′
            @test !(x′ ≈ x′′)
            @test x′ == F \ b′
            @test x′ ≈ (D64.(A₊)' * D64.(A₊)) \ D64.(b′)
            @inferred ldiv!(x′, cholesky!(NNLS.NormalEquation(), work), b′)
        end

        # QR factors
        #   Note: qr(A) = QR relates to cholesky(A'A) = LL' = U'U via:
        #   A'A = U'U = (QR)'(QR) = R'R => R = U (up to row sign)
        posdiag(R) = UpperTriangular(Diagonal(sign.(diag(R))) * R) # ensure diagonal is positive
        R = qr(A₊).R
        @test posdiag(R) ≈ posdiag(U)
    end # GC.@preserve

    return work
end

function verify_NNLS_tikh(m, n, μ)
    D64 = Double64
    A0, b0 = rand_NNLS_data(m, n)
    A = [A0; μ * LinearAlgebra.I(n)]
    b = [b0; zeros(n)]
    work = NNLS.NNLSWorkspace(A, b)

    NNLS.load!(work, A, b)
    @test work.A == A
    @test work.b == b

    for mode in [:direct]#, :shuffle]
        # Fill workspace with junk values
        work.A .= randn(m + n, n)
        work.b .= randn(m + n)
        work.x .= randn(n)
        work.w .= randn(n)
        work.zz .= randn(m + n)
        work.idx .= rand(Int, n)
        work.diag .= rand(Bool, n)
        work.rnorm[] = rand()
        work.mode[] = rand(1:100)
        work.nsetp[] = rand(0:min(m + n, n))

        if mode === :direct
            @inferred NNLS.nnls!(work, A, b, μ)
        elseif mode === :shuffle
            idx′, nsetp′ = Random.randperm(n), rand(0:min(m, n))
            @inferred NNLS.nnls!(work, A, b, μ, idx′, nsetp′)
        else
            error("Invalid mode :$mode")
        end
        @test work.mode[] == 0 # success

        GC.@preserve work begin
            x = NNLS.solution(work)
            w = NNLS.dual(work)
            n₊ = NNLS.ncomponents(work)
            U = NNLS.choleskyfactor(work, Val(:U))
            L = NNLS.choleskyfactor(work, Val(:L))

            # Solution partitioning
            idx = work.idx
            invidx = work.invidx
            @test isperm(idx)
            @test isperm(invidx)
            @test invperm(idx) == invidx

            i₊ = idx[1:n₊]
            i₀ = idx[n₊+1:end]
            x₊, x₀ = x[i₊], x[i₀]
            w₀, w₋ = w[i₊], w[i₀]
            A₊, A₀ = A[:, i₊], A[:, i₀]

            @test NNLS.components(work) == i₊
            @test setdiff(1:n, NNLS.components(work)) == sort(i₀)

            # Solution
            @test all(>(0), x₊)
            @test all(==(0), x₀)
            # @test x₊ ≈ (A₊' * A₊) \ (A₊' * b)
            @test x₊ ≈ (D64.(A₊)' * D64.(A₊)) \ (D64.(A₊)' * D64.(b))
            @test x₊ ≈ A₊ \ b

            # Dual (i.e. gradient)
            maxn₊ = μ == 0 ? min(m, n) : n
            if n₊ < maxn₊
                # Solution is not full rank and gradient has negative components
                @test NNLS.residualnorm(work) > 0

                @test count(<(0), w) == n - n₊
                @test count(==(0), w) == n₊
                @test all(<(0), w₋)
                @test all(==(0), w₀)

                @test w₋ ≈ -A₀' * (A₊ * x₊ - b) rtol = 1e-8 atol = 1e-12 * norm(A'b)
                @test w ≈ -A' * (A * x - b) rtol = 1e-8 atol = 1e-12 * norm(A'b)

                # Gradient of positive components is A0₊'(A0₊ * x₊ - b0) + μ^2 * x₊ = 0
                @test A0[:, i₊]' * (A0[:, i₊] * x₊ - b0) ≈ -μ^2 * x₊ rtol = 1e-8 atol = 1e-12 * norm(A0'b0)
            else
                # Solution is full rank, gradient is zero
                @test all(==(0), w)
                if μ == 0
                    # Should be exactly zero since b ∈ range(A) by construction, but allow for floating point error.
                    @test NNLS.residualnorm(work) <= 10 * eps()
                else
                    # Should be strictly positive, since b ∉ range(A) in general due to zero padding
                    @test NNLS.residualnorm(work) >= 10 * eps()
                end
            end

            # KKT conditions
            @test all(>=(0), x) # primal feasibility
            @test all(<=(0), w) # dual feasibility
            @test all(==(0), x .* w) # complementary slackness

            # Internals
            @test U == work.A[1:n₊, 1:n₊]
            @test L == work.A[1:n₊, 1:n₊]'
            @test U * x₊ ≈ work.b[1:n₊]

            @test work.zz[1:n₊] == x₊
            # @test work.A[n₊+1:end, i₀]' * work.zz[n₊+1:end] ≈ w₋ #TODO?
            @test work.zz[n₊+1:end] == work.b[n₊+1:end]
            @test NNLS.residualnorm(work) ≈ norm(A * x - b) rtol = 1e-12 atol = 1e-12

            if n₊ > 0
                b₊ = rand(n₊)
                U⁻¹b₊ = copy(b₊)
                @inferred NNLS.solve_triangular_system!(U⁻¹b₊, U, n₊, Val(false))
                @test U⁻¹b₊ ≈ U \ b₊

                L⁻¹b₊ = copy(b₊)
                @inferred NNLS.solve_triangular_system!(L⁻¹b₊, U, n₊, Val(true))
                @test L⁻¹b₊ ≈ L \ b₊

                U⁻¹b₊ = copy(b₊)
                @inferred NNLS.solve_triangular_system!(U⁻¹b₊, work.A, n₊, Val(false))
                @test U⁻¹b₊ ≈ U \ b₊

                L⁻¹b₊ = copy(b₊)
                @inferred NNLS.solve_triangular_system!(L⁻¹b₊, work.A, n₊, Val(true))
                @test L⁻¹b₊ ≈ L \ b₊
            end

            ## Cholesky factors
            @test A₊' * A₊ ≈ U' * U
            @test A₊' * A₊ ≈ L * L'

            F = cholesky!(NNLS.NormalEquation(), work)
            if n₊ > 0
                x′, b′ = rand(n₊), rand(n₊)
                x′′, b′′ = copy(x′), copy(b′)
                ldiv!(x′, F, b′)
                @test b′ ≈ b′′
                @test !(x′ ≈ x′′)
                @test x′ == F \ b′
                @test x′ ≈ (D64.(A₊)' * D64.(A₊)) \ D64.(b′)
                @inferred ldiv!(x′, cholesky!(NNLS.NormalEquation(), work), b′)
            end

            # QR factors
            #   Note: qr(A) = QR relates to cholesky(A'A) = LL' = U'U via:
            #   A'A = U'U = (QR)'(QR) = R'R => R = U (up to row sign)
            posdiag(R) = UpperTriangular(Diagonal(sign.(diag(R))) * R) # ensure diagonal is positive
            R = qr(A₊).R
            @test posdiag(R) ≈ posdiag(U)
        end # GC.@preserve
    end

    return work
end

@testset "NNLS" begin
    for (m, n) in NNLS_SIZES, μ in [0.0, 1e-6, 1e-2, 10.0, 1e4]
        verify_NNLS(m, n, μ)
    end
end

@testset "NNLS Tikh" begin
    for (m, n) in NNLS_SIZES, μ in [0.0, 1e-6, 1e-2, 10.0, 1e4]
        verify_NNLS_tikh(m, n, μ)
    end
end

function build_lcurve_corner_cached_fun(::Type{T} = Float64) where {T}
    # Mock lcurve function with (ξ(μ), η(μ)) = (μ, 1/μ)
    f = CachedFunction(_logμ -> SA[exp(_logμ), exp(-_logμ)], GrowableCache{T, SVector{2, T}}())
    f = LCurveCornerCachedFunction(f, GrowableCache{T, LCurveCornerPoint{T}}(), GrowableCache{T, LCurveCornerState{T}}())
    return f
end

function lcurve_corner_tests()
    function run_lcurve_corner(f)
        return lcurve_corner(f, log(0.1), log(10.0); xtol = 1e-6, Ptol = 1e-6, Ctol = 0)
    end

    #TODO: # Test allocations
    # f = build_lcurve_corner_cached_fun()
    # empty!(f)
    # @test @allocated(run_lcurve_corner(f)) > 0 # caches will be populated with first call
    # empty!(f)
    # @test @allocated(run_lcurve_corner(f)) == 0 # caches should be reused with second call

    # Maximum curvature point for the graph (x(μ), y(μ)) = (μ, 1/μ) occurs at μ=1, i.e. logμ=0
    f = build_lcurve_corner_cached_fun()
    @inferred empty!(f)
    @test @inferred(run_lcurve_corner(f)) ≈ 0 atol = 1e-3
end

@testset "lcurve_corner" begin
    lcurve_corner_tests()
end

function lsqnonneg_lcurve_tests(m, n)
    A, b = rand_NNLS_data(m, n)
    work = DECAES.lsqnonneg_lcurve_work(A, b)

    #TODO: # Test allocations
    # @test @allocated(DECAES.lsqnonneg_lcurve!(work)) == 0 # caches should be initialized to be sufficiently large that normally they don't need to grow

    @inferred DECAES.lsqnonneg_lcurve!(work)
end

@testset "lsqnonneg_lcurve" begin
    for (m, n) in NNLS_SIZES
        lsqnonneg_lcurve_tests(m, n)
    end
end

function lsqnonneg_chi2_tests(m, n)
    A, b = rand_NNLS_data(m, n)
    work = DECAES.lsqnonneg_chi2_work(A, b)

    # Test solver
    x_unreg = DECAES.solve!(work.nnls_prob)
    res²_min = DECAES.resnorm_sq(work.nnls_prob)
    res²_max = sum(abs2, b) # lim_{μ -> ∞} ||A*x(μ) - b||² = ||b||², since lim_{μ -> ∞} x(μ) = 0
    res²_target = √(res²_min * res²_max)
    # res²_target = res²_min + (res²_max - res²_min) / 10
    chi2_target = min(res²_target / res²_min, 1.01 + 0.99 * rand())

    for (method, rtol) in [
        :bisect => 0.01,
        :brent => 0.001,
    ]
        (; x, mu, chi2) = DECAES.lsqnonneg_chi2!(work, chi2_target; method)

        if res²_min <= 0
            # Unregularized solution should be returned
            @test x === x_unreg
            @test isfinite(mu)
            @test mu == 0
            @test chi2 == 1
        elseif res²_min <= 1e-12
            # Minimum is approx zero in floating point, chi2 will be ~noise
            @test mu >= 0
        elseif sum(abs2, x_unreg) == 0
            # Unreg solution is x = 0, and regularization can only reduce ||x||.
            # Since ||x|| = 0 is already minimized, μ is undefined; we enforce that μ = 0 is returned which is consistent with chi2 = 1.
            @test x == x_unreg
            @test isfinite(mu)
            @test mu == 0 # any value of μ should result in x == x_unreg and chi2 == 1
            @test chi2 == 1
        else
            @test mu > 0
            @test chi2 ≈ chi2_target rtol = rtol
        end

        @inferred DECAES.lsqnonneg_chi2!(work, chi2_target; method) # caches should be initialized to be sufficiently large that normally they don't need to grow
    end
end

@testset "lsqnonneg_chi2" begin
    for (m, n) in NNLS_SIZES
        lsqnonneg_chi2_tests(m, n)
    end
end

function test_lsqnonneg_gcv(m, n)
    A, b = rand_NNLS_data(m, n)
    work = DECAES.NNLSGCVRegProblem(A, b)
    logμ = randn()
    μ = exp(logμ)

    # Precompute singular values for GCV (also resizes an internal buffer)
    svdvals!(work.svd_work, A)
    @test work.γ == svdvals(A) # should match exactly (same underlying LAPACK call)

    # Test GCV degrees of freedom
    @test DECAES.gcv_dof(A, μ) ≈ tr(I - A * ((A' * A + μ^2 * I) \ A')) # "degrees of freedom" of normal equation matrix
    @test DECAES.∇gcv_dof(A, μ) ≈ ∇logfinitediff(_logμ -> DECAES.gcv_dof(A, exp(_logμ)), logμ, 1e-6) atol = 1e-4 rtol = 1e-4

    # Test GCV loss function
    gcv = DECAES.gcv!(work, logμ) # gcv! calls `DECAES.solve!` internally
    x = DECAES.solve!(work.nnls_prob_smooth_cache, exp(logμ))
    res² = sum(abs2, A * x - b)
    @test gcv ≈ res² / DECAES.gcv_dof(A, μ)^2

    # Test GCV gradient function
    _gcv, ∇gcv = DECAES.gcv_and_∇gcv!(work, logμ) # gcv_and_∇gcv! calls `DECAES.solve!` internally
    @test _gcv == gcv # primals should match exactly
    @test ∇gcv ≈ ∇logfinitediff(_logμ -> DECAES.gcv!(work, _logμ), logμ, 1e-6) atol = 1e-3 rtol = 1e-3

    # GCV minimization methods shouldn't fail for any m, n
    @test isfinite(DECAES.lsqnonneg_gcv!(work; method = :brent).mu)
    @test isfinite(DECAES.lsqnonneg_gcv!(work; method = :brent_newton).mu)
    @test isfinite(DECAES.lsqnonneg_gcv!(work; method = :nlopt).mu)

    #TODO: Test that different GCV minimization methods are consistent when m >= n

    #TODO: # Test allocations
    # @test @allocated(DECAES.gcv!(work, logμ)) == 0
    # @test @allocated(DECAES.lsqnonneg_gcv!(work)) == 0 # caches should be initialized to be sufficiently large that normally they don't need to grow

    # Test inference
    @inferred DECAES.gcv!(work, logμ)
    @inferred DECAES.lsqnonneg_gcv!(work)
end

@testset "lsqnonneg_gcv" begin
    for (m, n) in NNLS_SIZES
        test_lsqnonneg_gcv(m, n)
    end
end

function lsqnonneg_mdp_tests(m, n)
    A, b = rand_NNLS_data(m, n)
    work = DECAES.lsqnonneg_mdp_work(A, b)

    # Test solver
    x_unreg = DECAES.solve!(work.nnls_prob)
    res²_min = DECAES.resnorm_sq(work.nnls_prob)
    res²_max = sum(abs2, b) # lim_{μ -> ∞} ||A*x(μ) - b||² = ||b||², since lim_{μ -> ∞} x(μ) = 0
    # res²_target = √(res²_min * res²_max)
    res²_target = res²_min + (res²_max - res²_min) / 10
    res_target = √res²_target

    (; x, mu, chi2) = DECAES.lsqnonneg_mdp!(work, res_target)
    res² = sum(abs2, A * x - b)

    if mu <= 0
        @test mu == 0
        @test res² ≈ res²_min atol = 1e-12
        @test chi2 == 1
    else
        @test mu > 0
        @test res² ≈ res²_target rtol = 2e-3 atol = 1e-12 # internally we solve f(logμ) = res² - res²_target = 0 to tolerance abs(f) < 1e-3 * res²_target
        @test chi2 ≈ res² / res²_min rtol = 2e-3 atol = 1e-12 # should also hold when res²_min = 0, i.e. when chi2 = Inf
    end

    #TODO: # Test allocations
    # @test @allocated(DECAES.lsqnonneg_mdp!(work, res_target)) == 0 # caches should be initialized to be sufficiently large that normally they don't need to grow

    # Test inference
    @inferred DECAES.lsqnonneg_mdp!(work, res_target)
end

@testset "lsqnonneg_mdp" begin
    for (m, n) in NNLS_SIZES
        lsqnonneg_mdp_tests(m, n)
    end
end

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
@testset "least-squares gradients" begin
    for (m, n) in NNLS_SIZES
        A, b = rand_NNLS_data(m, n)
        A_μ = μ -> [A; μ * LinearAlgebra.I]
        B_μ = μ -> LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(A' * A + μ^2 * I))
        x_μ = μ -> A_μ(μ) \ [b; zeros(n)]

        μ = 0.99
        B = B_μ(μ)
        x = x_μ(μ)

        # Derivative of solution x w.r.t. μ
        f = _μ -> x_μ(_μ)
        dx_dμ = -2 * μ * (B \ x)
        @test dx_dμ ≈ -2 * μ * (B \ (B \ (A' * b)))
        @test dx_dμ ≈ ∇logfinitediff(f ∘ exp, log(μ), 1e-6) rtol = 1e-4

        # Derivative of A*x (or equivalently, A*x-b) w.r.t. μ
        f = _μ -> A * x_μ(_μ)
        dAx_dμ = A * dx_dμ
        @test dAx_dμ ≈ -2 * μ * (A * (B \ (B \ (A' * b))))
        @test dAx_dμ ≈ ∇logfinitediff(f ∘ exp, log(μ), 1e-6) rtol = 1e-4

        # Derivative of solution l2-norm ||x||^2 w.r.t. μ
        f = _μ -> sum(abs2, x_μ(_μ))
        dx²_dμ = 2 * dot(x, dx_dμ)
        @test dx²_dμ ≈ -4 * μ * dot(b, (A * (B \ (B \ (B \ (A' * b))))))
        @test dx²_dμ ≈ ∇logfinitediff(f ∘ exp, log(μ), 1e-6) rtol = 1e-4

        # Derivative of residual l2-norm ||A*x-b||^2 w.r.t. μ
        f = _μ -> sum(abs2, A * x_μ(_μ) - b)
        dAxb_dμ = -2μ^2 * dot(x, dx_dμ)
        @test dAxb_dμ ≈ 2 * dot(A * x - b, dAx_dμ)
        @test dAxb_dμ ≈ -4 * μ * dot(A * x - b, A * (B \ (B \ (A' * b))))
        @test dAxb_dμ ≈ ∇logfinitediff(f ∘ exp, log(μ), 1e-6) rtol = 1e-4
    end
end

function NNLSTikhonovRegProblem_tests(m, n)
    T = Double64 # need higher precision for finite differences
    A, b = rand_NNLS_data(m, n)
    A, b = T.(A), T.(b)
    work = NNLSTikhonovRegProblem(A, b)
    withsolve(f, μ) = (DECAES.solve!(work, μ); return f(work))

    GC.@preserve work for (i, μ) in enumerate(T[0.01, 0.05, 0.25, 0.99])
        @test xor(i > 1, isnan(DECAES.regparam(work))) # μ should be initialized to NaN
        @test DECAES.regparam!(work, μ) == μ
        @test DECAES.regparam(work) == μ

        x = DECAES.solve!(work, μ)
        @test all(>=(0), x)
        @test withsolve(DECAES.regnorm, μ) ≈ μ^2 * sum(abs2, x)
        @test withsolve(DECAES.resnorm_sq, μ) ≈ sum(abs2, A * x - b)
        @test withsolve(DECAES.seminorm_sq, μ) ≈ sum(abs2, x)

        @inferred DECAES.solve!(work, μ)
        @inferred DECAES.regparam(work)

        ∇μ = DECAES.∇regnorm(work)
        @test ∇μ ≈ ∇logfinitediff(_logμ -> withsolve(DECAES.regnorm, exp(_logμ)), log(μ), T(1e-6)) rtol = 1e-3
        @test @inferred(DECAES.∇regnorm(work)) isa T

        ∇μ = DECAES.∇resnorm_sq(work)
        @test ∇μ ≈ ∇logfinitediff(_logμ -> withsolve(DECAES.resnorm_sq, exp(_logμ)), log(μ), T(1e-6)) rtol = 1e-3
        @test @inferred(DECAES.∇resnorm_sq(work)) isa T

        ∇μ = DECAES.∇seminorm_sq(work)
        @test ∇μ ≈ ∇logfinitediff(_logμ -> withsolve(DECAES.seminorm_sq, exp(_logμ)), log(μ), T(1e-6)) rtol = 1e-3
        @test @inferred(DECAES.∇seminorm_sq(work)) isa T

        ∇μ = DECAES.solution_gradnorm(work)
        @test ∇μ ≈ norm(∇logfinitediff(_logμ -> withsolve(copy ∘ DECAES.solution, exp(_logμ)), log(μ), T(1e-6))) rtol = 1e-3
        @test @inferred(DECAES.solution_gradnorm(work)) isa T

        ∇²μ = DECAES.∇²resnorm_sq(work)
        @test ∇²μ ≈ ∇²logfinitediff(_logμ -> withsolve(DECAES.resnorm_sq, exp(_logμ)), log(μ), T(1e-6)) rtol = 1e-2 atol = 1e-2
        @test @inferred(DECAES.∇²resnorm_sq(work)) isa T

        ∇²μ = DECAES.∇²seminorm_sq(work)
        @test ∇²μ ≈ ∇²logfinitediff(_logμ -> withsolve(DECAES.seminorm_sq, exp(_logμ)), log(μ), T(1e-6)) rtol = 1e-2 atol = 1e-2
        @test @inferred(DECAES.∇²seminorm_sq(work)) isa T

        # Curvature computation
        DECAES.solve!(work, μ)
        ξ = DECAES.resnorm_sq(work)
        η = DECAES.seminorm_sq(work)
        if η > 0
            ξ_fun = _logμ -> withsolve(DECAES.resnorm_sq, exp(_logμ))
            η_fun = _logμ -> withsolve(DECAES.seminorm_sq, exp(_logμ))
            C_fun = _logμ -> withsolve(Base.Fix1(DECAES.curvature, identity), exp(_logμ))
            C_menger = DECAES.menger(ξ_fun, η_fun; h = T(1e-4))

            ξ′ = DECAES.∇resnorm_sq(work)
            η′ = DECAES.∇seminorm_sq(work)
            ξ′′ = DECAES.∇²resnorm_sq(work)
            η′′ = DECAES.∇²seminorm_sq(work)
            C = (ξ′ * η′′ - η′ * ξ′′) / √((ξ′^2 + η′^2)^3)

            @test C_fun(log(μ)) ≈ C
            @test C_fun(log(μ)) ≈ C_menger(log(μ)) rtol = 0.05 atol = 0.05

            C̄_fun = _logμ -> withsolve(Base.Fix1(DECAES.curvature, log), exp(_logμ))
            C̄_menger = DECAES.menger(log ∘ ξ_fun, log ∘ η_fun; h = T(1e-4))

            _ξ′ = ξ′ / ξ # d/dlogμ ξ(μ) = ξ'(μ) / ξ(μ)
            _η′ = η′ / η
            _ξ′′ = ξ′′ / ξ - _ξ′^2 # d²/d(logμ)² ξ(μ) = ξ''(μ) / ξ(μ) - (ξ'(μ) / ξ(μ))^2
            _η′′ = η′′ / η - _η′^2
            C̄ = (_ξ′ * _η′′ - _η′ * _ξ′′) / √((_ξ′^2 + _η′^2)^3)

            @test C̄_fun(log(μ)) ≈ C̄
            @test C̄_fun(log(μ)) ≈ C̄_menger(log(μ)) rtol = 0.05 atol = 0.05
        end
    end # GC.@preserve
end

@testset "NNLSTikhonovRegProblem" begin
    for (m, n) in NNLS_SIZES
        NNLSTikhonovRegProblem_tests(m, n)
    end
end

function NNLSTikhonovRegProblemCache_tests(m, n, ::Val{N} = Val(5)) where {N}
    A, b = rand_NNLS_data(m, n)
    work0 = NNLSTikhonovRegProblem(A, b)
    work = NNLSTikhonovRegProblemCache(A, b, Val(N))

    # Import to test at least as many μ values as the cache size
    count = 0
    GC.@preserve work work0 for (i, μ) in enumerate(exp10.(range(-3.0, 0.0; length = 2 * N)))
        DECAES.solve!(work0, μ)
        x0 = DECAES.solution(work0)

        DECAES.solve!(work, μ)
        x = DECAES.solution(work[])
        @test x ≈ x0
        @test DECAES.regparam(work[]) == μ

        DECAES.solve!(work, μ)
        @test x === DECAES.solution(work[]) # retrieves cached solution
        @test x ≈ x0
        @test DECAES.regparam(work[]) == μ

        count += 1
        if count <= N
            @test !any(isnan.(DECAES.regparam.(work.cache[1:count])))
            @test all(isnan.(DECAES.regparam.(work.cache[count+1:N])))
        else
            @test !any(isnan.(DECAES.regparam.(work.cache)))
        end
        @test work.idx[] == mod1(count, N)
    end
end

@testset "NNLSTikhonovRegProblemCache" begin
    for (m, n) in NNLS_SIZES
        NNLSTikhonovRegProblemCache_tests(m, n)
    end
end
