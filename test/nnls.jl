Random.seed!(0) # reproducible randomized tests: failures should be bisectable, not luck-of-the-draw

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
    work.diag .= rand(0:n, n)
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

    for mode in [:direct, :shuffle]
        # Fill workspace with junk values
        work.A .= randn(m + n, n)
        work.b .= randn(m + n)
        work.x .= randn(n)
        work.w .= randn(n)
        work.zz .= randn(m + n)
        work.idx .= rand(Int, n)
        work.diag .= rand(0:n, n)
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

# Direct workspace calling convention: load the workspace manually, preload the initial dual w, and call `unsafe_nnls!` with `init_dual = false`.
# The solver must use the preloaded w and derive all other internal state from the loaded workspace fields (A, b, x, idx, diag) alone.
@testset "NNLS direct workspace API" begin
    for (m, n) in NNLS_SIZES, μ in [0.0, 1e-2]
        A0, b0 = rand_NNLS_data(m, n)
        A, b = maybe_pad_NNLS_data(A0, b0, μ)
        work = NNLS.NNLSWorkspace(A, b)
        x_ref = copy(μ > 0 ? NNLS.nnls!(work, A, b, μ) : NNLS.nnls!(work, A, b))

        NNLS.load!(work, A, b)
        μ > 0 ? NNLS.init_nnls!(work, μ) : NNLS.init_nnls!(work)
        mul!(work.w, A0', b0) # caller-preloaded initial dual w₀ = A₀'b₀
        μ > 0 ? NNLS.unsafe_nnls!(work, A0, μ; init_dual = false) : NNLS.unsafe_nnls!(work, A0; init_dual = false)
        @test NNLS.solution(work) ≈ x_ref atol = 1e-12 rtol = 1e-8
        @test all(isfinite, NNLS.solution(work))
    end
end

####
#### Adversarial NNLS problems
####

# Test NNLS on ill-conditioned or degenerate problems: near-collinear columns, duplicated columns, zero columns, rank deficiency, degenerate right-hand sides.
# Solutions may be non-unique and active-set duals may vanish, so instead of the strict sign checks of `verify_NNLS` the solutions are verified with a tolerance-based global optimality certificate:
# NNLS is convex, so x ≥ 0 together with w = A'(b - Ax) ≤ ε and |x ⋅ w| ≤ ε (evaluated in Double64) certifies optimality regardless of conditioning or uniqueness.

# Multi-exponential decay basis: smooth strictly positive columns with near-collinear neighbours, sparse nonnegative spectrum
function expdecay_NNLS_data(m, n)
    t = range(0, 2; length = m)
    τ = exp10.(range(-1.5, 0.5; length = n))
    A = [exp(-tᵢ / τⱼ) for tᵢ in t, τⱼ in τ]
    x = zeros(n)
    for _ in 1:3
        x[rand(1:n)] += rand()
    end
    b = A * x + 1e-3 .* randn(m)
    return A, b
end

adversarial_NNLS_generators() = [
    "expdecay" => expdecay_NNLS_data,
    "hilbert" => (m, n) -> (A = [1 / (i + j - 1) for i in 1:m, j in 1:n]; (A, A * randn(n))),
    "duplicates" => (m, n) -> (A = rand(m, cld(n, 2))[:, mod1.(1:n, cld(n, 2))]; (A, A * randn(n))),
    "zerocols" => (m, n) -> (A = rand(m, n); A[:, rand(1:n, cld(n, 4))] .= 0; (A, rand(m))),
    "rankdeficient" => (m, n) -> (A = rand(m, max(1, min(m, n) ÷ 2)) * rand(max(1, min(m, n) ÷ 2), n); (A, A * randn(n))),
    "zerorhs" => (m, n) -> (rand(m, n), zeros(m)),
    "negcone" => (m, n) -> (A = rand(m, n); (A, -A * rand(n))), # b in -cone(A): x = 0 is optimal
]

function verify_NNLS_adversarial(A0, b0, μ = 0.0)
    D64 = Double64
    n = size(A0, 2)
    A, b = maybe_pad_NNLS_data(A0, b0, μ)
    work = NNLS.NNLSWorkspace(A, b)

    modes = μ > 0 ? [:direct, :shuffle, :fullseed] : [:direct]
    for mode in modes
        if mode === :direct && μ == 0
            @inferred NNLS.nnls!(work, A, b)
        elseif mode === :direct
            @inferred NNLS.nnls!(work, A, b, μ)
        elseif mode === :shuffle
            @inferred NNLS.nnls!(work, A, b, μ, Random.randperm(n), rand(0:n))
        else # seed with the full column set
            @inferred NNLS.nnls!(work, A, b, μ, collect(1:n), n)
        end
        @test work.mode[] == 0 # success

        x = NNLS.solution(work)
        i₊ = NNLS.components(work)

        # Primal feasibility and support partitioning
        @test all(isfinite, x)
        @test all(>=(0), x)
        @test all(>(0), x[i₊])
        @test all(==(0), x[setdiff(1:n, i₊)])
        @test isperm(work.idx)
        @test isperm(work.invidx)

        # Reported dual is nonpositive (exact termination invariant), and the reported residual norm matches the definition
        @test all(<=(0), NNLS.dual(work))
        @test NNLS.residualnorm(work) ≈ norm(A * x - b) rtol = 1e-10 atol = 1e-12 * max(1, norm(b))

        # Global optimality certificate at the computed solution
        w64 = D64.(A)' * (D64.(b) - D64.(A) * D64.(x))
        εw = 1e-10 * max(1, norm(D64.(A)' * D64.(b)))
        @test all(<=(εw), w64) # dual feasibility
        @test maximum(abs, x .* w64; init = zero(D64)) <= εw * max(1, maximum(x; init = 0.0)) # complementary slackness
    end

    return work
end

@testset "Adversarial NNLS ($name)" for (name, data) in adversarial_NNLS_generators()
    for (m, n) in [(8, 16), (16, 8), (16, 16), (32, 64), (48, 40)], μ in [0.0, 1e-6, 1e-2, 10.0]
        A0, b0 = data(m, n)
        verify_NNLS_adversarial(A0, b0, μ)
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

    (; x, mu, chi2) = @inferred DECAES.lsqnonneg_lcurve!(work)
    @test all(>=(0), x)
    @test isfinite(mu) && mu >= 0
    if mu > 0
        # Self-consistency: the returned solution is the exact Tikhonov-NNLS solution at the returned μ
        @test x ≈ DECAES.lsqnonneg_tikh(A, b, mu) rtol = 1e-8 atol = 1e-12 * norm(b)
        @test chi2 >= 1 - √eps() # res²(μ)/res²(0) ≥ 1 up to roundoff between the two evaluation paths

        # Slope-collapse guard invariant: the accepted corner never sits in the near-vertical μ → 0 tail:
        # log-log tangent slope |S| = res²/(‖x‖²μ²) ≤ (1 + ϵ) * slope_max.
        η² = sum(abs2, x)
        @test η² == 0 || sum(abs2, A * x - b) / (η² * mu^2) < 1.01 * DECAES.LCURVE_SLOPE_MAX[]
    else
        @test chi2 == 1
    end
end

@testset "lsqnonneg_lcurve" begin
    for (m, n) in NNLS_SIZES
        lsqnonneg_lcurve_tests(m, n)
    end
end

function lsqnonneg_reginska_tests(m, n)
    A, b = rand_NNLS_data(m, n)
    work = DECAES.lsqnonneg_reginska_work(A, b)

    (; x, mu, chi2) = @inferred DECAES.lsqnonneg_reginska!(work)
    @test all(>=(0), x)
    @test isfinite(mu) && mu >= 0
    if mu > 0
        # Self-consistency: the returned solution is the exact Tikhonov-NNLS solution at the returned μ
        @test x ≈ DECAES.lsqnonneg_tikh(A, b, mu) rtol = 1e-8 atol = 1e-12 * norm(b)

        # Stationarity of the minimum-product criterion: at the selected μ the log-log L-curve tangent slope is -1, i.e. the balance point res² = μ²‖x‖² holds with |log S| ≤ (local slope of g) × (brent xatol = 1e-4 on logμ).
        S = sum(abs2, A * x - b) / (sum(abs2, x) * mu^2)
        @test 0.9999 < S < 1.0001
        @test chi2 >= 1
    else
        @test chi2 == 1
    end
end

@testset "lsqnonneg_reginska" begin
    for (m, n) in NNLS_SIZES
        lsqnonneg_reginska_tests(m, n)
    end
end

# Reginska selects the *leftmost* balance point |S| = 1, the smallest local minimizer of Ψ = res²·‖x‖², certified by the leap scan.
# Verified against a brute-force reference: the returned μ must equal the leftmost downward crossing of g(logμ) = log res² − log‖x‖² − 2logμ.
function reginska_expdecay_data(m, n)
    t = range(0, 2; length = m)
    τ = exp10.(range(-1.5, 0.5; length = n))
    A = [exp(-tᵢ / τⱼ) for tᵢ in t, τⱼ in τ]
    x = zeros(n)
    for _ in 1:3
        x[rand(1:n)] += rand()
    end
    return A, A * x .+ 1e-3 .* randn(m)
end

function reginska_g(A, b, logμ)
    x = DECAES.lsqnonneg_tikh(A, b, exp(logμ))
    res², η² = sum(abs2, A * x - b), sum(abs2, x)
    return η² == 0 ? Inf : log(res²) - log(η²) - 2 * logμ
end

function reginska_leftmost_downcrossing(A, b, logμ_grid = range(-8, 2; length = 1000); atol = 1e-6)
    prevpos, prevl = false, first(logμ_grid)
    for logμ in logμ_grid
        g = reginska_g(A, b, logμ)
        if prevpos && g <= 0
            lo, hi = prevl, logμ
            while hi - lo > atol
                mid = (lo + hi) / 2
                if reginska_g(A, b, mid) > 0
                    lo = mid
                else
                    hi = mid
                end
            end
            return (lo + hi) / 2
        end
        prevpos, prevl = g > 0, logμ
    end
    return NaN
end

@testset "lsqnonneg_reginska leftmost crossing" begin
    for (m, n) in ((32, 40), (48, 40), (32, 60), (48, 32), (24, 48)), _ in 1:3
        A, b = reginska_expdecay_data(m, n)
        (; mu) = DECAES.lsqnonneg_reginska!(DECAES.lsqnonneg_reginska_work(A, b))
        mu > 0 || continue
        lc = reginska_leftmost_downcrossing(A, b)
        @test !isnan(lc) # an interior balance point exists
        @test abs(log(mu) - lc) < 0.001 # the leap scan lands on the leftmost crossing
    end
end

@testset "lsqnonneg_reginska degenerate (μ = 0)" begin
    # b = 0: res²_min = 0, no balance point ⇒ unregularized (μ = 0, chi2 = 1)
    r0 = DECAES.lsqnonneg_reginska!(DECAES.lsqnonneg_reginska_work(rand(8, 6), zeros(8)))
    @test r0.mu == 0 && r0.chi2 == 1 && all(==(0), r0.x)

    # b ∈ -cone(A): x_unreg = 0 (no active columns) ⇒ no balance point ⇒ μ = 0
    A = rand(8, 6)
    rc = DECAES.lsqnonneg_reginska!(DECAES.lsqnonneg_reginska_work(A, -A * rand(6)))
    @test rc.mu == 0 && rc.chi2 == 1
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
        :brent_gram => 0.001,
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

    # Precompute the squared singular values for GCV
    DECAES.spectrum!(work)
    @test work.γ² == svdvals(A) .^ 2

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

# Integration test for the gridded GCV dof interpolation. `lsqnonneg_gcv!` with a `GriddedSpectrumInterpolator` interpolates dof(μ) at the voxel's α instead of computing a per-voxel SVD;
# verify the μ-selection matches the exact per-call-SVD path. The decay bases and analytic ∂A/∂α are the real EPG ensemble used by the pipeline.
function test_gcv_gridded_interp()
    o = DECAES.mock_t2map_opts(Float64; MatrixSize = (1, 1, 1), nTE = 32, nT2 = 24, Silent = true)
    θ = DECAES.default_epg_parameters(o)
    T2t = DECAES.T2_component_times(o)
    Aα(α) = DECAES.epg_decay_basis(DECAES.restructure(θ, (; α)), T2t)
    M, N = o.nTE, o.nT2
    flip_angle_work = DECAES.FlipAngleOptimizationWorkspace(o, zeros(M, N), zeros(M))
    ensemble = flip_angle_work.decay_basis_set_ensemble
    interp = DECAES.GriddedSpectrumInterpolator(ensemble.decay_basis_set, ensemble.∇decay_basis_set, DECAES.flip_angles(o))
    αs = interp.αs

    function mock_signal(α)
        A = Aα(α)
        x = zeros(N)
        for _ in 1:3
            x[rand(1:N)] += rand()
        end
        b = A * x .+ 1e-3 .* randn(M)
        b ./= maximum(b)
        return b
    end

    CURR_GCV_INTERP_DOF = DECAES.GCV_INTERP_DOF[]
    DECAES.GCV_INTERP_DOF[] = true # the interpolated path is opt-in; the exact spectrum is the default
    try
        for trial in 1:20
            b = mock_signal(αs[1] + (αs[end] - αs[1]) * rand())

            # At a grid node the Hermite interpolant reproduces its endpoint data, so the interpolated dof equals the exact dof to roundoff
            αnode = αs[5+(trial%4)]
            γ²node = svdvals(Aα(αnode)) .^ 2
            for μ in (1e-3, 1e-2, 1e-1, 1.0)
                @test DECAES.gcv_dof_interp(interp, αnode, M, N, μ) ≈ DECAES.gcv_dof(M, N, γ²node, μ) rtol = 1e-12
            end

            # The selected μ then agrees only to the search tolerance: the exact path reaches the spectrum through `svdvals` and the interpolator's slices through `svd`, and a roundoff-level dof difference can move the Brent result by up to `atol` on a flat objective
            r_node_exact = DECAES.lsqnonneg_gcv!(DECAES.NNLSGCVRegProblem(Aα(αnode), b); method = :brent)
            r_node_interp = DECAES.lsqnonneg_gcv!(DECAES.NNLSGCVRegProblem(Aα(αnode), b, nothing, (interp, Ref(αnode))); method = :brent)
            @test r_node_exact.mu ≈ r_node_interp.mu rtol = 1e-3
            @test r_node_interp.x ≈ r_node_exact.x rtol = 1e-3 atol = 1e-8

            # At an interior α the interpolated dof carries the cubic-Hermite error, so the selected μ agrees only within a band
            αint = (αs[6] + αs[7]) / 2
            r_int_exact = DECAES.lsqnonneg_gcv!(DECAES.NNLSGCVRegProblem(Aα(αint), b); method = :brent)
            r_int_interp = DECAES.lsqnonneg_gcv!(DECAES.NNLSGCVRegProblem(Aα(αint), b, nothing, (interp, Ref(αint))); method = :brent)
            @test r_int_exact.mu ≈ r_int_interp.mu rtol = 1e-3
        end
    finally
        DECAES.GCV_INTERP_DOF[] = CURR_GCV_INTERP_DOF
    end
end

@testset "lsqnonneg_gcv gridded interpolation" test_gcv_gridded_interp()

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

# Gradient notes:
#
# d/dμ x(μ):
#   ∇x = -2 * μ * (B\(B\(A'b)))
#      = -2 * μ * (B\x)             <-- x = (A'A)\(A'b) = B\(A'b)
#
# d/dμ ||A*x(μ)-b||^2:
#   ∇μ = 2 * ((A*x-b)' * (A*∇x))
#      = 2 * (A'*(A*x-b))' * ∇x
#      = 2 * ((-μ^2*x)' * ∇x)       <-- A'*(A*x-b) = -μ^2*x, as 0 = w = [A; μI]' * ([A; μI]*x - [b;0]) = [A; μI]' * [A*x-b; μ*x] = A'*(A*x-b) + μ^2*x
#      = 4μ^3 * x' * (B\x)
#
# d/dμ ||x(μ)||^2:
#   ∇μ = -4μ * b' * (A*(B\(B\(B\(A'b)))))
#      = -4μ * b' * (A*(B\(B\x)))   <-- x = (A'A)\(A'b) = B\(A'b)
#      = -4μ * (B\(A'b))'* (B\x)    <-- B = B'
#      = -4μ * x' * (B\x)
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

# The μ-selection methods on adversarial (ill-conditioned / rank-deficient / degenerate) inputs.
# Each method's Gram fast path has conditioning/iteration guards that fall back to the exact QR solve; those guards fire only on ill-conditioned inputs, which the strictly-positive random data of the per-method testsets above never produces.
# The regularized (μ > 0) returns are certified: the returned x must be KKT-optimal for the Tikhonov problem min_{x≥0} ‖Ax−b‖² + μ²‖x‖² at the returned μ. By strong convexity, the Double64 dual/complementarity certificate is sufficient. This exercises the guarded Gram path + exact-QR final solve.
function verify_reg_kkt_regularized(A0, b0, x, mu)
    D64 = Double64
    A, b, x = D64.(A0), D64.(b0), D64.(x)
    w = A' * (b - A * x) .- D64(mu)^2 .* x # dual (negative half-gradient) of the Tikhonov objective
    ε = 1e-8 * max(1, norm(A' * b))
    @test all(>=(-1e-10), x) # primal feasibility
    @test all(<=(ε), w) # dual feasibility
    @test maximum(abs, x .* w; init = zero(D64)) <= ε * max(1, maximum(x; init = 0.0)) # complementary slackness
end

@testset "Adversarial regularized NNLS ($name)" for (name, data) in adversarial_NNLS_generators()
    for (m, n) in [(16, 8), (16, 16), (32, 24)]
        A, b = data(m, n)
        runs = Any[
            DECAES.lsqnonneg_lcurve!(DECAES.lsqnonneg_lcurve_work(A, b)),
            DECAES.lsqnonneg_gcv!(DECAES.lsqnonneg_gcv_work(A, b)),
            DECAES.lsqnonneg_reginska!(DECAES.lsqnonneg_reginska_work(A, b)),
            DECAES.lsqnonneg_chi2!(DECAES.lsqnonneg_chi2_work(A, b), 1.02),
        ]
        norm(b) > 0 && push!(runs, DECAES.lsqnonneg_mdp!(DECAES.lsqnonneg_mdp_work(A, b), 0.5 * norm(b)))
        for (; x, mu) in runs
            @test all(isfinite, x)
            @test all(>=(0), x)
            isfinite(mu) && mu > 0 && verify_reg_kkt_regularized(A, b, x, mu)
        end
    end
end
