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
        i₀ = idx[(n₊+1):end]
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
        @test work.zz[(n₊+1):end] == work.b[(n₊+1):end]
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
        work.nsetp[] = rand(0:min(m+n, n))

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
            i₀ = idx[(n₊+1):end]
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
            @test work.zz[(n₊+1):end] == work.b[(n₊+1):end]
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
        @test NNLS.solution(work) ≈ x_ref atol = 1e-12 rtol = 1e-12
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
        @test NNLS.residualnorm(work) ≈ norm(A * x - b) rtol = 1e-12 atol = 1e-12 * max(1, norm(b))

        # Global optimality certificate at the computed solution
        w64 = D64.(A)' * (D64.(b) - D64.(A) * D64.(x))
        εw = 1e-12 * max(1, norm(D64.(A)' * D64.(b)))
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

# Mock L-curve (ξ(t), η(t)) = (e^t, e^{-t}), whose exact geometry is v² = e^{2t} + e^{-2t}, κ = 2/v³ and ω = κv = 2/v².
# The active-set signature is constant, so the search sees one analytic branch, and `sig₀` below is chosen distinct from it so the unregularized endpoint is never reached.
# This is an arbitrary planar curve, not a Tikhonov path, so `tangent_angle` does not read its tangent; it exercises the golden-state mechanics only. See `build_tikhonov_corner_cached_fun` for the rotation-based sweep.
function build_lcurve_corner_cached_fun(::Type{T} = Float64) where {T}
    function mock_lcurve_point(t)
        v² = exp(2t) + exp(-2t)
        return LCurveCornerPoint(SA{T}[exp(t), exp(-t)], 2 / v²^(3 // 2), 2 / v², UInt128(1))
    end
    f = CachedFunction(mock_lcurve_point, GrowableCache{T, LCurveCornerPoint{T}}(64, isapprox))
    return LCurveCornerCachedFunction(f, GrowableCache{Int, LCurveCornerState{T}}(64))
end

function lcurve_corner_tests()
    # Maximum curvature of (x(μ), y(μ)) = (μ, 1/μ) occurs at μ=1, i.e. logμ=0. Each seed is far enough away that the corner is found by expansion, not by the initial state.
    for t₀ in (-5.0, -0.5, 3.0)
        f = build_lcurve_corner_cached_fun()
        @inferred empty!(f)
        @test @inferred(lcurve_corner(f, t₀, SA[NaN, NaN], UInt128(0); xtol = 1e-6, Ptol = 1e-6)) ≈ 0 atol = 1e-3
    end

    # A seed outside the caller's admissible interval slides the initial state into the interval instead of ending the search, and nothing is ever evaluated outside it.
    for t₀ in (6.0, -6.0)
        f = build_lcurve_corner_cached_fun()
        bounds = (-1.5, 1.5)
        @test lcurve_corner(f, t₀, SA[NaN, NaN], UInt128(0); xtol = 1e-6, Ptol = 1e-6, bounds) ≈ 0 atol = 1e-3
        @test all(t -> bounds[1] <= t <= bounds[2], keys(f.f.cache))
    end

    # Inverse golden expansions undo golden contractions exactly, so an expanded state contracts back onto its parent.
    f = build_lcurve_corner_cached_fun()
    φ, t₀, Δ = Base.MathConstants.φ, 0.7, 1.0
    x⃗ = SA[t₀-Δ/φ^2, t₀, t₀+Δ/φ^3, t₀+Δ/φ]
    s = DECAES.golden_state(x⃗, SA[f(x⃗[1]), f(x⃗[2]), f(x⃗[3]), f(x⃗[4])], SA[Inf, Inf])
    @test DECAES.move_right(f, DECAES.expand_left(f, s)).x⃗ ≈ s.x⃗
    @test DECAES.move_left(f, DECAES.expand_right(f, s)).x⃗ ≈ s.x⃗

    # The branch queue holds the discarded siblings in decreasing width, so retiring its oldest entry examines the largest unexplored region first. `lcurve_certify!` depends on that ordering.
    let f = build_lcurve_corner_cached_fun()
        φ, t₀, Δ = Base.MathConstants.φ, 0.7, 1.0
        x⃗ = SA[t₀-Δ/φ^2, t₀, t₀+Δ/φ^3, t₀+Δ/φ]
        s = DECAES.golden_state(x⃗, SA[f(x⃗[1]), f(x⃗[2]), f(x⃗[3]), f(x⃗[4])], SA[Inf, Inf])
        empty!(f.state_stack)
        for _ in 1:8
            push!(f.state_stack, (length(f.state_stack), s))
            s = DECAES.move(f, s, DECAES.contract_left(s))
        end
        widths = [v.x⃗[4] - v.x⃗[1] for (_, v) in f.state_stack]
        @test issorted(widths; rev = true)
        _, widest = popfirst!(f.state_stack)
        @test widest.x⃗[4] - widest.x⃗[1] == maximum(widths)
        @test length(f.state_stack) == 7
    end

    # Contraction preserves the interior ratios (φ⁻², φ⁻¹), a fixed point whose linearization has multiplier -1, so a deviation neither grows nor accumulates.
    # Stepping a new abscissa off a stored one gives -φ² instead, which reorders the abscissas at Float32 precision.
    let f = build_lcurve_corner_cached_fun(Float32), φ = Float32(Base.MathConstants.φ), t₀ = -1.836f0, Δ = 1.0f0
        x⃗ = SA[t₀-Δ/φ^2, t₀, t₀+Δ/φ^3, t₀+Δ/φ]
        s₀ = DECAES.golden_state(x⃗, SA[f(x⃗[1]), f(x⃗[2]), f(x⃗[3]), f(x⃗[4])], SA[Inf32, Inf32])
        for move in (DECAES.move_left, DECAES.move_right)
            s = s₀
            for _ in 1:20
                s = move(f, s)
                w = s.x⃗[4] - s.x⃗[1]
                @test issorted(s.x⃗)
                @test (s.x⃗[2] - s.x⃗[1]) / w ≈ 1 / φ^2 atol = 1.0f-2
                @test (s.x⃗[3] - s.x⃗[1]) / w ≈ 1 / φ atol = 1.0f-2
            end
        end
    end
end

@testset "lcurve_corner" begin
    lcurve_corner_tests()
end

# The corner search computes the curvature from the Gram fast path, which must agree with the QR path, and ω must be the derivative of the tangent angle.
function lcurve_geometry_tests(A, b; test_q = true)
    work = DECAES.lsqnonneg_lcurve_work(A, b)
    work_D64 = DECAES.lsqnonneg_lcurve_work(Double64.(A), Double64.(b))
    DECAES.reset_cache!(work.nnls_prob_smooth_cache)
    DECAES.solve_unreg!(work.nnls_prob, work.nnls_prob_seed)
    tikh = DECAES.lsqnonneg_tikh_work(A, b)
    h = √eps()
    rtol = 1e-6
    for t in range(-6.0, 2.0; length = 17)
        DECAES.solve!(tikh, exp(t))
        ∇ = DECAES.gradient_temps(tikh)

        # `inv_quadratic_form` is defined only after a successful Gram solve. The fallback path leaves the factorization stale and returns NaN, and `lcurve_point` takes q from the QR path there instead.
        DECAES.nnls_gram_setup!(work)
        ξ²_gram = DECAES.NNLS.solve!(work.nnls_gram, A, b, exp(t))
        if !isnan(ξ²_gram) && DECAES.seminorm_sq(tikh) > 0
            @test ξ²_gram ≈ DECAES.resnorm_sq(tikh) rtol = rtol
            @test DECAES.NNLS.seminorm_sq(work.nnls_gram) ≈ DECAES.seminorm_sq(tikh) rtol = rtol
            test_q && @test DECAES.NNLS.inv_quadratic_form(work.nnls_gram) ≈ ∇.xᵀB⁻¹x rtol = rtol
        end

        # `lcurve_point` is valid on both paths
        DECAES.nnls_gram_setup!(work)
        p = DECAES.lcurve_point(work, exp(t))
        @test exp(p.P[1]) ≈ DECAES.resnorm_sq(tikh) rtol = rtol
        if DECAES.seminorm_sq(tikh) == 0
            @test !isfinite(p.P[2]) # x = 0 has no L-curve point at all; the search reaches it only through `is_saturated`
            continue
        end
        @test exp(p.P[2]) ≈ DECAES.seminorm_sq(tikh) rtol = rtol
        @test p.κ ≈ DECAES.curvature(log, tikh, ∇) rtol = rtol

        # ω = dθ/dt for θ = -atan(ξ²/(μ²η²)), by central difference, where the stencil stays on one active set
        p₋, p₊ = DECAES.lcurve_point(work_D64, exp(Double64(t) - h)), DECAES.lcurve_point(work_D64, exp(Double64(t) + h))
        if p₋.sig == p.sig == p₊.sig && isfinite(p₋.P[2]) && isfinite(p₊.P[2])
            θ(pᵢ, tᵢ) = -atan(exp(pᵢ.P[1] - pᵢ.P[2] - 2 * tᵢ))
            @test p.ω ≈ (θ(p₊, t + h) - θ(p₋, t - h)) / 2h rtol = rtol
        end
    end
end

@testset "L-curve geometry (Gram path vs QR path)" begin
    # Well-conditioned overdetermined problems
    for (m, n) in ((32, 20), (48, 40), (64, 40))
        lcurve_geometry_tests(rand(m, n), rand(m))
    end

    # Adversarial sizes, including underdetermined and rank-deficient, which also exercise the QR-solver fallback
    for (m, n) in NNLS_SIZES
        lcurve_geometry_tests(rand_NNLS_data(m, n)...)
    end

    # Exactly duplicated columns: ∇²Jμ = 2AᵀA + 2μ²I ⪰ 2μ²I is strongly convex for every μ > 0, so x and q are unique and this is a conditioning stress test rather than a nonuniqueness exemption
    A, b = rand_NNLS_data(32, 20)
    lcurve_geometry_tests([A A], b)

    # An exponential basis at cond(A) ~ 1e17, the regime DECAES actually runs in
    t, τ = range(0, 2; length = 40), exp10.(range(-1.5, 0.5; length = 24))
    Aexp = [exp(-tᵢ / τⱼ) for tᵢ in t, τⱼ in τ]
    lcurve_geometry_tests(Aexp, Aexp * collect(1.0:24) .+ 1e-3 .* sin.(1:40))

    # A diagonal problem with a strictly positive solution keeps its support for every μ, so the ω derivative check rests on a proof rather than on sampled signatures
    Adiag = diagm(exp10.(range(-1.0, 1.0; length = 12)))
    lcurve_geometry_tests(Adiag, Adiag * ones(12))
end

# The μ → 0 tail carries positive limiting curvature η⁴/(2qξ²), so positive curvature alone cannot reject it; the two-sided comparison against the flanking evaluations can.
# A single column with a residual floor δ isolates the mechanism: the support is {1} for every μ, so κ is smooth, and the plateau grows without bound as δ → 0.
@testset "L-curve tail rejection" begin
    A = [1.0; 0.0;;]
    for δ in (1e-1, 1e-2, 1e-4, 1e-6, 1e-8)
        b = [1.0, δ]
        (; mu) = DECAES.lsqnonneg_lcurve(A, b)
        κ(t) = (w = DECAES.lsqnonneg_tikh_work(A, b); DECAES.solve!(w, exp(t)); DECAES.curvature(log, w))

        # R(μ) = δ² + μ⁴/(1+μ²)², so the plateau is reached once μ ≪ √δ, where κ → 1/(2δ²) analytically: a positive attractor, and an unboundedly large one.
        # Only its sign and scale are asserted, since evaluating κ against a residual pinned at δ² is prone to cancellation.
        @test κ(log(δ) / 2 - 5) > 1

        # Either the documented unregularized fallback, or a strict local maximum resolved well above `xtol`
        @test mu == 0 || κ(log(mu)) > max(κ(log(mu) - 1e-3), κ(log(mu) + 1e-3))
    end
end

# A strict local maximum of κ does not exclude the μ → 0 tail. Here N₀b₂/a² exceeds 2, so the linear term of the tail expansion is positive and a strict maximum is forced at small μ, on a support that never changes.
# The tail maximum has enormous curvature and almost no turning; the broad maximum is the L-curve elbow. The search must not select the tail one.
@testset "L-curve multi-component tail maximum" begin
    A = [1.0 0 0; 0 √1000 0; 0 0 √1000; 0 0 0]
    λ = [1.0, 1000.0, 1000.0]
    function geom(t, δ)
        ρ = exp(2t)
        x = λ ./ (λ .+ ρ)
        return DECAES.lcurve_geometry(δ^2 + sum(@. (ρ * x / λ)^2 * λ), sum(abs2, x), sum(@. x^2 / (λ + ρ)), exp(t))
    end

    for (δ, t_tail, κ_tail) in ((1e-1, -6.259206, 4.491026e2), (1e-2, -10.863286, 4.491018e4))
        # The tail feature is real: a strict local maximum with curvature orders above the broad one, and turning orders below it
        κ₋, κ₀, κ₊ = geom(t_tail - 1e-3, δ)[1], geom(t_tail, δ)[1], geom(t_tail + 1e-3, δ)[1]
        @test κ₀ > κ₋ && κ₀ > κ₊
        @test κ₀ ≈ κ_tail rtol = 1e-5
        @test geom(t_tail, δ)[2] < 1e-2 < geom(0.7783, δ)[2] # ω separates them where κ does not
        b = [1.0, √1000, √1000, δ]
        (; x, mu) = DECAES.lsqnonneg_lcurve(A, b)

        # The production invariant, which any accepted corner must satisfy however the search changes
        @test mu == 0 || sum(abs2, A * x - b) / (mu^2 * sum(abs2, x)) <= 1.001 * DECAES.LCURVE_SLOPE_MAX_DEFAULT
        @test mu == 0 || abs(log(mu) - t_tail) > 1 # never this tail maximum
    end
end

# On a fixed passive set the strict tail maximum has an exact high-SNR limit. With N = ‖x₀‖², a = x₀ᵀG⁻¹x₀, b₂ = x₀ᵀG⁻²x₀ and D = Nb₂ - 2a² > 0,
#       ρ* → (D/aN³)R₀²,   R₀|S*| → aN²/D,   ω*/R₀ → 2D/(aN²),
# so |S*| diverges and ω* vanishes as the residual floor closes: every finite slope guard eventually rejects the asymptotic μ → 0 collapse. A finite-residual admissible maximum is therefore a genuine bend, not collapse.
@testset "L-curve high-SNR tail asymptotics" begin
    TE, m = 10e-3, 48
    T2s = [50e-3, 80e-3, 120e-3]
    A = [exp(-i * TE / T2) for i in 1:m, T2 in T2s]
    x₀ = ones(3)
    G = A'A
    N, a, b₂ = 3.0, x₀' * (G \ x₀), sum(abs2, G \ x₀)
    Dc = N * b₂ - 2a^2
    @test Dc > 0 # the rising-tail condition
    r = qr(A, ColumnNorm()).Q[:, end]
    r = all(A * x₀ .+ 0.1 .* r .> 0) ? r : -r

    function tailmax(δ)
        b = A * x₀ .+ δ .* r
        function g(t)
            w = DECAES.lsqnonneg_tikh_work(A, b)
            DECAES.solve!(w, exp(t))
            Nv = DECAES.seminorm_sq(w)
            return Nv == 0 ? (-Inf, 0.0, Inf) : (DECAES.lcurve_geometry(DECAES.resnorm_sq(w), Nv, DECAES.gradient_temps(w).xᵀB⁻¹x, exp(t))..., DECAES.resnorm_sq(w) / (exp(2t) * Nv))
        end
        ts = range(-18.0, 2.0; length = 4001)
        i = argmax([g(t)[1] for t in ts])
        lo, hi = ts[i-1], ts[i+1]
        for _ in 1:100
            p, q = hi - (hi - lo) / Base.MathConstants.φ, lo + (hi - lo) / Base.MathConstants.φ
            g(p)[1] > g(q)[1] ? (hi = q) : (lo = p)
        end
        t = (lo + hi) / 2
        return (exp(2t), g(t)...)
    end

    # Convergence to the predicted constants, over two decades of residual floor
    for δ in (1e-2, 3e-3, 1e-3)
        R₀ = δ^2
        ρ★, _, ω★, s★ = tailmax(δ)
        @test ρ★ / R₀^2 ≈ Dc / (a * N^3) rtol = 1e-2
        @test R₀ * s★ ≈ a * N^2 / Dc rtol = 1e-2
        @test ω★ / R₀ ≈ 2Dc / (a * N^2) rtol = 1e-2
    end

    # The guard's transition is where |S*| crosses τ, near R₀ = aN²/(τD)
    @test tailmax(0.1)[4] < DECAES.LCURVE_SLOPE_MAX_DEFAULT < tailmax(0.03)[4]

    # δ = 0.1 is a finite-residual bend the guard admits, and the search localizes it; smaller floors are rejected and fall back
    for (δ, admissible) in ((0.1, true), (0.03, false), (0.01, false))
        b = A * x₀ .+ δ .* r
        (; mu) = DECAES.lsqnonneg_lcurve(A, b)
        ρ★ = tailmax(δ)[1]
        if admissible
            @test mu > 0
            @test log(mu) ≈ log(√ρ★) atol = 1e-3
        else
            @test mu == 0
        end
    end
end

# The same mechanism on the production basis, where the search must select the elbow rather than fall back.
# N₀b₂/a² = 1 + Var_p(λ⁻¹)/E_p[λ⁻¹]² over p = (Qᵀx₀)²/N₀ measures how far κ climbs off the plateau; a coherent exponential family exceeds the threshold of 2 by orders, where the near-orthogonal basis above barely clears it.
@testset "L-curve tail maximum on an EPG decay basis" begin
    o = DECAES.mock_t2map_opts(Float64; MatrixSize = (1, 1, 1), nTE = 48, nT2 = 40, Silent = true)
    A = DECAES.epg_decay_basis(DECAES.restructure(DECAES.default_epg_parameters(o), (; α = deg2rad(180.0))), DECAES.T2_component_times(o))
    P, x₀ = A[:, round.(Int, range(4, o.nT2 - 3; length = 6))], ones(6)
    G = P'P
    N₀, a, b₂ = sum(abs2, x₀), x₀' * (G \ x₀), sum(abs2, G \ x₀)
    @test N₀ * b₂ > 2 * a^2 # the rising-tail condition, far from tight on a coherent basis
    r = qr(P, ColumnNorm()).Q[:, end]
    r = all(P * x₀ .+ 0.1 .* r .> 0) ? r : -r
    for δ in (0.1, 0.03, 0.01)
        b = P * x₀ .+ δ .* r
        (; x, mu) = DECAES.lsqnonneg_lcurve(A, b)
        @test mu > 0 # a genuine admissible elbow, unlike the toy basis above
        @test sum(abs2, A * x - b) / (mu^2 * sum(abs2, x)) <= 1.001 * DECAES.LCURVE_SLOPE_MAX_DEFAULT
    end
end

# `Ptol` bounds the solution and not only its norm. Monotonicity of ∂F gives (ρ₁+ρ₂)x₁ᵀx₂ ≥ ρ₁N₁ + ρ₂N₂, hence ‖x₁-x₂‖² ≤ (ρ₂-ρ₁)/(ρ₁+ρ₂)·(N₁-N₂), across active-set changes.
@testset "L-curve Ptol solution-diameter bound" begin
    for _ in 1:200
        m, n = rand(4:40), rand(4:40)
        A, b = rand(m, n), rand(m)
        ρ₁, ρ₂ = sort(exp.(2 .* (4 .* randn(2) .- 3)))
        ρ₁ == ρ₂ && continue
        x₁, x₂ = DECAES.lsqnonneg_tikh(A, b, √ρ₁), DECAES.lsqnonneg_tikh(A, b, √ρ₂)
        N₁, N₂ = sum(abs2, x₁), sum(abs2, x₂)
        @test sum(abs2, x₁ .- x₂) <= (ρ₂ - ρ₁) / (ρ₁ + ρ₂) * (N₁ - N₂) + 1e-12
        N₂ > 0 && @test norm(x₁ - x₂) <= √(1 - N₂ / N₁) * norm(x₁) + 1e-12
    end
end

@testset "L-curve geometry (dimensionless form)" begin
    # The invariant 0 < z ≤ 2 follows from B ⪰ ρI.
    for _ in 1:100
        A, b, μ = rand(40, 30), rand(40), exp(4 * randn())
        w = DECAES.lsqnonneg_tikh_work(A, b)
        DECAES.solve!(w, μ)
        N = DECAES.seminorm_sq(w)
        N == 0 && continue
        @test 0 < 2 * μ^2 * DECAES.gradient_temps(w).xᵀB⁻¹x / N <= 2 + 1e-12
    end

    # ω = 2du with u = c - z(c+d) ≤ c and c² + d² = 1, so positive curvature bounds the turning by 2cd ≤ 1, and |S| = c/d bounds it by 2c²/|S| ≤ 2/|S|
    for _ in 1:100
        ξ², η², μ = exp(8 * randn()), exp(8 * randn()), exp(4 * randn())
        q = (η² / μ^2) * rand()
        κ, ω = DECAES.lcurve_geometry(ξ², η², q, μ)
        (isfinite(κ) && isfinite(ω) && κ > 0) || continue
        @test 0 < ω <= 1
        @test ω <= 2 / (ξ² / (μ^2 * η²)) + 1e-12
    end
end

# An exact fit returns the unregularized solution by convention, not because the criterion has no corner: the L-curve runs off to X = -∞ but its curvature can still have a finite interior maximum.
@testset "L-curve exact fit is a convention" begin
    A = [0.1 0.0; 0.0 1.0]
    b = A * [0.1, 0.1]
    @test DECAES.lsqnonneg_lcurve(A, b).mu == 0

    κ(t) = (w = DECAES.lsqnonneg_tikh_work(A, b); DECAES.solve!(w, exp(t)); DECAES.curvature(log, w))
    @test κ(-1.675) > κ(-1.8) && κ(-1.675) > κ(-1.55) # a strict interior maximum near μ = 0.187
    @test κ(-1.675) > 0
end

# The slope guard confines every admissible corner to an exact interval. Both bounds follow from monotonicity and KKT alone, and the second upper bound is the tighter one whenever ξ²₀/‖b‖² < τ/(τ+2).
@testset "L-curve admissible domain" begin
    τ = DECAES.LCURVE_SLOPE_MAX_DEFAULT
    for _ in 1:50
        m, n = rand(8:48), rand(8:48)
        A, b = rand(m, n), abs.(randn(m))
        x₀ = DECAES.lsqnonneg(A, b)
        R₀, N₀ = sum(abs2, A * x₀ - b), sum(abs2, x₀)
        (R₀ <= 0 || N₀ <= 0) && continue
        t₀ = (log(R₀) - log(N₀)) / 2
        Bn, C = sum(abs2, b), sum(cᵢ -> max(cᵢ, 0.0)^2, A' * b)
        ρmax = C * min(τ / R₀, (τ + 2) / Bn)

        # Admissibility forces ξ²₀ ≤ (τ/4)‖Ax₀‖² ≤ (τ/4)√(CN₀), which places the seed inside the interval for every 1 < τ < 6 + 2√17 unless no point of the path is admissible at all
        R₀ / Bn <= τ / (τ + 4) && @test t₀ < log(ρmax) / 2
        for t in range(t₀ - 6, t₀ + 20; length = 40)
            w = DECAES.lsqnonneg_tikh_work(A, b)
            DECAES.solve!(w, exp(t))
            N = DECAES.seminorm_sq(w)
            N == 0 && continue
            @test exp(2t) * N <= (Bn - R₀) / 4 + 1e-9 # ρη² ≤ ‖Ax₀‖²/4, from unregularized KKT
            DECAES.resnorm_sq(w) / (exp(2t) * N) <= τ || continue
            @test t >= t₀ - log(τ) / 2 - 1e-9
            @test exp(2t) <= ρmax * (1 + 1e-9)
        end

        # The search never returns a point outside its own admissible domain
        (; mu) = DECAES.lsqnonneg_lcurve(A, b)
        mu > 0 && @test t₀ - log(τ) / 2 - 1e-9 <= log(mu) <= log(ρmax) / 2 + 1e-9
    end
end

# The sweep ranks brackets by tangent rotation, which is exact only because the tangent direction is continuous across an active-set transition while κ is not.
# dη²/dρ = -2q and dξ²/dρ = 2ρq share the factor q, which jumps, so dP/dρ is discontinuous while the direction (ρη², -ξ²) is not. Hence θ is continuous and θᵢ₊₁ - θᵢ is the exact net signed rotation of an interval, whatever transitions it contains.
# The transition is placed exactly rather than searched for: a support mask need not vary monotonically in μ, so bisecting the predicate sig(t) == sig(t₋) is not guaranteed to isolate one.
# With AᵀA = [1 1/2; 1/2 1] and Aᵀb = [1, 2/5], the support {1} carries x = (4/5, 0) with the second component's dual 2/5 - (1/2)(4/5) reaching zero at ρ = 1/4, where it enters at x₂ = 0.
# x is unchanged there, so ξ² and η² are continuous, while q = x_PᵀB_P⁻¹x_P jumps from (4/5)²/(5/4) = 0.512 to (4/5)²(5/4)/((5/4)² - (1/2)²) ≈ 0.6095.
@testset "L-curve tangent is continuous where curvature is not" begin
    A = [1.0 0.5; 0.0 sqrt(0.75)] # AᵀA = [1 1/2; 1/2 1]
    b = A' \ [1.0, 0.4]
    work = DECAES.lsqnonneg_lcurve_work(A, b)
    DECAES.reset_cache!(work.nnls_prob_smooth_cache)
    DECAES.solve_unreg!(work.nnls_prob, work.nnls_prob_seed)
    p(ρ) = (DECAES.nnls_gram_setup!(work); DECAES.lcurve_point(work, sqrt(ρ)))
    ρ₋, ρ₊ = 0.25 - 1e-9, 0.25 + 1e-9
    p₋, p₊ = p(ρ₋), p(ρ₊)
    @test p₊.sig != p₋.sig # the second component has entered, so this is a genuine transition
    @test norm(p₊.P - p₋.P) < 1e-7 # ξ² and η² carry across
    @test DECAES.tangent_angle(p₊, log(ρ₊) / 2) ≈ DECAES.tangent_angle(p₋, log(ρ₋) / 2) atol = 1e-7 # and so does the tangent direction
    @test abs(p₊.κ - p₋.κ) > 1e-2 # while κ jumps, by orders more than either of those moved
end

# The sweep ranks brackets by the net rotation of the tangent, which `tangent_angle` reads off the Tikhonov identity θ = -arctan(ξ²/(μ²η²)); only an actual Tikhonov path makes that the curve's tangent.
# A = [diagm(γ); 0ᵀ] with b = [γ .* x; δ] keeps every component of x_ρ = γ²x/(γ² + ρ) positive, so NNLS coincides with ridge on one branch and ξ², η², q are exact.
function build_tikhonov_corner_cached_fun(γ, x, δ, ::Type{T} = Float64) where {T}
    function tikhonov_lcurve_point(t)
        ρ = exp(2t)
        xᵨ = (γ .^ 2 .* x) ./ (γ .^ 2 .+ ρ)
        ξ² = sum(((γ .* x .* ρ) ./ (γ .^ 2 .+ ρ)) .^ 2) + δ^2
        η² = sum(abs2, xᵨ)
        q = sum((xᵨ .^ 2) ./ (γ .^ 2 .+ ρ))
        return LCurveCornerPoint(SA{T}[log(ξ²), log(η²)], DECAES.lcurve_geometry(ξ², η², q, exp(t))..., UInt128(1))
    end
    f = CachedFunction(tikhonov_lcurve_point, GrowableCache{T, LCurveCornerPoint{T}}(64, isapprox))
    return LCurveCornerCachedFunction(f, GrowableCache{Int, LCurveCornerState{T}}(64))
end

# Two singular values two decades apart put a single admissible corner at t✶ ≈ -3.4384, where |S| ≈ 0.081.
@testset "L-curve sweep fallback" begin
    f() = build_tikhonov_corner_cached_fun([1.0, 1e-2], [1.0, 1.0], 1e-6)
    sweep(g, L, U, τ; nsweep = 32) = DECAES.lcurve_sweep!(g, L, U, log(τ); nsweep, max_candidates = 8, xtol = 1e-6, Ptol = 1e-6, max_backtrack = 4)
    t✶ = -3.4384

    # The curve is unimodal on [-6, -1], so any bracket containing the corner certifies it
    @test sweep(f(), -6.0, -1.0, 1e6) ≈ t✶ atol = 1e-4
    @test sweep(f(), -6.0, -1.0, 1e6; nsweep = 2) |> isnan # two samples span one interval, leaving no interior sample to bracket a maximum

    # Only certified corners are returned, never a sample: on a grid whose nearest point is 0.29 away the answer is still localized to `xtol`
    g = f()
    @test sweep(g, -6.0, -1.0, 1e6; nsweep = 8) ≈ t✶ atol = 1e-4
    @test g(t✶).κ > g(t✶ - 1e-3).κ && g(t✶).κ > g(t✶ + 1e-3).κ # and it is a strict maximum of the exact curvature

    # A domain lying entirely to one side of the corner leaves the sampled sequence monotone, and a monotone sequence has no interior maximum to certify
    @test sweep(f(), -1.0, 2.0, 1e6) |> isnan

    # The slope guard still applies to whatever the sweep finds: the corner has |S| ≈ 0.081, so any τ below that rejects it
    @test sweep(f(), -6.0, -1.0, 0.05) |> isnan
end

# Three singular values two decades apart put admissible corners at t ≈ -8.0588, where |S| ≈ 0.042, and t ≈ -3.4385, where |S| ≈ 0.081.
# Every interval of largest tangent rotation surrounds the right corner, so a slope guard admitting only the left one forces the sweep past its top-ranked candidates.
@testset "L-curve sweep reaches a lower-ranked candidate" begin
    sweep(τ) = DECAES.lcurve_sweep!(build_tikhonov_corner_cached_fun([1.0, 1e-2, 1e-4], [1.0, 1.0, 1.0], 1e-9), -11.0, -1.0, log(τ); nsweep = 32, max_candidates = 8, xtol = 1e-6, Ptol = 1e-6, max_backtrack = 4)
    @test sweep(1e6) ≈ -3.4385 atol = 1e-3 # unguarded, the top-ranked candidate certifies
    @test sweep(0.06) ≈ -8.0588 atol = 1e-3 # the top-ranked candidates are all rejected, and a later one certifies
    @test sweep(0.03) |> isnan # neither corner is admissible
end

# A residual at the level of the arithmetic is treated as an exact fit: ξ²₀ ≤ ε‖b‖² reads ‖Ax₀ - b‖ ≤ √ε‖b‖.
@testset "L-curve near-exact fit policy" begin
    for (δ, exact) in ((√eps() / 2, true), (2 * √eps(), false))
        A, b = [1.0; 0.0;;], [1.0, δ]
        work = DECAES.lsqnonneg_lcurve_work(A, b)
        (; mu, chi2) = DECAES.lsqnonneg_lcurve!(work)
        if exact
            @test mu == 0 && chi2 == 1 # returned without evaluating the curve
            @test length(work.lcurve_point_cache) == 0
        else
            @test length(work.lcurve_point_cache) > 0 # the curve is searched normally
        end
    end
end

# ρη² ≤ ‖Ax₀‖²/4 holds everywhere on the path, so |S| ≥ 4ξ²₀/(‖b‖² - ξ²₀) and ξ²₀/‖b‖² > τ/(τ+4) certifies that no point is admissible.
# For A = I the path is x_ρ = b₊/(1+ρ) and the bound is nearly attained: min|S| = 2√(u(1+u)) + 2u = 4u + 1 - 1/(4u) + O(u⁻²) against a guarantee of 4u, for u = ‖b₋‖²/‖b₊‖².
@testset "L-curve no-corner certificate" begin
    τ = DECAES.LCURVE_SLOPE_MAX_DEFAULT
    A = Matrix(1.0 * LinearAlgebra.I, 8, 8)
    for c in (0.1, 0.5, 1.0, 2.0, 5.0)
        b = [fill(-1.0, 7); c]
        R₀, B₀ = sum(abs2, min.(b, 0.0)), sum(abs2, b)
        w = DECAES.lsqnonneg_tikh_work(A, b)
        smin = minimum(range(-8.0, 8.0; length = 2001)) do t
            DECAES.solve!(w, exp(t))
            η² = DECAES.seminorm_sq(w)
            return η² == 0 ? Inf : DECAES.resnorm_sq(w) / (exp(2t) * η²)
        end
        @test 4 * R₀ / (B₀ - R₀) <= smin * (1 + 1e-9)

        work = DECAES.lsqnonneg_lcurve_work(A, b)
        (; mu) = DECAES.lsqnonneg_lcurve!(work)
        if R₀ / B₀ > τ / (τ + 4)
            @test smin > τ # no point of the path is admissible
            @test mu == 0 # so the unregularized solution is returned
            @test length(work.lcurve_point_cache) == 0 # without evaluating the curve at all
        else
            @test length(work.lcurve_point_cache) > 0
        end
    end
end

# Past 128 columns the active-set digest is no longer an exact mask, so equality is a probabilistic statement. The search must still terminate and return a valid solution.
@testset "lsqnonneg_lcurve (n > 128)" begin
    A, b = rand(64, 200), abs.(randn(64))
    @test allunique(DECAES.NNLS.column_digest.(1:1000)) # distinct columns take distinct digests well past the exactly representable range
    (; x, mu, chi2) = DECAES.lsqnonneg_lcurve(A, b)
    @test all(>=(0), x) && isfinite(mu) && mu >= 0
    @test mu == 0 || isapprox(x, DECAES.lsqnonneg_tikh(A, b, mu); rtol = 1e-12, atol = 1e-12)
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
        @test x ≈ DECAES.lsqnonneg_tikh(A, b, mu) rtol = 1e-12 atol = 1e-12 * norm(b)
        @test chi2 >= 1 - √eps() # res²(μ)/res²(0) ≥ 1 up to roundoff between the two evaluation paths

        w = DECAES.lsqnonneg_tikh_work(A, b)
        DECAES.solve!(w, mu)
        @test DECAES.curvature(log, w) > 0
        η² = sum(abs2, x)
        @test η² == 0 || sum(abs2, A * x - b) / (η² * mu^2) < 1.001 * DECAES.LCURVE_SLOPE_MAX_DEFAULT
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
        @test x ≈ DECAES.lsqnonneg_tikh(A, b, mu) rtol = 1e-12 atol = 1e-12 * norm(b)

        # Stationarity of the minimum-product criterion: at the selected μ the log-log L-curve tangent slope is -1, so the balance point res² = μ²‖x‖² holds with |log S| bounded by the local slope of g times the Brent abscissa tolerance.
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
function reginska_expdecay_data(m, n, noise = 1e-3)
    t = range(0, 2; length = m)
    τ = exp10.(range(-1.5, 0.5; length = n))
    A = [exp(-tᵢ / τⱼ) for tᵢ in t, τⱼ in τ]
    x = zeros(n)
    for _ in 1:3
        x[rand(1:n)] += rand()
    end
    return A, A * x .+ noise .* randn(m)
end

function reginska_log_abs_slope(A, b, logμ)
    x = DECAES.lsqnonneg_tikh(A, b, exp(logμ))
    res², η² = sum(abs2, A * x - b), sum(abs2, x)
    return η² == 0 ? Inf : log(res²) - log(η²) - 2 * logμ
end

function reginska_leftmost_downcrossing(f_log_abs_slope, A, b, logμ_grid = range(-8, 2; length = 1000); atol = 1e-6)
    prevpos, prevl = false, first(logμ_grid)
    for logμ in logμ_grid
        g = f_log_abs_slope(A, b, logμ)
        if prevpos && g <= 0
            lo, hi = prevl, logμ
            while hi - lo > atol
                mid = (lo + hi) / 2
                if f_log_abs_slope(A, b, mid) > 0
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
        lc = reginska_leftmost_downcrossing(reginska_log_abs_slope, A, b)
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

# Positive Lasso: x_μ = argmin_{x ≥ 0} ‖Ax - b‖² + μ‖x‖₁, whose KKT conditions are x ≥ 0, d = Aᵀ(b - Ax) - μ/2 ≤ 0, and x ⊙ d = 0
lasso_obj(A, b, x, μ) = sum(abs2, A * x - b) + μ * sum(x)
lasso_dual(A, b, x, μ) = A' * (b - A * x) .- μ / 2
lasso_regparam_max(A, b) = 2 * max(0, maximum(A' * b))

# Gap against the dual max_y bᵀy - ‖y‖²/2 subject to Aᵀy ≤ (μ/2)𝟙, whose feasible point y = θr carries θ = min(1, μ / (2 max_j Aⱼᵀr)).
# Weak duality bounds the distance of ½‖Ax - b‖² + (μ/2)𝟙ᵀx from its optimum, and complementarity makes the bound vanish at the solution, providing an independent quantitative suboptimality bound alongside the KKT conditions, which convexity already makes a global certificate.
function lasso_gap(A, b, x, μ)
    r, λ = b - A * x, μ / 2
    θ = maximum(A' * r; init = zero(eltype(r))) > λ ? λ / maximum(A' * r) : one(eltype(r))
    return (sum(abs2, r) / 2 + λ * sum(x)) - (θ * dot(b, r) - θ^2 * sum(abs2, r) / 2)
end

# Global minimum by exhaustive support search over linearly independent supports: a minimizer of this polyhedral problem has an extreme-point representation whose active columns are independent, so some enumerated support attains the global objective value, though not necessarily the one the solver returns
function lasso_brute(A, b, μ)
    n = size(A, 2)
    f★, x★ = Inf, zeros(n)
    for mask in 0:(1<<n-1)
        S = [j for j in 1:n if mask & (1 << (j - 1)) != 0]
        AS = A[:, S]
        rank(AS) < length(S) && continue
        xS = (AS' * AS) \ (AS' * b .- μ / 2)
        all(>(0), xS) || continue
        x = zeros(n)
        x[S] .= xS
        f = lasso_obj(A, b, x, μ)
        f < f★ && ((f★, x★) = (f, x))
    end
    return f★, x★
end

# Certify the KKT and duality-gap certificate of the fixed-μ solver, with ground truth evaluated in Double64.
function lasso_certify(A, b, x, μ; rtol = 1e-12)
    A, b, x, μ = Double64.(A), Double64.(b), Double64.(x), Double64(μ)
    d, scale = lasso_dual(A, b, x, μ), maximum(abs, A' * b)
    @test all(>=(0), x)
    @test maximum(d[x .== 0]; init = -Inf) <= rtol * scale
    @test maximum(abs, d[x .> 0]; init = 0.0) <= rtol * scale
    return μ > 0 && @test lasso_gap(A, b, x, μ) <= rtol * sum(abs2, b)
end

function lsqnonneg_lasso_tests(m, n)
    A, b = rand_NNLS_data(m, n)
    work = DECAES.lsqnonneg_lasso_work(A, b)
    μmax = lasso_regparam_max(A, b)
    scale = maximum(abs, A' * b)
    R, S = Float64[], Float64[]

    for μrel in (1e-8, 1e-3, 0.1, 0.5, 0.9, 1.0)
        μ = μrel * μmax
        x = DECAES.lsqnonneg_lasso!(work, μ)
        d = lasso_dual(A, b, x, μ)
        @test minimum(x) >= 0
        @test maximum(d[x .== 0]; init = -Inf) <= 1e-12 * scale
        @test maximum(abs, d[x .> 0]; init = 0.0) <= 1e-12 * scale
        @test DECAES.ncomponents(work) == count(>(0), x)
        @test DECAES.resnorm_sq(work) ≈ sum(abs2, A * x - b) rtol = 1e-12 atol = 1e-12 * sum(abs2, b)

        # Convexity makes the KKT conditions a global optimality certificate already; the duality gap adds an independent quantitative bound on the objective, computed from different quantities.
        # Its dual point collapses to zero at μ = 0 for any positive dual residual at all, so the certificate is meaningful only for μ > 0.
        μ > 0 && @test lasso_gap(A, b, x, μ) <= 1e-12 * sum(abs2, b)

        push!(R, sum(abs2, A * x - b))
        push!(S, sum(x))
    end

    # ‖Ax_μ-b‖² is nondecreasing and ‖x_μ‖₁ nonincreasing in μ, which is what brackets the χ² root by construction
    @test all(>=(-1e-12 * maximum(R)), diff(R))
    @test all(<=(1e-12 * maximum(S; init = 0.0)), diff(S))

    # x = 0 from the threshold onwards, since the KKT conditions at x = 0 read Aᵀb ≤ (μ/2)𝟙
    @test all(==(0), DECAES.lsqnonneg_lasso(A, b, μmax))
    μmax > 0 && @test any(>(0), DECAES.lsqnonneg_lasso(A, b, 0.999 * μmax))

    # μ = 0 is the unregularized problem, whose residual norm is unique even where its minimizer is not
    DECAES.lsqnonneg_lasso!(work, 0.0)
    @test DECAES.resnorm_sq(work) ≈ sum(abs2, A * DECAES.lsqnonneg(A, b) - b) rtol = 1e-12 atol = 1e-12 * sum(abs2, b)

    @inferred DECAES.lsqnonneg_lasso!(work, μmax / 2)
end

@testset "lsqnonneg_lasso" begin
    for (m, n) in NNLS_SIZES
        lsqnonneg_lasso_tests(m, n)
    end
end

# The separable case A = I, where the objective splits into ‖xⱼ-bⱼ‖² + μxⱼ over each coordinate, pins the convention of μ against a factor of two
@testset "lsqnonneg_lasso separable oracle" begin
    n = 8
    A = Matrix{Float64}(LinearAlgebra.I, n, n)
    b = collect(range(0.1, 1.0; length = n))
    for μ in (0.0, 0.05, 2 * b[3], 0.5, 1.0, 3.0)
        @test DECAES.lsqnonneg_lasso(A, b, μ) ≈ max.(b .- μ / 2, 0)
    end
end

# Comparison with exhaustive support search. The `n > m` cases saturate the active set at rank `m`, wherein a column already inside its span can still carry a positive dual,
# and so only the exchange of `resolve_dependency!` reaches the minimum. Such a column is appended before it is exchanged away, so the active set passes through `m + 1` columns.
@testset "lsqnonneg_lasso exhaustive support search" begin
    for (m, n) in ((2, 3), (2, 5), (3, 4), (3, 7), (4, 8), (5, 5), (8, 6)), _ in 1:3
        A, b = rand_NNLS_data(m, n)
        μmax = lasso_regparam_max(A, b)
        μmax == 0 && continue
        for μrel in (1e-6, 1e-3, 0.1, 0.5, 0.9)
            μ = μrel * μmax
            x = DECAES.lsqnonneg_lasso(A, b, μ)
            f★, _ = lasso_brute(A, b, μ)
            @test lasso_obj(A, b, x, μ) <= f★ * (1 + 1e-14) + 1e-14
        end
    end
end

# Exactly dependent columns are the case pure ℓ¹ regularization cannot factor away, having no diagonal shift with which to make the Hessian definite.
# The dependence coefficients decide the outcome: a column reproduced by others as A_k = A_{P\k} c is worth entering exactly when 𝟙ᵀc > 1, since it then carries the same fitted vector at strictly smaller 𝟙ᵀx.
@testset "lsqnonneg_lasso exact column dependence" begin
    a₁, a₂ = [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]
    b = 2 * a₁ + 3 * a₂

    # A duplicate has 𝟙ᵀc = 1 and buys nothing, leaving the split of its coefficient with the solver, so only the invariants of the minimizer are pinned
    A = [a₁ a₂ a₁]
    x = DECAES.lsqnonneg_lasso(A, b, 1e-6)
    @test A * x ≈ b rtol = 1e-5
    @test sum(x) ≈ 5 rtol = 1e-5

    # A scaled copy has 𝟙ᵀc = 2, so the fit moves onto it entirely and halves that part of 𝟙ᵀx
    A = [a₁ a₂ 2 * a₁]
    x = DECAES.lsqnonneg_lasso(A, b, 1e-6)
    @test x ≈ [0.0, 3.0, 1.0] rtol = 1e-5 atol = 1e-6

    # A sum of two columns likewise has 𝟙ᵀc = 2, and the fit moves onto it as far as nonnegativity of the columns it replaces allows
    A = [a₁ a₂ a₁+a₂]
    x = DECAES.lsqnonneg_lasso(A, b, 1e-6)
    @test x ≈ [0.0, 1.0, 2.0] rtol = 1e-5 atol = 1e-6

    for μ in (1e-3, 0.1, 1.0)
        f★, _ = lasso_brute(A, b, μ)
        @test lasso_obj(A, b, DECAES.lsqnonneg_lasso(A, b, μ), μ) <= f★ * (1 + 1e-14) + 1e-14
    end
end

# A column already inside the span of the active set is appended before the exchange that removes it, so a full-rank active set passes through m + 1 columns and the QR runs one reflector past the end of its m rows.
@testset "lsqnonneg_lasso rank saturation" begin
    A = [0.13217884 0.38673057 0.33919489; 0.87444120 0.41875302 0.08204501]
    b = [0.92680520, 0.62231336]
    for μ in (2.239389e-3, 1e-6, 1e-2, 0.1)
        x = DECAES.lsqnonneg_lasso(A, b, μ)
        f★, _ = lasso_brute(A, b, μ)
        @test lasso_obj(A, b, x, μ) <= f★ * (1 + 1e-14) + 1e-14
        lasso_certify(A, b, x, μ)
    end
end

# `NNLS.regparam_segment!` returns q = 𝟙ᵀG_PP⁻¹𝟙 and the end of the interval of μ over which `solve!` keeps the support it just solved on, where the path is affine and
#
#   ‖x_ν‖₁ = ‖x_μ‖₁ - (q/2)(ν - μ),
#   ‖Ax_ν - b‖² = ‖Ax_μ - b‖² + (q/4)(ν² - μ²).
#
# Checking both against independent solves exercises the two triangular solves, q itself, the leave events of the active coefficients and the inactive dual slopes at once.
@testset "NNLS.regparam_segment!" begin
    for (m, n) in ((6, 4), (12, 8), (8, 12), (24, 20)), _ in 1:4
        A, b = rand_NNLS_data(m, n)
        μmax = lasso_regparam_max(A, b)
        μmax == 0 && continue
        work = DECAES.lsqnonneg_lasso_work(A, b)

        for μrel in (0.05, 0.2, 0.5, 0.8)
            μ = μrel * μmax
            x = copy(DECAES.lsqnonneg_lasso!(work, μ))
            S = findall(>(0), x)
            isempty(S) && continue
            res², seminrm = sum(abs2, A * x - b), sum(x)
            q, μ_end = DECAES.NNLS.regparam_segment!(work, μ)

            @test q ≈ sum((A[:, S]' * A[:, S]) \ ones(length(S))) rtol = 1e-12
            @test μ < μ_end
            @test μ_end <= μmax * (1 + 1e-12) # the last leave event is μmax itself, reached here through 2xᵢ/uᵢ rather than through 2max_j Aⱼᵀb

            ν = (μ + μ_end) / 2
            y = DECAES.lsqnonneg_lasso(A, b, ν)
            @test findall(>(0), y) == S
            @test sum(y) ≈ seminrm - q * (ν - μ) / 2 rtol = 1e-12
            @test sum(abs2, A * y - b) ≈ res² + q * (ν - μ) * (ν + μ) / 4 rtol = 1e-12

            # The interval ends where the support changes, so the same support cannot survive past it
            @test findall(>(0), DECAES.lsqnonneg_lasso(A, b, μ_end * (1 + 1e-6))) != S
        end
    end
end

# The ℓ¹ passive solve carries no diagonal shift with which to condition A_P, so the families that stress ordinary NNLS stress it harder.
# Optimality is certified rather than compared against a reference: on a near-dependent support the normal equations are worse conditioned than the solver whose answer they would judge.
@testset "Adversarial lsqnonneg_lasso ($name)" for (name, data) in adversarial_NNLS_generators()
    for (m, n) in ((16, 8), (16, 16), (32, 24))
        A, b = data(m, n)
        μmax = lasso_regparam_max(A, b)
        for μrel in (1e-6, 1e-3, 0.1, 0.5, 0.9)
            x = DECAES.lsqnonneg_lasso(A, b, μrel * μmax)
            @test all(isfinite, x)
            lasso_certify(A, b, x, μrel * μmax; rtol = 1e-9)
        end

    end
end

# Check that chains of warm-started solves equal those of cold solves for a non-monotonic sequence of μs.
# The fitted vector Ax and penalty 𝟙ᵀx are the compared invariants.
@testset "lsqnonneg_lasso cold and warm solves agree" begin
    for (m, n) in ((4, 12), (12, 8), (13, 16), (32, 25))
        A, b = rand_NNLS_data(m, n)
        μmax = lasso_regparam_max(A, b)
        μmax == 0 && continue
        work = DECAES.lsqnonneg_lasso_work(A, b)
        DECAES.NNLS.reset!(work)
        for μrel in (0.1, 0.7, 0.3, 0.9, 0.5)
            μ = μrel * μmax
            x_warm = copy(DECAES.NNLS.solve!(work, μ)) # keeps the preceding active set
            x_cold = DECAES.lsqnonneg_lasso(A, b, μ)
            @test A * x_warm ≈ A * x_cold rtol = 1e-12 atol = 1e-12 * norm(b)
            @test sum(x_warm) ≈ sum(x_cold) rtol = 1e-12 atol = 1e-12
        end
    end
end

function test_lsqnonneg_gcv(m, n)
    A, b = rand_NNLS_data(m, n)
    work = DECAES.NNLSGCVRegProblem(A, b)
    logμ = randn()
    μ = exp(logμ)

    # Precompute the squared singular values for GCV
    LinearAlgebra.eigvals!(work)
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
    _gcv, ∇gcv = DECAES.gcv_and_dgcv_dlogμ!(work, logμ) # gcv_and_dgcv_dlogμ! calls `DECAES.solve!` internally
    @test _gcv == gcv # primals should match exactly
    @test ∇gcv ≈ ∇finitediff(_logμ -> DECAES.gcv!(work, _logμ), logμ, 1e-6) atol = 1e-3 rtol = 1e-3

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

@testset "GCV log-parameter gradient" begin
    A = [1.0 0.2 0.1; 0.1 2.0 0.3; 0.2 0.1 4.0; 0.3 0.2 0.1]
    b = [1.0, 0.8, 0.6, 0.4]
    work = DECAES.lsqnonneg_gcv_work(A, b)
    LinearAlgebra.eigvals!(work)
    logμ = log(0.2)
    _, ∇gcv = DECAES.gcv_and_dgcv_dlogμ!(work, logμ)
    @test ∇gcv ≈ ∇finitediff(_logμ -> DECAES.gcv!(work, _logμ), logμ, 1e-8) rtol = 1e-7
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
    search_prob = flip_angle_work.nnls_search_prob
    interp = DECAES.GriddedSpectrumInterpolator(search_prob.As, search_prob.∇As, DECAES.flip_angles(o))
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

            # The selected μ agrees only to the search tolerance: the exact path reaches the spectrum through `svdvals` and the interpolator's slices through `svd`, and a roundoff-level dof difference can move the Brent result by up to `atol` on a flat objective
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

# Scale covariance. Since ‖sAx − sb‖² + μ²‖x‖² = s²(‖Ax − b‖² + (μ/s)²‖x‖²), the Tikhonov path satisfies x_μ(sA, sb) = x_{μ/s}(A, b), so every selection rule must return s·μ* with x* and chi2 unchanged.
# The rules themselves are covariant: chi2 and mdp equate quantities that both scale as s², and the Regińska and L-curve criteria are invariant under the translation s induces on the log-log curve.
# `lsqnonneg_gcv` follows for the same reason, its criterion being covariant and its search interval derived from the data; see `gcv_bracket`.
@testset "scale covariance of reg methods" begin
    m, n = 48, 40
    t = range(0, 2; length = m)
    τ = exp10.(range(-1.5, 0.5; length = n))
    A = [exp(-tᵢ / τⱼ) for tᵢ in t, τⱼ in τ]
    x = zeros(n)
    x[[4, 17, 31]] .= (0.2, 0.5, 0.3)
    b = A * x .+ 1e-3 .* sin.(1:m)
    δ = 1e-2 * norm(b) # scaled alongside A and b, since the discrepancy level is a residual norm

    for (name, rtol, f) in [
        ("unreg", 1e-12, (A, b, s) -> (; mu = 0.0, x = DECAES.lsqnonneg(A, b), chi2 = 1.0)),
        ("chi2", 1e-3, (A, b, s) -> DECAES.lsqnonneg_chi2(A, b, 1.02)),
        ("mdp", 1e-3, (A, b, s) -> DECAES.lsqnonneg_mdp(A, b, s * δ)),
        ("reginska", 1e-9, (A, b, s) -> DECAES.lsqnonneg_reginska(A, b)),
        ("lcurve", 1e-9, (A, b, s) -> DECAES.lsqnonneg_lcurve(A, b)),
        ("gcv", 1e-3, (A, b, s) -> DECAES.lsqnonneg_gcv(A, b)),
    ]
        r = f(A, b, 1.0)
        for scale in (1e-9, 1e9)
            rs = f(scale .* A, scale .* b, scale)
            @test rs.mu / scale ≈ r.mu rtol = rtol atol = 1e-12
            @test rs.x ≈ r.x rtol = rtol atol = 1e-12
            @test rs.chi2 ≈ r.chi2 rtol = rtol
        end
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
            @test all(isnan.(DECAES.regparam.(work.cache[(count+1):N])))
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

# The μ-selection methods on ill-conditioned, rank-deficient, and degenerate inputs.
# Each method's Gram fast path has conditioning and iteration guards that fall back to the QR solve. They fire only on ill-conditioned inputs, which the strictly-positive random data of the per-method testsets above never produces.
# The returned x must be KKT-optimal for min_{x≥0} ‖Ax−b‖² + μ²‖x‖² at the returned μ, which at μ = 0 is the unregularized problem. Strong convexity makes the Double64 dual and complementarity certificate sufficient.
function verify_reg_kkt(A0, b0, x, mu)
    D64 = Double64
    A, b, x = D64.(A0), D64.(b0), D64.(x)
    w = A' * (b - A * x) .- D64(mu)^2 .* x # dual (negative half-gradient) of the Tikhonov objective
    ε = 1e-14 * max(1, norm(A' * b))
    @test all(>=(-1e-14), x) # primal feasibility
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
            @test isfinite(mu) && mu >= 0
            verify_reg_kkt(A, b, x, mu)
        end
    end
end
