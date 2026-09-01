Random.seed!(0) # reproducible randomized tests

# α-polish: the refinement of the flip-angle surrogate minimizer. The polish spends one true off-grid loss and gradient evaluation, minimizes both adjacent cubic Hermites, and returns the lex-minimum of true losses over the best evaluated node, α₀, and the candidate.
# The gradient is the envelope-theorem derivative g = 2(∂A/∂α·x)ᵀ(Ax − b), valid because x is the NNLS minimizer at α.
function alpha_polish_setup(::Type{T} = Float64) where {T}
    o = DECAES.mock_t2map_opts(T; MatrixSize = (1, 1, 1), nTE = 48, nT2 = 40, nRefAngles = 16, nRefAnglesMin = 5, Silent = true)
    θ = DECAES.default_epg_parameters(o)
    T2t = DECAES.T2_component_times(o)
    function loss_true(α, b)
        A = DECAES.epg_decay_basis(DECAES.restructure(θ, (; α)), T2t)
        x = DECAES.lsqnonneg(A, b)
        return sum(abs2, A * x - b)
    end
    function make_signal(α_true, noise = zero(T))
        A = DECAES.epg_decay_basis(DECAES.restructure(θ, (; α = α_true)), T2t)
        x = zeros(T, o.nT2)
        for _ in 1:rand(1:4)
            x[rand(1:o.nT2)] += rand(T)
        end
        b = A * x
        b ./= maximum(b)
        noise > 0 && (b .= max.(b .+ T(noise) .* randn(T, o.nTE), 0)) # noiseless signals never exercise the candidate-rejection path
        return b
    end
    function build(b)
        return DECAES.FlipAngleOptimizationWorkspace(o, zeros(T, o.nTE, o.nT2), copy(b))
    end
    function surrogate_minimizer(w)
        empty!(w.α_surrogate)
        DECAES.advance_warmstart!(w.nnls_search_prob)
        DECAES.reset!(w.α_searcher)
        DECAES.initialize!(w.α_surrogate, w.α_searcher; mineval = o.nRefAnglesMin, maxeval = o.nRefAngles)
        if DECAES.USE_DYADIC_REFINEMENT[]
            α, _ = DECAES.bisection_search(w.α_surrogate, w.α_searcher; maxeval = o.nRefAngles)
        else
            α, _ = DECAES.projected_search(w.α_surrogate, w.α_searcher; maxeval = o.nRefAngles)
        end
        return α[1]
    end
    return o, loss_true, make_signal, build, surrogate_minimizer
end

function test_alpha_polish()
    o, loss_true, make_signal, build, surrogate_minimizer = alpha_polish_setup()
    _, loss_true_D64, _, _, _ = alpha_polish_setup(Double64)
    for noise in (0.0, 0.02), _ in 1:20
        b = make_signal(deg2rad(100 + 75 * rand()), noise)

        # The search is deterministic, so a fresh workspace reaching only the search prefix yields the α₀ the polished run started from
        α₀ = surrogate_minimizer(build(b))

        wp = build(b)
        DECAES.optimize_flip_angle!(wp, o)
        αp = wp.α[]

        # Certificate: the returned angle's true loss never exceeds the best evaluated node's, whatever the interpolant proposed.
        # The allowance is roundoff between two independent NNLS solves, not slack in the selection, which compares its own evaluations exactly.
        @test loss_true(αp, b) <= (1 + 1e-10) * best_seen_loss(wp) + 1e-14
        @test loss_true(αp, b) <= (1 + 1e-10) * loss_true(α₀, b) + 1e-14
        @test π/2 < αp <= π # noisy signals routinely minimize at the grid's upper endpoint

        # Envelope-theorem soundness. Evaluated away from α₀ where f′ vanishes.
        α = α₀ + deg2rad(rand((-1, 1)) * rand(1:5))
        if deg2rad(100) < α < deg2rad(175)
            wg = build(b)
            surrogate_minimizer(wg)
            f, g, _ = DECAES.polish_loss_grad!(wg, α)
            @test f ≈ loss_true(α, b) rtol = 1e-6

            h, b_D64 = Double64(1e-12), Double64.(b)
            @test g ≈ (loss_true_D64(α + h, b_D64) - loss_true_D64(α - h, b_D64)) / 2h rtol = 1e-8
        end
    end
end

# Smallest loss among the nodes the search actually evaluated, which is the incumbent the final selection must beat
best_seen_loss(w) = minimum(w.α_surrogate.u[j] for j in eachindex(w.α_surrogate.u) if w.α_surrogate.seen[j])

# A signal whose Hermite candidate lands on a worse α: the sub-bracket minimizer sits where the interpolant undershoots the true loss, which the endpoint data cannot exclude.
# Pinned so the candidate-rejection path has a deterministic exercise; a random noisy signal reaches it only rarely.
const ALPHA_POLISH_REGRESSOR = [0.8422373004806307, 1.0091826247794164, 0.6428184716776855, 0.5474218056440463, 0.49007482076481235, 0.4414498947102849, 0.35792229991534513, 0.3959161871485733, 0.3530918960063172, 0.3394509594247156, 0.32341540148568765, 0.32801167912370516, 0.29491448589485336, 0.3307895905906243, 0.2917119077728464, 0.29840252787747223, 0.26000906747287317, 0.2785880979660586, 0.2364619558411132, 0.2922696232122398, 0.24858576438062197, 0.2676898045436556, 0.26618500265259215, 0.26734846712542065, 0.2869129895431217, 0.22680163431643924, 0.24080488853157997, 0.235545603116511, 0.2805689743797803, 0.25028792186470766, 0.2692806391269384, 0.2259881346943582, 0.2678662713428116, 0.2144992953881632, 0.24553142545692724, 0.24110889770675573, 0.22014554534697436, 0.2049518897846395, 0.24203924203691987, 0.22570841149516885, 0.22471961495387455, 0.17554128367607663, 0.2105506792087249, 0.1998381332750309, 0.23359528498137713, 0.1921590953698649, 0.19310486085739623, 0.2622939307433316]

function test_alpha_polish_guard()
    o, loss_true, _, build, surrogate_minimizer = alpha_polish_setup()
    b = ALPHA_POLISH_REGRESSOR

    wp = build(b)
    DECAES.optimize_flip_angle!(wp, o)
    α̂ = wp.α[]

    # The Hermite candidate is worse than α₀ here, so the selection must reject it and still beat every evaluated node
    @test loss_true(α̂, b) <= (1 + 1e-10) * best_seen_loss(wp) + 1e-14
    @test loss_true(α̂, b) <= (1 + 1e-10) * loss_true(surrogate_minimizer(build(b)), b) + 1e-14

    # Whichever candidate wins, the caller's basis must be the one at the returned angle, not at a rejected candidate
    @test wp.decay_basis ≈ DECAES.epg_decay_basis(DECAES.restructure(DECAES.default_epg_parameters(o), (; α = α̂)), DECAES.T2_component_times(o))
end

# At RefConAngle ≠ 180 the cosine series does not apply, so the polish builds A(α) by the EPG recurrence and takes derivative columns from the AD Jacobian on the support only.
# That backend is a direct evaluation of the same model, so it must agree with central differences of the true loss and satisfy the same certificate.
function test_alpha_polish_general_refcon()
    for β in (120.0, 150.0, 175.0)
        o = DECAES.mock_t2map_opts(Float64; MatrixSize = (1, 1, 1), nTE = 48, nT2 = 40, nRefAngles = 16, nRefAnglesMin = 5, RefConAngle = β, Silent = true)
        θ = DECAES.default_epg_parameters(o)
        T2t = DECAES.T2_component_times(o)
        function loss_true(α, b)
            A = DECAES.epg_decay_basis(DECAES.restructure(θ, (; α)), T2t)
            return sum(abs2, A * DECAES.lsqnonneg(A, b) - b)
        end

        # Differencing only, so the tolerance measures the AD gradient rather than the step size
        o_D64 = DECAES.mock_t2map_opts(Double64; MatrixSize = (1, 1, 1), nTE = 48, nT2 = 40, nRefAngles = 16, nRefAnglesMin = 5, RefConAngle = β, Silent = true)
        θ_D64, T2s_D64 = DECAES.default_epg_parameters(o_D64), DECAES.T2_component_times(o_D64)
        function loss_true64(α, b)
            A = DECAES.epg_decay_basis(DECAES.restructure(θ_D64, (; α)), T2s_D64)
            return sum(abs2, A * DECAES.lsqnonneg(A, b) - b)
        end

        for _ in 1:4
            A = DECAES.epg_decay_basis(DECAES.restructure(θ, (; α = deg2rad(100 + 75 * rand()))), T2t)
            x = zeros(o.nT2)
            for _ in 1:rand(1:4)
                x[rand(1:o.nT2)] += rand()
            end
            b = A * x
            b ./= maximum(b)
            b .= max.(b .+ 0.02 .* randn(o.nTE), 0)

            w = DECAES.FlipAngleOptimizationWorkspace(o, zeros(o.nTE, o.nT2), copy(b))
            @test w.α_polish_work !== nothing # the polish is constructed for every β
            DECAES.optimize_flip_angle!(w, o)
            α̂ = w.α[]

            @test π/2 < α̂ <= π
            @test loss_true(α̂, b) <= (1 + 1e-10) * best_seen_loss(w) + 1e-14

            # The AD envelope derivative matches central differences of the true loss, evaluated away from α̂ where f′ does not vanish
            α = α̂ + deg2rad(rand((-1, 1)) * rand(1:5))
            if deg2rad(100) < α < deg2rad(175)
                f, g, _ = DECAES.polish_loss_grad!(w, α)
                @test f ≈ loss_true(α, b) rtol = 1e-6
                h, b_D64 = Double64(1e-12), Double64.(b)
                @test g ≈ (loss_true64(α + h, b_D64) - loss_true64(α - h, b_D64)) / 2h rtol = 1e-8
            end
        end
    end
end

# The envelope second derivative comes from constrained variable projection, and its final −2qᵀG⁻¹q term is what distinguishes the profiled curvature from that of a frozen x.
# Dropping that term would still pass a value/gradient check, so it is verified against central differences of the exact envelope gradient.
function test_alpha_polish_hessian()
    o, loss_true, make_signal, build, surrogate_minimizer = alpha_polish_setup()
    _, _, _, build_D64, surrogate_minimizer_D64 = alpha_polish_setup(Double64)
    for _ in 1:10
        b = make_signal(deg2rad(100 + 75 * rand()), 0.02)
        w = build(b)
        α = surrogate_minimizer(w) + deg2rad(rand((-1, 1)) * rand(1:5)) # away from the minimizer, where f′ and f″ are both order one
        _, _, f″ = DECAES.polish_loss_grad!(w, α) # analytical, in Float64: this is the kernel under test

        # Central difference of the exact gradient, which is itself an envelope quantity requiring a fresh solve at each point. Only the differencing runs in Double64.
        h, b_D64 = Double64(1e-12), Double64.(b)
        w⁺, w⁻ = build_D64(b_D64), build_D64(b_D64)
        surrogate_minimizer_D64(w⁺)
        surrogate_minimizer_D64(w⁻)
        _, g⁺ = DECAES.polish_loss_grad!(w⁺, Double64(α) + h)
        _, g⁻ = DECAES.polish_loss_grad!(w⁻, Double64(α) - h)
        @test f″ ≈ (g⁺ - g⁻) / 2h rtol = 1e-8
    end
end

@testset "reconstructed EPG bases from outputs are consistent" begin
    for αdeg in (50.0, 90.0, 165.0, 180.0)
        o = DECAES.mock_t2map_opts(Float64; MatrixSize = (1, 1, 1), nTE = 32, nT2 = 20, SetFlipAngle = αdeg, SaveNNLSBasis = true, Silent = true)
        img = DECAES.mock_image(o)
        maps, _ = DECAES.redirect_to_devnull() do
            return DECAES.T2mapSEcorr(img, o)
        end
        @test maps["alpha"][1] ≈ αdeg
        θ = DECAES.restructure(DECAES.default_epg_parameters(o), (; α = deg2rad(αdeg)))
        @test maps["decaybasisset"] ≈ DECAES.epg_decay_basis(θ, DECAES.T2_component_times(o))
    end
end

@testset "EPGdecaycurve constructors are consistent" begin
    @test DECAES.EPGdecaycurve(32, 165.0, 9e-3, 50e-3, 1.0, 150.0) == DECAES.EPGdecaycurve(DECAES.EPGOptions((; ETL = 32, α = deg2rad(165.0), TE = 9e-3, T2 = 50e-3, T1 = 1.0, β = deg2rad(150.0))))
    @test DECAES.EPGdecaycurve(32, 165.0, 9e-3, 50e-3, 1.0) == DECAES.EPGdecaycurve(DECAES.EPGConstantFlipAngleOptions((; ETL = 32, α = deg2rad(165.0), TE = 9e-3, T2 = 50e-3, T1 = 1.0)))
end

@testset "α-polish" begin
    test_alpha_polish()
    test_alpha_polish_hessian()
    test_alpha_polish_guard()
    test_alpha_polish_general_refcon()
end

# When the polish problem is already solved, the T2 stage skips its own unregularized solve and takes that state instead.
# That is sound only if the basis really is A(α̂) and the workspace really holds its exact NNLS solution, so both halves are checked against an independent solve.
@testset "certified unregularized state" begin
    o, loss_true, make_signal, build = alpha_polish_setup()
    θ, T2t = DECAES.default_epg_parameters(o), DECAES.T2_component_times(o)
    ncertified = 0
    for _ in 1:20
        b = make_signal(deg2rad(100 + 75 * rand()), 0.02)
        w = build(b)
        DECAES.optimize_flip_angle!(w, o)

        (; nnls_work) = w.α_polish_work.prob
        DECAES.NNLS.issolved(nnls_work) || continue
        ncertified += 1

        @test w.decay_basis ≈ DECAES.epg_decay_basis(DECAES.restructure(θ, (; α = w.α[])), T2t)

        x = DECAES.lsqnonneg(w.decay_basis, b)
        @test DECAES.NNLS.solution(nnls_work) ≈ x rtol = 1e-10
        @test DECAES.NNLS.residual(nnls_work) ≈ b - w.decay_basis * x rtol = 1e-10 atol = 1e-14
        @test DECAES.NNLS.residualnorm(nnls_work) ≈ norm(b - w.decay_basis * x) rtol = 1e-10
        @test sort(DECAES.NNLS.components(nnls_work)) == findall(>(0), x)
    end
    @test ncertified > 0 # the adopted path must actually be reached
end

# The limiting returns μ = 0 and μ = ∞ carry their selected solution directly rather than through the regularized cache, which they never populate.
# Here b = Ax lies exactly in the nonnegative span of A, so the unregularized fit is exact and Ψ = ‖Ax_μ − b‖²‖x_μ‖² already attains its global minimum of zero at μ = 0.
# No positive balance point exists, and the selected solution is the unregularized one.
@testset "limiting regularization solution propagation" begin
    o = DECAES.mock_t2map_opts(Float64; MatrixSize = (1, 1, 1), nTE = 16, nT2 = 8, SetFlipAngle = 180.0, Reg = "reginska", SaveRegParam = true, Threaded = false, Silent = true)
    θ = DECAES.restructure(DECAES.default_epg_parameters(o), (; α = deg2rad(o.SetFlipAngle)))
    A = DECAES.epg_decay_basis(θ, DECAES.T2_component_times(o))
    x = zeros(o.nT2)
    x[3], x[6] = 0.4, 0.8
    b = A * x
    image = reshape(copy(b), 1, 1, 1, o.nTE)
    maps, dist = DECAES.T2mapSEcorr(image, o)

    @test maps["mu"][1] == 0
    @test vec(dist) ≈ DECAES.lsqnonneg(A, b) rtol = 1e-10 atol = 1e-12
end

# The limiting returns are precisely the ones the regularized cache never writes, so the failure needs a reused workspace: a first voxel that populates the cache, then a second that must not read it.
# Each returns a vector the cache does not own, μ = 0 the unregularized workspace and μ = ∞ the shared zero solution.
@testset "limiting regularization returns across a reused workspace" begin
    limiting_opts(; kwargs...) = DECAES.mock_t2map_opts(Float64; MatrixSize = (2, 1, 1), nTE = 16, nT2 = 8, SetFlipAngle = 180.0, SaveRegParam = true, Threaded = false, Silent = true, kwargs...)
    basis(o) = DECAES.epg_decay_basis(DECAES.restructure(DECAES.default_epg_parameters(o), (; α = deg2rad(o.SetFlipAngle))), DECAES.T2_component_times(o))
    function peaks(o, i, j, wi, wj)
        x = zeros(o.nT2)
        x[i], x[j] = wi, wj
        return x
    end

    # Regińska: an exactly-fit second voxel returns μ = 0 after a noisy first voxel leaves a regularized solution in the cache.
    let o = limiting_opts(; Reg = "reginska")
        A = basis(o)
        b₁ = A * peaks(o, 2, 7, 0.9, 0.3) .+ 1e-2 .* sin.(1:o.nTE)
        b₂ = A * peaks(o, 3, 6, 0.4, 0.8)
        image = zeros(2, 1, 1, o.nTE)
        image[1, 1, 1, :], image[2, 1, 1, :] = b₁, b₂
        maps, dist = DECAES.T2mapSEcorr(image, o)

        @test maps["mu"][1] > 0 # the first voxel must actually populate the cache
        @test maps["mu"][2] == 0
        @test dist[2, 1, 1, :] ≈ DECAES.lsqnonneg(A, b₂) rtol = 1e-10 atol = 1e-12
        @test !(dist[2, 1, 1, :] ≈ dist[1, 1, 1, :]) # the leak this guards against
    end

    # MDP: scaling the second voxel down raises δ = √m · NoiseLevel / decay_scale past ‖b‖, which returns the shared zero solution.
    let o = limiting_opts(; Reg = "mdp", NoiseLevel = 1e-2)
        A = basis(o)
        b₁ = A * peaks(o, 2, 7, 0.9, 0.3)
        b₂ = 1e-6 .* (A * peaks(o, 3, 6, 0.4, 0.8))
        image = zeros(2, 1, 1, o.nTE)
        image[1, 1, 1, :], image[2, 1, 1, :] = b₁, b₂
        maps, dist = DECAES.T2mapSEcorr(image, o)

        @test isfinite(maps["mu"][1]) && maps["mu"][1] > 0
        @test maps["mu"][2] == Inf
        @test all(iszero, dist[2, 1, 1, :])
    end
end

# Permitted (Reg, RegNorm) pairs
const REG_NORM_PAIRS = [("none", "l2"), ("chi2", "l2"), ("chi2", "l1"), ("mdp", "l2"), ("mdp", "l1"), ("reginska", "l2"), ("reginska", "l1"), ("lcurve", "l2"), ("lcurve", "l1"), ("gcv", "l2")]

# The regularizer acts on the final T2 fit after the flip angle has been chosen from unregularized fits, so changing the penalty norm must not change the flip-angle map.
@testset "$Reg l1 regularization" for (Reg, RegParams) in (("chi2", (; Chi2Factor = 1.02)), ("reginska", (;)), ("mdp", (; NoiseLevel = 2e-3)), ("lcurve", (;)))
    opts(RegNorm) = DECAES.mock_t2map_opts(Float64; MatrixSize = (3, 3, 1), nTE = 32, nT2 = 20, Reg, RegNorm, SaveRegParam = true, SaveResidualNorm = true, Threaded = false, Silent = true, RegParams...)
    o₂, o₁ = opts("l2"), opts("l1")
    image = DECAES.mock_image(o₂)
    maps₂, dist₂ = T2mapSEcorr(copy(image), o₂)
    maps₁, dist₁ = T2mapSEcorr(copy(image), o₁)

    @test maps₁["alpha"] == maps₂["alpha"]
    @test all(>=(0), dist₁)
    @test all(k -> all(isfinite, maps₁[k]), keys(maps₁))
    @test all(>(0), maps₁["mu"])
    @test count(>(0), dist₁) < count(>(0), dist₂)
end

# Supplying the fitted flip angle back as an input B1 map should reproduce the same fit.
@testset "flip angle round trip ($Reg-$RegNorm)" for (Reg, RegNorm) in REG_NORM_PAIRS
    RegParams = Reg == "chi2" ? (; Chi2Factor = 1.02) : Reg == "mdp" ? (; NoiseLevel = 2e-3) : (;)
    o = DECAES.mock_t2map_opts(Float64; MatrixSize = (4, 4, 1), nTE = 48, nT2 = 40, Reg, RegNorm, SaveRegParam = true, SaveResidualNorm = true, SaveDecayCurve = true, Threaded = false, Silent = true, RegParams...)

    image = DECAES.mock_image(o)
    maps₁, dist₁ = T2mapSEcorr(copy(image), o)

    maps₂ = DECAES.T2Maps(o)
    DECAES.load_B1map!(maps₂, maps₁["alpha"])
    maps₂, dist₂ = DECAES.T2mapSEcorr!(maps₂, DECAES.T2Distributions(o), copy(image), o)

    @test maps₁["alpha"] == maps₂["alpha"]
    @test isapprox(dist₁, dist₂; rtol = 1e-6, atol = 1e-14)
    for k in keys(maps₁)
        @test isapprox(maps₁[k], maps₂[k]; rtol = 1e-6, atol = 1e-14)
    end
end

# Test scale invariance/equivariance of outputs. Scale factor is a power of two such that scaling properties are satisfied bitwise.
@testset "regularization parameter scaling" begin
    opts(Reg, RegNorm, s) = DECAES.mock_t2map_opts(Float64; MatrixSize = (2, 2, 1), nTE = 32, nT2 = 20, Reg, RegNorm, Chi2Factor = 1.02, NoiseLevel = s * 1e-3, SaveRegParam = true, Threaded = false, Silent = true)
    image = DECAES.mock_image(opts("chi2", "l2", 1.0))
    scale = 128.0

    for (Reg, RegNorm) in (("chi2", "l2"), ("chi2", "l1"), ("reginska", "l2"), ("reginska", "l1"), ("mdp", "l2"), ("mdp", "l1"), ("lcurve", "l2"), ("lcurve", "l1"))
        maps, dist = T2mapSEcorr(copy(image), opts(Reg, RegNorm, 1.0))
        maps_scaled, dist_scaled = T2mapSEcorr(scale .* image, opts(Reg, RegNorm, scale)) # NoiseLevel is a physical residual scale, so it rides along with the image
        @test maps_scaled["mu"] == scale^(RegNorm == "l2" ? 0 : 1) .* maps["mu"]
        @test maps_scaled["chi2factor"] == maps["chi2factor"]
        @test dist_scaled == scale .* dist
    end
end

# Determinism across blocks: `reset_voxel_chains!` exists so that the deterministic block partition, not the worker count, fixes every warm-start chain.
# The voxel count must exceed `default_blocksize()`, since a single block cannot exercise a chain that spans one.
@testset "pipeline determinism (multi-block, thread count)" begin
    o = DECAES.mock_t2map_opts(Float64; MatrixSize = (5, 5, 4), nTE = 32, nT2 = 40, Silent = true)
    @test prod(o.MatrixSize) > DECAES.default_blocksize() # more than one block, or this proves nothing
    img = DECAES.mock_image(o)
    mserial, dserial = DECAES.redirect_to_devnull() do
        return T2mapSEcorr(copy(img), DECAES.mock_t2map_opts(Float64; MatrixSize = (5, 5, 4), nTE = 32, nT2 = 40, Silent = true, Threaded = false))
    end
    mthreaded, dthreaded = DECAES.redirect_to_devnull() do
        return T2mapSEcorr(copy(img), o)
    end
    @test isequal(dserial, dthreaded)
    @test all(k -> isequal(mserial[k], mthreaded[k]), keys(mserial))
end

# Determinism: cross-voxel warm-start chains reset at block boundaries, so the pipeline output is a pure function of the input, independent of run and thread scheduling.
# The suppressed differences are near-tie reselections at the KKT tolerance, so a tolerance test would hide a real nondeterminism bug; equality must be bitwise.
@testset "pipeline determinism (run-to-run)" begin
    o = DECAES.mock_t2map_opts(Float64; MatrixSize = (4, 4, 2), nTE = 32, nT2 = 40, Silent = true)
    img = DECAES.mock_image(o)
    m1, d1 = DECAES.redirect_to_devnull() do
        return T2mapSEcorr(copy(img), o)
    end
    m2, d2 = DECAES.redirect_to_devnull() do
        return T2mapSEcorr(copy(img), o)
    end
    @test isequal(d1, d2) # bitwise-identical T2 distributions (NaNs match)
    @test all(k -> isequal(m1[k], m2[k]), keys(m1)) # bitwise-identical maps
end
