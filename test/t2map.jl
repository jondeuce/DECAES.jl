Random.seed!(0) # reproducible randomized tests

# α-polish: the refinement of the flip-angle surrogate minimizer. The polish spends one true off-grid loss and gradient evaluation, minimizes both adjacent cubic Hermites, and returns the lex-minimum of true losses over the best evaluated node, α₀, and the candidate.
# The gradient is the envelope-theorem derivative g = 2(∂A/∂α·x)ᵀ(Ax − b), valid because x is the NNLS minimizer at α.
function alpha_polish_setup()
    o = DECAES.mock_t2map_opts(Float64; MatrixSize = (1, 1, 1), nTE = 48, nT2 = 40, nRefAngles = 16, nRefAnglesMin = 5, Silent = true)
    θ = DECAES.default_epg_parameters(o)
    T2t = DECAES.T2_component_times(o)
    function loss_true(α, b)
        A = DECAES.epg_decay_basis(DECAES.restructure(θ, (; α)), T2t)
        x = DECAES.lsqnonneg(A, b)
        return sum(abs2, A * x - b)
    end
    function make_signal(α_true, noise = 0.0)
        A = DECAES.epg_decay_basis(DECAES.restructure(θ, (; α = α_true)), T2t)
        x = zeros(o.nT2)
        for _ in 1:rand(1:4)
            x[rand(1:o.nT2)] += rand()
        end
        b = A * x
        b ./= maximum(b)
        noise > 0 && (b .= max.(b .+ noise .* randn(o.nTE), 0)) # noiseless signals never exercise the candidate-rejection path
        return b
    end
    function build(b)
        return DECAES.FlipAngleOptimizationWorkspace(o, zeros(o.nTE, o.nT2), copy(b))
    end
    # The discrete search's own continuous proposal α₀, before any true off-grid evaluation. Mirrors the search prefix of `optimize_flip_angle!`.
    function surrogate_minimizer(w)
        empty!(w.α_surrogate)
        DECAES.advance_warmstart!(w.nnls_search_prob)
        DECAES.reset!(w.α_searcher)
        DECAES.initialize!(w.α_surrogate, w.α_searcher; mineval = o.nRefAnglesMin, maxeval = o.nRefAngles)
        α, _ = DECAES.bisection_search(w.α_surrogate, w.α_searcher; maxeval = o.nRefAngles)
        return α[1]
    end
    return o, loss_true, make_signal, build, surrogate_minimizer
end

function test_alpha_polish()
    o, loss_true, make_signal, build, surrogate_minimizer = alpha_polish_setup()
    for noise in (0.0, 0.02), _ in 1:20
        b = make_signal(100.0 + 75.0 * rand(), noise)

        # The search is deterministic, so a fresh workspace reaching only the search prefix yields the α₀ the polished run started from
        α₀ = surrogate_minimizer(build(b))

        wp = build(b)
        DECAES.optimize_flip_angle!(wp, o)
        αp = wp.α[]

        # Certificate: the returned angle's true loss never exceeds the best evaluated node's, whatever the interpolant proposed.
        # The allowance is roundoff between two independent NNLS solves, not slack in the selection, which compares its own evaluations exactly.
        @test loss_true(αp, b) <= (1 + 1e-10) * best_seen_loss(wp) + 1e-14
        @test loss_true(αp, b) <= (1 + 1e-10) * loss_true(α₀, b) + 1e-14
        @test 90 < αp <= 180 # noisy signals routinely minimize at the grid's upper endpoint

        # Envelope-theorem soundness: the loss and gradient at α₀ match an independent NNLS solve and central differences of the true loss
        if 100 < α₀ < 175
            wg = build(b)
            surrogate_minimizer(wg)
            f₀, g₀ = DECAES.polish_loss_grad!(wg, α₀)
            @test f₀ ≈ loss_true(α₀, b) rtol = 1e-6

            h = 1e-4
            g_fd = (loss_true(α₀ + h, b) - loss_true(α₀ - h, b)) / 2h
            @test g₀ ≈ g_fd rtol = 5e-3 atol = 1e-6
        end
    end
end

# Smallest loss among the nodes the search actually evaluated, which is the incumbent the final selection must beat
best_seen_loss(w) = minimum(w.α_surrogate.u[j] for j in eachindex(w.α_surrogate.u) if w.α_surrogate.seen[j])

# A signal whose Hermite candidate lands on a worse α: the sub-bracket minimizer sits where the interpolant undershoots the true loss, which the endpoint data cannot exclude.
# Pinned so the candidate-rejection path has a deterministic exercise; random noisy signals reach it in roughly 1 in 200 searches.
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
function test_alpha_polish_general_beta()
    for β in (120.0, 150.0, 175.0)
        o = DECAES.mock_t2map_opts(Float64; MatrixSize = (1, 1, 1), nTE = 48, nT2 = 40, nRefAngles = 16, nRefAnglesMin = 5, RefConAngle = β, Silent = true)
        θ = DECAES.default_epg_parameters(o)
        T2t = DECAES.T2_component_times(o)
        @test !(θ isa DECAES.EPGConstantFlipAngleOptions) # the cosine backend must not apply here
        function loss_true(α, b)
            A = DECAES.epg_decay_basis(DECAES.restructure(θ, (; α)), T2t)
            return sum(abs2, A * DECAES.lsqnonneg(A, b) - b)
        end

        for _ in 1:4
            A = DECAES.epg_decay_basis(DECAES.restructure(θ, (; α = 100.0 + 75.0 * rand())), T2t)
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

            @test 90 < α̂ <= 180
            @test loss_true(α̂, b) <= (1 + 1e-10) * best_seen_loss(w) + 1e-14

            # The AD envelope derivative matches central differences of the true loss
            if 100 < α̂ < 175
                f, g = DECAES.polish_loss_grad!(w, α̂)
                @test f ≈ loss_true(α̂, b) rtol = 1e-6
                h = 1e-4
                @test g ≈ (loss_true(α̂ + h, b) - loss_true(α̂ - h, b)) / 2h rtol = 5e-3 atol = 1e-6
            end
        end
    end
end

@testset "α-polish" begin
    test_alpha_polish()
    test_alpha_polish_guard()
    test_alpha_polish_general_beta()
end

# When the polish problem is already solved, the T2 stage skips its own unregularized solve and takes that state instead.
# That is sound only if the basis really is A(α̂) and the workspace really holds its exact NNLS solution, so both halves are checked against an independent solve.
@testset "certified unregularized state" begin
    o, loss_true, make_signal, build = alpha_polish_setup()
    θ, T2t = DECAES.default_epg_parameters(o), DECAES.T2_component_times(o)
    ncertified = 0
    for _ in 1:20
        b = make_signal(100.0 + 75.0 * rand(), 0.02)
        w = build(b)
        DECAES.optimize_flip_angle!(w, o)
        DECAES.NNLS.issolved(w.α_polish_work.prob.nnls_work) || continue
        ncertified += 1

        @test w.decay_basis ≈ DECAES.epg_decay_basis(DECAES.restructure(θ, (; α = w.α[])), T2t)

        wk = w.α_polish_work.prob.nnls_work
        x = DECAES.lsqnonneg(w.decay_basis, b)
        @test DECAES.NNLS.solution(wk) ≈ x rtol = 1e-10
        @test DECAES.NNLS.residual(wk) ≈ b - w.decay_basis * x rtol = 1e-10 atol = 1e-14
        @test DECAES.NNLS.residualnorm(wk) ≈ norm(b - w.decay_basis * x) rtol = 1e-10
        @test sort(DECAES.NNLS.components(wk)) == findall(>(0), x)
    end
    @test ncertified > 0 # the adopted path must actually be reached
end

# Boundary paths return their selected solution directly rather than through the regularized cache, which they do not populate.
@testset "regularization boundary solution propagation" begin
    o = DECAES.mock_t2map_opts(Float64; MatrixSize = (1, 1, 1), nTE = 16, nT2 = 8, SetFlipAngle = 180.0, Reg = "reginska", SaveRegParam = true, Threaded = false, Silent = true)
    θ = DECAES.restructure(DECAES.default_epg_parameters(o), (; α = o.SetFlipAngle))
    A = DECAES.epg_decay_basis(θ, DECAES.T2_component_times(o))
    x = zeros(o.nT2)
    x[3], x[6] = 0.4, 0.8
    b = A * x
    image = reshape(copy(b), 1, 1, 1, o.nTE)
    maps, dist = DECAES.T2mapSEcorr(image, o)

    @test maps["mu"][1] == 0
    @test vec(dist) ≈ DECAES.lsqnonneg(A, b) rtol = 1e-10 atol = 1e-12
end

# The boundary returns are precisely the ones the regularized cache never writes, so the failure needs a reused workspace: a first voxel that populates the cache, then a second that must not read it.
# Both boundaries here return a vector the cache does not own, μ = 0 the unregularized workspace and μ = ∞ the shared zero solution.
@testset "regularization boundaries across a reused workspace" begin
    boundary_opts(; kwargs...) = DECAES.mock_t2map_opts(Float64; MatrixSize = (2, 1, 1), nTE = 16, nT2 = 8, SetFlipAngle = 180.0, SaveRegParam = true, Threaded = false, Silent = true, kwargs...)
    basis(o) = DECAES.epg_decay_basis(DECAES.restructure(DECAES.default_epg_parameters(o), (; α = o.SetFlipAngle)), DECAES.T2_component_times(o))
    function peaks(o, i, j, wi, wj)
        x = zeros(o.nT2)
        x[i], x[j] = wi, wj
        return x
    end

    # Regińska: an exactly-fit second voxel returns μ = 0 after a noisy first voxel leaves a regularized solution in the cache.
    let o = boundary_opts(; Reg = "reginska")
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
    let o = boundary_opts(; Reg = "mdp", NoiseLevel = 1e-2)
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
