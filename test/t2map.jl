Random.seed!(0) # reproducible randomized tests

# α-polish: the default-on refinement of the flip-angle surrogate minimizer. The polish spends one true off-grid (loss, gradient) evaluation and re-minimizes a cubic Hermite;
# the gradient is the envelope-theorem derivative g = 2(∂A/∂α·x)ᵀ(Ax − b), valid because x is the NNLS minimizer at α.
function alpha_polish_setup()
    o = DECAES.mock_t2map_opts(Float64; MatrixSize = (1, 1, 1), nTE = 48, nT2 = 40, nRefAngles = 16, nRefAnglesMin = 5, Silent = true)
    θ = DECAES.default_epg_parameters(o)
    T2t = DECAES.T2_component_times(o)
    function loss_true(α, b)
        A = DECAES.epg_decay_basis(DECAES.restructure(θ, (; α)), T2t)
        x = DECAES.lsqnonneg(A, b)
        return sum(abs2, A * x - b)
    end
    function make_signal(α_true)
        A = DECAES.epg_decay_basis(DECAES.restructure(θ, (; α = α_true)), T2t)
        x = zeros(o.nT2)
        for _ in 1:rand(1:4)
            x[rand(1:o.nT2)] += rand()
        end
        b = A * x
        return b ./ maximum(b)
    end
    function build(b)
        return DECAES.FlipAngleOptimizationWorkspace(o, zeros(o.nTE, o.nT2), copy(b))
    end
    return o, loss_true, make_signal, build
end

function test_alpha_polish()
    o, loss_true, make_signal, build = alpha_polish_setup()
    CURR_FLIP_ALPHA_POLISH = DECAES.FLIP_ALPHA_POLISH[]
    try
        for _ in 1:20
            b = make_signal(100.0 + 75.0 * rand())

            # Unpolished surrogate minimizer α₀ vs the polished α (same deterministic search on a fresh workspace, only the polish differs)
            wu = build(b)
            DECAES.FLIP_ALPHA_POLISH[] = false
            DECAES.optimize_flip_angle!(wu, o)
            α₀ = wu.α[]

            wp = build(b)
            DECAES.FLIP_ALPHA_POLISH[] = true
            DECAES.optimize_flip_angle!(wp, o)
            αp = wp.α[]

            # Contract: the polish never increases the true loss, and α stays in range
            @test loss_true(αp, b) <= (1 + 1e-8) * loss_true(α₀, b) + 1e-14
            @test 90 < αp < 180

            # Envelope-theorem soundness: the (loss, gradient) at α₀ match an independent NNLS solve and central differences of the true loss
            if 100 < α₀ < 175
                wg = build(b)
                DECAES.FLIP_ALPHA_POLISH[] = false
                DECAES.optimize_flip_angle!(wg, o)
                f₀, g₀ = DECAES.polish_loss_grad!(wg, α₀)
                @test f₀ ≈ loss_true(α₀, b) rtol = 1e-6

                h = 1e-4
                g_fd = (loss_true(α₀ + h, b) - loss_true(α₀ - h, b)) / 2h
                @test g₀ ≈ g_fd rtol = 5e-3 atol = 1e-6
            end
        end
    finally
        DECAES.FLIP_ALPHA_POLISH[] = CURR_FLIP_ALPHA_POLISH
    end
end

@testset "α-polish" begin
    test_alpha_polish()
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
