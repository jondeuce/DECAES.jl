####
#### Algorithm list
####

const EPG_Algorithms = Any[
    DECAES.EPGWork_Basic_Cplx,
    # DECAES.EPGWork_Vec,
    DECAES.EPGWork_ReIm,
    DECAES.EPGWork_ReIm_DualVector,
    DECAES.EPGWork_ReIm_DualVector_Split,
    DECAES.EPGWork_ReIm_DualVector_Split_Dynamic,
    DECAES.EPGWork_ReIm_DualFlat_Split_Dynamic,
    DECAES.EPGWork_ReIm_DualTuple_Split_Dynamic,
    DECAES.EPGWork_ReIm_Batched_Split_Dynamic,
    DECAES.EPGWork_ReIm_Tangent_DualTuple_Split_Dynamic,
    DECAES.EPGWork_ReIm_Tangent_Batched_Split_Dynamic,
    DECAES.EPGWork_ReIm_DualMVector_Split,
    # DECAES.EPGWork_ReIm_DualPaddedMVector_Vec_Split,
    DECAES.EPGWork_ReIm_DualPaddedVector_Split,
    # DECAES.EPGWork_ReIm_Generated,
]

const EPG_TestOptionTypes = (
    DECAES.EPGOptions,
    DECAES.EPGConstantFlipAngleOptions,
    DECAES.EPGIncreasingFlipAnglesOptions,
)

mock_θ(::Type{DECAES.EPGOptions}, ::Type{T}, ETL::Int) where {T} = DECAES.EPGOptions((; ETL, α = deg2rad(T(165.0)), TE = T(39e-3), T2 = T(1.1), T1 = T(151.0), β = deg2rad(T(150.0))))
mock_θ(::Type{DECAES.EPGConstantFlipAngleOptions}, ::Type{T}, ETL::Int) where {T} = DECAES.EPGConstantFlipAngleOptions((; ETL, α = deg2rad(T(165.0)), TE = T(39e-3), T2 = T(1.1), T1 = T(151.0)))
mock_θ(::Type{DECAES.EPGIncreasingFlipAnglesOptions}, ::Type{T}, ETL::Int) where {T} = DECAES.EPGIncreasingFlipAnglesOptions((; ETL, α = deg2rad(T(165.0)), α1 = deg2rad(T(165.0)), α2 = deg2rad(T(140.0)), TE = T(39e-3), T2 = T(1.1), T1 = T(151.0)))

supports(work, θ) = applicable(DECAES.epg_decay_curve!, DECAES.decaycurve(work), work, θ)

epg_reference(θ) = copy(DECAES.EPGdecaycurve!(DECAES.EPGWork_Basic_Cplx(Double64, DECAES.echotrainlength(θ)), θ))

function compare_epg(work₁::DECAES.AbstractEPGWorkspace{T}, work₂::DECAES.AbstractEPGWorkspace{T}, θ₁::DECAES.EPGParameterization{T}, θ₂::DECAES.EPGParameterization{T}; verbose = false) where {T}
    dc₁ = zeros(T, DECAES.echotrainlength(work₁))
    dc₂ = zeros(T, DECAES.echotrainlength(work₂))
    DECAES.EPGdecaycurve!(dc₁, work₁, θ₁)
    DECAES.EPGdecaycurve!(dc₂, work₂, θ₂)

    if verbose && !(dc₁ ≈ dc₂)
        @info "Comparing: $((nameof(typeof(work₁)), nameof(typeof(work₂))))"
        @info "  option types: $((nameof(typeof(θ₁)), nameof(typeof(θ₂))))"
        @info "  max error:   $(maximum(abs, dc₁ .- dc₂))"
        @info "  diff vector: $(abs.(dc₁ .- dc₂)')"
    end

    @test isapprox(dc₁, dc₂; rtol = cbrt(eps(T))^2, atol = 10 * eps(T))
end

function test_EPG_algorithms(; verbose = false)
    for T in (Float32, Float64)
        for ETL in (4, 5, 6, 7)
            for Opt in EPG_TestOptionTypes
                θ_ETL = mock_θ(Opt, T, ETL)

                works = DECAES.AbstractEPGWorkspace{T}[]
                for alg in EPG_Algorithms
                    w = alg(T, ETL)
                    supports(w, θ_ETL) && push!(works, w)
                end
                @test !isempty(works)

                ref = first(works)
                θ_ref = mock_θ(Opt, T, DECAES.echotrainlength(ref))

                for w in works
                    w === ref && continue
                    θ_w = mock_θ(Opt, T, DECAES.echotrainlength(w))
                    compare_epg(ref, w, θ_ref, θ_w; verbose)
                end
            end
        end

        # Default factory vs reference implementation
        for ETL in 4:64
            θ = mock_θ(DECAES.EPGOptions, T, ETL)
            @test supports(DECAES.EPGdecaycurve_work(T, ETL), θ)
            @test supports(DECAES.EPGWork_Basic_Cplx(T, ETL), θ)
            compare_epg(DECAES.EPGWork_Basic_Cplx(T, ETL), DECAES.EPGdecaycurve_work(T, ETL), θ, θ; verbose)
        end
    end
end

function test_EPG_algorithm_consistency(; verbose = false)
    # Constant-only fast kernels
    const_only_algs = (
        DECAES.EPGWork_ReIm_DualVector_Split_Dynamic,
        DECAES.EPGWork_ReIm_DualFlat_Split_Dynamic,
        DECAES.EPGWork_ReIm_DualTuple_Split_Dynamic,
        DECAES.EPGWork_ReIm_Batched_Split_Dynamic,
    )

    for T in (Float32, Float64)
        for ETL in (4, 5, 6, 7)
            α = deg2rad(T(165.0))
            TE = T(39e-3)
            T2 = T(1.1)
            T1 = T(151.0)

            # EPGOptions with β = π represents a constant train
            θ_opt = DECAES.EPGOptions((; ETL, α, TE, T2, T1, β = T(π)))
            θ_cst = DECAES.EPGConstantFlipAngleOptions((; ETL, α, TE, T2, T1))
            θ_inc = DECAES.EPGIncreasingFlipAnglesOptions((; ETL, α, α1 = T(π), α2 = T(π), TE, T2, T1))

            w_ref = DECAES.EPGWork_Basic_Cplx(T, ETL)
            @test supports(w_ref, θ_opt)

            # Constant-only algs vs EPGOptions(β=180)
            for Alg in const_only_algs
                w_c = Alg(T, ETL)
                @test supports(w_c, θ_cst)
                compare_epg(w_ref, w_c, θ_opt, θ_cst; verbose)
            end

            # Increasing(α1=180,α2=180) vs EPGOptions(β=180)
            @test supports(w_ref, θ_inc)
            compare_epg(w_ref, w_ref, θ_opt, θ_inc; verbose)
        end
    end
end

function test_EPGOptions()
    θ = DECAES.EPGOptions((; ETL = 10, α = deg2rad(169.0), TE = 9.0e-3, T2 = 10.1e-3, T1 = 0.98, β = deg2rad(176.0)))

    @testset "basics" begin
        @test Tuple(θ) == (10, deg2rad(169.0), 9.0e-3, 10.1e-3, 0.98, deg2rad(176.0))
        @test NamedTuple(θ) == (; ETL = 10, α = deg2rad(169.0), TE = 9.0e-3, T2 = 10.1e-3, T1 = 0.98, β = deg2rad(176.0))
    end

    @testset "destructure/restructure" begin
        θ′ = @inferred(DECAES.restructure(θ, (2.0, 1.0), Val((:β, :α))))
        @test Tuple(θ′) == (θ.ETL, 1.0, θ.TE, θ.T2, θ.T1, 2.0)

        x′ = @inferred(DECAES.destructure(θ, Val((:TE, :α))))
        @test x′ == SA[θ.TE, θ.α]
    end
end

function test_EPGFunctor()
    T = Float64
    ETL = 8
    θ = DECAES.EPGOptions((; ETL, α = deg2rad(169.0), TE = 9.0e-3, T2 = 10.1e-3, T1 = 0.98, β = deg2rad(176.0)))
    fun! = DECAES.EPGFunctor(θ, Val((:α, :T2)))
    jac! = DECAES.EPGJacobianFunctor(θ, Val((:α, :T2)))

    # EPGFunctor
    x = [θ.α, θ.T2]
    y = zeros(T, ETL)
    @inferred fun!(y, x)
    @test y == DECAES.EPGdecaycurve(θ)

    # EPGJacobianFunctor
    y .= 0
    J = zeros(T, ETL, 2)
    @inferred jac!(y, θ)
    @inferred jac!(J, y, θ)
    @test y ≈ DECAES.EPGdecaycurve(θ) atol = 10 * eps(T) # note: not exact because Dual's likely lead to different SIMD instructions etc.
    @test J == DECAES.DiffResults.jacobian(jac!.res)

    # Finite difference test; error should decrease as 𝒪(δx^2)
    δx = x .* T(1e-2)
    θ′ = DECAES.restructure(θ, x .+ δx, Val((:α, :T2)))
    @test J * δx ≈ DECAES.EPGdecaycurve(θ′) - y atol = 5e-4

    δx = x .* T(1e-4)
    θ′ = DECAES.restructure(θ, x .+ δx, Val((:α, :T2)))
    @test J * δx ≈ DECAES.EPGdecaycurve(θ′) - y atol = 5e-8

    δx = x .* T(1e-6)
    θ′ = DECAES.restructure(θ, x .+ δx, Val((:α, :T2)))
    @test J * δx ≈ DECAES.EPGdecaycurve(θ′) - y atol = 5e-12
end

# The lane-batched kernel computes decay bases in chunks of EPG_BATCH_WIDTH T2s at once.
# Compare against the per-curve reference basis for heterogeneous lanes, partial chunks, and remainder handling.
function test_EPG_batched_basis()
    for T in (Float32, Float64), ETL in (4, 7, 33, 48), nT2 in (1, 3, 8, 11, 40)
        θ = mock_θ(DECAES.EPGConstantFlipAngleOptions, T, ETL)
        T2_times = nT2 == 1 ? T[0.1] : exp10.(range(T(-2), T(0); length = nT2))
        basis_ref = zeros(T, ETL, nT2)
        basis_bat = zeros(T, ETL, nT2)
        DECAES.epg_decay_basis!(basis_ref, DECAES.EPGdecaycurve_work(θ), θ, T2_times)
        DECAES.epg_decay_basis!(basis_bat, DECAES.EPGWork_ReIm_Batched_Split_Dynamic(T, ETL), θ, T2_times)
        @test isapprox(basis_ref, basis_bat; rtol = cbrt(eps(T))^2, atol = 10 * eps(T))
    end
end

# Reference implementation for the cosine series representation: sample the EPG impulse response hᵢ at α_m = πm/N, then invert hᵢ(α_m) = Σₖ Âᵢₖ cos(πkm/N) by DCT-I,
#   Âᵢₖ = (2wₖ/N) Σₘ wₘ hᵢ(α_m) cos(πkm/N),
# where w₀ = w_N = 1/2 and wₖ = 1 otherwise.
function epg_cosine_coefficients_reference(θ::DECAES.EPGConstantFlipAngleOptions{T}) where {T}
    N = DECAES.echotrainlength(θ)
    work = DECAES.EPGWork_ReIm_Batched_Split_Dynamic(T, N)
    h = zeros(T, N, N + 1)
    for m in 0:N
        DECAES.epg_setup_lanes!(work, DECAES.restructure(θ, (; α = T(π) * m / N)), [θ.T2], 1, 1)
        DECAES.epg_impulse_response_batched!(work)
        h[:, m+1] .= work.h[1, 1:N]
    end
    w = [m == 0 || m == N ? T(1) / 2 : T(1) for m in 0:N]
    return [2 * w[k+1] / N * sum(w[m+1] * h[i, m+1] * cospi(T(k * m) / N) for m in 0:N) for i in 1:N, k in 0:N]
end

function test_EPG_cosine_coefficients()
    T = Float64
    T2_times = exp10.(range(T(-2), T(0); length = 11)) # one full batch of lanes and a partial one
    for ETL in (4, 7, 33, 48)
        θ = mock_θ(DECAES.EPGConstantFlipAngleOptions, T, ETL)
        Â = DECAES.epg_cosine_coefficients(θ, T2_times)
        for (j, T2) in enumerate(T2_times)
            Â_ref = epg_cosine_coefficients_reference(DECAES.restructure(mock_θ(DECAES.EPGConstantFlipAngleOptions, Double64, ETL), (; T2 = Double64(T2))))
            k = (0:ETL)' # p'th α-derivative error is proportional to kᵖ-weighted coefficient difference
            @test isapprox(Â[:, :, j], Â_ref; rtol = 10 * eps(T))
            @test isapprox(Â[:, :, j] .* k, Â_ref .* k; rtol = 25 * eps(T))
            @test isapprox(Â[:, :, j] .* k .^ 2, Â_ref .* k .^ 2; rtol = 50 * eps(T))
        end
    end
end

function test_EPG_cosine_basis()
    for T in (Float32, Float64), ETL in (4, 7, 33, 48), nT2 in (1, 3, 8, 11, 40)
        θ = mock_θ(DECAES.EPGConstantFlipAngleOptions, T, ETL)
        T2_times = nT2 == 1 ? T[0.1] : exp10.(range(T(-2), T(0); length = nT2))
        work_cosine_basis = DECAES.EPGCosineSeriesBasis(θ, T2_times)
        basis_ref = zeros(T, ETL, nT2)
        basis_cos = zeros(T, ETL, nT2)
        work_ref = DECAES.EPGdecaycurve_work(θ)
        for α in T[51.3, 90.0, 119.7, 147.2, 165.0, 180.0]
            DECAES.epg_decay_basis!(basis_ref, work_ref, DECAES.restructure(θ, (; α = deg2rad(α))), T2_times)
            DECAES.epg_decay_basis!(basis_cos, work_cosine_basis, deg2rad(α))
            @test isapprox(basis_ref, basis_cos; rtol = 20 * eps(T))
        end
        @test allocated_bytes(DECAES.epg_decay_basis!, basis_cos, work_cosine_basis, T(π)) == 0
    end
end

function test_EPG_cosine_basis_gradient()
    T = Float64
    for ETL in (4, 7, 33, 48), nT2 in (1, 8, 40)
        θ = mock_θ(DECAES.EPGConstantFlipAngleOptions, T, ETL)
        T2_times = nT2 == 1 ? T[0.1] : exp10.(range(T(-2), T(0); length = nT2))
        work_cosine_basis = DECAES.EPGCosineSeriesBasis(θ, T2_times)
        Acol, ∂Acol, ∂²Acol = zeros(T, ETL), zeros(T, ETL), zeros(T, ETL)

        # The differencing alone runs in Double64, so the reference carries no cancellation error and the tolerance measures the Float64 kernel rather than the step size.
        θ_D64 = mock_θ(DECAES.EPGConstantFlipAngleOptions, Double64, ETL)
        work_D64 = DECAES.EPGCosineSeriesBasis(θ_D64, Double64.(T2_times))
        B⁺, B⁻, B, B⁺₂, B⁻₂ = (zeros(Double64, ETL, nT2) for _ in 1:5)
        Bf = zeros(T, ETL, nT2)
        h₁, h₂ = Double64(1e-12), Double64(1e-8)
        for α_deg in T[91.3, 120.7, 150.2, 164.9, 179.1]
            α = deg2rad(α_deg)
            α_D64 = deg2rad(Double64(α_deg))
            DECAES.epg_decay_basis!(B⁺, work_D64, α_D64 + h₁)
            DECAES.epg_decay_basis!(B⁻, work_D64, α_D64 - h₁)
            DECAES.epg_decay_basis!(B⁺₂, work_D64, α_D64 + h₂)
            DECAES.epg_decay_basis!(B⁻₂, work_D64, α_D64 - h₂)
            DECAES.epg_decay_basis!(B, work_D64, α_D64)
            DECAES.epg_decay_basis!(Bf, work_cosine_basis, α) # Float64 value reference; also leaves the features current at α for the column kernel
            for j in 1:nT2
                DECAES.epg_decay_basis_∂α_col!(Acol, ∂Acol, ∂²Acol, work_cosine_basis, α, j)
                @test isapprox(Acol, Bf[:, j]; rtol = 10 * eps(T)) # value column matches the full-basis kernel
                @test isapprox(∂Acol, T.((B⁺[:, j] .- B⁻[:, j]) ./ 2h₁); rtol = 100 * eps(T)) # derivative column
                @test isapprox(∂²Acol, T.((B⁺₂[:, j] .- 2 .* B[:, j] .+ B⁻₂[:, j]) ./ h₂^2); rtol = 250 * eps(T)) # second-derivative column
            end
        end
    end
end

function test_EPG_flip_angle_derivatives()
    T = Float64
    h₁, h₂ = Double64(1e-12), Double64(1e-8)
    for Opt in EPG_TestOptionTypes, ETL in (4, 7, 33, 48)
        work = DECAES.EPGWork_ReIm_Tangent_DualTuple_Split_Dynamic(T, ETL)
        Â, ∂Â, ∂²Â = zeros(T, ETL), zeros(T, ETL), zeros(T, ETL)
        for α_deg in T[91.3, 120.7, 150.2, 164.9, 179.1]
            θ = DECAES.restructure(mock_θ(Opt, T, ETL), (; α = deg2rad(α_deg)))
            DECAES.epg_decay_curve_∂α!(Â, ∂Â, ∂²Â, work, θ)

            θ_D64 = DECAES.restructure(mock_θ(Opt, Double64, ETL), (; α = deg2rad(Double64(α_deg))))
            A, A₁⁺, A₁⁻, A⁺₂, A⁻₂ = (epg_reference(DECAES.restructure(θ_D64, (; α = θ_D64.α + δ))) for δ in (0, h₁, -h₁, h₂, -h₂))
            @test isapprox(Â, A; rtol = 50 * eps(T))
            @test isapprox(∂Â, (A₁⁺ .- A₁⁻) ./ 2h₁; rtol = 100 * eps(T))
            @test isapprox(∂²Â, (A⁺₂ .- 2 .* A .+ A⁻₂) ./ h₂^2; rtol = 500 * eps(T))

            # Any output is allowed to be `nothing` or a preallocated buffer
            for outputs in Iterators.product(((nothing, zeros(T, ETL)) for _ in 1:3)...)
                DECAES.epg_decay_curve_∂α!(outputs..., work, θ)
                for (output, ref) in zip(outputs, (Â, ∂Â, ∂²Â))
                    output === nothing || @test isapprox(output, ref; rtol = 50 * eps(T))
                end
            end
        end
        @test allocated_bytes(DECAES.epg_decay_curve_∂α!, Â, ∂Â, ∂²Â, work, mock_θ(Opt, T, ETL)) == 0
    end
end

function test_EPG_batched_flip_angle_derivatives()
    T = Float64
    for Opt in EPG_TestOptionTypes, ETL in (4, 7, 33, 48), nT2 in (1, 11, 14, 40)
        θ₀ = mock_θ(Opt, T, ETL)
        T2_times = nT2 == 1 ? T[0.1] : exp10.(range(T(-2), T(0); length = nT2))
        cols = circshift(nT2:-1:1, nT2 ÷ 3) # permuted to check that output column col holds T2_times[cols[col]]
        work_basis = DECAES.EPGWork_ReIm_Tangent_Batched_Split_Dynamic(T, ETL)
        work_curve = DECAES.EPGWork_ReIm_Tangent_DualTuple_Split_Dynamic(T, ETL)
        A, ∂A, ∂²A = zeros(T, ETL, nT2), zeros(T, ETL, nT2), zeros(T, ETL, nT2)
        Acol, ∂Acol, ∂²Acol = zeros(T, ETL), zeros(T, ETL), zeros(T, ETL)
        for α_deg in T[95.0, 140.0, 172.5]
            θ = DECAES.restructure(θ₀, (; α = deg2rad(α_deg)))
            DECAES.epg_decay_basis_∂α!(A, ∂A, ∂²A, work_basis, θ, T2_times, cols)
            for (col, j) in enumerate(cols)
                DECAES.epg_decay_curve_∂α!(Acol, ∂Acol, ∂²Acol, work_curve, DECAES.restructure(θ, (; T2 = T2_times[j])))
                @test isapprox(A[:, col], Acol; rtol = 20 * eps(T))
                @test isapprox(∂A[:, col], ∂Acol; rtol = 50 * eps(T))
                @test isapprox(∂²A[:, col], ∂²Acol; rtol = 100 * eps(T))
            end

            if Opt === DECAES.EPGConstantFlipAngleOptions
                work_cosine_basis = DECAES.EPGCosineSeriesBasis(θ₀, T2_times)
                A_cos, ∂A_cos, ∂²A_cos = zeros(T, ETL, nT2), zeros(T, ETL, nT2), zeros(T, ETL, nT2)
                DECAES.cosine_features!(work_cosine_basis, θ.α)
                DECAES.epg_decay_basis_∂α!(A_cos, ∂A_cos, ∂²A_cos, work_cosine_basis, θ.α, cols)
                @test isapprox(A, A_cos; rtol = 20 * eps(T))
                @test isapprox(∂A, ∂A_cos; rtol = 50 * eps(T))
                @test isapprox(∂²A, ∂²A_cos; rtol = 100 * eps(T))
                @test allocated_bytes(DECAES.epg_decay_basis_∂α!, A_cos, ∂A_cos, ∂²A_cos, work_cosine_basis, θ.α, cols) == 0
            end
        end
        @test allocated_bytes(DECAES.epg_decay_basis_∂α!, A, ∂A, ∂²A, work_basis, θ₀, T2_times, cols) == 0
    end
end

# `EPGdecaycurve` takes angles in degrees. Partials in the flip angle alone take the analytic tangent; partials in another argument as well take the generic dual-number path.
function test_EPG_flip_angle_derivative_dual()
    T = Float64
    ETL, TE, T2, T1 = 32, T(9e-3), T(45e-3), T(1.1)
    h = Double64(1e-12)
    for α in T[120.3, 163.1], β in ((), (T(150.0),))
        function θ_D64(α, T2)
            α, TE_D64, T1_D64 = deg2rad(Double64(α)), Double64(TE), Double64(T1)
            return isempty(β) ?
                   DECAES.EPGConstantFlipAngleOptions((; ETL, α, TE = TE_D64, T2 = Double64(T2), T1 = T1_D64)) :
                   DECAES.EPGOptions((; ETL, α, TE = TE_D64, T2 = Double64(T2), T1 = T1_D64, β = deg2rad(Double64(only(β)))))
        end
        ∂Acol_∂α = (epg_reference(θ_D64(α + h, T2)) .- epg_reference(θ_D64(α - h, T2))) ./ 2h
        ∂Acol_∂T2 = (epg_reference(θ_D64(α, T2 + h)) .- epg_reference(θ_D64(α, T2 - h))) ./ 2h

        g = ForwardDiff.derivative(α -> EPGdecaycurve(ETL, α, TE, T2, T1, β...), α)
        J = ForwardDiff.jacobian(x -> EPGdecaycurve(ETL, x[1], TE, x[2], T1, β...), [α, T2])

        @test isapprox(g, ∂Acol_∂α; rtol = 50 * eps(T))
        @test isapprox(J[:, 1], ∂Acol_∂α; rtol = 50 * eps(T))
        @test isapprox(J[:, 2], ∂Acol_∂T2; rtol = 50 * eps(T))
    end
end

@testset "EPG algorithms" test_EPG_algorithms()
@testset "EPG algorithm consistency" test_EPG_algorithm_consistency()
@testset "EPG batched basis" test_EPG_batched_basis()
@testset "EPG cosine series coefficients" test_EPG_cosine_coefficients()
@testset "EPG cosine series basis" test_EPG_cosine_basis()
@testset "EPG cosine series basis gradient" test_EPG_cosine_basis_gradient()
@testset "EPG flip angle derivatives" test_EPG_flip_angle_derivatives()
@testset "EPG batched flip angle derivatives" test_EPG_batched_flip_angle_derivatives()
@testset "EPG flip angle derivative dual numbers" test_EPG_flip_angle_derivative_dual()
@testset "EPGOptions" test_EPGOptions()
@testset "EPGFunctor" test_EPGFunctor()

nothing
