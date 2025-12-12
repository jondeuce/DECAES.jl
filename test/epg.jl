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

####
#### Test parameter constructors
####

mock_θ(::Type{DECAES.EPGOptions}, ::Type{T}, ETL::Int) where {T} = DECAES.EPGOptions((; ETL, α = T(165.0), TE = T(39e-3), T2 = T(1.1), T1 = T(151.0), β = T(150.0)))
mock_θ(::Type{DECAES.EPGConstantFlipAngleOptions}, ::Type{T}, ETL::Int) where {T} = DECAES.EPGConstantFlipAngleOptions((; ETL, α = T(165.0), TE = T(39e-3), T2 = T(1.1), T1 = T(151.0)))
mock_θ(::Type{DECAES.EPGIncreasingFlipAnglesOptions}, ::Type{T}, ETL::Int) where {T} = DECAES.EPGIncreasingFlipAnglesOptions((; ETL, α = T(165.0), α1 = T(165.0), α2 = T(140.0), TE = T(39e-3), T2 = T(1.1), T1 = T(151.0)))

supports(work, θ) = applicable(DECAES.epg_decay_curve!, DECAES.decaycurve(work), work, θ)

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

    @test isapprox(dc₁, dc₂; rtol = √eps(T), atol = 10 * eps(T))
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
    )

    for T in (Float32, Float64)
        for ETL in (4, 5, 6, 7)
            α  = T(165.0)
            TE = T(39e-3)
            T2 = T(1.1)
            T1 = T(151.0)

            # EPGOptions with β=180 represents constant train
            θ_opt = DECAES.EPGOptions((; ETL, α, TE, T2, T1, β = T(180.0)))
            θ_cst = DECAES.EPGConstantFlipAngleOptions((; ETL, α, TE, T2, T1))
            θ_inc = DECAES.EPGIncreasingFlipAnglesOptions((; ETL, α, α1 = T(180.0), α2 = T(180.0), TE, T2, T1))

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
    θ = DECAES.EPGOptions((; ETL = 10, α = 169.0, TE = 9.0e-3, T2 = 10.1e-3, T1 = 0.98, β = 176.0))

    @testset "basics" begin
        @test Tuple(θ) == (10, 169.0, 9.0e-3, 10.1e-3, 0.98, 176.0)
        @test NamedTuple(θ) == (; ETL = 10, α = 169.0, TE = 9.0e-3, T2 = 10.1e-3, T1 = 0.98, β = 176.0)
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
    θ = DECAES.EPGOptions((; ETL, α = 169.0, TE = 9.0e-3, T2 = 10.1e-3, T1 = 0.98, β = 176.0))
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

@testset "EPG algorithms" test_EPG_algorithms()
@testset "EPG algorithm consistency" test_EPG_algorithm_consistency()
@testset "EPGOptions" test_EPGOptions()
@testset "EPGFunctor" test_EPGFunctor()

nothing
