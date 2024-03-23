function compare_epg(
    work₁::DECAES.AbstractEPGWorkspace{T},
    work₂::DECAES.AbstractEPGWorkspace{T};
    verbose = false,
) where {T}
    xs = (; α = T(11e-3), TE = T(39e-3), T2 = T(1.1), T1 = T(151.0), β = T(163.0))
    θ₁ = DECAES.EPGOptions((; ETL = DECAES.echotrainlength(work₁), xs...))
    θ₂ = DECAES.EPGOptions((; ETL = DECAES.echotrainlength(work₂), xs...))
    dc₁ = DECAES.EPGdecaycurve!(work₁, θ₁)
    dc₂ = DECAES.EPGdecaycurve!(work₂, θ₂)
    if verbose && !(dc₁ ≈ dc₂)
        @info "Comparing: $((nameof(typeof(work₁)), nameof(typeof(work₂))))"
        @info "    max error:   $(maximum(abs, dc₁ .- dc₂))"
        @info "    diff vector: $(abs.(dc₁ .- dc₂)')"
    end
    @test isapprox(dc₁, dc₂; rtol = √eps(T), atol = 10 * eps(T))
end

function test_EPG_algorithms()
    # In principle ETL testing range need not be too large (only need to test four ETL values >=4 which are unique mod 4),
    # but since this file is also used for app precompilation, we should sweep over ETL values we expect to see in practice
    # for the default algorithm.
    #   NOTE: Generated function approach is extremely slow to compile for large ETL (around 16)
    epg_algs = DECAES.EPG_Algorithms
    for T in [Float32, Float64]
        for i in 1:length(epg_algs), ETL in [4, 5, 6, 7]
            j = rand([1:i-1; i+1:length(epg_algs)])
            algᵢ, algⱼ = epg_algs[i], epg_algs[j]
            compare_epg(algᵢ(T, ETL), algⱼ(T, ETL))
        end
        for ETL in 4:64
            alg = DECAES.EPGdecaycurve_work(T, ETL)
            compare_epg(alg, alg)
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
        @test @allocated(DECAES.restructure(θ, (2.0, 1.0), Val((:T1, :TE)))) == 0
        @test @allocated(DECAES.destructure(θ, Val((:T2, :ETL)))) == 0

        θ′ = DECAES.restructure(θ, (2.0, 1.0), Val((:β, :α)))
        @test Tuple(θ′) == (θ.ETL, 1.0, θ.TE, θ.T2, θ.T1, 2.0)

        x′ = DECAES.destructure(θ, Val((:TE, :α)))
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
    @test @allocated(fun!(y, x)) > 0 # Dual cache created on first call
    @test @allocated(fun!(y, x)) == 0 # Dual cache reused on second call
    @test y == DECAES.EPGdecaycurve(θ)

    # EPGJacobianFunctor
    y .= 0
    J = zeros(T, ETL, 2)
    @test @allocated(jac!(y, θ)) > 0 # Dual cache created on first call
    @test @allocated(jac!(J, y, θ)) == 0 # Dual cache reused on second call
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
@testset "EPGOptions" test_EPGOptions()
@testset "EPGFunctor" test_EPGFunctor()

nothing
