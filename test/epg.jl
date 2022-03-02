using Test
using DECAES
using Logging

function compare_epg(
        work₁::DECAES.AbstractEPGWorkspace{T,ETL},
        work₂::DECAES.AbstractEPGWorkspace{T,ETL},
    ) where {T,ETL}
    α, TE, T2, T1, β = T(163.0), T(11e-3), T(39e-3), T(1.1), T(151.0)
    θ   = DECAES.EPGOptions((; α, TE, T2, T1, β), Val(ETL), T)
    dc₁ = DECAES.EPGdecaycurve!(work₁, θ)
    dc₂ = DECAES.EPGdecaycurve!(work₂, θ)
    if !(dc₁ ≈ dc₂)
        @info "Comparing: $((nameof(typeof(work₁)), nameof(typeof(work₂))))"
        @info "    max error:   $(maximum(abs, dc₁ .- dc₂))"
        @info "    diff vector: $(round.(abs.(dc₁ .- dc₂)'; sigdigits = 4))"
    end
    @test dc₁ ≈ dc₂
end

@testset "EPG algorithms" begin
    # In principle ETL testing range need not be too large (only need to test four ETL values >=4 which are unique mod 4),
    # but since this file is also used for app precompilation, we should sweep over ETL values we expect to see in practice
    # for the default algorithm.
    #   NOTE: Generated function approach is extremely slow to compile for large ETL (around 16)
    epg_algs = DECAES.EPG_Algorithms
    for T in [Float32, Float64]
        for i in 1:length(epg_algs), ETL in [4,5,6,7]
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

nothing
