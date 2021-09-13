using Test
using DECAES
using Logging

function compare_epg(
        work₁::DECAES.AbstractEPGWorkspace{T,ETL},
        work₂::DECAES.AbstractEPGWorkspace{T,ETL},
    ) where {T,ETL}
    α, TE, T2, T1, β = T(163.0), T(11e-3), T(39e-3), T(1.1), T(151.0)
    θ   = DECAES.EPGOptions{T,ETL}(α, TE, T2, T1, β)
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
    # NOTE: Generated function approach is extremely slow to compile for large ETL (around 16)
    epg_algs = DECAES.EPGWork_List
    for i in 1:length(epg_algs), j in 1:i-1, T in [Float32, Float64], ETL in [4,5,6,7] # test four ETL >= 4 which are unique mod 4
        algᵢ, algⱼ = epg_algs[i], epg_algs[j]
        compare_epg(algᵢ(T, ETL), algⱼ(T, ETL))
    end
end

nothing
