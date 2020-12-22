using DECAES
using Test
using Logging

@testset "EPG algorithms" begin
    epg_algs = [
        DECAES.EPGWork_Basic_Cplx,
        DECAES.EPGWork_Vec,
        DECAES.EPGWork_Cplx,
        DECAES.EPGWork_Cplx_Vec_Unrolled,
    ]
    for ((i, algᵢ), (j, algⱼ)) in Iterators.product(enumerate(epg_algs), enumerate(epg_algs))
        for T in (Float32, Float64)
            j < i || continue
            @info i, j, T, algᵢ, algⱼ
            ETL = rand(4:64)
            α  = T(90.0) + T(90.0) * rand(T)
            TE = T(5e-3) + T(5e-3) * rand(T)
            T2 = T(10e-3) + T(190e-3) * rand(T)
            T1 = T(0.8) + T(0.4) * rand(T)
            β  = T(90.0) + T(90.0) * rand(T)
            y1 = DECAES.EPGdecaycurve!(algᵢ(T, ETL), α, TE, T2, T1, β)
            y2 = DECAES.EPGdecaycurve!(algⱼ(T, ETL), α, TE, T2, T1, β)
            @test y1 ≈ y2
        end
    end
end

