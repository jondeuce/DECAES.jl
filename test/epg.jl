using DECAES
using Test
using Logging

@testset "EPG algorithms" begin
    for ((i, algᵢ), (j, algⱼ)) in Iterators.product(enumerate(DECAES.EPGWork_List), enumerate(DECAES.EPGWork_List))
        for T in (Float32, Float64)
            j < i || continue
            # @info i, j, T, algᵢ, algⱼ
            for ETL in [6,7,11,12,31,32,63,64]
                α  = T(1.0) + T(179.0) * rand(T)
                TE = T(5e-3) + T(5e-3) * rand(T)
                T2 = T(10e-3) + T(190e-3) * rand(T)
                T1 = T(0.8) + T(0.4) * rand(T)
                β  = T(1.0) + T(179.0) * rand(T)
                y1 = DECAES.EPGdecaycurve!(algᵢ(T, ETL), α, TE, T2, T1, β)
                y2 = DECAES.EPGdecaycurve!(algⱼ(T, ETL), α, TE, T2, T1, β)
                # if !(y1 ≈ y2)
                #     @info "max error = $(maximum(abs, y1 .- y2))"
                #     @info "diff = $(round.(abs.(y1 .- y2)'; sigdigits = 4))"
                # end
                @test y1 ≈ y2
            end
        end
    end
end

nothing
