using BenchmarkTools, DECAES
const SUITE = BenchmarkGroup()

# Build suite of EPG algorithm benchmark groups
for ETL in [8, 16, 32, 48, 64]
    SUITE_ETL = SUITE["ETL=$ETL"] = BenchmarkGroup()
    for T in [Float32, Float64]
        SUITE_T = SUITE_ETL["T=$T"] = BenchmarkGroup()
        for alg in DECAES.EPG_Algorithms
            alg == DECAES.EPGWork_ReIm_Generated && continue
            α, TE, T2, T1, β = T(163.0), T(11e-3), T(39e-3), T(1.1), T(151.0)
            work = alg(T, ETL)
            θETL = Val(something(DECAES.echotrainlength(work), ETL))
            θ = DECAES.EPGOptions((; α, TE, T2, T1, β), θETL, T)
            SUITE_T["alg=$(nameof(alg))"] = @benchmarkable DECAES.EPGdecaycurve!($work, $θ)
        end
    end
end

if isinteractive()
    BenchmarkTools.run(SUITE; verbose = true)
end
