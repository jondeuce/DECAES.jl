using BenchmarkTools, DECAES

# EPG workspace implementations to compare; see `test/epg.jl`.
const EPG_ALGORITHMS = Any[
    DECAES.EPGWork_Basic_Cplx,
    # DECAES.EPGWork_Vec,
    DECAES.EPGWork_ReIm,
    DECAES.EPGWork_ReIm_DualVector,
    DECAES.EPGWork_ReIm_DualVector_Split,
    DECAES.EPGWork_ReIm_DualVector_Split_Dynamic,
    DECAES.EPGWork_ReIm_DualFlat_Split_Dynamic,
    DECAES.EPGWork_ReIm_DualTuple_Split_Dynamic,
    DECAES.EPGWork_ReIm_Batched_Split_Dynamic,
    DECAES.EPGWork_ReIm_DualMVector_Split,
    # DECAES.EPGWork_ReIm_DualPaddedMVector_Vec_Split,
    DECAES.EPGWork_ReIm_DualPaddedVector_Split,
    # DECAES.EPGWork_ReIm_Generated,
]

const EPG_OPTION_TYPES = (DECAES.EPGOptions, DECAES.EPGConstantFlipAngleOptions, DECAES.EPGIncreasingFlipAnglesOptions)

mock_θ(::Type{DECAES.EPGOptions}, ::Type{T}, ETL) where {T} = DECAES.EPGOptions((; ETL, α = deg2rad(T(163.0)), TE = T(11e-3), T2 = T(39e-3), T1 = T(1.1), β = deg2rad(T(151.0))))
mock_θ(::Type{DECAES.EPGConstantFlipAngleOptions}, ::Type{T}, ETL) where {T} = DECAES.EPGConstantFlipAngleOptions((; ETL, α = deg2rad(T(163.0)), TE = T(11e-3), T2 = T(39e-3), T1 = T(1.1)))
mock_θ(::Type{DECAES.EPGIncreasingFlipAnglesOptions}, ::Type{T}, ETL) where {T} = DECAES.EPGIncreasingFlipAnglesOptions((; ETL, α = deg2rad(T(163.0)), α1 = deg2rad(T(163.0)), α2 = deg2rad(T(140.0)), TE = T(11e-3), T2 = T(39e-3), T1 = T(1.1)))

const SUITE = BenchmarkGroup()

for ETL in [8, 16, 32, 48, 64], T in [Float32, Float64], Opt in EPG_OPTION_TYPES, alg in EPG_ALGORITHMS
    work = alg(T, ETL)
    θ = mock_θ(Opt, T, DECAES.echotrainlength(work))
    applicable(DECAES.epg_decay_curve!, DECAES.decaycurve(work), work, θ) || continue
    SUITE["ETL=$ETL"]["T=$T"]["θ=$(nameof(Opt))"]["alg=$(nameof(alg))"] = @benchmarkable DECAES.EPGdecaycurve!($work, $θ)
end

if isinteractive()
    BenchmarkTools.tune!(SUITE; verbose = true)
    BenchmarkTools.run(SUITE; verbose = true)
end
