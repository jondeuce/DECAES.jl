import Pkg
Pkg.activate(@__DIR__)
using DECAES
using BenchmarkTools
using PrettyTables

suite = BenchmarkGroup()

for ETL in [32,48,64]
    s_ETL = suite["ETL=$ETL"] = BenchmarkGroup()
    for T in [Float32, Float64]
        s_T = s_ETL["T=$T"] = BenchmarkGroup()
        for alg in DECAES.EPGWork_List
            α    = T(90.0) + T(90.0) * rand(T)
            TE   = T(5e-3) + T(5e-3) * rand(T)
            T2   = T(10e-3) + T(190e-3) * rand(T)
            T1   = T(0.8) + T(0.4) * rand(T)
            β    = T(90.0) + T(90.0) * rand(T)
            work = alg(T, ETL)
            opts = DECAES.EPGOptions{T,ETL}(α, TE, T2, T1, β)
            s_T["alg=$(nameof(alg))"] = @benchmarkable DECAES.EPGdecaycurve!($work, $opts)
        end
    end
end

# Tune and run benchmarks
tune!(suite)
results = run(suite; verbose = true, seconds = 1)
map(((name,res),) -> (@info(name); display(res)), leaves(results))
display(results)

# Display results
function default_pretty_table(io, data, header, row_names; backend = :text, kwargs...)
    is_minimum = (data,i,j) -> !isnan(data[i,j]) && data[i,j] ≈ minimum(filter(!isnan, data[i,:]))
    hl = PrettyTables.Highlighter(is_minimum, foreground = :blue, bold = true)
    PrettyTables.pretty_table(io, data, header; backend = :text, row_names, highlighters = (hl,), formatters = (v,i,j) -> round(v, sigdigits = 3), kwargs...)
end

function main()
    names, times = map(leaves(results)) do (name, res)
        name, time(minimum(res)) / 1000
    end |> xs -> ((x->x[1]).(xs), (x->x[2]).(xs))
    alg_names = (x->x[end]).(names) |> unique |> permutedims
    row_names = reshape(names, length(alg_names), :)[1,:] .|> x -> join(x[1:end-1], ", ")
    tbl_data  = reshape(times, length(alg_names), :) |> permutedims
    tbl_hdr   = vcat(alg_names, fill("Time [us]", size(alg_names)...))
    default_pretty_table(stdout, tbl_data, tbl_hdr, row_names; backend = :text)
    return nothing
end

main()

# v1.5.3
# ┌───────────────────┬──────────────────┬───────────────────────────────┬────────────────────────────┬──────────────────────────────────┬────────────────────────┬─────────────────┬──────────────────┬─────────────────────────────────────┐
# │                   │ alg=EPGWork_ReIm │ alg=EPGWork_Cplx_Vec_Unrolled │ alg=EPGWork_ReIm_DualCache │ alg=EPGWork_ReIm_DualCache_Split │ alg=EPGWork_Basic_Cplx │ alg=EPGWork_Vec │ alg=EPGWork_Cplx │ alg=EPGWork_ReIm_DualCache_Unrolled │
# │                   │        Time [us] │                     Time [us] │                  Time [us] │                        Time [us] │              Time [us] │       Time [us] │        Time [us] │                           Time [us] │
# ├───────────────────┼──────────────────┼───────────────────────────────┼────────────────────────────┼──────────────────────────────────┼────────────────────────┼─────────────────┼──────────────────┼─────────────────────────────────────┤
# │ ETL=48, T=Float64 │            0.833 │                          2.56 │                      0.844 │                            0.831 │                   6.28 │            1.97 │             3.86 │                               0.962 │
# │ ETL=48, T=Float32 │            0.822 │                          3.13 │                      0.825 │                            0.816 │                   6.02 │            1.85 │              3.7 │                               0.954 │
# │ ETL=64, T=Float64 │             1.45 │                          4.31 │                       1.54 │                             1.44 │                   10.5 │            3.33 │             6.62 │                                 1.6 │
# │ ETL=64, T=Float32 │             1.44 │                          4.95 │                       1.44 │                             1.46 │                   10.1 │             3.2 │             6.43 │                                1.61 │
# │ ETL=32, T=Float64 │            0.395 │                          1.34 │                      0.396 │                             0.39 │                   3.13 │           0.957 │             1.93 │                               0.487 │
# │ ETL=32, T=Float32 │            0.391 │                          1.51 │                      0.387 │                            0.381 │                   2.94 │           0.891 │             1.82 │                               0.484 │
# └───────────────────┴──────────────────┴───────────────────────────────┴────────────────────────────┴──────────────────────────────────┴────────────────────────┴─────────────────┴──────────────────┴─────────────────────────────────────┘

# v1.6.0-beta1
# ┌───────────────────┬────────────────────────────────────┬────────────────────────────┬──────────────────────────────────┬────────────────────────┬──────────────────┬──────────────────┬─────────────────────────────────────┐
# │                   │ alg=EPGWork_ReIm_DualMVector_Split │ alg=EPGWork_ReIm_DualCache │ alg=EPGWork_ReIm_DualCache_Split │ alg=EPGWork_Basic_Cplx │ alg=EPGWork_Cplx │ alg=EPGWork_ReIm │ alg=EPGWork_ReIm_DualCache_Unrolled │
# │                   │                          Time [us] │                  Time [us] │                        Time [us] │              Time [us] │        Time [us] │        Time [us] │                           Time [us] │
# ├───────────────────┼────────────────────────────────────┼────────────────────────────┼──────────────────────────────────┼────────────────────────┼──────────────────┼──────────────────┼─────────────────────────────────────┤
# │ ETL=48, T=Float64 │                               0.79 │                      0.808 │                            0.824 │                   10.7 │             2.99 │             0.84 │                               0.919 │
# │ ETL=48, T=Float32 │                              0.814 │                      0.832 │                            0.816 │                   13.0 │             3.47 │             0.83 │                               0.875 │
# │ ETL=64, T=Float64 │                               1.43 │                       1.46 │                             1.44 │                   19.0 │             5.05 │             1.46 │                                1.54 │
# │ ETL=64, T=Float32 │                               1.43 │                       1.45 │                             1.43 │                   23.2 │              7.4 │             1.45 │                                1.49 │
# │ ETL=32, T=Float64 │                              0.389 │                      0.397 │                            0.388 │                   4.94 │             1.54 │            0.399 │                               0.465 │
# │ ETL=32, T=Float32 │                              0.381 │                      0.392 │                            0.381 │                   5.92 │             1.65 │            0.389 │                               0.431 │
# └───────────────────┴────────────────────────────────────┴────────────────────────────┴──────────────────────────────────┴────────────────────────┴──────────────────┴──────────────────┴─────────────────────────────────────┘
