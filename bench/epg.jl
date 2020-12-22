using DECAES
using BenchmarkTools
using PrettyTables

module BenchEPG

using ..DECAES
using ..BenchmarkTools

const suite = BenchmarkGroup()

const epg_algs = [
    DECAES.EPGWork_Basic_Cplx,
    DECAES.EPGWork_Vec,
    DECAES.EPGWork_Cplx,
    DECAES.EPGWork_Cplx_Vec_Unrolled,
    DECAES.EPGWork_Fused_Cplx,
]

for ETL in [32,48,64]
    s_ETL = suite["ETL=$ETL"] = BenchmarkGroup()
    for T in [Float32, Float64]
        s_T = s_ETL["T=$T"] = BenchmarkGroup()
        for alg in epg_algs
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

end  # module

# Tune and run benchmarks
tune!(BenchEPG.suite)
results = run(BenchEPG.suite; verbose = true, seconds = 1)
map(((name,res),) -> (@info(name); display(res)), leaves(results))
display(results)

# Display results
function default_pretty_table(io, data, header, row_names; backend = :text, kwargs...)
    is_minimum = (data,i,j) -> !isnan(data[i,j]) && data[i,j] ≈ minimum(filter(!isnan, data[i,:]))
    hl = PrettyTables.Highlighter(is_minimum, foreground = :blue, bold = true)
    PrettyTables.pretty_table(io, data, header; backend = :text, row_names, highlighters = (hl,), formatters = (v,i,j) -> round(v, sigdigits = 3), kwargs...)
end

let
    names, times = map(leaves(results)) do (name, res)
        name, time(minimum(res)) / 1000
    end |> xs -> ((x->x[1]).(xs), (x->x[2]).(xs))
    alg_names = (x->x[end]).(names) |> unique |> permutedims
    row_names = reshape(names, length(alg_names), :)[1,:] .|> x -> join(x[1:end-1], ", ")
    tbl_data  = reshape(times, length(alg_names), :) |> permutedims
    tbl_hdr   = vcat(alg_names, fill("Time [us]", size(alg_names)...))
    default_pretty_table(stdout, tbl_data, tbl_hdr, row_names; backend = :text)
end

nothing
