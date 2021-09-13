if normpath(@__DIR__) ∉ LOAD_PATH
    pushfirst!(LOAD_PATH, normpath(@__DIR__, "../.."))
    pushfirst!(LOAD_PATH, normpath(@__DIR__))
end

using DECAES
using BenchmarkTools
using PrettyTables
using LibGit2

function commit_hash()
    repo = LibGit2.GitRepo(normpath(@__DIR__, "../.."))
    hash = LibGit2.GitShortHash(LibGit2.peel(LibGit2.GitCommit, LibGit2.head(repo)))
    return string(hash)
end

function build_suite()
    # Build suite of EPG algorithm benchmark groups
    suite = BenchmarkGroup()
    for ETL in [32,48,64]
        suite_ETL = suite["ETL=$ETL"] = BenchmarkGroup()
        for T in [Float32, Float64]
            suite_T = suite_ETL["T=$T"] = BenchmarkGroup()
            for alg in DECAES.EPG_Algorithms
                alg == DECAES.EPGWork_ReIm_Generated && continue
                α, TE, T2, T1, β = T(163.0), T(11e-3), T(39e-3), T(1.1), T(151.0)
                θ    = DECAES.EPGOptions{T,ETL}(α, TE, T2, T1, β)
                work = alg(T, ETL)
                suite_T["alg=$(nameof(alg))"] = @benchmarkable DECAES.EPGdecaycurve!($work, $θ)
            end
        end
    end
    return suite
end

# Display results
function default_pretty_table(io, data, header, row_names; backend = :text, kwargs...)
    is_minimum = (data,i,j) -> !isnan(data[i,j]) && data[i,j] ≈ minimum(filter(!isnan, data[i,:]))
    hl = PrettyTables.Highlighter(is_minimum, foreground = :blue, bold = true)
    PrettyTables.pretty_table(io, data, header; backend = :text, row_names, highlighters = (hl,), formatters = (v,i,j) -> round(v, sigdigits = 3), kwargs...)
end

function print_results(io, results)
    names, times = map(leaves(results)) do (name, results)
        name, time(minimum(results)) / 1000
    end |> xs -> ((x->x[1]).(xs), (x->x[2]).(xs))
    alg_names = (x->x[end]).(names) |> unique |> permutedims
    row_names = reshape(names, length(alg_names), :)[1,:] .|> x -> join(x[1:end-1], ", ")
    tbl_data  = reshape(times, length(alg_names), :) |> permutedims
    tbl_hdr   = vcat(alg_names, fill("Time [us]", size(alg_names)...))
    default_pretty_table(io, tbl_data, tbl_hdr, row_names; backend = :text)
end

function main()
    @info "building suite"
    suite = build_suite()

    @info "tuning suite"
    tune!(suite)

    @info "running benchmarks"
    results = run(suite; verbose = true, seconds = 1)

    @info "saving results"
    print_results(stdout, results)
    open(joinpath(@__DIR__, "epg_bench_$(commit_hash()).txt"); write = true) do io
        print_results(io, results)
    end
    return results
end

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

# v1.6.1
# ┌───────────────────┬─────────────────────────────────────────┬────────────────────────────────────┬────────────────────────────┬──────────────────────────────────┬────────────────────────────────────────┬────────────────────────┬─────────────────┬──────────────────┐
# │                   │ alg=EPGWork_ReIm_DualPaddedVector_Split │ alg=EPGWork_ReIm_DualMVector_Split │ alg=EPGWork_ReIm_DualCache │ alg=EPGWork_ReIm_DualCache_Split │ alg=EPGWork_ReIm_DualMVector_Vec_Split │ alg=EPGWork_Basic_Cplx │ alg=EPGWork_Vec │ alg=EPGWork_ReIm │
# │                   │                               Time [us] │                          Time [us] │                  Time [us] │                        Time [us] │                              Time [us] │              Time [us] │       Time [us] │        Time [us] │
# ├───────────────────┼─────────────────────────────────────────┼────────────────────────────────────┼────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼────────────────────────┼─────────────────┼──────────────────┤
# │ ETL=48, T=Float64 │                                   0.986 │                              0.821 │                      0.818 │                            0.802 │                                   1.17 │                   11.6 │            1.79 │             0.82 │
# │ ETL=48, T=Float32 │                                     1.0 │                              0.793 │                       0.81 │                            0.794 │                                  0.954 │                   13.1 │            1.85 │              0.8 │
# │ ETL=64, T=Float64 │                                    1.71 │                               1.39 │                       1.41 │                             1.39 │                                   2.01 │                   20.2 │            3.02 │             1.41 │
# │ ETL=64, T=Float32 │                                    1.74 │                               1.37 │                       1.41 │                             1.39 │                                   1.68 │                   23.1 │             3.4 │             1.41 │
# │ ETL=32, T=Float64 │                                   0.459 │                              0.375 │                      0.384 │                            0.376 │                                  0.542 │                   5.25 │           0.847 │            0.387 │
# │ ETL=32, T=Float32 │                                   0.473 │                              0.369 │                      0.379 │                            0.368 │                                  0.444 │                    5.8 │           0.794 │            0.377 │
# └───────────────────┴─────────────────────────────────────────┴────────────────────────────────────┴────────────────────────────┴──────────────────────────────────┴────────────────────────────────────────┴────────────────────────┴─────────────────┴──────────────────┘
