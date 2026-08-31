using BenchmarkTools, DECAES

# Each file in this folder is a DECAES settings file, passed to `main` as `@<file>`; see `settings-template.txt`.
const SETTINGS_DIR = get(ENV, "DECAES_BENCH_SETTINGS", joinpath(@__DIR__, "settings"))
@assert isdir(SETTINGS_DIR) "Settings folder not found: $SETTINGS_DIR. Create it, or point DECAES_BENCH_SETTINGS elsewhere."
@assert Threads.nthreads() > 1 "This benchmark requires multiple threads"

const SUITE = BenchmarkGroup()

for (i, settings) in enumerate(filter(isfile, readdir(SETTINGS_DIR; join = true)))
    suite = SUITE[basename(settings)] = BenchmarkGroup()
    args = ["@" * settings, "--quiet"]

    # The first run includes compilation, and is reported separately.
    i == 1 && (suite["main (first run)"] = @benchmarkable DECAES.main($args) samples = 1 evals = 1)
    suite["main"] = @benchmarkable DECAES.main($args) samples = 5 evals = 1 seconds = 3600
end

if isinteractive()
    BenchmarkTools.run(SUITE; verbose = true)
end
