using BenchmarkTools, DECAES
const SUITE = BenchmarkGroup()
const SETTINGS_FILES = readdir(joinpath(@__DIR__, "settings"); join = true)

@assert Threads.nthreads() > 1 "This benchmark requires multiple threads"

# Build suite of EPG algorithm benchmark groups
for (i, settings) in enumerate(SETTINGS_FILES)
    suite = SUITE[basename(settings)] = BenchmarkGroup()
    args = ["@" * settings, "--quiet"]
    if i == 1
        # Measure overhead of first run
        suite["main (first run)"] = @benchmarkable DECAES.main($args) samples=1 evals=1
    end
    # Measure performance of runs after compilation
    suite["main"] = @benchmarkable DECAES.main($args) samples=3 evals=1 seconds=3600
end

if isinteractive()
    BenchmarkTools.run(SUITE; verbose = true)
end
