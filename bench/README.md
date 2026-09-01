# Benchmarks

Tools for comparing DECAES revisions.
These are not part of the test suite.

## `main/`

End-to-end command-line benchmarks across DECAES revisions, Julia versions, thread counts, and optimization levels using [hyperfine](https://github.com/sharkdp/hyperfine).
Copy/modify `settings-template.txt`, set the paths, and run:

```bash
julia --startup-file=no --project=bench/main bench/main/main.jl @/path/to/settings.txt
```

Each run creates a timestamped directory containing `results.md`, `results.json`, the resolved settings, and copies of the scripts.
`plot.jl` loads the results under `main/results` and plots runtimes and speedups relative to `compare_version`.

## `cli/` and `epg/`

[AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl) benchmark suites.
Each directory contains a `bench_*.jl` suite and a `bench_*.sh` driver.
The drivers use the `benchpkg`, `benchpkgplot`, and `benchpkgtable` executables and compare `master,nnls` by default.
Override the revisions with `REVISIONS`:

```bash
REVISIONS=v0.6.0,master bench/epg/bench_epg.sh
```

Loading a suite in an interactive REPL runs it directly.

`cli/` benchmarks the full pipeline on real images.
Create `cli/settings/` from `cli/settings-template.txt`, or set `DECAES_BENCH_SETTINGS` to another directory.

`epg/` compares the supported `AbstractEPGWorkspace` implementations across echo-train lengths, numeric types, and flip-angle parameterizations.
