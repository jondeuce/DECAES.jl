# Typically causes a lot of false positives with ambiguities and/or unbound args checks; unfortunately have to periodically check this manually
Aqua.test_all(DECAES; ambiguities = false, unbound_args = true, stale_deps = (; ignore = [:Pkg, :Scratch])) # `Pkg` and `Scratch` are used only by `deps/build.jl`.
