# Typically causes a lot of false positives with ambiguities and/or unbound args checks; unfortunately have to periodically check this manually
Aqua.test_all(DECAES; ambiguities = false, unbound_args = true, stale_deps = (; ignore = [:Scratch])) # `Scratch` is used only by `deps/build.jl`.
