using Pkg
Pkg.instantiate()
Pkg.status()

using PackageCompiler
using DECAESApp

"""
    build(build_path::String; kwargs...)

Build DECAES into an executable [app](https://julialang.github.io/PackageCompiler.jl/stable/apps.html).
By default, `build_path` points to the folder "decaes_app" in the working directory.
Building will error if `build_path` exists, unless the keyword argument `force = true` is passed.
All keyword arguments `kwargs` are forwarded to `PackageCompiler.create_app`.
```
"""
function build(
        build_path = joinpath(pwd(), "decaes_app");
        kwargs...,
    )
    @assert !ispath(build_path) "The following path already exists and will not be overwritten: $(build_path)"
    create_app(
        pkgdir(DECAESApp),
        build_path;
        executables = ["decaes" => "julia_main"],
        precompile_execution_file = joinpath(pkgdir(DECAESApp), "app_precompile.jl"),
        include_lazy_artifacts = true,
        incremental = false,
        filter_stdlibs = false,
        force = false,
        kwargs...,
    )
    @info "DECAES: Build complete."
    @info "DECAES: Executable binary can be found here: $(joinpath(build_path, "bin", "decaes"))"
end

if !isinteractive()
    build()
end

nothing
