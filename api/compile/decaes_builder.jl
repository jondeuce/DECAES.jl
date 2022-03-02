# DECAES is assumed to be installed in the active project environment
using Pkg
using DECAES

# Copy Project.toml file to a temporary project folder and instantiate
Pkg.activate(; temp = true, io = devnull)
cp(joinpath(@__DIR__, "Project.toml"), Base.active_project(); force = true)
Pkg.instantiate(; io = devnull)

using PackageCompiler

"""
    build_decaes(build_path::String; kwargs...)

Build DECAES into an executable [app](https://julialang.github.io/PackageCompiler.jl/stable/apps.html).
By default, `build_path` points to the folder "decaes_app" in the working directory.
Building will error if `build_path` exists, unless the keyword argument `force = true` is passed.
All keyword arguments `kwargs` are forwarded to `PackageCompiler.create_app`.
```
"""
function build_decaes(
        build_path = joinpath(pwd(), "decaes_app");
        kwargs...,
    )
    @assert !ispath(build_path) "The following path already exists and will not be overwritten: $(build_path)"
    create_app(
        pkgdir(DECAES),
        build_path;
        executables = ["decaes" => "julia_main"],
        precompile_execution_file = joinpath(pkgdir(DECAES), "api", "compile", "decaes_precompile.jl"),
        include_lazy_artifacts = true,
        incremental = false,
        filter_stdlibs = false,
        force = false,
        kwargs...,
    )
    @info "DECAES: build complete."
    @info "DECAES: executable binary can be found here: $(joinpath(build_path, "bin", "decaes"))"
end

if !isinteractive()
    build_decaes()
end

nothing
