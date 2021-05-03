using DECAES, PackageCompiler

"""
    build_decaes(build_path::String; kwargs...)

Build DECAES into an executable [app](https://julialang.github.io/PackageCompiler.jl/dev/apps/).
By default, `build_path` points to the folder "decaes" in the working directory.
Building will error if `build_path` exists, unless the keyword argument `force = true` is passed.
All keyword arguments `kwargs` are forwarded to `PackageCompiler.create_app`.

Example usage:

```julia-repl
julia> using DECAES, PackageCompiler

julia> include(joinpath(pkgdir(DECAES), "api", "compile", "decaes_builder.jl"))

julia> build_decaes()
```
"""
function build_decaes(
        build_path = joinpath(pwd(), "decaes");
        kwargs...,
    )
    create_app(
        pkgdir(DECAES),
        build_path;
        app_name = "decaes",
        precompile_execution_file = joinpath(pkgdir(DECAES), "api", "compile", "decaes_precompile.jl"),
        audit = false,
        force = false,
        kwargs...
    )
    @info "DECAES: build complete."
    @info "DECAES: executable binary can be found here: $(joinpath(build_path, "bin", "decaes"))"
end

nothing
