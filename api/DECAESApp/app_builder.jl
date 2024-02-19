using DECAESApp
using PackageCompiler

function build(; create_symlink = true)
    @info "DECAES: Starting build."
    package_dir = pkgdir(DECAESApp)
    compiled_app = joinpath(package_dir, "build")
    exe = joinpath(compiled_app, "bin", "decaes")

    try
        create_app(
            package_dir,
            compiled_app;
            executables = ["decaes" => "julia_main"],
            precompile_execution_file = joinpath(package_dir, "app_precompile.jl"),
            incremental = false,
            filter_stdlibs = false,
            force = true,
            include_lazy_artifacts = true,
            include_transitive_dependencies = true,
        )
        @info "DECAES: Build complete."
        @info "DECAES: Executable binary can be found here: $(exe)"
    catch e
        rm(compiled_app; force = true, recursive = true)
        rethrow()
    end

    if create_symlink
        bin = mkpath(joinpath(homedir(), ".julia", "bin"))
        link = joinpath(bin, "decaes")
        rm(link; force = true)
        symlink(exe, link; dir_target = false)
        @info "DECAES: Symbolic link created: $(link)"
    end
end

if !isinteractive()
    build()
end

nothing
