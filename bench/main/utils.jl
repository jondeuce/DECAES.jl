using Pkg

temp_project_folder() = joinpath(@__DIR__, ".bench.tmp")
temp_project_folder(julia_version::String, pkg_spec::String) = joinpath(temp_project_folder(), julia_version, pkg_spec)

function prepare_project(julia_version::String, pkg_spec::String)
    project_folder = temp_project_folder(julia_version, pkg_spec)
    if all(ispath, joinpath.(project_folder, ["Project.toml", "Manifest.toml"]))
        Pkg.activate(project_folder)
    else
        Pkg.activate(mkpath(project_folder))
        try
            Pkg.add(Pkg.PackageSpec(name = "DECAES", version = pkg_spec))
        catch e1
            try
                Pkg.add(Pkg.PackageSpec(name = "DECAES", rev = pkg_spec))
            catch e2
                @warn sprint(showerror, e1, catch_backtrace())
                @warn sprint(showerror, e2, catch_backtrace())
                @error "Invalid version specifier: $pkg_spec"
            end
        end
    end
    # Pkg.precompile() # no need to precompile; will occur during untimed warmup run
end

nothing
