module DECAESApp

using DECAES

"""
Entrypoint function for compiling DECAES into an executable [app](https://julialang.github.io/PackageCompiler.jl/dev/apps/).
"""
function julia_main()::Cint
    try
        DECAES.main(ARGS)
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end

end # module DECAESApp
