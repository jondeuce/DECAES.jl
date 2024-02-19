using Pkg
using Scratch

DECAES_UUID = Base.UUID("d84fb938-a666-558e-89d9-d531edc6724f")
DECAES_INFO = Pkg.dependencies()[DECAES_UUID]
DECAES_SOURCE = normpath(@__DIR__, "..")

if get(ENV, "DECAES_COMPILE_APP", nothing) == "true"
    DECAES_PROJECT_SOURCE = joinpath(DECAES_SOURCE, "api", "DECAESApp")
    DECAES_PROJECT_SCRATCH = get_scratch!(DECAES_UUID, "App")
else
    DECAES_PROJECT_SOURCE = joinpath(DECAES_SOURCE, "api", "DECAESCLI")
    DECAES_PROJECT_SCRATCH = get_scratch!(DECAES_UUID, "CLI")
end

@info "DECAES: Copying environment into a scratch directory: $(DECAES_PROJECT_SOURCE)"
ispath(DECAES_PROJECT_SCRATCH) && rm(DECAES_PROJECT_SCRATCH; force = true, recursive = true)
cp(DECAES_PROJECT_SOURCE, DECAES_PROJECT_SCRATCH; force = true)

@info "DECAES: Instantiating environment: $(DECAES_PROJECT_SCRATCH)"
cd(DECAES_PROJECT_SCRATCH)
Pkg.activate(DECAES_PROJECT_SCRATCH)
if get(ENV, "CI", nothing) == "true" || DECAES_INFO.version.prerelease == ("DEV",)
    @info "DECAES: Developing local version: $(DECAES_SOURCE)"
    Pkg.develop(; path = DECAES_SOURCE)
    Pkg.resolve()
end
Pkg.instantiate()
Pkg.status()

if get(ENV, "DECAES_COMPILE_APP", nothing) == "true"
    @info "DECAES: Building App"
    include(joinpath(DECAES_PROJECT_SCRATCH, "app_builder.jl"))
else
    @info "DECAES: Building CLI"
    include(joinpath(DECAES_PROJECT_SCRATCH, "cli_builder.jl"))
end
