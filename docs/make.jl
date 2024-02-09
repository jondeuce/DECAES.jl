try
    @eval using Pkg, Documenter, DECAES
catch e
    Pkg.develop(PackageSpec(; path = joinpath(@__DIR__, "..")))
    Pkg.instantiate()
    @eval using Pkg, Documenter, DECAES
end

makedocs(;
    modules = [DECAES],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    sitename = "DECAES.jl",
    authors = "Jonathan Doucette",
    pages = [
        "Home" => "index.md",
        "cli.md",
        "ref.md",
    ],
)

deploydocs(;
    repo = "github.com/jondeuce/DECAES.jl.git",
    push_preview = true,
    deploy_config = Documenter.GitHubActions(),
)
