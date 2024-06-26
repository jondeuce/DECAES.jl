@testset "version numbers" begin
    DECAES_PATH = pkgdir(DECAES)
    decaes_proj = TOML.parsefile(joinpath(DECAES_PATH, "Project.toml"))
    decaes_app_proj = TOML.parsefile(joinpath(DECAES_PATH, "api", "DECAESApp", "Project.toml"))
    decaes_cli_proj = TOML.parsefile(joinpath(DECAES_PATH, "api", "DECAESCLI", "Project.toml"))

    @test DECAES.VERSION == VersionNumber(decaes_proj["version"])
    @test DECAES.VERSION == VersionNumber(decaes_app_proj["version"])

    (; major, minor, patch, prerelease) = DECAES.VERSION
    @test startswith(decaes_app_proj["compat"]["DECAES"], "=")
    @test startswith(decaes_cli_proj["compat"]["DECAES"], "=")
    @test VersionNumber(major, minor, patch) == VersionNumber(split(decaes_app_proj["compat"]["DECAES"], '=')[2])
    @test VersionNumber(major, minor, patch) == VersionNumber(split(decaes_cli_proj["compat"]["DECAES"], '=')[2])
    if prerelease != ()
        @test prerelease == ("DEV",)
    end

    for file in ["decaes.m", "decaes_pyjulia.py", "decaes_pyjuliacall.py", "decaes.sh"]
        contents = readchomp(joinpath(DECAES_PATH, "api", file))
        @test contains(contents, "was written for DECAES v$(DECAES.VERSION)")
    end
end

@testset "cli script" begin
    homedepot = first(DEPOT_PATH)
    bin = mkpath(joinpath(homedepot, "bin"))
    if Sys.iswindows()
        cli = joinpath(bin, "decaes.cmd")
        @test isfile(cli)
        @test success(`cmd /c $(cli) --help`)
    else
        cli = joinpath(bin, "decaes")
        @test isfile(cli)
        @test success(`bash -c "$(cli) --help"`)
    end
end
