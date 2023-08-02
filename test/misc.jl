@testset "version numbers" begin
    DECAES_PATH = pkgdir(DECAES)
    decaes_proj = TOML.parsefile(joinpath(DECAES_PATH, "Project.toml"))
    decaes_app_proj = TOML.parsefile(joinpath(DECAES_PATH, "api", "DECAESApp", "Project.toml"))

    @test DECAES.VERSION == VersionNumber(decaes_proj["version"])
    @test DECAES.VERSION == VersionNumber(decaes_app_proj["version"])
    @test DECAES.VERSION == VersionNumber(decaes_app_proj["compat"]["DECAES"])

    for file in ["decaes.m", "decaes.py", "decaes.sh"]
        contents = readchomp(joinpath(DECAES_PATH, "api", file))
        @test contains(contents, "was written for DECAES v$(DECAES.VERSION)")
    end
end
