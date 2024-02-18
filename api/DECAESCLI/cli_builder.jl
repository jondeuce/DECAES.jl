using Pkg
Pkg.resolve()
Pkg.instantiate()
Pkg.status()

using DECAES: DECAES
using Scratch: get_scratch!

const DECAES_CLI_PROJECT = get_scratch!(DECAES, "CLI")

# The below code was modified from Comonicon.jl:
#   https://github.com/comonicon/Comonicon.jl/blob/49e4972b61c9f08338a998f081a836c6af103639/src/builder/install.jl

function install()
    homedepot = first(DEPOT_PATH)
    bin = mkpath(joinpath(homedepot, "bin"))

    cli = joinpath(bin, "decaes")
    if Sys.iswindows()
        cli = cli * ".cmd"
    end

    ispath(cli) && rm(cli; force = true, recursive = true)
    open(cli; write = true) do io
        print(io, cli_script())
        return nothing
    end
    chmod(cli, 0o777)

    return nothing
end

function cli_script()
    exe = joinpath(Sys.BINDIR, Base.julia_exename())

    cmds = String[]
    push!(cmds, exe)
    push!(cmds, "--threads=auto")
    push!(cmds, "--startup-file=no")
    push!(cmds, "--color=yes")

    if Sys.iswindows()
        push!(cmds, "-e \"using DECAES; main()\" %*")
    else
        push!(cmds, "-- \"\${BASH_SOURCE[0]}\" \"\$@\"")
    end

    if Sys.iswindows()
        """
        @echo off
        setlocal
        set JULIA_PROJECT=$(DECAES_CLI_PROJECT)
        $(join(cmds, " ^\n    "))
        if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
        endlocal
        """
    else
        """
        #!/usr/bin/env bash
        #=
        JULIA_PROJECT=$(DECAES_CLI_PROJECT) \\
        exec $(join(cmds, " \\\n    "))
        =#
        using DECAES
        main()
        """
    end
end

if !isinteractive()
    install()
end

nothing
