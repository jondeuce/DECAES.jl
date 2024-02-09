####
#### Load DaemonMode
####

# Load the Julia package manager, which is itself a Julia package
using Pkg

# Load the DaemonMode package
try
    @eval using DaemonMode
catch e
    println("* Installing DaemonMode package for DECAES server")
    Pkg.add("DaemonMode"; io = devnull)
    @eval using DaemonMode
end

# Server port number. This can be changed to any valid port
const PORT = DaemonMode.PORT

####
#### Utilities
####

# Julia command for starting server.
# Note: inherits current Julia environment - make sure number of threads
#       are set before running this script!
julia_cmd = ```
            $(Base.julia_cmd())
            --project=$(Base.active_project())
            --threads=$(Threads.nthreads())
            --startup-file=no
            --optimize=3
            --quiet
            ```;

# Convenience macro to write a Julia expression into a temporary script
macro mktempscript(ex = nothing)
    quote
        local fname = tempname() * ".jl"
        open(fname; write = true) do io
            println(io, "const PORT = $PORT")
            println(io, $(string(ex)))
            return nothing
        end
        fname
    end
end

####
#### Start DECAES server, if necessary
####

# Check for kill server command
if !isempty(ARGS) && lowercase(ARGS[1]) == "--kill"
    try
        sendExitCode(PORT)
        println("* DECAES server killed")
        exit()
    catch e
        if e isa Base.IOError
            println("* DECAES server inactive; nothing to kill")
            exit()
        else
            rethrow(e)
        end
    end
end

# Ping server by trying to delete dummy file
function ping()
    daemon_script = @mktempscript begin
        using DaemonMode
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                return runargs(PORT)
            end
        end
    end
    ping_script = @mktempscript rm(ARGS[1]; force = true)
    ping_file = touch(tempname())
    try
        run(`$(julia_cmd) $(daemon_script) $(ping_script) $(ping_file)`)
        return !isfile(ping_file)
    catch e
        if e isa ProcessFailedException
            return false
        else
            rethrow(e)
        end
    finally
        rm.([ping_file, daemon_script, ping_script]; force = true)
    end
end

if !ping()
    # Server is not started
    println("* Starting DECAES server")

    server_script = @mktempscript begin
        using DaemonMode
        serve(PORT)
    end
    server_cmd = `$(julia_cmd) $(server_script) \&`
    run(detach(server_cmd); wait = false)

    while !ping()
        sleep(1)
    end
end

####
#### Run DECAES on server
####

decaes_script = @mktempscript begin
    using Pkg
    try
        @eval using DECAES
    catch e
        println("* Installing DECAES")
        Pkg.add("DECAES"; io = devnull)
        # Pkg.develop("DECAES"; io = devnull)
        @eval using DECAES
    end

    # Call the command line interface entrypoint function
    main()
end

pushfirst!(ARGS, decaes_script)
runargs(PORT)
