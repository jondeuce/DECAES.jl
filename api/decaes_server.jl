####
#### Utilities
####

# Julia command for starting server.
#   Note: inherits current environment - make sure number of threads
#         are set before running this script!
julia_cmd =
    ```
    $(Base.julia_cmd())
    --project=$(Base.active_project())
    --threads=$(Threads.nthreads())
    --startup-file=no
    --optimize=3
    --quiet
    ```;

# Convenience macro to write a Julia expression into a temporary script
# which will then be run on the DECAES server
macro mktempscript(ex = nothing, filename = nothing)
    quote
        local fname = $filename === nothing ? tempname() * ".jl" : $filename
        open(fname; write = true) do io
            println(io, $(string(ex)))
        end
        fname
    end
end

####
#### Start DECAES server, if necessary
####

# Load the Julia package manager, which is itself a Julia package
import Pkg

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

# Check if server is alive by sending a dummy script to run
tryconnect() =
    try
        ping_script = @mktempscript begin
            using DaemonMode
            redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    runfile(touch(tempname() * ".jl"))
                end
            end
        end
        run(`$(julia_cmd) $(ping_script)`)
        return true
    catch e
        if e isa ProcessFailedException
            return false
        else
            rethrow(e)
        end
    end

if !tryconnect()
    # Server is not started
    println("* Starting DECAES server")

    server_script = @mktempscript begin
        using DaemonMode
        serve()
    end
    server_cmd = `$(julia_cmd) $(server_script) \&`
    run(detach(server_cmd); wait = false)
    sleep(1)

    local tries, max_tries, delay = 0, 5, 0.5
    while !tryconnect()
        sleep(delay)
        delay *= 2
        tries += 1
        tries >= max_tries && error("Failed to connect to DECAES server")
    end
end

####
#### Run DECAES on server
####

decaes_script = @mktempscript begin
    import Pkg
    try
        @eval using DECAES
    catch e
        println("* Installing DECAES")
        # Pkg.add("DECAES"; io = devnull)
        Pkg.develop("DECAES"; io = devnull) #TODO
        @eval using DECAES
    end

    # Call the command line interface entrypoint function
    main()
end

pushfirst!(ARGS, decaes_script)
runargs()
