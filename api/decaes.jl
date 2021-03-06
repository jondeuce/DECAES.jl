# Load the Julia package manager, which is itself a Julia package
using Pkg

try
    # Load DECAES
    @eval using DECAES
catch e
    # DECAES not found; install DECAES and try loading again
    Pkg.add("DECAES")
    @eval using DECAES
end

# Call the command line interface entrypoint function
main()
