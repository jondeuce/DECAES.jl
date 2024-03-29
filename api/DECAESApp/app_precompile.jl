using DECAESApp
using DECAESApp.DECAES

#### Precompiling seems unneccessary with PrecompileTools.jl

#=
# Silence console output
DECAES.redirect_to_devnull() do
    # Run code to be precompiled which is not included in test suite coverage
    main(["--help"])
    main(["--version"])
    for Reg in ["lcurve", "gcv", "chi2"]
        NumVoxels = max(4, Threads.nthreads()) * DECAES.default_blocksize()
        DECAES.mock_T2_pipeline(; MatrixSize = (NumVoxels, 1, 1), Reg)
    end

    # Run test suite, excluding Matlab MWI Toolbox compatibility tests
    ENV["DECAES_RUN_MWI_TOOLBOX_TESTS"] = "0"
    include(joinpath(pkgdir(DECAES), "test", "runtests.jl"))

    return nothing
end
=#
