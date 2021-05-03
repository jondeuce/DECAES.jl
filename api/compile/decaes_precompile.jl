using DECAES

# Silence console output
DECAES.redirect_to_devnull() do
    # Run code to be precompiled which is not included in test suite coverage
    main(["--help"])

    # Run test suite, excluding Matlab MWI Toolbox compatibility tests
    ENV["DECAES_RUN_MWI_TOOLBOX_TESTS"] = "0"
    include(joinpath(pkgdir(DECAES), "test", "runtests.jl"))
end
