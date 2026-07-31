using Aqua
using Test

using DoubleFloats
using ForwardDiff
using LinearAlgebra
using Pkg
using PolynomialRoots: PolynomialRoots
using Random
using StaticArrays
using Statistics
using TOML

using DECAES
using DECAES: NNLS
using DECAES: NormalHermiteSplines
using DECAES:
    GrowableCache, GrowableCachePairs, CachedFunction, MappedArray,
    LCurveCornerPoint, LCurveCornerState, LCurveCornerCachedFunction,
    NNLSProblem, NNLSTikhonovRegProblem, NNLSTikhonovRegProblemCache,
    lcurve_corner

# Environment flags
is_ci() = lowercase(get(ENV, "CI", "false")) == "true"
RUN_MATLAB_TESTS = !is_ci() && get(ENV, "DECAES_RUN_MATLAB_TESTS", "") != "0"
MWI_TOOLBOX_PATH = get(ENV, "DECAES_MWI_TOOLBOX_PATH", "")
RUN_MWI_TOOLBOX_TESTS = get(ENV, "DECAES_RUN_MWI_TOOLBOX_TESTS", "") != "0"

# Try loading MATLAB.jl
if RUN_MATLAB_TESTS
    try
        @eval using MATLAB
        mxcall(:addpath, 0, joinpath(pkgdir(DECAES), "api"))
    catch e
        global RUN_MATLAB_TESTS = false
        @warn "Failed to load Julia package MATLAB.jl; skipping MATLAB tests"
        @warn sprint(showerror, e, catch_backtrace())
    end
end

# Try finding UBC MWI toolbox
mfile_exists(fname) = MATLAB.mxcall(:exist, 1, fname) == 2
if RUN_MATLAB_TESTS && RUN_MWI_TOOLBOX_TESTS
    try
        if !isempty(MWI_TOOLBOX_PATH)
            mxcall(:addpath, 0, MWI_TOOLBOX_PATH)
        end
        if !mfile_exists("T2map_SEcorr_nechoes_2019") || !mfile_exists("T2part_SEcorr_2019")
            global RUN_MWI_TOOLBOX_TESTS = false
            @warn "Files T2map_SEcorr_nechoes_2019.m and T2part_SEcorr_2019.m were not found on the default MATLAB path. " *
                  "Modify your default MATLAB path to include these files, or set the DECAES_MWI_TOOLBOX_PATH environment variable.\n\n" *
                  "For example, on unix-like systems run" *
                  "\n\n    export DECAES_MWI_TOOLBOX_PATH=/path/to/MWI_NNLS_toolbox_0319\n\n" *
                  "before testing DECAES, or add a command such as" *
                  "\n\n    addpath /path/to/MWI_NNLS_toolbox_0319\n\n" *
                  "to your startup.m file in MATLAB."
        end
    catch e
        global RUN_MWI_TOOLBOX_TESTS = false
        @warn "Failed to find the UBC MWI toolbox; skipping tests"
        @warn sprint(showerror, e, catch_backtrace())
    end
end

# Test files to run, in order. With no arguments the whole suite runs, otherwise names are matched as substrings of the file name:
#   julia --project=test test/runtests.jl
#   julia --project=test test/runtests.jl nnls
#   julia --project=test test/runtests.jl nnls splines
#   julia -e 'using Pkg; Pkg.test("DECAES"; test_args = ["nnls"])'

const TEST_FILES = ["misc", "nhs", "utils", "optimization", "splines", "nnls", "epg", "cli", "aqua"]
const SELECTED_TEST_FILES = isempty(ARGS) ? TEST_FILES : filter(name -> any(arg -> occursin(arg, name), ARGS), TEST_FILES)

for name in SELECTED_TEST_FILES
    @testset "$name.jl" verbose = true begin
        include("$name.jl")
    end
end
