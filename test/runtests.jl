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
using DECAES:
    GrowableCache, GrowableCachePairs, CachedFunction, MappedArray,
    LCurveCornerPoint, LCurveCornerState, LCurveCornerCachedFunction,
    NNLSProblem, NNLSTikhonovRegProblem, NNLSTikhonovRegProblemCache,
    lcurve_corner

is_ci() = lowercase(get(ENV, "CI", "false")) == "true"

# Test files to run, in order. With no arguments the whole suite runs, otherwise names are matched as substrings of the file name:
#   julia --project=test test/runtests.jl
#   julia --project=test test/runtests.jl nnls
#   julia --project=test test/runtests.jl nnls splines
#   julia -e 'using Pkg; Pkg.test("DECAES"; test_args = ["nnls"])'

const TEST_FILES = ["misc", "utils", "optimization", "splines", "nnls", "epg", "t2map", "cli", "aqua"]
const SELECTED_TEST_FILES = isempty(ARGS) ? TEST_FILES : filter(name -> any(arg -> occursin(arg, name), ARGS), TEST_FILES)

for name in SELECTED_TEST_FILES
    @testset "$name.jl" verbose = true begin
        include("$name.jl")
    end
end
