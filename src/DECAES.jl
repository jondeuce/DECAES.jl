module DECAES

using LinearAlgebra, SpecialFunctions, Statistics, Random
using StaticArrays, SIMD
using Optim, Dierckx, Polynomials, PolynomialRoots
using ArgParse, Parameters, TimerOutputs, Logging, LoggingExtras
using MAT, NIfTI

include("nnls.jl")
using .NNLS

include("utils.jl")
include("lsqnonneg.jl")
include("lsqnonneg_reg.jl")
include("lsqnonneg_lcurve.jl")
include("EPGdecaycurve.jl")
include("T2mapSEcorr.jl")
include("T2partSEcorr.jl")
include("main.jl")

export MAT, NIfTI # export module symbols
export T2mapSEcorr, T2mapOptions, T2partSEcorr, T2partOptions
export EPGdecaycurve, lsqnonneg, lsqnonneg_reg, lsqnonneg_lcurve
export main

include("precompile.jl")
_precompile_()

end
