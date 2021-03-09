module DECAES

using LinearAlgebra, SpecialFunctions, Statistics, Random
import NLopt, Dierckx, Polynomials, PolynomialRoots, ProgressMeter
using Parameters: @with_kw, @with_kw_noshow, @unpack
using StaticArrays: SVector, MVector, SizedVector, SA
using TimerOutputs: TimerOutput, @timeit_debug, reset_timer!
using SIMD: Vec, FloatingTypes, shufflevector
using ProgressMeter: Progress, BarGlyphs, tty_width, next!, finish!
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDFIELDS, TYPEDSIGNATURES

include("NNLS.jl")
using .NNLS

include("types.jl")
include("utils.jl")
include("lsqnonneg.jl")
include("lsqnonneg_reg.jl")
include("lsqnonneg_lcurve.jl")
include("EPGdecaycurve.jl")
include("T2mapSEcorr.jl")
include("T2partSEcorr.jl")

include("Main.jl")
using .Main

# Exported symbols
export MAT, NIfTI, ParXRec
export T2mapSEcorr, T2mapOptions, T2partSEcorr, T2partOptions
export EPGdecaycurve, EPGdecaycurve!, EPGdecaycurve_work
export lsqnonneg, lsqnonneg_reg, lsqnonneg_lcurve
export main

# include("precompile.jl")
# _precompile_()

end
