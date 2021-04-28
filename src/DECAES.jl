module DECAES

using LinearAlgebra, SpecialFunctions, Statistics, Random
import ArgParse, Dierckx, DocStringExtensions, Logging, MAT, NIfTI, NLopt, Parameters, PolynomialRoots, ProgressMeter, ProgressMeter, StaticArrays, SIMD, TimerOutputs, UnsafeArrays
using ArgParse: @add_arg_table!, add_arg_table!, add_arg_group!, ArgParseSettings, parse_args
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDFIELDS, TYPEDSIGNATURES
using Logging: ConsoleLogger, with_logger
using Parameters: @with_kw, @with_kw_noshow, @unpack
using ProgressMeter: Progress, BarGlyphs, tty_width, next!, finish!
using StaticArrays: SVector, MVector, SizedVector, SA
using SIMD: Vec, FloatingTypes, shufflevector
using TimerOutputs: TimerOutput, @timeit_debug, reset_timer!
using UnsafeArrays: @uviews, uviews, uview

include("NNLS.jl")
using .NNLS

include("ParXRec.jl")
import .ParXRec

include("types.jl")
include("utils.jl")
include("lsqnonneg.jl")
include("lsqnonneg_reg.jl")
include("EPGdecaycurve.jl")
include("T2mapSEcorr.jl")
include("T2partSEcorr.jl")
include("main.jl")

# Exported symbols
export MAT, NIfTI, ParXRec
export T2mapSEcorr, T2mapOptions, T2partSEcorr, T2partOptions
export EPGdecaycurve, EPGdecaycurve!, EPGdecaycurve_work
export lsqnonneg, lsqnonneg_chi2, lsqnonneg_gcv, lsqnonneg_lcurve
export main, julia_main, load_image

# include("precompile.jl")
# _precompile_()

end
