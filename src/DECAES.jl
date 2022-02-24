module DECAES

using LinearAlgebra, SpecialFunctions, Statistics, Random
import ArgParse, BangBang, Dierckx, DocStringExtensions, ForwardDiff, Logging, MAT, NIfTI, NLopt, ParXRec, Parameters, PolynomialRoots, ProgressMeter, SIMD, StaticArrays, TupleTools, UnPack, UnsafeArrays
using ArgParse: @add_arg_table!, ArgParseSettings, add_arg_group!, add_arg_table!, parse_args
using BangBang: setindex!!, setproperty!!, setproperties!!
using Base.MathConstants: φ
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDFIELDS, TYPEDSIGNATURES
using ForwardDiff: DiffResults
using Logging: ConsoleLogger, with_logger
using Parameters: @with_kw, @with_kw_noshow
using ProgressMeter: Progress, BarGlyphs
using StaticArrays: FieldVector, SA, SArray, SVector, SMatrix, SizedVector, MVector
using SIMD: FloatingTypes, Vec, shufflevector
using UnPack: @unpack, @pack!
using UnsafeArrays: @uviews, uviews, uview

include("NNLS.jl")
using .NNLS

include("NormalHermiteSplines.jl")
using .NormalHermiteSplines

include("types.jl")
include("utils.jl")
include("splines.jl")
include("lsqnonneg.jl")
include("lsqnonneg_reg.jl")
include("EPGdecaycurve.jl")
include("T2mapSEcorr.jl")
include("T2partSEcorr.jl")
include("main.jl")

# Exported symbols
export MAT, NIfTI, ParXRec, load_image
export T2mapSEcorr, T2mapOptions, T2partSEcorr, T2partOptions
export EPGdecaycurve, EPGdecaycurve!, EPGdecaycurve_work
export lsqnonneg, lsqnonneg_chi2, lsqnonneg_gcv, lsqnonneg_lcurve, lcurve_corner
export main, julia_main

# include("precompile.jl")
# _precompile_()

end
