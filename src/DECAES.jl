module DECAES

using Dates, LinearAlgebra, SpecialFunctions, Statistics, Random
import ArgParse, BangBang, Dierckx, DocStringExtensions, ForwardDiff, Logging, LoggingExtras, MAT, NIfTI, NLopt, ParXRec, Parameters, PolynomialRoots, ProgressMeter, SIMD, StaticArrays, TupleTools, UnPack, UnsafeArrays
using ArgParse: @add_arg_table!, ArgParseSettings, add_arg_group!, add_arg_table!, parse_args
using BangBang: setindex!!, setproperty!!, setproperties!!
using Base.MathConstants: φ
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDFIELDS, TYPEDSIGNATURES
using ForwardDiff: DiffResults
using Logging: ConsoleLogger, with_logger
using LoggingExtras: FileLogger, TeeLogger, TransformerLogger
using Parameters: @with_kw, @with_kw_noshow
using ProgressMeter: Progress, BarGlyphs
using Requires: @require
using SIMD: FloatingTypes, Vec, shufflevector
using StaticArrays: FieldVector, SA, SArray, SVector, SMatrix, SizedVector, MVector
using UnPack: @unpack, @pack!
using UnsafeArrays: @uviews, uviews, uview

function __init__()
    @require LoopVectorization="bdcacae8-1622-11e9-2a5c-532679323890" @eval using .LoopVectorization: @turbo
end

macro acc(ex)
    if isdefined(DECAES, Symbol("@turbo"))
        esc( :( @turbo $(ex) ) )
    else
        esc( :( @inbounds @simd $(ex) ) )
    end
end

include("NNLS.jl")
using .NNLS

include("NormalHermiteSplines.jl")
using .NormalHermiteSplines

include("types.jl")
include("utils.jl")
include("splines.jl")
include("lsqnonneg.jl")
include("EPGdecaycurve.jl")
include("T2mapSEcorr.jl")
include("T2partSEcorr.jl")
include("main.jl")

# Exported symbols
export MAT, NIfTI, ParXRec, load_image
export T2mapSEcorr, T2mapOptions, T2partSEcorr, T2partOptions
export EPGdecaycurve, EPGdecaycurve!, EPGdecaycurve_work
export lsqnonneg, lsqnonneg_chi2, lsqnonneg_gcv, lsqnonneg_lcurve, lcurve_corner
export main

end
