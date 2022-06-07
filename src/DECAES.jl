module DECAES

# Standard libraries
using Dates, LinearAlgebra, SpecialFunctions, Statistics, Random

# Imported modules
import ArgParse, BangBang, Dierckx, DocStringExtensions, ForwardDiff, Logging, LoggingExtras, MAT, NIfTI, NLopt, ParXRec, Parameters, PolynomialRoots, ProgressMeter, Requires, SIMD, StaticArrays, TupleTools, UnPack, UnsafeArrays

# Explicitly imported symbols
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
        # Misusing LoopVectorization can have serious consequences. Like @inbounds, misusing it can lead to segfaults and memory corruption. We expect that any time you use the @turbo macro with a given block of code that you:
        #   1. Are not indexing an array out of bounds. @turbo does not perform any bounds checking.
        #   2. Are not iterating over an empty collection. Iterating over an empty loop such as for i ∈ eachindex(Float64[]) is undefined behavior, and will likely result in the out of bounds memory accesses. Ensure that loops behave correctly.
        #   3. Are not relying on a specific execution order. @turbo can and will re-order operations and loops inside its scope, so the correctness cannot depend on a particular order. You cannot implement cumsum with @turbo.
        #   4. Are not using multiple loops at the same level in nested loops.
        esc( :( @turbo $(ex) ) )
    else
        # Your inner loop should have the following properties to allow vectorization:
        #   * The loop must be an innermost loop
        #   * The loop body must be straight-line code. Therefore, [`@inbounds`](@ref) is
        #       currently needed for all array accesses. The compiler can sometimes turn
        #       short `&&`, `||`, and `?:` expressions into straight-line code if it is safe
        #       to evaluate all operands unconditionally. Consider using the [`ifelse`](@ref)
        #       function instead of `?:` in the loop if it is safe to do so.
        #   * Accesses must have a stride pattern and cannot be "gathers" (random-index
        #       reads) or "scatters" (random-index writes).
        #   * The stride should be unit stride.
        # With the ivdep flag:
        #   * There exists no loop-carried memory dependencies
        #   * No iteration ever waits on a previous iteration to make forward progress.
        esc( :( @inbounds @simd ivdep $(ex) ) )
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
