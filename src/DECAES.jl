module DECAES

const VERSION = v"0.5.2-DEV"

# Standard libraries
using Dates, LinearAlgebra, SpecialFunctions, Statistics, Random
using Base.MathConstants: φ

# External libraries
using ArgParse: ArgParse, ArgParseSettings, add_arg_group!, add_arg_table!, parse_args
using Dierckx: Dierckx
using DocStringExtensions: DocStringExtensions, FIELDS, SIGNATURES, TYPEDFIELDS, TYPEDSIGNATURES
using ForwardDiff: ForwardDiff, DiffResults
using Logging: Logging, ConsoleLogger, with_logger
using LoggingExtras: LoggingExtras, FileLogger, TeeLogger, TransformerLogger
using MAT: MAT
using MuladdMacro: MuladdMacro, @muladd
using NIfTI: NIfTI
using NLopt: NLopt
using ParXRec: ParXRec
using Parameters: Parameters, @with_kw, @with_kw_noshow
using PolynomialRoots: PolynomialRoots
using PrecompileTools: PrecompileTools, @compile_workload, @setup_workload
using ProgressMeter: ProgressMeter, Progress, BarGlyphs
# using Roots: Roots
# using SIMD: SIMD, FloatingTypes, Vec, shufflevector
using Scratch: Scratch, @get_scratch!
using StaticArrays: StaticArrays, FieldVector, SA, SArray, SVector, SMatrix, SizedVector, MVector
using TupleTools: TupleTools
using UnsafeArrays: UnsafeArrays, uview

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

# Precompile
@compile_workload begin
    redirect_to_devnull() do
        main(["--help"])
        main(["--version"])
        for Reg in ["lcurve", "gcv", "chi2"]
            NumVoxels = max(4, Threads.nthreads()) * default_blocksize()
            mock_T2_pipeline(; NumVoxels, Reg)
        end
    end
end

end
