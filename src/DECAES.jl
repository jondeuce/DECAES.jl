module DECAES

const VERSION = v"0.6.0"

# Standard libraries
using Dates: Dates
using LinearAlgebra: LinearAlgebra, BLAS, LAPACK, axpy!, cholesky!, dot, mul!, norm, svdvals, svdvals!, ×, ⋅
using LinearAlgebra.BLAS: @blasfunc, BlasInt, libblastrampoline
using LinearAlgebra.LAPACK: chklapackerror
using Logging: Logging, ConsoleLogger, with_logger
using Pkg: Pkg
using Random: Random
using Statistics: Statistics, mean, std

# External libraries
using ArgParse: ArgParse, ArgParseSettings, add_arg_group!, add_arg_table!, parse_args
using Dierckx: Dierckx
using DocStringExtensions: DocStringExtensions, @doc, FIELDS, SIGNATURES, TYPEDFIELDS, TYPEDSIGNATURES
using ForwardDiff: ForwardDiff, DiffResults, Dual
using LoggingExtras: LoggingExtras, FileLogger, LevelOverrideLogger, TeeLogger, TransformerLogger
using MAT: MAT
using MuladdMacro: MuladdMacro, @muladd
using NIfTI: NIfTI
using NLopt: NLopt
using ParXRec: ParXRec
using Parameters: Parameters, @with_kw, @with_kw_noshow
# using PolynomialRoots: PolynomialRoots
using PrecompileTools: PrecompileTools, @compile_workload, @setup_workload
using ProgressMeter: ProgressMeter, Progress, BarGlyphs
# using Roots: Roots
# using SLEEFPirates: SLEEFPirates
# using SIMD: SIMD, FloatingTypes, Vec, shufflevector
using Scratch: Scratch, @get_scratch!, get_scratch!
using SpecialFunctions: SpecialFunctions, erfc, erfinv
using StaticArrays: StaticArrays, MVector, SA, SArray, SMatrix, SVector
using TupleTools: TupleTools
using UnsafeArrays: UnsafeArrays, uview

include("NNLS.jl")
using .NNLS: NNLS

include("NormalHermiteSplines.jl")
using .NormalHermiteSplines: NormalHermiteSplines, RK_H0, RK_H1, RK_H2

include("types.jl")
include("utils.jl")
include("optimization.jl")
include("splines.jl")
include("lsqnonneg.jl")
include("EPGdecaycurve.jl")
include("T2mapSEcorr.jl")
include("T2partSEcorr.jl")
include("main.jl")

# Exported symbols
export MAT, NIfTI, ParXRec, load_image
export T2mapOptions, T2mapSEcorr, T2partOptions, T2partSEcorr
export EPGdecaycurve, EPGdecaycurve!, EPGdecaycurve_work
export lcurve_corner, lsqnonneg, lsqnonneg_tikh, lsqnonneg_lcurve, lsqnonneg_gcv, lsqnonneg_chi2, lsqnonneg_mdp
export main

# Precompile
@compile_workload begin
    redirect_to_devnull() do
        main(["--help"])
        main(["--version"])
        mock_load_image()
        for Reg in ["none", "lcurve", "gcv", "chi2", "mdp"]
            NumVoxels = max(4, Threads.nthreads()) * default_blocksize()
            mock_T2_pipeline(; MatrixSize = (NumVoxels, 1, 1), Reg)
        end
    end
end

end # module DECAES
