module DECAES

using LinearAlgebra, SpecialFunctions, Statistics, Random
using ArgParse, Logging, LoggingExtras
import Optim, Dierckx, Polynomials, PolynomialRoots
import MAT, NIfTI
import LoopVectorization
using Parameters: @with_kw, @with_kw_noshow, @unpack
using StaticArrays: SVector, SizedVector, SA
using TimerOutputs: TimerOutput, @timeit_debug, reset_timer!
using SIMD: Vec, FloatingTypes, shufflevector

include("nnls.jl")
using .NNLS

include("types.jl")
include("utils.jl")
include("lsqnonneg.jl")
include("lsqnonneg_reg.jl")
include("lsqnonneg_lcurve.jl")
include("EPGdecaycurve.jl")
include("T2mapSEcorr.jl")
include("T2partSEcorr.jl")
include("main.jl")

# Global constants and settings computed during precompilation
const ALLOWED_FILE_SUFFIXES = (".mat", ".nii", ".nii.gz")
const ALLOWED_FILE_SUFFIXES_STRING = join(ALLOWED_FILE_SUFFIXES, ", ", ", and ")

const ARGPARSE_SETTINGS = create_argparse_settings(legacy = false)
const ARGPARSE_SETTINGS_LEGACY = create_argparse_settings(legacy = true)

const T2MAP_FIELDTYPES = Dict{Symbol,Type}(fieldnames(T2mapOptions{Float64}) .=> fieldtypes(T2mapOptions{Float64}))
const T2PART_FIELDTYPES = Dict{Symbol,Type}(fieldnames(T2partOptions{Float64}) .=> fieldtypes(T2partOptions{Float64}))

# Exported symbols
export MAT, NIfTI
export T2mapSEcorr, T2mapOptions, T2partSEcorr, T2partOptions
export EPGdecaycurve, EPGdecaycurve!, EPGdecaycurve_work
export lsqnonneg, lsqnonneg_reg, lsqnonneg_lcurve
export main

include("precompile.jl")
_precompile_()

end
