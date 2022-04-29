using Test
using DECAES

using DECAES.LinearAlgebra
using DECAES.NNLS
using DECAES.Random
using DECAES.StaticArrays
using DECAES: GrowableCache, GrowableCachePairs, CachedFunction, MappedArray
using DECAES: LCurveCornerPoint, LCurveCornerState, LCurveCornerCachedFunction
using DECAES: lcurve_corner

@testset "nnls.jl"    verbose = true begin; include("nnls.jl"); end
@testset "utils.jl"   verbose = true begin; include("utils.jl"); end
@testset "epg.jl"     verbose = true begin; include("epg.jl"); end
@testset "splines.jl" verbose = true begin; include("splines.jl"); end
@testset "cli.jl"     verbose = true begin; include("cli.jl"); end

# using Aqua
# @testset "Aqua tests" begin
#     Aqua.test_all(DECAES)
# end
