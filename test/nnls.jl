using Test
using DECAES
using DECAES.NNLS
using LinearAlgebra

function verify_solution()
    A = rand(5,5)
    b = rand(5)
    work = NNLSWorkspace(A, b)
    nnls!(work)
    GC.@preserve work begin
        x = NNLS.solution(work)
        w = NNLS.dual(work)
        L = LowerTriangular(work)
        Jx = NNLS.components(work)
        Jw = setdiff(1:length(x), Jx)
        Ax = A[:,Jx]
        Aw = A[:,Jw]
        @test all(>(0), x[Jx])
        @test all(==(0), x[Jw])
        @test all(<(0), w[Jw])
        @test all(==(0), w[Jx])
        @test w[Jw] ≈ Aw'b - Aw'Ax * x[Jx]
        @test Ax \ b ≈ x[Jx]
        @test Ax'Ax ≈ L'L
    end
    work
end

@testset "NNLS" begin
    verify_solution()
end
