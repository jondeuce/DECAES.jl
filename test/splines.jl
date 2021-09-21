using Test
using DECAES
using DECAES.NormalHermiteSplines
using StaticArrays
using Random
using LinearAlgebra

function test_mock_surrogate_search_problem(
        opts::T2mapOptions = DECAES.mock_t2map_opts(;
            MatrixSize = (1, 1, 1),
            nRefAngles = 8,
        )
    )
    function A(α, β)
        theta = DECAES.EPGOptions{Float64,32}(α, opts.TE, 0.0, opts.T1, β)
        T2_times = DECAES.logrange(opts.T2Range..., opts.nT2)
        DECAES.epg_decay_basis(theta, T2_times)
    end

    function f!(work, prob, α, β)
        DECAES.solve!(work, A(α, β), prob.b)
        return log(max(DECAES.chi2(work), eps()))
    end

    function fg_approx!(work, prob, α, β; h)
        l   = f!(work, prob, α, β)
        lα⁺ = f!(work, prob, α + h, β)
        lα⁻ = f!(work, prob, α - h, β)
        lβ⁺ = f!(work, prob, α, β + h)
        lβ⁻ = f!(work, prob, α, β - h)
        ∂l_∂α = (lα⁺ - lα⁻) / 2h
        ∂l_∂β = (lβ⁺ - lβ⁻) / 2h
        ∇l  = SA[∂l_∂α, ∂l_∂β]
        return l, ∇l
    end

    function fg_surrogate!(prob, I)
        l  = DECAES.loss!(prob, I)
        ∇l = DECAES.∇loss!(prob, I)
        return l, ∇l
    end

    prob = DECAES.mock_surrogate_search_problem(Val(2), Val(opts.nTE), opts)
    work = DECAES.lsqnonneg_work(zeros(opts.nTE, opts.nT2), zeros(opts.nTE))

    for I in CartesianIndices(prob.αs)
        α, β = prob.αs[I]
        l′, ∇l′ = fg_approx!(work, prob, α, β; h = 1e-6)
        l , ∇l  = fg_surrogate!(prob, I)
        @test l ≈ l′
        @test ∇l ≈ ∇l′ rtol = 1e-6 atol = 1e-8
    end
end

function test_opposite_corner()
    for bounds in [
            ((2,3),),
            ((2,3), (5,7)),
            ((2,3), (5,7), (-3,-1)),
        ]
        box = DECAES.BoundingBox(bounds)
        for I in box.corners
            Iopp = CartesianIndex(ntuple(d -> ifelse(I[d] == bounds[d][1], bounds[d][2], bounds[d][1]), length(bounds)))
            @test DECAES.opposite_corner(box, I) == Iopp
        end
    end
end

function test_minimal_bounding_box()
    grid  = DECAES.meshgrid(SVector{2,Float64}, 1:5, 1:10)
    surr  = DECAES.HermiteSplineSurrogate(I -> (1.0, zero(SVector{2,Float64})), grid)
    state = DECAES.DiscreteSurrogateSearcher(surr; mineval = 0, maxeval = 0)
    state.seen[[1,5], [1,10]] .= true

    #  1  -------  1  ----------  1
    #  |  0  0  0  |  0  0  0  0  |
    #  |  0  0  0  |  0  0  0  0  |
    #  |  0  0  0  |  0  0  0  0  |
    #  1  -------  1  ----------  1
    @test DECAES.minimal_bounding_box(state, SA[1.5, 2.0]) == DECAES.BoundingBox(((1,5), (1,5)))
    @test DECAES.minimal_bounding_box(state, SA[4.5, 6.5]) == DECAES.BoundingBox(((1,5), (5,10)))
    state.seen[[1,5], 5] .= true

    #  1  -------  1  -- 1 -----  1
    #  |  0  0  0  |  0  |  0  0  |
    #  |  -------  |  0  |  0  0  |
    #  |  0  0  0  |  0  |  0  0  |
    #  1  -------  1  -- 1 -----  1
    @test DECAES.minimal_bounding_box(state, SA[1.5, 2.0]) == DECAES.BoundingBox(((1,3), (1,5)))
    @test DECAES.minimal_bounding_box(state, SA[4.5, 2.0]) == DECAES.BoundingBox(((3,5), (1,5)))
    @test DECAES.minimal_bounding_box(state, SA[1.5, 6.5]) == DECAES.BoundingBox(((1,5), (5,7)))
    @test DECAES.minimal_bounding_box(state, SA[4.5, 6.5]) == DECAES.BoundingBox(((1,5), (5,7)))
    @test DECAES.minimal_bounding_box(state, SA[1.5, 8.5]) == DECAES.BoundingBox(((1,5), (7,10)))
    @test DECAES.minimal_bounding_box(state, SA[4.5, 8.5]) == DECAES.BoundingBox(((1,5), (7,10)))
end

@testset "Splines" begin
    @testset "mock surrogate search problem" begin
        test_mock_surrogate_search_problem()
    end
    @testset "bounding box" begin
        test_opposite_corner()
        test_minimal_bounding_box()
    end
end
