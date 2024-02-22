function test_poly()
    @testset "degree d = $d" for d in 0:5
        coeffs = randn(d+1)
        for p in [DECAES.Poly(coeffs), DECAES.Poly(coeffs...)]
            @test DECAES.coeffs(p) == coeffs
            @test DECAES.coeffs(p') == Float64[i * coeffs[i+1] for i in 1:d]
            @test DECAES.coeffs(cumsum(p)) == [i == 0 ? 0.0 : coeffs[i] / i for i in 0:d+1]
            @test DECAES.coeffs(cumsum(p)') ≈ DECAES.coeffs(p) # derivative of integral is identity
            @test DECAES.coeffs(cumsum(p')) ≈ [0.0; DECAES.coeffs(p)[2:end]] # integral of derivative is identity, up to a constant, which we fix to zero
        end
    end
end

function test_cubic_splines()
    @testset "npts = $npts, deg_spline = $deg_spline" for npts in 2:5, deg_spline in 1:min(npts - 1, 3)
        X = sort!(randn(npts))
        Y = randn(npts)

        spl = DECAES.make_spline(X, Y; deg_spline)
        (; x, y) = DECAES.spline_opt(X, Y; deg_spline)
        ŷ = minimum(spl, range(X[1], X[end]; length = 100))
        @test X[1] <= x <= X[end]
        @test spl(x) ≈ y
        @test ŷ >= y - 100 * eps()

        x̄ = DECAES.spline_root(X, Y, y - 1; deg_spline)
        @test isnan(x̄)

        ȳ = (minimum(Y) + maximum(Y)) / 2
        x̄ = DECAES.spline_root(X, Y, ȳ; deg_spline)
        @test X[1] <= x̄ <= X[end]
        @test spl(x̄) ≈ ȳ
    end
end

function test_cubic_hermite_splines()
    # Two real roots
    a, b, c = rand(), 2 + rand(), rand()
    coeffs = (c, b, a)
    x1, x2 = DECAES.roots_real_quadratic(coeffs)
    @test !isnan(x1) && !isnan(x2) # both roots are real

    y1, y2 = evalpoly.((x1, x2), ((c, b, a),))
    @test !isnan(y1) && !isnan(y2) # both values are real
    @test b^2 - 4a * c > 0 # discriminant is positive
    @test abs(y1) < √eps() && abs(y2) < √eps() # both values are near zero

    # No roots
    a, b, c = 0.75, rand(), rand() + 0.5
    coeffs = (c, b, a)
    x1, x2 = DECAES.roots_real_quadratic(coeffs)
    @test isnan(x1) && isnan(x2) # both roots NaN

    y1, y2 = evalpoly.((x1, x2), ((c, b, a),))
    @test isnan(y1) && isnan(y2) # both values NaN
    @test b^2 - 4a * c < 0 # discriminant is negative
end

function test_cubic_hermite_interpolator()
    u0, u1, m0, m1 = randn(4)
    spl = DECAES.CubicHermiteInterpolator(u0, u1, m0, m1)
    (; coeffs) = spl
    ∇coeffs = DECAES.deriv_coeffs(coeffs)

    @test evalpoly(-1.0, coeffs) ≈ u0 rtol = 1e-14 atol = 1e-14
    @test evalpoly(1.0, coeffs) ≈ u1 rtol = 1e-14 atol = 1e-14
    @test evalpoly(-1.0, ∇coeffs) ≈ m0 rtol = 1e-14 atol = 1e-14
    @test evalpoly(1.0, ∇coeffs) ≈ m1 rtol = 1e-14 atol = 1e-14

    @test evalpoly(0.7, coeffs) == spl(0.7)
    @test evalpoly(0.7, ∇coeffs) ≈ DECAES.ForwardDiff.derivative(spl, 0.7) rtol = 1e-14 atol = 1e-14

    xmin, umin = DECAES.minimize(spl)
    @test umin ≈ spl(xmin) rtol = 1e-14 atol = 1e-14
    @test all(spl(x) >= umin - 1e-14 for x in range(-1.0, 1.0; length = 1001))
end

function test_mock_surrogate_search_problem(
    opts::T2mapOptions = DECAES.mock_t2map_opts(;
        MatrixSize = (1, 1, 1),
        nRefAngles = 8,
    ),
)
    function A(α, β)
        theta = DECAES.EPGOptions((; ETL = 32, α = α, TE = opts.TE, T2 = 0.0, T1 = opts.T1, β = β))
        T2_times = DECAES.logrange(opts.T2Range..., opts.nT2)
        return DECAES.epg_decay_basis(theta, T2_times)
    end

    function f!(work, prob, α, β)
        DECAES.solve!(work, A(α, β), prob.b)
        return log(max(DECAES.chi2(work), eps()))
    end

    function fg_approx!(work, prob, α, β; h)
        l = f!(work, prob, α, β)
        lα⁺ = f!(work, prob, α + h, β)
        lα⁻ = f!(work, prob, α - h, β)
        lβ⁺ = f!(work, prob, α, β + h)
        lβ⁻ = f!(work, prob, α, β - h)
        ∂l_∂α = (lα⁺ - lα⁻) / 2h
        ∂l_∂β = (lβ⁺ - lβ⁻) / 2h
        ∇l = SA[∂l_∂α, ∂l_∂β]
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
        α, β    = prob.αs[I]
        l′, ∇l′ = fg_approx!(work, prob, α, β; h = 1e-6)
        l, ∇l   = fg_surrogate!(prob, I)
        @test l ≈ l′
        @test ∇l ≈ ∇l′ rtol = 1e-6 atol = 1e-8
    end
end

function test_bounding_box()
    for bounds in [
        ((2, 3),),
        ((2, 3), (5, 7)),
        ((2, 3), (5, 7), (-3, -1)),
    ]
        box = DECAES.BoundingBox(bounds)
        for I in box.corners
            Iopp = CartesianIndex(ntuple(d -> ifelse(I[d] == bounds[d][1], bounds[d][2], bounds[d][1]), length(bounds)))
            @test DECAES.opposite_corner(box, I) == Iopp
        end
    end
end

function test_discrete_searcher()
    grid = DECAES.meshgrid(SVector{2, Float64}, 1:5, 1:10)
    surr = DECAES.NormalHermiteSplineSurrogate(I -> (1.0, zero(SVector{2, Float64})), grid)
    state = DECAES.DiscreteSurrogateSearcher(surr; mineval = 0, maxeval = 0)
    state.seen[[1, 5], [1, 10]] .= true

    @test DECAES.sorted_corners(state, DECAES.BoundingBox(((2, 4), (3, 7))), SA[3.5, 6.5]) == SMatrix{2, 2}((CartesianIndex(4, 7), CartesianIndex(2, 7), CartesianIndex(4, 3), CartesianIndex(2, 3)))
    @test DECAES.sorted_corners(state, DECAES.BoundingBox(((2, 4), (3, 7))), SA[2.5, 6.5]) == SMatrix{2, 2}((CartesianIndex(2, 7), CartesianIndex(4, 7), CartesianIndex(2, 3), CartesianIndex(4, 3)))
    @test DECAES.sorted_corners(state, DECAES.BoundingBox(((2, 4), (3, 7))), SA[2.5, 4.5]) == SMatrix{2, 2}((CartesianIndex(2, 3), CartesianIndex(4, 3), CartesianIndex(2, 7), CartesianIndex(4, 7)))
    @test DECAES.sorted_corners(state, DECAES.BoundingBox(((2, 4), (3, 7))), SA[3.5, 4.5]) == SMatrix{2, 2}((CartesianIndex(4, 3), CartesianIndex(2, 3), CartesianIndex(4, 7), CartesianIndex(2, 7)))

    #  1  -------  1  ----------  1
    #  |  0  0  0  |  0  0  0  0  |
    #  |  0  0  0  |  0  0  0  0  |
    #  |  0  0  0  |  0  0  0  0  |
    #  1  -------  1  ----------  1
    @test DECAES.minimal_bounding_box(state, SA[1.5, 2.0]) == DECAES.BoundingBox(((1, 5), (1, 5)))
    @test DECAES.minimal_bounding_box(state, SA[4.5, 6.5]) == DECAES.BoundingBox(((1, 5), (5, 10)))
    state.seen[[1, 5], 5] .= true

    #  1  -------  1  -- 1 -----  1
    #  |  0  0  0  |  0  |  0  0  |
    #  |  -------  |  0  |  0  0  |
    #  |  0  0  0  |  0  |  0  0  |
    #  1  -------  1  -- 1 -----  1
    @test DECAES.minimal_bounding_box(state, SA[1.5, 2.0]) == DECAES.BoundingBox(((1, 3), (1, 5)))
    @test DECAES.minimal_bounding_box(state, SA[4.5, 2.0]) == DECAES.BoundingBox(((3, 5), (1, 5)))
    @test DECAES.minimal_bounding_box(state, SA[1.5, 6.5]) == DECAES.BoundingBox(((1, 5), (5, 7)))
    @test DECAES.minimal_bounding_box(state, SA[4.5, 6.5]) == DECAES.BoundingBox(((1, 5), (5, 7)))
    @test DECAES.minimal_bounding_box(state, SA[1.5, 8.5]) == DECAES.BoundingBox(((1, 5), (7, 10)))
    @test DECAES.minimal_bounding_box(state, SA[4.5, 8.5]) == DECAES.BoundingBox(((1, 5), (7, 10)))
end

@testset "Splines" begin
    @testset "poly" begin
        test_poly()
    end
    @testset "cubic" begin
        test_cubic_splines()
    end
    @testset "cubic hermite" begin
        @testset "basics" test_cubic_hermite_splines()
        @testset "interpolator" test_cubic_hermite_interpolator()
    end
    @testset "mock surrogate search problem" begin
        test_mock_surrogate_search_problem()
    end
    @testset "bounding box" begin
        @testset "basics" test_bounding_box()
        @testset "discrete searcher" test_discrete_searcher()
    end
end
