function test_poly()
    function polyroots(coeffs)
        if all(iszero, coeffs)
            # All coefficients are zero; by convention, return NaN roots
            return fill(NaN, length(coeffs) - 1)
        end
        rs = PolynomialRoots.roots(coeffs)
        if length(rs) < length(coeffs) - 1
            # If leading coefficient is zero, PolynomialRoots.roots returns one fewer root; append NaN to match length
            @assert coeffs[end] == 0
            push!(rs, NaN)
        end
        return sort!(rs; by = r -> (abs(imag(r)) > √eps(), real(r))) # sorted real roots, followed by complex roots
    end

    @testset "degree d = $d, coeffs[$i] = 0" for d in 0:5, i in 0:d+1
        coeffs = randn(d + 1)
        i > 0 && (coeffs[i] = 0.0) # zero out i'th coefficient to test degenerate polynomials
        for p in [DECAES.Poly(coeffs), DECAES.Poly(coeffs...)]
            @test DECAES.coeffs(p) == coeffs
            @test DECAES.coeffs(p') == Float64[i * coeffs[i+1] for i in 1:d]
            @test DECAES.coeffs(cumsum(p)) == [i == 0 ? 0.0 : coeffs[i] / i for i in 0:d+1]
            @test DECAES.coeffs(cumsum(p)') ≈ DECAES.coeffs(p) # derivative of integral is identity
            @test DECAES.coeffs(cumsum(p')) ≈ [0.0; DECAES.coeffs(p)[2:end]] # integral of derivative is identity, up to a constant, which we fix to zero

            # Test root-finding (only implemented for polynomials of degree <= 3)
            d <= 3 || continue
            rs = polyroots(coeffs)
            r̂s = DECAES.roots(p)
            @test length(r̂s) == length(rs) == d
            for i in eachindex(rs, r̂s)
                if !isnan(r̂s[i])
                    @test isapprox(r̂s[i], rs[i]; rtol = 1e-12, atol = 1e-12) # real roots should be close
                else
                    @test isnan(rs[i]) || abs(imag(rs[i])) > √eps() # NaN outputs should correspond to NaN or complex roots
                end
            end

            # Test minimization on an interval [a, b] #TODO `minimize(p::Poly, a::Real, b::Real)` method?
            a, b = sort(randn(2))
            xs = range(a, b; length = 1024)
            if d == 1
                x̄, px̄ = DECAES.minimize_linear((coeffs...,), a, b)
                @test a <= x̄ <= b
                @test px̄ == min(p(a), p(b))
            elseif d == 2
                x̄, px̄ = DECAES.minimize_quadratic((coeffs...,), a, b)
                @test a <= x̄ <= b
                @test all(p(x) >= px̄ - 1e-12 for x in xs)
            elseif d == 3
                x̄, px̄ = DECAES.minimize_cubic((coeffs...,), a, b)
                @test a <= x̄ <= b
                @test all(p(x) >= px̄ - 1e-12 for x in xs)
            end
        end
    end
end

function test_quadratic_roots()
    # Degenerate quadratics
    x1, x2 = DECAES.roots_real_quadratic((randn(), 0.0, 0.0))
    @test isnan(x1) && isnan(x2) # no roots; degenerate quadratic `const = 0`
    x1, x2 = DECAES.roots_real_quadratic((randn(), randn(), 0.0))
    @test !isnan(x1) && isnan(x2) # one root is real, the other at infinity NaN
    x1, x2 = DECAES.roots_real_quadratic((0.0, randn(), randn()))
    @test !isnan(x1) && !isnan(x2) # both roots are real
    @test xor(x1 == 0.0, x2 == 0.0) # one root is zero, the other is not

    # Repeated roots
    x1, x2 = DECAES.roots_real_quadratic((1.0, -2.0, 1.0))
    @test !isnan(x1) && !isnan(x2) # both roots are real
    @test x1 == x2 # repeated root

    # Two real roots
    a, b, c = rand(), 2 + rand(), rand()
    coeffs = (c, b, a)
    x1, x2 = DECAES.roots_real_quadratic(coeffs)
    @test !isnan(x1) && !isnan(x2) # both roots are real
    @test x1 <= x2 # roots are sorted

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

function test_cubic_splines()
    @testset "npts = $npts, deg_spline = $deg_spline" for npts in 2:5, deg_spline in 1:min(npts - 1, 3)
        X = sort!(randn(npts))
        Y = randn(npts)

        spl = DECAES.make_spline(X, Y; deg_spline)
        (; x, y) = DECAES.spline_opt(X, Y; deg_spline)
        ŷ = minimum(spl, range(X[1], X[end]; length = 100))
        @test X[1] <= x <= X[end]
        @test spl(x) ≈ y
        @test ŷ >= y - 1e-12

        x̄ = DECAES.spline_root(X, Y, y - 1; deg_spline)
        @test isnan(x̄)

        ȳ = (minimum(Y) + maximum(Y)) / 2
        x̄ = DECAES.spline_root(X, Y, ȳ; deg_spline)
        @test X[1] <= x̄ <= X[end]
        @test spl(x̄) ≈ ȳ
    end
end

function cubic_hermite_interpolator_params_iter()
    # Pairs of endpoint slopes
    ms = Iterators.flatten((
        ((s0 * rand(), s1 * rand()) for s0 in (-1, 0, +1), s1 in (-1, 0, +1)), # differing positive, negative, and zero slopes
        (rand() .* (s0, s1) for s0 in (-1, 0, +1), s1 in (-1, 0, +1)), # equal and/or opposite slopes
    ))

    # Pairs of endpoint values
    us = (randn() .+ (s * rand(), -s * rand()) for s in -1:1) # u0 < u1, u0 = u1, and u0 > u1

    return ((u0, u1, m0, m1) for ((u0, u1), (m0, m1)) in Iterators.product(us, ms))
end

function test_minimize_cubic_hermite_interpolator()
    for (u0, u1, m0, m1) in cubic_hermite_interpolator_params_iter()
        a, b = sort(randn(2))
        c, r = (a + b) / 2, (b - a) / 2
        spl = DECAES.CubicHermiteInterpolator(a, b, u0, u1, m0, m1)
        (; coeffs) = spl
        ∇coeffs = DECAES.deriv_coeffs(coeffs)

        @test evalpoly(-1.0, coeffs) ≈ u0 rtol = 1e-12 atol = 1e-12
        @test evalpoly(+1.0, coeffs) ≈ u1 rtol = 1e-12 atol = 1e-12
        @test evalpoly(-1.0, ∇coeffs) ≈ r * m0 rtol = 1e-12 atol = 1e-12
        @test evalpoly(+1.0, ∇coeffs) ≈ r * m1 rtol = 1e-12 atol = 1e-12

        t = 2 * rand() - 1
        x = c + r * t
        @test evalpoly(t, coeffs) ≈ spl(x)
        @test evalpoly(t, ∇coeffs) ≈ r * DECAES.ForwardDiff.derivative(spl, x) rtol = 1e-12 atol = 1e-12

        xmin, umin = DECAES.minimize(spl)
        @test umin ≈ spl(xmin) rtol = 1e-12 atol = 1e-12
        @test all(spl(x) >= umin - 1e-12 for x in range(a, b; length = 1024 + 1))
    end
end

function test_mock_surrogate_search_problem(
    opts::T2mapOptions = DECAES.mock_t2map_opts(;
        MatrixSize = (1, 1, 1),
        nRefAngles = 8,
        nTE = 11,
    ),
)
    function A(α, β)
        theta = DECAES.EPGOptions((; ETL = opts.nTE, α = α, TE = opts.TE, T2 = 0.0, T1 = opts.T1, β = β))
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
        @testset "basics" test_poly()
        @testset "quadratic" test_quadratic_roots()
    end
    @testset "cubic" begin
        test_cubic_splines()
    end
    @testset "cubic hermite interpolator" begin
        test_minimize_cubic_hermite_interpolator()
    end
    @testset "mock surrogate search problem" begin
        test_mock_surrogate_search_problem()
    end
    @testset "bounding box" begin
        @testset "basics" test_bounding_box()
        @testset "discrete searcher" test_discrete_searcher()
    end
end
