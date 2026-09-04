Random.seed!(0) # reproducible randomized tests

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
            a, b = randn() .+ (1 + rand()) .* (-0.5, 0.5)
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

                x̄, px̄ = DECAES.maximize_cubic((coeffs...,), a, b)
                @test a <= x̄ <= b
                @test all(p(x) <= px̄ + 1e-12 for x in xs)
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
    # Not-a-knot should reproduce cubic data exactly
    @testset "not-a-knot on cubic data, npts = $npts" for npts in 4:8
        c = ntuple(_ -> randn(), 4)
        x = sort(randn(npts))
        y = evalpoly.(x, Ref(c))
        spl = DECAES.cubic_spline(x, y)
        @test all(isapprox(spl(t), evalpoly(t, c); rtol = 1e-14, atol = 1e-14) for t in range(x[1], x[end]; length = 64))
    end

    @testset "bc = $bc, npts = $npts" for bc in (:notaknot, :natural, :zeroslope, (:notaknot, :zeroslope), (:zeroslope, :notaknot), (:natural, :zeroslope), (:notaknot, :natural)), npts in 2:8
        x = collect(range(-0.5, 2.0; length = npts))
        npts > 2 && (x[2:end-1] .+= (0.4 * step(range(-0.5, 2.0; length = npts))) .* (rand(npts - 2) .- 0.5)) # unequal spacing
        y = randn(npts)
        spl = DECAES.cubic_spline(x, y; bc)

        # Interpolation is exact at the knots
        @test all(spl(x[i]) ≈ y[i] for i in 1:npts)

        # Second derivatives agree from either side of each interior knot, and the end conditions hold
        h(i) = x[i+1] - x[i]
        δ(i) = (y[i+1] - y[i]) / h(i)
        curvature_right(i) = 2 * (3 * δ(i) - 2 * spl.m[i] - spl.m[i+1]) / h(i)
        curvature_left(i) = 2 * (spl.m[i-1] + 2 * spl.m[i] - 3 * δ(i - 1)) / h(i - 1)
        scale = maximum(abs, y) / minimum(h(i) for i in 1:npts-1)^2
        for i in 2:npts-1
            @test curvature_left(i) ≈ curvature_right(i) rtol = 1e-8 atol = 1e-8 * scale
        end
        bcl, bcr = DECAES.boundary_conditions(bc)
        npts < 3 && ((bcl, bcr) = (bcl === :notaknot ? :natural : bcl, bcr === :notaknot ? :natural : bcr)) # not-a-knot needs two intervals on its side
        bcl === :natural && @test abs(curvature_right(1)) <= 1e-8 * scale
        bcr === :natural && @test abs(curvature_left(npts)) <= 1e-8 * scale
        bcl === :zeroslope && @test spl.m[1] == 0
        bcr === :zeroslope && @test spl.m[npts] == 0

        # The minimizer is not larger than a brute-force search
        opt = DECAES.spline_opt(x, y; bc)
        x̄, ȳ = opt.x, opt.y
        xs = range(x[1], x[end]; length = 4096)
        @test x[1] <= x̄ <= x[end]
        @test spl(x̄) ≈ ȳ
        @test ȳ <= minimum(spl, xs) + 1e-14

        # A value below the minimum has no root; a value inside the range produces the leftmost root
        @test isnan(DECAES.spline_root(x, y, ȳ - 1; bc))
        v = (minimum(y) + maximum(y)) / 2
        x̂ = DECAES.spline_root(x, y, v; bc)
        @test x[1] <= x̂ <= x[end]
        @test spl(x̂) ≈ v
        @test all(!isapprox(spl(t), v; rtol = 1e-14, atol = 1e-14) for t in range(x[1], x̂; length = 64)[1:end-1] if t < x̂ - 1e-8)
    end

    @test_throws AssertionError DECAES.cubic_spline([0.0, 0.0], [1.0, 2.0])
    @test_throws AssertionError DECAES.cubic_spline([1.0, 0.0], [1.0, 2.0])
    @test_throws AssertionError DECAES.cubic_spline([0.0, Inf], [1.0, 2.0])
end

function test_cubic_spline_surrogate()
    npts = 4
    x = range(-0.5, 2.0; length = npts)

    coeffs = (-2, -1, -5, 3) # 3x^3 - 5x^2 - x - 2
    f = i -> evalpoly(x[i], coeffs)
    surr = DECAES.CubicSplineSurrogate(f, SVector.(x))

    @test surr.seen[3] == false
    @test all(iszero, surr.idx)
    @test surr.npts[] == 0

    DECAES.update!(surr, CartesianIndex(3))
    @test surr.seen[3] == true
    @test surr.idx == [3; zeros(Int, npts - 1)]
    @test surr.npts[] == 1

    DECAES.update!(surr, CartesianIndex(3))
    @test surr.seen[3] == true
    @test surr.idx == [3; zeros(Int, npts - 1)]
    @test surr.npts[] == 1

    for i in 1:npts
        i == 3 && continue
        @test surr.seen[i] == false
        DECAES.update!(surr, CartesianIndex(i))
        @test issorted(surr.idx[1:surr.npts[]])
        @test surr.seen[i] == true
    end

    @test all(==(true), surr.seen)
    @test surr.idx == 1:npts
    @test surr.npts[] == npts
    @test surr.u == f.(1:npts)

    # Four points should make a Cubic spline an exact surrogate of a cubic function
    p, u = DECAES.suggest_point(surr)
    xtrue, utrue = DECAES.minimize_cubic(float.(coeffs), x[1], x[end])
    @test p[1] ≈ xtrue
    @test u ≈ utrue
end

function test_cubic_hermite_spline_surrogate()
    npts = 4
    x = range(-0.5, 2.0; length = npts)

    coeffs = (-2, -1, -5, 3) # 3x^3 - 5x^2 - x - 2
    ∇coeffs = (-1, -10, 9) # 9x^2 - 10x - 1
    fg = i -> (evalpoly(x[i], coeffs), SVector(evalpoly(x[i], ∇coeffs)))
    surr = DECAES.CubicHermiteSplineSurrogate(fg, SVector.(x))

    @test surr.seen[4] == false
    @test all(iszero, surr.idx)
    @test surr.npts[] == 0

    DECAES.update!(surr, CartesianIndex(4))
    @test surr.seen[4] == true
    @test surr.idx == [4; zeros(Int, npts - 1)]
    @test surr.npts[] == 1

    DECAES.update!(surr, CartesianIndex(4))
    @test surr.seen[4] == true
    @test surr.idx == [4; zeros(Int, npts - 1)]
    @test surr.npts[] == 1

    @test surr.seen[1] == false
    DECAES.update!(surr, CartesianIndex(1))
    @test surr.seen[1] == true
    @test surr.idx == [1; 4; zeros(Int, npts - 2)]
    @test surr.npts[] == 2

    second(x) = x[2]
    @test surr.u[surr.idx[1:2]] == first.(fg.([1, 4]))
    @test surr.∇u[surr.idx[1:2]] == second.(fg.([1, 4]))

    # Two points + two gradients at the endpoints should make a Cubic Hermite spline an exact surrogate of a cubic function
    p, u = DECAES.suggest_point(surr)
    xtrue, utrue = DECAES.minimize_cubic(float.(coeffs), x[1], x[end])
    @test p[1] ≈ xtrue
    @test u ≈ utrue

    x = range(-0.5, 2.0; length = 9)
    fg = i -> (evalpoly(x[i], coeffs), SVector(evalpoly(x[i], ∇coeffs)))
    surr = DECAES.CubicHermiteSplineSurrogate(fg, SVector.(x))
    state = DECAES.DiscreteSurrogateSearcher(surr; mineval = 2, maxeval = 9)
    p, u = DECAES.projected_search(surr, state; maxeval = 9)
    i = searchsortedlast(x, p[1])
    @test state.numeval[] == 4
    @test state.seen[i] && state.seen[i+1]
    @test p[1] ≈ xtrue
    @test u ≈ utrue
end

function hermite_boundary_conditions_iter()
    # Pairs of endpoint slopes
    ms = Iterators.flatten((
        ((s0 * rand(), s1 * rand()) for s0 in (-1, 0, +1), s1 in (-1, 0, +1)), # differing positive, negative, and zero slopes
        (rand() .* (s0, s1) for s0 in (-1, 0, +1), s1 in (-1, 0, +1)), # equal and/or opposite slopes
    ))

    # Pairs of endpoint values
    us = (randn() .+ (s * rand(), -s * rand()) for s in (-1, 0, +1)) # u0 < u1, u0 = u1, and u0 > u1

    iter = ((u0, u1, m0, m1) for ((u0, u1), (m0, m1)) in Iterators.product(us, ms))
    return Iterators.take(Iterators.cycle(iter), 1_000)
end

function test_minimize_cubic_hermite_interpolator()
    for (u0, u1, m0, m1) in hermite_boundary_conditions_iter()
        a, b = randn() .+ (1 + rand()) .* (-0.5, 0.5)
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
        @test evalpoly(t, ∇coeffs) ≈ r * DECAES.ForwardDiff.derivative(spl, x)

        xmin, umin = DECAES.minimize(spl)
        @test umin ≈ spl(xmin) rtol = 1e-12 atol = 1e-12
        @test all(spl(x) >= umin - 1e-12 for x in range(a, b; length = 1024 + 1))
    end
end

function test_mock_surrogate_search_problem(opts::T2mapOptions = DECAES.mock_t2map_opts(; MatrixSize = (1, 1, 1), nRefAngles = 8, nTE = 11))
    function A(α, β)
        theta = DECAES.EPGOptions((; ETL = opts.nTE, α = α, TE = opts.TE, T2 = 0.0, T1 = opts.T1, β = β))
        T2_times = DECAES.logrange(opts.T2Range..., opts.nT2)
        return DECAES.epg_decay_basis(theta, T2_times)
    end

    function f!(work, prob, α, β)
        DECAES.solve!(work, A(α, β), prob.b)
        return DECAES.resnorm_sq(work)
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
        return DECAES.loss_with_grad!(prob, I)
    end

    prob = DECAES.mock_surrogate_search_problem(Val(2), Val(opts.nTE), opts)
    work = DECAES.lsqnonneg_work(zeros(opts.nTE, opts.nT2), zeros(opts.nTE))

    for I in CartesianIndices(prob.αs)
        α, β    = prob.αs[I]
        l′, ∇l′ = fg_approx!(work, prob, α, β; h = 1e-6)

        # Exact QR evaluation path: strict agreement with the independent reference solve
        DECAES.SURROGATE_USE_FAST_GRAM[] = false
        DECAES.reset_warmstart!(prob)
        l, ∇l = fg_surrogate!(prob, I)
        @test l ≈ l′ rtol = 1e-6 atol = 1e-8
        @test ∇l ≈ ∇l′ rtol = 1e-6 atol = 1e-8

        # Precomputed-Gram evaluation path: the same KKT point, so it is held to the reference at the same tolerances as the exact path and to the exact path itself at solver roundoff
        DECAES.SURROGATE_USE_FAST_GRAM[] = true
        DECAES.reset_warmstart!(prob)
        lg, ∇lg = fg_surrogate!(prob, I)
        @test lg ≈ l′ rtol = 1e-6 atol = 1e-8
        @test ∇lg ≈ ∇l′ rtol = 1e-6 atol = 1e-8

        @test lg ≈ l rtol = 1e-10 atol = 1e-10
        @test ∇lg ≈ ∇l rtol = 1e-10 atol = 1e-10
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
    state = DECAES.DiscreteSurrogateSearcher(grid)
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

# The 1D initialization plans `K₀` endpoint-inclusive nodes directly rather than by dyadic recursion.
# No endpoint-inclusive design with `K₀` nodes has a smaller maximum gap, and at the flip-angle defaults the two planners agree exactly.
function test_initialization_plan()
    plan(K, K₀) = DECAES.plan_initialize!(DECAES.DiscreteSurrogateSearcher([SVector{1, Float64}(k) for k in 1:K]); mineval = K₀, maxeval = K)
    function dyadic(K, K₀)
        state = DECAES.DiscreteSurrogateSearcher([SVector{1, Float64}(k) for k in 1:K])
        p = empty!(state.plan)
        for d in 1:K₀
            DECAES.plan_initialize!(p, DECAES.BoundingBox((K,)), d; mineval = K₀, maxeval = K)
            length(p) >= K₀ && break
        end
        return sort!(p)
    end

    @test first.(Tuple.(plan(64, 9))) == [1, 8, 16, 24, 32, 40, 48, 56, 64]
    @test plan(64, 5) == dyadic(64, 5)
    @test plan(64, 9) == dyadic(64, 9)

    for K in 2:40, K₀ in 2:K
        j = first.(Tuple.(plan(K, K₀)))
        @test length(j) == K₀ && allunique(j)
        @test j[1] == 1 && j[end] == K # endpoints included
        @test issorted(j)
        @test maximum(diff(j)) <= cld(K - 1, K₀ - 1) # no endpoint-inclusive design can do better
    end
end

# `bisection_search` may return only a resolved proposal: one that coincides with an evaluated node or lies in a cell whose endpoints are both evaluated.
# Testing the box built for the previous proposal admits a return whose minimizer moved into an unevaluated cell, which leaves the polish interpolating from data the search never evaluated.
function test_search_resolution_invariant()
    grid = [SVector{1, Float64}(α) for α in range(50.0, 180.0; length = 64)]

    # Multi-well objectives whose minimizer migrates between cells as nodes are added
    for trial in 1:200
        c = 50.0 .+ 130.0 .* rand(3)
        w = 0.05 .+ 0.5 .* rand(3)
        a = 0.2 .+ rand(3)
        f(α) = -sum(a[k] * exp(-((α - c[k]) / (10 * w[k]))^2) for k in 1:3)
        df(α) = sum(2 * a[k] * (α - c[k]) / (10 * w[k])^2 * exp(-((α - c[k]) / (10 * w[k]))^2) for k in 1:3)

        surr = DECAES.CubicHermiteSplineSurrogate(I -> (f(grid[I][1]), SVector{1, Float64}(df(grid[I][1]))), grid)
        state = DECAES.DiscreteSurrogateSearcher(surr; mineval = 5, maxeval = length(grid))
        x, _ = DECAES.bisection_search(surr, state; maxeval = length(grid))

        @test DECAES.is_resolved(state, x) || state.numeval[] >= length(grid)
        @test state.numeval[] <= length(grid) # evaluation bound |E| <= K

        # A resolved proposal has true endpoint data on both sides, which is what the polish assumes
        i = searchsortedlast(grid, x; by = first)
        if 1 <= i < length(grid) && grid[i][1] < x[1]
            @test state.seen[i] && state.seen[i+1]
        end
    end
end

@testset "Splines" begin
    @testset "poly" begin
        @testset "basics" test_poly()
        @testset "quadratic" test_quadratic_roots()
    end
    @testset "cubic" begin
        @testset "interpolation" test_cubic_splines()
    end
    @testset "surrogate splines" begin
        @testset "cubic" test_cubic_spline_surrogate()
        @testset "cubic hermite" test_cubic_hermite_spline_surrogate()
    end
    @testset "hermite interpolators" begin
        @testset "cubic" test_minimize_cubic_hermite_interpolator()
    end
    @testset "mock surrogate search problem" begin
        test_mock_surrogate_search_problem()
    end
    @testset "bounding box" begin
        @testset "basics" test_bounding_box()
        @testset "discrete searcher" test_discrete_searcher()
    end
    @testset "initialization plan" begin
        test_initialization_plan()
    end
    @testset "search resolution invariant" begin
        test_search_resolution_invariant()
    end
end
