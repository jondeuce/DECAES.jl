using Test
using DECAES
using StaticArrays
using BenchmarkTools
using Random
using LinearAlgebra

using NormalHermiteSplines
const nhs = NormalHermiteSplines

using CairoMakie
set_theme!(Theme(resolution = (600,450)))

function spline(x, u, du = nothing)
    if du === nothing
        nhs.interpolate(x, u, nhs.RK_H1())
    else
        nhs.interpolate(x, u, x, du, nhs.RK_H1())
    end
end

function hermite_spline_opt(spl::nhs.NormalSpline)

    # return (; x = first(x), y = first(u), spl = spl, ret = 0)

    # xr = range(minimum(x), maximum(x), length = 50)
    # ymin, imin = findmin(xr) do xi
    #     nhs.evaluate_one(spl, xi)
    # end
    # xmin = xr[imin]
    # return (; x = xmin, y = ymin, spl = spl, ret = 0)

    # opt = NLopt.Opt(:LN_COBYLA, 1) # local, gradient-free, linear approximation of objective
    # opt = NLopt.Opt(:LN_BOBYQA, 1) # local, gradient-free, quadratic approximation of objective
    opt = NLopt.Opt(:GN_ORIG_DIRECT_L, 1) # global, gradient-free, systematically divides search space into smaller hyper-rectangles via a branch-and-bound technique, systematic division of the search domain into smaller and smaller hyperrectangles, "more biased towards local search"
    # opt = NLopt.Opt(:GN_AGS, 1) # global, gradient-free, employs the Hilbert curve to reduce the source problem to the univariate one.
    # opt = NLopt.Opt(:GD_STOGO, 1) # global, with-gradient, systematically divides search space into smaller hyper-rectangles via a branch-and-bound technique, and searching them by a gradient-based local-search algorithm (a BFGS variant)
    opt.lower_bounds = minimum(x)
    opt.upper_bounds = maximum(x)
    opt.xtol_rel = 0.01
    opt.min_objective = function (x, g)
        if length(g) > 0
            @inbounds g[1] = Float64(nhs.evaluate_derivative(spl, x[1]))
        end
        @inbounds Float64(nhs.evaluate_one(spl, x[1]))
    end
    minf, minx, ret = NLopt.optimize(opt, [mean(x)])
    return (; x = first(minx), y = minf, spl = spl, ret = ret)
end

wrap_f(f) = x::SVector{1} -> f(x[1])
wrap_df(df) = x::SVector{1} -> SVector{1}(df(x[1]))

function plot_splines()
    f = x -> sin(x) * exp(-x^2)
    df = x -> cos(x) * exp(-x^2) + sin(x) * (-2x) * exp(-x^2)
    # f = x -> min(abs(x), 0.5)
    # df = x -> ifelse(abs(x) < 0.5, sign(x), zero(x))
    f_ = wrap_f(f)
    df_ = wrap_df(df)

    lb = -2.5
    ub = 2.5
    xs = [lb; lb .+ (ub .- lb) .* sort(rand(8)); ub]
    us = f.(xs)
    dus = df.(xs)

    xps = collect(range(lb, ub, length = 100))

    fig = Figure()
    ax = fig[1, 1] = Axis(fig)
    lines!(lb..ub, f, label = "f", color = :blue)
    # scatter!(xs, us, label = "pts", color = :red)

    opt = DECAES.ADAM{1,Float64}(0.1)
    x′ = SA{Float64}[rand(xs)]
    for i in 1:10
        i == 1 && scatter!([x′[1]], [f_(x′)[1]], label = "0: x = $(round(x′[1], sigdigits = 2))")
        # x′, opt = DECAES.optimize(df_, x′, SA[lb], SA[ub], opt; maxiter = 10, xtol_rel = 1e-2)
        x′, opt = @btime $(DECAES.optimize)($df_, $x′, SA[$lb], SA[$ub], $opt; maxiter = 10, xtol_rel = 1e-2)
        scatter!([x′[1]], [f_(x′)[1]], label = "$i: x = $(round(x′[1], sigdigits = 2))")
    end

    fig[1, 2] = Legend(fig, ax, title = "", framevisible = false)

    return fig
end

function benchmark_spline()
    x = sort(randn(MersenneTwister(0), 10))
    u = randn(MersenneTwister(0), 10)
    du = randn(MersenneTwister(0), 10)
    xi = Ref(mean(x))
    spl = spline(x, u)
    dspl = spline(x, u, du)
    @btime $(nhs._gram!)($(spl._gram.data), $(spl._nodes), $(spl._kernel))
    @btime $(nhs._gram!)($(dspl._gram.data), $(dspl._nodes), $(dspl._d_nodes), $(dspl._d_dirs), $(dspl._kernel))
    @btime $(nhs.evaluate_one)($spl, $xi[])
    @btime $(nhs.evaluate_one)($dspl, $xi[])
    @btime $(nhs.evaluate_derivative)($spl, $xi[])
    @btime $(nhs.evaluate_derivative)($dspl, $xi[])
end

function test_mock_surrogate_search_problem(
        o::T2mapOptions = DECAES.mock_t2map_opts(; MatrixSize = (1, 1, 1))
    )
    function fA(α, β)
        theta = DECAES.EPGOptions{Float64,32}(α, o.TE, 0.0, o.T1, β)
        T2_times = DECAES.logrange(o.T2Range..., o.nT2)
        DECAES.epg_decay_basis(theta, T2_times)
    end

    function f!(work, prob, α, β)
        A = fA(α, β)
        DECAES.solve!(work, A, prob.b)
        return DECAES.chi2(work)
    end

    function ∇f_approx!(work, prob, α, β; h)
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

    function ∇f_surrogate!(prob, I)
        l  = DECAES.loss!(prob, I)
        ∇l = DECAES.∇loss!(prob, I)
        return l, ∇l
    end

    prob = DECAES.mock_surrogate_search_problem(o)
    work = DECAES.lsqnonneg_work(zeros(o.nTE, o.nT2), zeros(o.nTE))

    for I in CartesianIndices(prob.αs)
        α, β = prob.αs[I]
        l′, ∇l′ = ∇f_approx!(work, prob, α, β; h = 1e-3)
        l , ∇l  = ∇f_surrogate!(prob, I)
        @test l == l′
        @test ∇l ≈ ∇l′ rtol = 1e-3 atol = 1e-6
    end
end
