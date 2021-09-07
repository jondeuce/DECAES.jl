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

function DECAES.spline_opt(
        spl::nhs.NormalSpline,
        prob::DECAES.NNLSDiscreteSurrogateSearch{D};
        # alg = :LN_COBYLA,        # local, gradient-free, linear approximation of objective
        # alg = :LN_BOBYQA,        # local, gradient-free, quadratic approximation of objective
        alg = :LD_SLSQP,         # local, with-gradient, "Sequential Least-Squares Quadratic Programming"; uses dense-matrix methods (ordinary BFGS, not low-storage BFGS)
        # alg = :GN_ORIG_DIRECT_L, # global, gradient-free, systematically divides search space into smaller hyper-rectangles via a branch-and-bound technique, systematic division of the search domain into smaller and smaller hyperrectangles, "more biased towards local search"
        # alg = :GN_AGS,           # global, gradient-free, employs the Hilbert curve to reduce the source problem to the univariate one.
        # alg = :GD_STOGO,         # global, with-gradient, systematically divides search space into smaller hyper-rectangles via a branch-and-bound technique, and searching them by a gradient-based local-search algorithm (a BFGS variant)
    ) where {D}

    nhs.evaluate!(prob.ℓ, spl, prob.αs)
    uopt, i = findmin(prob.ℓ)
    xopt = prob.αs[i]

    opt = DECAES.NLopt.Opt(alg, D)
    opt.lower_bounds = Float64[prob.αs[begin]...]
    opt.upper_bounds = Float64[prob.αs[end]...]
    opt.xtol_rel = 0.001
    opt.min_objective = function (x, g)
        if length(g) > 0
            @inbounds g[1] = Float64(nhs.evaluate_derivative(spl, x[1]))
        end
        @inbounds Float64(nhs.evaluate_one(spl, x[1]))
    end
    minf, minx, ret = DECAES.NLopt.optimize(opt, Float64[xopt[1]])
    return (; xopt = SA{Float64}[minx[1]], uopt = minf)
end

wrap_f(f) = x::SVector{1} -> f(x[1])
wrap_df(df) = x::SVector{1} -> SVector{1}(df(x[1]))

function plot_splines(; kwargs...)

    T = Float64
    S = SVector{1,T}
    ETL = 32
    opts = DECAES.mock_t2map_opts(; MatrixSize = (1, 1, 1), nTE = ETL, kwargs...)
    prob = DECAES.mock_surrogate_search_problem(Val(1); opts = opts)
    θ = DECAES.EPGOptions{T,ETL}(opts.SetFlipAngle, opts.TE, 0.0, opts.T1, opts.SetRefConAngle)

    function f_true(α)
        θα = DECAES.EPGOptions(θ, (α=α,))
        T2s = DECAES.logrange(opts.T2Range..., opts.nT2)
        A = DECAES.epg_decay_basis(θα, T2s)
        DECAES.solve!(prob.nnls_work, A, prob.b)
        return log(DECAES.chi2(prob.nnls_work))
    end

    function build_spline(Is, ::Val{use_grad} = Val(false)) where {use_grad}
        n = length(Is)
        x, u, e, du = zeros(S, n), zeros(T, n), fill(ones(S), n), zeros(T, n)
        for (i,I) in enumerate(Is)
            x[i] = prob.αs[I]
            u[i] = DECAES.loss!(prob, I)
            du[i] = DECAES.∇loss!(prob, I)[1] # ∇loss! must be called after loss!, as it relies on internal state
            u[i], du[i] = log(u[i]), du[i] / u[i] # transform to log-scale
        end
        spl = use_grad ?
            nhs.interpolate(x, u, x, e, du, nhs.RK_H1()) :
            nhs.interpolate(x, u, nhs.RK_H1())
        return (; x, u, e, du, spl)
    end

    fig = Figure()
    ax = fig[1, 1] = Axis(fig)
    lines!(opts.MinRefAngle..180.0, f_true, label = "f (true)", color = :blue, linewidth = 3, linestyle = :dash)

    for nα in [3]
        Is = CartesianIndex.(unique(round.(Int, range(1, length(prob.αs); length = nα))))

        DECAES.@unpack x, u, spl = build_spline(Is, Val(false))
        DECAES.@unpack xopt, uopt = DECAES.spline_opt(spl, prob)
        lines!(opts.MinRefAngle..180.0, x_ -> nhs.evaluate_one(spl, x_), label = "no grad", color = :green, linewidth = 4)
        scatter!([xopt[1]], [uopt], label = "min", color = :darkgreen, marker = :diamond, markersize = 20)

        DECAES.@unpack x, u, spl = build_spline(Is, Val(true))
        DECAES.@unpack xopt, uopt = DECAES.spline_opt(spl, prob)
        lines!(opts.MinRefAngle..180.0, x_ -> nhs.evaluate_one(spl, x_), label = "w/ grad", color = :red, linewidth = 4)
        scatter!([xopt[1]], [uopt], label = "min", color = :darkred, marker = :diamond, markersize = 20)

        scatter!(first.(x), u, label = "samples", color = :black)
    end

    fig[1, 2] = Legend(fig, ax, "HERMITE", framevisible = true)

    return fig
end

function benchmark_spline()
    x = sort(randn(MersenneTwister(0), 10))
    u = randn(MersenneTwister(0), 10)
    du = randn(MersenneTwister(0), 10)
    xi = Ref(mean(x))
    spl = nhs.interpolate(x, u, nhs.RK_H1())
    dspl = nhs.interpolate(x, u, x, du, nhs.RK_H1())
    @btime $(nhs._gram!)($(spl._gram.data), $(spl._nodes), $(spl._kernel))
    @btime $(nhs._gram!)($(dspl._gram.data), $(dspl._nodes), $(dspl._d_nodes), $(dspl._d_dirs), $(dspl._kernel))
    @btime $(nhs.evaluate_one)($spl, $xi[])
    @btime $(nhs.evaluate_one)($dspl, $xi[])
    @btime $(nhs.evaluate_derivative)($spl, $xi[])
    @btime $(nhs.evaluate_derivative)($dspl, $xi[])
end

function test_mock_surrogate_search_problem(
        opts::T2mapOptions = DECAES.mock_t2map_opts(; MatrixSize = (1, 1, 1))
    )
    function fA(α, β)
        theta = DECAES.EPGOptions{Float64,32}(α, opts.TE, 0.0, opts.T1, β)
        T2_times = DECAES.logrange(opts.T2Range..., opts.nT2)
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

    prob = DECAES.mock_surrogate_search_problem(Val(2); opts = opts)
    work = DECAES.lsqnonneg_work(zeros(opts.nTE, opts.nT2), zeros(opts.nTE))

    for I in CartesianIndices(prob.αs)
        α, β = prob.αs[I]
        l′, ∇l′ = ∇f_approx!(work, prob, α, β; h = 1e-3)
        l , ∇l  = ∇f_surrogate!(prob, I)
        @test l == l′
        @test ∇l ≈ ∇l′ rtol = 1e-3 atol = 1e-6
    end
end

function bounding_neighbours_test()
    xgrid = DECAES.meshgrid(SVector{2,Float64}, 1:5, 1:10)
    I = CartesianIndex.([(1,3), (1,10), (2,1), (3,7), (4,2), (5,1), (5,9)])
    x = xgrid[I]
    ℓ = rand(length(I))
    state = DECAES.DiscreteBisectionSearch(xgrid, I, x, ℓ)
    @test DECAES.bounding_neighbours(state, SA[1.5, 3.5]) == (CartesianIndex(1, 3), CartesianIndex(2, 7))
    @test DECAES.bounding_neighbours(state, SA[0.0, 0.0]) == (CartesianIndex(1, 1), CartesianIndex(1, 1))
    @test DECAES.bounding_neighbours(state, SA[6.0, 11.0]) == (CartesianIndex(5, 10), CartesianIndex(5, 10))

    f = x -> sum(sin.(x).^2) * sum(x.^2)
    xgrid = DECAES.meshgrid(SVector{1,Float64}, range(-10*rand(), 10*rand(), length = 21))
    I = CartesianIndex.(round.(Int, range(1, length(xgrid), length = 5)))
    x = xgrid[I]
    ℓ = f.(x)

    state = DECAES.DiscreteBisectionSearch(xgrid, I, x, ℓ)
    out = DECAES.bisection_search(f, state)
    out_legacy = DECAES.surrogate_spline_opt(i -> f(xgrid[i]), first.(xgrid); mineval = 5)
    @test out.x[1] ≈ out_legacy.x && out.y ≈ out_legacy.y
end
