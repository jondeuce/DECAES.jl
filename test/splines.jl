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

    nhs.evaluate!(prob.v, spl, prob.αs)
    ymin, imin = findmin(prob.v)
    xmin = prob.αs[imin]
    return (; x = xmin, y = ymin)

    # opt = NLopt.Opt(alg, D)
    # opt.lower_bounds = minimum(x)
    # opt.upper_bounds = maximum(x)
    # opt.xtol_rel = 0.01
    # opt.min_objective = function (x, g)
    #     if length(g) > 0
    #         @inbounds g[1] = Float64(nhs.evaluate_derivative(spl, x[1]))
    #     end
    #     @inbounds Float64(nhs.evaluate_one(spl, x[1]))
    # end
    # minf, minx, ret = NLopt.optimize(opt, [mean(x)])
    # return (; x = first(minx), y = minf, spl = spl, ret = ret)
end

spline(x, u, du = nothing) = du === nothing ?
    nhs.interpolate(x, u, nhs.RK_H1()) :
    nhs.interpolate(x, u, x, du, nhs.RK_H1())

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
        lines!(opts.MinRefAngle..180.0, x_ -> nhs.evaluate_one(spl, x_), label = "no grad", color = :green, linewidth = 4)

        DECAES.@unpack x, u, spl = build_spline(Is, Val(true))
        lines!(opts.MinRefAngle..180.0, x_ -> nhs.evaluate_one(spl, x_), label = "with grad", color = :red, linewidth = 4)

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
