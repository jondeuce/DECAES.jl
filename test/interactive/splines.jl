if normpath(@__DIR__) ∉ LOAD_PATH
    pushfirst!(LOAD_PATH, normpath(@__DIR__, "../.."))
    pushfirst!(LOAD_PATH, normpath(@__DIR__))
end

using DECAES
using DECAES.NormalHermiteSplines
using StaticArrays
using Random
using LinearAlgebra

# Packages from this local env
using BenchmarkTools
using LaTeXStrings
using CairoMakie
set_theme!(theme_ggplot2(); resolution = (600,450), font = "CMU Serif")

function plot_neighbours(::Val{D} = Val(2)) where {D}
    grid  = DECAES.meshgrid(SVector{D,Float64}, [range(0, 1; length = 25) for _ in 1:D]...)
    surr  = DECAES.HermiteSplineSurrogate(I -> 1.0, I -> zero(SVector{D,Float64}), grid)
    state = DECAES.DiscreteSurrogateBisector(surr; mineval = 5, maxeval = typemax(Int))

    second(x) = x[2]
    xy(xs) = D == 1 ? (first.(xs), ones(length(xs))) : vec.((first.(xs), second.(xs)))

    while true
        xnew = rand(SVector{D,Float64})
        box = DECAES.minimal_bounding_box(state, xnew)
        oldseen = copy(state.seen)
        DECAES.evaluate_box!(surr, state, box; maxeval = typemax(Int))
        newseen = state.seen .!= oldseen

        fig = Figure()
        ax = fig[1,1] = Axis(fig)
        scatter!(xy(state.grid)...; markersize = 5, color = :black, axis = (; aspect = AxisAspect(1.0)), label = "grid")
        scatter!(xy(state.grid[oldseen])...; markersize = 20, color = :darkblue, label = "old pts")
        scatter!(xy(state.grid[newseen])...; markersize = 20, color = :orange, label = "new pts")
        scatter!(xy([xnew])...; markersize = 25, color = :red, marker = :diamond, label = "query")
        fig[1,2] = Legend(fig, ax)
        display(fig)
        readline()
    end
end

function plot_bisection_search(
        ::Val{D} = Val(2);
        surrtype = :hermite,
        npts     = 32,
        mineval  = 5,
        maxeval  = npts,
        flip     = 175.0,
        refcon   = 150.0,
    ) where {D}
    surrtype === :cubic && @assert D == 1 "cubic splines only support 1D"

    # grid    = DECAES.meshgrid(SVector{1,Float64}, range(0, 1; length = npts))
    # xopt    = rand()
    # f_true  = x -> sum(@. 5 * sin(5 * (x - xopt))^2 + (x - xopt)^2)
    # ∇f_true = x -> @. 50 * sin(5 * (x - xopt)) * cos(5 * (x - xopt)) + 2 * (x - xopt)
    # surr    = surrtype === :cubic ?
    #     DECAES.CubicSplineSurrogate(I -> f_true(grid[I]), grid, SVector{1,Float64}[], Float64[]) :
    #     DECAES.HermiteSplineSurrogate(I -> f_true(grid[I]), I -> ∇f_true(grid[I]), grid, zeros(Float64, size(grid)), SVector{1,Float64}[], Float64[], SVector{1,Float64}[], Float64[])

    # build surrogate
    opts = DECAES.mock_t2map_opts(; MatrixSize = (1,1,1), nTE = 32, SetFlipAngle = flip, SetRefConAngle = refcon, nRefAngles = npts)
    prob = DECAES.mock_surrogate_search_problem(Val(D), Val(32); opts = opts)
    surr = surrtype === :cubic ?
        DECAES.CubicSplineSurrogate(prob) :
        DECAES.HermiteSplineSurrogate(prob)
    f_true = function(x...)
        α, β = length(x) == 1 ? (x[1], opts.SetRefConAngle) : (x[1], x[2])
        θ = DECAES.EPGOptions{Float64,32}(α, opts.TE, 0.0, opts.T1, β)
        A = DECAES.epg_decay_basis(θ, DECAES.logrange(opts.T2Range..., opts.nT2))
        nnls_prob = DECAES.NNLSProblem(A, prob.b)
        DECAES.solve!(nnls_prob, A, prob.b)
        return log(DECAES.chi2(nnls_prob))
    end

    # solve discrete search problem
    state = DECAES.DiscreteSurrogateBisector(surr; mineval = mineval, maxeval = maxeval)
    minx, miny = DECAES.bisection_search(surr, state; maxeval = maxeval)

    # reconstruct surrogate from evaluated points and plot
    if surrtype === :cubic
        spl = DECAES._make_spline(first.(surr.p), surr.u)
    else
        σ    = DECAES.interpolate(surr.p, surr.u, surr.s, surr.e, surr.du, DECAES.RK_H1())
        spl  = (x...) -> DECAES.evaluate_one(σ, SVector(x...))
        σ₀   = DECAES.interpolate(surr.p, surr.u, DECAES.RK_H1())
        spl₀ = (x...) -> DECAES.evaluate_one(σ₀, SVector(x...))
    end

    if D == 1
        fig = Figure()
        ax = fig[1,1] = Axis(fig)
        lines!(surr.grid[1][1]..surr.grid[end][1], f_true; color = :darkblue, label = "f")
        lines!(surr.grid[1][1]..surr.grid[end][1], x -> spl(x); color = :darkred, label = "spl")
        scatter!(first.(surr.p), surr.u; markersize = 10, color = :blue, label = "pts")
        scatter!(first.(minx), [miny]; markersize = 10, color = :red, marker = :diamond, label = "min")
        fig[1,2] = Legend(fig, ax)
        return fig
    else
        xs = LinRange(surr.grid[1,1][1], surr.grid[end,end][1], 4*npts)
        ys = LinRange(surr.grid[1,1][2], surr.grid[end,end][2], 4*npts)
        zs_withgrad = spl.(xs, ys')
        zs_nograd = spl₀.(xs, ys')
        zs_true = f_true.(xs, ys')

        fig = Figure()
        ax1 = Axis(fig[1,1]; title = L"$\sigma(x)$: no gradient")
        contourf!(ax1, xs, ys, zs_nograd; limits = extrema(zs_true), levels = 50)
        scatter!(ax1, (p->p[1]).(surr.p), (p->p[2]).(surr.p), surr.u; markersize = 5, color = :black, label = "pts")
        scatter!(ax1, [minx[1]], [minx[2]]; markersize = 10, color = :red, marker = :diamond, label = "min")

        ax2 = Axis(fig[1,2]; title = L"$\sigma(x)$: with gradient")
        contourf!(ax2, xs, ys, zs_withgrad; limits = extrema(zs_true), levels = 50)
        scatter!(ax2, (p->p[1]).(surr.p), (p->p[2]).(surr.p), surr.u; markersize = 5, color = :black, label = "pts")
        scatter!(ax2, [minx[1]], [minx[2]]; markersize = 10, color = :red, marker = :diamond, label = "min")

        ax3 = Axis(fig[1,3]; title = L"f(x)")
        cont3 = contourf!(ax3, xs, ys, zs_true; limits = extrema(zs_true), levels = 50)
        scatter!(ax3, (p->p[1]).(surr.p), (p->p[2]).(surr.p), surr.u; markersize = 5, color = :black, label = "pts")
        scatter!(ax3, [minx[1]], [minx[2]]; markersize = 10, color = :red, marker = :diamond, label = "min")

        # Colorbar(fig[2,1:3], cont3; vertical = false, flipaxis = false)
        hidedecorations!.([ax1, ax2, ax3])
        return fig
    end
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
