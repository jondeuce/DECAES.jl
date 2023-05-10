import Pkg
Pkg.activate(@__DIR__)

using DECAES
using DECAES.NormalHermiteSplines
const nhs = DECAES.NormalHermiteSplines

# Packages from this local env
using Random
using LinearAlgebra
using StaticArrays
using BenchmarkTools
using LaTeXStrings
using CairoMakie
set_theme!(theme_ggplot2(); resolution = (500,400), font = "CMU Serif")

function plot_neighbours(::Val{D} = Val(2)) where {D}
    grid  = DECAES.meshgrid(SVector{D,Float64}, [range(0, 1; length = 25) for _ in 1:D]...)
    surr  = DECAES.HermiteSplineSurrogate(I -> (1.0, zero(SVector{D,Float64})), grid)
    state = DECAES.DiscreteSurrogateSearcher(surr; mineval = 5, maxeval = typemax(Int))

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
        scatter!(xy([xnew])...; markersize = 25, color = :red, marker = :star5, label = "query")
        fig[1,2] = Legend(fig, ax)
        display(fig)
        readline()
    end
end

function plot_bisection_search(
        ::Val{D},
        ::Val{ETL},
        prob::DECAES.NNLSDiscreteSurrogateSearch{D,T},
        opts::DECAES.T2mapOptions{T};
        surrtype = :hermite,
        npts     = opts.nRefAngles,
        mineval  = opts.nRefAnglesMin,
        maxeval  = npts,
    ) where {D,T,ETL}
    surrtype === :cubic && @assert D == 1 "cubic splines only support 1D"

    # build surrogate
    surr = surrtype === :cubic ?
        DECAES.CubicSplineSurrogate(prob) :
        DECAES.HermiteSplineSurrogate(prob)
    f_true = function(x...)
        α, β = length(x) == 1 ? (x[1], opts.RefConAngle) : (x[1], x[2])
        θ = DECAES.EPGOptions((; α = α, TE = opts.TE, T2 = 0.0, T1 = opts.T1, β = β), Val(ETL), T)
        A = DECAES.epg_decay_basis(θ, DECAES.logrange(opts.T2Range..., opts.nT2))
        nnls_prob = DECAES.NNLSProblem(A, prob.b)
        DECAES.solve!(nnls_prob, A, prob.b)
        return log(DECAES.chi2(nnls_prob))
    end

    # solve discrete search problem
    state = DECAES.DiscreteSurrogateSearcher(surr; mineval = mineval, maxeval = maxeval)
    minx, miny = DECAES.bisection_search(surr, state; maxeval = min(D == 1 ? 5 : 12, maxeval))
    # x₀ = (minx + DECAES.centre(state, DECAES.minimal_bounding_box(state, minx))) / 2
    # x₀ = DECAES.centre(state, DECAES.minimal_bounding_box(state, minx))
    # x₀ = DECAES.is_inside(state, minx) ? minx : DECAES.nearest_interior_gridpoint(surr.grid, minx)[2]
    x₀ = DECAES.nearest_interior_gridpoint(surr.grid, minx)[2]
    minx, miny = DECAES.local_search(surr, x₀, state; maxeval = maxeval)
    # xmid = DECAES.centre(state, DECAES.minimal_bounding_box(state, minx))
    # xopt = DECAES.nearest_interior_gridpoint(surr.grid, minx)[2]
    # minx₂, miny₂ = DECAES.local_search(surr, xopt, state; maxeval = maxeval)
    # minx₁, miny₁ = DECAES.local_search(surr, xmid, state; maxeval = maxeval)
    # minx = ifelse(miny₁ < miny₂, minx₁, minx₂)

    # reconstruct surrogate from evaluated points and plot
    if surrtype === :cubic
        nodes = () -> surr.p
        values = () -> surr.u
        spl = DECAES.make_spline(first.(nodes()), values())
    else
        nodes = () -> map(x -> nhs._unnormalize(surr.spl, x), nhs._get_nodes(surr.spl))
        values = () -> nhs._get_values(surr.spl)
        spl = (x...) -> nhs.evaluate(surr.spl, SVector(x...))
    end

    if D == 1
        fig = Figure()
        ax = fig[1,1] = Axis(fig; xlabel = L"Flip Angle $\alpha$")
        lines!(surr.grid[1][1]..surr.grid[end][1], f_true; color = :darkblue, linestyle = :dash, label = L"||CX-d||_2^2")
        lines!(surr.grid[1][1]..surr.grid[end][1], x -> spl(x); color = :darkred, label = "Spline")
        scatter!(first.(nodes()), values(); markersize = 8, color = :blue, label = "Evaluated")
        scatter!(first.(minx), [miny]; markersize = 8, color = :red, marker = :circle, label = "Minimizer")
        fig[1,2] = Legend(fig, ax)
        return fig
    else
        xs = LinRange(surr.grid[1,1][1], surr.grid[end,end][1], 100)
        ys = LinRange(surr.grid[1,1][2], surr.grid[end,end][2], 100)
        zs = spl.(xs, ys')
        zs_true = f_true.(xs, ys')
        zmin_true, Imin_true = findmin(zs_true)

        fig = Figure()
        ax1 = Axis(fig[1,1]; title = L"$\sigma(x)$")
        contourf!(ax1, xs, ys, zs; levels = 50) # limits = extrema(zs_true)
        scatter!(ax1, (p->p[1]).(nodes()), (p->p[2]).(nodes()), values(); marker = :rect, markersize = 10, color = :black, label = "pts")
        scatter!(ax1, [minx[1]], [minx[2]]; markersize = 15, color = :red, marker = :star5, label = "spl min")

        ax2 = Axis(fig[1,2]; title = L"f(x)")
        cont2 = contourf!(ax2, xs, ys, zs_true; levels = 50) # limits = extrema(zs_true)
        scatter!(ax2, (p->p[1]).(nodes()), (p->p[2]).(nodes()), values(); marker = :rect, markersize = 10, color = :black, label = "pts")
        scatter!(ax2, [xs[Imin_true[1]]], [ys[Imin_true[2]]]; markersize = 15, color = :white, marker = :star5, label = "true min")
        scatter!(ax2, [minx[1]], [minx[2]]; markersize = 15, color = :red, marker = :star5, label = "spl min")

        # Colorbar(fig[2,1:2], cont2; vertical = false, flipaxis = false)
        hidedecorations!.([ax1, ax2])
        return fig
    end
end

function plot_bisection_search(
        ::Val{D}; 
        flip   = 150.0,
        refcon = 180.0,
        npts   = 32,
        kwargs...,
    ) where {D}
    opts = DECAES.mock_t2map_opts(; MatrixSize = (1,1,1), nTE = 32, SetFlipAngle = flip, RefConAngle = refcon, nRefAngles = npts)
    prob = DECAES.mock_surrogate_search_problem(Val(D), Val(32), opts)
    plot_bisection_search(Val(D), Val(32), prob, opts; npts = npts, kwargs...)
end

function plot_bisection_search(b::AbstractVector, opts::T2mapOptions, ::Val{D}; kwargs...) where {D}
    prob = DECAES.mock_surrogate_search_problem(b, opts, Val(D))
    plot_bisection_search(Val(D), Val(opts.nTE), prob, opts; npts = opts.nRefAngles, kwargs...)
end

function sequential_plot_bisection_search(; 
        flip = 135.0,
        npts = 32,
        snr = 40,
        seed = 0,
        kwargs...,
    )
    Random.seed!(seed)
    opts = DECAES.mock_t2map_opts(; MatrixSize = (1,1,1), nTE = 32, SetFlipAngle = flip, nRefAngles = npts, nRefAnglesMin = 3)
    b = vec(DECAES.mock_image(opts; SNR = snr))
    for maxeval in 3:7
        prob = DECAES.mock_surrogate_search_problem(b, opts, Val(1), Val(32))
        fig = plot_bisection_search(Val(1), Val(32), prob, opts; npts = npts, maxeval = maxeval, kwargs...)
        display(fig)
        # save("hermite_spline_maxeval-$(maxeval).png", fig)
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
    @btime $(nhs.evaluate)($spl, $xi[])
    @btime $(nhs.evaluate)($dspl, $xi[])
    @btime $(nhs.evaluate_derivative)($spl, $xi[])
    @btime $(nhs.evaluate_derivative)($dspl, $xi[])
end
