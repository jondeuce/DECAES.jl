if normpath(@__DIR__) ∉ LOAD_PATH
    push!(LOAD_PATH, normpath(@__DIR__))
end

using DECAES
using DECAES.NormalHermiteSplines
using StaticArrays
using Random
using LinearAlgebra

using BenchmarkTools
using CairoMakie
set_theme!(Theme(resolution = (600,450)))

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
        npts     = 25,
        mineval  = 5,
        maxeval  = npts,
    ) where {D}
    surrtype === :cubic && @assert D == 1 "cubic splines only support 1D"

    # grid  = DECAES.meshgrid(SVector{1,Float64}, range(0, 1; length = npts))
    # xopt   = rand()
    # ftrue  = x -> sum(@. 5 * sin(5 * (x - xopt))^2 + (x - xopt)^2)
    # ∇ftrue = x -> @. 50 * sin(5 * (x - xopt)) * cos(5 * (x - xopt)) + 2 * (x - xopt)
    # surr   = surrtype === :cubic ?
    #     DECAES.CubicSplineSurrogate(I -> ftrue(grid[I]), grid, SVector{1,Float64}[], Float64[]) :
    #     DECAES.HermiteSplineSurrogate(I -> ftrue(grid[I]), I -> ∇ftrue(grid[I]), grid, zeros(Float64, size(grid)), SVector{1,Float64}[], Float64[], SVector{1,Float64}[], Float64[])

    # build surrogate
    opts = DECAES.mock_t2map_opts(; MatrixSize = (1,1,1), nTE = 32, SetFlipAngle = 150.0, SetRefConAngle = 90.0)
    prob = DECAES.mock_surrogate_search_problem(Val(D), Val(32); opts = opts)
    surr = surrtype === :cubic ?
        DECAES.CubicSplineSurrogate(prob) :
        DECAES.HermiteSplineSurrogate(prob)
    ftrue = function(x)
        α, β = D == 1 ? (x[1], opts.SetRefConAngle) : (x[1], x[2])
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
    spl = if surrtype === :cubic
        DECAES._make_spline(first.(surr.p), surr.u)
    else
        herm = DECAES.interpolate(surr.p, surr.u, surr.s, surr.e, surr.du, DECAES.RK_H1())
        (x...) -> DECAES.evaluate_one(herm, SVector(x...))
    end

    if D == 1
        fig = Figure()
        ax = fig[1,1] = Axis(fig)
        lines!(ax, surr.grid[1][1]..surr.grid[end][1], ftrue; color = :darkblue, label = "f")
        lines!(ax, surr.grid[1][1]..surr.grid[end][1], x -> spl(x); color = :darkred, label = "spline")
        scatter!(ax, first.(surr.p), surr.u; markersize = 15, color = :blue, label = "samples")
        scatter!(ax, first.(minx), [miny]; markersize = 15, color = :red, marker = :diamond, label = "min")
        fig[1,2] = Legend(fig, ax)
        return fig
    else
        xs, ys = (p->p[1]).(surr.grid), (p->p[2]).(surr.grid)
        zs = spl.(xs, ys)
        fig = Figure()
        ax = Axis(fig[1,1])
        pcont = contourf!(ax, xs[:,1], ys[:,2], zs)
        scatter!(ax, (p->p[1]).(surr.p), (p->p[2]).(surr.p), surr.u; markersize = 15, color = :blue, label = "samples")
        scatter!(ax, [minx[1]], [minx[2]]; markersize = 15, color = :red, marker = :diamond, label = "min")
        Colorbar(fig[1,2], pcont)
        Legend(fig[1,3], ax)
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
