if normpath(@__DIR__) ∉ LOAD_PATH
    pushfirst!(LOAD_PATH, normpath(@__DIR__, "../.."))
    pushfirst!(LOAD_PATH, normpath(@__DIR__))
end

using DECAES
using Statistics
using LaTeXStrings
using CairoMakie
set_theme!(theme_minimal(); resolution = (800, 600), font = "CMU Serif")

const AbstractTensor{N} = AbstractArray{T, N} where {T}

gridsize(n) = (imax = floor(Int, sqrt(n)); jmax = ceil(Int, n / imax); (imax, jmax))

function global_reduction(reducer, img::AbstractTensor{4}, mask = (img[:, :, :, 1] .> 0))
    return [reducer(img[mask, echo]) for echo in axes(img, 4)]
end

function plot_echoes(img::AbstractTensor{4}; echoes)
    @assert size(img, 3) == 1
    fig = Figure()
    nplots = length(echoes)
    limits = extrema(vec(img))
    for (i, I) in enumerate(CartesianIndices(gridsize(nplots)))
        i > nplots && break
        ax = Axis(fig[I[1], I[2]]; aspect = AxisAspect(1))
        hidedecorations!(ax)
        heatmap!(img[:, :, 1, echoes[i]]; limits = limits)
        text!("$(echoes[i])"; color = :white, position = (1.5, 1.5), align = (:left, :baseline))
    end
    return fig
end

function plot_band!(ax, ydata::AbstractTensor{4}, xdata = 1:size(ydata, 4); meankwargs, bandkwargs)
    μ = global_reduction(mean, ydata)
    σ = global_reduction(std, ydata)
    (meankwargs !== nothing) && scatterlines!(ax, xdata, μ; meankwargs...)
    return (bandkwargs !== nothing) && band!(ax, xdata, μ .- σ, μ .+ σ; bandkwargs...)
end

function plot_global_signal(img::AbstractTensor{4};
    meankwargs = (label = L"\mu", color = :red, markersize = 5, linewidth = 1),
    bandkwargs = (label = L"\mu \pm \sigma", color = (:red, 0.5)),
)
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot_band!(ax, img; meankwargs = meankwargs, bandkwargs = bandkwargs)
    axislegend(ax)
    return fig
end

function compare_refcon_correction(
    img::AbstractTensor{4},
    mask::AbstractTensor{3},
    t2maps_norefcon::AbstractDict,
    t2dist_norefcon::AbstractTensor{4},
    t2maps_refcon::AbstractDict,
    t2dist_refcon::AbstractTensor{4},
)
    (; t2times, echotimes) = t2maps_norefcon
    t2times, echotimes = 1000 .* t2times, 1000 .* echotimes
    globalmean(x) = vec(mean(x[mask, :]; dims = 1))

    fig = Figure()
    ax = Axis(fig[1, 1]; ylabel = "mean signal [a.u.]", xlabel = "echo time [ms]")
    lines!(ax, echotimes, globalmean(img); label = "data", markersize = 5, linewidth = 3, color = :red)
    lines!(ax, echotimes, globalmean(t2maps_norefcon["decaycurve"]); label = "no correction", markersize = 5, linewidth = 3, color = :blue)
    lines!(ax, echotimes, globalmean(t2maps_refcon["decaycurve"]); label = "w/ correction", markersize = 5, linewidth = 3, color = :green)
    hideydecorations!(ax; label = false)
    axislegend(ax)

    ax = Axis(fig[1, 2]; ylabel = "mean difference [a.u.]", xlabel = "echo time [ms]")
    hlines!(ax, [0.0]; label = "zero", linewidth = 3, linestyle = :dot, color = :red)
    lines!(ax, echotimes, globalmean(img .- t2maps_norefcon["decaycurve"]); label = "no correction", markersize = 5, linewidth = 3, color = :blue)
    lines!(ax, echotimes, globalmean(img .- t2maps_refcon["decaycurve"]); label = "w/ correction", markersize = 5, linewidth = 3, color = :green)
    hideydecorations!(ax; label = false)
    axislegend(ax)

    ax = Axis(fig[2, 1]; xlabel = "fit error [a.u.]", ylabel = "relative count")
    hist!(ax, t2maps_norefcon["resnorm"][mask]; label = "no correction", color = (:blue, 0.5), scale_to = 0.9, offset = 1, bins = 50)
    hist!(ax, t2maps_refcon["resnorm"][mask]; label = "w/ correction", color = (:green, 0.5), scale_to = 0.9, offset = 0, bins = 50)
    hideydecorations!(ax; label = false)
    axislegend(ax)

    ax = Axis(fig[2, 2]; xlabel = "T2 times [ms]", ylabel = "mean T2 dist. [a.u.]", xscale = log10)
    band!(ax, t2times, zero(t2times), globalmean(t2dist_norefcon); label = "no correction", linewidth = 3, color = (:blue, 0.5))
    band!(ax, t2times, zero(t2times), globalmean(t2dist_refcon); label = "w/ correction", linewidth = 3, color = (:green, 0.5))
    scatter!(ax, t2times, globalmean(t2dist_norefcon); markersize = 5, color = :blue)
    scatter!(ax, t2times, globalmean(t2dist_refcon); markersize = 5, color = :green)
    hideydecorations!(ax; label = false)
    axislegend(ax)

    return fig
end
