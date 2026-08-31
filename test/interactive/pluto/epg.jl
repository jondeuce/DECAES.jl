### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 64ae8c40-b077-4ea1-b94d-e337b61e32dd
begin
    import Pkg
    Pkg.activate(@__DIR__)
    using CairoMakie, DECAES, PlutoUI
end

# ╔═╡ c0ea4c27-5a82-433a-a152-a313554d420e
md"""
# Interactive T₂ mapping

This notebook is a compact interface for inspecting an echo train and computing its T₂ distribution. It runs from simulated data by default. Enter a NIfTI path to analyze an image.
"""

# ╔═╡ 6313bc11-9f72-43c3-8770-c71e5462df44
md"""
**Data:** $(@bind data_source PlutoUI.Select(["Simulated echo train", "NIfTI image"]))

**Image path:** $(@bind image_path PlutoUI.TextField(default=""))

**Voxel:** x $(@bind voxel_x PlutoUI.NumberField(1:512, default=1)), y $(@bind voxel_y PlutoUI.NumberField(1:512, default=1)), z $(@bind voxel_z PlutoUI.NumberField(1:512, default=1))
"""

# ╔═╡ 39ec1de3-b127-4f31-9c8e-a62342b5ea60
md"""
**Echo spacing:** $(@bind TE_ms PlutoUI.Slider(5.0:0.5:20.0, default=10.0, show_value=true)) ms

**Number of echoes:** $(@bind nTE PlutoUI.Slider(16:4:64, default=32, show_value=true))

**Noise:** $(@bind noise_fraction PlutoUI.Slider(0.0:0.002:0.08, default=0.02, show_value=true))

**Regularization:** $(@bind reg PlutoUI.Select(["lcurve" => "L-curve", "chi2" => "χ²", "reginska" => "Regińska", "gcv" => "GCV", "none" => "None"]))
"""

# ╔═╡ c369469d-47d5-43f8-9419-b5ff7bd74ec4
begin
    if data_source == "NIfTI image"
        isempty(image_path) && error("Enter a NIfTI image path.")
        image = DECAES.load_image(image_path)
        voxel = ntuple(d -> clamp((voxel_x, voxel_y, voxel_z)[d], firstindex(image, d), lastindex(image, d)), 3)
        decay = Float64.(vec(image[voxel..., :]))
        local_nTE = length(decay)
    else
        local_nTE = nTE
        times = TE_ms .* (1:local_nTE)
        clean = @. 0.18 * exp(-times / 18) + 0.82 * exp(-times / 82)
        decay = clean .+ noise_fraction .* sin.(1.7 .* (1:local_nTE))
    end
    decay ./= maximum(decay)
end

# ╔═╡ 445f1bd7-29c2-4775-be92-7625015689f6
begin
    opts = DECAES.T2mapOptions(; MatrixSize=(1, 1, 1), nTE=local_nTE, TE=TE_ms / 1000, nT2=80, T2Range=(TE_ms / 1000, 2.0), Reg=reg, Chi2Factor=reg == "chi2" ? 1.02 : nothing, SaveRegParam=true, Threaded=false, Silent=true)
    maps, distribution = DECAES.T2mapSEcorr(reshape(decay, 1, 1, 1, :), opts)
    t2 = exp.(range(log(opts.T2Range[1]), log(opts.T2Range[2]); length=opts.nT2))
    spectrum = vec(distribution)
end

# ╔═╡ ac43348e-f081-4274-ab67-7e6d20f81c59
begin
    fig = Figure(size=(900, 360))
    ax1 = Axis(fig[1, 1], xlabel="Echo time (ms)", ylabel="Normalized signal", title="Echo train")
    scatterlines!(ax1, TE_ms .* (1:local_nTE), decay, color=:black)
    ax2 = Axis(fig[1, 2], xlabel="T₂ (ms)", ylabel="Amplitude", xscale=log10, title="T₂ distribution")
    lines!(ax2, 1000 .* t2, spectrum, linewidth=3)
    fig
end

# ╔═╡ 3772bb8d-b46c-4bf6-95f1-1a50e89385b7
md"""
Selected regularization parameter: **$(round(maps["mu"][1]; sigdigits=4))**

Geometric-mean T₂: **$(round(1000maps["ggm"][1]; digits=1)) ms**
"""

# ╔═╡ Cell order:
# ╠═64ae8c40-b077-4ea1-b94d-e337b61e32dd
# ╟─c0ea4c27-5a82-433a-a152-a313554d420e
# ╟─6313bc11-9f72-43c3-8770-c71e5462df44
# ╟─39ec1de3-b127-4f31-9c8e-a62342b5ea60
# ╠═c369469d-47d5-43f8-9419-b5ff7bd74ec4
# ╠═445f1bd7-29c2-4775-be92-7625015689f6
# ╠═ac43348e-f081-4274-ab67-7e6d20f81c59
# ╟─3772bb8d-b46c-4bf6-95f1-1a50e89385b7
