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

# ╔═╡ 6f460dbd-1fe9-4d4d-bd24-1dd24a465e75
begin
    import Pkg
    Pkg.activate(@__DIR__)
    using CairoMakie, DECAES, LaTeXStrings, LinearAlgebra, PlutoUI
end

# ╔═╡ b2d0ca29-568f-4b87-8f80-a10cf1b0dd31
md"""
# Choosing a regularization parameter

DECAES selects the regularization parameter ``\mu`` from properties of the complete solution path. Use the controls below to change the data, selection rule, and penalty. The plots show the selected point against a dense path computed independently.
"""

# ╔═╡ 4581a4b9-b1cb-4916-959b-65baefa932ae
md"""
**Data:** $(@bind data_source PlutoUI.Select(["Two peaks", "Broad distribution", "NIfTI voxel"]))

**Noise level:** $(@bind noise_fraction PlutoUI.Slider(0.002:0.002:0.06, default=0.02, show_value=true))

**Image path:** $(@bind image_path PlutoUI.TextField(default=""))

**Voxel:** x $(@bind voxel_x PlutoUI.NumberField(1:512, default=1)), y $(@bind voxel_y PlutoUI.NumberField(1:512, default=1)), z $(@bind voxel_z PlutoUI.NumberField(1:512, default=1))
"""

# ╔═╡ 57f4c9e0-1a2e-45ed-9dc5-66c81945ee3e
md"""
**Penalty:** $(@bind penalty PlutoUI.Select(["l2" => "ℓ² — Tikhonov", "l1" => "ℓ¹ — lasso"]))

**Method:** $(@bind method PlutoUI.Select(penalty == "l2" ? ["lcurve" => "L-curve", "chi2" => "χ²", "mdp" => "Morozov discrepancy", "reginska" => "Regińska", "gcv" => "GCV"] : ["lcurve" => "L-curve", "chi2" => "χ²", "mdp" => "Morozov discrepancy", "reginska" => "Regińska"]))

**χ² target:** $(@bind chi2_target PlutoUI.Slider(1.002:0.002:1.10, default=1.02, show_value=true))
"""

# ╔═╡ d8b0c8b9-2382-42b4-949f-e97e91fa5804
begin
    TE = 0.010
    t2 = exp.(range(log(TE), log(2.0); length=60))
    if data_source == "NIfTI voxel"
        isempty(image_path) && error("Enter a NIfTI image path.")
        image = DECAES.load_image(image_path)
        voxel = ntuple(d -> clamp((voxel_x, voxel_y, voxel_z)[d], firstindex(image, d), lastindex(image, d)), 3)
        b = Float64.(vec(image[voxel..., :]))
        b ./= maximum(b)
        noise_σ = noise_fraction * norm(b) / sqrt(length(b))
        x_true = nothing
    else
        nTE = 40
        if data_source == "Two peaks"
            x_true = @. 0.20 * exp(-0.5 * (log(t2 / 0.020) / 0.10)^2) + exp(-0.5 * (log(t2 / 0.090) / 0.16)^2)
        else
            x_true = @. exp(-0.5 * (log(t2 / 0.075) / 0.55)^2)
        end
        x_true ./= sum(x_true)
        A = exp.(-(TE .* (1:nTE)) ./ t2')
        clean = A * x_true
        noise_σ = noise_fraction * maximum(clean)
        b = clean .+ noise_σ .* sin.(sqrt(2) .* (1:nTE))
    end
    A = exp.(-(TE .* (1:length(b))) ./ t2')
end;

# ╔═╡ 56aa6caf-c931-4622-bb83-39213d85d019
begin
    x₀ = DECAES.lsqnonneg(A, b)
    R₀ = sum(abs2, A * x₀ - b)
    δ_requested = noise_σ * sqrt(length(b))
    δ = clamp(δ_requested, sqrt(R₀) * 1.002, norm(b) * 0.998)

    if penalty == "l2"
        result = method == "lcurve" ? DECAES.lsqnonneg_lcurve(A, b) :
                 method == "chi2" ? DECAES.lsqnonneg_chi2(A, b, chi2_target) :
                 method == "mdp" ? DECAES.lsqnonneg_mdp(A, b, δ) :
                 method == "reginska" ? DECAES.lsqnonneg_reginska(A, b) :
                 DECAES.lsqnonneg_gcv(A, b)
    else
        result = method == "lcurve" ? DECAES.lsqnonneg_lcurve_lasso(A, b) :
                 method == "chi2" ? DECAES.lsqnonneg_chi2_lasso(A, b, chi2_target) :
                 method == "mdp" ? DECAES.lsqnonneg_mdp_lasso(A, b, δ) :
                 DECAES.lsqnonneg_reginska_lasso(A, b)
    end
    x_selected = copy(result.x)
    μ_selected = result.mu
end;

# ╔═╡ f9fc152b-926f-451b-8bd6-65e58bc9a4b7
begin
    μs = exp.(range(log(1e-5), log(2.0); length=121))
    path = [copy(penalty == "l2" ? DECAES.lsqnonneg_tikh(A, b, μ) : DECAES.lsqnonneg_lasso(A, b, μ)) for μ in μs]
    residual² = [sum(abs2, A * x - b) for x in path]
    seminorm = penalty == "l2" ? [norm(x) for x in path] : [sum(x) for x in path]
    logR = log.(residual²)
    logN = 2log.(seminorm)
    Δt = log(μs[2] / μs[1])
    dX = (logR[3:end] .- logR[1:end-2]) ./ (2Δt)
    dY = (logN[3:end] .- logN[1:end-2]) ./ (2Δt)
    ddX = (logR[3:end] .- 2logR[2:end-1] .+ logR[1:end-2]) ./ Δt^2
    ddY = (logN[3:end] .- 2logN[2:end-1] .+ logN[1:end-2]) ./ Δt^2
    curvature = (dX .* ddY .- dY .* ddX) ./ (dX.^2 .+ dY.^2).^(3 / 2)
    gcv = penalty == "l2" ? [residual²[i] / DECAES.gcv_dof(A, μs[i])^2 for i in eachindex(μs)] : fill(NaN, length(μs))
    reginska_balance = penalty == "l2" ? log.(residual²) .- 2log.(μs) .- 2log.(seminorm) : log.(residual²) .- log.(μs) .- log.(seminorm)
end;

# ╔═╡ 89c75ca9-b622-4c51-b922-ab242771e141
begin
    selected_R = sum(abs2, A * x_selected - b)
    selected_N = penalty == "l2" ? norm(x_selected) : sum(x_selected)
    diagnostic, target = if method == "lcurve"
        curvature, 0.0
    elseif method == "chi2"
        residual² ./ R₀ .- chi2_target, 0.0
    elseif method == "mdp"
        sqrt.(residual²) .- δ, 0.0
    elseif method == "reginska"
        reginska_balance, 0.0
    else
        gcv, minimum(gcv)
    end
    diagnostic_μs = method == "lcurve" ? μs[2:end-1] : μs
    diagnostic_label = method == "lcurve" ? L"\kappa(t)" : method == "chi2" ? L"R(\mu)/R(0)-\chi^2_{\mathrm{target}}" : method == "mdp" ? L"\Vert Ax(\mu)-b\Vert_2-\delta" : method == "reginska" ? (penalty == "l2" ? L"\log\!\left(R/(\mu^2\Vert x\Vert_2^2)\right)" : L"\log\!\left(R/(\mu\Vert x\Vert_1)\right)") : L"\mathcal{G}(\mu)"
end

# ╔═╡ d66d4f7c-daa0-4f7c-951f-aefb15578226
begin
    fig = Figure(size=(1050, 720))
    penalty_name = penalty == "l2" ? "ℓ² Tikhonov penalty" : "ℓ¹ lasso penalty"
    Label(fig[0, :], "$(method) selection with $penalty_name", fontsize=22)
    ax1 = Axis(fig[1, 1], xlabel=L"T_2\;\mathrm{(ms)}", ylabel="Amplitude", xscale=log10, title="Recovered distribution")
    x_true === nothing || lines!(ax1, 1000t2, x_true, color=(:gray, 0.7), linewidth=3, label="truth")
    lines!(ax1, 1000t2, x_selected, color=:dodgerblue, linewidth=3, label="selected")
    axislegend(ax1)

    ax2 = Axis(fig[1, 2], xlabel="Echo time (ms)", ylabel="Signal", title="Data and selected fit")
    scatter!(ax2, 1000TE .* (1:length(b)), b, color=:black, label="data")
    lines!(ax2, 1000TE .* (1:length(b)), A * x_selected, color=:dodgerblue, linewidth=3, label="fit")
    axislegend(ax2)

    lcurve_y_label = penalty == "l2" ? L"\log\Vert x\Vert_2^2" : L"2\log\Vert x\Vert_1"
    ax3 = Axis(fig[2, 1], xlabel=L"\log\Vert Ax-b\Vert_2^2", ylabel=lcurve_y_label, title="L-curve ($penalty_name)")
    lines!(ax3, logR, logN, color=log10.(μs), colormap=:viridis, linewidth=3)
    scatter!(ax3, [log(selected_R)], [2log(selected_N)], color=:red, markersize=16)

    ax4 = Axis(fig[2, 2], xlabel=L"\mu", ylabel=diagnostic_label, xscale=log10, title="$(method) selection diagnostic ($penalty_name)")
    lines!(ax4, diagnostic_μs, diagnostic, linewidth=3)
    hlines!(ax4, [target], color=:gray, linestyle=:dash)
    vlines!(ax4, [μ_selected], color=:red, linewidth=2)
    fig
end


# ╔═╡ e0279819-5547-4536-80e7-f32b80ef8897
md"""
### Selected solution

| quantity | value |
|:--|--:|
| method | **$(method)** |
| penalty | **$(penalty)** |
| ``\mu`` | **$(round(μ_selected; sigdigits=5))** |
| residual norm | **$(round(sqrt(selected_R); sigdigits=5))** |
| ``\chi^2 = R(\mu)/R(0)`` | **$(round(result.chi2; sigdigits=5))** |
| discrepancy target ``\delta`` | **$(round(δ; sigdigits=5))** |

The red point and vertical line are the parameter returned by DECAES. For χ² and Morozov selection, the diagnostic crosses zero at the requested residual. Regińska selection crosses the log-slope balance condition. GCV minimizes its score. The L-curve method selects a positive-curvature corner of the parametric curve.
"""

# ╔═╡ Cell order:
# ╠═6f460dbd-1fe9-4d4d-bd24-1dd24a465e75
# ╟─b2d0ca29-568f-4b87-8f80-a10cf1b0dd31
# ╟─4581a4b9-b1cb-4916-959b-65baefa932ae
# ╟─57f4c9e0-1a2e-45ed-9dc5-66c81945ee3e
# ╟─d8b0c8b9-2382-42b4-949f-e97e91fa5804
# ╟─56aa6caf-c931-4622-bb83-39213d85d019
# ╟─f9fc152b-926f-451b-8bd6-65e58bc9a4b7
# ╟─89c75ca9-b622-4c51-b922-ab242771e141
# ╟─d66d4f7c-daa0-4f7c-951f-aefb15578226
# ╟─e0279819-5547-4536-80e7-f32b80ef8897
