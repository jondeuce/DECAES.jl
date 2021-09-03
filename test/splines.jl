using Test
using DECAES
using StaticArrays
using BenchmarkTools

using NormalHermiteSplines
const nhs = NormalHermiteSplines

# using CairoMakie
# set_theme!(Theme(resolution = (600,450)))

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
    # nhs.evaluate_one(spl, xi)
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

function plot_splines()
    # f = x -> sin(x) * exp(-x^2)
    # df = x -> cos(x) * exp(-x^2) + sin(x) * (-2x) * exp(-x^2)
    f = x -> min(abs(x), 0.5)
    df = x -> ifelse(abs(x) < 0.5, sign(x), zero(x))

    x = sort(2 .* rand(10) .- 1)
    u = f.(x)
    du = df.(x)

    xp = collect(range(-1, 1, length = 100))
    p = plot(; legend = :bottomleft)
    plot!(p, xp, f.(xp), line = (2, :solid, :black), label = "f")
    scatter!(p, x, u, marker = (5, :red, :circle), label = "points")

    for use_grad in [true, false]
        minx, minf, spl, ret = DECAES.hermite_spline_opt(x, u, use_grad ? du : nothing)
        plot!(p, xp, nhs.evaluate(spl, xp), line = (3, use_grad ? :dash : :dot), label = "spline (grad = $use_grad)")
        scatter!(p, [minx], [minf], marker = (10, :diamond), label = "minimum (grad = $use_grad)")
    end

    display(p)
end

function benchmark_spline()
    x = sort(randn(10))
    u = randn(10)
    du = randn(10)
    xi = Ref(mean(x))
    spl = spline(x, u)
    dspl = spline(x, u, du)
    @btime $(nhs.evaluate_one)($spl, $xi[])
    @btime $(nhs.evaluate_one)($dspl, $xi[])
    @btime $(nhs.evaluate_derivative)($spl, $xi[])
    @btime $(nhs.evaluate_derivative)($dspl, $xi[])
end

function test_mock_surrogate_search_problem(
        o::T2mapOptions = mock_t2map_opts(; MatrixSize = (1, 1, 1))
    )
    function fA(α, β)
        theta = EPGOptions{Float64,32}(α, o.TE, 0.0, o.T1, β)
        T2_times = logrange(o.T2Range..., o.nT2)
        epg_decay_basis(theta, T2_times)
    end

    p = mock_surrogate_search_problem(o)
    alphas, betas = p.αs
    nnls_work = lsqnonneg_work(zeros(o.nTE, o.nT2), zeros(o.nTE))

    function f!(work, α, β)
        A = fA(α, β)
        solve!(work, A, p.b)
        return chi2(work)
    end

    function ∇f_approx!(work, α, β; h)
        l = f!(work, α, β)
        lα⁺ = f!(work, α + h, β)
        lα⁻ = f!(work, α - h, β)
        lβ⁺ = f!(work, α, β + h)
        lβ⁻ = f!(work, α, β - h)
        dl_dα = (lα⁺ - lα⁻) / 2h
        dl_dβ = (lβ⁺ - lβ⁻) / 2h
        return l, dl_dα, dl_dβ
    end

    function ∇f_surrogate!(work, α, β)
        _, i = find_nearest(alphas, α)
        _, j = find_nearest(betas, β)
        @assert alphas[i] == α
        @assert betas[j] == β
        I = CartesianIndex(i, j)
        l = loss!(p, I)
        dl_dα, dl_dβ = ∇loss!(p, I)
        return l, dl_dα, dl_dβ
    end

    for α in alphas, β in betas
        # α = rand(alphas) # alphas[div(end,2)]
        # β = rand(betas) # betas[div(end,2)]
        l1, ∂α1, ∂β1 = ∇f_approx!(nnls_work, α, β; h = 1e-1)
        l2, ∂α2, ∂β2 = ∇f_approx!(nnls_work, α, β; h = 1e-2)
        l3, ∂α3, ∂β3 = ∇f_approx!(nnls_work, α, β; h = 1e-3)
        l , ∂α , ∂β  = ∇f_surrogate!(nnls_work, α, β)
        @test l1 == l2 == l3 == l
        @test abs(∂α3 - ∂α) < abs(∂α2 - ∂α) < abs(∂α1 - ∂α)
        @test abs(∂β3 - ∂β) < abs(∂β2 - ∂β) < abs(∂β1 - ∂β)
    end

    # function chi2_alpha_fun(flip_angles, i)
    # # First argument `flip_angles` has been used implicitly in creating `decay_basis_set` already
    # @timeit_debug TIMER() "lsqnonneg!" begin
    # solve!(nnls_work, decay_basis_set[i], decay_data)
    # return chi2(nnls_work)
    # end
    # end
end
