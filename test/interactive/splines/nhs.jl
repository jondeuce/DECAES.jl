using .NormalHermiteSplines: NormalHermiteSplines, RK_H2

struct NormalHermiteSplineSurrogate{D, T, F, RK} <: AbstractSurrogate{D, T}
    fg::F
    grid::Array{SVector{D, T}, D}
    seen::Array{Bool, D}
    ugrid::Array{T, D}
    ∇ugrid::Array{SVector{D, T}, D}
    spl::NormalHermiteSplines.ElasticNormalSpline{D, T, RK}
end

function NormalHermiteSplineSurrogate(fg, grid::Array{SVector{D, T}, D}, kernel = RK_H2(one(T))) where {D, T}
    return NormalHermiteSplineSurrogate(
        fg,
        grid,
        fill(false, size(grid)),
        fill(T(NaN), size(grid)),
        fill(fill(T(NaN), SVector{D, T}), size(grid)),
        NormalHermiteSplines.ElasticNormalSpline(first(grid), last(grid), maximum(size(grid)), kernel),
    )
end

function update!(surr::NormalHermiteSplineSurrogate{D, T}, I::CartesianIndex{D}) where {D, T}
    @inbounds(surr.seen[I]) && return surr
    u, ∇u = surr.fg(I)
    @inbounds begin
        p = surr.grid[I]
        surr.seen[I] = true
        surr.ugrid[I] = u
        surr.∇ugrid[I] = ∇u
    end
    insert!(surr.spl, p, u)
    @inbounds for i in 1:D
        eᵢ = basisvector(SVector{D, T}, i)
        insert!(surr.spl, p, eᵢ, ∇u[i])
    end
    return surr
end

function Base.empty!(surr::NormalHermiteSplineSurrogate{D, T}) where {D, T}
    empty!(surr.spl)
    surr.seen .= false
    surr.ugrid .= T(NaN)
    surr.∇ugrid .= (fill(T(NaN), SVector{D, T}),)
    return surr
end

function suggest_point(surr::NormalHermiteSplineSurrogate{1, T}) where {T}
    @assert length(surr.grid) >= 2 "Grid must have at least two points"
    @inbounds for i in eachindex(surr.grid)
        if !surr.seen[i]
            surr.ugrid[i], surr.∇ugrid[i] = NormalHermiteSplines._evaluate_with_gradient(surr.spl, surr.grid[i])
        end
    end

    @inbounds p₀, u₀, ∇u₀, I = surr.grid[1], surr.ugrid[1], surr.∇ugrid[1], 1
    @inbounds for i in 2:length(surr.grid)
        pᵢ, uᵢ, ∇uᵢ = surr.grid[i], surr.ugrid[i], surr.∇ugrid[i]
        uᵢ < u₀ && ((p₀, u₀, ∇u₀, I) = (pᵢ, uᵢ, ∇uᵢ, i))
    end

    @inbounds if I == 1 || (I < length(surr.grid) && ∇u₀[1] < 0)
        p₁, p₂ = p₀, surr.grid[I+1]
    else
        p₁, p₂ = surr.grid[I-1], p₀
    end
    f = Base.Fix1(NormalHermiteSplines.evaluate, surr.spl)
    x, u = brent_minimize(f, p₁[1], p₂[1]; xrtol = T(1e-4), xatol = T(1e-4), maxiters = 10)
    return u₀ < u ? (p₀, u₀) : (SA{T}[x], u)
end

function NormalHermiteSplineSurrogate(prob::NNLSDiscreteSurrogateSearch{D, T}) where {D, T}
    return NormalHermiteSplineSurrogate(Base.Fix1(loss_with_grad!, prob), prob.αs, RK_H2(one(T)))
end

function nearest_gridpoint(grid::AbstractArray{SVector{D, T}, D}, x::SVector{D, T}) where {D, T}
    @inbounds xlo, xhi = first(grid), last(grid)
    @inbounds Ilo, Ihi = first(CartesianIndices(grid)), last(CartesianIndices(grid))
    lo, hi = SVector(Tuple(Ilo)), SVector(Tuple(Ihi))
    i = @. clamp(round(Int, (x - xlo) * (hi - lo) / (xhi - xlo) + lo), lo, hi)
    I = CartesianIndex(Tuple(i))
    return I, @inbounds(grid[I])
end
nearest_gridpoint(state::DiscreteSurrogateSearcher{D, T}, x::SVector{D, T}) where {D, T} = nearest_gridpoint(state.grid, x)

function nearest_interior_gridpoint(grid::AbstractArray{SVector{D, T}, D}, x::SVector{D, T}) where {D, T}
    R = CartesianIndices(grid)
    oneI = oneunit(first(R))
    Ilo, Ihi = first(R) + oneI, last(R) - oneI
    I, _ = nearest_gridpoint(@views(grid[Ilo:Ihi]), x)
    I += Ilo - oneI
    return I, @inbounds(grid[I])
end
nearest_interior_gridpoint(state::DiscreteSurrogateSearcher{D, T}, x::SVector{D, T}) where {D, T} = nearest_interior_gridpoint(state.grid, x)

function local_search(surr::NormalHermiteSplineSurrogate{D, T}, x₀::SVector{D, T}, state::Union{Nothing, DiscreteSurrogateSearcher{D, T}} = nothing; maxeval::Int = 100, xtol_rel = 1e-4, xtol_abs = 1e-4) where {D, T}
    if state !== nothing
        box = BoundingBox(size(surr.grid))
        for I in corners(box)
            update!(surr, state, I; maxeval)
        end
    end

    opt = NLopt.Opt(:LD_LBFGS, D)
    opt.lower_bounds = Float64[first(surr.grid)...]
    opt.upper_bounds = Float64[last(surr.grid)...]
    opt.xtol_rel = xtol_rel
    opt.xtol_abs = xtol_abs
    opt.maxeval = maxeval
    function objective(x, g)
        x⃗ = SVector{D, T}(x)
        if !isempty(g)
            g .= NormalHermiteSplines.evaluate_gradient(surr.spl, x⃗)
        end
        return NormalHermiteSplines.evaluate(surr.spl, x⃗)
    end
    opt.min_objective = objective
    minf, minx, _ = NLopt.optimize(opt, Vector{Float64}(x₀))
    return SVector{D, T}(minx), T(minf)
end

function spline_opt(spl::NormalHermiteSplines.NormalSpline{D, T}, prob::NNLSDiscreteSurrogateSearch{D, T}; alg = :LD_SLSQP) where {D, T}
    NormalHermiteSplines.evaluate!(prob.u, spl, prob.αs)
    α₀ = prob.αs[argmin(prob.u)]
    opt = NLopt.Opt(alg, D)
    opt.lower_bounds = Float64[prob.αs[begin]...]
    opt.upper_bounds = Float64[prob.αs[end]...]
    opt.xtol_rel = 0.001
    function objective(x, g)
        x⃗ = SVector{D, T}(x)
        if !isempty(g)
            u, ∇u = NormalHermiteSplines._evaluate_with_gradient(spl, x⃗)
            g .= ∇u
        else
            u = NormalHermiteSplines.evaluate(spl, x⃗)
        end
        return u
    end
    opt.min_objective = objective
    minf, minx, _ = NLopt.optimize(opt, Vector{Float64}(α₀))
    return SVector{D, T}(minx), T(minf)
end
