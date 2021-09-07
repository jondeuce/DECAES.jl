"""
Lightweight polynomial type
"""
struct Poly{T, A<:AbstractVector{T}}
    c::A
end
Poly(c::Number...) = Poly(c)
Poly(c::Tuple) = Poly([float.(c)...])
Poly(c::AbstractVector{Int}) = Poly(float.(c))

coeffs(p::Poly) = p.c
degree(p::Poly) = length(coeffs(p)) - 1
(p::Poly)(x) = evalpoly(x, coeffs(p))

add!(p::Poly, a::Number) = (p.c[1] += a; return p)
sub!(p::Poly, a::Number) = (p.c[1] -= a; return p)

# Recall: p(x) = Σᵢ cᵢ ⋅ xⁱ⁻¹
derivative(p::Poly{T}) where {T} = Poly(T[(i-1) * coeffs(p)[i] for i in 2:degree(p)+1]) # ∂/∂x p(x) = Σᵢ (i-1) ⋅ cᵢ ⋅ xⁱ⁻²
integral(p::Poly{T}) where {T} = Poly(T[i == 0 ? zero(T) : coeffs(p)[i] / i for i in 0:degree(p)+1]) # ∫₀ˣ p(x) = Σᵢ (cᵢ / i) ⋅ xⁱ
PolynomialRoots.roots(p::Poly) = PolynomialRoots.roots(coeffs(p))
Base.extrema(p::Poly) = PolynomialRoots.roots(derivative(p))

####
#### Spline utils
####

# These algorithms were changed compared to the original MATLAB version:
# instead of brute force searching through the splines, numerical methods
# are performed which are much more efficient and much more accurate.
# For direct comparison of results with the MATLAB version, the brute force
# version is implimented and can be used by setting the LEGACY flag.
const LEGACY = Ref(false)
spline_opt(args...; kwargs...)  = LEGACY[] ? _spline_opt_legacy_slow(args...; kwargs...)  : _spline_opt(args...; kwargs...)
spline_root(args...; kwargs...) = LEGACY[] ? _spline_root_legacy_slow(args...; kwargs...) : _spline_root(args...; kwargs...)

function _make_spline(X, Y; deg_spline = min(3, length(X)-1))
    # @assert length(X) == length(Y) && length(X) > 1
    spl = Dierckx.Spline1D(X, Y; k = deg_spline, bc = "extrapolate")
end

function _build_polynomials(spl)
    k = spl.k
    t = Dierckx.get_knots(spl)[1:end-1]
    coeffs = zeros(k+1, length(t))
    coeffs[1, :] .= spl.(t)
    for m in 1:k
        coeffs[m+1, :] .= Dierckx.derivative(spl, t, m) ./ factorial(m)
    end
    return [Poly(coeffs[:, j]) for j in 1:size(coeffs, 2)]
end

# Global minimization through fitting a spline to data (X, Y)
function _spline_opt(spl::Dierckx.Spline1D)
    knots = Dierckx.get_knots(spl)
    polys = _build_polynomials(spl)
    x, y = knots[1], polys[1](0) # initial lefthand point
    @inbounds for (i,p) in enumerate(polys)
        x₀, x₁ = knots[i], knots[i+1] # spline section endpoints
        _x, _y = x₁, p(x₁ - x₀) # check right endpoint
        (_y < y) && (x = _x; y = _y)
        for rᵢ in extrema(p) # extrema(p) returns the zeros of derivative(p)
            if imag(rᵢ) ≈ 0 # real roots only
                xᵢ = x₀ + real(rᵢ)
                if x₀ <= xᵢ <= x₁ # filter roots within range
                    _x, _y = xᵢ, p(real(rᵢ))
                    (_y < y) && (x = _x; y = _y)
                end
            end
        end
    end
    return (; x, y)
end
_spline_opt(X::AbstractVector, Y::AbstractVector; deg_spline = min(3, length(X)-1)) = _spline_opt(_make_spline(X, Y; deg_spline))

# MATLAB spline optimization performs global optimization by sampling the spline
# fit to data (X, Y) at points X[1]:0.001:X[end], and uses the minimum value.
# This isn't very efficient; instead we use exact spline optimization and return
# the (x,y) pair such that x ∈ X[1]:0.001:X[end] is nearest to the exact optimum
function _spline_opt_legacy(spl::Dierckx.Spline1D)
    xopt, yopt = _spline_opt(spl)
    knots = Dierckx.get_knots(spl)
    xs = knots[1]:eltype(knots)(0.001):knots[end] # from MATLAB version
    _, i0 = find_nearest(xs, xopt) # find nearest x in xs to xopt

    # Note that the above finds the x value nearest the true minimizer, but we need the x value corresponding to
    # the y value nearest the true minimum. Since we are near the minimum, search for a local minimum.
    @unpack x, y, i = local_gridsearch(spl, xs, i0)

    return (; x, y)
end
_spline_opt_legacy(X::AbstractVector, Y::AbstractVector; deg_spline = min(3, length(X)-1)) = _spline_opt_legacy(_make_spline(X, Y; deg_spline))

# Similar to above, but removes the extra trick and instead performs
# exactly what the MATLAB implementation does
function _spline_opt_legacy_slow(spl::Dierckx.Spline1D)
    knots = Dierckx.get_knots(spl)
    xs = knots[1]:eltype(knots)(0.001):knots[end] # from MATLAB version
    x, y = xs[1], spl(xs[1])
    for (i,xᵢ) in enumerate(xs)
        (i == 1) && continue
        yᵢ = spl(xᵢ)
        (yᵢ < y) && ((x, y) = (xᵢ, yᵢ))
    end
    return (; x, y)
end
_spline_opt_legacy_slow(X::AbstractVector, Y::AbstractVector; deg_spline = min(3, length(X)-1)) = _spline_opt_legacy_slow(_make_spline(X, Y; deg_spline))

# Root finding through fitting a spline to data (X, Y)
function _spline_root(spl::Dierckx.Spline1D, value::Number = 0)
    knots = Dierckx.get_knots(spl)
    polys = _build_polynomials(spl)
    x = eltype(knots)(NaN)
    @inbounds for (i,p) in enumerate(polys)
        x₀, x₁ = knots[i], knots[i+1] # spline section endpoints
        # Solve `p(rᵢ) = value` via `p(rᵢ) - value = 0`
        for rᵢ in PolynomialRoots.roots(sub!(p, value))
            if imag(rᵢ) ≈ 0 # real roots only
                xᵢ = x₀ + real(rᵢ)
                if x₀ <= xᵢ <= x₁ # filter roots within range
                    x = isnan(x) ? xᵢ : min(x, xᵢ)
                end
            end
        end
    end
    return x
end
_spline_root(X::AbstractVector, Y::AbstractVector, value::Number = 0; deg_spline = min(3, length(X)-1)) = _spline_root(_make_spline(X, Y; deg_spline), value)

# Brute force root finding through fitting a spline to data (X, Y):
# MATLAB implementation of spline root finding performs root finding by sampling the
# spline fit to data (X, Y) at points X[1]:0.001:X[end], and uses the nearest value.
# This isn't very efficient; instead we use exact spline root finding and return
# the nearest x ∈ X[1]:0.001:X[end] such that the y value is nearest zero
function _spline_root_legacy(spl::Dierckx.Spline1D, value = 0)
    # Find x value nearest to the root
    knots = Dierckx.get_knots(spl)
    xs = knots[1]:eltype(knots)(0.001):knots[end] # from MATLAB version
    xroot = _spline_root(spl, value)
    _, i0 = find_nearest(xs, xroot) # find nearest x in xs to xroot

    # Note that the above finds the x value nearest the true root, but we need the x value corresponding to the
    # y value nearest to `value`. Since we are near the root, search for a local minimum in abs(spl(x)-value).
    dy(x) = abs(spl(x) - value)
    @unpack x, y, i = local_gridsearch(dy, xs, i0)

    return x
end
_spline_root_legacy(X::AbstractVector, Y::AbstractVector, value = 0; deg_spline = min(3, length(X)-1)) = _spline_root_legacy(_make_spline(X, Y; deg_spline), value)

# Similar to above, but removes the extra trick and instead performs
# exactly what the MATLAB implementation does
function _spline_root_legacy_slow(spl::Dierckx.Spline1D, value = 0)
    knots = Dierckx.get_knots(spl)
    xs = knots[1]:eltype(knots)(0.001):knots[end] # from MATLAB version
    x, y = xs[1], abs(spl(xs[1]) - value)
    for (i,xᵢ) in enumerate(xs)
        (i == 1) && continue
        yᵢ = abs(spl(xᵢ) - value)
        (yᵢ < y) && ((x, y) = (xᵢ, yᵢ))
    end
    return x
end
_spline_root_legacy_slow(X::AbstractVector, Y::AbstractVector, value = 0; deg_spline = min(3, length(X)-1)) = _spline_root_legacy_slow(_make_spline(X, Y; deg_spline), value)

####
#### Global optimization using surrogate splines
####

struct NNLSDiscreteSurrogateSearch{D, T, TA <: AbstractArray{T}, TdA <: AbstractArray{T}, Tb <: AbstractVector{T}, W}
    As::TA
    ∇As::TdA
    αs::Array{SVector{D,T}, D}
    b::Tb
    ℓ::Array{T, D}
    ∂Ax⁺::Vector{T}
    Ax⁺b::Vector{T}
    nnls_work::W
end

function NNLSDiscreteSurrogateSearch(
        As::AbstractArray{T},   # size(As)  = (M, N, P1..., PD)
        ∇As::AbstractArray{T},  # size(∇As) = (M, N, D, P1..., PD)
        αs::NTuple{D},          # size(αs)  = (P1..., PD)
        b::AbstractVector{T},   # size(b)   = (M,)
    ) where {D,T}
    M, N = size(As, 1), size(As, 2)
    @assert ndims(As) == 2 + D && ndims(∇As) == 3 + D # ∇As has extra dimension for parameter gradients
    @assert size(∇As)[1:3] == (M, N, D) # matrix dimensions must match, and gradient dimension must equal number of parameters
    @assert size(As)[3:end] == size(∇As)[4:end] == length.(αs) # dimension size must match parameters lengths
    @assert size(b) == (M,)

    αs   = meshgrid(SVector{D,T}, αs...)
    ℓ    = zeros(T, size(αs))
    ∂Ax⁺ = zeros(T, M)
    Ax⁺b = zeros(T, M)
    nnls_work = lsqnonneg_work(zeros(T, M, N), zeros(T, M))
    NNLSDiscreteSurrogateSearch(As, ∇As, αs, b, ℓ, ∂Ax⁺, Ax⁺b, nnls_work)
end

load!(p::NNLSDiscreteSurrogateSearch{D,T}, b::AbstractVector{T}) where {D,T} = copyto!(p.b, b)

function loss!(p::NNLSDiscreteSurrogateSearch{D,T}, I::CartesianIndex{D}) where {D,T}
    @unpack As, b, nnls_work = p
    solve!(nnls_work, uview(As, :, :, I), b)
    return chi2(nnls_work)
end

function ∇loss!(p::NNLSDiscreteSurrogateSearch{D,T}, I::CartesianIndex{D}) where {D,T}
    @unpack As, ∇As, b, ∂Ax⁺, Ax⁺b, nnls_work = p
    x = solution(nnls_work)
    @inbounds @. Ax⁺b = -b
    @inbounds for j in 1:size(As, 2)
        (x[j] > 0) && axpy!(x[j], uview(As, :, j, I), Ax⁺b)
    end
    return svector(D) do d
        @inbounds ∂Ax⁺ .= zero(T)
        @inbounds for j in 1:size(∇As, 2)
            (x[j] > 0) && axpy!(x[j], uview(∇As, :, j, d, I), ∂Ax⁺)
        end
        2 * dot(∂Ax⁺, Ax⁺b)
    end
end

abstract type AbstractOracle end
struct CubicSplineOracle <: AbstractOracle end
struct HermiteSplineOracle <: AbstractOracle end

struct DiscreteBisectionSearch{D, T}
    xgrid::Array{SVector{D,T}, D}
    I::Vector{CartesianIndex{D}}
    x::Vector{SVector{D,T}}
    ℓ::Vector{T}
end

function bisection_search(
        f,
        state::DiscreteBisectionSearch{D, T},
        oracle::AbstractOracle = CubicSplineOracle(),
    ) where {D,T}

    while true
        # get new suggestion from oracle, and find nearest neighbours containing the suggestion
        x, y = suggest_point(state, oracle)
        IL, IR = bounding_neighbours(state, x)
        if IR - IL <= one(IL)
            # `x` is in bounding box of `xL = state.xgrid[IL]` and `xR = state.xgrid[IR]`
            return (; x, y)
        else
            # evaluate `f` at midpoint of bounding box of `x`
            IM = CartesianIndex(Tuple(IL + IR) .÷ 2)
            xM = state.xgrid[IM]
            ℓM = f(xM)
            push!(state.I, IM)
            push!(state.x, xM)
            push!(state.ℓ, ℓM)
        end
    end
end

function suggest_point(state::DiscreteBisectionSearch{1,T}, ::CubicSplineOracle) where {T}
    p = sortperm(state.x; by = first)
    @unpack x, y = spline_opt(first.(state.x[p]), state.ℓ[p])
    return (x = SA{T}[x], y = T(y))
end

# Update observed evaluations, returning true if converged
function bounding_neighbours(
        state::DiscreteBisectionSearch{D,T},
        x::SVector{D,T},
    ) where {D,T}

    # # NOTE: if `state.xgrid[state.I]` does not contain any pairs of points
    # # which bound `x` then `IL` and `IR` will default to garbage values
    # IL = IR = one(CartesianIndex{D})
    # d²max = T(Inf)
    # for j in 1:length(state.I)
    #     for i in 1:length(state.I)
    #         i == j && continue
    #         Ii, Ij = state.I[i], state.I[j]
    #         xi, xj = state.xgrid[Ii], state.xgrid[Ij]
    #         d² = sum(abs2.(xi - xj))
    #         if all(xi .<= x .<= xj) && d² <= dmax
    #             IL, IR, d²max = Ii, Ij, d²
    #         end
    #     end
    # end
    # return IL, IR

    # NOTE: if `x` is outside of the bounding box of `state.xgrid[state.I]`,
    # `IL` and `IR` will lie on the boundary of `R`
    R = CartesianIndices(state.xgrid)
    IL, IR = foldl(state.I; init = (first(R), last(R))) do (IL, IR), I
        IL = ifelse.(Tuple(state.xgrid[I] .< x), Tuple(max(I, IL)), Tuple(IL))
        IR = ifelse.(Tuple(state.xgrid[I] .> x), Tuple(min(I, IR)), Tuple(IR))
        return CartesianIndex(IL), CartesianIndex(IR)
    end
end

function surrogate_spline_opt(
        f, X::AbstractVector{T};
        mineval::Int = 2,
        maxeval::Int = length(X),
    ) where {T}

    mineval = max(mineval, 2)
    maxeval = min(maxeval, length(X))

    # Check if X has less than mineval points
    if 2 <= length(X) <= mineval
        Y = T[f(i) for i in 1:length(X)]
        return spline_opt(X, Y)
    end

    # Update observed evaluations, returning true if converged
    function update_state!(Is, Xi, Yi, x, y)
        converged = false
        for i in 1:length(Is)-1
            # Find interval containing x
            if Xi[i] <= x <= Xi[i+1]
                iL, iR = Is[i], Is[i+1]
                if iL + 1 == iR
                    # interval cannot be reduced further
                    converged = true
                else
                    # insert and evaluate midpoint
                    iM = (iL + iR) ÷ 2
                    insert!(Is, i+1, iM)
                    insert!(Xi, i+1, X[iM])
                    insert!(Yi, i+1, f(iM))
                end
                break
            end
        end
        return converged
    end

    # Initialize state
    Is = round.(Int, range(1, length(X), length = mineval))
    Xi = X[Is]
    Yi = T[f(i) for i in Is]

    # Surrogate minimization
    while true
        x, y = spline_opt(Xi, Yi) # current global minimum estimate
        if update_state!(Is, Xi, Yi, x, y) # update and check for convergence
            return (; x, y)
        end
    end
end

function surrogate_spline_opt(
        p::NNLSDiscreteSurrogateSearch{1,T};
        mineval::Int = 2,
        maxeval::Int = length(p.αs),
    ) where {T}
    f(i) = loss!(p, CartesianIndex(i))
    X = reinterpret(T, p.αs)
    surrogate_spline_opt(f, X; mineval, maxeval)
end

function mock_surrogate_search_problem(
        ::Val{D} = Val(2),
        ::Val{ETL} = Val(32);
        opts::T2mapOptions = mock_t2map_opts(; MatrixSize = (1,1,1), nTE = ETL),
        alphas = range(50, 180, length = opts.nRefAngles),
        betas = range(50, 180, length = opts.nRefAngles),
    ) where {D,ETL}

    # Mock CPMG image
    @assert opts.nTE == ETL
    b = vec(mock_image(opts))
    opt_vars = D == 1 ? (:α,) : (:α, :β)
    opt_ranges = D == 1 ? (alphas,) : (alphas, betas)
    As = zeros(ETL, opts.nT2, length.(opt_ranges)...)
    ∇As = zeros(ETL, opts.nT2, D, length.(opt_ranges)...)
    T2s = logrange(opts.T2Range..., opts.nT2)
    θ = EPGOptions(ETL, opts.SetFlipAngle, opts.TE, 0.0, opts.T1, opts.SetRefConAngle)
    j! = EPGJacobianFunctor(θ, opt_vars)

    _, Rαs = SplitCartesianIndices(As, Val(2))
    for Iαs in Rαs
        @inbounds for j in 1:opts.nT2
            θαs = D == 1 ?
                EPGOptions(θ, (T2 = T2s[j], α = alphas[Iαs[1]],)) :
                EPGOptions(θ, (T2 = T2s[j], α = alphas[Iαs[1]], β = alphas[Iαs[2]]))
            j!(uview(∇As, :, j, :, Iαs), uview(As, :, j, Iαs), θαs)
        end
    end

    return NNLSDiscreteSurrogateSearch(As, ∇As, opt_ranges, b)
end
