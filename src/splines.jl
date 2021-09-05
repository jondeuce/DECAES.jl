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

struct NNLSDiscreteSurrogateSearch{D, T, TA <: AbstractArray{T}, TdA <: AbstractArray{T}, Tα <: AbstractArray{SVector{D,T}}, W}
    As::TA
    ∇As::TdA
    αs::Tα
    b::Vector{T}
    ∂αAx::Vector{T}
    Axb::Vector{T}
    nnls_work::W
end

function NNLSDiscreteSurrogateSearch(
        As::AbstractArray{T},
        ∇As::AbstractArray{T},
        αs::NTuple{D},
        b = zeros(T, size(As, 1)),
    ) where {D,T}
    @assert ndims(As) == 2 + D
    @assert ndims(∇As) == 3 + D # ∇As has extra dimension for parameter gradients
    @assert all(size(As)[1:2] .== size(∇As)[1:2]) # matrix dimensions must match
    @assert size(∇As)[3] == D # length of gradient dimension equals number of parameters
    @assert all(size(As)[3:end] .== size(∇As)[4:end] .== length.(αs)) # dimension size must match parameters lengths

    # size(As)  = (M, N, P1..., PD)
    # size(∇As) = (M, N, D, P1..., PD)
    # size(αs)  = (P1..., PD)
    M, N = size(As, 1), size(As, 2)
    αs   = meshgrid(SVector{D,T}, αs...)
    ∂αAx = zeros(T, M)
    Axb  = zeros(T, M)
    nnls_work = lsqnonneg_work(zeros(T, M, N), zeros(T, M))
    NNLSDiscreteSurrogateSearch(As, ∇As, αs, b, ∂αAx, Axb, nnls_work)
end

load!(p::NNLSDiscreteSurrogateSearch{D,T}, b::AbstractVector{T}) where {D,T} = copyto!(p.b, b)

function loss!(p::NNLSDiscreteSurrogateSearch{D,T}, I::CartesianIndex{D}) where {D,T}
    A = uview(p.As, :, :, I)
    b = p.b
    solve!(p.nnls_work, A, b)
    return chi2(p.nnls_work)
end

function ∇loss!(p::NNLSDiscreteSurrogateSearch{D,T}, I::CartesianIndex{D}) where {D,T}
    x = solution(p.nnls_work)
    pos = x .> 0
    x⁺ = x[pos]
    b = p.b
    return svector(D) do d
        dA = p.∇As[:, pos, d, I]
        A = p.As[:, pos, I]
        2 * dot(dA * x⁺, A * x⁺ - b)
    end
end

function surrogate_spline_opt(
        p::NNLSDiscreteSurrogateSearch{1,T};
        mineval::Int = 2,
        maxeval::Int = length(p.αs),
    ) where {T}

    f(i)    = loss!(p, CartesianIndex(i))
    X       = reinterpret(T, p.αs)
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
                iM = (iL + iR) ÷ 2
                if iM == iL || iM == iR
                    # interval cannot be reduced further
                    converged = true
                else
                    # insert and evaluate midpoint
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

function mock_surrogate_search_problem(
        o::T2mapOptions = mock_t2map_opts(; MatrixSize = (1,1,1));
        alphas = range(50, 180, length = o.nRefAngles),
        betas = range(50, 180, length = o.nRefAngles),
    )

    # Mock CPMG image
    b = vec(mock_image(o))
    nα, nβ = length(alphas), length(betas)
    As = zeros(o.nTE, o.nT2, nα, nβ)
    ∇As = zeros(o.nTE, o.nT2, 2, nα, nβ)
    T2s = logrange(o.T2Range..., o.nT2)
    θ = EPGOptions(o.nTE, o.SetFlipAngle, o.TE, 0.0, o.T1, o.SetRefConAngle)
    jfuncs = [EPGJacobianFunctor(θ, (:α, :β)) for _ in 1:Threads.nthreads()]

    _, Rαβ = SplitCartesianIndices(As, Val(2))
    @time Threads.@threads for Iαβ in Rαβ
        @inbounds for j in 1:o.nT2
            α, β = alphas[Iαβ[1]], betas[Iαβ[2]]
            j!   = jfuncs[Threads.threadid()]
            j!(uview(∇As, :, j, :, Iαβ), uview(As, :, j, Iαβ), EPGOptions(θ, α, θ.TE, T2s[j], θ.T1, β))
        end
    end

    return NNLSDiscreteSurrogateSearch(As, ∇As, (alphas, betas), b)
end
