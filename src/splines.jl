####
#### Polynomial utilities
####

#### Lightweight polynomial type

struct Poly{T <: AbstractFloat, A <: AbstractVector{T}}
    c::A
end
Poly(c::Number...) = Poly(c)
Poly(c::Tuple) = Poly(SVector(promote(c...)))
Poly(c::NTuple{N, Int}) where {N} = Poly(SVector(float.(c)))
Poly(c::AbstractVector{Int}) = Poly(float.(c))

coeffs(p::Poly) = p.c
degree(p::Poly) = length(coeffs(p)) - 1
(p::Poly)(x) = evalpoly(x, coeffs(p))

add!(p::Poly, a::Number) = (p.c[1] += a; return p)
sub!(p::Poly, a::Number) = (p.c[1] -= a; return p)

# Recall: p(x) = Σᵢ cᵢ ⋅ xⁱ⁻¹
Base.adjoint(p::Poly) = Poly(deriv_coeffs(coeffs(p))) # ∂/∂x p(x) = Σᵢ (i-1) ⋅ cᵢ ⋅ xⁱ⁻²
Base.cumsum(p::Poly) = Poly(integral_coeffs(coeffs(p))) # ∫₀ˣ p(x) = Σᵢ (cᵢ / i) ⋅ xⁱ
roots(p::Poly) = roots(coeffs(p))
Base.extrema(p::Poly) = roots(p')

deriv_coeffs(c::AbstractVector{T}) where {T} = T[i * c[i+1] for i in 1:length(c)-1]
deriv_coeffs(c::SVector{N, T}) where {N, T} = SVector{N - 1, typeof(one(T) * 1)}(ntuple(i -> i * c[i+1], Val(N - 1)))
deriv_coeffs(c::SVector{0, T}) where {T} = c
deriv_coeffs(c::Tuple) = Tuple(deriv_coeffs(SVector(c)))

integral_coeffs(c::AbstractVector{T}) where {T} = [i == 0 ? zero(T) / 1 : c[i] / i for i in 0:length(c)]
integral_coeffs(c::SVector{N, T}) where {N, T} = SVector{N + 1, typeof(one(T) / 1)}(ntuple(i -> i == 1 ? zero(T) / 1 : c[i-1] / (i - 1), Val(N + 1)))
integral_coeffs(c::Tuple) = Tuple(integral_coeffs(SVector(c)))

function roots(c::AbstractVector{T}) where {T <: AbstractFloat}
    deg = length(c) - 1
    @assert 0 <= deg <= 3 "Degree of polynomial must be 0, 1, 2, or 3"
    return deg == 0 ? T[] :
           deg == 1 ? T[root_real_linear(@inbounds((c[1], c[2])))] :
           deg == 2 ? T[roots_real_quadratic(@inbounds((c[1], c[2], c[3])))...] :
           T[roots_real_cubic(@inbounds((c[1], c[2], c[3], c[4])))...]
end
roots(::SVector{1, T}) where {T <: AbstractFloat} = SVector{0, T}()
roots(c::SVector{2, T}) where {T <: AbstractFloat} = SVector{1, T}(root_real_linear(Tuple(c)))
roots(c::SVector{3, T}) where {T <: AbstractFloat} = SVector{2, T}(roots_real_quadratic(Tuple(c)))
roots(c::SVector{4, T}) where {T <: AbstractFloat} = SVector{3, T}(roots_real_cubic(Tuple(c)))
roots(::SVector{5, T}) where {T <: AbstractFloat} = error("Degree of polynomial must be 0, 1, 2, or 3")

#### Cubic Hermite interpolator

struct CubicHermiteInterpolator{T <: AbstractFloat}
    u0::T
    u1::T
    m0::T
    m1::T
    dom::NTuple{2, T}
    coeffs::NTuple{4, T}
end

function CubicHermiteInterpolator(a::T, b::T, u0::T, u1::T, m0::T, m1::T) where {T <: AbstractFloat}
    r = (b - a) / 2
    m0, m1 = r * m0, r * m1
    Δu, Δm = u1 - u0, m1 - m0
    Σu, Σm = u1 + u0, m1 + m0
    coeffs = (Σu / 2 - Δm / 4, (3 * Δu - Σm) / 4, Δm / 4, (Σm - Δu) / 4)
    return CubicHermiteInterpolator(u0, u1, m0, m1, (a, b), coeffs)
end
CubicHermiteInterpolator(a, b, u0, u1, m0, m1) = CubicHermiteInterpolator(promote(map(float, (a, b, u0, u1, m0, m1))...)...)

@inline (spl::CubicHermiteInterpolator)(x) = evalpoly(tocanonical(x, spl.dom), spl.coeffs)

@inline function tocanonical(x, (a, b))
    c, r = (a + b) / 2, (b - a) / 2
    t = (x - c) / r
    return t
end

@inline function todomain(t, (a, b))
    c, r = (a + b) / 2, (b - a) / 2
    x = muladd(r, t, c)
    return x
end

function minimize(spl::CubicHermiteInterpolator{T}) where {T}
    (; u0, u1, m0, m1, dom, coeffs) = spl
    xend, uend = u0 < u1 ? (dom[1], u0) : (dom[2], u1)

    # See: https://github.com/ZJU-FAST-Lab/LBFGS-Lite/blob/35450d6256aad2e1c137ec955adfdc90710da80b/include/lbfgs.hpp#L391
    Δ = 3 * (u0 - u1)
    θ = Δ / 2 + (m0 + m1)
    γ = θ^2 - m0 * m1
    γ = γ > 0 ? -√γ : zero(T)
    p = -(Δ + (m0 + m1))
    q = 2 * γ + (m0 - m1)
    if abs(p) < abs(q)
        t = p / q
        y = evalpoly(t, coeffs)
        if y < uend
            return todomain(t, dom), y
        else
            return xend, uend
        end
    else
        return xend, uend
    end
end

function signedroots(spl::CubicHermiteInterpolator{T}, atol::T = zero(T)) where {T}
    (; dom, coeffs) = spl
    (t1, t2, t3), (s1, s2, s3) = signed_roots_real_cubic(coeffs)
    xlo, xhi = dom[1] + atol, dom[2] - atol
    x1, s1 = !isnan(t1) ? (x = todomain(t1, dom); xlo <= x <= xhi ? (x, s1) : (T(NaN), T(NaN))) : (T(NaN), T(NaN))
    x2, s2 = !isnan(t2) ? (x = todomain(t2, dom); xlo <= x <= xhi ? (x, s2) : (T(NaN), T(NaN))) : (T(NaN), T(NaN))
    x3, s3 = !isnan(t3) ? (x = todomain(t3, dom); xlo <= x <= xhi ? (x, s3) : (T(NaN), T(NaN))) : (T(NaN), T(NaN))
    (x1, s1), (x2, s2), (x3, s3) = TupleTools.sort(((x1, s1), (x2, s2), (x3, s3)); by = first, lt = lt_nan) # sort, treating NaN's as Inf
    return (x1, x2, x3), (s1, s2, s3)
end
roots(spl::CubicHermiteInterpolator) = signedroots(spl)[1]

#### Minimizing polynomials

function minimize_linear(coeffs::NTuple{2, T}, a::T, b::T, ua::T = evalpoly(a, coeffs), ub::T = evalpoly(b, coeffs)) where {T}
    # Minimize linear polynomial f(x) = c₁*x + c₀ over the interval [a, b].
    return ua < ub ? (a, ua) : (b, ub) # linear function attains minimum at one of the endpoints
end

function minimize_quadratic(coeffs::NTuple{3, T}, a::T, b::T, ua::T = evalpoly(a, coeffs), ub::T = evalpoly(b, coeffs)) where {T}
    # Minimize quadratic polynomial f(x) = c₂*x² + c₁*x + c₀ over the interval [a, b].
    xend, uend = ua < ub ? (a, ua) : (b, ub) # endpoint minimum
    if coeffs[3] == 0
        return xend, uend
    else
        x = -coeffs[2] / (2 * coeffs[3]) # vertex of the parabola: x = -c₁ / 2c₂
        if a < x < b
            u = evalpoly(x, coeffs)
            return u < uend ? (x, u) : (xend, uend)
        else
            return xend, uend
        end
    end
end

function extremize_cubic(coeffs::NTuple{4, T}, a::T, b::T, ua::T = evalpoly(a, coeffs), ub::T = evalpoly(b, coeffs)) where {T}
    # Extremize cubic polynomial f(x) = c₃*x³ + c₂*x² + c₁*x + c₀ over the interval [a, b].
    (xlo, ulo), (xhi, uhi) = ua < ub ? ((a, ua), (b, ub)) : ((b, ub), (a, ua)) # sort endpoint values
    x1, x2 = roots_real_quadratic(deriv_coeffs(coeffs)) # roots of the derivative

    if isnan(x1) || isnan(x2) && (x1 <= a || x1 >= b) || !isnan(x2) && (x1 >= b || x2 <= a || (x1 <= a && x2 >= b) || x1 == x2)
        # No real local extrema in the interval; return endpoint extrema
        return (xlo, ulo), (xhi, uhi)
    elseif coeffs[4] == 0
        # Spline is quadratic; check sign of quadratic coefficient
        x = x1 # note: x2 is NaN, x1 is NaN if coeffs[3] == 0 i.e. spline is linear
        if !isnan(x) && a < x < b
            u = evalpoly(x, coeffs)
            if coeffs[3] > 0
                u < ulo && ((xlo, ulo) = (x, u))
            else
                u > uhi && ((xhi, uhi) = (x, u))
            end
        end
        return (xlo, ulo), (xhi, uhi)
    else
        # Two real roots; local minimum corresponds to the larger (smaller) root when the cubic coefficient is positive (negative)
        if coeffs[4] > 0
            x1, x2 = x2, x1 # x1 corresponds to local min/x2 to local max
        end
        if a < x1 < b
            u1 = evalpoly(x1, coeffs)
            u1 < ulo && ((xlo, ulo) = (x1, u1))
        end
        if a < x2 < b
            u2 = evalpoly(x2, coeffs)
            u2 > uhi && ((xhi, uhi) = (x2, u2))
        end
        return (xlo, ulo), (xhi, uhi)
    end
end

minimize_cubic(coeffs::NTuple{4, T}, a::T, b::T, args...) where {T} = extremize_cubic(coeffs, a, b, args...)[1]
maximize_cubic(coeffs::NTuple{4, T}, a::T, b::T, args...) where {T} = extremize_cubic(coeffs, a, b, args...)[2]

#### Real roots of polynomials

function root_real_linear(coeffs::NTuple{2, T}) where {T <: AbstractFloat}
    # Root of linear equation a*x + b = 0. Returns NaN if a = 0.
    b, a = coeffs
    return a == 0 ? T(NaN) : -b / a
end

function root_real_linear(a::T, b::T, ua::T, ub::T, value::T = zero(T)) where {T <: AbstractFloat}
    # Root of linear equation `f(x) = c₁*x + c₀ = value` where `c₁, c₀` are defined implicitly by `f(a) = ua` and `f(b) = ub`.
    min(ua, ub) <= value <= max(ua, ub) || return T(NaN)
    ua == ub && return T(NaN) # degenerate linear function `f(x) = c₀`
    x = ua == value ? a :
        ub == value ? b :
        clamp(a + (b - a) * ((value - ua) / (ub - ua)), a, b)
    return x
end

function signed_roots_real_quadratic(coeffs::NTuple{3, T}) where {T <: AbstractFloat}
    # Robust solution to the quadratic equation a*x² + b*x + c = 0.
    # Coefficients are given in increasing order: (c, b, a).
    # Returns one of:
    #   1) Tuple of sorted real roots (x1, x2),
    #   2) Tuple (NaN, NaN) if no real roots exist or if quadratic degenerates to constant (a = b = 0),
    #   3) Tuple (x1, NaN) if quadratic degenerates to linear (a = 0),
    # See: https://math.stackexchange.com/a/2007723
    c, b, a = coeffs
    if a == 0
        if b == 0
            # Constant: c = 0
            return (T(NaN), T(NaN)), (T(NaN), T(NaN))
        else
            # Linear: bx + c = 0
            x = -c / b
            return (x, T(NaN)), (strictsign(b), T(NaN))
        end
    elseif c == 0
        # Factor out x: x * (ax + b) = 0
        x1, x2 = minmax(zero(T), -b / a)
        s1, s2 = b == 0 ? (zero(T), zero(T)) : (-strictsign(a), strictsign(a))
        return (x1, x2), (s1, s2)
    end

    Δ = b^2 - 4 * a * c
    if Δ < 0
        # No real roots
        return (T(NaN), T(NaN)), (T(NaN), T(NaN))
    elseif Δ == 0
        # One repeated real root: x = -b / 2a
        x = -b / 2a
        return (x, x), (zero(T), zero(T))
    else
        # Two real roots
        x1 = -(b + strictsign(b) * √Δ) / 2
        if x1 == 0
            x2 = x1 # Note: should never occur? since a, c != 0 and b² > 4ac
            return minmax(x1, x2), (zero(T), zero(T))
        else
            x2 = c / x1 # Viète's formulas
            x1 = x1 / a
            return minmax(x1, x2), (-strictsign(a), strictsign(a))
        end
    end
end
signed_roots_real_quadratic(coeffs::Tuple) = (@assert length(coeffs) == 3; return signed_roots_real_quadratic(promote(map(float, coeffs)...)))
roots_real_quadratic(coeffs::Tuple) = signed_roots_real_quadratic(coeffs)[1]

function signed_roots_real_cubic(coeffs::NTuple{4, T}) where {T <: AbstractFloat}
    # Return real roots of the cubic polynomial c₃*x³ + c₂*x² + c₁*x + c₀ = 0.
    # Coefficients are given in increasing order: (c₀, c₁, c₂, c₃).
    # Returns Tuple (x₁, x₂, x₃) such that if the cubic has `0 <= r <= 3` real roots,
    # then (x₁, ..., xᵣ) are sorted real roots and (xᵣ₊₁, ..., x₃) are NaN.
    if coeffs[4] == 0
        # Spline is quadratic; third root is at infinity
        (x1, x2), (s1, s2) = signed_roots_real_quadratic(coeffs[1:3])
        return (x1, x2, T(NaN)), (s1, s2, T(NaN))
    end
    if coeffs[1] == 0
        # Factor out x: x * (c₃*x² + c₂*x + c₁) = 0
        (x1, x2), (s1, s2) = signed_roots_real_quadratic(coeffs[2:4])
        if isnan(x1)
            return (zero(T), T(NaN), T(NaN)), (strictsign(coeffs[4]), T(NaN), T(NaN))
        else
            # Note: c₃ != 0, and therefore x1, x2 are both real or both NaN
            x1, x2, x3 = TupleTools.sort((zero(T), x1, x2))
            s1, s2, s3 = x1 == x2 ? (zero(T), zero(T), one(T)) : x2 == x3 ? (one(T), zero(T), zero(T)) : (one(T), -one(T), one(T))
            coeffs[4] < 0 && ((s1, s2, s3) = (-s1, -s2, -s3))
            return (x1, x2, x3), (s1, s2, s3)
        end
    end
    # Reduced cubic: x³ + a*x² + b*x + c = 0
    a = coeffs[3] / coeffs[4]
    b = coeffs[2] / coeffs[4]
    c = coeffs[1] / coeffs[4]
    a² = a^2
    R = (a * (2 * a² - 9 * b) + 27 * c) / 54
    Q = (a² - 3 * b) / 9
    R², Q³ = R^2, Q^3
    if R² < Q³
        # Three real roots
        tmpsqrtQ = -6 * √Q
        thirdθ = acos(R / √Q³) / 3 #TODO: more accurate than `atan(√(Q³ - R²), R) / 3`?
        twothirdsπ = 2 * T(π) / 3
        x1 = muladd(tmpsqrtQ, cos(thirdθ), -a) / 3
        x2 = muladd(tmpsqrtQ, cos(thirdθ + twothirdsπ), -a) / 3
        x3 = muladd(tmpsqrtQ, cos(thirdθ - twothirdsπ), -a) / 3
        x1, x2, x3 = TupleTools.sort((x1, x2, x3))
        s1, s2, s3 = x1 == x2 ? (zero(T), zero(T), one(T)) : x2 == x3 ? (one(T), zero(T), zero(T)) : (one(T), -one(T), one(T))
        coeffs[4] < 0 && ((s1, s2, s3) = (-s1, -s2, -s3))
        return (x1, x2, x3), (s1, s2, s3)
    else
        # One real root, two complex roots
        A = -strictsign(R) * ∛(abs(R) + √(R² - Q³))
        B = A == 0 ? zero(T) : Q / A
        x1 = A + B - a / 3
        return (x1, T(NaN), T(NaN)), (strictsign(coeffs[4]), T(NaN), T(NaN))
    end
end
signed_roots_real_cubic(coeffs::Tuple) = (@assert length(coeffs) == 4; return signed_roots_real_cubic(promote(map(float, coeffs)...)))
roots_real_cubic(coeffs::Tuple) = signed_roots_real_cubic(coeffs)[1]

####
#### Spline utils
####

function make_spline(X::AbstractVector, Y::AbstractVector; deg_spline = min(3, length(X) - 1))
    @assert length(X) == length(Y) && length(X) > 1
    return Dierckx.Spline1D(X, Y; k = deg_spline, bc = "extrapolate")
end

function build_polynomials(spl::Dierckx.Spline1D, knots = Dierckx.get_knots(spl))
    k = spl.k
    t = knots[1:end-1]
    coeffs = zeros(k + 1, length(t))
    @inbounds coeffs[1, :] .= spl(t)
    @inbounds for m in 1:k
        coeffs[m+1, :] .= Dierckx.derivative(spl, t, m) ./ factorial(m)
    end
    return [Poly(@views coeffs[:, j]) for j in 1:size(coeffs, 2)]
end

function build_polynomial!(coeffs, spl::Dierckx.Spline1D, t)
    @assert length(coeffs) == spl.k + 1
    mfact = 1
    @inbounds coeffs[1] = spl(t)
    @inbounds for m in 1:spl.k
        mfact *= m
        coeffs[m+1] = Dierckx.derivative(spl, t, m) / mfact
    end
    return coeffs
end

# Fit a spline to data `(X, Y)` and minimize `spl(x)`
function spline_opt(X::AbstractVector, Y::AbstractVector; deg_spline = min(3, length(X) - 1))
    @assert length(X) == length(Y) "X and Y must have the same length"
    @assert length(X) > 1 "X and Y must have at least 2 elements"
    @assert 0 < deg_spline <= 3 "Degree of spline must be 1, 2, or 3"
    if deg_spline == 1
        # Linear spline achievies minimum at one of the nodes
        y, i = findmin(Y)
        return @inbounds (; x = X[i], y)
    end
    spl = make_spline(X, Y; deg_spline)
    knots = Dierckx.get_knots(spl)
    polys = build_polynomials(spl, knots)
    x, y = @inbounds knots[1], polys[1](0) # initial lefthand point
    atol = 100 * eps(eltype(knots)) # tolerance for real roots
    @inbounds for (i, p) in enumerate(polys)
        x₀, x₁ = knots[i], knots[i+1] # spline section endpoints
        _x, _y = x₁, p(x₁ - x₀) # check right endpoint
        (_y < y) && (x = _x; y = _y)
        for rᵢ in extrema(p) # extrema(p) returns the zeros of p'
            isnan(rᵢ) && break # real roots only
            xᵢ = x₀ + rᵢ
            if x₀ - atol <= xᵢ <= x₁ + atol # filter roots within range
                _x, _y = clamp(xᵢ, x₀, x₁), p(rᵢ)
                (_y < y) && (x = _x; y = _y)
            end
        end
    end
    return (; x, y)
end

# Fit a spline to data `(X, Y)` and solve `spl(x) = value`
function spline_root(X::AbstractVector, Y::AbstractVector, value::Number = 0; deg_spline = min(3, length(X) - 1))
    x = eltype(X)(NaN)
    if deg_spline == 1
        # Linear spline has at most one root in each section
        @inbounds for i in 1:length(X)-1
            x₀, y₀, x₁, y₁ = X[i], Y[i], X[i+1], Y[i+1]
            if y₀ <= value <= y₁ || y₁ <= value <= y₀
                x = y₀ == y₁ ? x₀ :
                    y₀ ≈ value ? x₀ :
                    y₁ ≈ value ? x₁ :
                    clamp(x₀ + (x₁ - x₀) * ((value - y₀) / (y₁ - y₀)), x₀, x₁)
                break
            end
        end
    else
        spl = make_spline(X, Y; deg_spline)
        knots = Dierckx.get_knots(spl)
        polys = build_polynomials(spl)
        atol = 100 * eps(eltype(knots)) # tolerance for real roots
        @inbounds for (i, p) in enumerate(polys)
            # Solve `p(rᵢ) = value` via `p(rᵢ) - value = 0`
            x₀, x₁ = knots[i], knots[i+1] # spline section endpoints
            for rᵢ in roots(sub!(p, value))
                isnan(rᵢ) && break # real roots only
                xᵢ = x₀ + rᵢ
                if x₀ - atol <= xᵢ <= x₁ + atol # filter roots within range
                    xᵢ = clamp(xᵢ, x₀, x₁)
                    x = isnan(x) ? xᵢ : min(x, xᵢ) # find the leftmost root
                end
            end
            !isnan(x) && break # found a root
        end
    end
    return x
end

####
#### Legacy spline utils
####
####    These algorithms were changed compared to the original MATLAB version:
####    instead of brute force searching through the splines, numerical methods
####    are performed which are much more efficient and much more accurate.
####    For direct comparison of results with the MATLAB version, the brute force
####    version is implimented and can be used by setting the `legacy` flag.
####

# MATLAB spline optimization performs global optimization by sampling the spline
# fit to data (X, Y) at points X[1]:0.001:X[end], and uses the minimum value.
function spline_opt_legacy(X::AbstractVector, Y::AbstractVector; deg_spline = min(3, length(X) - 1))
    spl = make_spline(X, Y; deg_spline)
    knots = Dierckx.get_knots(spl)
    xs = knots[1]:eltype(knots)(0.001):knots[end] # from MATLAB version
    x, y = xs[1], spl(xs[1])
    for (i, xᵢ) in enumerate(xs)
        (i == 1) && continue
        yᵢ = spl(xᵢ)
        (yᵢ < y) && ((x, y) = (xᵢ, yᵢ))
    end
    return (; x, y)
end

# Brute force root finding through fitting a spline to data (X, Y).
# MATLAB implementation of spline root finding performs root finding by sampling the
# spline fit to data (X, Y) at points X[1]:0.001:X[end], and uses the nearest value.
function spline_root_legacy(spl::Dierckx.Spline1D, value = 0)
    knots = Dierckx.get_knots(spl)
    xs = knots[1]:eltype(knots)(0.001):knots[end] # from MATLAB version
    x, y = xs[1], abs(spl(xs[1]) - value)
    for (i, xᵢ) in enumerate(xs)
        (i == 1) && continue
        yᵢ = abs(spl(xᵢ) - value)
        (yᵢ < y) && ((x, y) = (xᵢ, yᵢ))
    end
    return x
end
spline_root_legacy(X::AbstractVector, Y::AbstractVector, value = 0; deg_spline = min(3, length(X) - 1)) = spline_root_legacy(make_spline(X, Y; deg_spline), value)

####
#### Surrogate functions over discrete grids
####

abstract type AbstractSurrogate{D, T} end

gridwidths(surr::AbstractSurrogate) = Tuple(abs.(last(surr.grid) - first(surr.grid)))
gridspacings(surr::AbstractSurrogate) = gridwidths(surr) ./ size(surr.grid)

struct CubicSplineSurrogate{T, F} <: AbstractSurrogate{1, T}
    f::F
    grid::Vector{SVector{1, T}}
    p::Vector{SVector{1, T}}
    u::Vector{T}
    npts::Base.RefValue{Int}
    legacy::Bool
end

function CubicSplineSurrogate(f, grid::Vector{SVector{1, T}}; legacy = false) where {T}
    return CubicSplineSurrogate(f, grid, SVector{1, T}[], T[], Ref(0), legacy)
end

function update!(surr::CubicSplineSurrogate, I::CartesianIndex{1})
    p = surr.grid[I]
    pos = surr.npts[] + 1
    @inbounds for i in 1:surr.npts[]
        (p[1] <= surr.p[i][1]) && (pos = i; break)
    end
    u = surr.f(I)
    insertat!(surr.p, pos, p, surr.npts[] + 1)
    insertat!(surr.u, pos, u, surr.npts[] + 1)
    surr.npts[] += 1
    return surr
end

function insertat!(x::AbstractVector{T}, i, v::T, len = length(x)) where {T}
    if len > length(x)
        append!(x, similar(x, max(length(x), 1)))
    end
    last = v
    @inbounds for j in i:len
        x[j], last = last, x[j]
    end
    return x
end

function Base.empty!(surr::CubicSplineSurrogate)
    surr.npts[] = 0
    return surr
end

function suggest_point(surr::CubicSplineSurrogate{T}) where {T}
    npts = surr.npts[]
    ps = reinterpret(T, view(surr.p, 1:npts))
    us = view(surr.u, 1:npts)
    p, u = surr.legacy ? spline_opt_legacy(ps, us) : spline_opt(ps, us)
    return SVector{1, T}(p), T(u)
end

struct NormalHermiteSplineSurrogate{D, T, F, RK} <: AbstractSurrogate{D, T}
    fg::F
    grid::Array{SVector{D, T}, D}
    ugrid::Array{T, D}
    ∇ugrid::Array{SVector{D, T}, D}
    spl::NormalHermiteSplines.ElasticNormalSpline{D, T, RK}
end

function NormalHermiteSplineSurrogate(fg, grid::Array{SVector{D, T}, D}, kernel = RK_H1(one(T))) where {D, T}
    return NormalHermiteSplineSurrogate(
        fg,
        grid,
        fill(T(NaN), size(grid)),
        fill(fill(T(NaN), SVector{D, T}), size(grid)),
        NormalHermiteSplines.ElasticNormalSpline(first(grid), last(grid), maximum(size(grid)), kernel),
    )
end

function update!(surr::NormalHermiteSplineSurrogate{D, T}, I::CartesianIndex{D}) where {D, T}
    u, ∇u = surr.fg(I)
    @inbounds begin
        p = surr.grid[I]
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
    surr.ugrid .= T(NaN)
    surr.∇ugrid .= (fill(T(NaN), SVector{D, T}),)
    return surr
end

function suggest_point(surr::NormalHermiteSplineSurrogate{D, T}) where {D, T}
    _, I = findmin(NormalHermiteSplines.evaluate!(vec(surr.ugrid), surr.spl, vec(surr.grid)))
    @inbounds p = surr.grid[I]
    p, u = local_search(surr, p)
    return p, u
end

# Specialize the 1D case to use the faster and more robust Brent-Dekker method
function suggest_point(surr::NormalHermiteSplineSurrogate{1, T}) where {T}
    @assert length(surr.grid) >= 2 "Grid must have at least 2 points"
    @inbounds p₀, u₀, ∇u₀, I = surr.grid[1], surr.ugrid[1], surr.∇ugrid[1], 1
    @inbounds for i in 2:length(surr.grid)
        pᵢ = surr.grid[i]
        uᵢ, ∇uᵢ = NormalHermiteSplines._evaluate_with_gradient(surr.spl, pᵢ)
        surr.ugrid[i], surr.∇ugrid[i] = uᵢ, ∇uᵢ
        if uᵢ < u₀
            p₀, u₀, ∇u₀, I = pᵢ, uᵢ, ∇uᵢ, i
        end
    end

    @inbounds if I == 1 || (I < length(surr.grid) && ∇u₀[1] < 0)
        # Grid minimizer is at the left endpoint, or downhill to the right
        p₁, p₂ = p₀, surr.grid[I+1]
        u₁, u₂ = u₀, surr.ugrid[I+1]
        ∇u₁, ∇u₂ = ∇u₀, surr.∇ugrid[I+1]
    else # I == length(surr.grid) || (I > 1 && ∇u₀[1] >= 0)
        # Grid minimizer is at the right endpoint, or downhill to the left
        p₁, p₂ = surr.grid[I-1], p₀
        u₁, u₂ = surr.ugrid[I-1], u₀
        ∇u₁, ∇u₂ = surr.∇ugrid[I-1], ∇u₀
    end

    # Use Brent's method to search for a minimum on the interval (p₁, p₂)
    xᵒᵖᵗ, uᵒᵖᵗ = brents_method(p₁[1], p₂[1]; xrtol = T(1e-4), xatol = T(1e-4), maxiters = 10) do x
        p = SA{T}[x]
        u = NormalHermiteSplines.evaluate(surr.spl, p)
        return u
    end
    pᵒᵖᵗ = SA{T}[xᵒᵖᵗ]

    # Brent's method doesn't evaluate the boundaries of the search interval; check manually
    if u₀ < uᵒᵖᵗ
        pᵒᵖᵗ, uᵒᵖᵗ = p₀, u₀
    end

    return pᵒᵖᵗ, uᵒᵖᵗ
end

####
#### Bounding box for multi-dimensional bisection search
####

struct BoundingBox{D, S, N}
    bounds::NTuple{D, NTuple{2, Int}}
    corners::SArray{S, CartesianIndex{D}, D, N}
end
corners(box::BoundingBox) = box.corners
bounds(box::BoundingBox) = box.bounds
widths(box::BoundingBox{D}) where {D} = ntuple(d -> abs(box.bounds[d][2] - box.bounds[d][1]), D)
Base.show(io::IO, ::MIME"text/plain", box::BoundingBox{D}) where {D} = print(io, "$D-D BoundingBox with dimensions: " * join(bounds(box), " × "))

BoundingBox(widths::NTuple{D, Int}) where {D} = BoundingBox(tuple.(1, widths))
BoundingBox(bounds::NTuple{D, NTuple{2, Int}}) where {D} = BoundingBox(bounds, corners(bounds))

@generated function corners(bounds::NTuple{D, NTuple{2, Int}}) where {D}
    corners = Iterators.product([(true, false) for d in 1:D]...)
    S = Tuple{ntuple(d -> 2, D)...}
    vals = [:(CartesianIndex($(ntuple(d -> I[d] ? :(bounds[$d][1]) : :(bounds[$d][2]), D)...))) for I in corners]
    return :(Base.@_inline_meta; $SArray{$S, CartesianIndex{$D}, $D, $(2^D)}(tuple($(vals...))))
end

function opposite_corner(box::BoundingBox{D}, I::CartesianIndex{D}) where {D}
    @inbounds lo, hi = first(corners(box)), last(corners(box))
    return lo + hi - I
end

function bisect(box::BoundingBox{D}) where {D}
    _, i = findmax(widths(box))
    left_bounds = ntuple(D) do d
        return i !== d ? box.bounds[d] : (box.bounds[i][1], (box.bounds[i][1] + box.bounds[i][2]) ÷ 2)
    end
    right_bounds = ntuple(D) do d
        return i !== d ? box.bounds[d] : ((box.bounds[i][1] + box.bounds[i][2]) ÷ 2, box.bounds[i][2])
    end
    return BoundingBox(left_bounds), BoundingBox(right_bounds)
end

splittable(box::BoundingBox{D}) where {D} = any(widths(box) .> 1)

####
#### Searching on a discrete grid using a surrogate function
####

struct DiscreteSurrogateSearcher{D, T}
    grid::Array{SVector{D, T}, D}
    seen::Array{Bool, D}
    numeval::Base.RefValue{Int}
end
function DiscreteSurrogateSearcher(surr::AbstractSurrogate; mineval::Int, maxeval::Int)
    @assert mineval <= maxeval
    state = DiscreteSurrogateSearcher(surr.grid, fill(false, size(surr.grid)), Ref(0))
    return initialize!(surr, state; mineval, maxeval)
end

function initialize!(surr::AbstractSurrogate{D}, state::DiscreteSurrogateSearcher{D}; mineval::Int, maxeval::Int) where {D}
    # Evaluate at least `mineval` points by repeatedly bisecting the grid in a breadth-first manner
    box = BoundingBox(size(state.grid))
    for depth in 1:mineval # should never reach `mineval` depth, this is just to ensure the loop terminates in case `mineval` is greater than the number of gridpoints
        initialize!(surr, state, box, depth; mineval, maxeval)
        state.numeval[] >= mineval && break
    end
    return state
end

function initialize!(surr::AbstractSurrogate{D}, state::DiscreteSurrogateSearcher{D}, box::BoundingBox{D}, depth::Int; mineval::Int, maxeval::Int) where {D}
    depth <= 0 && return state
    evaluate_box!(surr, state, box; maxeval)
    state.numeval[] ≥ mineval && return state
    left, right = bisect(box)
    initialize!(surr, state, left, depth - 1; mineval, maxeval)
    initialize!(surr, state, right, depth - 1; mineval, maxeval)
    return state
end

function update!(surr::AbstractSurrogate{D}, state::DiscreteSurrogateSearcher{D}, I::CartesianIndex{D}; maxeval::Int) where {D}
    # Update the surrogate function with a new point, returning whether the maximum number of function evaluations has been reached or not
    state.numeval[] >= maxeval && return true # check if already exceeded number of evals
    @inbounds state.seen[I] && return false # point already evaluated
    update!(surr, I) # update surrogate
    @inbounds state.seen[I] = true # mark as now seen
    @inbounds state.numeval[] += 1 # increment function call counter
    return state.numeval[] >= maxeval
end

####
#### Global optimization using multi-dimensional bisection with surrogate functions
####

function bisection_search(
    surr::AbstractSurrogate{D, T},
    state::DiscreteSurrogateSearcher{D, T};
    maxeval::Int,
) where {D, T}

    # Algorithm:
    #   0. Get initial optimum suggestion from surrogate
    #   REPEAT:
    #       1. Find smallest bounding box containing the optimum suggestion
    #       2. Evaluate the box corners
    #       3. Get new optimum suggestion from surrogate:
    #           IF: Box is sufficiently small or if the maximum number of evaluations has been reached:
    #               RETURN: Current optimum
    #           ELSE:
    #               GOTO:   1.
    x, u = suggest_point(surr)
    while true
        box = minimal_bounding_box(state, x)
        evaluate_box!(surr, state, box, x; maxeval)
        x, u = suggest_point(surr)
        if state.numeval[] ≥ maxeval || converged(state, box)
            return x, u
        end
    end
end

# Update observed evaluations, returning true if converged
function minimal_bounding_box(
    state::DiscreteSurrogateSearcher{D, T},
    x::SVector{D, T},
) where {D, T}

    box = BoundingBox(size(state.grid))
    while true
        left, right = bisect(box)
        if contains(state, left, x) # left box contains `x`
            if !is_evaluated(state, left) || !splittable(left)
                return left # left box not fully evaluated, or we have reached bottom; return
            else
                box = left # whole left box already evaluated; continue search
            end
        else # contains(state, right, x), i.e. right box contains `x`
            if !is_evaluated(state, right) || !splittable(right)
                return right # right box not fully evaluated, or we have reached bottom; return
            else
                box = right # whole right box already evaluated; continue search
            end
        end
    end
end

function evaluate_box!(
    surr::AbstractSurrogate{D, T},
    state::DiscreteSurrogateSearcher{D, T},
    box::BoundingBox{D},
    x::Union{Nothing, SVector{D, T}} = nothing;
    maxeval::Int,
) where {D, T}
    cs = x === nothing ? corners(box) : sorted_corners(state, box, x)
    @inbounds for I in cs
        is_evaluated(state, box) && break # box sufficiently evaluated
        update!(surr, state, I; maxeval) && break # update surrogate, breaking if max evals reached
    end
    return state
end

function is_evaluated(state::DiscreteSurrogateSearcher{D}, box::BoundingBox{D}) where {D}
    # Box is considered sufficiently evaluated when all of the corners have been evaluted
    return count(I -> @inbounds(state.seen[I]), corners(box)) >= 2^D
end

function converged(::DiscreteSurrogateSearcher{D}, box::BoundingBox{D}) where {D}
    # Convergence is defined as: bounding box has at least one side of length <= 1
    return any(widths(box) .<= 1)
end

function centre(state::DiscreteSurrogateSearcher{D, T}, box::BoundingBox{D}) where {D, T}
    @inbounds lo = state.grid[first(corners(box))]
    @inbounds hi = state.grid[last(corners(box))]
    return (lo + hi) / 2
end

function sorted_corners(state::DiscreteSurrogateSearcher{D, T}, box::BoundingBox{D}, x::SVector{D, T}) where {D, T}
    dist²(I) = @inbounds sum(abs2.(state.grid[I] - x))
    cs = corners(box)
    return typeof(cs)(TupleTools.sort(Tuple(cs); by = dist²))
end

function contains(state::DiscreteSurrogateSearcher{D, T}, box::BoundingBox{D}, x::SVector{D, T}) where {D, T}
    @inbounds lo = state.grid[first(corners(box))]
    @inbounds hi = state.grid[last(corners(box))]
    return all(lo .<= x .<= hi)
end

function is_inside(grid::AbstractArray{SVector{D, T}, D}, x::SVector{D, T}) where {D, T}
    @inbounds lo = first(grid)
    @inbounds hi = last(grid)
    return all(lo .< x .< hi)
end
is_inside(state::DiscreteSurrogateSearcher{D, T}, x::SVector{D, T}) where {D, T} = is_inside(state.grid, x)

####
#### Local optimization using surrogate functions
####

function local_search(surr::NormalHermiteSplineSurrogate{D, T}, x₀::SVector{D, T}) where {D, T}
    return error("Placeholder method --- not implemented")
end

#=
function local_search(
    surr::NormalHermiteSplineSurrogate{D, T},
    x₀::SVector{D, T},
    state::Union{Nothing, DiscreteSurrogateSearcher{D, T}} = nothing;
    maxiters::Int = 100,
    maxeval::Int = maxiters,
    xtol_rel = 1e-4,
    xtol_abs = 1e-4,
    initial_step = maximum(gridwidths(surr)) / 100,
    xeval_radius = √sum(abs2, gridspacings(surr)) - √eps(T),
) where {D, T}

    if state !== nothing
        # Initialize surrogate with domain corners
        box = BoundingBox(size(surr.grid))
        for I in corners(box)
            update!(surr, state, I; maxeval)
        end
    end

    x, xlast = x₀, x₀
    xlo, xhi = first(surr.grid), last(surr.grid)
    opt = ADAM{D, T}(initial_step)

    for _ in 1:maxiters
        if state !== nothing
            # Find nearest gridpoint to `x` and update surrogate
            I, xI = nearest_gridpoint(state, x)
            dmin = minimum(NormalHermiteSplines._get_nodes(surr.spl)) do p
                return norm(xI - NormalHermiteSplines._unnormalize(surr.spl, p))
            end
            if dmin > xeval_radius
                update!(surr, state, I; maxeval)
            end
        end

        # Perform gradient descent step using surrogate function
        ∇u = NormalHermiteSplines.evaluate_gradient(surr.spl, x)
        Δx, opt = update(∇u, opt)
        xlast, x = x, @. clamp(x - Δx, xlo, xhi)

        # Check for convergence
        maximum(abs.(x - xlast)) <= max(T(xtol_abs), T(xtol_rel) * maximum(abs.(x))) && break
    end

    u = NormalHermiteSplines.evaluate(surr.spl, x)

    return x, u
end

function local_search(
        surr::NormalHermiteSplineSurrogate{D,T},
        x₀::SVector{D,T};
        maxeval::Int = 100,
        xtol_rel = 1e-4,
        xtol_abs = 1e-4,
    ) where {D,T}

    # alg = :LN_COBYLA # local, gradient-free, linear approximation of objective
    # alg = :LN_BOBYQA # local, gradient-free, quadratic approximation of objective
    # alg = :LD_SLSQP # local, with-gradient, "Sequential Least-Squares Quadratic Programming"; uses dense-matrix methods (ordinary BFGS, not low-storage BFGS)
    alg = :LD_LBFGS # local, with-gradient, low-storage BFGS

    opt = NLopt.Opt(alg, D)
    opt.lower_bounds = Vector{Float64}(first(surr.grid))
    opt.upper_bounds = Vector{Float64}(last(surr.grid))
    opt.xtol_rel = xtol_rel
    opt.xtol_abs = xtol_abs
    opt.maxeval = maxeval
    opt.min_objective = function (x, g)
        x⃗ = SVector{D,T}(ntuple(d -> @inbounds(x[d]), D))
        @inbounds if length(g) > 0
            g .= Float64.(NormalHermiteSplines.evaluate_gradient(surr.spl, x⃗))
        end
        return Float64(NormalHermiteSplines.evaluate(spl, x⃗))
    end
    minf, minx, ret = NLopt.optimize(opt, Vector{Float64}(x₀))

    x = SVector{D,T}(ntuple(d -> @inbounds(minx[d]), D))
    u = T(minf)

    return x, u
end

function nearest_gridpoint(grid::AbstractArray{SVector{D, T}, D}, x::SVector{D, T}) where {D, T}
    @inbounds xlo, xhi = first(grid), last(grid)
    @inbounds Ilo, Ihi = first(CartesianIndices(grid)), last(CartesianIndices(grid))
    lo, hi = SVector(Tuple(Ilo)), SVector(Tuple(Ihi))
    i = @. clamp(round(Int, (x - xlo) * (hi - lo) / (xhi - xlo) + lo), lo, hi)
    I = CartesianIndex(Tuple(i))
    xI = @inbounds grid[I]
    return I, xI
end
nearest_gridpoint(state::DiscreteSurrogateSearcher{D, T}, x::SVector{D, T}) where {D, T} = nearest_gridpoint(state.grid, x)

function nearest_interior_gridpoint(grid::AbstractArray{SVector{D, T}, D}, x::SVector{D, T}) where {D, T}
    R = CartesianIndices(grid)
    One = CartesianIndex(ntuple(d -> 1, D))
    Ilo, Ihi = first(R) + One, last(R) - One
    return nearest_gridpoint(@views(grid[Ilo:Ihi]), x)
end
nearest_interior_gridpoint(state::DiscreteSurrogateSearcher{D, T}, x::SVector{D, T}) where {D, T} = nearest_interior_gridpoint(state.grid, x)
=#

####
#### Global optimization for NNLS problem
####

struct NNLSDiscreteSurrogateSearch{D, T, TA <: AbstractArray{T}, TdA <: AbstractArray{T}, Tb <: AbstractVector{T}, W}
    As::TA
    ∇As::TdA
    αs::Array{SVector{D, T}, D}
    b::Tb
    u::Array{T, D}
    ∂Ax⁺::Vector{T}
    Ax⁺b::Vector{T}
    nnls_work::W
    legacy::Bool
end

function NNLSDiscreteSurrogateSearch(
    As::AbstractArray{T},  # size(As)  = (M, N, P1..., PD)
    ∇As::AbstractArray{T}, # size(∇As) = (M, N, D, P1..., PD)
    αs::NTuple{D},         # size(αs)  = (P1..., PD)
    b::AbstractVector{T};  # size(b)   = (M,)
    legacy::Bool = false,
) where {D, T}
    M, N = size(As, 1), size(As, 2)
    @assert ndims(As) == 2 + D && ndims(∇As) == 3 + D # ∇As has extra dimension for parameter gradients
    @assert size(∇As)[1:3] == (M, N, D) # matrix dimensions must match, and gradient dimension must equal number of parameters
    @assert size(As)[3:end] == size(∇As)[4:end] == length.(αs) # dimension size must match parameters lengths
    @assert size(b) == (M,)

    αs = meshgrid(SVector{D, T}, αs...)
    u = zeros(T, size(αs))
    ∂Ax⁺ = zeros(T, M)
    Ax⁺b = zeros(T, M)
    nnls_work = lsqnonneg_work(zeros(T, M, N), zeros(T, M))
    return NNLSDiscreteSurrogateSearch(As, ∇As, αs, b, u, ∂Ax⁺, Ax⁺b, nnls_work, legacy)
end

load!(prob::NNLSDiscreteSurrogateSearch{D, T}, b::AbstractVector{T}) where {D, T} = copyto!(prob.b, b)

function loss!(prob::NNLSDiscreteSurrogateSearch{D, T}, I::CartesianIndex{D}) where {D, T}
    (; As, b, nnls_work) = prob
    solve!(nnls_work, uview(As, :, :, I), b)
    ℓ = chi2(nnls_work)
    u = prob.legacy ? ℓ : log(max(ℓ, eps(T))) # loss capped at eps(T) from below to avoid log(0) error
    return u
end

function ∇loss!(prob::NNLSDiscreteSurrogateSearch{D, T}, I::CartesianIndex{D}) where {D, T}
    (; As, ∇As, b, ∂Ax⁺, Ax⁺b, nnls_work) = prob
    ℓ = chi2(nnls_work)
    ℓ <= eps(T) && return zero(SVector{D, T}) # loss capped at eps(T) from below; return zero gradient
    x = solution(nnls_work)
    @inbounds @. Ax⁺b = -b
    @inbounds for j in 1:size(As, 2)
        (x[j] > 0) && axpy!(x[j], uview(As, :, j, I), Ax⁺b)
    end
    ∇u = ntuple(D) do d
        @inbounds ∂Ax⁺ .= zero(T)
        @inbounds for j in 1:size(∇As, 2)
            (x[j] > 0) && axpy!(x[j], uview(∇As, :, j, d, I), ∂Ax⁺)
        end
        ∂ℓ = 2 * dot(∂Ax⁺, Ax⁺b)
        ∂u = prob.legacy ? ∂ℓ : ∂ℓ / ℓ
        return ∂u
    end
    return SVector{D, T}(∇u)
end

function CubicSplineSurrogate(prob::NNLSDiscreteSurrogateSearch{1, T}; legacy = false) where {T}
    f = Base.Fix1(cubic_spline_surrogate_loss!, prob)
    return CubicSplineSurrogate(f, prob.αs, SVector{1, T}[], T[], Ref(0), legacy)
end
cubic_spline_surrogate_loss!(prob::NNLSDiscreteSurrogateSearch{1, T}, I::CartesianIndex{1}) where {T} = loss!(prob, I)

function NormalHermiteSplineSurrogate(prob::NNLSDiscreteSurrogateSearch{D, T}) where {D, T}
    fg = Base.Fix1(hermite_spline_surrogate_gradloss!, prob)
    return NormalHermiteSplineSurrogate(fg, prob.αs, RK_H1(one(T)))
end
function hermite_spline_surrogate_gradloss!(prob::NNLSDiscreteSurrogateSearch{D, T}, I::CartesianIndex{D}) where {D, T}
    u = loss!(prob, I)
    ∇u = ∇loss!(prob, I)
    return u, ∇u
end

function surrogate_spline_opt(
    prob::NNLSDiscreteSurrogateSearch{D},
    surr::AbstractSurrogate{D};
    mineval::Int = min(2^D, length(prob.αs)),
    maxeval::Int = length(prob.αs),
) where {D}
    state = DiscreteSurrogateSearcher(surr; mineval, maxeval)
    return bisection_search(surr, state; maxeval)
end

function spline_opt(
    spl::NormalHermiteSplines.NormalSpline{D, T},
    prob::NNLSDiscreteSurrogateSearch{D, T};
    # alg = :LN_COBYLA,        # local, gradient-free, linear approximation of objective
    # alg = :LN_BOBYQA,        # local, gradient-free, quadratic approximation of objective
    # alg = :GN_ORIG_DIRECT_L, # global, gradient-free, systematically divides search space into smaller hyper-rectangles via a branch-and-bound technique, systematic division of the search domain into smaller and smaller hyperrectangles, "more biased towards local search"
    # alg = :GN_AGS,           # global, gradient-free, employs the Hilbert curve to reduce the source problem to the univariate one.
    # alg = :GD_STOGO,         # global, with-gradient, systematically divides search space into smaller hyper-rectangles via a branch-and-bound technique, and searching them by a gradient-based local-search algorithm (a BFGS variant)
    alg = :LD_SLSQP,         # local, with-gradient, "Sequential Least-Squares Quadratic Programming"; uses dense-matrix methods (ordinary BFGS, not low-storage BFGS)
) where {D, T}

    NormalHermiteSplines.evaluate!(prob.u, spl, prob.αs)
    _, i = findmin(prob.u)
    α₀ = prob.αs[i]

    opt = NLopt.Opt(alg, D)
    opt.lower_bounds = Float64[prob.αs[begin]...]
    opt.upper_bounds = Float64[prob.αs[end]...]
    opt.xtol_rel = 0.001
    opt.min_objective = function (x, g)
        if length(g) > 0
            u, ∇u = NormalHermiteSplines._evaluate_with_gradient(spl, SVector{D, T}(x))
            @inbounds g .= Float64.(∇u)
        else
            u = NormalHermiteSplines.evaluate(spl, SVector{D, T}(x))
        end
        return Float64(u)
    end
    minf, minx, ret = NLopt.optimize(opt, Vector{Float64}(α₀))
    x, f = SVector{D, T}(minx), T(minf)
    return x, f
end

function mock_surrogate_search_problem(
    b::AbstractVector{T},
    opts::T2mapOptions{T},
    ::Val{D},
    ::Val{ETL};
    alphas = range(50, 180; length = opts.nRefAngles),
    betas = range(50, 180; length = opts.nRefAngles),
) where {D, T, ETL}

    # Mock CPMG image
    @assert opts.nTE == ETL
    opt_vars = D == 1 ? (:α,) : (:α, :β)
    opt_ranges = D == 1 ? (alphas,) : (alphas, betas)
    As = zeros(T, ETL, opts.nT2, length.(opt_ranges)...)
    ∇As = zeros(T, ETL, opts.nT2, D, length.(opt_ranges)...)
    T2s = logrange(opts.T2Range..., opts.nT2)
    θ = EPGOptions((; ETL, α = T(165.0), TE = opts.TE, T2 = zero(T), T1 = opts.T1, β = T(180.0)))
    j! = EPGJacobianFunctor(θ, Val(opt_vars))

    _, Rαs = SplitCartesianIndices(As, Val(2))
    for Iαs in Rαs
        @inbounds for j in 1:opts.nT2
            θαs = D == 1 ?
                  restructure(θ, (T2 = T2s[j], α = alphas[Iαs[1]])) :
                  restructure(θ, (T2 = T2s[j], α = alphas[Iαs[1]], β = alphas[Iαs[2]]))
            j!(uview(∇As, :, j, :, Iαs), uview(As, :, j, Iαs), θαs)
        end
    end

    return NNLSDiscreteSurrogateSearch(As, ∇As, opt_ranges, b)
end
function mock_surrogate_search_problem(::Val{D}, ::Val{ETL}, opts = mock_t2map_opts(; MatrixSize = (1, 1, 1), nTE = ETL); kwargs...) where {D, ETL}
    b = vec(mock_image(opts; kwargs...))
    return mock_surrogate_search_problem(b, opts, Val(D), Val(ETL); kwargs...)
end
function mock_surrogate_search_problem(b::AbstractVector, opts::T2mapOptions, ::Val{D}; kwargs...) where {D}
    @assert length(b) == opts.nTE
    return mock_surrogate_search_problem(b, opts, Val(D), Val(length(b)); kwargs...)
end
