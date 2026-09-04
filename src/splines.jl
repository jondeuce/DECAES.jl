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

@inline (spl::CubicHermiteInterpolator)(x, bc::Val = Val(:extrapolate)) = evalpoly(tocanonical(x, spl.dom, bc), spl.coeffs)

@inline function tocanonical(x, (a, b), ::Val{bc} = Val(:extrapolate)) where {bc}
    c, r = (a + b) / 2, (b - a) / 2
    t = (x - c) / r
    (bc === :nearest) && (t = clamp(t, -one(typeof(t)), one(typeof(t))))
    return t
end

@inline function todomain(t, (a, b), ::Val{bc} = Val(:extrapolate)) where {bc}
    c, r = (a + b) / 2, (b - a) / 2
    x = muladd(r, t, c)
    (bc === :nearest) && (x = clamp(x, a, b))
    return x
end

@inline incanonical(t) = !isnan(t) && -one(t) <= t <= one(t)

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
            return todomain(t, dom, Val(:nearest)), y
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
    x1, s1 = incanonical(t1) ? (x = todomain(t1, dom, Val(:nearest)); xlo <= x <= xhi ? (x, s1) : (T(NaN), T(NaN))) : (T(NaN), T(NaN))
    x2, s2 = incanonical(t2) ? (x = todomain(t2, dom, Val(:nearest)); xlo <= x <= xhi ? (x, s2) : (T(NaN), T(NaN))) : (T(NaN), T(NaN))
    x3, s3 = incanonical(t3) ? (x = todomain(t3, dom, Val(:nearest)); xlo <= x <= xhi ? (x, s3) : (T(NaN), T(NaN))) : (T(NaN), T(NaN))
    (x1, s1), (x2, s2), (x3, s3) = sorttuple(((x1, s1), (x2, s2), (x3, s3)); by = first, lt = lt_nan) # sort, treating NaN's as Inf
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
            x1, x2, x3 = sorttuple((zero(T), x1, x2))
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
        x1, x2, x3 = sorttuple((x1, x2, x3))
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
#### Interpolating cubic splines
####

# Reusable C² cubic spline represented by Hermite segments. Boundary conditions are `:notaknot`, `:natural`, or `:zeroslope`, specified as a `Symbol` or as a tuple `(left, right)` thereof.
const BoundaryConditions = Union{Symbol, NTuple{2, Symbol}}

@inline function boundary_conditions(bc::BoundaryConditions)
    bcl, bcr = bc isa Symbol ? (bc, bc) : bc
    @assert bcl ∈ (:notaknot, :natural, :zeroslope) "boundary condition must be one of :notaknot, :natural, or :zeroslope, got :$bcl"
    @assert bcr ∈ (:notaknot, :natural, :zeroslope) "boundary condition must be one of :notaknot, :natural, or :zeroslope, got :$bcr"
    return (bcl, bcr)
end

struct CubicSpline{T <: AbstractFloat}
    x::Vector{T}  # knots, sorted ascending
    u::Vector{T}  # values at the knots
    m::Vector{T}  # slopes at the knots; also the right-hand side during the solve
    dl::Vector{T} # subdiagonal
    d::Vector{T}  # diagonal
    du::Vector{T} # superdiagonal
    npts::Base.RefValue{Int}
end
CubicSpline{T}(maxpts::Int) where {T <: AbstractFloat} = CubicSpline{T}(ntuple(_ -> zeros(T, maxpts), 6)..., Ref(0))

@inline Base.length(spl::CubicSpline) = spl.npts[]
@inline nsegments(spl::CubicSpline) = max(length(spl) - 1, 0)

# The `i`th segment as a cubic Hermite interpolant, optionally shifted so that its roots solve `spl(x) = value`.
@inline function segment(spl::CubicSpline{T}, i::Int, value::T = zero(T)) where {T}
    (; x, u, m) = spl
    @inbounds return CubicHermiteInterpolator(x[i], x[i+1], u[i] - value, u[i+1] - value, m[i], m[i+1])
end

# Interpolate `(xs, us)` into `spl`.
function interpolate!(spl::CubicSpline{T}, xs::AbstractVector, us::AbstractVector, bc::BoundaryConditions = :notaknot) where {T}
    n = length(xs)
    bcl, bcr = boundary_conditions(bc)
    @assert length(us) == n "xs and us must have the same length, got $(length(xs)) and $(length(us))"
    @assert 1 <= n <= length(spl.x) "spline holds at most $(length(spl.x)) knots, got $n"
    @assert all(i -> isfinite(T(xs[i])), 1:n) && (n == 1 || all(i -> T(xs[i]) < T(xs[i+1]), 1:n-1)) "spline knots must be finite and strictly increasing"

    (; x, u, m, dl, d, du) = spl
    spl.npts[] = n
    @inbounds for i in 1:n
        x[i], u[i] = xs[i], us[i]
    end

    # Interval widths and secant slopes
    h(i) = @inbounds x[i+1] - x[i]
    δ(i) = @inbounds (u[i+1] - u[i]) / h(i)

    # Not-a-knot requires two intervals
    n < 3 && (bcl === :notaknot) && (bcl = :natural)
    n < 3 && (bcr === :notaknot) && (bcr = :natural)

    if n == 1
        @inbounds m[1] = zero(T)
        return spl
    elseif n == 3 && bcl === :notaknot && bcr === :notaknot
        # Quadratic through three knots
        @inbounds c₂ = (δ(2) - δ(1)) / (x[3] - x[1])
        @inbounds m[1], m[2], m[3] = δ(1) - c₂ * h(1), δ(1) + c₂ * h(1), δ(1) + c₂ * (h(1) + 2 * h(2))
        return spl
    end

    @inbounds for i in 2:n-1
        dl[i], d[i], du[i] = h(i), 2 * (h(i - 1) + h(i)), h(i - 1)
        m[i] = 3 * (h(i) * δ(i - 1) + h(i - 1) * δ(i))
    end

    @inbounds if bcl === :zeroslope
        d[1], du[1], m[1] = one(T), zero(T), zero(T)
    elseif bcl === :natural
        d[1], du[1], m[1] = T(2), one(T), 3 * δ(1)
    else # :notaknot
        h₁, h₂ = h(1), h(2)
        d[1], du[1] = h₂, h₁ + h₂
        m[1] = ((h₁ + 2 * (h₁ + h₂)) * h₂ * δ(1) + h₁^2 * δ(2)) / (h₁ + h₂)
    end

    @inbounds if bcr === :zeroslope
        dl[n], d[n], m[n] = zero(T), one(T), zero(T)
    elseif bcr === :natural
        dl[n], d[n], m[n] = one(T), T(2), 3 * δ(n - 1)
    else # :notaknot
        hₙ, hₙ₋₁ = h(n - 1), h(n - 2)
        dl[n], d[n] = hₙ + hₙ₋₁, hₙ₋₁
        m[n] = ((hₙ + 2 * (hₙ + hₙ₋₁)) * hₙ₋₁ * δ(n - 1) + hₙ^2 * δ(n - 2)) / (hₙ + hₙ₋₁)
    end

    # Thomas algorithm, overwriting `du` with the swept superdiagonal and `m` with the solution
    @inbounds du[1] /= d[1]
    @inbounds m[1] /= d[1]
    @inbounds for i in 2:n
        p = d[i] - dl[i] * du[i-1]
        du[i] /= p
        m[i] = (m[i] - dl[i] * m[i-1]) / p
    end
    @inbounds for i in n-1:-1:1
        m[i] -= du[i] * m[i+1]
    end

    return spl
end

# Index of the segment containing `x`, clamping to the end segments outside the domain
@inline function findsegment(spl::CubicSpline, x)
    i = searchsortedlast(@views(spl.x[1:length(spl)]), x)
    return clamp(i, 1, nsegments(spl))
end

@inline function (spl::CubicSpline{T})(x) where {T}
    length(spl) == 1 && return @inbounds spl.u[1]
    return segment(spl, findsegment(spl, x))(x)
end

# Minimizer of the spline over its domain and the value there
function minimize(spl::CubicSpline{T}) where {T}
    length(spl) == 1 && return @inbounds (spl.x[1], spl.u[1])
    x, u = minimize(segment(spl, 1))
    for i in 2:nsegments(spl)
        xᵢ, uᵢ = minimize(segment(spl, i))
        uᵢ < u && ((x, u) = (xᵢ, uᵢ))
    end
    return (x, u)
end

# Leftmost solution of `spl(x) = value` over the domain, or NaN if there is none
function spline_root(spl::CubicSpline{T}, value::T = zero(T)) where {T}
    for i in 1:nsegments(spl)
        seg = segment(spl, i, value)
        for r in roots(seg)
            isnan(r) && break # real roots only, sorted with NaN last
            return r
        end
    end
    return T(NaN)
end

# Fit a spline to the data `(X, Y)` and minimize it, or solve `spl(x) = value`
spline_opt(X::AbstractVector, Y::AbstractVector; bc::BoundaryConditions = :notaknot) = ((x, y) = minimize(cubic_spline(X, Y; bc)); return (; x, y))
spline_root(X::AbstractVector, Y::AbstractVector, value::Number = 0; bc::BoundaryConditions = :notaknot) = (spl = cubic_spline(X, Y; bc); return spline_root(spl, eltype(spl.x)(value)))
cubic_spline(X::AbstractVector, Y::AbstractVector; bc::BoundaryConditions = :notaknot) = interpolate!(CubicSpline{floattype((zero(eltype(X)), zero(eltype(Y))))}(length(X)), X, Y, bc)

####
#### Surrogate functions over discrete grids
####

abstract type AbstractSurrogate{D, T} end

#### CubicSplineSurrogate

struct CubicSplineSurrogate{T, F} <: AbstractSurrogate{1, T}
    f::F
    grid::Vector{SVector{1, T}}
    seen::Vector{Bool}
    u::Vector{T}
    idx::Vector{Int}
    npts::Base.RefValue{Int}
    spline::CubicSpline{T}
    bc::NTuple{2, Symbol}
end

function CubicSplineSurrogate(f, grid::Vector{SVector{1, T}}; bc::BoundaryConditions = :notaknot) where {T}
    return CubicSplineSurrogate(
        f,
        grid,
        fill(false, length(grid)),
        fill(T(NaN), length(grid)),
        zeros(Int, length(grid)),
        Ref(0),
        CubicSpline{T}(length(grid)),
        boundary_conditions(bc),
    )
end

function Base.empty!(surr::CubicSplineSurrogate)
    surr.npts[] = 0
    surr.seen .= false
    return surr
end

function update!(surr::CubicSplineSurrogate, I::CartesianIndex{1})
    @inbounds(surr.seen[I]) && return surr
    u = surr.f(I)
    @inbounds surr.seen[I] = true
    @inbounds surr.u[I] = u
    insertsorted!(surr.idx, I[1], surr.npts[] += 1)
    return surr
end

function suggest_point(surr::CubicSplineSurrogate{T}) where {T}
    npts = surr.npts[]
    idx = @views surr.idx[1:npts]
    ps = @views reinterpret(T, surr.grid)[idx]
    us = @views surr.u[idx]
    p, u = minimize(interpolate!(surr.spline, ps, us, surr.bc))
    return SVector{1, T}(p), T(u)
end

#### CubicHermiteSplineSurrogate

struct CubicHermiteSplineSurrogate{T, F} <: AbstractSurrogate{1, T}
    fg::F
    grid::Vector{SVector{1, T}}
    seen::Vector{Bool}
    u::Vector{T}
    ∇u::Vector{SVector{1, T}}
    idx::Vector{Int}
    npts::Base.RefValue{Int}
end

function CubicHermiteSplineSurrogate(fg, grid::Vector{SVector{1, T}}) where {T}
    return CubicHermiteSplineSurrogate(
        fg,
        grid,
        fill(false, size(grid)),
        fill(T(NaN), size(grid)),
        fill(SVector{1, T}(T(NaN)), size(grid)),
        zeros(Int, size(grid)),
        Ref(0),
    )
end

function update!(surr::CubicHermiteSplineSurrogate{T}, I::CartesianIndex{1}) where {T}
    @inbounds(surr.seen[I]) && return surr
    u, ∇u = surr.fg(I)
    @inbounds surr.seen[I] = true
    @inbounds surr.u[I] = u
    @inbounds surr.∇u[I] = ∇u
    insertsorted!(surr.idx, I[1], surr.npts[] += 1)
    return surr
end

function Base.empty!(surr::CubicHermiteSplineSurrogate{T}) where {T}
    surr.npts[] = 0
    surr.seen .= false
    surr.u .= T(NaN)
    surr.∇u .= (SVector{1, T}(T(NaN)),)
    return surr
end

# Best evaluated node. Its loss is a true NNLS evaluation, so it is the incumbent the final selection certifies against.
# `idx` is sorted by angle and `argmin` keeps the first minimizer, so scanning in reverse resolves ties toward the largest angle, where refocusing flip angles concentrate.
best_seen_index(surr::CubicHermiteSplineSurrogate) = argmin(I -> surr.u[I], view(surr.idx, surr.npts[]:-1:1))

function suggest_point(surr::CubicHermiteSplineSurrogate{T}) where {T}
    @assert length(surr.grid) >= 2 "Grid must have at least 2 points"
    @assert surr.npts[] >= 1 "No points have been added to the surrogate"

    plast, ulast, ∇ulast = @inbounds begin
        I0 = surr.idx[1]
        surr.grid[I0], surr.u[I0], surr.∇u[I0]
    end

    p, u = plast, ulast
    @inbounds for i in 2:surr.npts[]
        I = surr.idx[i]
        pcurr, ucurr, ∇ucurr = surr.grid[I], surr.u[I], surr.∇u[I]
        spl = CubicHermiteInterpolator(plast[1], pcurr[1], ulast, ucurr, ∇ulast[1], ∇ucurr[1])
        _x, _u = minimize(spl)
        if _u < u
            p, u = SVector{1, T}(_x), _u
        end
        plast, ulast, ∇ulast = pcurr, ucurr, ∇ucurr
    end

    return p, u
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
    grid::Array{SVector{D, T}, D} # parameter grid being searched
    seen::Array{Bool, D} # whether the surrogate has been evaluated at each grid point
    numeval::Base.RefValue{Int} # number of evaluations performed
    plan::Vector{CartesianIndex{D}} # scratch buffer of planned initialization points; see `initialize!`
end
function DiscreteSurrogateSearcher(grid::Array{SVector{D, T}, D}) where {D, T}
    return DiscreteSurrogateSearcher(grid, fill(false, size(grid)), Ref(0), sizehint!(CartesianIndex{D}[], length(grid)))
end
function DiscreteSurrogateSearcher(surr::AbstractSurrogate; mineval::Int, maxeval::Int)
    @assert mineval <= maxeval
    state = DiscreteSurrogateSearcher(surr.grid)
    return initialize!(surr, state; mineval, maxeval)
end

# Clear the evaluation state so the searcher can be reused for a new search without reallocating.
# The flip-angle optimization reuses one searcher per thread, resetting it once per voxel.
function reset!(state::DiscreteSurrogateSearcher)
    fill!(state.seen, false)
    state.numeval[] = 0
    return state
end

function initialize!(surr::AbstractSurrogate{D}, state::DiscreteSurrogateSearcher{D}; mineval::Int, maxeval::Int) where {D}
    # Neighbouring gridpoints have similar loss-function state, such as the nearby decay bases of the NNLS surrogate search, so evaluating in sorted order chains warm starts through small parameter jumps.
    planned = plan_initialize!(state; mineval, maxeval)
    for I in planned
        update!(surr, state, I; maxeval)
    end
    return state
end

function plan_initialize!(state::DiscreteSurrogateSearcher{1}; mineval::Int, maxeval::Int)
    # `K₀` indices whose gaps are ⌊(K−1)/(K₀−1)⌋ or ⌈(K−1)/(K₀−1)⌉.
    K = length(state.grid)
    K₀ = clamp(mineval, 2, K)
    planned = empty!(state.plan)
    for q in 1:K₀
        push!(planned, CartesianIndex(1 + ((q - 1) * (K - 1)) ÷ (K₀ - 1)))
    end
    return planned
end

function plan_initialize!(state::DiscreteSurrogateSearcher{D}; mineval::Int, maxeval::Int) where {D}
    # Higher dimensions use a recursive dyadic ordering.
    box = BoundingBox(size(state.grid))
    planned = empty!(state.plan)
    for depth in 1:mineval # should never reach `mineval` depth, this is just to ensure the loop terminates in case `mineval` is greater than the number of gridpoints
        plan_initialize!(planned, box, depth; mineval, maxeval)
        length(planned) >= mineval && break
    end
    sort!(planned) # `CartesianIndex` sorts in column-major order, matching the grid layout; `sort!` on a sized buffer allocates nothing
    return planned
end

# Dry-run mirror of the recursive initialization: record the indices `evaluate_box!` and `update!` would evaluate, evaluating nothing. Membership in `planned` plays the role of `state.seen` and `state.numeval`.
function plan_initialize!(planned::Vector{CartesianIndex{D}}, box::BoundingBox{D}, depth::Int; mineval::Int, maxeval::Int) where {D}
    depth <= 0 && return planned
    cs = corners(box)
    for I in cs
        count(in(planned), cs) >= 2^D && break # box sufficiently evaluated
        length(planned) >= maxeval && break # max evals reached
        I in planned || push!(planned, I)
    end
    length(planned) >= mineval && return planned
    left, right = bisect(box)
    plan_initialize!(planned, left, depth - 1; mineval, maxeval)
    plan_initialize!(planned, right, depth - 1; mineval, maxeval)
    return planned
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
    #               GOTO: 1.
    x, u = suggest_point(surr)
    while true
        box = minimal_bounding_box(state, x)
        evaluate_box!(surr, state, box, x; maxeval)
        x, u = suggest_point(surr)
        if state.numeval[] ≥ maxeval || is_resolved(state, x)
            return x, u
        end
    end
end

function projected_search(
    surr::AbstractSurrogate{1, T},
    state::DiscreteSurrogateSearcher{1, T};
    maxeval::Int,
) where {T}
    x, u = suggest_point(surr)

    @inbounds while state.numeval[] < maxeval
        i = clamp(searchsortedlast(state.grid, x; by = first), 1, length(state.grid) - 1)

        Il, Ir = state.grid[i] == x ? (CartesianIndex(max(i - 1, 1)), CartesianIndex(min(i + 1, length(state.grid)))) :
                 state.grid[i+1] == x ? (CartesianIndex(i), CartesianIndex(min(i + 2, length(state.grid)))) :
                 (CartesianIndex(i), CartesianIndex(i + 1))

        state.seen[Il] && state.seen[Ir] && break

        I = state.seen[Il] ? Ir :
            state.seen[Ir] ? Il :
            abs(first(state.grid[Il]) - first(x)) <= abs(first(state.grid[Ir]) - first(x)) ? Il : Ir

        update!(surr, state, I; maxeval)

        x, u = suggest_point(surr)
    end

    return x, u
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

function is_resolved(state::DiscreteSurrogateSearcher{D, T}, x::SVector{D, T}) where {D, T}
    # Resolved means `x` is an evaluated node, or sits in a cell whose corners are all evaluated. `minimal_bounding_box` searches only the dyadic hierarchy, so it can miss a qualifying cell and call a resolved `x` unresolved, costing extra evaluations.
    box = minimal_bounding_box(state, x)
    is_evaluated(state, box) && return true
    return any(I -> @inbounds(state.seen[I] && state.grid[I] == x), corners(box))
end

function centre(state::DiscreteSurrogateSearcher{D, T}, box::BoundingBox{D}) where {D, T}
    @inbounds lo = state.grid[first(corners(box))]
    @inbounds hi = state.grid[last(corners(box))]
    return (lo + hi) / 2
end

function sorted_corners(state::DiscreteSurrogateSearcher{D, T}, box::BoundingBox{D}, x::SVector{D, T}) where {D, T}
    dist²(I) = @inbounds sum(abs2.(state.grid[I] - x))
    cs = corners(box)
    return typeof(cs)(sorttuple(Tuple(cs); by = dist²))
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
#### Global optimization for NNLS problem
####

# Runtime toggle for the precomputed-Gram fast path of the surrogate search's NNLS evaluations (see `NNLS.NNLSPrecomputedGram`); `false` recovers the exact-QR evaluation path for every solve.
const SURROGATE_USE_FAST_GRAM = Ref(true)

struct NNLSDiscreteSurrogateSearch{D, T, TA <: AbstractArray{T}, TdA <: AbstractArray{T}, Tb <: AbstractVector{T}, W, WG}
    As::TA # decay bases over the parameter grid; As[:, :, I] is A(αs[I])
    ∇As::TdA # parameter derivatives of the grid bases; ∇As[:, :, d, I] is ∂A(αs[I])/∂αs[I][d]
    Gs::Array{T, 3} # Gram matrices of the grid bases, by linear grid index; Gs[:, :, lin] is AᵀA there
    αs::Array{SVector{D, T}, D} # parameter grid
    b::Tb # decay curve data
    u::Array{T, D} # loss values at the evaluated grid points
    nnls_work::W # exact QR solver workspace
    nnls_gram::WG # Gram fast path for the loss evaluations (see `SURROGATE_USE_FAST_GRAM`)
    seen_pts::Vector{CartesianIndex{D}} # grid points evaluated during the current search
    seen_idx::Matrix{Int} # column p holds the active set, as original column indices, of the solve at grid point p by linear index
    seen_nsetp::Vector{Int} # active-set size per grid point, by linear index
    seen_stamp::Vector{Int} # voxel counter when seen_idx[:, p] was last written, 0 meaning never; enables cross-voxel warm starts
    voxel::Base.RefValue{Int} # monotonic voxel counter; a grid point last written at voxel-1 seeds the same grid point this voxel, same A and nearby b
end

function NNLSDiscreteSurrogateSearch(
    As::AbstractArray{T},  # size(As)  = (M, N, P1..., PD)
    ∇As::AbstractArray{T}, # size(∇As) = (M, N, D, P1..., PD)
    Gs::Array{T, 3},       # size(Gs)  = (N, N, prod(P1..., PD))
    αs::NTuple{D},         # size(αs)  = (P1..., PD)
    b::AbstractVector{T},  # size(b)   = (M,)
) where {D, T}
    M, N = size(As, 1), size(As, 2)
    @assert ndims(As) == 2 + D && ndims(∇As) == 3 + D # ∇As has extra dimension for parameter gradients
    @assert size(∇As)[1:3] == (M, N, D) # matrix dimensions must match, and gradient dimension must equal number of parameters
    @assert size(As)[3:end] == size(∇As)[4:end] == length.(αs) # dimension size must match parameters lengths
    @assert size(Gs) == (N, N, prod(length.(αs)))
    @assert size(b) == (M,)

    αs = meshgrid(SVector{D, T}, αs...)
    u = zeros(T, size(αs))
    nnls_work = lsqnonneg_work(zeros(T, M, N), zeros(T, M))
    nnls_gram = NNLS.NNLSPrecomputedGram(T, N)
    seen_pts = sizehint!(CartesianIndex{D}[], length(αs))
    seen_idx = zeros(Int, N, length(αs))
    seen_nsetp = zeros(Int, length(αs))
    seen_stamp = zeros(Int, length(αs))
    return NNLSDiscreteSurrogateSearch(As, ∇As, Gs, αs, b, u, nnls_work, nnls_gram, seen_pts, seen_idx, seen_nsetp, seen_stamp, Ref(1))
end

load!(prob::NNLSDiscreteSurrogateSearch{D, T}, b::AbstractVector{T}) where {D, T} = copyto!(prob.b, b)

# Fully reset warm-start state: the next `loss!` solves cold and every per-gridpoint seed is discarded, this-voxel and cross-voxel alike, so the search is independent of any prior state.
reset_warmstart!(prob::NNLSDiscreteSurrogateSearch) = (empty!(prob.seen_pts); fill!(prob.seen_stamp, 0); prob.voxel[] = 1; prob)

# Advance to the next voxel without discarding the per-gridpoint active sets. A grid point last evaluated in the previous voxel has the same decay basis `As` and a nearby signal `b`, so it seeds the same grid point this voxel; see `loss!`.
# Only the this-voxel `seen_pts` list is cleared. NNLS converges to the same solution from any seed, so this changes solve speed, not results.
advance_warmstart!(prob::NNLSDiscreteSurrogateSearch) = (prob.voxel[] += 1; empty!(prob.seen_pts); prob)

function loss!(prob::NNLSDiscreteSurrogateSearch{D, T}, I::CartesianIndex{D}) where {D, T}
    (; As, b, nnls_work, seen_pts, seen_idx, seen_nsetp, seen_stamp, voxel) = prob
    lin = LinearIndices(size(prob.u))[I]

    # Choose the warm-start seed in order of preference. First this grid point's active set from the immediately-previous voxel, which shares its decay basis and has a nearby signal.
    # Otherwise the nearest grid point already evaluated this voxel, since nearby parameters have nearly identical decay bases and hence nearly identical active sets, and even a far seed beats a cold solve. Otherwise solve cold.
    seedlin = 0
    @inbounds begin
        prevstamp = seen_stamp[lin]
        if prevstamp > 0 && prevstamp == voxel[] - 1
            seedlin = lin
        else
            bestd = typemax(Int)
            for J in seen_pts
                d = 0
                for k in 1:D
                    d += abs(J[k] - I[k])
                end
                if d < bestd
                    seedlin, bestd = LinearIndices(size(prob.u))[J], d
                end
            end
        end
    end
    np0 = seedlin == 0 ? 0 : @inbounds(seen_nsetp[seedlin])

    # Precomputed-Gram fast path, reading the seed out of seen_idx before it is overwritten below. The exact QR solve takes over on toggle-off or a conditioning or iteration guard failure.
    solved = SURROGATE_USE_FAST_GRAM[] && @views loss_gram!(prob, I, lin, seen_idx[:, max(seedlin, 1)], np0)
    @inbounds if !solved
        if np0 > 0
            @views solve!(nnls_work, As[:, :, I], b, seen_idx[:, seedlin], np0)
        else
            @views solve!(nnls_work, As[:, :, I], b)
        end
    end

    # Record the active set for future warm starts; each grid point is evaluated at most once per search, so no overwrite occurs.
    work = nnls_work.nnls_work
    ns = NNLS.ncomponents(work)
    seen_nsetp[lin] = ns
    @inbounds for t in 1:ns
        seen_idx[t, lin] = work.idx[t]
    end
    seen_stamp[lin] = voxel[]
    push!(seen_pts, I)
    u = resnorm_sq(nnls_work)
    return u
end

# Envelope-theorem gradient ∂u/∂αd = 2·(∂A_P x_P)ᵀ(A_P x_P − b), valid by Danskin's theorem because x is the NNLS minimizer at α.
# The solve already left r = b − A_P x_P current, so the fit residual is −r and the gradient is −2·Σ_{t} x_t ⟨∂A[:, P_t], r⟩:
# accumulating that as one dot product per support column avoids rebuilding the residual, materializing ∂A_P x_P, and scanning all n columns for positivity.
function ∇loss!(prob::NNLSDiscreteSurrogateSearch{D, T}, I::CartesianIndex{D}) where {D, T}
    (; ∇As, nnls_work) = prob
    work = nnls_work.nnls_work
    r = NNLS.residual(work)
    x = NNLS.solution(work)
    m, p = size(∇As, 1), NNLS.ncomponents(work)
    ∇u = ntuple(D) do d
        s = zero(T)
        @inbounds for t in 1:p
            j = work.idx[t]
            sj = zero(T)
            @simd for i in 1:m
                sj = muladd(∇As[i, j, d, I], r[i], sj)
            end
            s = muladd(x[j], sj, s)
        end
        return -2 * s
    end
    return SVector{D, T}(∇u)
end

function loss_with_grad!(prob::NNLSDiscreteSurrogateSearch{D, T}, I::CartesianIndex{D}) where {D, T}
    u = loss!(prob, I)
    ∇u = ∇loss!(prob, I)
    return u, ∇u
end

# Evaluate the loss at grid point I via the precomputed-Gram fast path, seeded with the active set idx0[1:np0] (np0 = 0 solves cold).
# On success the solution, active set, and exact residual norm are written into the NNLS workspace, which is everything the downstream consumers read, and `true` is returned. `false` means the caller must run the exact QR solver instead.
function loss_gram!(prob::NNLSDiscreteSurrogateSearch{D, T}, I::CartesianIndex{D}, lin::Int, idx0, np0::Int) where {D, T}
    (; As, Gs, b, nnls_work, nnls_gram) = prob
    A, G = view(As, :, :, I), view(Gs, :, :, lin)
    m = size(A, 1)
    NNLS.load!(nnls_gram, A, b)
    NNLS.set_active!(nnls_gram, G, idx0, np0)
    NNLS.solve!(nnls_gram, G, m) || return false

    # Scatter the results into the NNLS workspace; the residual norm is computed from the exact m-space residual r = b − A_P x_P
    work = nnls_work.nnls_work
    p = nnls_gram.np[]
    fill!(work.x, zero(T))
    @inbounds for t in 1:p
        work.x[nnls_gram.P[t]] = nnls_gram.xp[t]
        work.idx[t] = nnls_gram.P[t]
    end
    r = work.r
    @inbounds @simd ivdep for i in 1:m
        r[i] = b[i]
    end
    t = 1
    @inbounds while t + 1 <= p # two support columns per pass, halving the loads and stores of r
        x1, x2 = nnls_gram.xp[t], nnls_gram.xp[t+1]
        j1, j2 = nnls_gram.P[t], nnls_gram.P[t+1]
        @simd ivdep for i in 1:m
            r[i] = r[i] - x1 * A[i, j1] - x2 * A[i, j2]
        end
        t += 2
    end
    res² = zero(T)
    if t <= p
        xt, jt = @inbounds(nnls_gram.xp[t]), @inbounds(nnls_gram.P[t])
        @inbounds @simd for i in 1:m # the last column folds the norm into its own pass
            ri = r[i] - xt * A[i, jt]
            r[i] = ri
            res² = muladd(ri, ri, res²)
        end
    else
        @inbounds @simd for i in 1:m
            res² = muladd(r[i], r[i], res²)
        end
    end
    work.rnorm[] = sqrt(res²)
    work.nsetp[] = p
    work.mode[] = 0
    work.solved[] = true
    return true
end

function CubicSplineSurrogate(prob::NNLSDiscreteSurrogateSearch{1, T}; bc::BoundaryConditions = :notaknot) where {T}
    f = Base.Fix1(loss!, prob)
    return CubicSplineSurrogate(f, prob.αs; bc)
end

function CubicHermiteSplineSurrogate(prob::NNLSDiscreteSurrogateSearch{1, T}) where {T}
    fg = Base.Fix1(loss_with_grad!, prob)
    return CubicHermiteSplineSurrogate(fg, prob.αs)
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

function mock_surrogate_search_problem(
    b::AbstractVector{T},
    opts::T2mapOptions{T},
    ::Val{D},
    ::Val{ETL};
    alphas = range(deg2rad(50.0), π; length = opts.nRefAngles),
    betas = range(deg2rad(50.0), π; length = opts.nRefAngles),
) where {D, T, ETL}

    # Mock CPMG image
    @assert opts.nTE == ETL
    opt_vars = D == 1 ? (:α,) : (:α, :β)
    opt_ranges = D == 1 ? (alphas,) : (alphas, betas)
    As = zeros(T, ETL, opts.nT2, length.(opt_ranges)...)
    ∇As = zeros(T, ETL, opts.nT2, D, length.(opt_ranges)...)
    T2s = logrange(opts.T2Range..., opts.nT2)
    θ = EPGOptions((; ETL, α = deg2rad(T(165.0)), TE = opts.TE, T2 = zero(T), T1 = opts.T1, β = T(π)))
    j! = EPGJacobianFunctor(θ, Val(opt_vars))

    _, Rαs = SplitCartesianIndices(As, Val(2))
    for Iαs in Rαs
        @inbounds for j in 1:opts.nT2
            θαs = D == 1 ?
                  restructure(θ, (T2 = T2s[j], α = alphas[Iαs[1]])) :
                  restructure(θ, (T2 = T2s[j], α = alphas[Iαs[1]], β = betas[Iαs[2]]))
            @views j!(∇As[:, j, :, Iαs], As[:, j, Iαs], θαs)
        end
    end

    Gs = zeros(T, opts.nT2, opts.nT2, prod(length.(opt_ranges)))
    @views for (lin, Iαs) in enumerate(Rαs)
        mul!(Gs[:, :, lin], As[:, :, Iαs]', As[:, :, Iαs])
    end

    return NNLSDiscreteSurrogateSearch(As, ∇As, Gs, opt_ranges, b)
end
function mock_surrogate_search_problem(::Val{D}, ::Val{ETL}, opts = mock_t2map_opts(; MatrixSize = (1, 1, 1), nTE = ETL); kwargs...) where {D, ETL}
    b = vec(mock_image(opts; kwargs...))
    return mock_surrogate_search_problem(b, opts, Val(D), Val(ETL); kwargs...)
end
function mock_surrogate_search_problem(b::AbstractVector, opts::T2mapOptions, ::Val{D}; kwargs...) where {D}
    @assert length(b) == opts.nTE
    return mock_surrogate_search_problem(b, opts, Val(D), Val(length(b)); kwargs...)
end
