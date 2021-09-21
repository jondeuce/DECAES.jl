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
    @inbounds for (i, p) in enumerate(polys)
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
    @inbounds for (i, p) in enumerate(polys)
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
#### Surrogate functions over discrete grids
####

abstract type AbstractSurrogate{D,T} end

struct CubicSplineSurrogate{T,F} <: AbstractSurrogate{1,T}
    f::F
    grid::Vector{SVector{1,T}}
    p::Vector{SVector{1,T}}
    u::Vector{T}
end

function CubicSplineSurrogate(f, grid::Vector{SVector{1,T}}) where {T}
    CubicSplineSurrogate(f, grid, SVector{1,T}[], T[])
end

function update!(surr::CubicSplineSurrogate, I::CartesianIndex{1})
    p = surr.grid[I]
    pos = length(surr.p) + 1
    @inbounds for i in 1:length(surr.p)
        (p[1] <= surr.p[i][1]) && (pos = i; break)
    end
    insert!(surr.p, pos, p)
    insert!(surr.u, pos, surr.f(I))
    return surr
end

function suggest_point(surr::CubicSplineSurrogate{T}) where {T}
    p, u = spline_opt(reinterpret(T, surr.p), surr.u)
    return SVector{1,T}(p), T(u)
end

struct HermiteSplineSurrogate{D,T,F,RK} <: AbstractSurrogate{D,T}
    fg::F
    grid::Array{SVector{D,T},D}
    ugrid::Array{T,D}
    spl::NormalHermiteSplines.ElasticNormalSpline{D,T,RK}
end

function HermiteSplineSurrogate(fg, grid::Array{SVector{D,T},D}, kernel = RK_H1(one(T))) where {D,T}
    spl = NormalHermiteSplines.ElasticNormalSpline(first(grid), last(grid), maximum(size(grid)), kernel)
    HermiteSplineSurrogate(fg, grid, zeros(T, size(grid)), spl)
end

function update!(surr::HermiteSplineSurrogate{D,T}, I::CartesianIndex{D}) where {D,T}
    u, ∇u = surr.fg(I)
    @inbounds p = surr.grid[I]
    insert!(surr.spl, p, u)
    @inbounds for i in 1:D
        eᵢ = basisvector(SVector{D,T}, i)
        insert!(surr.spl, p, eᵢ, ∇u[i])
    end
    return surr
end

function suggest_point(surr::HermiteSplineSurrogate{D,T}) where {D,T}
    u, I = findmin(evaluate!(vec(surr.ugrid), surr.spl, vec(surr.grid)))
    @inbounds p = surr.grid[I]

    p₀, p₁ = first(surr.grid), last(surr.grid)
    Δp = maximum(Tuple(abs.(p₁ - p₀)))
    p, u = local_search(surr, p; maxiter = 100, initial_step = Δp/100)

    return (p, u)
end

####
#### Bounding box for multi-dimensional bisection search
####

struct BoundingBox{D,S,N}
    bounds::NTuple{D,NTuple{2,Int}}
    corners::SArray{S,CartesianIndex{D},D,N}
end
Base.show(io::IO, ::MIME"text/plain", box::BoundingBox{D}) where {D} = print(io, "$D-D BoundingBox with dimensions: " * join(box.bounds, " × "))

BoundingBox(widths::NTuple{D,Int}) where {D} = BoundingBox(tuple.(1, widths))
BoundingBox(bounds::NTuple{D,NTuple{2,Int}}) where {D} = BoundingBox(bounds, corners(bounds))

@generated function corners(bounds::NTuple{D,NTuple{2,Int}}) where {D}
    corners = Iterators.product([(true, false) for d in 1:D]...)
    S = Tuple{ntuple(d -> 2, D)...}
    vals = [:(CartesianIndex($(ntuple(d -> I[d] ? :(bounds[$d][1]) : :(bounds[$d][2]), D)...))) for I in corners]
    :(Base.@_inline_meta; SArray{$S, CartesianIndex{$D}, $D, $(2^D)}(tuple($(vals...))))
end

function opposite_corner(box::BoundingBox{D}, I::CartesianIndex{D}) where {D}
    @inbounds lo, hi = box.corners[begin], box.corners[end]
    return lo + hi - I
end

function bisect(box::BoundingBox{D}) where {D}
    _, i = findmax(ntuple(d -> abs(box.bounds[d][2] - box.bounds[d][1]), D))
    left_bounds = ntuple(D) do d
        i !== d ? box.bounds[d] : (box.bounds[i][1], (box.bounds[i][1] + box.bounds[i][2]) ÷ 2)
    end
    right_bounds = ntuple(D) do d
        i !== d ? box.bounds[d] : ((box.bounds[i][1] + box.bounds[i][2]) ÷ 2, box.bounds[i][2])
    end
    return BoundingBox(left_bounds), BoundingBox(right_bounds)
end

splittable(box::BoundingBox{D}) where {D} = any(ntuple(d -> abs(box.bounds[d][2] - box.bounds[d][1]) > 1, D))

####
#### Searching on a discrete grid using a surrogate function
####

struct DiscreteSurrogateSearcher{D, T}
    grid::Array{SVector{D,T},D}
    seen::Array{Bool,D}
    numeval::Base.RefValue{Int}
end
function DiscreteSurrogateSearcher(surr::AbstractSurrogate; mineval::Int, maxeval::Int)
    @assert mineval <= maxeval
    state = DiscreteSurrogateSearcher(surr.grid, fill(false, size(surr.grid)), Ref(0))
    initialize!(surr, state; mineval = mineval, maxeval = maxeval)
end

function initialize!(surr::AbstractSurrogate{D}, state::DiscreteSurrogateSearcher{D}; mineval::Int, maxeval::Int) where {D}
    # Evaluate at least `mineval` points by repeatedly bisecting the grid in a breadth-first manner
    box = BoundingBox(size(state.grid))
    for depth in 1:mineval # should never reach `mineval` depth, this is just to ensure the loop terminates in case `mineval` is greater than the number of gridpoints
        initialize!(surr, state, box, depth; mineval = mineval, maxeval = maxeval)
        state.numeval[] >= mineval && break
    end
    return state
end

function initialize!(surr::AbstractSurrogate{D}, state::DiscreteSurrogateSearcher{D}, box::BoundingBox{D}, depth::Int; mineval::Int, maxeval::Int) where {D}
    depth <= 0 && return state
    evaluate_box!(surr, state, box; maxeval = maxeval)
    state.numeval[] ≥ mineval && return state
    left, right = bisect(box)
    initialize!(surr, state, left, depth-1; mineval = mineval, maxeval = maxeval)
    initialize!(surr, state, right, depth-1; mineval = mineval, maxeval = maxeval)
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
        surr::AbstractSurrogate{D,T},
        state::DiscreteSurrogateSearcher{D,T};
        maxeval::Int,
    ) where {D,T}

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
        evaluate_box!(surr, state, box; maxeval = maxeval)
        x, u = suggest_point(surr)
        if state.numeval[] ≥ maxeval || converged(state, box)
            return (x, u)
        end
    end
end

# Update observed evaluations, returning true if converged
function minimal_bounding_box(
        state::DiscreteSurrogateSearcher{D,T},
        x::SVector{D,T},
    ) where {D,T}

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

function evaluate_box!(surr::AbstractSurrogate{D}, state::DiscreteSurrogateSearcher{D}, box::BoundingBox{D}; maxeval::Int) where {D}
    @inbounds for I in box.corners
        is_evaluated(state, box) && break # box sufficiently evaluated
        update!(surr, state, I; maxeval = maxeval) && break # update surrogate, breaking if max evals reached
    end
    return state
end

function is_evaluated(state::DiscreteSurrogateSearcher{D}, box::BoundingBox{D}) where {D}
    # Box is considered sufficiently evaluated when all of the corners have been evaluted
    count(I -> @inbounds(state.seen[I]), box.corners) >= 2^D
end

function converged(::DiscreteSurrogateSearcher{D}, box::BoundingBox{D}) where {D}
    # Convergence is defined as: bounding box has at least one side of length <= 1
    any(ntuple(d -> abs(box.bounds[d][2] - box.bounds[d][1]) <= 1, D))
end

function contains(state::DiscreteSurrogateSearcher{D,T}, box::BoundingBox{D}, x::SVector{D,T}) where {D,T}
    @inbounds bottomleft = state.grid[box.corners[begin]]
    @inbounds topright = state.grid[box.corners[end]]
    all(bottomleft .<= x .<= topright)
end

####
#### Local optimization using surrogate functions
####

function local_search(
        surr::HermiteSplineSurrogate{D,T},
        x₀::SVector{D,T},
        state::Union{Nothing, DiscreteSurrogateSearcher{D,T}} = nothing;
        maxiter::Int = 100,
        maxeval::Int = maxiter,
        xtol_rel = 1e-6,
        xtol_abs = 1e-4,
        initial_step = 0.1,
    ) where {D,T}

    if state !== nothing
        box = BoundingBox(size(surr.grid))
        for I in box.corners
            update!(surr, state, I; maxeval = maxeval) # initialize surrogate with domain corners
        end
    end

    x, xlo, xhi = x₀, first(surr.grid), last(surr.grid)
    Δx = SVector{D,T}(ntuple(d -> T(Inf), D))
    opt = ADAM{D,T}(initial_step)

    for _ in 1:maxiter
        if state !== nothing
            # Find nearest gridpoint to `x` and update surrogate
            I = nearest_gridpoint(state, x)
            update!(surr, state, I; maxeval = maxeval)
        end

        # Perform gradient descent step using surrogate function
        ∇u = NormalHermiteSplines.evaluate_gradient(surr.spl, x)
        Δx, opt = update(∇u, opt)
        x = @. clamp(x - Δx, xlo, xhi)

        # Check for convergence
        maximum(abs.(Δx)) <= max(T(xtol_abs), T(xtol_rel) * maximum(abs.(x))) && break
    end

    u = NormalHermiteSplines.evaluate(surr.spl, x)

    return (x, u)
end

function nearest_gridpoint(state::DiscreteSurrogateSearcher{D,T}, x::SVector{D,T}) where {D,T}
    @inbounds xlo, xhi = state.grid[begin], state.grid[end]
    @inbounds Ilo, Ihi = CartesianIndices(state.grid)[begin], CartesianIndices(state.grid)[end]
    lo, hi = SVector(Tuple(Ilo)), SVector(Tuple(Ihi))
    i = @. clamp(round(Int, (x - xlo) * (hi - lo) / (xhi - xlo) + lo), lo, hi)
    return CartesianIndex(Tuple(i))
end

function test_nearest_gridpoint()
    grid = meshgrid(SVector, range(-3, 4, length=4), range(1, 5, length=7))
    I = nearest_gridpoint(grid, SVector(10.0, -10.0))
    I, grid[I]
end

####
#### Global optimization for NNLS problem
####

struct NNLSDiscreteSurrogateSearch{D, T, TA <: AbstractArray{T}, TdA <: AbstractArray{T}, Tb <: AbstractVector{T}, W}
    As::TA
    ∇As::TdA
    αs::Array{SVector{D,T},D}
    b::Tb
    u::Array{T,D}
    ∂Ax⁺::Vector{T}
    Ax⁺b::Vector{T}
    nnls_work::W
end

function NNLSDiscreteSurrogateSearch(
        As::AbstractArray{T},  # size(As)  = (M, N, P1..., PD)
        ∇As::AbstractArray{T}, # size(∇As) = (M, N, D, P1..., PD)
        αs::NTuple{D},         # size(αs)  = (P1..., PD)
        b::AbstractVector{T},  # size(b)   = (M,)
    ) where {D,T}
    M, N = size(As, 1), size(As, 2)
    @assert ndims(As) == 2 + D && ndims(∇As) == 3 + D # ∇As has extra dimension for parameter gradients
    @assert size(∇As)[1:3] == (M, N, D) # matrix dimensions must match, and gradient dimension must equal number of parameters
    @assert size(As)[3:end] == size(∇As)[4:end] == length.(αs) # dimension size must match parameters lengths
    @assert size(b) == (M,)

    αs   = meshgrid(SVector{D,T}, αs...)
    u    = zeros(T, size(αs))
    ∂Ax⁺ = zeros(T, M)
    Ax⁺b = zeros(T, M)
    nnls_work = lsqnonneg_work(zeros(T, M, N), zeros(T, M))
    NNLSDiscreteSurrogateSearch(As, ∇As, αs, b, u, ∂Ax⁺, Ax⁺b, nnls_work)
end

load!(prob::NNLSDiscreteSurrogateSearch{D,T}, b::AbstractVector{T}) where {D,T} = copyto!(prob.b, b)

function loss!(prob::NNLSDiscreteSurrogateSearch{D,T}, I::CartesianIndex{D}) where {D,T}
    @unpack As, b, nnls_work = prob
    solve!(nnls_work, uview(As, :, :, I), b)
    ℓ = chi2(nnls_work)
    u = LEGACY[] ? ℓ : log(max(ℓ, eps(T))) # loss capped at eps(T) from below to avoid log(0) error
    return u
end

function ∇loss!(prob::NNLSDiscreteSurrogateSearch{D,T}, I::CartesianIndex{D}) where {D,T}
    @unpack As, ∇As, b, ∂Ax⁺, Ax⁺b, nnls_work = prob
    ℓ = chi2(nnls_work)
    ℓ <= eps(T) && return zero(SVector{D,T}) # loss capped at eps(T) from below; return zero gradient
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
        ∂u = LEGACY[] ? ∂ℓ : ∂ℓ / ℓ
        return ∂u
    end
    return SVector{D,T}(∇u)
end

function CubicSplineSurrogate(prob::NNLSDiscreteSurrogateSearch{1,T}) where {T}
    f(I) = loss!(prob, I)
    CubicSplineSurrogate(f, prob.αs, SVector{1,T}[], T[])
end

function HermiteSplineSurrogate(prob::NNLSDiscreteSurrogateSearch{D,T}, kernel = RK_H1(one(T))) where {D,T}
    function fg(I)
        u = loss!(prob, I)
        ∇u = ∇loss!(prob, I)
        return u, ∇u
    end
    return HermiteSplineSurrogate(fg, prob.αs, kernel)
end

function surrogate_spline_opt(
        prob::NNLSDiscreteSurrogateSearch{D},
        surr::AbstractSurrogate{D};
        mineval::Int = min(2^D, length(prob.αs)),
        maxeval::Int = length(prob.αs),
    ) where {D}
    state = DiscreteSurrogateSearcher(surr; mineval = mineval, maxeval = maxeval)
    bisection_search(surr, state; maxeval = maxeval)
end

function spline_opt(
        spl::NormalSpline{D,T},
        prob::NNLSDiscreteSurrogateSearch{D,T};
        # alg = :LN_COBYLA,        # local, gradient-free, linear approximation of objective
        # alg = :LN_BOBYQA,        # local, gradient-free, quadratic approximation of objective
        # alg = :GN_ORIG_DIRECT_L, # global, gradient-free, systematically divides search space into smaller hyper-rectangles via a branch-and-bound technique, systematic division of the search domain into smaller and smaller hyperrectangles, "more biased towards local search"
        # alg = :GN_AGS,           # global, gradient-free, employs the Hilbert curve to reduce the source problem to the univariate one.
        # alg = :GD_STOGO,         # global, with-gradient, systematically divides search space into smaller hyper-rectangles via a branch-and-bound technique, and searching them by a gradient-based local-search algorithm (a BFGS variant)
        alg = :LD_SLSQP,         # local, with-gradient, "Sequential Least-Squares Quadratic Programming"; uses dense-matrix methods (ordinary BFGS, not low-storage BFGS)
    ) where {D,T}

    evaluate!(prob.u, spl, prob.αs)
    _, i = findmin(prob.u)
    α₀ = prob.αs[i]

    opt = NLopt.Opt(alg, D)
    opt.lower_bounds = Float64[prob.αs[begin]...]
    opt.upper_bounds = Float64[prob.αs[end]...]
    opt.xtol_rel = 0.001
    opt.min_objective = function (x, g)
        if length(g) > 0
            @inbounds g[1] = Float64(evaluate_derivative(spl, x[1]))
        end
        @inbounds Float64(evaluate(spl, x[1]))
    end
    minf, minx, ret = NLopt.optimize(opt, Vector{Float64}(α₀))
    return (SVector{D,T}(minx), T(minf))
end

function mock_surrogate_search_problem(
        b::AbstractVector,
        opts::T2mapOptions,
        ::Val{D},
        ::Val{ETL};
        alphas = range(50, 180, length = opts.nRefAngles),
        betas = range(50, 180, length = opts.nRefAngles),
    ) where {D,ETL}

    # Mock CPMG image
    @assert opts.nTE == ETL
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
function mock_surrogate_search_problem(::Val{D}, ::Val{ETL}, opts = mock_t2map_opts(; MatrixSize = (1,1,1), nTE = ETL); kwargs...) where {D,ETL}
    b = vec(mock_image(opts))
    mock_surrogate_search_problem(b, opts, Val(D), Val(ETL); kwargs...)
end
function mock_surrogate_search_problem(b::AbstractVector, opts::T2mapOptions, ::Val{D}; kwargs...) where {D}
    @assert length(b) == opts.nTE
    mock_surrogate_search_problem(b, opts, Val(D), Val(length(b)); kwargs...)
end
