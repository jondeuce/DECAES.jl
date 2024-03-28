####
#### Unregularized NNLS problem
####

struct NNLSProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W}
    A::TA
    b::Tb
    m::Int
    n::Int
    nnls_work::W
end
function NNLSProblem(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    nnls_work = NNLS.NNLSWorkspace(A, b)
    return NNLSProblem(A, b, m, n, nnls_work)
end

# Solve NNLS problem
solve!(work::NNLSProblem, A = work.A, b = work.b) = NNLS.nnls!(work.nnls_work, A, b)

@inline solution(work::NNLSProblem) = NNLS.solution(work.nnls_work)
@inline ncomponents(work::NNLSProblem) = NNLS.ncomponents(work.nnls_work)
@inline resnorm(work::NNLSProblem) = NNLS.residualnorm(work.nnls_work)
@inline resnorm_sq(work::NNLSProblem) = resnorm(work)^2

@doc raw"""
    lsqnonneg(A::AbstractMatrix, b::AbstractVector)

Compute the nonnegative least-squares (NNLS) solution ``X`` of the problem:

```math
X = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2.
```

# Arguments

  - `A::AbstractMatrix`: Left hand side matrix acting on `x`
  - `b::AbstractVector`: Right hand side vector

# Outputs

  - `X::AbstractVector`: NNLS solution
"""
lsqnonneg(A::AbstractMatrix, b::AbstractVector) = lsqnonneg!(lsqnonneg_work(A, b))
lsqnonneg_work(A::AbstractMatrix, b::AbstractVector) = NNLSProblem(A, b)
lsqnonneg!(work::NNLSProblem) = solve!(work)
lsqnonneg!(work::NNLSProblem{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T} = solve!(work, A, b)

####
#### Lazy wrappers for LHS matrix and RHS vector for augmented Tikhonov-regularized NNLS problems
####

struct PaddedVector{T, Tb <: AbstractVector{T}} <: AbstractVector{T}
    b::Tb
    pad::Int
end
Base.size(x::PaddedVector) = (length(x.b) + x.pad,)
Base.parent(x::PaddedVector) = x.b

function Base.copyto!(y::AbstractVector{T}, x::PaddedVector{T}) where {T}
    @assert size(x) == size(y)
    (; b, pad) = x
    m = length(b)
    @simd for i in 1:m
        y[i] = b[i]
    end
    @simd for i in m+1:m+pad
        y[i] = zero(T)
    end
    return y
end

struct TikhonovPaddedMatrix{T, TA <: AbstractMatrix{T}} <: AbstractMatrix{T}
    A::TA
    μ::Base.RefValue{T}
end
TikhonovPaddedMatrix(A::AbstractMatrix, μ::Real) = TikhonovPaddedMatrix(A, Ref(μ))
Base.size(P::TikhonovPaddedMatrix) = ((m, n) = size(P.A); return (m + n, n))
Base.parent(P::TikhonovPaddedMatrix) = P.A
mu(P::TikhonovPaddedMatrix) = P.μ[]
mu!(P::TikhonovPaddedMatrix, μ::Real) = P.μ[] = μ

function Base.copyto!(B::AbstractMatrix{T}, P::TikhonovPaddedMatrix{T}) where {T}
    @assert size(P) == size(B)
    A, μ = parent(P), mu(P)
    m, n = size(A)
    @inbounds for j in 1:n
        @simd for i in 1:m
            B[i, j] = A[i, j]
        end
        @simd for i in m+1:m+n
            B[i, j] = ifelse(i == m + j, μ, zero(T))
        end
    end
    return B
end

####
#### Tikhonov regularized NNLS problem
####

struct NNLSTikhonovRegProblem{
    T,
    TA <: AbstractMatrix{T},
    Tb <: AbstractVector{T},
    W <: NNLSProblem{T, TikhonovPaddedMatrix{T, TA}, PaddedVector{T, Tb}},
    B,
}
    A::TA
    b::Tb
    m::Int
    n::Int
    nnls_prob::W
    buffers::B
end
function NNLSTikhonovRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}, μ::Real = T(NaN)) where {T}
    m, n = size(A)
    nnls_prob = NNLSProblem(TikhonovPaddedMatrix(A, μ), PaddedVector(b, n))
    buffers = (x = zeros(T, n), y = zeros(T, n))
    return NNLSTikhonovRegProblem(A, b, m, n, nnls_prob, buffers)
end

@doc raw"""
    lsqnonneg_tikh(A::AbstractMatrix, b::AbstractVector, μ::Real)

Compute the Tikhonov-regularized nonnegative least-squares (NNLS) solution ``X_{\mu}`` of the problem:

```math
X_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu^2 ||x||_2^2.
```

# Arguments

  - `A::AbstractMatrix`: Left hand side matrix acting on `x`
  - `b::AbstractVector`: Right hand side vector
  - `μ::Real`: Regularization parameter

# Outputs

  - `X::AbstractVector`: NNLS solution
"""
lsqnonneg_tikh(A::AbstractMatrix, b::AbstractVector, μ::Real) = lsqnonneg_tikh!(lsqnonneg_tikh_work(A, b), μ)
lsqnonneg_tikh_work(A::AbstractMatrix, b::AbstractVector) = NNLSTikhonovRegProblem(A, b)
lsqnonneg_tikh!(work::NNLSTikhonovRegProblem, μ::Real) = solve!(work, μ)

mu(work::NNLSTikhonovRegProblem) = mu(work.nnls_prob.A)
mu!(work::NNLSTikhonovRegProblem, μ::Real) = mu!(work.nnls_prob.A, μ)

function solve!(work::NNLSTikhonovRegProblem, μ::Real)
    # Set regularization parameter and solve NNLS problem
    mu!(work, μ)
    solve!(work.nnls_prob)
    return solution(work)
end

@inline solution(work::NNLSTikhonovRegProblem) = NNLS.solution(work.nnls_prob.nnls_work)
@inline ncomponents(work::NNLSTikhonovRegProblem) = NNLS.ncomponents(work.nnls_prob.nnls_work)

@inline loss(work::NNLSTikhonovRegProblem) = NNLS.residualnorm(work.nnls_prob.nnls_work)^2

regnorm(work::NNLSTikhonovRegProblem) = mu(work)^2 * seminorm_sq(work) # μ²||x||²
∇regnorm(work::NNLSTikhonovRegProblem) = 2 * mu(work) * seminorm_sq(work) + mu(work)^2 * ∇seminorm_sq(work) # d/dμ [μ²||x||²] = 2μ||x||² + μ² d/dμ [||x||²]

resnorm(work::NNLSTikhonovRegProblem) = √(resnorm_sq(work)) # ||Ax-b||
resnorm_sq(work::NNLSTikhonovRegProblem) = max(loss(work) - regnorm(work), 0) # ||Ax-b||²
∇resnorm_sq(work::NNLSTikhonovRegProblem, ∇ = gradient_temps(work)) = 4 * ∇.μ^3 * ∇.xᵀB⁻¹x # d/dμ [||Ax-b||²]
∇²resnorm_sq(work::NNLSTikhonovRegProblem, ∇² = hessian_temps(work)) = 12 * ∇².μ^2 * ∇².xᵀB⁻¹x - 24 * ∇².μ^4 * ∇².xᵀB⁻ᵀB⁻¹x # d²/dμ² [||Ax-b||²]

seminorm(work::NNLSTikhonovRegProblem) = √(seminorm_sq(work)) # ||x||
seminorm_sq(work::NNLSTikhonovRegProblem) = GC.@preserve work sum(abs2, NNLS.positive_solution(work.nnls_prob.nnls_work)) # ||x||²
∇seminorm_sq(work::NNLSTikhonovRegProblem, ∇ = gradient_temps(work)) = -4 * ∇.μ * ∇.xᵀB⁻¹x # d/dμ [||x||²]
∇²seminorm_sq(work::NNLSTikhonovRegProblem, ∇² = hessian_temps(work)) = -4 * ∇².xᵀB⁻¹x + 24 * ∇².μ^2 * ∇².xᵀB⁻ᵀB⁻¹x # d²/dμ² [||x||²]

solution_gradnorm(work::NNLSTikhonovRegProblem, ∇² = hessian_temps(work)) = √(solution_gradnorm_sq(work, ∇²)) # ||dx/dμ|| = ||-2μ * B⁻¹x|| = 2μ * ||B⁻¹x||
solution_gradnorm_sq(work::NNLSTikhonovRegProblem, ∇² = hessian_temps(work)) = 4 * ∇².μ^2 * ∇².xᵀB⁻ᵀB⁻¹x # ||dx/dμ||² = ||-2μ * B⁻¹x||² = 4μ² * xᵀB⁻ᵀB⁻¹x

# L-curve: (ξ(μ), η(μ)) = (||Ax-b||^2, ||x||^2)
curvature(::typeof(identity), work::NNLSTikhonovRegProblem, ∇ = gradient_temps(work)) = inv(2 * ∇.xᵀB⁻¹x * √(1 + ∇.μ^4)^3)

# L-curve: (ξ̄(μ), η̄(μ)) = (log||Ax-b||^2, log||x||^2)
function curvature(::typeof(log), work::NNLSTikhonovRegProblem, ∇ = gradient_temps(work))
    ℓ² = loss(work) # ℓ² = ||Ax-b||^2 + μ²||x||^2 = ξ² + μ²η²
    ξ² = resnorm_sq(work)
    η² = seminorm_sq(work)
    ξ⁴, η⁴ = ξ²^2, η²^2
    C̄ = ξ² * η² * (ξ² * η² - (2 * ∇.xᵀB⁻¹x) * ∇.μ^2 * ℓ²) / (2 * ∇.xᵀB⁻¹x * √(ξ⁴ + ∇.μ^4 * η⁴)^3)
    return C̄
end

function gradient_temps(work::NNLSTikhonovRegProblem{T}) where {T}
    GC.@preserve work begin
        (; nnls_work) = work.nnls_prob
        B = cholesky!(NNLS.NormalEquation(), nnls_work) # B = A'A + μ²I = U'U
        x₊ = NNLS.positive_solution(nnls_work)
        tmp = uview(work.buffers.y, 1:length(x₊))

        μ = mu(work)
        copyto!(tmp, x₊)
        NNLS.solve_triangular_system!(tmp, B, Val(true)) # tmp = U'\x
        xᵀB⁻¹x = sum(abs2, tmp) # x'B\x = x'(U'U)\x = ||U'\x||^2

        return (; μ, xᵀB⁻¹x)
    end
end

function hessian_temps(work::NNLSTikhonovRegProblem{T}) where {T}
    GC.@preserve work begin
        (; nnls_work) = work.nnls_prob
        B = cholesky!(NNLS.NormalEquation(), nnls_work) # B = A'A + μ²I = U'U
        x₊ = NNLS.positive_solution(nnls_work)
        tmp = uview(work.buffers.y, 1:length(x₊))

        μ = mu(work)
        copyto!(tmp, x₊)
        NNLS.solve_triangular_system!(tmp, B, Val(true)) # tmp = U'\x
        xᵀB⁻¹x = sum(abs2, tmp) # x'B\x = x'(U'U)\x = ||U'\x||^2

        NNLS.solve_triangular_system!(tmp, B, Val(false)) # tmp = U\(U'\x) = (U'U)\x
        xᵀB⁻ᵀB⁻¹x = sum(abs2, tmp) # x'B'\B\x = ||B\x||^2 = ||(U'U)\x||^2

        return (; μ, xᵀB⁻¹x, xᵀB⁻ᵀB⁻¹x)
    end
end

function chi2_relerr!(work::NNLSTikhonovRegProblem, χ²target, logμ, ∇logμ = nothing)
    # NOTE: assumes `solve!(work, μ)` has been called and that the solution is ready
    μ = exp(logμ)
    res² = resnorm_sq(work)
    relerr = log(res² / χ²target) # better behaved than res² / χ²target - 1 for large res²?
    if ∇logμ !== nothing && length(∇logμ) > 0
        ∂res²_∂μ = ∇resnorm_sq(work)
        ∂relerr_∂logμ = μ * ∂res²_∂μ / res²
        @inbounds ∇logμ[1] = ∂relerr_∂logμ
    end
    return relerr
end
chi2_relerr⁻¹(χ²target, relerr) = χ²target * exp(relerr)

# Helper struct which wraps `N` caches of `NNLSTikhonovRegProblem` workspaces.
# Useful for optimization problems where the last function call may not be
# the optimium, but perhaps it was one or two calls previous and is still in the
# `NNLSTikhonovRegProblemCache` and a recomputation can be avoided.
struct NNLSTikhonovRegProblemCache{N, W <: NTuple{N}}
    cache::W
    idx::Base.RefValue{Int}
end
function NNLSTikhonovRegProblemCache(A::AbstractMatrix{T}, b::AbstractVector{T}, ::Val{N} = Val(5)) where {T, N}
    cache = ntuple(_ -> NNLSTikhonovRegProblem(A, b), N)
    idx = Ref(1)
    return NNLSTikhonovRegProblemCache(cache, idx)
end
increment_cache_index!(work::NNLSTikhonovRegProblemCache{N}) where {N} = (work.idx[] = mod1(work.idx[] + 1, N))
set_cache_index!(work::NNLSTikhonovRegProblemCache{N}, i) where {N} = (work.idx[] = mod1(i, N))
reset_cache!(work::NNLSTikhonovRegProblemCache{N}) where {N} = foreach(w -> mu!(w, NaN), work.cache)
get_cache(work::NNLSTikhonovRegProblemCache) = work.cache[work.idx[]]

function solve!(work::NNLSTikhonovRegProblemCache, μ::Real)
    i = findfirst(w -> μ == mu(w), work.cache)
    if i === nothing
        increment_cache_index!(work)
        solve!(get_cache(work), μ)
    else
        set_cache_index!(work, i)
    end
    return solution(get_cache(work))
end

####
#### Chi2 method for choosing the Tikhonov regularization parameter
####

struct NNLSChi2RegProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W1, W2}
    A::TA
    b::Tb
    m::Int
    n::Int
    nnls_prob::W1
    nnls_prob_smooth_cache::W2
end
function NNLSChi2RegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    nnls_prob = NNLSProblem(A, b)
    nnls_prob_smooth_cache = NNLSTikhonovRegProblemCache(A, b)
    return NNLSChi2RegProblem(A, b, m, n, nnls_prob, nnls_prob_smooth_cache)
end

@inline solution(work::NNLSChi2RegProblem) = solution(get_cache(work.nnls_prob_smooth_cache))
@inline ncomponents(work::NNLSChi2RegProblem) = ncomponents(get_cache(work.nnls_prob_smooth_cache))

@doc raw"""
    lsqnonneg_chi2(A::AbstractMatrix, b::AbstractVector, chi2_target::Real)

Compute the Tikhonov-regularized nonnegative least-squares (NNLS) solution ``X_{\mu}`` of the problem:

```math
X_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu^2 ||x||_2^2
```

where ``\mu`` is determined by solving:

```math
\chi^2(\mu) = \frac{||AX_{\mu} - b||_2^2}{||AX_{0} - b||_2^2} = \chi^2_{\mathrm{target}}.
```

That is, ``\mu`` is chosen such that the squared residual norm of the regularized problem is `chi2_target`
times larger than the squared residual norm of the unregularized problem.

# Arguments

  - `A::AbstractMatrix`: Decay basis matrix
  - `b::AbstractVector`: Decay curve data
  - `chi2_target::Real`: Target ``\chi^2(\mu)``; typically a small value, e.g. 1.02 representing a 2% increase

# Outputs

  - `X::AbstractVector`: Regularized NNLS solution
  - `mu::Real`: Resulting regularization parameter ``\mu``
  - `chi2::Real`: Resulting ``\chi^2(\mu)``, which should be approximately equal to `chi2_target`
"""
function lsqnonneg_chi2(A::AbstractMatrix, b::AbstractVector, chi2_target::Real, args...; kwargs...)
    work = lsqnonneg_chi2_work(A, b)
    return lsqnonneg_chi2!(work, chi2_target, args...; kwargs...)
end
lsqnonneg_chi2_work(A::AbstractMatrix, b::AbstractVector) = NNLSChi2RegProblem(A, b)

function lsqnonneg_chi2!(work::NNLSChi2RegProblem{T}, chi2_target::T, legacy::Bool = false; method::Symbol = legacy ? :legacy : :bisect) where {T}
    # Non-regularized solution
    solve!(work.nnls_prob)
    x_unreg = solution(work.nnls_prob)
    res²_min = resnorm_sq(work.nnls_prob)

    if res²_min == 0 || ncomponents(work.nnls_prob) == 0
        # 1. If non-regularized solution is exact, the only solution to res²(μ) = chi2_target * res²_min = 0 is μ = 0, since res²(μ) > 0 for all μ > 0.
        # 2. If non-regularized solution is zero, any value of μ > 0 also results in x(μ) = 0, and so res²(μ) = chi2_target * res²_min has either no solutions if chi2_target > 1, or infinitely many solutions if chi2_target = 1; choose μ = 0 and chi2_target = 1.
        x_final = x_unreg
        return (; x = x_final, mu = zero(T), chi2 = one(T))
    end

    # Prepare to solve
    res²_target = chi2_target * res²_min
    reset_cache!(work.nnls_prob_smooth_cache)

    if method === :legacy
        # Use the legacy algorithm: double μ starting from an initial guess, then interpolate the root using a cubic spline fit
        mu_final, res²_final = chi2_search_from_minimum(res²_min, chi2_target; legacy) do μ
            μ == 0 && return res²_min
            solve!(work.nnls_prob_smooth_cache, μ)
            return resnorm_sq(get_cache(work.nnls_prob_smooth_cache))
        end
        if mu_final == 0
            x_final = x_unreg
        else
            x_final = solve!(work.nnls_prob_smooth_cache, mu_final)
        end

    elseif method === :bisect
        f = function (logμ)
            solve!(work.nnls_prob_smooth_cache, exp(logμ))
            return chi2_relerr!(get_cache(work.nnls_prob_smooth_cache), res²_target, logμ)
        end

        # Find bracketing interval containing root, then perform bisection search with slightly higher tolerance to not waste f evals
        a, b, fa, fb = bracket_root_monotonic(f, T(-4.0), T(1.0); dilate = T(1.5), mono = +1, maxiters = 6) # maxiters = 100

        if fa * fb < 0
            # Bracketing interval found
            a, fa, c, fc, b, fb = bisect_root(f, a, b, fa, fb; xatol = T(0.05), xrtol = T(0.0), ftol = (chi2_target - 1) / 100) # maxiters = 100

            # Root of secant line through `(a, fa), (b, fb)` or `(c, fc), (b, fb)` to improve bisection accuracy
            tmp = fa * fc < 0 ? root_real_linear(a, c, fa, fc) : fc * fb < 0 ? root_real_linear(c, b, fc, fb) : T(NaN)
            d, fd = isnan(tmp) ? (c, fc) : (tmp, f(tmp))

            # Return regularization parameter with lowest abs(relerr)
            logmu_final, relerr_final = abs(fd) < abs(fc) ? (d, fd) : (c, fc)
        else
            # No bracketing interval found; choose point with smallest value of f (note: this branch should never be reached)
            logmu_final, relerr_final = !isfinite(fa) ? (b, fb) : !isfinite(fb) ? (a, fa) : abs(fa) < abs(fb) ? (a, fa) : (b, fb)
        end

        if isfinite(relerr_final)
            mu_final, res²_final = exp(logmu_final), chi2_relerr⁻¹(res²_target, relerr_final)
            x_final = solve!(work.nnls_prob_smooth_cache, mu_final)
        else
            x_final, mu_final, res²_final = x_unreg, zero(T), one(T)
        end

    elseif method === :brent
        f = function (logμ)
            solve!(work.nnls_prob_smooth_cache, exp(logμ))
            return chi2_relerr!(get_cache(work.nnls_prob_smooth_cache), res²_target, logμ)
        end

        # Find bracketing interval containing root
        a, b, fa, fb = bracket_root_monotonic(f, T(-4.0), T(1.0); dilate = T(1.5), mono = +1, maxiters = 100)

        if fa * fb < 0
            # Find root using Brent's method
            logmu_final, relerr_final = brent_root(f, a, b, fa, fb; xatol = T(0.0), xrtol = T(0.0), ftol = (chi2_target - 1) / 1000, maxiters = 100)
        else
            # No bracketing interval found; choose point with smallest value of f (note: this branch should never be reached)
            logmu_final, relerr_final = !isfinite(fa) ? (b, fb) : !isfinite(fb) ? (a, fa) : abs(fa) < abs(fb) ? (a, fa) : (b, fb)
        end

        if isfinite(relerr_final)
            mu_final, res²_final = exp(logmu_final), chi2_relerr⁻¹(res²_target, relerr_final)
            x_final = solve!(work.nnls_prob_smooth_cache, mu_final)
        else
            x_final, mu_final, res²_final = x_unreg, zero(T), one(T)
        end
    else
        error("Unknown root-finding method: :$method")
    end

    return (; x = x_final, mu = mu_final, chi2 = res²_final / res²_min)
end

function chi2_search_from_minimum(f, res²min::T, χ²fact::T, μmin::T = T(1e-3), μfact = T(2.0); legacy = false) where {T}
    # Minimize energy of spectrum; loop to find largest μ that keeps chi-squared in desired range
    μ_cache = T[zero(T)]
    res²_cache = T[res²min]
    μnew = μmin
    while true
        # Cache function value at μ = μnew
        res²new = f(μnew)
        push!(μ_cache, μnew)
        push!(res²_cache, res²new)

        # Break when χ²fact reached, else increase regularization
        (res²new >= χ²fact * res²min) && break
        μnew *= μfact
    end

    # Solve res²(μ) = χ²fact * res²min using a spline fitting root finding method
    if legacy
        # Legacy algorithm fits spline to all (μ, res²) values observed, including for μ=0.
        # This poses several problems:
        #   1) while unlikely, it is possible for the spline to return a negative regularization parameter
        #   2) the μ values are exponentially spaced, leading to poorly conditioned splines
        μ = spline_root_legacy(μ_cache, res²_cache, χ²fact * res²min)
    else
        if length(μ_cache) == 2
            # Solution is contained in [0,μmin]; `spline_root` with two points performs root finding via simple linear interpolation
            μ = spline_root(μ_cache, res²_cache, χ²fact * res²min; deg_spline = 1)
            μ = isnan(μ) ? μmin : μ
        else
            # Perform spline fit on log-log scale on data with μ > 0. This solves the above problems with the legacy algorithm:
            #   1) Root is found in terms of logμ, guaranteeing μ > 0
            #   2) logμ is linearly spaced, leading to well-conditioned splines
            logμ = spline_root(log.(μ_cache[2:end]), log.(res²_cache[2:end]), log(χ²fact * res²min); deg_spline = 1)
            μ = isnan(logμ) ? μmin : exp(logμ)
        end
    end

    # Compute the final regularized solution
    res² = f(μ)

    return μ, res²
end

####
#### Morozov discrepency principle (MDP) method for choosing the Tikhonov regularization parameter
####

struct NNLSMDPRegProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W1, W2}
    A::TA
    b::Tb
    m::Int
    n::Int
    nnls_prob::W1
    nnls_prob_smooth_cache::W2
end
function NNLSMDPRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    nnls_prob = NNLSProblem(A, b)
    nnls_prob_smooth_cache = NNLSTikhonovRegProblemCache(A, b)
    return NNLSMDPRegProblem(A, b, m, n, nnls_prob, nnls_prob_smooth_cache)
end

@inline solution(work::NNLSMDPRegProblem) = solution(get_cache(work.nnls_prob_smooth_cache))
@inline ncomponents(work::NNLSMDPRegProblem) = ncomponents(get_cache(work.nnls_prob_smooth_cache))

@doc raw"""
    lsqnonneg_mdp(A::AbstractMatrix, b::AbstractVector, δ::Real)

Compute the Tikhonov-regularized nonnegative least-squares (NNLS) solution ``X_{\mu}`` of the problem:

```math
X_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu^2 ||x||_2^2
```

where ``\mu`` is chosen using Morozov's Discrepency Principle (MDP)[1,2]:

```math
\mu = \operatorname{sup}\; \left\{ \nu \ge 0 : ||AX_{\nu} - b|| \le \delta \right\}.
```

That is, ``\mu`` is maximized subject to the constraint that the residual norm of the regularized problem is at most ``\delta``[1].

# Arguments

  - `A::AbstractMatrix`: Decay basis matrix
  - `b::AbstractVector`: Decay curve data
  - `δ::Real`: Upper bound on regularized residual norm

# Outputs

  - `X::AbstractVector`: Regularized NNLS solution
  - `mu::Real`: Resulting regularization parameter ``\mu``
  - `chi2::Real`: Resulting increase in residual norm relative to the unregularized ``\mu = 0`` solution

# References

  1. Morozov VA. Methods for Solving Incorrectly Posed Problems. Springer Science & Business Media, 2012.
  2. Clason C, Kaltenbacher B, Resmerita E. Regularization of Ill-Posed Problems with Non-negative Solutions. In: Bauschke HH, Burachik RS, Luke DR (eds) Splitting Algorithms, Modern Operator Theory, and Applications. Cham: Springer International Publishing, pp. 113–135.
"""
function lsqnonneg_mdp(A::AbstractMatrix, b::AbstractVector, δ::Real, args...; kwargs...)
    work = lsqnonneg_mdp_work(A, b)
    return lsqnonneg_mdp!(work, δ, args...; kwargs...)
end
lsqnonneg_mdp_work(A::AbstractMatrix, b::AbstractVector) = NNLSMDPRegProblem(A, b)

function lsqnonneg_mdp!(work::NNLSMDPRegProblem{T}, δ::T) where {T}
    # Non-regularized solution
    solve!(work.nnls_prob)
    x_unreg = solution(work.nnls_prob)
    res²_min = resnorm_sq(work.nnls_prob)

    #TODO: throw error if δ > ||b||, since ||A * x(μ) - b|| <= ||b|| when A, x, b are componentwise nonnegative, or return... what?
    # if δ >= norm(work.nnls_prob.b)
    #     error("δ = $δ is greater than the norm of the data vector ||b|| = $(norm(work.nnls_prob.b))")
    # end
    if δ <= res²_min
        x_final = x_unreg
        return (; x = x_final, mu = zero(T), chi2 = one(T))
    end

    # Prepare to solve
    reset_cache!(work.nnls_prob_smooth_cache)

    function f(logμ)
        solve!(work.nnls_prob_smooth_cache, exp(logμ))
        return resnorm_sq(get_cache(work.nnls_prob_smooth_cache)) - δ^2
    end

    # Find bracketing interval containing root
    a, b, fa, fb = bracket_root_monotonic(f, T(-4.0), T(1.0); dilate = T(1.5), mono = +1, maxiters = 100)

    if fa * fb < 0
        # Find root using Brent's method
        logmu_final, err_final = brent_root(f, a, b, fa, fb; xatol = T(0.0), xrtol = T(0.0), ftol = δ / 1000, maxiters = 100)
    else
        # No bracketing interval found; choose point with smallest value of f (note: this branch should never be reached)
        logmu_final, err_final = !isfinite(fa) ? (b, fb) : !isfinite(fb) ? (a, fa) : abs(fa) < abs(fb) ? (a, fa) : (b, fb)
    end

    if isfinite(err_final)
        mu_final, res²_final = exp(logmu_final), δ^2 + err_final
        x_final = solve!(work.nnls_prob_smooth_cache, mu_final)
    else
        x_final, mu_final, res²_final = x_unreg, zero(T), one(T)
    end

    return (; x = x_final, mu = mu_final, chi2 = res²_final / res²_min)
end

####
#### L-curve method for choosing the Tikhonov regularization parameter
####

struct NNLSLCurveRegProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W1, W2, C1, C2}
    A::TA
    b::Tb
    m::Int
    n::Int
    nnls_prob::W1
    nnls_prob_smooth_cache::W2
    lsqnonneg_lcurve_fun_cache::C1
    lcurve_corner_caches::C2
end
function NNLSLCurveRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    nnls_prob = NNLSProblem(A, b)
    nnls_prob_smooth_cache = NNLSTikhonovRegProblemCache(A, b)
    lsqnonneg_lcurve_fun_cache = GrowableCache{T, SVector{2, T}}(64, isapprox)
    lcurve_corner_caches = (
        GrowableCache{T, LCurveCornerPoint{T}}(64, isapprox),
        GrowableCache{T, LCurveCornerState{T}}(64, isapprox),
    )
    return NNLSLCurveRegProblem(A, b, m, n, nnls_prob, nnls_prob_smooth_cache, lsqnonneg_lcurve_fun_cache, lcurve_corner_caches)
end

@inline solution(work::NNLSLCurveRegProblem) = solution(get_cache(work.nnls_prob_smooth_cache))
@inline ncomponents(work::NNLSLCurveRegProblem) = ncomponents(get_cache(work.nnls_prob_smooth_cache))

@doc raw"""
    lsqnonneg_lcurve(A::AbstractMatrix, b::AbstractVector)

Compute the Tikhonov-regularized nonnegative least-squares (NNLS) solution ``X_{\mu}`` of the problem:

```math
X_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu^2 ||L x||_2^2
```

where ``L`` is the identity matrix, and ``\mu`` is chosen by locating the corner of the "L-curve"[1].
Details of L-curve theory can be found in Hansen (1992)[2].

# Arguments

  - `A::AbstractMatrix`: Decay basis matrix
  - `b::AbstractVector`: Decay curve data

# Outputs

  - `X::AbstractVector`: Regularized NNLS solution
  - `mu::Real`: Resulting regularization parameter ``\mu``
  - `chi2::Real`: Resulting increase in residual norm relative to the unregularized ``\mu = 0`` solution

# References

  1. A. Cultrera and L. Callegaro, "A simple algorithm to find the L-curve corner in the regularization of ill-posed inverse problems". IOPSciNotes, vol. 1, no. 2, p. 025004, Aug. 2020, https://doi.org/10.1088/2633-1357/abad0d.
  2. Hansen, P.C., 1992. Analysis of Discrete Ill-Posed Problems by Means of the L-Curve. SIAM Review, 34(4), 561-580, https://doi.org/10.1137/1034115.
"""
function lsqnonneg_lcurve(A::AbstractMatrix, b::AbstractVector)
    work = lsqnonneg_lcurve_work(A, b)
    return lsqnonneg_lcurve!(work)
end
lsqnonneg_lcurve_work(A::AbstractMatrix, b::AbstractVector) = NNLSLCurveRegProblem(A, b)

function lsqnonneg_lcurve!(work::NNLSLCurveRegProblem{T, N}) where {T, N}
    # Compute the regularization using the L-curve method
    reset_cache!(work.nnls_prob_smooth_cache)

    # A point on the L-curve is given by (ξ(μ), η(μ)) = (log||Ax-b||^2, log||x||^2)
    #   Note: Squaring the norms is convenient for computing gradients of (ξ(μ), η(μ));
    #         this scales the L-curve, but does not change μ* = argmax C(ξ(μ), η(μ)).
    function f_lcurve(logμ)
        solve!(work.nnls_prob_smooth_cache, exp(logμ))
        ξ = log(resnorm_sq(get_cache(work.nnls_prob_smooth_cache)))
        η = log(seminorm_sq(get_cache(work.nnls_prob_smooth_cache)))
        return SA{T}[ξ, η]
    end

    # Build cached function and solve
    f_lcurve_cached = CachedFunction(f_lcurve, empty!(work.lsqnonneg_lcurve_fun_cache))
    f = LCurveCornerCachedFunction(f_lcurve_cached, empty!.(work.lcurve_corner_caches)...)

    logmu_bounds = (T(-8), T(2))
    logmu_final = lcurve_corner(f, logmu_bounds...)

    # Return the final regularized solution
    mu_final = exp(logmu_final)
    x_final = solve!(work.nnls_prob_smooth_cache, mu_final)
    x_unreg = solve!(work.nnls_prob)
    chi2_final = resnorm_sq(get_cache(work.nnls_prob_smooth_cache)) / resnorm_sq(work.nnls_prob)

    return (; x = x_final, mu = mu_final, chi2 = chi2_final)
end

struct LCurveCornerState{T}
    x⃗::SVector{4, T} # grid of regularization parameters
    P⃗::SVector{4, SVector{2, T}} # points (residual norm, solution seminorm) evaluated at x⃗
end
@inline Base.iterate(s::LCurveCornerState, args...) = iterate((s.x⃗, s.P⃗), args...)

struct LCurveCornerPoint{T}
    P::SVector{2, T} # grid point
    C::T # curvature
end
LCurveCornerPoint(P::SVector{2, T}) where {T} = LCurveCornerPoint(P, T(-Inf))
@inline Base.iterate(p::LCurveCornerPoint, args...) = iterate((p.P, p.C), args...)

struct LCurveCornerCachedFunction{T, F <: CachedFunction{T, SVector{2, T}}, C1 <: GrowableCache{T, LCurveCornerPoint{T}}, C2 <: GrowableCache{T, LCurveCornerState{T}}}
    f::F
    point_cache::C1
    state_cache::C2
end
@inline Base.empty!(f::LCurveCornerCachedFunction) = (empty!(f.f); empty!(f.point_cache); empty!(f.state_cache); f)
@inline (f::LCurveCornerCachedFunction{T})(x::T) where {T} = f.f(x)

@doc raw"""
    lcurve_corner(f, xlow, xhigh)

Find the corner of the L-curve via curvature maximization using a modified version of Algorithm 1 from Cultrera and Callegaro (2020)[1].

# References

  1. A. Cultrera and L. Callegaro, "A simple algorithm to find the L-curve corner in the regularization of ill-posed inverse problems". IOPSciNotes, vol. 1, no. 2, p. 025004, Aug. 2020, https://doi.org/10.1088/2633-1357/abad0d.
"""
function lcurve_corner(f::LCurveCornerCachedFunction{T}, xlow::T = -8.0, xhigh::T = 2.0; xtol = 0.05, Ptol = 0.05, Ctol = 0.01, backtracking = true) where {T}
    # Initialize state
    state = initial_state(f, T(xlow), T(xhigh))

    # Tolerances are relative to initial curve size
    Ptopleft, Pbottomright = state.P⃗[1], state.P⃗[4]
    Pdiam = norm(Ptopleft - Pbottomright)
    Ptol = Pdiam * T(Ptol) # convergence occurs when diameter of L-curve state is less than Ptol
    Ctol = Pdiam * T(Ctol) # note: *not* a tolerance on curvature, but on the minimum diameter of the L-curve state used to estimate curvature (see `Pfilter` below)

    # For very small regularization points on the L-curve may be extremely close, leading to
    # numerically unstable curvature estimates. Assign these points -Inf curvature.
    Pfilter = P -> min(norm(P - Ptopleft), norm(P - Pbottomright)) > T(Ctol)
    update_curvature!(f, state, Pfilter)

    # msg(s, state) = (@info "$s: [x⃗, P⃗, C⃗] = "; display(hcat(state.x⃗, state.P⃗, [f.point_cache[x].C for x in state.x⃗])))
    # msg("Starting", state)

    iter = 0
    while true
        iter += 1
        if backtracking
            # Find state with minimum diameter which contains the current best estimate maximum curvature point
            (x, (P, C)), _, _ = mapfindmax(T, ((x, (P, C)),) -> C, pairs(f.point_cache))
            for (_, s) in f.state_cache
                if (s.x⃗[2] == x || s.x⃗[3] == x) && abs(s.x⃗[4] - s.x⃗[1]) <= abs(state.x⃗[4] - state.x⃗[1])
                    state = s
                end
            end
        end

        # Move state toward region of lower curvature
        if f.point_cache[state.x⃗[2]].C > f.point_cache[state.x⃗[3]].C
            state = move_left(f, state)
            update_curvature!(f, state, Pfilter)
            # msg("C₂ > C₃; moved left", state)
        else
            state = move_right(f, state)
            update_curvature!(f, state, Pfilter)
            # msg("C₃ ≥ C₂; moved right", state)
        end
        backtracking && push!(f.state_cache, (iter, state))
        is_converged(state; xtol = T(xtol), Ptol = T(Ptol)) && break
    end

    x = f.point_cache[state.x⃗[2]].C > f.point_cache[state.x⃗[3]].C ? state.x⃗[2] : state.x⃗[3]
    # msg("Converged", state)

    return x
end

function initial_state(f::LCurveCornerCachedFunction{T}, x₁::T, x₄::T) where {T}
    x₂ = (T(φ) * x₁ + x₄) / (T(φ) + 1)
    x₃ = x₁ + (x₄ - x₂)
    x⃗ = SA[x₁, x₂, x₃, x₄]
    P⃗ = SA[f(x₁), f(x₂), f(x₃), f(x₄)]
    Base.Cartesian.@nexprs 4 i -> push!(f.point_cache, (x⃗[i], LCurveCornerPoint(P⃗[i])))
    return LCurveCornerState(x⃗, P⃗)
end

is_converged(state::LCurveCornerState; xtol, Ptol) = abs(state.x⃗[4] - state.x⃗[1]) < xtol || norm(state.P⃗[1] - state.P⃗[4]) < Ptol

function move_left(f::LCurveCornerCachedFunction{T}, state::LCurveCornerState{T}) where {T}
    (; x⃗, P⃗) = state
    x⃗ = SA[x⃗[1], (T(φ)*x⃗[1]+x⃗[3])/(T(φ)+1), x⃗[2], x⃗[3]]
    P⃗ = SA[P⃗[1], f(x⃗[2]), P⃗[2], P⃗[3]] # only P⃗[2] is recalculated
    return LCurveCornerState{T}(x⃗, P⃗)
end

function move_right(f::LCurveCornerCachedFunction{T}, state::LCurveCornerState{T}) where {T}
    (; x⃗, P⃗) = state
    x⃗ = SA[x⃗[2], x⃗[3], x⃗[2]+(x⃗[4]-x⃗[3]), x⃗[4]]
    P⃗ = SA[P⃗[2], P⃗[3], f(x⃗[3]), P⃗[4]] # only P⃗[3] is recalculated
    return LCurveCornerState(x⃗, P⃗)
end

function update_curvature!(f::LCurveCornerCachedFunction{T}, state::LCurveCornerState{T}, Pfilter = nothing) where {T}
    (; x⃗, P⃗) = state
    for i in 1:4
        x, P, C = x⃗[i], P⃗[i], T(-Inf)
        if Pfilter === nothing || Pfilter(P)
            # Compute curvature from nearest neighbours
            x₋, x₊ = T(-Inf), T(+Inf)
            P₋, P₊ = P, P
            for (_x, (_P, _)) in pairs(f.point_cache)
                (x₋ < _x < x) && ((x₋, P₋) = (_x, _P))
                (x < _x < x₊) && ((x₊, P₊) = (_x, _P))
            end
            C = menger(P₋, P, P₊)
        end
        f.point_cache[x] = LCurveCornerPoint(P, C)
    end
    return state
end

function menger(Pⱼ::V, Pₖ::V, Pₗ::V) where {V <: SVector{2}}
    Δⱼₖ, Δₖₗ, Δₗⱼ = Pⱼ - Pₖ, Pₖ - Pₗ, Pₗ - Pⱼ
    P̄ⱼP̄ₖ, P̄ₖP̄ₗ, P̄ₗP̄ⱼ = Δⱼₖ ⋅ Δⱼₖ, Δₖₗ ⋅ Δₖₗ, Δₗⱼ ⋅ Δₗⱼ
    Cₖ = 2 * (Δⱼₖ × Δₖₗ) / √(P̄ⱼP̄ₖ * P̄ₖP̄ₗ * P̄ₗP̄ⱼ)
    return Cₖ
end

function menger(f; h = 1e-3)
    function menger_curvature_inner(x)
        fⱼ, fₖ, fₗ = f(x - h), f(x), f(x + h)
        Pⱼ, Pₖ, Pₗ = SA[x-h, fⱼ], SA[x, fₖ], SA[x+h, fₗ]
        return menger(Pⱼ, Pₖ, Pₗ)
    end
end

function menger(x, y; h = 1e-3)
    function menger_curvature_inner(t)
        x₋, x₀, x₊ = x(t - h), x(t), x(t + h)
        y₋, y₀, y₊ = y(t - h), y(t), y(t + h)
        x′, x′′ = (x₊ - x₋) / 2h, (x₊ - 2x₀ + x₋) / h^2
        y′, y′′ = (y₊ - y₋) / 2h, (y₊ - 2y₀ + y₋) / h^2
        return (x′ * y′′ - y′ * x′′) / √((x′^2 + y′^2)^3)
    end
end

#=
lin_interp(x, x₁, x₂, y₁, y₂) = y₁ + (y₂ - y₁) * (x - x₁) / (x₂ - x₁)
exp_interp(x, x₁, x₂, y₁, y₂) = y₁ + log1p(expm1(y₂ - y₁) * (x - x₁) / (x₂ - x₁))

function menger(x::Dierckx.Spline1D, y::Dierckx.Spline1D)
    function menger_curvature_inner(t)
        x′  = Dierckx.derivative(x, t; nu = 1)
        x′′ = Dierckx.derivative(x, t; nu = 2)
        y′  = Dierckx.derivative(y, t; nu = 1)
        y′′ = Dierckx.derivative(y, t; nu = 2)
        return (x′ * y′′ - y′ * x′′) / √((x′^2 + y′^2)^3)
    end
end

function menger(y::Dierckx.Spline1D)
    function menger_curvature_inner(t)
        y′  = Dierckx.derivative(y, t; nu = 1)
        y′′ = Dierckx.derivative(y, t; nu = 2)
        return y′′ / √((1 + y′^2)^3)
    end
end

function menger(xⱼ::T, xₖ::T, xₗ::T, Pⱼ::V, Pₖ::V, Pₗ::V; interp_uniform = true, linear_deriv = true) where {T, V <: SVector{2, T}}
    if interp_uniform
        h = min(abs(xₖ - xⱼ), abs(xₗ - xₖ)) / T(φ)
        h₋ = h₊ = h
        x₋, x₀, x₊ = xₖ - h, xₖ, xₖ + h
        P₀ = Pₖ
        P₋ = exp_interp.(x₋, xⱼ, xₖ, Pⱼ, Pₖ)
        P₊ = exp_interp.(x₊, xₖ, xₗ, Pₖ, Pₗ)
    else
        P₋, P₀, P₊ = Pⱼ, Pₖ, Pₗ
        x₋, x₀, x₊ = xⱼ, xₖ, xₗ
        h₋, h₊ = x₀ - x₋, x₊ - x₀
    end
    ξ₋, ξ₀, ξ₊ = P₋[1], P₀[1], P₊[1]
    η₋, η₀, η₊ = P₋[2], P₀[2], P₊[2]

    if linear_deriv
        ξ′ = (ξ₊ - ξ₋) / (h₊ + h₋)
        η′ = (η₊ - η₋) / (h₊ + h₋)
    else
        ξ′ = (h₋^2 * ξ₊ + (h₊ + h₋) * (h₊ - h₋) * ξ₀ - h₊^2 * ξ₋) / (h₊ * h₋ * (h₊ + h₋))
        η′ = (h₋^2 * η₊ + (h₊ + h₋) * (h₊ - h₋) * η₀ - h₊^2 * η₋) / (h₊ * h₋ * (h₊ + h₋))
    end

    ξ′′ = 2 * (h₋ * ξ₊ - (h₊ + h₋) * ξ₀ + h₊ * ξ₋) / (h₊ * h₋ * (h₊ + h₋))
    η′′ = 2 * (h₋ * η₊ - (h₊ + h₋) * η₀ + h₊ * η₋) / (h₊ * h₋ * (h₊ + h₋))

    return (ξ′ * η′′ - η′ * ξ′′) / √((ξ′^2 + η′^2)^3)
end

function maximize_curvature(state::LCurveCornerState{T}) where {T}
    # Maximize curvature and transform back from t-space to x-space
    (; x⃗, P⃗) = state
    t₁, t₂, t₃, t₄ = T(0), 1 / T(3), 2 / T(3), T(1)
    t_opt, P_opt, C_opt = maximize_curvature(P⃗...)
    x_opt =
        t₁ <= t_opt < t₂ ? lin_interp(t_opt, t₁, t₂, x⃗[1], x⃗[2]) :
        t₂ <= t_opt < t₃ ? lin_interp(t_opt, t₂, t₃, x⃗[2], x⃗[3]) :
        lin_interp(t_opt, t₃, t₄, x⃗[3], x⃗[4])
    return x_opt
end

function maximize_curvature(P₁::V, P₂::V, P₃::V, P₄::V; bezier = true) where {T, V <: SVector{2, T}}
    # Analytically maximize curvature of parametric cubic spline fit to data.
    #   see: https://cs.stackexchange.com/a/131032
    a, e = P₁[1], P₁[2]
    b, f = P₂[1], P₂[2]
    c, g = P₃[1], P₃[2]
    d, h = P₄[1], P₄[2]

    if bezier
        ξ = t -> a * (1 - t)^3 + 3 * b * (1 - t)^2 * t + 3 * c * (1 - t) * t^2 + d * t^3
        η = t -> e * (1 - t)^3 + 3 * f * (1 - t)^2 * t + 3 * g * (1 - t) * t^2 + h * t^3
        P = t -> SA{T}[ξ(t), η(t)]

        # In order to keep the equations in a more compact form, introduce the following substitutions:
        m = d - 3 * c + 3 * b - a
        n = c - 2 * b + a
        o = b - a
        p = h - 3 * g + 3 * f - e
        q = g - 2 * f + e
        r = f - e
    else
        ξ = t -> a * (t - 1 / 3) * (t - 2 / 3) * (t - 1 / 1) / T((0 / 1 - 1 / 3) * (0 / 1 - 2 / 3) * (0 / 1 - 1 / 1)) + # (-9  * a * t^3)/2  + (+18 * a * t^2)/2 + (-11 * a * t)/2 + a
                 b * (t - 0 / 1) * (t - 2 / 3) * (t - 1 / 1) / T((1 / 3 - 0 / 1) * (1 / 3 - 2 / 3) * (1 / 3 - 1 / 1)) + # (+27 * b * t^3)/2  + (-45 * b * t^2)/2 + (+18 * b * t)/2 +
                 c * (t - 0 / 1) * (t - 1 / 3) * (t - 1 / 1) / T((2 / 3 - 0 / 1) * (2 / 3 - 1 / 3) * (2 / 3 - 1 / 1)) + # (-27 * c * t^3)/2  + (+36 * c * t^2)/2 + ( -9 * c * t)/2 +
                 d * (t - 0 / 1) * (t - 1 / 3) * (t - 2 / 3) / T((1 / 1 - 0 / 1) * (1 / 1 - 1 / 3) * (1 / 1 - 2 / 3))   # (+9  * d * t^3)/2  + ( -9 * d * t^2)/2 + ( +2 * d * t)/2
        η = t -> e * (t - 1 / 3) * (t - 2 / 3) * (t - 1 / 1) / T((0 / 1 - 1 / 3) * (0 / 1 - 2 / 3) * (0 / 1 - 1 / 1)) + # (-9  * e * t^3)/2  + (+18 * e * t^2)/2 + (-11 * e * t)/2 + e
                 f * (t - 0 / 1) * (t - 2 / 3) * (t - 1 / 1) / T((1 / 3 - 0 / 1) * (1 / 3 - 2 / 3) * (1 / 3 - 1 / 1)) + # (+27 * f * t^3)/2  + (-45 * f * t^2)/2 + (+18 * f * t)/2 +
                 g * (t - 0 / 1) * (t - 1 / 3) * (t - 1 / 1) / T((2 / 3 - 0 / 1) * (2 / 3 - 1 / 3) * (2 / 3 - 1 / 1)) + # (-27 * g * t^3)/2  + (+36 * g * t^2)/2 + ( -9 * g * t)/2 +
                 h * (t - 0 / 1) * (t - 1 / 3) * (t - 2 / 3) / T((1 / 1 - 0 / 1) * (1 / 1 - 1 / 3) * (1 / 1 - 2 / 3))   # (+9  * h * t^3)/2  + ( -9 * h * t^2)/2 + ( +2 * h * t)/2
        P = t -> SA{T}[ξ(t), η(t)]

        # In order to keep the equations in a more compact form, introduce the following substitutions:
        m = (-9 * a + 27 * b - 27 * c + 9 * d) / 2
        n = (+18 * a - 45 * b + 36 * c - 9 * d) / 6
        o = (-11 * a + 18 * b - 9 * c + 2 * d) / 6
        p = (-9 * e + 27 * f - 27 * g + 9 * h) / 2
        q = (+18 * e - 45 * f + 36 * g - 9 * h) / 6
        r = (-11 * e + 18 * f - 9 * g + 2 * h) / 6
    end

    # This leads to the following simplified derivatives:
    ξ′  = t -> 3 * (m * t^2 + 2 * n * t + o)
    ξ′′ = t -> 6 * (m * t + n)
    η′  = t -> 3 * (p * t^2 + 2 * q * t + r)
    η′′ = t -> 6 * (p * t + q)

    # Curvature and its derivative:
    k  = t -> ((3 * (m * t^2 + 2 * n * t + o)) * (6 * (p * t + q)) - (3 * (p * t^2 + 2 * q * t + r)) * (6 * (m * t + n))) / ((3 * (m * t^2 + 2 * n * t + o))^2 + (3 * (p * t^2 + 2 * q * t + r))^2)^(3 / 2)
    k′ = t -> (-18 * m * (p * t^2 + 2 * q * t + r) + 18 * p * (m * t^2 + 2 * n * t + o) - 18 * (m * t + n) * (2 * p * t + 2 * q) + 18 * (2 * m * t + 2 * n) * (p * t + q)) / (9 * (p * t^2 + 2 * q * t + r)^2 + 9 * (m * t^2 + 2 * n * t + o)^2)^(3 / 2) - (3 * (18 * (p * t + q) * (m * t^2 + 2 * n * t + o) - 18 * (m * t + n) * (p * t^2 + 2 * q * t + r)) * (18 * (2 * p * t + 2 * q) * (p * t^2 + 2 * q * t + r) + 18 * (2 * m * t + 2 * n) * (m * t^2 + 2 * n * t + o))) / (2 * (9 * (p * t^2 + 2 * q * t + r)^2 + 9 * (m * t^2 + 2 * n * t + o)^2)^(5 / 2))

    # Solve analytically
    coeffs = MVector{6, Complex{T}}(
        (1296 * m * p^2 + 1296 * m^3) * q - 1296 * n * p^3 - 1296 * m^2 * n * p,
        (1620 * m * p^2 + 1620 * m^3) * r + 3240 * m * p * q^2 + (3240 * m^2 * n - 3240 * n * p^2) * q - 1620 * o * p^3 + ((-1620 * m^2 * o) - 3240 * m * n^2) * p,
        (5184 * m * p * q + 1296 * n * p^2 + 6480 * m^2 * n) * r + 1296 * m * q^3 - 1296 * n * p * q^2 + ((-6480 * o * p^2) - 1296 * m^2 * o + 1296 * m * n^2) * q + ((-5184 * m * n * o) - 1296 * n^3) * p,
        1296 * m * p * r^2 + (1944 * m * q^2 + 6480 * n * p * q - 1296 * o * p^2 + 1296 * m^2 * o + 8424 * m * n^2) * r - 8424 * o * p * q^2 - 6480 * m * n * o * q + ((-1296 * m * o^2) - 1944 * n^2 * o) * p,
        2592 * n * p * r^2 + (3888 * n * q^2 - 2592 * o * p * q + 2592 * m * n * o + 3888 * n^3) * r - 3888 * o * q^3 + ((-2592 * m * o^2) - 3888 * n^2 * o) * q,
        -324 * m * r^3 + (1944 * n * q + 324 * o * p) * r^2 + ((-1944 * o * q^2) - 324 * m * o^2 + 1944 * n^2 * o) * r - 1944 * n * o^2 * q + 324 * o^3 * p,
    )
    roots = MVector{6, Complex{T}}(undef)
    PolynomialRoots.roots5!(roots, coeffs, eps(T), true)

    # Check roots and return maximum
    t₁, t₄ = zero(T), one(T)
    k₁, k₄ = k(t₁), k(t₄)
    tmax, Pmax, kmax = k₁ > k₄ ? (t₁, P₁, k₁) : (t₄, P₄, k₄)
    for rᵢ in roots
        _t, _s = real(rᵢ), imag(rᵢ)
        !(_s ≈ 0) && continue # real roots only
        !(t₁ <= _t <= t₄) && continue # filter roots within range
        ((_k = k(_t)) > kmax) && ((tmax, Pmax, kmax) = (_t, P(_t), _k))
    end

    return tmax, Pmax, kmax
end

function directed_angle(v₁::V, v₂::V) where {T, V <: SVector{2, T}}
    α = atan(v₁[2], v₁[1]) - atan(v₂[2], v₂[1])
    return α < 0 ? 2 * T(π) + α : α
end
directed_angle(Pⱼ::V, Pₖ::V, Pₗ::V) where {V <: SVector{2}} = directed_angle(Pⱼ - Pₖ, Pₗ - Pₖ)

function kahan_angle(v₁::V, v₂::V) where {T, V <: SVector{2, T}}
    # Kahan's method for computing the angle between v₁ and v₂.
    #   see: https://scicomp.stackexchange.com/a/27694
    a, b, c = norm(v₁), norm(v₂), norm(v₁ - v₂)
    a, b = max(a, b), min(a, b)
    μ = b ≥ c ? c - (a - b) : (b - (a - c))
    num = ((a - b) + c) * max(μ, zero(T))
    den = (a + (b + c)) * ((a - c) + b)
    α = 2 * atan(√(num / den))
    return v₁ × v₂ > 0 ? 2 * T(π) - α : α
end
kahan_angle(Pⱼ::V, Pₖ::V, Pₗ::V) where {V <: SVector{2}} = kahan_angle(Pⱼ - Pₖ, Pₗ - Pₖ)
=#

####
#### GCV method for choosing the Tikhonov regularization parameter
####

struct NNLSGCVRegProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W0, W1, W2}
    A::TA
    b::Tb
    m::Int
    n::Int
    γ::Vector{T}
    svd_work::W0
    nnls_prob::W1
    nnls_prob_smooth_cache::W2
end
function NNLSGCVRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    svd_work = SVDValsWorkspace(A) # workspace for computing singular values
    nnls_prob = NNLSProblem(A, b)
    nnls_prob_smooth_cache = NNLSTikhonovRegProblemCache(A, b)
    γ = svd_work.S # store reference to (generalized) singular values for convenience
    return NNLSGCVRegProblem(A, b, m, n, γ, svd_work, nnls_prob, nnls_prob_smooth_cache)
end

@inline solution(work::NNLSGCVRegProblem) = solution(get_cache(work.nnls_prob_smooth_cache))
@inline ncomponents(work::NNLSGCVRegProblem) = ncomponents(get_cache(work.nnls_prob_smooth_cache))

@doc raw"""
    lsqnonneg_gcv(A::AbstractMatrix, b::AbstractVector)

Compute the Tikhonov-regularized nonnegative least-squares (NNLS) solution ``X_{\mu}`` of the problem:

```math
X_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu^2 ||L x||_2^2
```

where ``L`` is the identity matrix, and ``\mu`` is chosen via the Generalized Cross-Validation (GCV) method:

```math
\mu = \underset{\nu \ge 0}{\operatorname{argmin}}\; \frac{||AX_{\nu} - b||_2^2}{\mathcal{T}(\nu)^2}
```

where ``\mathcal{T}(\mu)`` is the "degrees of freedom" of the regularized system

```math
\mathcal{T}(\mu) = \operatorname{tr}(I - A (A^T A + \mu^2 L^T L) A^T).
```

Details of the GCV method can be found in Hansen (1992)[1].

# Arguments

  - `A::AbstractMatrix`: Decay basis matrix
  - `b::AbstractVector`: Decay curve data

# Outputs

  - `X::AbstractVector`: Regularized NNLS solution
  - `mu::Real`: Resulting regularization parameter ``\mu``
  - `chi2::Real`: Resulting increase in residual norm relative to the unregularized ``\mu = 0`` solution

# References

  1. Hansen, P.C., 1992. Analysis of Discrete Ill-Posed Problems by Means of the L-Curve. SIAM Review, 34(4), 561-580, https://doi.org/10.1137/1034115.
"""
function lsqnonneg_gcv(A::AbstractMatrix, b::AbstractVector)
    work = lsqnonneg_gcv_work(A, b)
    return lsqnonneg_gcv!(work)
end
lsqnonneg_gcv_work(A::AbstractMatrix, b::AbstractVector) = NNLSGCVRegProblem(A, b)

function lsqnonneg_gcv!(work::NNLSGCVRegProblem{T, N}; method = :brent) where {T, N}
    # Find μ by minimizing the function G(μ) (GCV method)
    reset_cache!(work.nnls_prob_smooth_cache)

    # Precompute singular values for GCV computation
    svdvals!(work.svd_work, work.A)

    # Non-zero lower bound for GCV to avoid log(0) in the objective function
    gcv_low = gcv_lower_bound(work)

    if method === :nlopt
        # alg = :LN_COBYLA # local, gradient-free, linear approximation of objective
        alg = :LN_BOBYQA # local, gradient-free, quadratic approximation of objective
        # alg = :GN_AGS # global, gradient-free, hilbert curve based dimension reduction
        # alg = :LN_NELDERMEAD # local, gradient-free, simplex method
        # alg = :LN_SBPLX # local, gradient-free, subspace searching simplex method
        # alg = :LD_CCSAQ # local, first-order (rough ranking: [:LD_MMA, :LD_SLSQP, :LD_LBFGS, :LD_CCSAQ, :LD_AUGLAG])
        opt               = NLopt.Opt(alg, 1)
        opt.lower_bounds  = -8.0
        opt.upper_bounds  = 2.0
        opt.xtol_abs      = 1e-4
        opt.xtol_rel      = 1e-4
        opt.ftol_abs      = 0.0
        opt.ftol_rel      = 0.0
        opt.min_objective = (logμ, ∇logμ) -> @inbounds Float64(log(max(gcv!(work, logμ[1]), gcv_low)))
        minf, minx, ret   = NLopt.optimize(opt, [-4.0])
        logmu_final       = @inbounds T(minx[1])
    elseif method === :brent
        logmu_final, _ = brent_minimize(-8.0, 2.0; xrtol = T(0.05), xatol = T(1e-4), maxiters = 10) do logμ
            return log(max(gcv!(work, logμ), gcv_low))
        end
    else
        error("Unknown minimization method: $method")
    end

    # Return the final regularized solution
    mu_final = exp(logmu_final)
    x_final = solve!(work.nnls_prob_smooth_cache, mu_final)
    x_unreg = solve!(work.nnls_prob)
    chi2_final = resnorm_sq(get_cache(work.nnls_prob_smooth_cache)) / resnorm_sq(work.nnls_prob)

    return (; x = x_final, mu = mu_final, chi2 = chi2_final)
end

# Implements equation (32) from:
#
#   Analysis of Discrete Ill-Posed Problems by Means of the L-Curve
#   Hansen et al. 1992 (https://epubs.siam.org/doi/10.1137/1034115)
#
# where here L = Id and λ = μ.
function gcv!(work::NNLSGCVRegProblem, logμ)
    # Unpack buffers
    (; m, n, γ) = work

    # Solve regularized NNLS problem
    μ = exp(logμ)
    solve!(work.nnls_prob_smooth_cache, μ)
    cache = get_cache(work.nnls_prob_smooth_cache)

    # Compute GCV
    res² = resnorm_sq(cache) # squared residual norm ||A * x(μ) - b||^2
    dof = gcv_dof(m, n, γ, μ) # degrees of freedom; γ are (generalized) singular values
    gcv = res² / dof^2

    return gcv
end

function gcv_and_∇gcv!(work::NNLSGCVRegProblem, logμ)
    # Unpack buffers
    (; m, n, γ) = work

    # Solve regularized NNLS problem
    μ = exp(logμ)
    solve!(work.nnls_prob_smooth_cache, μ)
    cache = get_cache(work.nnls_prob_smooth_cache)

    # Compute primal
    res² = resnorm_sq(cache) # squared residual norm ||A * x(μ) - b||^2
    dof = gcv_dof(m, n, γ, μ) # degrees of freedom; γ are (generalized) singular values
    gcv = res² / dof^2

    # Compute derivative: ∂/∂λ [resnorm_sq(λ) / dof(λ)^2] = ∇resnorm_sq(λ) / dof(λ)^2 - 2 * resnorm_sq(λ) * ∇dof(λ) / dof(λ)^3
    ∇res² = ∇resnorm_sq(cache)
    ∇dof = ∇gcv_dof(m, n, γ, μ)
    ∇gcv = (∇res² - 2 * res² * ∇dof / dof) / dof^2

    return gcv, ∇gcv
end

# Non-trivial lower bound of the GCV function
#   GCV(μ) = ||A * x(μ) - b||^2 / 𝒯(μ)^2
# where 𝒯(μ) is the "degrees of freedom" of the regularized system
#   𝒯(μ) = tr(I - A * (A'A + μ²I)⁻¹ * A')
#        ∈ [max(m - n, 0), m)
# The trivial lower bound GCV(μ) = 0 can (sometimes) be achieved when μ = 0 if ||A * x(μ = 0) - b|| = 0.
# Let ε > 0 be the RMSE threshold below which we consider the solution exact, i.e. bound ||A * x(μ) - b|| / √m >= ε.
# Then, GCV(μ) = ||A * x(μ) - b||^2 / 𝒯(μ)^2 >= (√m * ε)^2 / m^2 = ε^2 / m
gcv_lower_bound(m::Int, n::Int, ε::Real) = ε^2 / m
gcv_lower_bound(work::NNLSGCVRegProblem{T}, ε::T = eps(T)) where {T} = gcv_lower_bound(work.m, work.n, ε)

#=
# Equivalent direct method (less efficient)
function gcv!(work::NNLSGCVRegProblem, logμ, ::Val{extract_subproblem} = Val(false)) where {extract_subproblem}
    # Unpack buffers
    (; A, b, m, n, Aμ, A_buf, Aᵀ_buf, AᵀA_buf) = work

    # Solve regularized NNLS problem and record residual norm ||A * x(μ) - b||^2
    μ = exp(logμ)
    solve!(work.nnls_prob_smooth_cache, μ)
    res² = resnorm_sq(get_cache(work.nnls_prob_smooth_cache))

    if extract_subproblem
        # Extract equivalent unconstrained least squares subproblem from NNLS problem
        # by extracting columns of A which correspond to nonzero components of x(μ)
        idx = NNLS.components(get_cache(work.nnls_prob_smooth_cache).nnls_prob.nnls_work)
        n′ = length(idx)
        A′ = reshape(view(A_buf, 1:m*n′), m, n′)
        At′ = reshape(view(Aᵀ_buf, 1:n′*m), n′, m)
        AtA′ = reshape(view(AᵀA_buf, 1:n′*n′), n′, n′)
        copyto!(A′, view(A, :, idx))
    else
        # Use full matrix
        A′ = A
        At′ = Aᵀ_buf
        AtA′ = AᵀA_buf
    end

    # Efficient compution of
    #   Aμ = A * (A'A + μ²I)⁻¹ * A'
    # where the matrices have sizes
    #   A: (m, n), Aμ: (m, m), At: (n, m), AtA: (n, n)
    mul!(AtA′, A′', A′) # A'A
    @simd for i in 1:n
        AtA′[i, i] += μ^2 # A'A + μ²I
    end
    ldiv!(At′, cholesky!(Symmetric(AtA′)), A′') # (A'A + μ²I)⁻¹ * A'
    mul!(Aμ, A′, At′) # A * (A'A + μ²I)⁻¹ * A'

    # Return Generalized cross-validation. See equations 27 and 32 in
    #   Hansen, P.C., 1992. Analysis of Discrete Ill-Posed Problems by Means of the L-Curve. SIAM Review, 34(4), 561-580
    #   https://doi.org/10.1137/1034115
    dof = m - tr(Aμ) # tr(I - Aμ) = m - tr(Aμ) for m x m matrix Aμ; can be considered as the "degrees of freedom" (Hansen, 1992)
    gcv = res² / dof^2 # ||A * x(μ) - b||^2 / tr(I - Aμ)^2

    return gcv
end
=#

# Equation (27) from Hansen et al. 1992 (https://epubs.siam.org/doi/10.1137/1034115),
# specialized for L = identity:
#
#   tr(I_m - A * (A'A + λ^2 * L'L)⁻¹ * A') = m - n + sum_i λ^2 / (γ_i^2 + λ^2)
#
# where γ_i are the generalized singular values, which are equivalent to ordinary
# singular values when L = identity, and size(A) = (m, n).
# Can be considered as the "degrees of freedom".
function gcv_dof(m::Int, n::Int, γ::AbstractVector{T}, λ::T) where {T}
    dof = T(max(m - n, 0)) # handle underdetermined systems (m < n)
    λ² = abs2(λ)
    @simd for γᵢ in γ
        γᵢ² = abs2(γᵢ)
        dof += λ² / (γᵢ² + λ²)
    end
    return dof
end
gcv_dof(A::AbstractMatrix{T}, λ::T) where {T} = gcv_dof(size(A)..., svdvals(A), λ)

# DOF derivative: ∂/∂λ gcv_dof(m, n, γ, λ)
function ∇gcv_dof(m::Int, n::Int, γ::AbstractVector{T}, λ::T) where {T}
    ∇dof = zero(T)
    λ² = abs2(λ)
    @simd for γᵢ in γ
        γᵢ² = abs2(γᵢ)
        ∇dof += 2 * λ * γᵢ² / (γᵢ² + λ²)^2
    end
    return ∇dof
end
∇gcv_dof(A::AbstractMatrix{T}, λ::T) where {T} = ∇gcv_dof(size(A)..., svdvals(A), λ)
