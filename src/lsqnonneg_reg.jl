####
#### Lazy wrappers for LHS matrix and RHS vector for augmented Tikhonov-regularized NNLS problems
####

struct PaddedVector{T, Vd <: AbstractVector{T}} <: AbstractVector{T}
    d::Vd
    pad::Int
end
Base.size(x::PaddedVector) = (length(x.d) + x.pad,)

function Base.copyto!(y::AbstractVector{T}, x::PaddedVector{T}) where {T}
    @assert size(x) == size(y)
    @unpack d, pad = x
    m = length(x.d)
    @inbounds @simd ivdep for i in 1:m
        y[i] = d[i]
    end
    @inbounds @simd ivdep for i in m+1:m+pad
        y[i] = zero(T)
    end
    return y
end

struct TikhonovPaddedMatrix{T, MC <: AbstractMatrix{T}} <: AbstractMatrix{T}
    C::MC
    μ::Base.RefValue{T}
end
TikhonovPaddedMatrix(C::AbstractMatrix, μ) = TikhonovPaddedMatrix(C, Ref(μ))
Base.size(A::TikhonovPaddedMatrix) = ((m,n) = size(A.C); return (m+n,n))

function Base.copyto!(B::AbstractMatrix{T}, A::TikhonovPaddedMatrix{T}) where {T}
    @assert size(A) == size(B)
    @unpack C, μ = A
    m, n = size(C)
    @inbounds @simd ivdep for j in 1:n
        @simd ivdep for i in 1:m
            B[i,j] = C[i,j]
        end
        @simd ivdep for i in m+1:m+n
            B[i,j] = ifelse(i == m+j, μ[], zero(T))
        end
    end
    return B
end

cache!(cache::AbstractArray, x, f) = cache !== nothing && push!(cache, (; x, f))
cache!(cache::Dict, x, f) = cache !== nothing && (cache[x] = f)

lin_interp(x, x₁, x₂, y₁, y₂) = y₁ + (y₂ - y₁) * (x - x₁)  / (x₂ - x₁)
exp_interp(x, x₁, x₂, y₁, y₂) = y₁ + log1p(expm1(y₂ - y₁) * (x - x₁)  / (x₂ - x₁))

####
#### Tikhonov regularized NNLS problem
####

struct NNLSTikhonovRegProblem{T, MC <: AbstractMatrix{T}, Vd <: AbstractVector{T}, W}
    C::MC
    d::Vd
    m::Int
    n::Int
    nnls_work::W
end
function NNLSTikhonovRegProblem(C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    m, n = size(C)
    nnls_work = NNLSProblem(TikhonovPaddedMatrix(C, T(NaN)), PaddedVector(d, n))
    NNLSTikhonovRegProblem(C, d, m, n, nnls_work)
end

mu(work::NNLSTikhonovRegProblem) = work.nnls_work.C.μ[]
mu!(work::NNLSTikhonovRegProblem, μ) = work.nnls_work.C.μ[] = μ

function solve!(work::NNLSTikhonovRegProblem, μ)
    # Set regularization parameter and solve NNLS problem
    mu!(work, μ)
    solve!(work.nnls_work)
    return solution(work)
end

solution(work::NNLSTikhonovRegProblem) = solution(work.nnls_work)

resnorm_sq(work::NNLSTikhonovRegProblem) = chi2(work)
resnorm(work::NNLSTikhonovRegProblem) = sqrt(resnorm_sq(work))

seminorm_sq(work::NNLSTikhonovRegProblem) = solution(work) ⋅ solution(work)
seminorm(work::NNLSTikhonovRegProblem) = sqrt(seminorm_sq(work))

loss(work::NNLSTikhonovRegProblem) = chi2(work.nnls_work)

reg(work::NNLSTikhonovRegProblem) = mu(work)^2 * seminorm_sq(work)

chi2(work::NNLSTikhonovRegProblem) = max(loss(work) - reg(work), 0)

# Extract columns spanning solution space of regularized NNLS problem
function subproblem(work::NNLSTikhonovRegProblem{T}) where {T}
    @unpack C, d, m, n, nnls_work = work
    μ = mu(work)
    x = solution(work)
    Jnz = findall(!≈(0), x)
    nnz = length(Jnz)
    return TikhonovPaddedMatrix(C[:,Jnz], μ), PaddedVector(d, nnz), x[Jnz]
end

function ∇reg(work::NNLSTikhonovRegProblem)
    @unpack C, d = work
    μ = mu(work)
    x = solution(work)
    C′, d′, x′ = subproblem(work)
    b′ = PaddedVector(C′.C' \ x′, d′.pad)
    ∇nnls_work = NNLSProblem(C′, b′)
    solve!(∇nnls_work)
    ∇x′ = -solution(∇nnls_work)
    ∇μ² = 2 * ((C′.C * x′ - d′.d) ⋅ (C′.C * ∇x′)) #+ (x′ ⋅ x′) + 2 * μ^2 * (x′ ⋅ ∇x′)
    ∇μ = 2 * μ * ∇μ²
    return ∇μ
end

function chi2factor_relerr!(work::NNLSTikhonovRegProblem, logμ, ∇logμ = nothing; χ²target)
    μ = exp(logμ)
    solve!(work, μ)
    χ² = chi2(work)
    relerr = log(χ² / χ²target) # better behaved than χ² / χ²target - 1 for large χ²
    if ∇logμ !== nothing && length(∇logμ) > 0
        ∂χ²_∂logμ = ∇reg(work)
        ∂relerr_∂logμ = μ * ∂χ²_∂logμ / χ²
        @inbounds ∇logμ[1] = ∂relerr_∂logμ
    end
    return relerr
end
chi2factor_relerr⁻¹(relerr; χ²target) = χ²target * exp(relerr)

function chi2factor_loss!(work::NNLSTikhonovRegProblem, logμ, ∇logμ = nothing; χ²target)
    relerr = chi2factor_relerr!(work, logμ, ∇logμ; χ²target)
    loss = abs(relerr)
    if ∇logμ !== nothing && length(∇logμ) > 0
        @inbounds ∂relerr_∂logμ = ∇logμ[1]
        ∂loss_∂logμ = sign(relerr) * ∂relerr_∂logμ
        @inbounds ∇logμ[1] = ∂loss_∂logμ
    end
    return loss
end

"""
Helper struct which wraps `N` caches of `NNLSTikhonovRegProblem` workspaces.
Useful for optimization problems where the last function call may not be
the optimium, but perhaps it was one or two calls previous and is still in the
`NNLSTikhonovRegProblemCache` and a recomputation can be avoided.
"""
struct NNLSTikhonovRegProblemCache{N,W}
    cache::NTuple{N,W}
    idx::Base.RefValue{Int}
end
function NNLSTikhonovRegProblemCache(C::AbstractMatrix{T}, d::AbstractVector{T}, ::Val{N} = Val(5)) where {T,N}
    cache = ntuple(_ -> NNLSTikhonovRegProblem(C, d), N)
    idx = Ref(1)
    return NNLSTikhonovRegProblemCache(cache, idx)
end
increment_cache_index!(work::NNLSTikhonovRegProblemCache{N}) where {N} = (work.idx[] = mod1(work.idx[] + 1, N))
set_cache_index!(work::NNLSTikhonovRegProblemCache{N}, i) where {N} = (work.idx[] = mod1(i, N))
reset_cache!(work::NNLSTikhonovRegProblemCache{N}) where {N} = foreach(w -> mu!(w, NaN), work.cache)
get_cache(work::NNLSTikhonovRegProblemCache) = work.cache[work.idx[]]

function solve!(work::NNLSTikhonovRegProblemCache, μ)
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
#### Chi2 method for choosing Tikhonov regularization parameter
####

struct NNLSChi2RegProblem{T, MC <: AbstractMatrix{T}, Vd <: AbstractVector{T}, W1, W2}
    C::MC
    d::Vd
    m::Int
    n::Int
    nnls_work::W1
    nnls_work_smooth_cache::W2
end
function NNLSChi2RegProblem(C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    m, n = size(C)
    nnls_work = NNLSProblem(C, d)
    nnls_work_smooth_cache = NNLSTikhonovRegProblemCache(C, d)
    NNLSChi2RegProblem(C, d, m, n, nnls_work, nnls_work_smooth_cache)
end

"""
    lsqnonneg_chi2(C::AbstractMatrix, d::AbstractVector, Chi2Factor::Real)

Returns the regularized NNLS solution, X, that incurrs an increase in ``\\chi^2`` approximately by a factor of `Chi2Factor`.
The regularized NNLS problem solved internally is:

```math
X = \\mathrm{argmin}_{x \\ge 0} ||Cx - d||_2^2 + \\mu^2 ||x||_2^2
```

where ``\\mu`` is determined by approximating a solution to the nonlinear equation

```math
\\frac{\\chi^2(\\mu)}{\\chi^2_{min}} = \\mathrm{Chi2Factor}
\\quad
\\text{where}
\\quad
\\chi^2_{min} = \\chi^2(\\mu = 0)
```

# Arguments
- `C::AbstractMatrix`: Decay basis matrix
- `d::AbstractVector`: Decay curve data
- `Chi2Factor::Real`: Desired ``\\chi^2`` increase due to regularization

# Outputs
- `X::AbstractVector`: Regularized NNLS solution
- `mu::Real`: Resulting regularization parameter ``\\mu``
- `Chi2Factor::Real`: Actual increase ``\\chi^2(\\mu)/\\chi^2_{min}``, which will be approximately equal to the input `Chi2Factor`
"""
function lsqnonneg_chi2(C, d, Chi2Factor)
    work = lsqnonneg_chi2_work(C, d)
    lsqnonneg_chi2!(work, Chi2Factor)
end
lsqnonneg_chi2_work(C, d) = NNLSChi2RegProblem(C, d)

function lsqnonneg_chi2!(work::NNLSChi2RegProblem{T}, Chi2Factor::T; bisection = true) where {T}
    # Non-regularized solution
    @timeit_debug TIMER() "Non-Reg. lsqnonneg!" begin
        solve!(work.nnls_work)
        chi2_min = chi2(work.nnls_work)
    end

    # Prepare to solve
    χ²target = Chi2Factor * chi2_min
    reset_cache!(work.nnls_work_smooth_cache)

    @timeit_debug TIMER() "chi2factor search" begin
        if LEGACY[]
            # Use the legacy algorithm: double μ starting from an initial guess, then interpolate the root using a cubic spline fit 
            mu_final, chi2_final = chi2factor_search_from_minimum(chi2_min, Chi2Factor) do μ
                μ == 0 && return chi2_min
                solve!(work.nnls_work_smooth_cache, μ)
                return chi2(get_cache(work.nnls_work_smooth_cache))
            end
            if mu_final == 0
                x_final = solution(work.nnls_work)
            else
                x_final = solve!(work.nnls_work_smooth_cache, mu_final)
            end

        elseif bisection
            # Find bracketing interval containing root, then perform bisection search
            f = function (logμ)
                increment_cache_index!(work.nnls_work_smooth_cache)
                return chi2factor_relerr!(get_cache(work.nnls_work_smooth_cache), logμ; χ²target)
            end
            cache = NamedTuple{(:x, :f), NTuple{2,T}}[]
            a, b, fa, fb = bracketing_interval(f, T(-4.0), T(1.0), T(1.5); maxiters = 6, cache)
            bisect(f, a, b, fa, fb; xtol = T(0.05), ftol = (Chi2Factor-1)/100, cache)

            # Spline rootfinding on evaluated points to improve accuracy
            sort!(cache; by = d -> d.x)
            logmu_root = spline_root([d.x for d in cache], [d.f for d in cache]; deg_spline = 1)
            logmu_root !== nothing && !any(d -> d.x ≈ logmu_root, cache) && cache!(cache, logmu_root, f(logmu_root))

            # Return regularization which minimizes relerr
            _, i = findmin([abs(d.f) for d in cache])
            logmu_final, relerr_final = cache[i]
            mu_final, chi2_final = exp(logmu_final), chi2factor_relerr⁻¹(relerr_final; χ²target)
            x_final = solve!(work.nnls_work_smooth_cache, mu_final)

        else
            # Instead of rootfinding, reformulate as a minimization problem. Solve using NLopt.
            alg = :LN_COBYLA # local, gradient-free, linear approximation of objective
            # alg = :LN_BOBYQA # local, gradient-free, quadratic approximation of objective
            # alg = :GN_AGS # global, gradient-free, hilbert curve based dimension reduction
            # alg = :LN_NELDERMEAD # local, gradient-free, simplex method
            # alg = :LN_SBPLX # local, gradient-free, subspace searching simplex method
            # alg = :LD_CCSAQ # local, first-order (rough ranking: [:LD_MMA, :LD_SLSQP, :LD_LBFGS, :LD_CCSAQ, :LD_AUGLAG])

            opt = NLopt.Opt(alg, 1)
            opt.lower_bounds  = -8.0
            opt.upper_bounds  = 2.0
            opt.xtol_rel      = 0.05
            opt.min_objective = function (logμ, ∇logμ)
                @inbounds _logμ = logμ[1]
                increment_cache_index!(work.nnls_work_smooth_cache)
                loss = chi2factor_loss!(get_cache(work.nnls_work_smooth_cache), _logμ, ∇logμ; χ²target)
                return Float64(loss)
            end
            minf, minx, ret   = NLopt.optimize(opt, [-4.0])

            mu_final = exp(T(minx[1]))
            x_final = solve!(work.nnls_work_smooth_cache, mu_final)
            chi2_final = chi2(get_cache(work.nnls_work_smooth_cache))
        end
    end

    return (x = x_final, mu = mu_final, chi2factor = chi2_final/chi2_min)
end

function chi2factor_search_from_minimum(f, χ²min::T, χ²fact::T, μmin::T = T(1e-3), μfact = T(2.0)) where {T}
    # Minimize energy of spectrum; loop to find largest μ that keeps chi-squared in desired range
    μ_cache = T[zero(T)]
    χ²_cache = T[χ²min]
    μnew = μmin
    while true
        # Cache function value at μ = μnew
        χ²new = f(μnew)
        push!(μ_cache, μnew)
        push!(χ²_cache, χ²new)

        # Break when χ²fact reached, else increase regularization
        (χ²new >= χ²fact * χ²min) && break
        μnew *= μfact
    end

    # Solve χ²(μ) = χ²fact * χ²min using a spline fitting root finding method
    if LEGACY[]
        # Legacy algorithm fits spline to all (μ, χ²) values observed, including for μ=0.
        # This poses several problems:
        #   1) while unlikely, it is possible for the spline to return a negative regularization parameter
        #   2) the μ values are exponentially spaced, leading to poorly conditioned splines
        μ = spline_root(μ_cache, χ²_cache, χ²fact * χ²min)
        μ = μ === nothing ? μmin : μ
    else
        if length(μ_cache) == 2
            # Solution is contained in [0,μmin]; `spline_root` with two points performs root finding via simple linear interpolation
            μ = spline_root(μ_cache, χ²_cache, χ²fact * χ²min; deg_spline = 1)
            μ = μ === nothing ? μmin : μ
        else
            # Perform spline fit on log-log scale on data with μ > 0. This solves the above problems with the legacy algorithm:
            #   1) Root is found in terms of logμ, guaranteeing μ > 0
            #   2) logμ is linearly spaced, leading to well-conditioned splines
            logμ = spline_root(log.(μ_cache[2:end]), log.(χ²_cache[2:end]), log(χ²fact * χ²min); deg_spline = 1)
            μ = logμ === nothing ? μmin : exp(logμ)
        end
    end

    # Compute the final regularized solution
    χ² = f(μ)

    return μ, χ²
end

function chi2factor_search_from_guess(f, χ²min::T, χ²fact::T, μ₀::T = T(1e-2), μfact = T(1.5), μmin::T = T(1e-4)) where {T}
    # Find interval containing χ²target = χ²fact * χ²min
    χ²target = χ²fact * χ²min
    μnew = μ₀
    χ²new = f(μnew)
    logμ_cache = T[log(μnew)]
    logχ²_cache = T[log(χ²new)]

    if χ²new > χ²target
        while μnew > μmin && !(μnew ≈ μmin)
            μnew = max(μnew/μfact, μmin)
            χ²new = f(μnew)
            pushfirst!(logμ_cache, log(μnew))
            pushfirst!(logχ²_cache, log(χ²new))
            (χ²new < χ²target && length(logμ_cache) >= 3) && break
        end
    else
        while true
            μnew *= μfact
            χ²new = f(μnew)
            push!(logμ_cache, log(μnew))
            push!(logχ²_cache, log(χ²new))
            (χ²new > χ²target && length(logμ_cache) >= 3) && break
        end
    end

    if logμ_cache[1] ≈ log(μmin) && logχ²_cache[1] > log(χ²target)
        # μ decreased to μmin but χ²target was not reached; linearly interpolate between μ=0 and μ=μmin points
        μ = spline_root([zero(T), μmin], [log(χ²min), logχ²_cache[1]], log(χ²target))
        μ = μ === nothing ? μmin : μ
        χ² = f(μ)
    else
        # Find optimal μ and evaluate at the interpolated solution
        logμ = spline_root(logμ_cache, logχ²_cache, log(χ²target))
        μ = logμ === nothing ? μmin : exp(logμ)
        χ² = f(μ)
    end

    return μ, χ²
end

####
#### Rootfinding methods. `secant_method` and `bisection_method` are modified codes from Roots.jl:
####
####    https://github.com/JuliaMath/Roots.jl/blob/8a5ff76e8e8305d4ad5719fe1dd665d8a7bd7ec3/src/simple.jl
####
#### The MIT License (MIT) Copyright (c) 2013 John C. Travers
#### Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#### The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#### THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

function secant_method(f, xs; xtol = zero(float(real(first(xs)))), xrtol = 8*eps(one(float(real(first(xs))))), maxiters = 1000, cache = nothing)
    if length(xs) == 1 # secant needs a, b; only a given
        a  = float(xs[1])
        h  = eps(one(real(a)))^(1/3)
        da = h * oneunit(a) + abs(a) * h^2 # adjust for if eps(a) > h
        b  = a + da
    else
        a, b = promote(float(xs[1]), float(xs[2]))
    end
    fa, fb = f(a), f(b)
    cache!(cache, a, fa)
    cache!(cache, b, fb)
    secant(f, a, b, fa, fb; xtol, xrtol, maxiters, cache)
end

function secant(f, a::T, b::T, fa::T, fb::T; xtol = zero(T), xrtol = 8eps(T), maxiters = 1000, cache = nothing) where {T}
    # No function change; return arbitrary endpoint
    if fb == fa
        return (a, fa)
    end

    cnt = 0
    mbest = abs(fa) < abs(fb) ? a : b
    fbest = min(abs(fa), abs(fb))
    uatol = xtol / oneunit(xtol) * oneunit(real(a))
    adjustunit = oneunit(real(fb)) / oneunit(real(b))

    while cnt < maxiters
        m = b - (b - a) * fb / (fb - fa)
        fm = f(m)
        cache!(cache, m, fm)

        abs(fm) < abs(fbest) && ((mbest, fbest) = (m, fm))
        iszero(fm) && return (m, fm)
        isnan(fm) || isinf(fm) && return (mbest, fbest) # function failed; bail out
        abs(fm) <= adjustunit * max(uatol, abs(m) * xrtol) && return (m, fm)
        fm == fb && return (m, fm)

        a, b, fa, fb = b, m, fb, fm
        cnt += 1
    end

    return (mbest, fbest) # maxiters reached
end

function bisection_method(f, a::Number, b::Number; xtol = nothing, xrtol = nothing, ftol = nothing, cache = nothing, maxiters = 1000)
    x₁, x₂ = float.((a,b))
    y₁, y₂ = f(x₁), f(x₂)
    cache!(cache, x₁, y₁)
    cache!(cache, x₂, y₂)

    T = eltype(x₁)
    xtol = xtol === nothing ? zero(T) : abs(xtol)
    xrtol = xrtol === nothing ? zero(one(T)) : abs(xrtol)
    ftol = ftol === nothing ? zero(T) : abs(ftol)

    bisect(f, x₁, x₂, y₁, y₂; xtol, xrtol, ftol, cache, maxiters)
end

function bisect(f, x₁::T, x₂::T, y₁::T, y₂::T; xtol = zero(T), xrtol = zero(one(T)), ftol = zero(T), cache = nothing, maxiters = 1000) where {T}
    # No sign change; return arbitrary endpoint
    if y₁ * y₂ >= 0
        return (x₁, y₁)
    end

    if y₂ < 0
        x₁, x₂, y₁, y₂ = x₂, x₁, y₂, y₁
    end

    xₘ = (x₁ + x₂)/2
    yₘ = f(xₘ)
    cache!(cache, xₘ, yₘ)

    cnt = 1
    while cnt < maxiters
        if iszero(yₘ) || isnan(yₘ) || abs(x₁ - x₂) <= xtol + max(abs(x₁), abs(x₂)) * xrtol || abs(yₘ) <= ftol
            return (xₘ, yₘ)
        end

        if yₘ < 0
            x₁, y₁ = xₘ, yₘ
        else
            x₂, y₂ = xₘ, yₘ
        end

        xₘ = (x₁ + x₂)/2
        yₘ = f(xₘ)
        cache!(cache, xₘ, yₘ)

        cnt += 1
    end

    return (xₘ, yₘ)
end

function bracketing_interval(f, a, δ, dilate = 1; maxiters = 1000, cache = nothing)
    # Initialize cache
    fa = f(a)
    cache!(cache, a, fa)
    cnt = 0
    if fa > 0
        b = a - δ
        fb = f(b)
        cache!(cache, b, fb)
        δ *= dilate
        while fb > 0 && cnt < maxiters
            a, fa = b, fb
            b = a - δ
            fb = f(b)
            cache!(cache, b, fb)
            δ *= dilate
            cnt += 1
        end
    else
        b = a + δ
        fb = f(b)
        cache!(cache, b, fb)
        δ *= dilate
        while fb < 0 && cnt < maxiters
            a, fa = b, fb
            b = a + δ
            fb = f(b)
            cache!(cache, b, fb)
            δ *= dilate
            cnt += 1
        end
    end
    return a, b, fa, fb
end

####
#### L-curve method for choosing Tikhonov regularization parameter
####

struct NNLSLCurveRegProblem{T, MC <: AbstractMatrix{T}, Vd <: AbstractVector{T}, W1, W2}
    C::MC
    d::Vd
    m::Int
    n::Int
    nnls_work::W1
    nnls_work_smooth_cache::W2
end
function NNLSLCurveRegProblem(C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    m, n = size(C)
    nnls_work = NNLSProblem(C, d)
    nnls_work_smooth_cache = NNLSTikhonovRegProblemCache(C, d)
    NNLSLCurveRegProblem(C, d, m, n, nnls_work, nnls_work_smooth_cache)
end

"""
    lsqnonneg_lcurve(C::AbstractMatrix, d::AbstractVector)

Returns the regularized NNLS solution, X, of the equation

```math
X = \\mathrm{argmin}_{x \\ge 0} ||Cx - d||_2^2 + \\mu^2 ||L x||_2^2
```

where ``L`` is the identity matrix and ``\\mu`` is chosen by locating the corner of the "L-curve".

Details of L-curve theory and the Generalized Cross-Validation (GCV) method can be found in:
[Hansen, P.C., 1992. Analysis of Discrete Ill-Posed Problems by Means of the L-Curve. SIAM Review, 34(4), 561-580](https://doi.org/10.1137/1034115)

# Arguments
- `C::AbstractMatrix`: Decay basis matrix
- `d::AbstractVector`: Decay curve data

# Outputs
- `X::AbstractVector`: Regularized NNLS solution
- `mu::Real`: Resulting regularization parameter ``\\mu``
- `Chi2Factor::Real`: Resulting increase in ``\\chi^2`` relative to unregularized (``\\mu = 0``) solution
"""
function lsqnonneg_lcurve(C, d)
    work = lsqnonneg_lcurve_work(C, d)
    lsqnonneg_lcurve!(work)
end
lsqnonneg_lcurve_work(C, d) = NNLSLCurveRegProblem(C, d)

function lsqnonneg_lcurve!(work::NNLSLCurveRegProblem{T,N}) where {T,N}
    # Compute the regularization using the L-curve method
    reset_cache!(work.nnls_work_smooth_cache)
    logmu_bounds = (T(-8), T(2))
    logmu_final = lcurve_corner(logmu_bounds...) do μ
        solve!(work.nnls_work_smooth_cache, μ)
        ξ = log(resnorm(get_cache(work.nnls_work_smooth_cache)))
        η = log(seminorm(get_cache(work.nnls_work_smooth_cache)))
        return SA{T}[ξ, η]
    end

    # Return the final regularized solution
    mu_final = exp(logmu_final)
    x_final = solve!(work.nnls_work_smooth_cache, mu_final)
    x_unreg = solve!(work.nnls_work)
    chi2factor_final = chi2(get_cache(work.nnls_work_smooth_cache)) / chi2(work.nnls_work)

    return (x = x_final, mu = mu_final, chi2factor = chi2factor_final)
end

"""
    lcurve_corner(f, xlow, xhigh)

Find the corner of the L-curve via curvature maximization using Algorithm 1 from:

A. Cultrera and L. Callegaro, “A simple algorithm to find the L-curve corner in the regularization of ill-posed inverse problems”
IOPSciNotes, vol. 1, no. 2, p. 025004, Aug. 2020, doi: 10.1088/2633-1357/abad0d
"""
function lcurve_corner(f, xlow::T = -8.0, xhigh::T = 2.0; xtol::T = 0.05, Ptol::T = 0.05, Ctol::T = 0.01, cache = nothing, refine = false, backtracking = true, verbose = false, kwargs...) where {T}
    # Initialize state
    msg(s, state) = verbose && (@info "$s: [x⃗, P⃗, C⃗] = "; display(hcat(state.x⃗, state.P⃗, [cache[x].C for x in state.x⃗])))
    cache === nothing && (cache = Dict{T, NamedTuple{(:P, :C), Tuple{SVector{2,T}, T}}}())
    state = LCurveCornerState(f, T(xlow), T(xhigh), cache; kwargs...)
    state_cache = backtracking ? [state] : nothing

    # Tolerances are relative to initial curve size
    Ptopleft, Pbottomright = state.P⃗[1], state.P⃗[4]
    Ptol = norm(Ptopleft - Pbottomright) * Ptol
    Ctol = norm(Ptopleft - Pbottomright) * Ctol

    # For very small regularization, points on the L-curve may be extremely close,
    # and curvature calculations will be highly ill-posed. Similarly, while it should
    # not typically occur, additionally check for points on the L-curve becoming
    # extremely close for large regularization. Points which don't satisfy `Pfilter`
    # are assigned curvature -Inf.
    Pfilter = P -> min(norm(P - Ptopleft), norm(P - Pbottomright)) > Ctol
    update_curvature!(state, cache; Pfilter)
    msg("Starting", state)

    while true
        if backtracking
            # Find state with minimum diameter which contains the current best estimate maximum curvature point
            (x, (P, C)), _, _ = mapfindmax(((x, (P, C)),) -> C, cache)
            for s in state_cache
                if (s.x⃗[2] == x || s.x⃗[3] == x) && abs(s.x⃗[4] - s.x⃗[1]) <= abs(state.x⃗[4] - state.x⃗[1])
                    state = s
                end
            end
        end

        # Move state toward region of lower curvature
        if cache[state.x⃗[2]].C > cache[state.x⃗[3]].C
            state = move_left(f, state, cache)
            update_curvature!(state, cache; Pfilter)
            msg("C₂ > C₃; moved left", state)
        else
            state = move_right(f, state, cache)
            update_curvature!(state, cache; Pfilter)
            msg("C₃ ≥ C₂; moved right", state)
        end
        backtracking && push!(state_cache, state)
        is_converged(state; xtol, Ptol) && break
    end

    if refine
        x = refine!(f, state, cache; Pfilter)
        msg("Converged; refined solution", state)
    else
        x = cache[state.x⃗[2]].C > cache[state.x⃗[3]].C ? state.x⃗[2] : state.x⃗[3]
        msg("Converged", state)
    end

    return x
end

struct LCurveCornerState{T, V <: SVector{2,T}}
    x⃗::SVector{4,T} # grid of regularization parameters
    P⃗::SVector{4,V} # points (residual norm, solution seminorm) evaluated at x⃗
end
function LCurveCornerState(f, x₁::T, x₄::T, cache = nothing) where {T}
    x₂   = (T(φ) * x₁ + x₄) / (T(φ) + 1)
    x₃   = x₁ + (x₄ - x₂)
    x⃗    = SA{T}[x₁, x₂, x₃, x₄]
    P⃗    = f.(exp.(x⃗))
    foreach(i -> cache!(cache, x⃗[i], (P = P⃗[i], C = T(-Inf))), 1:4)
    return LCurveCornerState(x⃗, P⃗)
end

is_converged(state::LCurveCornerState; xtol, Ptol) = abs(state.x⃗[4] - state.x⃗[1]) < xtol || norm(state.P⃗[1] - state.P⃗[4]) < Ptol

function maybecall!(f, x::T, state::LCurveCornerState{T}, cache) where {T}
    for (_x, (_P, _C)) in cache
        x ≈ _x && return _P
    end
    P = f(exp(x)) # only recalculate if not cached
    cache!(cache, x, (P = P, C = T(-Inf)))
    return P
end

function move_left(f, state::LCurveCornerState{T}, cache) where {T}
    @unpack x⃗, P⃗ = state
    x⃗ = SVector(x⃗[1], (T(φ) * x⃗[1] + x⃗[3]) / (T(φ) + 1), x⃗[2], x⃗[3])
    P⃗ = SVector(P⃗[1], maybecall!(f, x⃗[2], state, cache), P⃗[2], P⃗[3]) # only P⃗[2] is recalculated
    return LCurveCornerState(x⃗, P⃗)
end

function move_right(f, state::LCurveCornerState{T}, cache) where {T}
    @unpack x⃗, P⃗ = state
    x⃗ = SVector(x⃗[2], x⃗[3], x⃗[2] + (x⃗[4] - x⃗[3]), x⃗[4])
    P⃗ = SVector(P⃗[2], P⃗[3], maybecall!(f, x⃗[3], state, cache), P⃗[4]) # only P⃗[3] is recalculated
    return LCurveCornerState(x⃗, P⃗)
end

function update_curvature!(state::LCurveCornerState{T}, cache; Pfilter = nothing, menger_curvature = true, global_search = false) where {T}
    @unpack x⃗, P⃗ = state
    Cfun(P₋, P, P₊) = menger_curvature ?
        menger(P₋, P, P₊) :
        360 - rad2deg(kahan_angle(P₋, P, P₊))
    for i in 1:4
        x, P, C = x⃗[i], P⃗[i], T(-Inf)
        if Pfilter === nothing || Pfilter(P)
            if global_search
                # Search for maximum curvature over all neighbours
                for (x₋, (P₋, _)) in cache, (x₊, (P₊, _)) in cache
                    (x₋ < x⃗[i] < x₊) && (C = max(C, Cfun(P₋, P, P₊)))
                end
            else
                # Compute curvature from nearest neighbours
                x₋, x₊ = T(-Inf), T(+Inf)
                P₋, P₊ = P, P
                for (_x, (_P, _)) in cache
                    (x₋ < _x < x ) && ((x₋, P₋) = (_x, _P))
                    (x  < _x < x₊) && ((x₊, P₊) = (_x, _P))
                end
                C = Cfun(P₋, P, P₊)
            end
        end
        cache[x] = (; P, C)
    end
    return state
end

function refine!(f, state::LCurveCornerState{T}, cache; Pfilter = nothing, analytical = false) where {T}
    # Fit spline to (negative) curvature estimates on grid and minimize over the spline
    if analytical
        x_opt, _, _ = maximize_curvature(state)
    else
        C_spl = _make_spline(state.x⃗, [-cache[x].C for x in state.x⃗])
        x_opt, _ = spline_opt(C_spl)
    end
    maybecall!(f, x_opt, state, cache)
    update_curvature!(state, cache; Pfilter)
    (x_opt, (_, _)), _, _ = mapfindmax(((x,(P,C)),) -> C, cache)
    return x_opt
end

function menger(Pⱼ::V, Pₖ::V, Pₗ::V) where {V <: SVector{2}}
    Δⱼₖ, Δₖₗ, Δₗⱼ = Pⱼ - Pₖ, Pₖ - Pₗ, Pₗ - Pⱼ
    P̄ⱼP̄ₖ, P̄ₖP̄ₗ, P̄ₗP̄ⱼ = Δⱼₖ ⋅ Δⱼₖ, Δₖₗ ⋅ Δₖₗ, Δₗⱼ ⋅ Δₗⱼ
    Cₖ = 2 * (Δⱼₖ × Δₖₗ) / √(P̄ⱼP̄ₖ * P̄ₖP̄ₗ * P̄ₗP̄ⱼ)
end

function menger(xⱼ::T, xₖ::T, xₗ::T, Pⱼ::V, Pₖ::V, Pₗ::V; interp_uniform = true, linear_deriv = true) where {T, V <: SVector{2,T}}
    if interp_uniform
        h  = min(abs(xₖ - xⱼ), abs(xₗ - xₖ)) / T(φ)
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
        ξ′  = (ξ₊ - ξ₋) / (h₊ + h₋)
        η′  = (η₊ - η₋) / (h₊ + h₋)
    else
        ξ′ = (h₋^2 * ξ₊ + (h₊ + h₋) * (h₊ - h₋) * ξ₀ - h₊^2 * ξ₋) / (h₊ * h₋ * (h₊ + h₋))
        η′ = (h₋^2 * η₊ + (h₊ + h₋) * (h₊ - h₋) * η₀ - h₊^2 * η₋) / (h₊ * h₋ * (h₊ + h₋))
    end

    ξ′′ = 2 * (h₋ * ξ₊ - (h₊ + h₋) * ξ₀ + h₊ * ξ₋) / (h₊ * h₋ * (h₊ + h₋))
    η′′ = 2 * (h₋ * η₊ - (h₊ + h₋) * η₀ + h₊ * η₋) / (h₊ * h₋ * (h₊ + h₋))

    return (ξ′ * η′′ - η′ * ξ′′) / √((ξ′^2 + η′^2)^3)
end

function menger(f; h = 1e-3)
    function menger_curvature_inner(x)
        fⱼ, fₖ, fₗ = f(x-h), f(x), f(x+h)
        Pⱼ, Pₖ, Pₗ = SA[x-h,fⱼ], SA[x,fₖ], SA[x+h,fₗ]
        menger(Pⱼ, Pₖ, Pₗ)
    end
end

function menger(x, y; h = 1e-3)
    function menger_curvature_inner(t)
        x₋, x₀, x₊ = x(t-h), x(t), x(t+h)
        y₋, y₀, y₊ = y(t-h), y(t), y(t+h)
        x′, x′′ = (x₊ - x₋) / 2h, (x₊ - 2x₀ + x₋) / h^2
        y′, y′′ = (y₊ - y₋) / 2h, (y₊ - 2y₀ + y₋) / h^2
        return (x′ * y′′ - y′ * x′′) / √((x′^2 + y′^2)^3)
    end
end

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

function maximize_curvature(state::LCurveCornerState{T}) where {T}
    # Maximize curvature and transform back from t-space to x-space
    @unpack x⃗, P⃗ = state
    t₁, t₂, t₃, t₄ = T(0), 1/T(3), 2/T(3), T(1)
    t_opt, P_opt, C_opt = maximize_curvature(P⃗...)
    x_opt =
        t₁ <= t_opt < t₂ ? lin_interp(t_opt, t₁, t₂, x⃗[1], x⃗[2]) :
        t₂ <= t_opt < t₃ ? lin_interp(t_opt, t₂, t₃, x⃗[2], x⃗[3]) :
        lin_interp(t_opt, t₃, t₄, x⃗[3], x⃗[4])
    return x_opt
end

function maximize_curvature(P₁::V, P₂::V, P₃::V, P₄::V; bezier = true) where {T, V <: SVector{2,T}}
    # Analytically maximize curvature of parametric cubic spline fit to data.
    #   see: https://cs.stackexchange.com/a/131032
    a, e = P₁[1], P₁[2]
    b, f = P₂[1], P₂[2]
    c, g = P₃[1], P₃[2]
    d, h = P₄[1], P₄[2]

    if bezier
        ξ = t -> a*(1-t)^3+3*b*(1-t)^2*t+3*c*(1-t)*t^2+d*t^3
        η = t -> e*(1-t)^3+3*f*(1-t)^2*t+3*g*(1-t)*t^2+h*t^3
        P = t -> SA{T}[ξ(t), η(t)]

        # In order to keep the equations in a more compact form, introduce the following substitutions:
        m = d-3*c+3*b-a
        n = c-2*b+a
        o = b-a
        p = h-3*g+3*f-e
        q = g-2*f+e
        r = f-e
    else
        ξ = t -> a*(t-1/3)*(t-2/3)*(t-1/1)/T((0/1-1/3)*(0/1-2/3)*(0/1-1/1)) + # (-9  * a * t^3)/2  + (+18 * a * t^2)/2 + (-11 * a * t)/2 + a
                 b*(t-0/1)*(t-2/3)*(t-1/1)/T((1/3-0/1)*(1/3-2/3)*(1/3-1/1)) + # (+27 * b * t^3)/2  + (-45 * b * t^2)/2 + (+18 * b * t)/2 +
                 c*(t-0/1)*(t-1/3)*(t-1/1)/T((2/3-0/1)*(2/3-1/3)*(2/3-1/1)) + # (-27 * c * t^3)/2  + (+36 * c * t^2)/2 + ( -9 * c * t)/2 +
                 d*(t-0/1)*(t-1/3)*(t-2/3)/T((1/1-0/1)*(1/1-1/3)*(1/1-2/3))   # (+9  * d * t^3)/2  + ( -9 * d * t^2)/2 + ( +2 * d * t)/2
        η = t -> e*(t-1/3)*(t-2/3)*(t-1/1)/T((0/1-1/3)*(0/1-2/3)*(0/1-1/1)) + # (-9  * e * t^3)/2  + (+18 * e * t^2)/2 + (-11 * e * t)/2 + e
                 f*(t-0/1)*(t-2/3)*(t-1/1)/T((1/3-0/1)*(1/3-2/3)*(1/3-1/1)) + # (+27 * f * t^3)/2  + (-45 * f * t^2)/2 + (+18 * f * t)/2 +
                 g*(t-0/1)*(t-1/3)*(t-1/1)/T((2/3-0/1)*(2/3-1/3)*(2/3-1/1)) + # (-27 * g * t^3)/2  + (+36 * g * t^2)/2 + ( -9 * g * t)/2 +
                 h*(t-0/1)*(t-1/3)*(t-2/3)/T((1/1-0/1)*(1/1-1/3)*(1/1-2/3))   # (+9  * h * t^3)/2  + ( -9 * h * t^2)/2 + ( +2 * h * t)/2
        P = t -> SA{T}[ξ(t), η(t)]

        # In order to keep the equations in a more compact form, introduce the following substitutions:
        m = ( -9 * a + 27 * b - 27 * c + 9 * d)/2
        n = (+18 * a - 45 * b + 36 * c - 9 * d)/6
        o = (-11 * a + 18 * b -  9 * c + 2 * d)/6
        p = ( -9 * e + 27 * f - 27 * g + 9 * h)/2
        q = (+18 * e - 45 * f + 36 * g - 9 * h)/6
        r = (-11 * e + 18 * f -  9 * g + 2 * h)/6
    end

    # This leads to the following simplified derivatives:
    ξ′  = t -> 3*(m*t^2+2*n*t+o)
    ξ′′ = t -> 6*(m*t+n)
    η′  = t -> 3*(p*t^2+2*q*t+r)
    η′′ = t -> 6*(p*t+q)

    # Curvature and its derivative:
    k  = t -> ((3*(m*t^2+2*n*t+o)) * (6*(p*t+q)) - (3*(p*t^2+2*q*t+r)) * (6*(m*t+n))) / ((3*(m*t^2+2*n*t+o))^2 + (3*(p*t^2+2*q*t+r))^2)^(3/2)
    k′ = t -> (-18*m*(p*t^2+2*q*t+r)+18*p*(m*t^2+2*n*t+o)-18*(m*t+n)*(2*p*t+2*q)+18*(2*m*t+2*n)*(p*t+q))/(9*(p*t^2+2*q*t+r)^2+9*(m*t^2+2*n*t+o)^2)^(3/2)-(3*(18*(p*t+q)*(m*t^2+2*n*t+o)-18*(m*t+n)*(p*t^2+2*q*t+r))*(18*(2*p*t+2*q)*(p*t^2+2*q*t+r)+18*(2*m*t+2*n)*(m*t^2+2*n*t+o)))/(2*(9*(p*t^2+2*q*t+r)^2+9*(m*t^2+2*n*t+o)^2)^(5/2))

    # Solve analytically
    coeffs = MVector{6,Complex{T}}(
        (1296*m*p^2+1296*m^3)*q-1296*n*p^3-1296*m^2*n*p,
        (1620*m*p^2+1620*m^3)*r+3240*m*p*q^2+(3240*m^2*n-3240*n*p^2)*q-1620*o*p^3+((-1620*m^2*o)-3240*m*n^2)*p,
        (5184*m*p*q+1296*n*p^2+6480*m^2*n)*r+1296*m*q^3-1296*n*p*q^2+((-6480*o*p^2)-1296*m^2*o+1296*m*n^2)*q+((-5184*m*n*o)-1296*n^3)*p,
        1296*m*p*r^2+(1944*m*q^2+6480*n*p*q-1296*o*p^2+1296*m^2*o+8424*m*n^2)*r-8424*o*p*q^2-6480*m*n*o*q+((-1296*m*o^2)-1944*n^2*o)*p,
        2592*n*p*r^2+(3888*n*q^2-2592*o*p*q+2592*m*n*o+3888*n^3)*r-3888*o*q^3+((-2592*m*o^2)-3888*n^2*o)*q,
        -324*m*r^3+(1944*n*q+324*o*p)*r^2+((-1944*o*q^2)-324*m*o^2+1944*n^2*o)*r-1944*n*o^2*q+324*o^3*p,
    )
    roots = MVector{6,Complex{T}}(undef)
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

function directed_angle(v₁::V, v₂::V) where {T, V <: SVector{2,T}}
    α = atan(v₁[2], v₁[1]) - atan(v₂[2], v₂[1])
    α < 0 ? 2*T(π) + α : α
end
directed_angle(Pⱼ::V, Pₖ::V, Pₗ::V) where {V <: SVector{2}} = directed_angle(Pⱼ - Pₖ, Pₗ - Pₖ)

function kahan_angle(v₁::V, v₂::V) where {T, V <: SVector{2,T}}
    # Kahan's method for computing the angle between v₁ and v₂.
    #   see: https://scicomp.stackexchange.com/a/27694
    a, b, c = norm(v₁), norm(v₂), norm(v₁ - v₂)
    a, b = max(a,b), min(a,b)
    μ = b ≥ c ? c - (a - b) : (b - (a - c))
    num = ((a - b) + c) * max(μ, zero(μ))
    den = (a + (b + c)) * ((a - c) + b)
    α = 2 * atan(√(num / den))
    v₁ × v₂ > 0 ? 2*T(π) - α : α
end
kahan_angle(Pⱼ::V, Pₖ::V, Pₗ::V) where {V <: SVector{2}} = kahan_angle(Pⱼ - Pₖ, Pₗ - Pₖ)

####
#### GCV method for choosing Tikhonov regularization parameter
####

struct NNLSGCVRegProblem{T, MC <: AbstractMatrix{T}, Vd <: AbstractVector{T}, W1, W2}
    C::MC
    d::Vd
    m::Int
    n::Int
    Aμ::Matrix{T}
    C_buf::Matrix{T}
    Ct_buf::Matrix{T}
    CtC_buf::Matrix{T}
    nnls_work::W1
    nnls_work_smooth_cache::W2
end
function NNLSGCVRegProblem(C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    m, n = size(C)
    Aμ = zeros(T, m, m)
    C_buf = zeros(T, m, n)
    Ct_buf = zeros(T, n, m)
    CtC_buf = zeros(T, n, n)
    nnls_work = NNLSProblem(C, d)
    nnls_work_smooth_cache = NNLSTikhonovRegProblemCache(C, d)
    NNLSGCVRegProblem(C, d, m, n, Aμ, C_buf, Ct_buf, CtC_buf, nnls_work, nnls_work_smooth_cache)
end

"""
    lsqnonneg_gcv(C::AbstractMatrix, d::AbstractVector)

Returns the regularized NNLS solution, X, of the equation

```math
X = \\mathrm{argmin}_{x \\ge 0} ||Cx - d||_2^2 + \\mu^2 ||L x||_2^2
```

where ``L`` is the identity matrix and ``\\mu`` is chosen by the Generalized Cross-Validation (GCV) method.

Details of the GCV method and L-curve theory can be can be found in:
[Hansen, P.C., 1992. Analysis of Discrete Ill-Posed Problems by Means of the L-Curve. SIAM Review, 34(4), 561-580](https://doi.org/10.1137/1034115)

# Arguments
- `C::AbstractMatrix`: Decay basis matrix
- `d::AbstractVector`: Decay curve data

# Outputs
- `X::AbstractVector`: Regularized NNLS solution
- `mu::Real`: Resulting regularization parameter ``\\mu``
- `Chi2Factor::Real`: Resulting increase in ``\\chi^2`` relative to unregularized (``\\mu = 0``) solution
"""
function lsqnonneg_gcv(C, d)
    work = lsqnonneg_gcv_work(C, d)
    lsqnonneg_gcv!(work)
end
lsqnonneg_gcv_work(C, d) = NNLSGCVRegProblem(C, d)

function lsqnonneg_gcv!(work::NNLSGCVRegProblem{T,N}) where {T,N}
    # Find μ by minimizing the function G(μ) (GCV method)
    @timeit_debug TIMER() "L-curve Optimization" begin
        reset_cache!(work.nnls_work_smooth_cache)
        # opt = NLopt.Opt(:LN_COBYLA, 1) # local, gradient-free, linear approximation of objective
        opt = NLopt.Opt(:LN_BOBYQA, 1) # local, gradient-free, quadratic approximation of objective
        opt.lower_bounds  = -8.0
        opt.upper_bounds  = 2.0
        opt.xtol_rel      = 0.05
        opt.min_objective = (logμ, ∇logμ) -> Float64(gcv!(work, logμ[1]))
        minf, minx, ret   = NLopt.optimize(opt, [-4.0])
    end

    # Return the final regularized solution
    mu_final = exp(T(minx[1]))
    x_final  = solve!(work.nnls_work_smooth_cache, mu_final)
    x_unreg  = solve!(work.nnls_work)
    chi2factor_final = chi2(get_cache(work.nnls_work_smooth_cache)) / chi2(work.nnls_work)

    return (x = x_final, mu = mu_final, chi2factor = chi2factor_final)
end

# Implements equation (32) from:
# 
#   Analysis of Discrete Ill-Posed Problems by Means of the L-Curve
#   Hansen et al. 1992 (https://epubs.siam.org/doi/10.1137/1034115)
# 
# where here A = C, b = d, λ = μ, and L = identity.
function gcv!(work::NNLSGCVRegProblem, logμ; extract_subproblem = false)
    # Unpack buffers
    @unpack C, d, m, n, Aμ, C_buf, Ct_buf, CtC_buf = work

    # Solve regularized NNLS problem and record chi2 = ||C*X_reg - d||^2 which is returned
    μ = exp(logμ)
    solve!(work.nnls_work_smooth_cache, μ)
    χ² = chi2(get_cache(work.nnls_work_smooth_cache))
    x = solution(get_cache(work.nnls_work_smooth_cache))

    @timeit_debug TIMER() "Aμ" begin
        if extract_subproblem
            # Extract equivalent unconstrained least squares subproblem from NNLS problem
            # by extracting columns of C which correspond to nonzero components of x
            n′ = 0
            for (j,xⱼ) in enumerate(x)
                xⱼ ≈ 0 && continue
                n′ += 1
                @inbounds @simd ivdep for i in 1:m
                    C_buf[i,n′] = C[i,j]
                end
            end
            C′ = reshape(uview(C_buf, 1:m*n′), m, n′)
            Ct′ = reshape(uview(Ct_buf, 1:n′*m), n′, m)
            CtC′ = reshape(uview(CtC_buf, 1:n′*n′), n′, n′)
        else
            # Use full matrix
            C′ = C
            Ct′ = Ct_buf
            CtC′ = CtC_buf
        end

        # Efficient compution of
        #   Aμ = C * (C'C + μ^2*I)^-1 * C'
        # where the matrices have sizes
        #   C: (m,n), Aμ: (m,m), Ct: (n,m), CtC: (n,n)
        mul!(CtC′, C′', C′) # C'C
        @inbounds @simd ivdep for i in 1:n
            CtC′[i,i] += μ^2 # C'C + μ^2*I
        end
        ldiv!(Ct′, cholesky!(Symmetric(CtC′)), C′') # (C'C + μ^2*I)^-1 * C'
        mul!(Aμ, C′, Ct′) # C * (C'C + μ^2*I)^-1 * C'

        # Return Generalized cross-validation. See equations 27 and 32 in
        #   Hansen, P.C., 1992. Analysis of Discrete Ill-Posed Problems by Means of the L-Curve. SIAM Review, 34(4), 561-580
        #   https://doi.org/10.1137/1034115
        trace = m - tr(Aμ) # tr(I - Aμ) = m - tr(Aμ) for m x m matrix Aμ
        gcv = χ² / trace^2 # ||C*X_reg - d||^2 / tr(I - Aμ)^2
    end

    return gcv
end

# Equation (27) from Hansen et al. 1992 (https://epubs.siam.org/doi/10.1137/1034115),
# specialized for L = identity:
# 
#   tr(I_m - A * (A'A + λ^2 * L'L)^-1 * A') = m - n + sum_i λ^2 / (γ_i^2 + λ^2)
# 
# where γ_i are the generalized singular values, which are equivalent to ordinary
# singular values when L = identity, and size(A) = (m,n).
function gcv_tr!(A, λ)
    m, n = size(A)
    γ  = svdvals!(A)
    γ .= λ.^2 ./ (γ.^2 .+ λ.^2)
    max(m - n, 0) + sum(γ)
end
gcv_tr(A, λ) = gcv_tr!(copy(A), λ)
gcv_tr_brute(A, λ) = tr(I - A * ((A'A + λ^2 * I) \ A'))

