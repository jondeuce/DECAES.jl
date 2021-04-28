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
    nnls_work = NNLSProblem(TikhonovPaddedMatrix(C, zero(T)), PaddedVector(d, n))
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

loss(work::NNLSTikhonovRegProblem) = chi2(work.nnls_work)

function regularization(work::NNLSTikhonovRegProblem)
    μ = mu(work)
    x = solution(work)
    return μ^2 * (x ⋅ x)
end

chi2(work::NNLSTikhonovRegProblem) = max(loss(work) - regularization(work), 0)

# Extract columns spanning solution space of regularized NNLS problem
function subproblem(work::NNLSTikhonovRegProblem{T}) where {T}
    @unpack C, d, m, n, nnls_work = work
    μ = mu(work)
    x = solution(work)
    Jnz = findall(!≈(0), x)
    nnz = length(Jnz)
    return TikhonovPaddedMatrix(C[:,Jnz], μ), PaddedVector(d, nnz), x[Jnz]
end

function ∇regularization(work::NNLSTikhonovRegProblem)
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
    relerr = χ² / χ²target - 1
    if ∇logμ !== nothing && length(∇logμ) > 0
        ∂χ²_∂logμ = ∇regularization(work)
        ∂relerr_∂logμ = μ * ∂χ²_∂logμ / χ²target
        @inbounds ∇logμ[1] = ∂relerr_∂logμ
    end
    return relerr
end

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


####
#### Chi2 method for choosing Tikhonov regularization parameter
####

struct NNLSChi2RegProblem{T, MC <: AbstractMatrix{T}, Vd <: AbstractVector{T}, W1, W2}
    C::MC
    d::Vd
    m::Int
    n::Int
    nnls_work::W1
    nnls_work_smooth::W2
end
function NNLSChi2RegProblem(C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    m, n = size(C)
    nnls_work = NNLSProblem(C, d)
    nnls_work_smooth = NNLSTikhonovRegProblem(C, d)
    NNLSChi2RegProblem(C, d, m, n, nnls_work, nnls_work_smooth)
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

function lsqnonneg_chi2!(work::NNLSChi2RegProblem{T}, Chi2Factor::T) where {T}
    # Non-regularized solution
    @timeit_debug TIMER() "Non-Reg. lsqnonneg!" begin
        solve!(work.nnls_work)
        chi2_min = chi2(work.nnls_work)
    end

    @timeit_debug TIMER() "chi2factor search" begin
        if LEGACY[]
            mu_final, chi2_final = chi2factor_search_from_minimum(chi2_min, Chi2Factor) do μ
                solve!(work.nnls_work_smooth, μ)
                return chi2(work.nnls_work_smooth)
            end
        else
            # Find bracketing interval, then bisect
            f = logμ -> chi2factor_relerr!(work.nnls_work_smooth, logμ; χ²target = Chi2Factor * chi2_min)
            cache = NamedTuple{(:x, :f), NTuple{2,T}}[]
            a, b, fa, fb = bracketing_interval(f, T(-4.0), T(1.0), T(1.5); maxiters = 6, cache)
            bisect(f, a, b, fa, fb; atol = T(0.25), cache)

            # Spline rootfinding on evaluated points to improve accuracy
            sort!(cache; by = d -> d.x)
            logmu_root = spline_root([d.x for d in cache], [d.f for d in cache]; deg_spline = 1)
            logmu_root !== nothing && cache!(cache, logmu_root, f(logmu_root))

            # Return regularization which minimizes relerr
            _, i = findmin([abs(d.f) for d in cache])
            logmu_final, relerr_final = cache[i]
            mu_final, chi2_final = exp(logmu_final), Chi2Factor * chi2_min * (relerr_final + 1)

            #= Optimize using NLopt
            # Gradient-free:
            opt = NLopt.Opt(:LN_COBYLA, 1) # local, gradient-free, linear approximation of objective
            # opt = NLopt.Opt(:LN_BOBYQA, 1) # local, gradient-free, quadratic approximation of objective
            # opt = NLopt.Opt(:GN_AGS, 1)
            # opt = NLopt.Opt(:LN_NELDERMEAD, 1) # local, gradient-free, linear approximation of objective
            # opt = NLopt.Opt(:LN_SBPLX, 1) # local, gradient-free, linear approximation of objective

            # First-order methods
            #   -Rough algorithm ranking: [:LD_MMA, :LD_SLSQP, :LD_LBFGS, :LD_CCSAQ, :LD_AUGLAG]
            # opt = NLopt.Opt(:LD_CCSAQ, 1)

            opt.lower_bounds  = [log(eps(T))]
            opt.upper_bounds  = [log(one(T))]
            opt.maxeval       = 10
            μbest, χ²best = Ref(zero(T)), Ref(T(Inf))
            opt.min_objective = function (logμ, ∇logμ)
                @inbounds _logμ = logμ[1]
                χ²target = Chi2Factor * chi2_min
                loss = chi2factor_loss!(work.nnls_work_smooth, _logμ, ∇logμ; χ²target)
                if abs(χ² - χ²target) < abs(χ²best[] - χ²target)
                    μbest[] = exp(_logμ)
                    χ²best[] = chi2(work.nnls_work_smooth)
                end
                return loss
            end
            NLopt.optimize(opt, [log(T(1e-2))])
            mu_final, chi2_final = μbest[], χ²best[]
            =#
        end
    end

    return (x = solution(work.nnls_work_smooth), mu = mu_final, chi2factor = chi2_final/chi2_min)
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
            μ = spline_root(μ_cache, χ²_cache, χ²fact * χ²min)
            μ = μ === nothing ? μmin : μ
        else
            # Perform spline fit on log-log scale on data with μ > 0. This solves the above problems with the legacy algorithm:
            #   1) Root is found in terms of logμ, guaranteeing μ > 0
            #   2) logμ is linearly spaced, leading to well-conditioned splines
            logμ = @views spline_root(log.(μ_cache[2:end]), log.(χ²_cache[2:end]), log(χ²fact * χ²min))
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
        while true
            (logμ_cache[1] ≈ μmin) && break # in case μmin is approximately μ₀/μfact^k for some k, e.g. μ₀ = 1e-2, μfact = 10, μmin = 1e-4
            μnew = max(μnew/μfact, μmin)
            χ²new = f(μnew)
            pushfirst!(logμ_cache, log(μnew))
            pushfirst!(logχ²_cache, log(χ²new))
            ((χ²new < χ²target && length(logμ_cache) >= 3) || μnew ≈ μmin) && break
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

function secant_method(f, xs; atol = zero(float(real(first(xs)))), rtol = 8*eps(one(float(real(first(xs))))), maxiters = 1000, cache = nothing)

    if length(xs) == 1 # secant needs x0, x1; only x0 given
        a = float(xs[1])
        h = eps(one(real(a)))^(1/3)
        da = h * oneunit(a) + abs(a) * h^2 # adjust for if eps(a) > h
        b = a + da
    else
        a, b = promote(float(xs[1]), float(xs[2]))
    end
    fa, fb = f(a), f(b)
    cache!(cache, a, fa)
    cache!(cache, b, fb)
    secant(f, a, b, fa, fb; atol, rtol, maxiters, cache)
end

function secant(f, a::T, b::T, fa::T, fb::T; atol = zero(T), rtol = 8eps(T), maxiters = 1000, cache = nothing) where {T}

    # No function change; return arbitrary endpoint
    fb == fa && return (a, fa)

    cnt = 0
    mbest = abs(fa) < abs(fb) ? a : b
    fbest = min(abs(fa), abs(fb))
    uatol = atol / oneunit(atol) * oneunit(real(a))
    adjustunit = oneunit(real(fb)) / oneunit(real(b))

    while cnt < maxiters
        m = b - (b - a) * fb / (fb - fa)
        fm = f(m)
        cache!(cache, m, fm)

        abs(fm) < abs(fbest) && ((mbest, fbest) = (m, fm))
        iszero(fm) && return (m, fm)
        isnan(fm) || isinf(fm) && return (mbest, fbest) # function failed; bail out
        abs(fm) <= adjustunit * max(uatol, abs(m) * rtol) && return (m, fm)
        fm == fb && return (m, fm)

        a, b, fa, fb = b, m, fb, fm
        cnt += 1
    end

    return (mbest, fbest) # maxiters reached
end

function bisection_method(f, a::Number, b::Number; atol = nothing, rtol = nothing, cache = nothing, maxiters = 1000)

    x1, x2 = float.((a,b))
    y1, y2 = f(x1), f(x2)
    cache!(cache, x1, y1)
    cache!(cache, x2, y2)

    T = eltype(x1)
    atol = atol === nothing ? zero(T) : abs(atol)
    rtol = rtol === nothing ? zero(one(T)) : abs(rtol)

    bisect(f, x1, x2, y1, y2; atol, rtol, cache, maxiters)
end

function bisect(f, x1::T, x2::T, y1::T, y2::T; atol = zero(T), rtol = zero(one(T)), cache = nothing, maxiters = 1000) where {T}

    y1 * y2 >= 0 && return (x1, y1) # arbitrary

    if y2 < 0
        x1, x2, y1, y2 = x2, x1, y2, y1
    end

    xm = (x1+x2)/2
    ym = f(xm)
    cache!(cache, xm, ym)

    cnt = 1
    while cnt < maxiters

        if iszero(ym) || isnan(ym) || abs(x1 - x2) <= atol + max(abs(x1), abs(x2)) * rtol
            return (xm, ym)
        end

        if ym < 0
            x1, y1 = xm, ym
        else
            x2, y2 = xm, ym
        end

        xm = (x1+x2)/2
        ym = f(xm)
        cache!(cache, xm, ym)

        cnt += 1
    end

    return (xm, ym)
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

cache!(cache, x, f) = cache !== nothing && push!(cache, (; x, f))

####
#### L-curve method for choosing Tikhonov regularization parameter
####

"""
    lsqnonneg_lcurve(C::AbstractMatrix, d::AbstractVector)

Returns the regularized NNLS solution, X, of the equation

```math
X = \\mathrm{argmin}_{x \\ge 0} ||Cx - d||_2^2 + \\mu^2 ||L x||_2^2
```

where ``L`` is the identity matrix and ``\\mu`` is chosen by the L-curve theory using the Generalized Cross-Validation method.
Details of L-curve and GCV methods can be found in:
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
lsqnonneg_lcurve!(work) = lsqnonneg_gcv!(work) #TODO
lsqnonneg_lcurve_work(C, d) = lsqnonneg_gcv_work(C, d)

####
#### GCV method for choosing Tikhonov regularization parameter
####

struct NNLSGCVRegProblem{T, MC <: AbstractMatrix{T}, Vd <: AbstractVector{T}, W1, W2}
    C::MC
    d::Vd
    m::Int
    n::Int
    nnls_work::W1
    nnls_work_smooth::W2
end
function NNLSGCVRegProblem(C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    m, n = size(C)
    nnls_work = NNLSProblem(C, d)
    nnls_work_smooth = NNLSTikhonovRegProblem(C, d)
    NNLSGCVRegProblem(C, d, m, n, nnls_work, nnls_work_smooth)
end

function lsqnonneg_gcv(C, d)
    work = lsqnonneg_gcv_work(C, d)
    lsqnonneg_gcv!(work)
end
lsqnonneg_gcv_work(C, d) = NNLSGCVRegProblem(C, d)

function lsqnonneg_gcv!(work::NNLSGCVRegProblem{T}) where {T}
    # Find μ by minimizing the function G(μ) (GCV method)
    @timeit_debug TIMER() "L-curve Optimization" begin
        opt = NLopt.Opt(:LN_BOBYQA, 1)
        opt.lower_bounds  = [sqrt(eps(T))]
        opt.upper_bounds  = [T(0.1)]
        opt.xtol_rel      = sqrt(eps(T))
        opt.min_objective = (μ, ∇μ) -> gcv!(work, μ[1])
        minf, minx, ret   = NLopt.optimize(opt, [T(1e-2)])
        mu_final = minx[1]
    end

    # Non-regularized solution
    @timeit_debug TIMER() "Non-Reg. lsqnonneg!" begin
        solve!(work.nnls_work)
        chi2_min = chi2(work.nnls_work)
    end

    # Compute the final regularized solution
    @timeit_debug TIMER() "Final Reg. lsqnonneg!" begin
        solve!(work.nnls_work_smooth, mu_final)
        chi2_final = chi2(work.nnls_work_smooth)
    end

    return (x = solution(work.nnls_work_smooth), mu = mu_final, chi2factor = chi2_final/chi2_min)
end

# Implements equation (32) from:
#   Analysis of Discrete Ill-Posed Problems by Means of the L-Curve
#   Hansen et al. 1992 (https://epubs.siam.org/doi/10.1137/1034115)
# 
# Notation dictionary (this function -> the above work)
#   C -> A
#   d -> b
#   μ -> λ
#   I -> L (i.e. L == identity here)
function gcv!(work::NNLSGCVRegProblem, μ)
    # Unpack buffers
    @unpack C, d, m, n = work
    A_mu = zeros(T, m, m)
    Ct_tmp = zeros(T, n, m)
    CtC_tmp = zeros(T, n, n)

    # Solve regularized NNLS problem and record chi2 = ||C*X_reg - d||^2 which is returned
    solve!(work.nnls_work_smooth, μ)
    chi2 = chi2(work.nnls_work_smooth)

    # Efficient compution of
    #   A_mu = C * (C'C + μ^2*I)^-1 * C'
    # where the matrices have sizes
    #   C: (m,n), A_mu: (m,m), Ct_tmp: (n,m), CtC_tmp: (n,n)
    @timeit_debug TIMER() "A_mu" begin
        m, n = size(C)
        mul!(CtC_tmp, C', C) # C'C
        @inbounds for i in 1:n
            CtC_tmp[i,i] += μ^2 # C'C + μ^2*I
        end
        ldiv!(Ct_tmp, cholesky!(Symmetric(CtC_tmp)), C') # (C'C + μ^2*I)^-1 * C'
        mul!(A_mu, C, Ct_tmp) # C * (C'C + μ^2*I)^-1 * C'
    end

    # Return Generalized cross-validation. See equations 27 and 32 in
    #   Hansen, P.C., 1992. Analysis of Discrete Ill-Posed Problems by Means of the L-Curve. SIAM Review, 34(4), 561-580
    #   https://doi.org/10.1137/1034115
    trace = m - tr(A_mu) # tr(I - A_mu) = m - tr(A_mu) for m x m matrix A_mu
    gcv = chi2 / trace^2 # ||C*X_reg - d||^2 / tr(I - A_mu)^2

    return gcv
end
