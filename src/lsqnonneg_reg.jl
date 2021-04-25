####
#### Tikhonov regularized NNLS problem
####

struct NNLSTikhonovRegProblem{T, MC <: AbstractMatrix{T}, Vd <: AbstractVector{T}, Vμ <: AbstractVector{T}, MCμ <: AbstractMatrix{T}, Vdμ <: AbstractVector{T}, W}
    C::MC
    d::Vd
    m::Int
    n::Int
    μ_diag::Vμ
    C_smooth::MCμ
    d_smooth::Vdμ
    nnls_work::W
end
function NNLSTikhonovRegProblem(C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    m, n = size(C)
    μ_diag = zeros(T, n)
    C_smooth = Vcat(C, Diagonal(μ_diag))
    d_smooth = Vcat(d, zeros(T, n))
    nnls_work = NNLSProblem(C_smooth, d_smooth)
    NNLSTikhonovRegProblem(C, d, m, n, μ_diag, C_smooth, d_smooth, nnls_work)
end

function solve!(work::NNLSTikhonovRegProblem, μ)
    # Solve NNLS problem
    @inbounds work.μ_diag .= μ
    solve!(work.nnls_work)
    return solution(work)
end

residuals!(work::NNLSTikhonovRegProblem) = residuals!(work.nnls_work)

solution(work::NNLSTikhonovRegProblem) = solution(work.nnls_work)

chi2(work::NNLSTikhonovRegProblem) = sum(abs2, uview(work.nnls_work.resid, 1:work.m))

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
        residuals!(work.nnls_work)
        chi2_min = chi2(work.nnls_work)
    end

    @timeit_debug TIMER() "chi2factor search" begin
        if LEGACY[]
            mu_final, chi2_final = chi2factor_search_from_minimum(chi2_min, Chi2Factor) do μ
                solve!(work.nnls_work_smooth, μ)
                residuals!(work.nnls_work_smooth)
                return chi2(work.nnls_work_smooth)
            end
        else
            mu_final, chi2_final = chi2factor_search_from_guess(chi2_min, Chi2Factor) do μ
                solve!(work.nnls_work_smooth, μ)
                residuals!(work.nnls_work_smooth)
                return chi2(work.nnls_work_smooth)
            end
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

"""
    lsqnonneg_reg(C::AbstractMatrix, d::AbstractVector, Chi2Factor::Real)

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
function lsqnonneg_reg(C, d, Chi2Factor)
    work = lsqnonneg_reg_work(C, d)
    lsqnonneg_reg!(work, C, d, Chi2Factor)
end

function lsqnonneg_reg!(work, C::AbstractMatrix{T}, d::AbstractVector{T}, Chi2Factor::T) where {T}
    # Initialize buffers
    lsqnonneg_init!(work, C, d)

    # Non-regularized solution
    @timeit_debug TIMER() "Non-Reg. lsqnonneg!" begin
        chi2_min = lsqnonneg_solve!(work, C, d)
    end

    @timeit_debug TIMER() "chi2factor search" begin
        if LEGACY[]
            mu_final, chi2_final = chi2factor_search_from_minimum(chi2_min, Chi2Factor) do μ
                lsqnonneg_solve!(work, C, d, μ)
            end
        else
            mu_final, chi2_final = chi2factor_search_from_guess(chi2_min, Chi2Factor) do μ
                lsqnonneg_solve!(work, C, d, μ)
            end
        end
    end

####
#### L-curve method for choosing Tikhonov regularization parameter
####

"""
    lsqnonneg_lcurve(C::AbstractMatrix, d::AbstractVector)

Returns the regularized NNLS solution, X, of the equation

```math
X = \\mathrm{argmin}_{x \\ge 0} ||Cx - d||_2^2 + \\mu^2 ||H x||_2^2
```

where ``H`` is the identity matrix and ``\\mu`` is chosen by the L-curve theory using the Generalized Cross-Validation method.
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
        residuals!(work.nnls_work)
        chi2_min = chi2(work.nnls_work)
    end

    # Compute the final regularized solution
    @timeit_debug TIMER() "Final Reg. lsqnonneg!" begin
        solve!(work.nnls_work_smooth, mu_final)
        residuals!(work.nnls_work_smooth)
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
    residuals!(work.nnls_work_smooth)
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
