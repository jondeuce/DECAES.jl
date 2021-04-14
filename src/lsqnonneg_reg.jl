####
#### Utilities for regularized NNLS
####

# Work buffers
function lsqnonneg_reg_work(C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    @assert size(C,1) == length(d)
    m, n = size(C) # m = output dim (num echoes), n = input dim (num basis functions)
    d_backproj = zeros(T, m)
    resid = zeros(T, m)
    nnls_work = lsqnonneg_work(C, d)
    C_smooth = zeros(T, m + n, n)
    d_smooth = zeros(T, m + n)
    nnls_work_smooth = lsqnonneg_work(C_smooth, d_smooth)
    A_mu = zeros(T, m, m)
    Ct_tmp = zeros(T, n, m)
    CtC_tmp = zeros(T, n, n)
    x = zeros(T, n)
    mu_opt = Ref(T(NaN))
    chi2fact_opt = Ref(T(NaN))
    return @ntuple(
        d_backproj, resid, nnls_work,
        C_smooth, d_smooth, nnls_work_smooth,
        A_mu, Ct_tmp, CtC_tmp,
        x, mu_opt, chi2fact_opt
    )
end

function lsqnonneg_init!(work, C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    @unpack C_smooth, d_smooth = work
    m, n = size(C)
    @assert length(d_smooth) == m+n && length(d) == m
    @assert size(C_smooth) == (m+n, n)

    # Assign top and bottom of C_smooth, d_smooth
    uview(C_smooth, 1:m, :) .= C
    uview(C_smooth, m.+(1:n), :) .= 0
    uview(d_smooth, 1:m) .= d
    uview(d_smooth, m.+(1:n)) .= 0

    return nothing
end

function lsqnonneg_solve!(work, C::AbstractMatrix{T}, d::AbstractVector{T}, mu = nothing) where {T}
    @unpack nnls_work, d_backproj, resid = work
    @unpack nnls_work_smooth, C_smooth, d_smooth = work
    m, n = size(C)

    # Solve
    if mu === nothing
        # Non-regularized distribution
        nnls_workspace, A, b = work.nnls_work, C, d
    else
        # Regularized distribution
        nnls_workspace, A, b = work.nnls_work_smooth, C_smooth, d_smooth
        set_diag!(uview(A, m.+(1:n), :), mu)
    end
    lsqnonneg!(nnls_workspace, A, b)

    # Find predicted curve and calculate residuals and chi-squared
    mul!(d_backproj, C, nnls_workspace.x)
    resid .= d .- d_backproj
    chi2 = sum(abs2, resid)

    return chi2
end

function chi2factor_search_from_minimum(f, χ²min::T, χ²fact::T, μmin::T = T(1e-3), μfact = T(2.0)) where {T}
    # Minimize energy of spectrum; loop to find largest mu that keeps chi-squared in desired range
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

    # Assign output
    work.x .= work.nnls_work_smooth.x
    work.mu_opt[] = mu_final
    work.chi2fact_opt[] = chi2_final/chi2_min

    return (x = work.x, mu = work.mu_opt[], chi2factor = work.chi2fact_opt[])
end

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
    work = lsqnonneg_reg_work(C, d)
    lsqnonneg_lcurve!(work, C, d)
end

lsqnonneg_lcurve!(work, C, d) = lsqnonneg_gcv!(work, C, d) #TODO

####
#### GCV
####

function lsqnonneg_gcv(C, d)
    work = lsqnonneg_reg_work(C, d)
    lsqnonneg_gcv!(work, C, d)
end

function lsqnonneg_gcv!(work, C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    # Initialize buffers
    lsqnonneg_init!(work, C, d)

    # Find mu by minimizing the function G(mu) (GCV method)
    @timeit_debug TIMER() "L-curve Optimization" begin
        opt = NLopt.Opt(:LN_BOBYQA, 1)
        opt.lower_bounds  = [sqrt(eps(T))]
        opt.upper_bounds  = [T(0.1)]
        opt.xtol_rel      = sqrt(eps(T))
        opt.min_objective = (μ, ∇μ) -> gcv(μ[1], C, d, work)
        minf, minx, ret   = NLopt.optimize(opt, [T(1e-2)])
        mu = minx[1]
    end

    # Compute the final regularized solution
    @timeit_debug TIMER() "Final Reg. lsqnonneg!" begin
        chi2_final = lsqnonneg_solve!(work, C, d, mu)
    end

    # Non-regularized solution
    @timeit_debug TIMER() "Non-Reg. lsqnonneg!" begin
        chi2_min = lsqnonneg_solve!(work, C, d)
    end

    # Assign output
    work.x .= work.nnls_work_smooth.x
    work.mu_opt[] = mu
    work.chi2fact_opt[] = chi2_final/chi2_min

    return (x = work.x, mu = work.mu_opt[], chi2factor = work.chi2fact_opt[])
end

function gcv(μ, C, d, work)
    # Unpack buffers
    @unpack A_mu, Ct_tmp, CtC_tmp = work

    # Solve regularized NNLS problem and record chi2 = ||C*X_reg - d||^2 which is returned
    chi2 = lsqnonneg_solve!(work, C, d, mu)

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
