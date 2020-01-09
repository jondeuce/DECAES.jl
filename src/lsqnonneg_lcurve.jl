"""
    lsqnonneg_lcurve(C::AbstractMatrix, d::AbstractVector)

Returns the regularized NNLS solution, X, of the equation

```math
X = \\mathrm{argmin}_{x \\ge 0} ||Cx - d||_2^2 + \\mu||H x||_2^2
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
    lsqnonneg_lcurve!(work, C, d)
end

function lsqnonneg_lcurve_work(C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    @assert size(C,1) == length(d)
    d_backproj = zeros(T, length(d))
    resid = zeros(T, length(d))
    nnls_work = lsqnonneg_work(C, d)
    C_smooth = zeros(T, size(C,1) + size(C,2), size(C,2))
    d_smooth = zeros(T, length(d) + size(C,2))
    nnls_work_smooth = lsqnonneg_work(C_smooth, d_smooth)
    @views C_smooth_top, d_smooth_top = C_smooth[1:size(C,1), :], d_smooth[1:length(d)]
    @views C_smooth_bottom, d_smooth_bottom = C_smooth[size(C,1)+1:end, :], d_smooth[length(d)+1:end]
    A_mu = zeros(T, size(C,1), size(C,1))
    Ct_tmp = zeros(T, size(C,2), size(C,1))
    CtC_tmp = zeros(T, size(C,2), size(C,2))
    x = zeros(T, size(C,2))
    mu_opt = Ref(T(NaN))
    chi2fact_opt = Ref(T(NaN))
    return @ntuple(
        d_backproj, resid, nnls_work,
        C_smooth, d_smooth, nnls_work_smooth,
        C_smooth_top, C_smooth_bottom, d_smooth_bottom, d_smooth_top,
        A_mu, Ct_tmp, CtC_tmp,
        x, mu_opt, chi2fact_opt
    )
end

function lsqnonneg_lcurve!(work, C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    # Unpack workspace
    @unpack nnls_work, d_backproj, resid = work
    @unpack nnls_work_smooth, C_smooth, d_smooth = work
    @unpack C_smooth_top, C_smooth_bottom, d_smooth_bottom, d_smooth_top = work

    # Assign top and bottom of C_smooth, d_smooth
    @assert size(C,1) == length(d)
    @assert size(C_smooth,1) == length(d_smooth) == size(C,1) + size(C,2)
    C_smooth_top .= C; C_smooth_bottom .= 0
    d_smooth_top .= d; d_smooth_bottom .= 0

    # Find mu by minimizing the function G(mu) (GCV method)
    mu = @timeit_debug TIMER "L-curve Optimization" begin
        opt_result = Optim.optimize(μ -> lcurve_GCV(μ, C, d, work), T(0), T(0.1), Optim.Brent(); rel_tol = T(1e-6))
        Optim.minimizer(opt_result)
    end

    # Compute the regularized solution
    @timeit_debug TIMER "Reg. lsqnonneg!" begin
        set_diag!(C_smooth_bottom, mu)
        lsqnonneg!(nnls_work_smooth, C_smooth, d_smooth)
        mul!(d_backproj, C, nnls_work_smooth.x)
        resid .= d .- d_backproj
        chi2_final = sum(abs2, resid)
    end

    # Find non-regularized solution
    @timeit_debug TIMER "Non-reg. lsqnonneg!" begin
        lsqnonneg!(nnls_work, C, d)
        mul!(d_backproj, C, nnls_work.x)
        resid .= d .- d_backproj
        chi2_min = sum(abs2, resid)
    end

    # Assign output
    work.x .= nnls_work_smooth.x
    work.mu_opt[] = mu
    work.chi2fact_opt[] = chi2_final/chi2_min

    return (x = work.x, mu = work.mu_opt[], chi2factor = work.chi2fact_opt[])
end

function lcurve_GCV(μ, C, d, work)
    @unpack d_backproj, resid = work
    @unpack nnls_work_smooth, C_smooth, C_smooth_bottom, d_smooth = work
    @unpack A_mu, Ct_tmp, CtC_tmp = work

    # C_smooth = [C; μ * I], d_smooth = [d; 0],
    # X_reg = lsqnonneg(C_smooth, d_smooth), resid = d - C * X_reg
    @timeit_debug TIMER "lsqnonneg!" begin
        set_diag!(C_smooth_bottom, μ)
        lsqnonneg!(nnls_work_smooth, C_smooth, d_smooth)
        mul!(d_backproj, C, nnls_work_smooth.x)
        resid .= d .- d_backproj
    end

    # Efficient compution of
    #   A_mu = C * (C'C + μ*I)^-1 * C'
    # where the matrices have sizes
    #   C: (m,n), A_mu: (m,m), Ct_tmp: (n,m), CtC_tmp: (n,n)
    @timeit_debug TIMER "A_mu" begin
        mul!(CtC_tmp, C', C) # C'C
        @inbounds for i in 1:size(CtC_tmp,1)
            CtC_tmp[i,i] += μ # C'C + μ*I
        end
        ldiv!(Ct_tmp, lu!(CtC_tmp), C') # (C'C + μ*I)^-1 * C'
        mul!(A_mu, C, Ct_tmp) # C * (C'C + μ*I)^-1 * C'
    end

    # Compute trace(I - A_mu), avoiding the matrix subtraction
    trace = zero(eltype(A_mu))
    @inbounds for i in 1:size(A_mu,1) # tr(I - A_mu)
        trace += (1 - A_mu[i,i])
    end

    # Return ||C*X_reg - d||^2 / tr(I - A_mu)^2
    gcv = sum(abs2, resid) / trace^2

    return gcv
end