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

function lsqnonneg_reg_work(C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    @assert size(C,1) == length(d)
    d_backproj = zeros(T, length(d))
    resid = zeros(T, length(d))
    nnls_work = lsqnonneg_work(C, d)
    C_smooth = zeros(T, size(C,1) + size(C,2), size(C,2))
    d_smooth = zeros(T, length(d) + size(C,2))
    nnls_work_smooth = lsqnonneg_work(C_smooth, d_smooth)
    @views C_smooth_top, d_smooth_top = C_smooth[1:size(C,1), :], d_smooth[1:length(d)]
    @views C_smooth_bottom, d_smooth_bottom = C_smooth[size(C,1)+1:end, :], d_smooth[length(d)+1:end]
    x = zeros(T, size(C,2))
    mu_opt = Ref(T(NaN))
    chi2fact_opt = Ref(T(NaN))
    return @ntuple(
        d_backproj, resid, nnls_work,
        C_smooth, d_smooth, nnls_work_smooth,
        C_smooth_top, C_smooth_bottom, d_smooth_bottom, d_smooth_top,
        x, mu_opt, chi2fact_opt
    )
end

function lsqnonneg_reg!(work, C::AbstractMatrix{T}, d::AbstractVector{T}, Chi2Factor::T) where {T}
    # Unpack workspace
    @unpack nnls_work, d_backproj, resid = work
    @unpack nnls_work_smooth, C_smooth, d_smooth = work
    @unpack C_smooth_top, C_smooth_bottom, d_smooth_bottom, d_smooth_top = work

    # Assign top and bottom of C_smooth, d_smooth
    @assert size(C,1) == length(d)
    @assert size(C_smooth,1) == length(d_smooth) == size(C,1) + size(C,2)
    C_smooth_top .= C; C_smooth_bottom .= 0
    d_smooth_top .= d; d_smooth_bottom .= 0

    # Non-regularized solution
    nnls_work.x .= 0
    @timeit_debug TIMER() "Non-Reg. lsqnonneg!" begin
        lsqnonneg!(nnls_work, C, d)
    end
    mul!(d_backproj, C, nnls_work.x)
    resid .= d .- d_backproj
    chi2_min = sum(abs2, resid)

    # Initialzation of various components
    nnls_work_smooth.x .= nnls_work.x
    mu_cache = [zero(T)]
    chi2_cache = [chi2_min]

    # Minimize energy of spectrum; loop to find largest mu that keeps chi-squared in desired range
    while chi2_cache[end] ≤ Chi2Factor * chi2_min
        # Chi2Factor not reached; push larger mu to mu_cache
        push!(mu_cache, mu_cache[end] > 0 ? 2*mu_cache[end] : T(0.001))

        # Compute T2 distribution with smoothing
        set_diag!(C_smooth_bottom, mu_cache[end])
        @timeit_debug TIMER() "Loop Reg. lsqnonneg!" begin
            lsqnonneg!(nnls_work_smooth, C_smooth, d_smooth)
        end

        # Find predicted curve and calculate residuals and chi-squared
        mul!(d_backproj, C, nnls_work_smooth.x)
        resid .= d .- d_backproj
        push!(chi2_cache, sum(abs2, resid))
    end

    # Smooth the chi2(mu) curve using a spline fit and find the mu value such
    # that chi2 increases by Chi2Factor, i.e. chi2(mu) = Chi2Factor * chi2_min
    @timeit_debug TIMER() "spline_root" begin
        mu = spline_root(mu_cache, chi2_cache, Chi2Factor * chi2_min)
    end

    # Compute the regularized solution
    set_diag!(C_smooth_bottom, mu)
    @timeit_debug TIMER() "Final Reg. lsqnonneg!" begin
        lsqnonneg!(nnls_work_smooth, C_smooth, d_smooth)
    end
    mul!(d_backproj, C, nnls_work_smooth.x)
    resid .= d .- d_backproj
    chi2_final = sum(abs2, resid)

    # Assign output
    work.x .= nnls_work_smooth.x
    work.mu_opt[] = mu
    work.chi2fact_opt[] = chi2_final/chi2_min

    return (x = work.x, mu = work.mu_opt[], chi2factor = work.chi2fact_opt[])
end
