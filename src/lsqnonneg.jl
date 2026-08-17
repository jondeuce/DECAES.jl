####
#### Unregularized NNLS problem
####

struct NNLSProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W}
    A::TA # decay basis matrix
    b::Tb # decay curve data
    m::Int # number of rows of A
    n::Int # number of columns of A
    nnls_work::W # underlying NNLS solver workspace
end
function NNLSProblem(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    nnls_work = NNLS.NNLSWorkspace(A, b)
    return NNLSProblem(A, b, m, n, nnls_work)
end

# Solve NNLS problem
solve!(work::NNLSProblem, args...; kwargs...) = solve!(work, work.A, work.b, args...; kwargs...)
# solve!(work::NNLSProblem, A::AbstractMatrix, b::AbstractVector, args...; kwargs...) = NNLS.nnls!(work.nnls_work, A, b, args...; kwargs...)

# The nnls algorithm selects candidate x[j] based on the largest negative gradient of ||Ax - b||, i.e. j = argmax_j w[j] where w = -A'(Ax - b) is the dual vector.
# In DECAES, the initial dual vector w_0 = A'b is sorted because A[i, j], b[j] >= 0 and A[i, j+1] > A[i, j], and thus the last column of A will always be chosen first.
# Thence j = n and we can bypass the first iteration and initialize the gradient with x_0 = [0; x[n]], where x[n] >= 0 due to the nonnegativity of A, b.
#   NOTE: This will not fail even for generic A and b, it just forces NNLS to start with column j = n. From there, it may remove the column if necessary.
function solve!(
    work::NNLSProblem{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T};
    kwargs...,
) where {T}
    m, n = size(A)
    f = work.nnls_work.b
    x = work.nnls_work.x
    w = work.nnls_work.w
    z = work.nnls_work.zz
    idx = work.nnls_work.idx

    # x = A[:, end] \ b
    den = zero(T)
    @inbounds @simd for i in 1:m
        den += A[i, n] * A[i, n]
    end

    # An exactly zero last column leaves den = 0; x[n] = 0 is then optimal for it, and the dual below reduces to the cold A'b.
    xj = zero(T)
    if den > 0
        @inbounds @simd for i in 1:m
            xj += (A[i, n] / den) * b[i]
        end
    end

    # w = -A'*(Ax - b)
    @inbounds @simd for i in 1:m
        z[i] = b[i] - A[i, n] * xj
    end

    @inbounds for j in 1:n-1
        wj = zero(T)
        @simd for i in 1:m
            wj += A[i, j] * z[i]
        end
        w[j] = wj
    end
    @inbounds w[end] = 0
    @inbounds w[end] = all(<=(0), w)

    # Initialize nnls workspace; A is not copied, since the solver reads pristine column data directly from the caller's matrix
    @inbounds for i in 1:m
        f[i] = b[i]
    end

    @inbounds for j in 1:n
        x[j] = 0
        idx[j] = j
    end

    return NNLS.unsafe_nnls!(work.nnls_work, A; kwargs..., init_dual = false)
end

# Warm-started unregularized solve: seeds the passive set with the original column indices idx0[1:nsetp0], e.g. `NNLS.components` saved from a solve against a nearby matrix, such as an adjacent flip angle's decay basis in the surrogate search.
# Seeds are stashed in hpos, entered without the positivity check, and a feasibility pass drops any that come out non-positive, so the result satisfies the same KKT conditions as a cold solve regardless of seed quality.
# The initial dual is recomputed from the seeded residual, so no dual preload is needed here.
function solve!(
    work::NNLSProblem{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    idx0::AbstractVector{Int},
    nsetp0::Int;
    kwargs...,
) where {T}
    m, n = size(A)
    f = work.nnls_work.b
    x = work.nnls_work.x
    idx = work.nnls_work.idx
    invidx = work.nnls_work.invidx
    hpos = work.nnls_work.hpos
    @assert 0 <= nsetp0 <= min(n, length(idx0))

    # Stash the seeds first: idx0 may alias this workspace's own idx (which is re-initialized below)
    @inbounds for t in 1:nsetp0
        j = idx0[t]
        @assert 1 <= j <= n
        hpos[t] = j
    end

    # Initialize nnls workspace (A is not copied; see `solve!(work, A, b)`)
    @inbounds @simd ivdep for i in 1:m
        f[i] = b[i]
    end
    @inbounds for j in 1:n
        x[j] = 0
        idx[j] = j
        invidx[j] = j # seeding tracks positions via invidx; must start as the identity
    end

    return NNLS.unsafe_nnls!(work.nnls_work, A; kwargs..., nwarm = nsetp0)
end

function solve!(
    work::NNLSProblem{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    μ::T;
    kwargs...,
) where {T}
    if A isa TikhonovPaddedMatrix
        A0 = parent(A)
        m, n = size(A0)
    else
        M, N = size(A)
        m, n = M - N, N
        A0 = view(A, 1:m, :)
    end
    if b isa PaddedVector
        b0 = parent(b)
    else
        b0 = view(b, 1:m)
    end

    f = work.nnls_work.b
    x = work.nnls_work.x
    w = work.nnls_work.w
    z = work.nnls_work.zz
    idx = work.nnls_work.idx
    diag = work.nnls_work.diag

    # x = A[:, end] \ b
    den = zero(T)
    @inbounds @simd for i in 1:m
        den += A0[i, n] * A0[i, n]
    end
    den += μ^2

    xj = zero(T)
    @inbounds @simd for i in 1:m
        xj += (A0[i, n] / den) * b0[i]
    end

    # w = -A'*(Ax - b)
    @inbounds @simd for i in 1:m
        z[i] = b0[i] - A0[i, n] * xj
    end

    @inbounds for j in 1:n-1
        wj = zero(T)
        @simd for i in 1:m
            wj += A0[i, j] * z[i]
        end
        w[j] = wj
    end
    @inbounds w[end] = 0
    @inbounds w[end] = all(<=(0), w)

    # Initialize nnls workspace; A is not copied, since the solver reads pristine column data directly from the caller's
    # matrix and candidate columns are materialized in a scratch buffer with their λ entry placed on the fly
    @inbounds for i in 1:m
        f[i] = b0[i]
    end
    @inbounds for j in 1:n
        x[j] = 0
        f[m+j] = 0
        idx[j] = j
        diag[j] = 0 # no λ row activated
    end

    return NNLS.unsafe_nnls!(work.nnls_work, A0, μ; kwargs..., init_dual = false)
end

# Warm-started Tikhonov solve: like `solve!(work, A, b, μ)` above, but seeds the passive set with the original column indices `idx0[1:nsetp0]`, e.g. saved via `NNLS.components` from a solve at a nearby μ.
# Seeding follows the same protocol as `NNLS.nnls!(work, A, b, λ, idx0, nsetp0)`: seeds are stashed in `hpos`, entered without the positivity check, and a feasibility pass drops any that come out non-positive, so the result satisfies the same KKT conditions as a cold solve regardless of seed quality.
# The initial dual is recomputed from the seeded residual, so no dual preload is needed here.
function solve!(
    work::NNLSProblem{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    μ::T,
    idx0::AbstractVector{Int},
    nsetp0::Int;
    kwargs...,
) where {T}
    if A isa TikhonovPaddedMatrix
        A0 = parent(A)
        m, n = size(A0)
    else
        M, N = size(A)
        m, n = M - N, N
        A0 = view(A, 1:m, :)
    end
    if b isa PaddedVector
        b0 = parent(b)
    else
        b0 = view(b, 1:m)
    end
    @assert 0 <= nsetp0 <= min(n, length(idx0))

    f = work.nnls_work.b
    x = work.nnls_work.x
    idx = work.nnls_work.idx
    invidx = work.nnls_work.invidx
    diag = work.nnls_work.diag
    hpos = work.nnls_work.hpos

    # Stash the seeds first: idx0 may alias this workspace's own idx (which is re-initialized below)
    @inbounds for t in 1:nsetp0
        j = idx0[t]
        @assert 1 <= j <= n
        hpos[t] = j
    end

    # Initialize nnls workspace (A is not copied; see `solve!(work, A, b, μ)`)
    @inbounds for i in 1:m
        f[i] = b0[i]
    end
    @inbounds for j in 1:n
        x[j] = 0
        f[m+j] = 0
        idx[j] = j
        invidx[j] = j # seeding tracks positions via invidx; must start as the identity
        diag[j] = 0 # no λ row activated
    end

    return NNLS.unsafe_nnls!(work.nnls_work, A0, μ; kwargs..., nwarm = nsetp0)
end

# Source for the unregularized solve that each μ-selection method starts from.
# A bare workspace only seeds: its passive set warm-starts the solve, changing the cost but never the solution.
# A problem over the same `A` and `b`, which the flip-angle polish leaves solved at the fitted angle, also lets a solved state be used outright.
const NNLSUnregSource{T} = Union{Nothing, NNLS.NNLSWorkspace{T}, NNLSProblem{T}}

solve_unreg!(prob::NNLSProblem, ::Nothing) = solve!(prob)
solve_unreg!(prob::NNLSProblem, seed::NNLS.NNLSWorkspace) = solve!(prob, prob.A, prob.b, seed.idx, NNLS.ncomponents(seed))
solve_unreg!(prob::NNLSProblem, src::NNLSProblem) = NNLS.issolved(src.nnls_work) ? adopt_solution!(prob, src.nnls_work) : solve_unreg!(prob, src.nnls_work)

# Consumers read the solution, residual, residual norm, and passive set, never the triangular factor, so an O(m + n) transfer of those replaces the solve.
# Kept out of line so that `solve_unreg!` stays small enough to inline into its callers.
function adopt_solution!(prob::NNLSProblem, src::NNLS.NNLSWorkspace)
    dst = prob.nnls_work
    copyto!(dst.x, src.x)
    copyto!(dst.r, src.r)
    copyto!(dst.idx, src.idx)
    dst.rnorm[] = src.rnorm[]
    dst.mode[] = src.mode[]
    dst.nsetp[] = src.nsetp[]
    dst.solved[] = true
    return prob
end

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
    b::Tb # decay curve data
    pad::Int
end
Base.size(x::PaddedVector) = (length(x.b) + x.pad,)
Base.parent(x::PaddedVector) = x.b

function Base.copyto!(y::AbstractVector{T}, x::PaddedVector{T}) where {T}
    @assert size(x) == size(y)
    (; b, pad) = x
    m = length(b)
    @inbounds @simd for i in 1:m
        y[i] = b[i]
    end
    @inbounds @simd for i in m+1:m+pad
        y[i] = zero(T)
    end
    return y
end

struct TikhonovPaddedMatrix{T, TA <: AbstractMatrix{T}} <: AbstractMatrix{T}
    A::TA # decay basis matrix
    μ::Base.RefValue{T}
end
TikhonovPaddedMatrix(A::AbstractMatrix, μ::Real) = TikhonovPaddedMatrix(A, Ref(μ))
Base.size(P::TikhonovPaddedMatrix) = ((m, n) = size(P.A); return (m + n, n))
Base.parent(P::TikhonovPaddedMatrix) = P.A
regparam(P::TikhonovPaddedMatrix) = P.μ[]
regparam!(P::TikhonovPaddedMatrix, μ::Real) = P.μ[] = μ

function Base.copyto!(B::AbstractMatrix{T}, P::TikhonovPaddedMatrix{T}) where {T}
    @assert size(P) == size(B)
    A, μ = parent(P), regparam(P)
    m, n = size(A)
    @inbounds for j in 1:n
        @simd for i in 1:m
            B[i, j] = A[i, j]
        end
        @simd for i in m+1:m+n
            B[i, j] = zero(T)
        end
    end
    @inbounds for j in 1:n
        B[m+j, j] = μ
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
    W <: NNLSProblem{T, <:TikhonovPaddedMatrix{T}, <:PaddedVector{T}},
    B,
}
    A::TA # decay basis matrix
    b::Tb # decay curve data
    m::Int # number of rows of A
    n::Int # number of columns of A
    nnls_prob::W # NNLS problem over the μ-augmented system [A; μI] x = [b; 0]
    buffers::B # scratch for the μ-derivative and curvature computations (see `gradient_temps`, `hessian_temps`)
end
function NNLSTikhonovRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}, μ::Real = T(NaN)) where {T}
    m, n = size(A)
    nnls_prob = NNLSProblem(TikhonovPaddedMatrix(A, μ), PaddedVector(b, n))
    buffers = (; null_soln = zeros(T, n), tmp = zeros(T, n))
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

regparam(work::NNLSTikhonovRegProblem) = regparam(work.nnls_prob.A)
regparam!(work::NNLSTikhonovRegProblem, μ::Real) = regparam!(work.nnls_prob.A, μ)

# Solve the Tikhonov-regularized NNLS problem with regularization parameter `μ`
function solve!(work::NNLSTikhonovRegProblem, μ::Real; kwargs...)
    regparam!(work, μ)
    solve!(work.nnls_prob, μ; kwargs...)
    return solution(work)
end

# Warm-started solve: seed the passive set with the column indices idx0[1:nsetp0].
function solve!(work::NNLSTikhonovRegProblem, μ::Real, idx0::AbstractVector{Int}, nsetp0::Int; kwargs...)
    regparam!(work, μ)
    solve!(work.nnls_prob, μ, idx0, nsetp0; kwargs...)
    return solution(work)
end

@inline solution(work::NNLSTikhonovRegProblem) = NNLS.solution(work.nnls_prob.nnls_work)
@inline ncomponents(work::NNLSTikhonovRegProblem) = NNLS.ncomponents(work.nnls_prob.nnls_work)

@inline loss(work::NNLSTikhonovRegProblem) = NNLS.residualnorm(work.nnls_prob.nnls_work)^2

regnorm(work::NNLSTikhonovRegProblem) = regparam(work)^2 * seminorm_sq(work) # μ²||x||²
∇regnorm(work::NNLSTikhonovRegProblem) = 2 * regparam(work) * seminorm_sq(work) + regparam(work)^2 * ∇seminorm_sq(work) # d/dμ [μ²||x||²] = 2μ||x||² + μ² d/dμ [||x||²]

resnorm(work::NNLSTikhonovRegProblem) = √(resnorm_sq(work)) # ||Ax-b||
resnorm_sq(work::NNLSTikhonovRegProblem) = max(loss(work) - regnorm(work), 0) # ||Ax-b||²
∇resnorm_sq(work::NNLSTikhonovRegProblem, ∇ = gradient_temps(work)) = 4 * ∇.μ^3 * ∇.xᵀB⁻¹x # d/dμ [||Ax-b||²]
∇²resnorm_sq(work::NNLSTikhonovRegProblem, ∇² = hessian_temps(work)) = 12 * ∇².μ^2 * ∇².xᵀB⁻¹x - 24 * ∇².μ^4 * ∇².xᵀB⁻ᵀB⁻¹x # d²/dμ² [||Ax-b||²]

seminorm(work::NNLSTikhonovRegProblem) = √(seminorm_sq(work)) # ||x||
seminorm_sq(work::NNLSTikhonovRegProblem) = sum(abs2, NNLS.positive_solution(work.nnls_prob.nnls_work)) # ||x||²
∇seminorm_sq(work::NNLSTikhonovRegProblem, ∇ = gradient_temps(work)) = -4 * ∇.μ * ∇.xᵀB⁻¹x # d/dμ [||x||²]
∇²seminorm_sq(work::NNLSTikhonovRegProblem, ∇² = hessian_temps(work)) = -4 * ∇².xᵀB⁻¹x + 24 * ∇².μ^2 * ∇².xᵀB⁻ᵀB⁻¹x # d²/dμ² [||x||²]

solution_gradnorm(work::NNLSTikhonovRegProblem, ∇² = hessian_temps(work)) = √(solution_gradnorm_sq(work, ∇²)) # ||dx/dμ|| = ||-2μ * B⁻¹x|| = 2μ * ||B⁻¹x||
solution_gradnorm_sq(work::NNLSTikhonovRegProblem, ∇² = hessian_temps(work)) = 4 * ∇².μ^2 * ∇².xᵀB⁻ᵀB⁻¹x # ||dx/dμ||² = ||-2μ * B⁻¹x||² = 4μ² * xᵀB⁻ᵀB⁻¹x

# L-curve: (ξ(μ), η(μ)) = (||Ax-b||^2, ||x||^2)
curvature(::typeof(identity), work::NNLSTikhonovRegProblem, ∇ = gradient_temps(work)) = inv(2 * ∇.xᵀB⁻¹x * √(1 + ∇.μ^4)^3)

# L-curve: (ξ̄(μ), η̄(μ)) = (log||Ax-b||^2, log||x||^2)
function curvature(::typeof(log), work::NNLSTikhonovRegProblem, ∇ = gradient_temps(work))
    # Analytically, we have that
    #       d(η²)/d(ξ²) = d(η²)/dμ / d(ξ²)/dμ = -1 / μ²     (1)
    #   =>  d(logη²)/d(logξ²) = -(ξ² / η²) / μ²             (2)
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
        tmp = uview(work.buffers.tmp, 1:NNLS.ncomponents(nnls_work))
        NNLS.positive_solution!(nnls_work, tmp)

        μ = regparam(work)
        NNLS.solve_triangular_system!(tmp, B, Val(true)) # tmp = U'\x
        xᵀB⁻¹x = sum(abs2, tmp) # x'B\x = x'(U'U)\x = ||U'\x||^2

        return (; μ, xᵀB⁻¹x)
    end
end

function hessian_temps(work::NNLSTikhonovRegProblem{T}) where {T}
    GC.@preserve work begin
        (; nnls_work) = work.nnls_prob
        B = cholesky!(NNLS.NormalEquation(), nnls_work) # B = A'A + μ²I = U'U
        tmp = uview(work.buffers.tmp, 1:NNLS.ncomponents(nnls_work))
        NNLS.positive_solution!(nnls_work, tmp)

        μ = regparam(work)
        NNLS.solve_triangular_system!(tmp, B, Val(true)) # tmp = U'\x
        xᵀB⁻¹x = sum(abs2, tmp) # x'B\x = x'(U'U)\x = ||U'\x||^2

        NNLS.solve_triangular_system!(tmp, B, Val(false)) # tmp = U\(U'\x) = (U'U)\x
        xᵀB⁻ᵀB⁻¹x = sum(abs2, tmp) # x'B'\B\x = ||B\x||^2 = ||(U'U)\x||^2

        return (; μ, xᵀB⁻¹x, xᵀB⁻ᵀB⁻¹x)
    end
end

function chi2_relerr!(work::NNLSTikhonovRegProblem, res²_target, logμ, ∇logμ = nothing)
    # NOTE: assumes `solve!(work, μ)` has been called and that the solution is ready
    μ = exp(logμ)
    res² = resnorm_sq(work)
    relerr = (res² - res²_target) / res²_target
    if ∇logμ !== nothing && length(∇logμ) > 0
        ∂res²_∂μ = ∇resnorm_sq(work)
        ∂relerr_∂logμ = μ * ∂res²_∂μ / res²_target
        @inbounds ∇logμ[1] = ∂relerr_∂logμ
    end
    return relerr
end
chi2_relerr⁻¹(res²_target, relerr) = res²_target * (1 + relerr)

# Helper struct which wraps `N` caches of `NNLSTikhonovRegProblem` workspaces.
# Useful for optimization problems where the last function call may not be the optimium,
# but perhaps it was recent and is still in the `NNLSTikhonovRegProblemCache`, avoiding a recomputation.
struct NNLSTikhonovRegProblemCache{T, N, W <: AbstractVector}
    cache::W # a Vector rather than an NTuple, so runtime slot indexing compiles to a load instead of a branch chain
    idx::Base.RefValue{Int} # index of the most recently written slot
end
function NNLSTikhonovRegProblemCache(A::AbstractMatrix{T}, b::AbstractVector{T}, ::Val{N} = Val(8)) where {T, N}
    cache = [NNLSTikhonovRegProblem(A, b) for _ in 1:N]
    idx = Ref(1)
    return NNLSTikhonovRegProblemCache{T, N, typeof(cache)}(cache, idx)
end
reset_cache!(work::NNLSTikhonovRegProblemCache) = (foreach(w -> regparam!(w, NaN), work.cache); nothing)
Base.getindex(work::NNLSTikhonovRegProblemCache) = work.cache[get_cache_index(work)]

function next_cache_index!(work::NNLSTikhonovRegProblemCache{T, N}) where {T, N}
    for i in 1:N
        if isnan(regparam(work.cache[i]))
            set_cache_index!(work, i)
            return work.idx[]
        end
    end
    set_cache_index!(work, work.idx[] + 1)
    return work.idx[]
end
@inline get_cache_index(work::NNLSTikhonovRegProblemCache) = work.idx[]
@inline set_cache_index!(work::NNLSTikhonovRegProblemCache{T, N}, i) where {T, N} = (work.idx[] = mod1(i, N))

function solve!(work::NNLSTikhonovRegProblemCache{T}, μ::T) where {T}
    # Find index of cached workspace with μi nearest to μ
    @assert μ > 0 "Regularization parameter μ must be positive, got μ = $μ"

    emptycache = true
    imax, Δlogμmax = 0, T(Inf)
    for (i, w) in enumerate(work.cache)
        μi = regparam(w)
        if !isnan(μi)
            emptycache = false
            Δlogμ = μ == μi ? zero(T) : T(abs(log1p((μ - μi) / μi)))
            Δlogμ < Δlogμmax && ((imax, Δlogμmax) = (i, Δlogμ))
            Δlogμmax == 0 && break
        end
    end

    if emptycache || imax == 0
        # No cached solve is an exact match, so solve from scratch.
        # A nearest match can also fail to exist when μ is so large that (μ - μi) / μi rounds to -1 and Δlogμ overflows to Inf.
        next_cache_index!(work)
        solve!(work[], μ)
    elseif Δlogμmax > 0
        # No exact match, so solve into the next cache slot, warm-started from the active set of the nearest cached solve.
        # The seeded solve stashes the seed indices before touching its own workspace, so this is safe even when the next slot is the seed slot itself.
        src = work.cache[imax].nnls_prob.nnls_work
        idx0, nsetp0 = src.idx, NNLS.ncomponents(src)
        next_cache_index!(work)
        solve!(work[], μ, idx0, nsetp0)
    else
        # Exact match; return cached solution
        set_cache_index!(work, imax)
    end

    return solution(work[])
end

####
#### Gram matrix-based fast path for the μ-search
####

# Seed the Gram fast path from the unregularized solution's active set.
# Requires `solve!(work.nnls_prob)` to have been called; `work` is any of the μ-selection problem types (chi2/gcv/lcurve) exposing `.nnls_gram`, `.A`, `.nnls_prob`.
function nnls_gram_setup!(work)
    gp = work.nnls_gram
    NNLS.load!(gp, work.A, work.b)
    wk = work.nnls_prob.nnls_work
    NNLS.set_active!(gp, work.A, wk.idx, NNLS.ncomponents(wk))
    return gp
end

# Evaluate (‖Ax(μ)-b‖², ‖x(μ)‖²) via the Gram fast path, warm-chained across μ, falling back to the exact QR solver if the Gram path fails a conditioning/iteration guard.
# Used by the gcv and lcurve μ-searches (see `lsqnonneg_gcv!`, `lsqnonneg_lcurve!`); the selected μ is then recomputed with an exact final solve.
function nnls_gram_losses!(work, μ::T) where {T}
    gp = work.nnls_gram
    res² = NNLS.solve!(gp, work.A, work.b, μ)
    if isnan(res²)
        solve!(work.nnls_prob_smooth_cache, μ)
        cache = work.nnls_prob_smooth_cache[]
        wk = cache.nnls_prob.nnls_work
        NNLS.set_active!(gp, work.A, wk.idx, NNLS.ncomponents(wk))
        return resnorm_sq(cache), seminorm_sq(cache)
    end
    return res², NNLS.seminorm_sq(gp)
end

# Exact final solve at the selected μ, seeded from the Gram path's active set and stored in a fresh μ-cache slot so the usual `solution(work)` accessors see it.
function nnls_gram_polish_solve!(work, μ::T) where {T}
    gp = work.nnls_gram
    cache = work.nnls_prob_smooth_cache
    next_cache_index!(cache)
    return solve!(cache[], μ, gp.P, gp.np[])
end

####
#### Chi2 method for choosing the Tikhonov regularization parameter
####

struct NNLSChi2RegProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W1, W2, W3, S}
    A::TA # decay basis matrix
    b::Tb # decay curve data
    m::Int # number of rows of A
    n::Int # number of columns of A
    nnls_prob::W1 # unregularized NNLS problem, i.e. μ = 0
    nnls_prob_smooth_cache::W2 # cache of recent Tikhonov solves, warm-started from the nearest cached μ
    nnls_gram::W3 # Gram fast path for the μ-search evaluations; see `NNLSGram`
    nnls_prob_seed::S # source for the unregularized solve; see `NNLSUnregSource`
end
function NNLSChi2RegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}, nnls_prob_seed::NNLSUnregSource{T} = nothing) where {T}
    m, n = size(A)
    nnls_prob = NNLSProblem(A, b)
    nnls_prob_smooth_cache = NNLSTikhonovRegProblemCache(A, b)
    nnls_gram = NNLS.NNLSGram(A, b)
    return NNLSChi2RegProblem(A, b, m, n, nnls_prob, nnls_prob_smooth_cache, nnls_gram, nnls_prob_seed)
end

@inline solution(work::NNLSChi2RegProblem) = solution(work.nnls_prob_smooth_cache[])
@inline ncomponents(work::NNLSChi2RegProblem) = ncomponents(work.nnls_prob_smooth_cache[])

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
lsqnonneg_chi2_work(A::AbstractMatrix, b::AbstractVector, nnls_prob_seed = nothing) = NNLSChi2RegProblem(A, b, nnls_prob_seed)

function lsqnonneg_chi2!(work::NNLSChi2RegProblem{T}, chi2_target::T, legacy::Bool = false; method::Symbol = legacy ? :legacy : :brent_gram) where {T}
    # Non-regularized solution, warm-started from `work.nnls_prob_seed` when present: `method === :brent_gram` solves the same χ²(μ) = target root problem to the same tolerance as :brent, using the Gram fast path for search evaluations and an exact final solve.
    # On voxels where the χ²(μ) curve is flat the selected μ is reproducible only within the root-tolerance band; use :brent for an evaluation-path identical to the reference implementation.
    solve_unreg!(work.nnls_prob, work.nnls_prob_seed)
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
            return resnorm_sq(work.nnls_prob_smooth_cache[])
        end
        if mu_final == 0
            x_final = x_unreg
        else
            x_final = solve!(work.nnls_prob_smooth_cache, mu_final)
        end

    elseif method === :bisect
        function f_bisect(logμ)
            solve!(work.nnls_prob_smooth_cache, exp(logμ))
            return chi2_relerr!(work.nnls_prob_smooth_cache[], res²_target, logμ)
        end

        # Find bracketing interval containing root, then perform bisection search with slightly higher tolerance to not waste f evals
        a, b, fa, fb = bracket_root_monotonic(f_bisect, T(-4.0), T(1.0); dilate = T(1.5), mono = +1, maxiters = 6)

        if fa * fb < 0
            # Bracketing interval found
            a, fa, c, fc, b, fb = bisect_root(f_bisect, a, b, fa, fb; xatol = T(0.0), xrtol = T(0.0), ftol = T(1e-3) * (chi2_target - 1), maxiters = 100)

            # Root of secant line through `(a, fa), (b, fb)` or `(c, fc), (b, fb)` to improve bisection accuracy
            tmp = fa * fc < 0 ? root_real_linear(a, c, fa, fc) : fc * fb < 0 ? root_real_linear(c, b, fc, fb) : T(NaN)
            d, fd = isnan(tmp) ? (c, fc) : (tmp, f_bisect(tmp))

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
        # Search evaluations use the exact QR solver via the μ-cache (identical evaluation path to the reference implementation, so the selected μ tracks it to solver roundoff; solves are warm-started across μ, which perturbs only the QR roundoff, not the active set)
        function f_brent(logμ)
            solve!(work.nnls_prob_smooth_cache, exp(logμ))
            return chi2_relerr!(work.nnls_prob_smooth_cache[], res²_target, logμ)
        end

        # Find bracketing interval containing root
        a, b, fa, fb = bracket_root_monotonic(f_brent, T(-4.0), T(1.0); dilate = T(1.5), mono = +1, maxiters = 6)

        if fa * fb < 0
            # Find root using Brent's method
            logmu_final, relerr_final = brent_root(f_brent, a, b, fa, fb; xatol = T(0.0), xrtol = T(0.0), ftol = T(1e-3) * (chi2_target - 1), maxiters = 100)
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

    elseif method === :brent_gram
        # Fast variant: search evaluations use the Gram fast path, warm-chained across μ. Only the residual norm feeds the root finder, and the final solution is recomputed with the exact QR solver below.
        nnls_gram_setup!(work) # seed the Gram fast path from the unregularized active set
        function f_brent_gram(logμ)
            res², _ = nnls_gram_losses!(work, exp(logμ))
            return (res² - res²_target) / res²_target
        end

        # Find bracketing interval containing root
        a, b, fa, fb = bracket_root_monotonic(f_brent_gram, T(-4.0), T(1.0); dilate = T(1.5), mono = +1, maxiters = 6)

        if fa * fb < 0
            # Find root using Brent's method
            logmu_final, relerr_final = brent_root(f_brent_gram, a, b, fa, fb; xatol = T(0.0), xrtol = T(0.0), ftol = T(1e-3) * (chi2_target - 1), maxiters = 100)
        else
            # No bracketing interval found; choose point with smallest value of f (note: this branch should never be reached)
            logmu_final, relerr_final = !isfinite(fa) ? (b, fb) : !isfinite(fb) ? (a, fa) : abs(fa) < abs(fb) ? (a, fa) : (b, fb)
        end

        if isfinite(relerr_final)
            mu_final, res²_final = exp(logmu_final), chi2_relerr⁻¹(res²_target, relerr_final)
            x_final = nnls_gram_polish_solve!(work, mu_final)
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

struct NNLSMDPRegProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W1, W2, W3, S}
    A::TA # decay basis matrix
    b::Tb # decay curve data
    m::Int # number of rows of A
    n::Int # number of columns of A
    nnls_prob::W1 # unregularized NNLS problem, i.e. μ = 0
    nnls_prob_smooth_cache::W2 # cache of recent Tikhonov solves, warm-started from the nearest cached μ
    nnls_gram::W3 # Gram fast path for the μ-search evaluations; see `NNLSGram`
    nnls_prob_seed::S # source for the unregularized solve; see `NNLSUnregSource`
end
function NNLSMDPRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}, nnls_prob_seed::NNLSUnregSource{T} = nothing) where {T}
    m, n = size(A)
    nnls_prob = NNLSProblem(A, b)
    nnls_prob_smooth_cache = NNLSTikhonovRegProblemCache(A, b)
    nnls_gram = NNLS.NNLSGram(A, b)
    return NNLSMDPRegProblem(A, b, m, n, nnls_prob, nnls_prob_smooth_cache, nnls_gram, nnls_prob_seed)
end

@inline solution(work::NNLSMDPRegProblem) = solution(work.nnls_prob_smooth_cache[])
@inline ncomponents(work::NNLSMDPRegProblem) = ncomponents(work.nnls_prob_smooth_cache[])

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
lsqnonneg_mdp_work(A::AbstractMatrix, b::AbstractVector, nnls_prob_seed = nothing) = NNLSMDPRegProblem(A, b, nnls_prob_seed)

function lsqnonneg_mdp!(work::NNLSMDPRegProblem{T}, δ::T) where {T}
    @assert δ > 0 "Residual norm δ must be a positive value, but got δ = $δ"

    # Non-regularized solution, warm-started from `work.nnls_prob_seed` when present
    solve_unreg!(work.nnls_prob, work.nnls_prob_seed)
    x_unreg = solution(work.nnls_prob)
    res²_min = resnorm_sq(work.nnls_prob)

    if δ <= √res²_min
        # Limit as δ -> res_min⁺ from above is the unregularized solution
        return (; x = x_unreg, mu = zero(T), chi2 = one(T))
    end

    res²_max = sum(abs2, work.nnls_prob.b)
    if δ >= √res²_max
        # Limit as δ -> ||b|| from below is the infinitely regularized solution, i.e. x = 0, since ||A * x(μ -> +∞) - b|| -> ||b||.
        x_final = work.nnls_prob_smooth_cache[].buffers.null_soln # zero solution
        return (; x = x_final, mu = T(Inf), chi2 = res²_max / res²_min)
    end

    # Prepare to solve. The residual-norm root ‖Ax(μ)-b‖² = δ² is found with the Gram fast path for search evaluations, seeded by the unregularized solve.
    reset_cache!(work.nnls_prob_smooth_cache)
    nnls_gram_setup!(work)

    function f(logμ)
        res², _ = nnls_gram_losses!(work, exp(logμ))
        return res² - δ^2
    end

    # Find bracketing interval containing root
    a, b, fa, fb = bracket_root_monotonic(f, T(-4.0), T(1.0); dilate = T(1.5), mono = +1, maxiters = 6)

    if fa * fb < 0
        # Find root using Brent's method
        logmu_final, err_final = brent_root(f, a, b, fa, fb; xatol = T(0.0), xrtol = T(0.0), ftol = T(1e-3) * δ^2, maxiters = 100)
    else
        # No bracketing interval found; choose point with smallest value of f (note: this branch should never be reached)
        logmu_final, err_final = !isfinite(fa) ? (b, fb) : !isfinite(fb) ? (a, fa) : abs(fa) < abs(fb) ? (a, fa) : (b, fb)
    end

    if isfinite(err_final)
        mu_final, res²_final = exp(logmu_final), δ^2 + err_final
        x_final = nnls_gram_polish_solve!(work, mu_final)
    else
        x_final, mu_final, res²_final = x_unreg, zero(T), one(T)
    end

    return (; x = x_final, mu = mu_final, chi2 = res²_final / res²_min)
end

####
#### L-curve method for choosing the Tikhonov regularization parameter
####

struct NNLSLCurveRegProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W1, W2, W3, C1, C2, S}
    A::TA # decay basis matrix
    b::Tb # decay curve data
    m::Int # number of rows of A
    n::Int # number of columns of A
    nnls_prob::W1 # unregularized NNLS problem, i.e. μ = 0
    nnls_prob_smooth_cache::W2 # cache of recent Tikhonov solves, warm-started from the nearest cached μ
    nnls_gram::W3 # Gram fast path for the μ-search evaluations; see `NNLSGram`
    lsqnonneg_lcurve_fun_cache::C1 # cache of (log res², log ‖x‖²) points on the L-curve
    lcurve_corner_caches::C2 # corner-search point and state caches (see `lcurve_corner`)
    nnls_prob_seed::S # source for the unregularized solve; see `NNLSUnregSource`
end
function NNLSLCurveRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}, nnls_prob_seed::NNLSUnregSource{T} = nothing) where {T}
    m, n = size(A)
    nnls_prob = NNLSProblem(A, b)
    nnls_prob_smooth_cache = NNLSTikhonovRegProblemCache(A, b)
    nnls_gram = NNLS.NNLSGram(A, b)
    lsqnonneg_lcurve_fun_cache = GrowableCache{T, SVector{2, T}}(64, isapprox)
    lcurve_corner_caches = (
        GrowableCache{T, LCurveCornerPoint{T}}(64, isapprox),
        GrowableCache{T, LCurveCornerState{T}}(64, isapprox),
    )
    return NNLSLCurveRegProblem(A, b, m, n, nnls_prob, nnls_prob_smooth_cache, nnls_gram, lsqnonneg_lcurve_fun_cache, lcurve_corner_caches, nnls_prob_seed)
end

@inline solution(work::NNLSLCurveRegProblem) = solution(work.nnls_prob_smooth_cache[])
@inline ncomponents(work::NNLSLCurveRegProblem) = ncomponents(work.nnls_prob_smooth_cache[])

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
function lsqnonneg_lcurve(A::AbstractMatrix, b::AbstractVector; kwargs...)
    work = lsqnonneg_lcurve_work(A, b)
    return lsqnonneg_lcurve!(work; kwargs...)
end
lsqnonneg_lcurve_work(A::AbstractMatrix, b::AbstractVector, nnls_prob_seed = nothing) = NNLSLCurveRegProblem(A, b, nnls_prob_seed)

# Slope-collapse guard threshold for the L-curve corner: the maximum admissible log-log tangent slope |S| = -res²/(‖x‖²·μ²) at a corner candidate.
# The max-curvature search can latch onto spurious high curvature near μ→0, where the residual bottoms out as ‖Ax-b‖² → res²_min and the log-residual axis flattens, so the L-curve turns near-vertical and the "corner" collapses onto its top-left tail.
# A near-vertical collapse has |S|≫1, a genuine elbow has |S| ~ 0.1 to 10; candidate points steeper than this are not accepted as corners. This is built into the search's admissibility filter; see `lcurve_corner`'s `slope_max` kwarg.
# Set to Inf to disable and recover the pure max-curvature search.
const LCURVE_SLOPE_MAX = Ref(10.0)

function lsqnonneg_lcurve!(work::NNLSLCurveRegProblem{T}; kwargs...) where {T}
    # Compute the regularization using the L-curve method
    reset_cache!(work.nnls_prob_smooth_cache)

    # The corner search's (ξ, η) points are evaluated through the Gram fast path. The corner curvature is computed from these cached points and the final solution is recomputed via `nnls_gram_polish_solve!`.
    solve_unreg!(work.nnls_prob, work.nnls_prob_seed) # unregularized solution seeds the Gram fast path
    nnls_gram_setup!(work)

    # A point on the L-curve is given by (ξ(μ), η(μ)) = (log||Ax-b||^2, log||x||^2)
    #   Note: Squaring the norms is convenient for computing gradients of (ξ(μ), η(μ));
    #         this scales the L-curve, but does not change μ* = argmax C(ξ(μ), η(μ)).
    function f_lcurve(logμ)
        res², η² = nnls_gram_losses!(work, exp(logμ))
        return SA{T}[log(res²), log(η²)]
    end

    # Build cached function and solve via pointwise max-curvature, following Cultrera-Callegaro, rejecting corners in the near-vertical μ→0 collapse tail with the slope guard `LCURVE_SLOPE_MAX`.
    f_lcurve_cached = CachedFunction(f_lcurve, empty!(work.lsqnonneg_lcurve_fun_cache))
    f = LCurveCornerCachedFunction(f_lcurve_cached, empty!.(work.lcurve_corner_caches)...)
    logmu_final = lcurve_corner(f, T(-8), T(2); slope_max = T(LCURVE_SLOPE_MAX[]), kwargs...)

    # A degenerate cornerless curve admits no corner; see `lcurve_corner`.
    # Return the unregularized solution rather than an arbitrary near-zero μ.
    isnan(logmu_final) && return (; x = solution(work.nnls_prob), mu = zero(T), chi2 = one(T))

    # Return the final regularized solution (recomputed exactly; the unregularized solve was already done above to seed the Gram path)
    mu_final = exp(logmu_final)
    x_final = nnls_gram_polish_solve!(work, mu_final)
    chi2_final = resnorm_sq(work.nnls_prob_smooth_cache[]) / resnorm_sq(work.nnls_prob)

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
function lcurve_corner(f::LCurveCornerCachedFunction{T}, xlow::T = -8.0, xhigh::T = 2.0; xtol = 1e-4, Ptol = 1e-4, Ctol = 1e-4, slope_max = T(Inf), backtracking = true) where {T}
    # Initialize state
    state = initial_state(f, T(xlow), T(xhigh))

    # Note: tolerances are absolute because typically the L-curve is on a log-log scale, and atol on log-log is equivalent to rtol on linear-linear
    Ptopleft, Pbottomright = state.P⃗[1], state.P⃗[4]
    Ptol = T(Ptol) # convergence occurs when diameter of L-curve state is less than Ptol
    Ctol = T(Ctol) # note: *not* a tolerance on curvature, but on the minimum diameter of the L-curve state used to estimate curvature (see `Pfilter` below)

    # A candidate point is x = logμ paired with P = (log‖Ax-b‖², log‖x‖²), and is admissible as a corner only if two tests pass.
    # First, it must not be numerically too close to an endpoint: points on the L-curve can be extremely close for tiny μ, and the curvature estimate is then unstable.
    # Second, its log-log tangent slope |S| = -res²/(μ²‖x‖²) must not be too steep, i.e. log|S| = P[1] - P[2] - 2x ≤ log(slope_max).
    # Steep near-vertical points are the μ→0 collapse artifact of the log transform, where the residual has bottomed out at res_min, whereas a genuine elbow has |S| ~ 0.1 to 10.
    # Setting slope_max = Inf drops the second test and recovers the pure max-curvature search. Inadmissible points are assigned -Inf curvature, so the search never accepts them.
    log_slope_max = log(T(slope_max))
    Pfilter = (x, P) -> (min(norm(P - Ptopleft), norm(P - Pbottomright)) > T(Ctol)) && (P[1] - P[2] - 2 * x <= log_slope_max)
    update_curvature!(f, state, Pfilter)

    # msg(s, state) = (@info "$s: [x⃗, P⃗, C⃗] = "; display(hcat(state.x⃗, state.P⃗, [f.point_cache[x].C for x in state.x⃗])))
    # msg("Starting", state)

    iter = 0
    while !is_converged(state; xtol = T(xtol), Ptol = T(Ptol))
        iter += 1
        if backtracking
            # Find state with minimum diameter which contains the current best estimate maximum curvature point
            (x, (_, _)), _, _ = mapfindmax(T, ((x, (P, C)),) -> C, pairs(f.point_cache))
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
    end

    (x, (_, C)), _, _ = mapfindmax(T, ((x, (P, C)),) -> C, pairs(f.point_cache))
    # msg("Converged", state)

    # Every evaluated point is inadmissible when the maximum curvature is still -Inf, which happens when all points are endpoint-near or steeper than `slope_max`, as on a degenerate cornerless L-curve.
    # Return NaN rather than an arbitrary collapse point, and let the caller fall back; `lsqnonneg_lcurve!` returns the unregularized solution in that case.
    return C == T(-Inf) ? T(NaN) : x
end

function initial_state(f::LCurveCornerCachedFunction{T}, x₁::T, x₄::T) where {T}
    φ = T(Base.MathConstants.φ)
    x₂ = (φ * x₁ + x₄) / (φ + 1)
    x₃ = x₁ + (x₄ - x₂)
    x⃗ = SA[x₁, x₂, x₃, x₄]
    P⃗ = SA[f(x₁), f(x₂), f(x₃), f(x₄)]
    Base.Cartesian.@nexprs 4 i -> push!(f.point_cache, (x⃗[i], LCurveCornerPoint(P⃗[i])))
    return LCurveCornerState(x⃗, P⃗)
end

is_converged(state::LCurveCornerState; xtol, Ptol) = abs(state.x⃗[4] - state.x⃗[1]) < xtol || norm(state.P⃗[1] - state.P⃗[4]) < Ptol

function move_left(f::LCurveCornerCachedFunction{T}, state::LCurveCornerState{T}) where {T}
    (; x⃗, P⃗) = state
    φ = T(Base.MathConstants.φ)
    x⃗ = SA[x⃗[1], (φ*x⃗[1]+x⃗[3])/(φ+1), x⃗[2], x⃗[3]]
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
        if Pfilter === nothing || Pfilter(x, P)
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
        φ = T(Base.MathConstants.φ)
        h = min(abs(xₖ - xⱼ), abs(xₗ - xₖ)) / φ
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
#### Reginska (minimum-product) method for choosing the Tikhonov regularization parameter
####

struct NNLSReginskaRegProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W1, W2, W3, S}
    A::TA # decay basis matrix
    b::Tb # decay curve data
    m::Int # number of rows of A
    n::Int # number of columns of A
    nnls_prob::W1 # unregularized NNLS problem, i.e. μ = 0
    nnls_prob_smooth_cache::W2 # cache of recent Tikhonov solves, warm-started from the nearest cached μ
    nnls_gram::W3 # Gram fast path for the μ-search evaluations; see `NNLSGram`
    nnls_prob_seed::S # source for the unregularized solve; see `NNLSUnregSource`
end
function NNLSReginskaRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}, nnls_prob_seed::NNLSUnregSource{T} = nothing) where {T}
    m, n = size(A)
    nnls_prob = NNLSProblem(A, b)
    nnls_prob_smooth_cache = NNLSTikhonovRegProblemCache(A, b)
    nnls_gram = NNLS.NNLSGram(A, b)
    return NNLSReginskaRegProblem(A, b, m, n, nnls_prob, nnls_prob_smooth_cache, nnls_gram, nnls_prob_seed)
end

@inline solution(work::NNLSReginskaRegProblem) = solution(work.nnls_prob_smooth_cache[])
@inline ncomponents(work::NNLSReginskaRegProblem) = ncomponents(work.nnls_prob_smooth_cache[])

@doc raw"""
    lsqnonneg_reginska(A::AbstractMatrix, b::AbstractVector)

Compute the Tikhonov-regularized nonnegative least-squares (NNLS) solution ``X_{\mu}`` of the problem:

```math
X_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu^2 ||x||_2^2
```

where ``\mu`` is chosen by Regińska's minimum-product criterion[1]:

```math
\mu = \underset{\nu > 0}{\operatorname{argmin}}\; \Psi(\nu) = ||AX_{\nu} - b||_2^2 \, ||X_{\nu}||_2^2,
```

taking the smallest local minimizer of ``\Psi``. Stationarity of ``\Psi`` is equivalent to the log-log L-curve tangent slope equalling ``-1``, so the selected ``\mu`` is the balance point ``||AX_{\mu} - b|| = \mu ||X_{\mu}||`` nearest the L-curve corner; unlike curvature maximization the criterion is parameter-free and cannot collapse into the near-vertical ``\mu \to 0`` tail. The smallest local minimizer is required because ``\Psi \to 0`` trivially as ``\mu \to \infty`` (``X_{\mu} \to 0``); if ``\Psi`` has no interior local minimum, the unregularized solution is returned with ``\mu = 0``.

# Arguments

  - `A::AbstractMatrix`: Decay basis matrix
  - `b::AbstractVector`: Decay curve data

# Outputs

  - `X::AbstractVector`: Regularized NNLS solution
  - `mu::Real`: Resulting regularization parameter ``\mu``
  - `chi2::Real`: Resulting increase in residual norm relative to the unregularized ``\mu = 0`` solution

# References

  1. T. Regińska, "A Regularization Parameter in Discrete Ill-Posed Problems". SIAM Journal on Scientific Computing, 17(3), 740-749, 1996, https://doi.org/10.1137/S1064827593252672.
"""
function lsqnonneg_reginska(A::AbstractMatrix, b::AbstractVector; kwargs...)
    work = lsqnonneg_reginska_work(A, b)
    return lsqnonneg_reginska!(work; kwargs...)
end
lsqnonneg_reginska_work(A::AbstractMatrix, b::AbstractVector, nnls_prob_seed = nothing) = NNLSReginskaRegProblem(A, b, nnls_prob_seed)

function lsqnonneg_reginska!(
    work::NNLSReginskaRegProblem{T}; atol = 1e-4, h = 0.5, # h floors the leap, which both bounds the step count and sets the resolution at which a crossing pair can be stepped over; see the scan below
) where {T}
    reset_cache!(work.nnls_prob_smooth_cache)

    # Evaluations run on the Gram fast path (one warm-chained μ-solve yields both ‖Ax-b‖² and ‖x‖²); the final solution is recomputed exactly at the selected μ
    solve_unreg!(work.nnls_prob, work.nnls_prob_seed) # unregularized solution seeds the Gram fast path
    x_unreg = solution(work.nnls_prob)
    res²_min = resnorm_sq(work.nnls_prob)

    # An exact unregularized fit makes the minimum-product criterion zero at μ = 0. The test must be relative, not res²_min == 0, since a computed exact fit leaves a residual at the roundoff level and Φ(0) would then be a ratio of noise.
    # A consistent system solved in floating point leaves ‖r‖ ≲ ε·κ₂(A)·‖b‖, so res² ≤ eps(T)·‖b‖² admits relative residual norms up to √ε, which is that bound at κ₂(A) ≃ ε^(-1/2).
    # The threshold is a conditioning assumption, not a free tolerance: a basis conditioned worse than ε^(-1/2) needs it loosened.
    b_nrm² = sum(abs2, work.b)
    if res²_min <= eps(T) * b_nrm² || ncomponents(work.nnls_prob) == 0
        return (; x = x_unreg, mu = zero(T), chi2 = one(T))
    end
    η²_unreg = sum(abs2, x_unreg)
    nnls_gram_setup!(work)

    # g(logμ) = log|S| = log res² − log ‖x‖² − 2 logμ, the log-log L-curve tangent slope magnitude (continuous in μ; derivative-free from one Gram evaluation).
    # Ψ = res²·‖x‖² satisfies dlogΨ/dlogμ = ξ'·(1 + S) with ξ' ≥ 0, so the smallest local minimizer of Ψ is exactly the leftmost downward crossing g = 0, i.e. the balance point |S| = 1.
    # `res²` is returned alongside, since it also certifies when the scan may stop; see below.
    function g_and_res²(logμ)
        res², η² = nnls_gram_losses!(work, exp(logμ))
        return (η² == 0 ? T(+Inf) : log(res²) - log(η²) - 2 * logμ), res²
    end
    g(logμ) = first(g_and_res²(logμ))

    # |S| → ∞ at BOTH ends (μ → 0: μ²‖x‖² → 0 with res² → res²_min > 0; μ → ∞: ‖x‖² ~ C/μ⁴), so |S| = 1 generically has an even number of crossings and the leftmost one must be certified by a left-to-right scan (early-exit at the first sign change; each evaluation is one cheap warm-chained Gram solve).
    # Monotonicity of res²(μ) (nondecreasing) and ‖x‖² (nonincreasing) gives g(b) ≥ g(a) − 2(b − a) for b > a with no smoothness assumption, so from a point with g(a) > 0 the first crossing lies at or beyond a + g(a)/2: the scan leaps by max(h, g(a)/2) - the same crossing-detection resolution h as a uniform scan where g is small, exponentially fewer evaluations where g is large (g ≈ −2 logμ + O(1) as μ → 0).
    # Φ(μ) = ‖Ax(μ) − b‖/‖x(μ)‖ is nondecreasing, since res² is nondecreasing and ‖x‖² nonincreasing, and the balance points are exactly its fixed points, so every one of them satisfies μ ≥ Φ(0): the scan starts there and needs no lower bound.
    # Each leap logμ -> logμ + g/2 is one step of that same fixed-point map, so starting at Φ(0) is starting the iteration at its natural first iterate rather than climbing to it from an arbitrary constant.
    # That map alone cannot bracket a crossing: g > 0 is Φ(μ) > μ, and applying the nondecreasing Φ gives Φ(Φ(μ)) ≥ Φ(μ), i.e. g ≥ 0 at every iterate, so the scan approaches the crossing from below and passes it only when g underflows.
    # Flooring the leap at h advances logμ by at least h per step, bounding the scan at (logμ_cert − logμ₀)/h steps and handing Brent a genuine sign change. The overstep past the certified interval (a, a + g/2) is then at most h − g/2.
    # It needs no upper bound either: complementarity gives bᵀr = res² + μ²‖x‖², which at a balance point reads bᵀr = 2·res², and Cauchy-Schwarz then bounds res² ≤ ‖b‖²/4 there.
    # Since res² is nondecreasing, the first scan point exceeding ‖b‖²/4 proves no balance point lies at or beyond it, and res² → ‖b‖² guarantees that test eventually fires.
    logμ₀ = (log(res²_min) - log(η²_unreg)) / 2
    res²_max = b_nrm² / 4

    a, ga = logμ₀, g(logμ₀)
    if ga <= 0
        logmu_final = logμ₀ # Φ(0) is itself the balance point, to within the resolution of one Gram evaluation
    else
        b, gb = a, ga
        while true
            b = a + max(h, ga / 2)
            gb, res²_b = g_and_res²(b)
            gb <= 0 && break
            res²_b > res²_max && return (; x = x_unreg, mu = zero(T), chi2 = one(T)) # no balance point exists
            a, ga = b, gb
        end
        logmu_final, _ = brent_root(g, a, b, ga, gb; xatol = T(atol), xrtol = T(0), ftol = T(0), maxiters = 100)
    end

    mu_final = exp(logmu_final)
    x_final = nnls_gram_polish_solve!(work, mu_final)
    chi2_final = resnorm_sq(work.nnls_prob_smooth_cache[]) / res²_min

    return (; x = x_final, mu = mu_final, chi2 = chi2_final)
end

####
#### GCV method for choosing the Tikhonov regularization parameter
####

struct NNLSGCVRegProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W0, W1, W2, W3, V, S}
    A::TA # decay basis matrix
    b::Tb # decay curve data
    m::Int # number of rows of A
    n::Int # number of columns of A
    γ²::Vector{T} # squared singular values of A, i.e. nonzero eigenvalues of A'A
    spectrum_work::W0 # workspace for computing the singular values of A
    nnls_prob::W1 # unregularized NNLS problem, i.e. μ = 0
    nnls_prob_smooth_cache::W2 # cache of recent Tikhonov solves, warm-started from the nearest cached μ
    nnls_gram::W3 # Gram fast path for the μ-search evaluations; see `NNLSGram`
    dof_interpolator::V # 2-tuple (GriddedSpectrumInterpolator over the α-grid decay bases, flip-angle Ref) or nothing: the source for the opt-in interpolated dof(μ), see `GCV_INTERP_DOF`
    nnls_prob_seed::S # source for the unregularized solve; see `NNLSUnregSource`
end
function NNLSGCVRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}, nnls_prob_seed::NNLSUnregSource{T} = nothing, dof_interpolator::Union{Nothing, Tuple{GriddedSpectrumInterpolator{T}, Base.RefValue{T}}} = nothing) where {T}
    m, n = size(A)
    spectrum_work = SVDValsWorkspace(A)
    nnls_prob = NNLSProblem(A, b)
    nnls_prob_smooth_cache = NNLSTikhonovRegProblemCache(A, b)
    nnls_gram = NNLS.NNLSGram(A, b)
    γ² = similar(spectrum_work.S)
    return NNLSGCVRegProblem(A, b, m, n, γ², spectrum_work, nnls_prob, nnls_prob_smooth_cache, nnls_gram, dof_interpolator, nnls_prob_seed)
end

@inline solution(work::NNLSGCVRegProblem) = solution(work.nnls_prob_smooth_cache[])
@inline ncomponents(work::NNLSGCVRegProblem) = ncomponents(work.nnls_prob_smooth_cache[])
@inline function LinearAlgebra.eigvals!(work::NNLSGCVRegProblem, A = work.A)
    γ = svdvals!(work.spectrum_work, A)
    work.γ² .= abs2.(γ)
    return work.γ²
end

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
function lsqnonneg_gcv(A::AbstractMatrix, b::AbstractVector; kwargs...)
    work = lsqnonneg_gcv_work(A, b)
    return lsqnonneg_gcv!(work; kwargs...)
end
lsqnonneg_gcv_work(A::AbstractMatrix, b::AbstractVector, nnls_prob_seed = nothing, dof_interpolator = nothing) = NNLSGCVRegProblem(A, b, nnls_prob_seed, dof_interpolator)

# Runtime toggle for approximating dof(μ) by cubic-Hermite interpolation across the α-grid spectral slices, when a `dof_interpolator` is supplied.
# Off by default: the exact per-voxel spectrum costs one SVD, and the interpolant's error grows as μ → 0, where dof stiffens in α.
const GCV_INTERP_DOF = Ref(false)

function lsqnonneg_gcv!(work::NNLSGCVRegProblem{T}; method = :brent, init = -4.0, bounds = (-8.0, 2.0), rtol = 0.0, atol = 1e-4, maxiters = 20) where {T}
    # Find μ by minimizing the function G(μ) (GCV method)
    @assert bounds[1] < init < bounds[2] "Initial value must be within bounds"
    logμ₋, logμ₊ = T.(bounds)
    logμ₀ = T(init)

    # Precompute the squared singular values, which are all dof(μ) needs; the opt-in alternative interpolates dof across the α-grid slices per evaluation instead (see `gcv_dof_interp`)
    dof_interpolator = work.dof_interpolator
    use_dof_interp = dof_interpolator !== nothing && GCV_INTERP_DOF[] && method === :brent
    use_dof_interp || eigvals!(work)

    # The gradient-free search (:brent, the default) evaluates the GCV objective 𝒢(μ) = ‖Ax(μ)-b‖² / dof(μ)² through the Gram fast path (dof is μ-cheap from the singular values; only the residual needs an NNLS solve).
    # The gradient-based methods keep the exact μ-cache solves, since ∇resnorm_sq needs the QR triangular factor. Forfeiting the Gram fast path roughly doubles the cost per evaluation, which is why :brent is the default.
    # Curvature would not repay it either, since 𝒢 is C¹ but not C²: dof is smooth in μ, and d‖x‖²/dμ = 2Σⱼ xⱼx'ⱼ survives a passive-set change because the component entering or leaving does so at xⱼ = 0, whereas d²‖x‖²/dμ² carries (x'ⱼ)² and jumps.
    # The final solution is always recomputed exactly.
    use_gram = method === :brent
    reset_cache!(work.nnls_prob_smooth_cache)
    if use_gram
        solve_unreg!(work.nnls_prob, work.nnls_prob_seed) # unregularized solution seeds the Gram fast path
        nnls_gram_setup!(work)
    end
    # 𝒢 is minimized directly. It is strictly positive for μ > 0 and b ≠ 0: KKT complementarity gives xᵀd = 0, hence (Ax)ᵀr = μ²‖x‖², and with Ax = b − r this is bᵀr = res² + μ²‖x‖², so res² = 0 would force x = 0 and then b = 0.
    function 𝒢(logμ)
        use_gram || return gcv!(work, logμ)
        μ = exp(logμ)
        res², _ = nnls_gram_losses!(work, μ)
        dof = use_dof_interp ? gcv_dof_interp(dof_interpolator[1], dof_interpolator[2][], work.m, work.n, μ) : gcv_dof(work.m, work.n, work.γ², μ)
        return res² / dof^2
    end
    𝒢_and_∇𝒢(logμ) = gcv_and_dgcv_dlogμ!(work, logμ)

    if method === :nlopt
        # alg = :LN_COBYLA # local, gradient-free, linear approximation of objective
        alg = :LN_BOBYQA # local, gradient-free, quadratic approximation of objective
        # alg = :GN_AGS # global, gradient-free, hilbert curve based dimension reduction
        # alg = :LN_NELDERMEAD # local, gradient-free, simplex method
        # alg = :LN_SBPLX # local, gradient-free, subspace searching simplex method
        # alg = :LD_CCSAQ # local, first-order (rough ranking: [:LD_MMA, :LD_SLSQP, :LD_LBFGS, :LD_CCSAQ, :LD_AUGLAG])
        opt               = NLopt.Opt(alg, 1)
        opt.lower_bounds  = Float64(logμ₋)
        opt.upper_bounds  = Float64(logμ₊)
        opt.xtol_abs      = Float64(atol)
        opt.xtol_rel      = Float64(rtol)
        opt.ftol_abs      = 0.0
        opt.ftol_rel      = 0.0
        opt.min_objective = (logμ, ∇logμ) -> @inbounds Float64(𝒢(T(logμ[1])))
        minf, minx, ret   = NLopt.optimize(opt, Float64[logμ₀])
        logmu_final       = @inbounds T(minx[1])
        𝒢_final           = T(minf)
    elseif method === :brent
        # Gradient-free golden-section/parabolic search over the full bounds. The GCV minimization is bracket-shrink bound (convergence needs the bracket width, not just a good point, to reach `atol`), so a warm start cannot speed it without narrowing the bounds a priori (sacrificing determinism, risking collapse).
        logmu_final, 𝒢_final = brent_minimize(𝒢, logμ₋, logμ₊; xrtol = T(rtol), xatol = T(atol), maxiters)
    elseif method === :brent_newton
        𝒢₋, ∇𝒢₋ = 𝒢_and_∇𝒢(logμ₋)
        𝒢₊, ∇𝒢₊ = 𝒢_and_∇𝒢(logμ₊)
        logμ_bdry, 𝒢_bdry = 𝒢₋ < 𝒢₊ ? (logμ₋, 𝒢₋) : (logμ₊, 𝒢₊)
        if ∇𝒢₋ < 0 && ∇𝒢₊ > 0
            𝒢₀, ∇𝒢₀ = 𝒢_and_∇𝒢(logμ₀)
            logmu_final, 𝒢_final = brent_newton_minimize(𝒢_and_∇𝒢, logμ₋, logμ₊, logμ₀, 𝒢₀, ∇𝒢₀; xrtol = T(rtol), xatol = T(atol), maxiters)
        else
            logmu_final, 𝒢_final = logμ_bdry, 𝒢_bdry
        end
        if 𝒢_bdry < 𝒢_final
            logmu_final, 𝒢_final = logμ_bdry, 𝒢_bdry
        end
    else
        error("Unknown minimization method: $method")
    end

    # Return the final regularized solution (recomputed exactly; if the Gram path was used the unregularized solve was already done above to seed it)
    mu_final = exp(logmu_final)
    x_final = use_gram ? nnls_gram_polish_solve!(work, mu_final) : solve!(work.nnls_prob_smooth_cache, mu_final)
    use_gram || solve_unreg!(work.nnls_prob, work.nnls_prob_seed)
    chi2_final = resnorm_sq(work.nnls_prob_smooth_cache[]) / resnorm_sq(work.nnls_prob)

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
    #   NOTE: assumes `eigvals!(work)` has been called and that `work.γ²` is ready
    (; m, n, γ²) = work

    # Solve regularized NNLS problem
    μ = exp(logμ)
    solve!(work.nnls_prob_smooth_cache, μ)
    cache = work.nnls_prob_smooth_cache[]

    # Compute GCV
    res² = resnorm_sq(cache) # squared residual norm ||A * x(μ) - b||^2
    dof = gcv_dof(m, n, γ², μ) # degrees of freedom
    gcv = res² / dof^2

    return gcv
end

function gcv_and_dgcv_dlogμ!(work::NNLSGCVRegProblem, logμ)
    # Unpack buffers
    #   NOTE: assumes `eigvals!(work)` has been called and that `work.γ²` is ready
    (; m, n, γ²) = work

    # Solve regularized NNLS problem
    μ = exp(logμ)
    solve!(work.nnls_prob_smooth_cache, μ)
    cache = work.nnls_prob_smooth_cache[]

    # Compute primal
    res² = resnorm_sq(cache) # squared residual norm ||A * x(μ) - b||^2
    dof = gcv_dof(m, n, γ², μ) # degrees of freedom
    gcv = res² / dof^2

    # Compute the derivative with respect to logμ
    ∇res² = ∇resnorm_sq(cache)
    ∇dof = ∇gcv_dof(m, n, γ², μ)
    ∇gcv = μ * (∇res² - 2 * res² * ∇dof / dof) / dof^2

    return gcv, ∇gcv
end

# Equation (27) from Hansen et al. 1992 (https://epubs.siam.org/doi/10.1137/1034115), specialized for L = identity:
#
#   tr(I_m - A * (A'A + λ^2 * L'L)⁻¹ * A') = m - n + sum_i λ^2 / (γ_i^2 + λ^2)
#
# where γ_i are the generalized singular values, which are equivalent to ordinary singular values when L = identity, and size(A) = (m, n).
# Can be considered as the "degrees of freedom".
function gcv_dof(m::Int, n::Int, γ²::AbstractVector{T}, λ::T) where {T}
    dof = T(max(m - n, 0)) # handle underdetermined systems (m < n)
    λ² = abs2(λ)
    @simd for γᵢ² in γ²
        dof += λ² / (γᵢ² + λ²)
    end
    return dof
end
gcv_dof(A::AbstractMatrix{T}, λ::T) where {T} = gcv_dof(size(A)..., svdvals(A) .^ 2, λ)

# DOF derivative w.r.t. the flip angle α: dof = max(m−n, 0) + Σᵢ λ²/(γᵢ²+λ²) and so ∂dof/∂α = Σᵢ [−λ²/(γᵢ²+λ²)²]·dγᵢ²/dα, where dγᵢ²/dα = 2σᵢ·uᵢᵀ(∂A/∂α)vᵢ is the analytic α-derivative of the squared singular values supplied by `dγ²`.
# The derivative ∂dof/∂α is smooth through branch crossings, since dof is a symmetric spectral function.
function dgcv_dof_dα(m::Int, n::Int, γ²::AbstractVector{T}, dγ²::AbstractVector{T}, λ::T) where {T}
    ∂dof = zero(T)
    λ² = abs2(λ)
    @simd for i in eachindex(γ², dγ²)
        ∂dof -= λ² / (γ²[i] + λ²)^2 * dγ²[i]
    end
    return ∂dof
end

# GCV dof(μ) at flip angle α, cubic-Hermite interpolated in α between the bracketing grid slices of `interp`, clamped to the grid range.
# The dof is interpolated directly: sorted singular value curves σᵢ(α) kink (C⁰) where branches cross, so interpolating γ caps the accuracy at the kink scale.
# The dof itself, dof(μ, α) = max(m − n, 0) + μ²·tr((A(α)ᵀA(α) + μ²I)⁻¹), is a symmetric function of the spectrum and hence analytic in α, so cubic Hermite with the analytic ∂dof/∂α (`dgcv_dof_dα`) is kink-free and O(h⁴) accurate.
function gcv_dof_interp(interp::GriddedSpectrumInterpolator{T}, α::T, m::Int, n::Int, μ::T) where {T}
    (; αs, γ², dγ², ready) = interp
    i = clamp(searchsortedlast(αs, α), 1, length(αs) - 1)
    ready[i] || gridded_spectrum_slice!(interp, i)
    ready[i+1] || gridded_spectrum_slice!(interp, i + 1)
    γl, γr = view(γ², :, i), view(γ², :, i + 1)
    dl, dr = view(dγ², :, i), view(dγ², :, i + 1)
    spl = CubicHermiteInterpolator(αs[i], αs[i+1], gcv_dof(m, n, γl, μ), gcv_dof(m, n, γr, μ), dgcv_dof_dα(m, n, γl, dl, μ), dgcv_dof_dα(m, n, γr, dr, μ))
    return spl(α, Val(:nearest)) # clamp to the bracketing cell
end

# DOF derivative: ∂/∂λ gcv_dof(m, n, γ, λ)
function ∇gcv_dof(m::Int, n::Int, γ²::AbstractVector{T}, λ::T) where {T}
    ∇dof = zero(T)
    λ² = abs2(λ)
    @simd for γᵢ² in γ²
        ∇dof += 2 * λ * γᵢ² / (γᵢ² + λ²)^2
    end
    return ∇dof
end
∇gcv_dof(A::AbstractMatrix{T}, λ::T) where {T} = ∇gcv_dof(size(A)..., svdvals(A) .^ 2, λ)
