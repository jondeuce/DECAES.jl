####
#### Regularization methods
####

abstract type RegularizationMethod end
struct NoRegularization <: RegularizationMethod end
struct Tikhonov{T} <: RegularizationMethod
    mu::T
end
struct LCurve <: RegularizationMethod end
struct LCurveLasso <: RegularizationMethod end
struct GCV <: RegularizationMethod end
struct Reginska <: RegularizationMethod end
struct ReginskaLasso <: RegularizationMethod end
struct ChiSquared{T} <: RegularizationMethod
    Chi2Factor::T
end
struct ChiSquaredLasso{T} <: RegularizationMethod
    Chi2Factor::T
end
struct MDP{T} <: RegularizationMethod
    NoiseLevel::T
end
struct MDPLasso{T} <: RegularizationMethod
    NoiseLevel::T
end

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

# The nnls algorithm selects candidate x[j] based on the largest negative gradient of ‖Ax - b‖, i.e. j = argmax_j w[j] where w = -A'(Ax - b) is the dual vector.
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

# Warm-started unregularized solve: seeds the passive set with column indices `idx0[1:nsetp0]`, e.g. saved by `NNLS.components` at a nearby flip angle.
# Seeds enter without the positivity check; a feasibility pass then drops any that come out non-positive, so the KKT conditions hold whatever the seed.
# The dual is recomputed from the seeded residual, so unlike the cold solves there is nothing to preload.
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

    # Initialize nnls workspace; A is not copied, since the solver reads pristine column data directly from the caller's matrix,
    # and candidate columns are materialized in a scratch buffer with their λ entry placed on the fly
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

# Warm-started Tikhonov solve: `solve!(work, A, b, μ)` seeded with the passive set `idx0[1:nsetp0]`, e.g. saved by `NNLS.components` at a nearby μ. Seeding protocol as above.
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

# Source for the unregularized solve every μ-selection method starts from.
# A workspace supplies a passive set to warm-start from, changing the cost but not the solution.
# A problem over the same `A` and `b`, as the flip-angle polish leaves behind, additionally allows copying a finished solution outright.
const NNLSUnregSource{T} = Union{Nothing, NNLS.NNLSWorkspace{T}, NNLSProblem{T}}

solve_unreg!(prob::NNLSProblem, ::Nothing) = solve!(prob)
solve_unreg!(prob::NNLSProblem, seed::NNLS.NNLSWorkspace) = solve!(prob, prob.A, prob.b, seed.idx, NNLS.ncomponents(seed))
solve_unreg!(prob::NNLSProblem, src::NNLSProblem) = NNLS.issolved(src.nnls_work) ? adopt_solution!(prob, src.nnls_work) : solve_unreg!(prob, src.nnls_work)

# Consumers read the solution, residual, residual norm, and passive set, never the triangular factor, so copying those O(m + n) values replaces the solve.
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
@inline seminorm_sq(work::NNLSProblem) = sum(abs2, NNLS.positive_solution(work.nnls_work))

@doc raw"""
    lsqnonneg(A::AbstractMatrix, b::AbstractVector)

Compute the nonnegative least-squares (NNLS) solution ``x`` of the problem:

```math
x_{0} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2.
```

# Arguments

  - `A::AbstractMatrix`: Left hand side matrix acting on `x`
  - `b::AbstractVector`: Right hand side vector

# Outputs

  - `x::AbstractVector`: NNLS solution
"""
lsqnonneg(A::AbstractMatrix, b::AbstractVector) = lsqnonneg!(lsqnonneg_work(A, b))
lsqnonneg_work(A::AbstractMatrix, b::AbstractVector) = NNLSProblem(A, b)
lsqnonneg!(work::NNLSProblem) = solve!(work)
lsqnonneg!(work::NNLSProblem{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T} = solve!(work, A, b)

# Unregularized counterpart of the `NNLS*RegProblem` types: with no μ to select, the workspace holds only the problem and the source its solve starts from.
struct NNLSUnregProblem{T, W <: NNLSProblem{T}, S}
    nnls_prob::W
    nnls_prob_seed::S # source for the unregularized solve; see `NNLSUnregSource`
end
NNLSUnregProblem(A::AbstractMatrix{T}, b::AbstractVector{T}, nnls_prob_seed::NNLSUnregSource{T} = nothing) where {T} = NNLSUnregProblem(NNLSProblem(A, b), nnls_prob_seed)

@inline solution(work::NNLSUnregProblem) = solution(work.nnls_prob)
lsqnonneg!(work::NNLSUnregProblem) = (solve_unreg!(work.nnls_prob, work.nnls_prob_seed); solution(work))

####
#### Lazy wrappers for LHS matrix and RHS vector for augmented Tikhonov-regularized NNLS problems
####

struct PaddedVector{T, Tb <: AbstractVector{T}} <: AbstractVector{T}
    b::Tb # decay curve data
    pad::Int
end
Base.size(v::PaddedVector) = (length(v.b) + v.pad,)
Base.parent(v::PaddedVector) = v.b

function Base.copyto!(y::AbstractVector{T}, v::PaddedVector{T}) where {T}
    @assert size(v) == size(y)
    (; b, pad) = v
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

Compute the Tikhonov-regularized nonnegative least-squares (NNLS) solution ``x_{\mu}`` of the problem:

```math
x_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu^2 ||x||_2^2.
```

# Arguments

  - `A::AbstractMatrix`: Left hand side matrix acting on `x`
  - `b::AbstractVector`: Right hand side vector
  - `μ::Real`: Regularization parameter

# Outputs

  - `x::AbstractVector`: NNLS solution
"""
lsqnonneg_tikh(A::AbstractMatrix, b::AbstractVector, μ::Real) = lsqnonneg_tikh!(lsqnonneg_tikh_work(A, b), μ)
lsqnonneg_tikh_work(A::AbstractMatrix, b::AbstractVector) = NNLSTikhonovRegProblem(A, b)
lsqnonneg_tikh!(work::NNLSTikhonovRegProblem, μ::Real) = solve!(work, μ)

regparam(work::NNLSTikhonovRegProblem) = regparam(work.nnls_prob.A)
regparam!(work::NNLSTikhonovRegProblem, μ::Real) = regparam!(work.nnls_prob.A, μ)

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

regnorm(work::NNLSTikhonovRegProblem) = regparam(work)^2 * seminorm_sq(work) # μ²‖x‖²
∇regnorm(work::NNLSTikhonovRegProblem) = 2 * regparam(work) * seminorm_sq(work) + regparam(work)^2 * ∇seminorm_sq(work) # d/dμ [μ²‖x‖²] = 2μ‖x‖² + μ² d/dμ [‖x‖²]

resnorm(work::NNLSTikhonovRegProblem) = √(resnorm_sq(work)) # ‖Ax-b‖
resnorm_sq(work::NNLSTikhonovRegProblem) = max(loss(work) - regnorm(work), 0) # ‖Ax-b‖²
∇resnorm_sq(work::NNLSTikhonovRegProblem, ∇ = gradient_temps(work)) = 4 * ∇.μ^3 * ∇.xᵀB⁻¹x # d/dμ [‖Ax-b‖²]
∇²resnorm_sq(work::NNLSTikhonovRegProblem, ∇² = hessian_temps(work)) = 12 * ∇².μ^2 * ∇².xᵀB⁻¹x - 24 * ∇².μ^4 * ∇².xᵀB⁻ᵀB⁻¹x # d²/dμ² [‖Ax-b‖²]

seminorm(work::NNLSTikhonovRegProblem) = √(seminorm_sq(work)) # ‖x‖
seminorm_sq(work::NNLSTikhonovRegProblem) = sum(abs2, NNLS.positive_solution(work.nnls_prob.nnls_work)) # ‖x‖²
∇seminorm_sq(work::NNLSTikhonovRegProblem, ∇ = gradient_temps(work)) = -4 * ∇.μ * ∇.xᵀB⁻¹x # d/dμ [‖x‖²]
∇²seminorm_sq(work::NNLSTikhonovRegProblem, ∇² = hessian_temps(work)) = -4 * ∇².xᵀB⁻¹x + 24 * ∇².μ^2 * ∇².xᵀB⁻ᵀB⁻¹x # d²/dμ² [‖x‖²]

solution_gradnorm(work::NNLSTikhonovRegProblem, ∇² = hessian_temps(work)) = √(solution_gradnorm_sq(work, ∇²)) # ‖dx/dμ‖ = ‖-2μ * B⁻¹x‖ = 2μ * ‖B⁻¹x‖
solution_gradnorm_sq(work::NNLSTikhonovRegProblem, ∇² = hessian_temps(work)) = 4 * ∇².μ^2 * ∇².xᵀB⁻ᵀB⁻¹x # ‖dx/dμ‖² = ‖-2μ * B⁻¹x‖² = 4μ² * xᵀB⁻ᵀB⁻¹x

# L-curve: (ξ(μ), η(μ)) = (‖Ax-b‖^2, ‖x‖^2)
curvature(::typeof(identity), work::NNLSTikhonovRegProblem, ∇ = gradient_temps(work)) = inv(2 * ∇.xᵀB⁻¹x * √(1 + ∇.μ^4)^3)

# L-curve: (ξ̄(μ), η̄(μ)) = (log‖Ax-b‖^2, log‖x‖^2)
curvature(::typeof(log), work::NNLSTikhonovRegProblem, ∇ = gradient_temps(work)) = lcurve_geometry(resnorm_sq(work), seminorm_sq(work), ∇.xᵀB⁻¹x, ∇.μ)[1]

# Curvature κ and turning rate ω = θ̇ of the log-log L-curve P(t) = (log ξ², log η²) at t = log μ, from ξ² = ‖Ax-b‖², η² = ‖x‖², q = xᵀB⁻¹x.
# At fixed active set Ṗ = (4ρ²q/ξ², -4ρq/η²) with ρ = μ², so the log-log slope is -ξ²/(ρη²).
# Writing H = hypot(ξ², ρη²), c = ξ²/H, d = ρη²/H, z = 2ρq/η², whence c² + d² = 1,
#   κ = cd·u/z,    ω = 2d·u,    u = c - z(c + d).
# Every factor is O(1) but the intrinsic 1/z, and u alone cancels at zero curvature. B ⪰ ρI gives q ≤ η²/ρ, hence 0 < z ≤ 2.
# κ selects the corner; ω tells it apart from the μ → 0 tail, where κ tends to a plateau η⁴/(2qξ²) but the arc speed vanishes.
function lcurve_geometry(ξ²::T, η²::T, q::T, μ::T) where {T}
    ρ = μ^2
    H = hypot(ξ², ρ * η²)
    c, d = ξ² / H, (ρ * η²) / H
    z = 2 * ((ρ * q) / η²)
    u = muladd(-z, c + d, c)
    return (c * d * u / z, 2 * d * u)
end

function gradient_temps(work::NNLSTikhonovRegProblem{T}) where {T}
    (; nnls_work) = work.nnls_prob
    B = cholesky!(NNLS.NormalEquation(), nnls_work) # B = A'A + μ²I = U'U
    tmp = view(work.buffers.tmp, 1:NNLS.ncomponents(nnls_work))
    NNLS.positive_solution!(nnls_work, tmp)

    μ = regparam(work)
    NNLS.solve_triangular_system!(tmp, B, Val(true)) # tmp = U'\x
    xᵀB⁻¹x = sum(abs2, tmp) # x'B\x = x'(U'U)\x = ‖U'\x‖^2

    return (; μ, xᵀB⁻¹x)
end

function hessian_temps(work::NNLSTikhonovRegProblem{T}) where {T}
    GC.@preserve work begin
        (; nnls_work) = work.nnls_prob
        B = cholesky!(NNLS.NormalEquation(), nnls_work) # B = A'A + μ²I = U'U
        tmp = view(work.buffers.tmp, 1:NNLS.ncomponents(nnls_work))
        NNLS.positive_solution!(nnls_work, tmp)

        μ = regparam(work)
        NNLS.solve_triangular_system!(tmp, B, Val(true)) # tmp = U'\x
        xᵀB⁻¹x = sum(abs2, tmp) # x'B\x = x'(U'U)\x = ‖U'\x‖^2

        NNLS.solve_triangular_system!(tmp, B, Val(false)) # tmp = U\(U'\x) = (U'U)\x
        xᵀB⁻ᵀB⁻¹x = sum(abs2, tmp) # x'B'\B\x = ‖B\x‖^2 = ‖(U'U)\x‖^2

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

# χ²(μ) = ‖Ax(μ)-b‖² / ‖Ax(0)-b‖², with zero-residual guards.
chi2_ratio(res²::T, res²_min::T) where {T} = res²_min == 0 ? (res² == 0 ? one(T) : T(Inf)) : res² / res²_min

# `N` `NNLSTikhonovRegProblem` workspaces in rotation. A μ-search rarely ends at its last evaluation, so keeping the recent ones lets the selected μ be recovered without re-solving.
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
        # Nothing to warm-start from: the cache is empty, or every Δlogμ overflowed to Inf, which happens once μ is small enough relative to μi that (μ - μi) / μi rounds to -1.
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

# Seed the Gram fast path from the unregularized solution's active set, so `work.nnls_prob` must already be solved.
# `work` is any of the μ-selection problem types exposing `.nnls_gram`, `.A`, `.b`, and `.nnls_prob`.
function nnls_gram_setup!(work)
    (; nnls_gram) = work
    NNLS.load!(nnls_gram, work.A, work.b)
    (; nnls_work) = work.nnls_prob
    NNLS.set_active!(nnls_gram, work.A, nnls_work.idx, NNLS.ncomponents(nnls_work))
    return nnls_gram
end

# Evaluate (‖Ax(μ)-b‖², ‖x(μ)‖²) on the Gram fast path, warm-chained across μ, falling back to the QR solver when a conditioning or iteration guard trips.
# The gcv and lcurve searches evaluate through this, then recompute the selected μ with `nnls_gram_polish_solve!`.
function nnls_gram_losses!(work, μ::T) where {T}
    (; nnls_gram) = work
    res² = NNLS.solve!(nnls_gram, work.A, work.b, μ)
    if isnan(res²)
        solve!(work.nnls_prob_smooth_cache, μ)
        cache = work.nnls_prob_smooth_cache[]
        (; nnls_work) = cache.nnls_prob
        NNLS.set_active!(nnls_gram, work.A, nnls_work.idx, NNLS.ncomponents(nnls_work))
        return resnorm_sq(cache), seminorm_sq(cache)
    end
    return res², NNLS.seminorm_sq(nnls_gram)
end

# Final solve via QR at the selected μ, seeded from the Gram path's active set and written to a fresh μ-cache slot so that `solution(work)` finds it.
function nnls_gram_polish_solve!(work, μ::T) where {T}
    (; nnls_gram) = work
    cache = work.nnls_prob_smooth_cache
    next_cache_index!(cache)
    return solve!(cache[], μ, nnls_gram.P, nnls_gram.np[])
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

Compute the Tikhonov-regularized nonnegative least-squares (NNLS) solution ``x_{\mu}`` of the problem:

```math
x_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu^2 ||x||_2^2
```

where ``\mu`` is determined by solving:

```math
\chi^2(\mu) = \frac{||Ax_{\mu} - b||_2^2}{||Ax_{0} - b||_2^2} = \chi^2_{\mathrm{target}}.
```

That is, ``\mu`` is chosen such that the squared residual norm of the regularized problem is `chi2_target`
times larger than the squared residual norm of the unregularized problem.

# Arguments

  - `A::AbstractMatrix`: Decay basis matrix
  - `b::AbstractVector`: Decay curve data
  - `chi2_target::Real`: Target ``\chi^2(\mu)``; typically a small value, e.g. 1.02 representing a 2% increase

# Outputs

  - `x::AbstractVector`: Regularized NNLS solution
  - `mu::Real`: Resulting regularization parameter ``\mu``
  - `chi2::Real`: Resulting ``\chi^2(\mu)``, which should be approximately equal to `chi2_target`
"""
function lsqnonneg_chi2(A::AbstractMatrix, b::AbstractVector, chi2_target::Real, args...; kwargs...)
    work = lsqnonneg_chi2_work(A, b)
    return lsqnonneg_chi2!(work, chi2_target, args...; kwargs...)
end
lsqnonneg_chi2_work(A::AbstractMatrix, b::AbstractVector, nnls_prob_seed = nothing) = NNLSChi2RegProblem(A, b, nnls_prob_seed)

function lsqnonneg_chi2!(work::NNLSChi2RegProblem{T}, chi2_target::T; method::Symbol = :brent_gram) where {T}
    # Non-regularized solution, warm-started from `work.nnls_prob_seed` when present
    solve_unreg!(work.nnls_prob, work.nnls_prob_seed)
    x_unreg = solution(work.nnls_prob)
    res²_min = resnorm_sq(work.nnls_prob)

    if res²_min == 0 || ncomponents(work.nnls_prob) == 0
        # An exact unregularized fit makes the target res²(μ) = chi2_target * res²_min zero, whose only root is μ = 0, since res²(μ) > 0 for μ > 0.
        # Or, since a zero unregularized solution gives x(μ) = 0 for every μ > 0, the target has no roots when chi2_target > 1 and every μ is a root when chi2_target = 1; take μ = 0.
        x_final = x_unreg
        return (; x = x_final, mu = zero(T), chi2 = one(T))
    end

    # Prepare to solve
    res²_target = chi2_target * res²_min
    reset_cache!(work.nnls_prob_smooth_cache)

    if method === :bisect
        function f_bisect(logμ)
            solve!(work.nnls_prob_smooth_cache, exp(logμ))
            return chi2_relerr!(work.nnls_prob_smooth_cache[], res²_target, logμ)
        end

        # Find bracketing interval containing root, then perform bisection search with slightly higher tolerance to not waste f evals
        a, b, fa, fb = bracket_root_monotonic(f_bisect, T(-4.0), T(1.0); dilate = T(1.5), mono = +1, maxiters = 6)

        if fa * fb < 0
            # Bracketing interval found
            c, fc, (a, b), (fa, fb) = bisect_root(f_bisect, a, b, fa, fb; xatol = T(0.0), xrtol = T(0.0), ftol = T(1e-3) * (chi2_target - 1), maxiters = 100)

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
            x_final, mu_final, res²_final = x_unreg, zero(T), res²_min
        end

    elseif method === :brent
        # Search evaluations use the QR solver via the μ-cache, the same evaluation path as the reference implementation, so the selected μ tracks it to solver roundoff. Warm-starting across μ perturbs that roundoff but not the active set.
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
            x_final, mu_final, res²_final = x_unreg, zero(T), res²_min
        end

    elseif method === :brent_gram
        # The same root problem and tolerance as `:brent`, but evaluated on the Gram fast path, warm-chained across μ, with only the residual norm reaching the root finder.
        # Where χ²(μ) is flat this pins μ only to within the root-tolerance band, whereas `:brent` evaluates the full QR solve at each μ and pins it exactly.
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
            x_final, mu_final, res²_final = x_unreg, zero(T), res²_min
        end
    else
        error("Unknown root-finding method: :$method")
    end

    return (; x = x_final, mu = mu_final, chi2 = chi2_ratio(res²_final, res²_min))
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

Compute the Tikhonov-regularized nonnegative least-squares (NNLS) solution ``x_{\mu}`` of the problem:

```math
x_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu^2 ||x||_2^2
```

where ``\mu`` is chosen using Morozov's Discrepancy Principle (MDP)[1,2]:

```math
\mu = \operatorname{sup}\; \left\{ \nu \ge 0 : ||Ax_{\nu} - b|| \le \delta \right\}.
```

That is, ``\mu`` is maximized subject to the constraint that the residual norm of the regularized problem is at most ``\delta``[1].

# Arguments

  - `A::AbstractMatrix`: Decay basis matrix
  - `b::AbstractVector`: Decay curve data
  - `δ::Real`: Upper bound on regularized residual norm

# Outputs

  - `x::AbstractVector`: Regularized NNLS solution
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
        # Limit as δ -> ‖b‖ from below is the infinitely regularized solution, i.e. x = 0, since ‖A * x(μ -> +∞) - b‖ -> ‖b‖.
        x_final = work.nnls_prob_smooth_cache[].buffers.null_soln # zero solution
        return (; x = x_final, mu = T(Inf), chi2 = chi2_ratio(res²_max, res²_min))
    end

    # Search evaluations for the root ‖Ax(μ)-b‖² = δ² take the Gram fast path, seeded from the unregularized solve.
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
        x_final, mu_final, res²_final = x_unreg, zero(T), res²_min
    end

    return (; x = x_final, mu = mu_final, chi2 = chi2_ratio(res²_final, res²_min))
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
    lcurve_point_cache::C1 # cache of evaluated L-curve points; see `LCurveCornerPoint`
    lcurve_state_stack::C2 # corner-search branch stack; see `lcurve_corner`
    nnls_prob_seed::S # source for the unregularized solve; see `NNLSUnregSource`
end
function NNLSLCurveRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}, nnls_prob_seed::NNLSUnregSource{T} = nothing) where {T}
    m, n = size(A)
    nnls_prob = NNLSProblem(A, b)
    nnls_prob_smooth_cache = NNLSTikhonovRegProblemCache(A, b)
    nnls_gram = NNLS.NNLSGram(A, b)
    lcurve_point_cache = GrowableCache{T, LCurveCornerPoint{T}}(64, (t₁, t₂) -> isapprox(t₁, t₂; atol = 10 * eps(T), rtol = 10 * eps(T)))
    lcurve_state_stack = GrowableCache{Int, LCurveCornerState{T}}(64)
    return NNLSLCurveRegProblem(A, b, m, n, nnls_prob, nnls_prob_smooth_cache, nnls_gram, lcurve_point_cache, lcurve_state_stack, nnls_prob_seed)
end

@inline solution(work::NNLSLCurveRegProblem) = solution(work.nnls_prob_smooth_cache[])
@inline ncomponents(work::NNLSLCurveRegProblem) = ncomponents(work.nnls_prob_smooth_cache[])

# Default slope guard: the largest log-log tangent slope |S| = ξ²/(μ²η²) accepted at a corner, keeping the tangent at least arctan(1/τ) off the vertical μ → 0 asymptote.
# A guard is needed because positive curvature does not exclude the μ → 0 tail wherein κ tends to a positive plateau, even rising out of the plateau when ‖x₀‖²·x₀ᵀG⁻²x₀ > 2(x₀ᵀG⁻¹x₀)².
const LCURVE_SLOPE_MAX_DEFAULT = 10.0

@doc raw"""
    lsqnonneg_lcurve(A::AbstractMatrix, b::AbstractVector; max_slope = $(LCURVE_SLOPE_MAX_DEFAULT))

Compute the Tikhonov-regularized nonnegative least-squares (NNLS) solution ``x_{\mu}`` of the problem:

```math
x_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu^2 ||L x||_2^2
```

where ``L`` is the identity matrix, and ``\mu`` is chosen at a corner of the "L-curve"[1], a local maximum of the curvature of ``\mu \mapsto (\log||Ax_\mu - b||_2^2, \log||x_\mu||_2^2)``.
The L-curve may have several corners; `max_slope` excludes those in its near-vertical ``\mu \to 0`` tail, where the fit is barely regularized. If no corner is found, ``\mu = 0`` and the unregularized solution is returned.
Details of L-curve theory can be found in Hansen (1992)[2].

# Arguments

  - `A::AbstractMatrix`: Decay basis matrix
  - `b::AbstractVector`: Decay curve data
  - `max_slope::Real = $(LCURVE_SLOPE_MAX_DEFAULT)`: reject corners at which ``||Ax_\mu - b||_2^2 / (\mu^2 ||x_\mu||_2^2)`` exceeds `max_slope`. Pass `Inf` to accept any corner.

# Outputs

  - `x::AbstractVector`: Regularized NNLS solution
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

function lsqnonneg_lcurve!(work::NNLSLCurveRegProblem{T}; max_slope = LCURVE_SLOPE_MAX_DEFAULT, kwargs...) where {T}
    reset_cache!(work.nnls_prob_smooth_cache)

    # Search points come from the Gram fast path, which also supplies κ, ω and the active set; the selected μ is recomputed via QR by `nnls_gram_polish_solve!`.
    solve_unreg!(work.nnls_prob, work.nnls_prob_seed) # unregularized solution seeds the Gram fast path
    nnls_gram_setup!(work)

    # A point on the L-curve is given by (ξ(μ), η(μ)) = (log‖Ax-b‖^2, log‖x‖^2)
    # Note: squaring the norms is convenient for computing gradients of (ξ(μ), η(μ)); this scales the L-curve, but does not change μ* = argmax κ(ξ(μ), η(μ)).
    ξ²₀, η²₀, b² = resnorm_sq(work.nnls_prob), seminorm_sq(work.nnls_prob), sum(abs2, work.b)

    # A zero solution puts log η² at -∞ and reduces the curve to a single point; a fit exact to working precision puts log ξ² there and leaves only roundoff to read curvature from.
    # Exactness is judged as ‖Ax₀ - b‖ ≤ ε‖b‖: declining to regularize is a convention, applied only where the residual has reached the level of the arithmetic.
    (ξ²₀ <= eps(T) * b² || η²₀ <= 0) && return (; x = solution(work.nnls_prob), mu = zero(T), chi2 = one(T))

    # Seed at the balance point ξ²₀ = μ²η²₀ from the unregularized solution.
    f = LCurveCornerCachedFunction(CachedFunction(logμ -> lcurve_point(work, exp(logμ)), empty!(work.lcurve_point_cache)), empty!(work.lcurve_state_stack))
    P₀ = SA{T}[log(ξ²₀), log(η²₀)]
    t₀ = (P₀[1] - P₀[2]) / 2

    # A finite slope guard confines every admissible corner to an interval computed from the data, assuming nothing about the active set.
    # ξ² is nondecreasing and η² nonincreasing, so |S| ≥ e^{2(t₀-t)} and |S| ≤ τ forces t ≥ t₀ - ½logτ.
    # KKT gives (Aᵀb)ᵀx = xᵀAᵀAx + ρη², and x ≥ 0 with Cauchy-Schwarz gives (Aᵀb)ᵀx ≤ ‖(Aᵀb)₊‖·η.
    # Bounding ξ² below by ξ²₀ and by ‖b‖² - (Aᵀb)ᵀx - ρη² then gives ρ ≤ ‖(Aᵀb)₊‖²·min{τ/ξ²₀, (τ+2)/‖b‖²}.
    # Unregularized optimality gives Aᵀb ≤ AᵀAx₀ componentwise, hence ρη² ≤ ‖Ax₀‖²/4 and |S| ≥ 4ξ²₀/(‖b‖² - ξ²₀) everywhere.
    # So ξ²₀/‖b‖² > τ/(τ+4) leaves the path with no admissible point. For 1 < τ < 6 + 2√17 that is also the only way the seed can fall outside [t₋, t₊].
    τ = T(max_slope)
    logmu_bounds = if isfinite(τ)
        (τ + 4) * ξ²₀ > τ * b² && return (; x = solution(work.nnls_prob), mu = zero(T), chi2 = one(T))
        c₊² = sum(cᵢ -> max(cᵢ, zero(T))^2, work.nnls_gram.c)
        cmax = work.nnls_gram.cscale[]
        logC = isfinite(c₊²) ? log(c₊²) : 2 * log(cmax) + log(sum(cᵢ -> (max(cᵢ, zero(T)) * inv(cmax))^2, work.nnls_gram.c))
        (t₀ - log(τ) / 2, (logC + log(min(τ / ξ²₀, (τ + 2) / b²))) / 2)
    else
        (T(-Inf), T(Inf))
    end
    logmu_final = lcurve_corner(f, t₀, P₀, NNLS.active_signature(work.nnls_gram); max_slope = τ, bounds = logmu_bounds, kwargs...)

    # A degenerate curve admits no corner; see `lcurve_corner`. Return the unregularized solution rather than an arbitrary near-zero μ
    isnan(logmu_final) && return (; x = solution(work.nnls_prob), mu = zero(T), chi2 = one(T))

    # Return the final regularized solution, recomputed via QR
    mu_final = exp(logmu_final)
    x_final = nnls_gram_polish_solve!(work, mu_final)
    chi2_final = chi2_ratio(resnorm_sq(work.nnls_prob_smooth_cache[]), resnorm_sq(work.nnls_prob))

    return (; x = x_final, mu = mu_final, chi2 = chi2_final)
end

# The L-curve at μ: the log-log point, its analytic κ and ω, and a digest of the active set.
function lcurve_point(work::NNLSLCurveRegProblem{T}, μ::T) where {T}
    (; nnls_gram) = work
    ξ² = NNLS.solve!(nnls_gram, work.A, work.b, μ)
    if isnan(ξ²)
        solve!(work.nnls_prob_smooth_cache, μ)
        cache = work.nnls_prob_smooth_cache[]
        (; nnls_work) = cache.nnls_prob
        NNLS.set_active!(nnls_gram, work.A, nnls_work.idx, NNLS.ncomponents(nnls_work))
        ξ², η², q = resnorm_sq(cache), seminorm_sq(cache), gradient_temps(cache).xᵀB⁻¹x
    else
        η², q = NNLS.seminorm_sq(nnls_gram), NNLS.inv_quadratic_form(nnls_gram)
    end
    κ, ω = lcurve_geometry(ξ², η², q, μ)
    return LCurveCornerPoint(SA{T}[log(ξ²), log(η²)], κ, ω, NNLS.active_signature(nnls_gram))
end

struct LCurveCornerPoint{T}
    P::SVector{2, T} # log-log L-curve point (log‖Ax-b‖², log‖x‖²)
    κ::T # analytic signed curvature of the log-log curve; see `lcurve_geometry`
    ω::T # analytic angular velocity dθ/dt of the tangent, t = logμ
    sig::UInt128 # digest of the active set; see `NNLS.active_signature`
end

# A golden state is four abscissas t₁ < t₂ < t₃ < t₄ with t₂ = t₁ + Δ/φ², t₃ = t₁ + Δ/φ and Δ = t₄ - t₁.
# `C` and `κ⃗ₒ` are stored rather than looked up so that every decision is a function of the branch state alone.
struct LCurveCornerState{T}
    t⃗::SVector{4, T} # grid of log-regularization parameters
    p⃗::SVector{4, LCurveCornerPoint{T}} # L-curve points evaluated at t⃗
    C::SVector{2, T} # Menger curvatures of the triples (p₁, p₂, p₃) and (p₂, p₃, p₄)
    κ⃗ₒ::SVector{2, T} # curvature at the nearest point this branch evaluated outside [t₁, t₄] on each side, +Inf where it has evaluated none
end

struct LCurveCornerCachedFunction{T, F <: CachedFunction{T, LCurveCornerPoint{T}}, C <: GrowableCache{Int, LCurveCornerState{T}}}
    f::F
    state_stack::C # states whose discarded sibling branch is unexplored, innermost last
end
Base.empty!(f::LCurveCornerCachedFunction) = (empty!(f.f); empty!(f.state_stack); f)
(f::LCurveCornerCachedFunction{T})(t::T) where {T} = f.f(t)

@doc raw"""
    lcurve_corner(f, t₀, P₀, sig₀; kwargs...)

Locate a corner of the L-curve, following Cultrera and Callegaro (2020)[1] with a dynamically bracketed search.

`f(t)` returns the L-curve point at ``t = \log\mu`` with its curvature, angular velocity, and active-set digest; `P₀` and `sig₀` correspond to the unregularized solution at which the curve terminates.
Returns ``\log\mu`` at a corner, or `NaN` if none is found. A returned corner has positive curvature, is a local maximum, and satisfies the `max_slope` guard.

# Keyword arguments

  - `init_width`: width in ``\log\mu`` of the initial bracket, and the scale at which the first corner is sought.
  - `xtol`, `Ptol`: absolute tolerances on ``\log\mu`` and on the chord between log-log curve evaluation points.
  - `max_expand`, `max_backtrack`, `nsweep`, `max_candidates`: search budgets.
  - `max_slope`: reject corners whose log-log tangent slope exceeds this.
  - `bounds`: interval of ``\log\mu`` to search within.

# References

  1. A. Cultrera and L. Callegaro, "A simple algorithm to find the L-curve corner in the regularization of ill-posed inverse problems". IOPSciNotes, vol. 1, no. 2, p. 025004, Aug. 2020, https://doi.org/10.1088/2633-1357/abad0d.
"""
function lcurve_corner(f::LCurveCornerCachedFunction{T}, t₀::T, P₀::SVector{2, T}, sig₀::UInt128; init_width = 1.0, xtol = 1e-4, Ptol = 1e-4, max_expand::Int = 64, max_backtrack::Int = 4, nsweep::Int = 32, max_candidates::Int = 8, max_slope = T(Inf), bounds = (T(-Inf), T(Inf))) where {T}
    # Note: tolerances are absolute because typically the L-curve is on a log-log scale, and atol on log-log is equivalent to rtol on linear-linear
    xtol, Ptol, max_log_slope = T(xtol), T(Ptol), log(T(max_slope))

    # The search domain is where ρ = μ² is finite and normal, intersected with the caller's admissible interval.
    # The state slides to fit the domain rather than shrinking around the seed, so the first corner is still sought at the scale `init_width`.
    # Δ ≤ U - L together with 1/φ + 1/φ² = 1 gives t₁ ≥ L and t₄ ≤ U. The clamp is inert for 1 < τ < 6 + 2√17, where the caller has already excluded a seed outside [t₋, t₊].
    φ = T(Base.MathConstants.φ)
    L, U = max(log(floatmin(T)) / 2, T(bounds[1])), min(log(floatmax(T)) / 2, T(bounds[2]))
    L < U || return T(NaN)
    Δ = min(T(init_width), prevfloat(U - L))
    tₛ = clamp(t₀, L + Δ / φ^2, U - Δ / φ)
    t⃗ = SA[tₛ-Δ/φ^2, tₛ, tₛ+Δ/φ^3, tₛ+Δ/φ]
    init = golden_state(t⃗, SA[f(t⃗[1]), f(t⃗[2]), f(t⃗[3]), f(t⃗[4])], SA[T(Inf), T(Inf)])

    # A reversal brackets a curvature basin at the Menger scale, but only proposes it: the analytic κ may have no maximum inside it and genuine maxima outside.
    # Contraction and backtracking both stay within the candidate, so a failed certification resumes expanding outward; otherwise the search could never leave a basin it had entered.
    best = init
    for left in (contract_left(init), !contract_left(init))
        state, budget = init, max_expand
        while budget > 0
            found, candidate, state, chainbest, spent = expand_to_reversal(f, state, left, P₀, sig₀, budget, L, U)
            budget -= spent
            rank(chainbest) > rank(best) && (best = chainbest)
            !found && break
            t = lcurve_certify!(f, candidate; xtol, Ptol, max_log_slope, max_backtrack)
            !isnan(t) && return t
        end
    end

    # Both directions are exhausted; the best-turning state visited is the last candidate.
    t = lcurve_certify!(f, best; xtol, Ptol, max_log_slope, max_backtrack)
    !isnan(t) && return t

    # Nothing the branch search reached certifies, so fall back to a brute force sweep. NaN from the sweep leaves `lsqnonneg_lcurve!` returning the unregularized solution.
    return lcurve_sweep!(f, L, U, max_log_slope; nsweep, max_candidates, xtol, Ptol, max_backtrack)
end

# Sample the admissible domain uniformly and certify the brackets those samples propose, returning NaN if none certifies.
# Two passes, each trying up to `max_candidates` brackets in rank order: intervals of positive net tangent rotation, tried alone and then widened by one sample on either side, followed by the sampled discrete maxima of κ.
function lcurve_sweep!(f::LCurveCornerCachedFunction{T}, L::T, U::T, max_log_slope::T; nsweep::Int, max_candidates::Int, xtol::T, Ptol::T, max_backtrack::Int) where {T}
    nsweep >= 3 || return T(NaN)
    δ = (U - L) / (nsweep - 1)

    # Net rotation of the tangent across each interval, largest first. θ is continuous, so θᵢ₊₁ - θᵢ = ∫ω dt exactly; see `tangent_angle`.
    # The rotation is signed, and an interval can hide a bend cancelled by an opposite one, so this only ranks proposals and certifies nothing.
    prev = (T(Inf), 0)
    for _ in 1:max_candidates
        best = (T(0), 0) # only intervals that turn positively are worth proposing
        for i in 1:nsweep-1
            xᵢ = L + (i - 1) * δ
            cand = (tangent_angle(f(xᵢ + δ), xᵢ + δ) - tangent_angle(f(xᵢ), xᵢ), -i)
            cand < prev && cand > best && (best = cand)
        end
        best[2] == 0 && break
        prev = best
        a = L + (-best[2] - 1) * δ
        y = certify_span!(f, a, a + δ, δ, L, U; xtol, Ptol, max_log_slope, max_backtrack)
        !isnan(y) && return y
        y = certify_span!(f, max(a - δ, L), min(a + 2δ, U), δ, L, U; xtol, Ptol, max_log_slope, max_backtrack)
        !isnan(y) && return y
    end

    # Sampled discrete maxima of κ, largest first. The left comparison admits ties, resolving right as `contract_left` does, so a maximum falling between two equal samples is not lost; a monotone or constant sequence yields nothing.
    prev = (T(Inf), 0)
    for _ in 1:max_candidates
        best = (T(-Inf), 0)
        for i in 2:nsweep-1
            xᵢ = L + (i - 1) * δ
            p = f(xᵢ)
            (p.κ >= f(xᵢ - δ).κ && p.κ > f(xᵢ + δ).κ && is_admissible(p, xᵢ, max_log_slope)) || continue
            cand = (p.κ, -i)
            cand < prev && cand > best && (best = cand)
        end
        best[2] == 0 && break
        prev = best
        tⱼ = L + (-best[2] - 1) * δ
        y = certify_span!(f, tⱼ - δ, tⱼ + δ, δ, L, U; xtol, Ptol, max_log_slope, max_backtrack)
        !isnan(y) && return y
    end

    return T(NaN)
end

# Angle of the log-log tangent, whose direction (ρη², -ξ²) is continuous across an active-set transition.
tangent_angle(p::LCurveCornerPoint{T}, t::T) where {T} = -atan(exp(p.P[1] - p.P[2] - 2t))

# Search a proposed bracket for a corner, passing in the samples just outside it as `κ⃗ₒ`.
# Corners often sit on a bracket's edge, since κ rises up to an active-set transition and drops across it, leaving the largest sampled value the last one before the jump.
# Confirming such a corner takes a point beyond the edge, which is what `κ⃗ₒ` supplies.
function certify_span!(f::LCurveCornerCachedFunction{T}, a::T, c::T, δ::T, L::T, U::T; xtol::T, Ptol::T, max_log_slope::T, max_backtrack::Int) where {T}
    c - a > 0 || return T(NaN)
    φ = T(Base.MathConstants.φ)
    t⃗ = SA[a, a+(c-a)/φ^2, a+(c-a)/φ, c]
    κ⃗ₒ = SA[a - δ >= L ? f(a - δ).κ : T(Inf), c + δ <= U ? f(c + δ).κ : T(Inf)]
    return lcurve_certify!(f, golden_state(t⃗, SA[f(t⃗[1]), f(t⃗[2]), f(t⃗[3]), f(t⃗[4])], κ⃗ₒ); xtol, Ptol, max_log_slope, max_backtrack)
end

# Search `state` for a corner, and if none is found retry in a part the search skipped, up to `max_backtrack` times.
# Each contraction keeps half the state and discards the other, which `lcurve_localize!` records; the discards come back out oldest first, hence largest first.
# Retries are bounded because each searches from scratch, so an unbounded count would cost O(depth²) solves per voxel.
function lcurve_certify!(f::LCurveCornerCachedFunction{T}, state::LCurveCornerState{T}; xtol::T, Ptol::T, max_log_slope::T, max_backtrack::Int) where {T}
    empty!(f.state_stack)
    for _ in 0:max_backtrack
        t = lcurve_localize!(f, state; xtol, Ptol, max_log_slope)
        !isnan(t) && return t
        isempty(f.state_stack) && break
        _, parent = popfirst!(f.state_stack)
        state = move(f, parent, !contract_left(parent))
    end
    return T(NaN)
end

# Expand in one direction until the preferred contraction direction reverses.
# Returns whether a candidate was found, the state just before the reversal as the candidate bracket, the state expansion resumes from, the best-ranked state visited, and the expansions spent.
# Reaching a domain endpoint exhausts that side alone and leaves the other at its starting state, so both directions run before `rank` is consulted.
function expand_to_reversal(f::LCurveCornerCachedFunction{T}, state::LCurveCornerState{T}, left::Bool, P₀::SVector{2, T}, sig₀::UInt128, budget::Int, L::T, U::T) where {T}
    best = state
    prev = contract_left(state)
    for k in 1:budget
        (; t⃗) = state
        t = left ? t⃗[1] - (t⃗[4] - t⃗[1]) / T(Base.MathConstants.φ) : t⃗[4] + (t⃗[4] - t⃗[1]) / T(Base.MathConstants.φ)
        L < t < U || return (false, state, state, best, k - 1)
        expanded = expand(f, state, left)

        # Saturation is tested before the reversal: a saturated endpoint makes its Menger triples degenerate, so the expanded state's direction carries no information.
        is_saturated(left ? expanded.p⃗[1] : expanded.p⃗[4], P₀, sig₀) && return (false, state, state, best, k)

        if contract_left(expanded) != prev
            # The expansion that reversed also evaluated a point just outside `state`. Record it: a corner on that edge of `state` is confirmed only against a point beyond the edge.
            κₒ = left ? SA[expanded.p⃗[1].κ, state.κ⃗ₒ[2]] : SA[state.κ⃗ₒ[1], expanded.p⃗[4].κ]
            return (true, LCurveCornerState(state.t⃗, state.p⃗, state.C, κₒ), expanded, best, k)
        end

        rank(expanded) > rank(best) && (best = expanded)
        prev = contract_left(expanded)
        state = expanded
    end

    return (false, state, state, best, budget)
end

# Contract the bracket until a point certifies as a local maximum of the analytic curvature, returning NaN if none does.
function lcurve_localize!(f::LCurveCornerCachedFunction{T}, state::LCurveCornerState{T}; xtol::T, Ptol::T, max_log_slope::T) where {T}
    while true
        (; t⃗, p⃗) = state
        κᵢ = max(p⃗[2].κ, p⃗[3].κ)
        if is_equal_signature(state) && κᵢ > max(p⃗[1].κ, p⃗[4].κ)
            # `brent_minimize` bounds its returned bracket by 4·xatol and no better, so it is asked for a quarter of the target. An exhausted budget leaves the bracket wide and certifies nothing.
            t, _, (lo, hi) = brent_minimize(s -> -f(s).κ, t⃗[1], t⃗[4]; xatol = xtol / 4, xrtol = zero(T), maxiters = 40)
            hi - lo <= xtol && lo < t < hi && f(t).κ > max(f(lo).κ, f(hi).κ) && f(t).κ >= κᵢ && is_admissible(f(t), t, max_log_slope) && return t
        end
        is_converged(state; xtol, Ptol) && return best_corner(state, max_log_slope)
        push!(f.state_stack, (length(f.state_stack), state)) # the discarded sibling is `move(f, state, !contract_left(state))`, instantiated only if this branch certifies nothing
        state = move(f, state, contract_left(state))
    end
end

# Highest-curvature point of a state that exceeds both its neighbours, or NaN if none does.
# An interior point's neighbours are the adjacent state points; an endpoint's are its one adjacent state point and the matching entry of `κ⃗ₒ`, which is +Inf where nothing past the endpoint has been evaluated.
# A corner is therefore never declared at the edge of the region searched so far.
function best_corner(state::LCurveCornerState{T}, max_log_slope::T) where {T}
    (; t⃗, p⃗, κ⃗ₒ) = state
    κ⃗ = SA[κ⃗ₒ[1], p⃗[1].κ, p⃗[2].κ, p⃗[3].κ, p⃗[4].κ, κ⃗ₒ[2]]
    t, κ = T(NaN), T(-Inf)
    for i in 1:4
        κ⃗[i+1] > κ && κ⃗[i+1] > κ⃗[i] && κ⃗[i+1] > κ⃗[i+2] && is_admissible(p⃗[i], t⃗[i], max_log_slope) && ((t, κ) = (t⃗[i], κ⃗[i+1]))
    end
    return t
end

# A corner must curve the right way and must lie clear of the near-vertical μ → 0 tail.
# Both halves are needed: κ tends to a positive plateau η⁴/(2qξ²) in that tail and can rise out of it into a genuine maximum, so positive curvature alone does not exclude it. See `LCURVE_SLOPE_MAX_DEFAULT`.
is_admissible(p::LCurveCornerPoint{T}, t::T, max_log_slope::T) where {T} = p.κ > 0 && p.P[1] - p.P[2] - 2t <= max_log_slope

# Contract toward the higher state-local curvature. A negative right-hand curvature places the corner to the left regardless, which is the positive-curvature safeguard of Cultrera-Callegaro; exact equality resolves right.
contract_left(state::LCurveCornerState) = state.C[2] < 0 || state.C[1] > state.C[2]

# Whether all four abscissas returned the same active set, which gates the analytic-curvature maximization.
is_equal_signature(state::LCurveCornerState) = state.p⃗[1].sig == state.p⃗[2].sig == state.p⃗[3].sig == state.p⃗[4].sig

# Contraction stops when the abscissas span less than `xtol` or the endpoints span less than `Ptol` of L-curve arc. The two are not comparable: `xtol` resolves the abscissa, `Ptol` the solution.
# Monotonicity of the Tikhonov path gives ‖x₁ - x₂‖² ≤ tanh(t₂ - t₁)·(η²₁ - η²₂), across active-set changes and not merely within one branch, and so
#   ‖Δx‖/‖x‖ ≤ √(tanh(Δt)·(1 - e^{-Δlog η²})) ≤ √Ptol.
# That bounds the spread of solutions within one state, not the distance to the corner an `xtol`-resolved search would return: stopping on `Ptol` leaves the state wide in t, so `best_corner` compares distant neighbours and accepts a coarser feature of κ.
is_converged(state::LCurveCornerState; xtol, Ptol) = abs(state.t⃗[4] - state.t⃗[1]) < xtol || norm(state.p⃗[1].P - state.p⃗[4].P) < Ptol

# Whether the path has run out at this endpoint, so that expanding past it would add nothing: to the left x has reached the unregularized solution, to the right it has collapsed to x = 0 and log‖x‖² is -∞.
# The left test needs both parts. P alone moves arbitrarily little between two distinct solutions on a flat stretch, and the digest alone repeats whenever an unrelated μ happens to share the active set.
is_saturated(p::LCurveCornerPoint{T}, P₀::SVector{2, T}, sig₀::UInt128) where {T} = !isfinite(p.P[2]) || (p.sig == sig₀ && maximum(abs, p.P - P₀) <= √eps(T))

isfinite_else(a::T, b::T) where {T} = isfinite(a) ? a : b

# Ranking of the states visited during expansion, consulted only when both directions reach an endpoint without ever reversing.
# ω leads rather than κ: the two share a sign, but ω weights curvature by the speed |dP/dt|, and vanishing speed is what marks the μ → 0 tail. Ties fall to the higher state-local curvature, then to the tighter state.
rank(state::LCurveCornerState{T}) where {T} = (max(zero(T), isfinite_else(state.p⃗[2].ω, zero(T)), isfinite_else(state.p⃗[3].ω, zero(T))), max(isfinite_else(state.C[1], typemin(T)), isfinite_else(state.C[2], typemin(T))), state.t⃗[1] - state.t⃗[4])

golden_state(t⃗::SVector{4, T}, p⃗::SVector{4, LCurveCornerPoint{T}}, κ⃗ₒ::SVector{2, T}) where {T} = LCurveCornerState(t⃗, p⃗, SA[state_curvature(p⃗[1], p⃗[2], p⃗[3]), state_curvature(p⃗[2], p⃗[3], p⃗[4])], κ⃗ₒ)

# Menger curvature of one state triple, rejected as unresolved when the circumcircle is indeterminate.
function state_curvature(pⱼ::LCurveCornerPoint{T}, pₖ::LCurveCornerPoint{T}, pₗ::LCurveCornerPoint{T}) where {T}
    Pⱼ, Pₖ, Pₗ = pⱼ.P, pₖ.P, pₗ.P

    # A zero solution puts log η² at -∞ and leaves κ = 0/0. The expansion chain stops on `is_saturated` before consulting such a point, but the four seed abscissas are not screened, so the triple rejects it here.
    (isfinite(pⱼ.κ) && isfinite(pₖ.κ) && isfinite(pₗ.κ) && all(isfinite, Pⱼ) && all(isfinite, Pₖ) && all(isfinite, Pₗ)) || return T(-Inf)
    Δⱼₖ, Δₖₗ = Pⱼ - Pₖ, Pₖ - Pₗ
    scale = 1 + max(norm(Pⱼ), norm(Pₖ), norm(Pₗ))
    if min(norm(Δⱼₖ), norm(Δₖₗ), norm(Pₗ - Pⱼ)) <= √eps(T) * scale || abs(Δⱼₖ × Δₖₗ) <= √eps(T) * norm(Δⱼₖ) * norm(Δₖₗ)
        return pⱼ.sig == pₖ.sig == pₗ.sig ? pₖ.κ : T(-Inf)
    end

    return menger(Pⱼ, Pₖ, Pₗ)
end

move(f::LCurveCornerCachedFunction, state::LCurveCornerState, left::Bool) = left ? move_left(f, state) : move_right(f, state)
expand(f::LCurveCornerCachedFunction, state::LCurveCornerState, left::Bool) = left ? expand_left(f, state) : expand_right(f, state)

# Golden contraction: one new point, width divided by φ, golden proportions preserved.
function move_left(f::LCurveCornerCachedFunction{T}, state::LCurveCornerState{T}) where {T}
    (; t⃗, p⃗, κ⃗ₒ) = state
    φ = T(Base.MathConstants.φ)
    t = (φ * t⃗[1] + t⃗[3]) / (φ + 1) # t₁ + Δ/φ³
    return golden_state(SA[t⃗[1], t, t⃗[2], t⃗[3]], SA[p⃗[1], f(t), p⃗[2], p⃗[3]], SA[κ⃗ₒ[1], p⃗[4].κ])
end

function move_right(f::LCurveCornerCachedFunction{T}, state::LCurveCornerState{T}) where {T}
    (; t⃗, p⃗, κ⃗ₒ) = state
    φ = T(Base.MathConstants.φ)
    t = (φ * t⃗[4] + t⃗[2]) / (φ + 1) # t₄ - Δ/φ³
    return golden_state(SA[t⃗[2], t⃗[3], t, t⃗[4]], SA[p⃗[2], p⃗[3], f(t), p⃗[4]], SA[p⃗[1].κ, κ⃗ₒ[2]])
end

# Inverse golden expansion: one new point, width multiplied by φ. Exactly inverse to a contraction, so `move_right(expand_left(state)) == state` and `move_left(expand_right(state)) == state`.
function expand_left(f::LCurveCornerCachedFunction{T}, state::LCurveCornerState{T}) where {T}
    (; t⃗, p⃗, κ⃗ₒ) = state
    φ = T(Base.MathConstants.φ)
    t = t⃗[1] - (t⃗[4] - t⃗[1]) / φ
    return golden_state(SA[t, t⃗[1], t⃗[2], t⃗[4]], SA[f(t), p⃗[1], p⃗[2], p⃗[4]], SA[T(Inf), κ⃗ₒ[2]])
end

function expand_right(f::LCurveCornerCachedFunction{T}, state::LCurveCornerState{T}) where {T}
    (; t⃗, p⃗, κ⃗ₒ) = state
    φ = T(Base.MathConstants.φ)
    t = t⃗[4] + (t⃗[4] - t⃗[1]) / φ
    return golden_state(SA[t⃗[1], t⃗[3], t⃗[4], t], SA[p⃗[1], p⃗[3], p⃗[4], f(t)], SA[κ⃗ₒ[1], T(Inf)])
end

function menger(Pⱼ::V, Pₖ::V, Pₗ::V) where {V <: SVector{2}}
    Δⱼₖ, Δₖₗ, Δₗⱼ = Pⱼ - Pₖ, Pₖ - Pₗ, Pₗ - Pⱼ
    P̄ⱼP̄ₖ, P̄ₖP̄ₗ, P̄ₗP̄ⱼ = Δⱼₖ ⋅ Δⱼₖ, Δₖₗ ⋅ Δₖₗ, Δₗⱼ ⋅ Δₗⱼ
    Cₖ = 2 * (Δⱼₖ × Δₖₗ) / √(P̄ⱼP̄ₖ * P̄ₖP̄ₗ * P̄ₗP̄ⱼ)
    return Cₖ
end

function menger(f; h = 1e-3)
    function menger_curvature_inner(t)
        fⱼ, fₖ, fₗ = f(t - h), f(t), f(t + h)
        Pⱼ, Pₖ, Pₗ = SA[t-h, fⱼ], SA[t, fₖ], SA[t+h, fₗ]
        return menger(Pⱼ, Pₖ, Pₗ)
    end
end

function menger(ξ, η; h = 1e-3)
    function menger_curvature_inner(t)
        ξ₋, ξ₀, ξ₊ = ξ(t - h), ξ(t), ξ(t + h)
        η₋, η₀, η₊ = η(t - h), η(t), η(t + h)
        ξ′, ξ′′ = (ξ₊ - ξ₋) / 2h, (ξ₊ - 2ξ₀ + ξ₋) / h^2
        η′, η′′ = (η₊ - η₋) / 2h, (η₊ - 2η₀ + η₋) / h^2
        return (ξ′ * η′′ - η′ * ξ′′) / √((ξ′^2 + η′^2)^3)
    end
end

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

Compute the Tikhonov-regularized nonnegative least-squares (NNLS) solution ``x_{\mu}`` of the problem:

```math
x_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu^2 ||x||_2^2
```

where ``\mu`` is chosen by Regińska's minimum-product criterion[1]:

```math
\mu = \underset{\nu > 0}{\operatorname{argmin}}\; \Psi(\nu) = ||Ax_{\nu} - b||_2^2 \, ||x_{\nu}||_2^2,
```

taking the smallest local minimizer of ``\Psi``.
Stationarity of ``\Psi`` is equivalent to a log-log L-curve tangent slope of ``-1``, so the selected ``\mu`` is the balance point ``||Ax_{\mu} - b|| = \mu ||x_{\mu}||``.
The smallest local minimizer is taken because ``\Psi \to 0`` trivially as ``\mu \to \infty``, where ``x_{\mu} \to 0``.
If ``\Psi`` has no interior local minimum, the unregularized solution is returned with ``\mu = 0``.

# Arguments

  - `A::AbstractMatrix`: Decay basis matrix
  - `b::AbstractVector`: Decay curve data

# Outputs

  - `x::AbstractVector`: Regularized NNLS solution
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
    work::NNLSReginskaRegProblem{T}; atol = 1e-4, h = 0.5, # floors the leap, bounding the step count and setting the resolution at which a crossing pair can be stepped over; see the scan below
) where {T}
    reset_cache!(work.nnls_prob_smooth_cache)

    # Evaluations run on the Gram fast path, one warm-chained μ-solve yielding both ‖Ax-b‖² and ‖x‖²; the final solution is recomputed via QR at the selected μ
    solve_unreg!(work.nnls_prob, work.nnls_prob_seed) # unregularized solution seeds the Gram fast path
    x_unreg = solution(work.nnls_prob)
    res²_min = resnorm_sq(work.nnls_prob)

    # An exact unregularized fit makes the minimum-product criterion zero at μ = 0.
    b² = sum(abs2, work.b)
    if res²_min <= eps(T) * b² || ncomponents(work.nnls_prob) == 0
        return (; x = x_unreg, mu = zero(T), chi2 = one(T))
    end
    η²_unreg = sum(abs2, x_unreg)
    nnls_gram_setup!(work)

    # g(logμ) = log|S| = log res² − log ‖x‖² − 2 logμ is the log-log L-curve tangent slope magnitude, continuous in μ and derivative-free from one Gram evaluation.
    # Ψ = res²·‖x‖² satisfies dlogΨ/dlogμ = ξ'·(1 + S) with ξ' ≥ 0, so the smallest local minimizer of Ψ is exactly the leftmost downward crossing g = 0, i.e. the balance point |S| = 1.
    # `res²` is returned alongside because it also certifies when the scan may stop; see below.
    function g_and_res²(logμ)
        res², η² = nnls_gram_losses!(work, exp(logμ))
        return (η² == 0 ? T(+Inf) : log(res²) - log(η²) - 2 * logμ), res²
    end
    g(logμ) = first(g_and_res²(logμ))

    # |S| → ∞ at both ends, since μ²‖x‖² → 0 with res² → res²_min > 0 as μ → 0, and ‖x‖² ~ C/μ⁴ as μ → ∞. So |S| = 1 generically has an even number of crossings, and a left-to-right scan exiting at the first sign change is what identifies the leftmost.
    # res² is nondecreasing and ‖x‖² nonincreasing, so g(b) ≥ g(a) − 2(b − a) for b > a with no smoothness assumption, and from g(a) > 0 the first crossing lies at or beyond a + g(a)/2.
    # Leaping by max(h, g(a)/2) therefore keeps the crossing-detection resolution h of a uniform scan where g is small, and takes exponentially fewer evaluations where g is large, g ≈ −2 logμ + O(1) as μ → 0.
    # Those same monotonicities make Φ(μ) = ‖Ax(μ) − b‖/‖x(μ)‖ nondecreasing, and the balance points are its fixed points, so all of them satisfy μ ≥ Φ(0) and the scan can start there with no lower bound.
    # The leap logμ -> logμ + g/2 is a step of that fixed-point map, which alone never brackets: g > 0 means Φ(μ) > μ, and applying the nondecreasing Φ gives g ≥ 0 at every iterate.
    # The floor h is what carries the scan past the crossing, bounding it at (logμ_cert − logμ₀)/h steps and handing Brent a genuine sign change, overstepping the certified interval (a, a + g/2) by at most h − g/2.
    # No upper bound is needed either: complementarity gives bᵀr = res² + μ²‖x‖², which at a balance point reads bᵀr = 2·res², and Cauchy-Schwarz bounds res² ≤ ‖b‖²/4 there.
    # res² is nondecreasing, so the first scan point exceeding ‖b‖²/4 proves no balance point lies at or beyond it, and res² → ‖b‖² makes that test eventually fire.
    logμ₀ = (log(res²_min) - log(η²_unreg)) / 2
    res²_max = b² / 4

    a, ga = logμ₀, g(logμ₀)
    if ga <= 0
        logmu_final = logμ₀ # Φ(0) is itself the balance point, to within the resolution of one Gram evaluation
    else
        b, gb = a, ga
        while true
            b = a + max(T(h), ga / 2)
            gb, res²_b = g_and_res²(b)
            gb <= 0 && break
            res²_b > res²_max && return (; x = x_unreg, mu = zero(T), chi2 = one(T)) # no balance point exists
            a, ga = b, gb
        end
        logmu_final, _ = brent_root(g, a, b, ga, gb; xatol = T(atol), xrtol = T(0), ftol = T(0), maxiters = 100)
    end

    mu_final = exp(logmu_final)
    x_final = nnls_gram_polish_solve!(work, mu_final)
    chi2_final = chi2_ratio(resnorm_sq(work.nnls_prob_smooth_cache[]), res²_min)

    return (; x = x_final, mu = mu_final, chi2 = chi2_final)
end

####
#### GCV method for choosing the Tikhonov regularization parameter
####

struct NNLSGCVRegProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, WS, WD, W1, W2, W3, V, S}
    A::TA # decay basis matrix
    b::Tb # decay curve data
    m::Int # number of rows of A
    n::Int # number of columns of A
    γ²::Vector{T} # squared singular values of A, i.e. nonzero eigenvalues of A'A
    spectrum_work::WS # workspace for computing the singular values of A
    deflation_work::WD # workspace for `deflated_eigvals!`, or nothing without an α-grid
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
    deflation_work = dof_interpolator === nothing ? nothing : SVDValsWorkspace(similar(A, T, (max(m, n), min(m, n))))
    return NNLSGCVRegProblem(A, b, m, n, γ², spectrum_work, deflation_work, nnls_prob, nnls_prob_smooth_cache, nnls_gram, dof_interpolator, nnls_prob_seed)
end

@inline solution(work::NNLSGCVRegProblem) = solution(work.nnls_prob_smooth_cache[])
@inline ncomponents(work::NNLSGCVRegProblem) = ncomponents(work.nnls_prob_smooth_cache[])

# Runtime toggle for deflating numerically null spectral modes.
const GCV_DEFLATE_SPECTRUM = Ref(true)

# Compute the spectrum read by `gcv_dof`.
@inline function LinearAlgebra.eigvals!(work::NNLSGCVRegProblem, A = work.A)
    (; γ², spectrum_work, deflation_work, dof_interpolator) = work
    (dof_interpolator === nothing || !GCV_DEFLATE_SPECTRUM[]) && return eigvals_full!(γ², spectrum_work, A)
    interp, α = dof_interpolator
    return deflated_eigvals!(γ², spectrum_work, deflation_work, A, (@views interp.Q[:, :, findnearestindex(interp.αs, α[])]))
end

@doc raw"""
    lsqnonneg_gcv(A::AbstractMatrix, b::AbstractVector)

Compute the Tikhonov-regularized nonnegative least-squares (NNLS) solution ``x_{\mu}`` of the problem:

```math
x_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu^2 ||L x||_2^2
```

where ``L`` is the identity matrix, and ``\mu`` is chosen via the Generalized Cross-Validation (GCV) method:

```math
\mu = \underset{\nu \ge 0}{\operatorname{argmin}}\; \frac{||Ax_{\nu} - b||_2^2}{\mathcal{T}(\nu)^2}
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

  - `x::AbstractVector`: Regularized NNLS solution
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

function lsqnonneg_gcv!(work::NNLSGCVRegProblem{T}; method = :brent, rtol = 0.0, atol = 1e-4, maxiters = 20) where {T}
    # Find μ by minimizing the function G(μ) (GCV method)
    # Precompute the squared singular values, which are all that dof(μ) needs. The opt-in alternative instead interpolates dof across the α-grid slices at each evaluation; see `gcv_dof_interp`.
    dof_interpolator = work.dof_interpolator
    use_dof_interp = dof_interpolator !== nothing && GCV_INTERP_DOF[] && method === :brent
    use_dof_interp || eigvals!(work)
    𝒟(μ) = use_dof_interp ? gcv_dof_interp(dof_interpolator[1], dof_interpolator[2][], work.m, work.n, μ) : gcv_dof(work.m, work.n, work.γ², μ)

    # `:brent`, the default, evaluates 𝒢(μ) = ‖Ax(μ)-b‖² / dof(μ)² on the Gram fast path. Only the residual needs an NNLS solve; dof is a cheap function of μ once the singular values are known.
    # The final solution is always recomputed via QR for accuracy.
    use_gram = method === :brent
    reset_cache!(work.nnls_prob_smooth_cache)
    solve_unreg!(work.nnls_prob, work.nnls_prob_seed) # the unregularized solution anchors the search interval and seeds the Gram fast path
    nnls_gram_setup!(work) # also loads Aᵀb, which the search interval needs

    b², R₀, N₀ = sum(abs2, work.b), resnorm_sq(work.nnls_prob), seminorm_sq(work.nnls_prob)
    r₀ = NNLS.residual(work.nnls_prob.nnls_work)
    Ax₀² = sum(i -> abs2(work.b[i] - r₀[i]), eachindex(work.b))
    d = max(work.m - work.n, 0) # dof(0⁺), the limit `gcv_dof` takes as μ → 0

    # x₀ = 0 forces x_μ = 0 for every μ since ‖x_μ‖ is nonincreasing.
    N₀ <= 0 && return (; x = solution(work.nnls_prob), mu = zero(T), chi2 = one(T))

    # On a fit exact to working precision 𝒢 = R₀/dof² has approximately zero numerator, and its infimum R₀/d² = 0 is attained at μ = 0 whenever d > 0.
    # For d = 0 this is not necessarily the case: dof vanishes at μ = 0, too, and thus 𝒢 tends to a finite plateau, and the infimum can sit at μ = ∞.
    R₀ <= eps(T) * b² && d > 0 && return (; x = solution(work.nnls_prob), mu = zero(T), chi2 = one(T))

    c₊ = sqrt(sum(cⱼ -> max(cⱼ, zero(T))^2, work.nnls_gram.c))
    logμ₋, logμ₊ = gcv_bracket(𝒟, work.m, d, b², R₀, N₀, Ax₀², c₊, (log(R₀) - log(N₀)) / 2)

    if isnan(logμ₋)
        # If the bracket is not certified, anchor on the envelope balance point ‖Ax₀‖² = μ²N₀, which remains finite if R₀ vanishes.
        t₀ = (log(Ax₀²) - log(N₀)) / 2
        logμ₀, (logμ₋, logμ₊) = t₀, (t₀ - 5, t₀ + 5)
    else
        logμ₀ = (logμ₋ + logμ₊) / 2
    end

    # 𝒢 needs no guard: it is strictly positive for μ > 0 and b ≠ 0, since KKT complementarity gives xᵀd = 0, hence (Ax)ᵀr = μ²‖x‖², and with Ax = b − r this reads bᵀr = res² + μ²‖x‖², so res² = 0 would force x = 0 and then b = 0.
    function 𝒢(logμ)
        use_gram || return gcv!(work, logμ)
        μ = exp(logμ)
        res², _ = nnls_gram_losses!(work, μ)
        return res² / 𝒟(μ)^2
    end
    𝒢_and_∇𝒢(logμ) = gcv_and_dgcv_dlogμ!(work, logμ)

    if method === :brent
        # Gradient-free golden-section/parabolic search over the full bounds. Convergence needs the bracket width to reach `atol`, not merely a good point, so a warm start cannot speed it up without narrowing the bounds a priori.
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

    # Return the final regularized solution, (re)computed via QR
    mu_final = exp(logmu_final)
    x_final = use_gram ? nnls_gram_polish_solve!(work, mu_final) : solve!(work.nnls_prob_smooth_cache, mu_final)
    chi2_final = chi2_ratio(resnorm_sq(work.nnls_prob_smooth_cache[]), resnorm_sq(work.nnls_prob))

    return (; x = x_final, mu = mu_final, chi2 = chi2_final)
end

# Search interval for the GCV minimizer, certified against a bound U ≥ inf 𝒢 that costs no NNLS solve. Returns NaN endpoints where no such certificate exists.
# The feasible ray s·x₀ supplies U: complementarity gives ‖A(s·x₀) - b‖² = R₀ + (1-s)²‖Ax₀‖², so minimizing that plus μ²s²N₀ over s bounds 𝒢 above by Φ.
# Below, R ≥ R₀ forces dof(μ) ≥ √(R₀/U) at every minimizer, which bounds μ since dof increases from d to m.
# Above, KKT gives (Aᵀb)ᵀx_μ = ‖Ax_μ‖² + μ²‖x_μ‖², which with (Aᵀb)ᵀx_μ ≤ ‖(Aᵀb)₊‖·‖x_μ‖ yields ‖Ax_μ‖ ≤ ‖(Aᵀb)₊‖/2μ and ‖x_μ‖ ≤ ‖(Aᵀb)₊‖/μ².
# Either R ≥ (‖b‖ - ‖Ax_μ‖)₊² or R ≥ ‖b‖² - 2‖(Aᵀb)₊‖²/μ², against dof ≤ m, then bounds μ from above.
# Both endpoints scale like A, so the interval translates rigidly by log s under A → s·A.
function gcv_bracket(𝒟, m::Int, d::Int, b²::T, R₀::T, N₀::T, Ax₀²::T, c₊::T, t_bal::T) where {T}
    # Φ ≥ 𝒢 pointwise, so every value of Φ is admissible as U; the smallest U certifies the tightest interval.
    # Φ inherits 𝒢's own endpoint limits, Φ(0⁺) = R₀/d² and Φ(∞) = ‖b‖²/m², since ‖Ax₀‖² = ‖b‖² - R₀ by complementarity; Φ dips below them iff a certificate exists.
    function Φ(t)
        μ = exp(t)
        ρN₀ = μ^2 * N₀
        return (R₀ + Ax₀² * ρN₀ / (Ax₀² + ρN₀)) / 𝒟(μ)^2
    end
    tΦ₁, tΦ₂ = bracket_minimum(Φ, t_bal, one(T); dilate = T(1.5), maxiters = 12)
    isnan(tΦ₁) && return (T(NaN), T(NaN)) # Φ descends without turning, so its infimum is one of the endpoint limits and nothing interior is certified
    _, U = brent_minimize(Φ, tΦ₁, tΦ₂; xatol = T(1e-2), xrtol = zero(T)) # Φ is smooth at its minimum, so the error in U falls off as the square of the resolution in the argmin

    # A finite minimizer exists only where U undercuts both endpoint limits, 𝒢(0⁺) = R₀/d² and 𝒢(∞) = ‖b‖²/m².
    # Writing the left condition through dof rather than through R₀/d² covers d = 0, where A has full row rank and dof(0⁺) vanishes.
    𝒟min = sqrt(R₀ / U)
    (d < 𝒟min < m && m^2 * U < b²) || return (T(NaN), T(NaN))

    Δ𝒟(t) = 𝒟(exp(t)) - 𝒟min
    t₁, t₂, Δ₁, Δ₂ = bracket_root_monotonic(Δ𝒟, t_bal, one(T); dilate = T(1.5), mono = +1, maxiters = 12)
    Δ₁ * Δ₂ > 0 && return (T(NaN), T(NaN)) # dof never reaches √(R₀/U), so A is more rank deficient than d = max(m - n, 0) assumes
    _, _, (t₋, _), _ = bisect_root(Δ𝒟, t₁, t₂, Δ₁, Δ₂; xatol = T(1e-3), maxiters = 100) # the lower end of the final bracket keeps dof ≤ √(R₀/U), so it remains a valid bound

    t₊ = log(c₊ * min(sqrt(2 / (b² - m^2 * U)), inv(2 * (sqrt(b²) - m * sqrt(U)))))
    return t₋ < t₊ ? (t₋, t₊) : (T(NaN), T(NaN))
end

# Implements equation (32) from:
#
#   Analysis of Discrete Ill-Posed Problems by Means of the L-Curve
#   Hansen et al. 1992 (https://epubs.siam.org/doi/10.1137/1034115)
#
# where here L = Id and λ = μ.
function gcv!(work::NNLSGCVRegProblem, logμ)
    # Unpack buffers
    #   NOTE: assumes `eigvals!(work)` has filled `work.γ²`
    (; m, n, γ²) = work

    # Solve regularized NNLS problem
    μ = exp(logμ)
    solve!(work.nnls_prob_smooth_cache, μ)
    cache = work.nnls_prob_smooth_cache[]

    # Compute GCV
    res² = resnorm_sq(cache) # squared residual norm ‖A * x(μ) - b‖^2
    dof = gcv_dof(m, n, γ², μ) # degrees of freedom
    gcv = res² / dof^2

    return gcv
end

function gcv_and_dgcv_dlogμ!(work::NNLSGCVRegProblem, logμ)
    # Unpack buffers
    #   NOTE: assumes `eigvals!(work)` has filled `work.γ²`
    (; m, n, γ²) = work

    # Solve regularized NNLS problem
    μ = exp(logμ)
    solve!(work.nnls_prob_smooth_cache, μ)
    cache = work.nnls_prob_smooth_cache[]

    # Compute primal
    res² = resnorm_sq(cache) # squared residual norm ‖A * x(μ) - b‖^2
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

# DOF derivative with respect to the flip angle α. From dof = max(m−n, 0) + Σᵢ λ²/(γᵢ²+λ²), ∂dof/∂α = −Σᵢ λ²·(dγᵢ²/dα)/(γᵢ²+λ²)², with `dγ²` supplying dγᵢ²/dα = 2σᵢ·uᵢᵀ(∂A/∂α)vᵢ.
# It is smooth through branch crossings, dof being a symmetric function of the spectrum.
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
# The dof itself, dof(μ, α) = max(m − n, 0) + μ²·tr((A(α)ᵀA(α) + μ²I)⁻¹), is a symmetric function of the spectrum and hence analytic in α, so cubic Hermite with the analytic ∂dof/∂α from `dgcv_dof_dα` is kink-free and O(h⁴) accurate.
function gcv_dof_interp(interp::GriddedSpectrumInterpolator{T}, α::T, m::Int, n::Int, μ::T) where {T}
    (; αs, γ², dγ²) = interp
    i = clamp(searchsortedlast(αs, α), 1, length(αs) - 1)
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

####
#### ℓ¹-regularized NNLS problem
####

@inline solution(work::NNLS.NNLSLassoWorkspace) = NNLS.solution(work)
@inline ncomponents(work::NNLS.NNLSLassoWorkspace) = NNLS.ncomponents(work)
@inline resnorm_sq(work::NNLS.NNLSLassoWorkspace) = sum(abs2, NNLS.residual(work))
@inline seminorm(work::NNLS.NNLSLassoWorkspace) = sum(NNLS.solution(work)) # ‖x‖₁ = 𝟙ᵀx where x ≥ 0

@doc raw"""
    lsqnonneg_lasso(A::AbstractMatrix, b::AbstractVector, μ::Real)

Compute the ``\ell^1``-regularized nonnegative least-squares (NNLS) solution ``x_{\mu}`` of the problem:

```math
x_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu ||x||_1.
```

Nonnegativity makes ``||x||_1 = \mathbf{1}^T x`` linear, so this is a smooth bound-constrained convex quadratic program, solved by a Lawson-Hanson active-set method.

# Arguments

  - `A::AbstractMatrix`: Left hand side matrix acting on `x`
  - `b::AbstractVector`: Right hand side vector
  - `μ::Real`: Regularization parameter

# Outputs

  - `x::AbstractVector`: NNLS solution
"""
lsqnonneg_lasso(A::AbstractMatrix, b::AbstractVector, μ::Real; kwargs...) = lsqnonneg_lasso!(lsqnonneg_lasso_work(A, b), μ; kwargs...)
lsqnonneg_lasso_work(A::AbstractMatrix, b::AbstractVector) = NNLS.NNLSLassoWorkspace(A, b)
lsqnonneg_lasso!(work::NNLS.NNLSLassoWorkspace, μ::Real; kwargs...) = NNLS.solve!(NNLS.reset!(work), μ; kwargs...)

# Compute the μ = 0 endpoint of the ℓ¹ L-curve, which every selector measures its criterion against.
# Returns μmax, the μ = 0 residual R₀ and seminorm N₀, and ‖b‖², with the workspace in a solved state.
function lasso_baseline!(work)
    (; nnls_prob, lasso_work) = work
    solve_unreg!(nnls_prob, work.nnls_prob_seed)
    NNLS.reset!(lasso_work, solution(nnls_prob))
    μmax = NNLS.regparam_max(lasso_work)
    NNLS.solve!(lasso_work, zero(eltype(work.b)))
    return (; μmax, R₀ = resnorm_sq(lasso_work), N₀ = seminorm(lasso_work), b² = sum(abs2, work.b))
end

####
#### Chi2 method for choosing the ℓ¹ regularization parameter
####

struct NNLSChi2LassoRegProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W1, W2, S}
    A::TA # decay basis matrix
    b::Tb # decay curve data
    m::Int # number of rows of A
    n::Int # number of columns of A
    nnls_prob::W1 # unregularized NNLS problem, i.e. μ = 0
    lasso_work::W2 # ℓ¹-regularized solver workspace, warm-started across the μ-search
    nnls_prob_seed::S # source for the unregularized solve; see `NNLSUnregSource`
end
function NNLSChi2LassoRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}, nnls_prob_seed::NNLSUnregSource{T} = nothing) where {T}
    m, n = size(A)
    return NNLSChi2LassoRegProblem(A, b, m, n, NNLSProblem(A, b), NNLS.NNLSLassoWorkspace(A, b), nnls_prob_seed)
end

@inline solution(work::NNLSChi2LassoRegProblem) = solution(work.lasso_work)
@inline ncomponents(work::NNLSChi2LassoRegProblem) = ncomponents(work.lasso_work)

@doc raw"""
    lsqnonneg_chi2_lasso(A::AbstractMatrix, b::AbstractVector, chi2_target::Real)

Compute the ``\ell^1``-regularized nonnegative least-squares (NNLS) solution ``x_{\mu}`` of the problem:

```math
x_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu ||x||_1
```

where ``\mu`` is determined by solving:

```math
\chi^2(\mu) = \frac{||Ax_{\mu} - b||_2^2}{||Ax_{0} - b||_2^2} = \chi^2_{\mathrm{target}}.
```

This is the ``\ell^1`` counterpart of [`lsqnonneg_chi2`](@ref).

# Arguments

  - `A::AbstractMatrix`: Decay basis matrix
  - `b::AbstractVector`: Decay curve data
  - `chi2_target::Real`: Target ``\chi^2(\mu)``; typically a small value, e.g. 1.02 representing a 2% increase

# Outputs

  - `x::AbstractVector`: Regularized NNLS solution
  - `mu::Real`: Resulting regularization parameter ``\mu``
  - `chi2::Real`: Resulting ``\chi^2(\mu)``, which should be approximately equal to `chi2_target`
"""
function lsqnonneg_chi2_lasso(A::AbstractMatrix, b::AbstractVector, chi2_target::Real)
    work = lsqnonneg_chi2_lasso_work(A, b)
    return lsqnonneg_chi2_lasso!(work, chi2_target)
end
lsqnonneg_chi2_lasso_work(A::AbstractMatrix, b::AbstractVector, nnls_prob_seed = nothing) = NNLSChi2LassoRegProblem(A, b, nnls_prob_seed)

function lsqnonneg_chi2_lasso!(work::NNLSChi2LassoRegProblem{T}, chi2_target::Real) where {T}
    (; lasso_work) = work
    @assert chi2_target >= 1 "chi2_target must be at least 1, but chi2_target = $chi2_target."
    χ²_target = T(chi2_target)

    (; μmax, R₀, b²) = lasso_baseline!(work)
    res²_min = R₀

    if res²_min <= eps(T) * b² || ncomponents(lasso_work) == 0
        # An exact fit has target residual zero, whose only root is μ = 0, and a zero unregularized solution remains zero for every μ.
        return (; x = solution(lasso_work), mu = zero(T), chi2 = one(T))
    end

    res²_target = χ²_target * res²_min
    if res²_target >= b²
        # The requested residual is not reached before the solution vanishes; report the zero solution and the χ² it does reach
        x_final = NNLS.solve!(lasso_work, μmax)
        return (; x = x_final, mu = μmax, chi2 = chi2_ratio(resnorm_sq(lasso_work), res²_min))
    end

    mu_final = lasso_target_regparam!(lasso_work, res²_target; atol = eps(T) * b²)
    return (; x = solution(lasso_work), mu = mu_final, chi2 = chi2_ratio(resnorm_sq(lasso_work), res²_min))
end

# Root of the monotone residual ‖Ax_μ - b‖² = res²_target on [0, μmax], given ‖Ax₀ - b‖² < res²_target < ‖b‖².
# On one support the residual is affine in μ², ‖Ax_ν - b‖² = ‖Ax_μ - b‖² + q(ν² - μ²)/4, so the model root
#
#   ν = √(μ² + 4(res²_target - ‖Ax_μ - b‖²)/q)
#
# is the root itself whenever `regparam_segment!` places it at or before the next support change, and is returned there without reference to a tolerance.
# Past that endpoint ν is only a Newton step in μ², bracketed by the monotonicity of the residual: each solve replaces the end of the bracket it falls on, and a step leaving the bracket is replaced by a geometric bisection.
# The bracket closes onto a representable interior point every iteration, and the loop ends when there is none left; the root then lies between two adjacent parameters and the lower is returned.
function lasso_target_regparam!(lasso_work::NNLS.NNLSLassoWorkspace{T}, res²_target::T; ftol::T = √eps(T), atol::T = zero(T)) where {T}
    lo, hi, μ = zero(T), NNLS.regparam_max(lasso_work), zero(T)

    while true
        NNLS.solve!(lasso_work, μ)
        res² = resnorm_sq(lasso_work)

        ν = T(NaN)
        if res² < res²_target
            lo = μ
            q, μ_end = NNLS.regparam_segment!(lasso_work, μ)
            q <= 0 && return error("The ℓ¹ path direction vanished on a nonempty support, where q = ‖R⁻ᵀ𝟙‖² is positive.")
            ν = √(μ^2 + 4 * (res²_target - res²) / q)
            ν <= μ_end && return (NNLS.solve!(lasso_work, ν); ν) # the segment model holds to μ_end, so ν is the root and not an iterate
        else
            hi = μ
        end

        abs(res² - res²_target) <= ftol * res²_target + atol && return μ
        ν = lo < ν < hi ? ν : lo > 0 ? √(lo * hi) : hi / 2
        lo < ν < hi || return (NNLS.solve!(lasso_work, lo); lo)
        μ = ν
    end
end

####
#### Morozov discrepancy principle (MDP) method for choosing the ℓ¹ regularization parameter
####

struct NNLSMDPLassoRegProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W1, W2, S}
    A::TA # decay basis matrix
    b::Tb # decay curve data
    m::Int # number of rows of A
    n::Int # number of columns of A
    nnls_prob::W1 # unregularized NNLS problem, i.e. μ = 0
    lasso_work::W2 # ℓ¹-regularized solver workspace, warm-started across the μ-search
    nnls_prob_seed::S # source for the unregularized solve; see `NNLSUnregSource`
end
function NNLSMDPLassoRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}, nnls_prob_seed::NNLSUnregSource{T} = nothing) where {T}
    m, n = size(A)
    return NNLSMDPLassoRegProblem(A, b, m, n, NNLSProblem(A, b), NNLS.NNLSLassoWorkspace(A, b), nnls_prob_seed)
end

@inline solution(work::NNLSMDPLassoRegProblem) = solution(work.lasso_work)
@inline ncomponents(work::NNLSMDPLassoRegProblem) = ncomponents(work.lasso_work)

@doc raw"""
    lsqnonneg_mdp_lasso(A::AbstractMatrix, b::AbstractVector, δ::Real)

Compute the ``\ell^1``-regularized nonnegative least-squares (NNLS) solution ``x_{\mu}`` of the problem:

```math
x_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu ||x||_1
```

where ``\mu`` is chosen using Morozov's Discrepancy Principle (MDP)[1,2]:

```math
\mu = \operatorname{sup}\; \left\{ \nu \ge 0 : ||Ax_{\nu} - b|| \le \delta \right\}.
```

This is the ``\ell^1`` counterpart of [`lsqnonneg_mdp`](@ref).

# Arguments

  - `A::AbstractMatrix`: Decay basis matrix
  - `b::AbstractVector`: Decay curve data
  - `δ::Real`: Upper bound on regularized residual norm

# Outputs

  - `x::AbstractVector`: Regularized NNLS solution
  - `mu::Real`: Resulting regularization parameter ``\mu``
  - `chi2::Real`: Resulting ratio ``||Ax_{\mu} - b||_2^2 / ||Ax_0 - b||_2^2`` of squared residual norms

# References

  1. Morozov VA. Methods for Solving Incorrectly Posed Problems. Springer Science & Business Media, 2012.
  2. Clason C, Kaltenbacher B, Resmerita E. Regularization of Ill-Posed Problems with Non-negative Solutions. In: Bauschke HH, Burachik RS, Luke DR (eds) Splitting Algorithms, Modern Operator Theory, and Applications. Cham: Springer International Publishing, pp. 113–135.
"""
function lsqnonneg_mdp_lasso(A::AbstractMatrix, b::AbstractVector, δ::Real)
    work = lsqnonneg_mdp_lasso_work(A, b)
    return lsqnonneg_mdp_lasso!(work, δ)
end
lsqnonneg_mdp_lasso_work(A::AbstractMatrix, b::AbstractVector, nnls_prob_seed = nothing) = NNLSMDPLassoRegProblem(A, b, nnls_prob_seed)

function lsqnonneg_mdp_lasso!(work::NNLSMDPLassoRegProblem{T}, δ::Real) where {T}
    (; lasso_work) = work
    @assert δ > 0 "Residual norm δ must be a positive value, but got δ = $δ"
    δ² = T(δ)^2

    (; μmax, R₀, b²) = lasso_baseline!(work)
    res²_min = R₀

    if b² == 0
        # No data to fit: x = 0 is the unique minimizer for every μ > 0 and an optimal one at μ = 0, and its residual is already zero
        return (; x = solution(lasso_work), mu = zero(T), chi2 = one(T))
    end

    if δ² <= res²_min
        # No μ ≥ 0 attains a residual this small, ‖Ax_μ - b‖² being nondecreasing
        return (; x = solution(lasso_work), mu = zero(T), chi2 = one(T))
    end

    if δ² >= b²
        # Every μ ≥ μmax satisfies the discrepancy with x = 0 and residual ‖b‖², so the admissible set is the ray [μmax, ∞) and its supremum is infinite; report μmax, since in general we search only over [0, μmax].
        x_final = NNLS.solve!(lasso_work, μmax)
        return (; x = x_final, mu = μmax, chi2 = chi2_ratio(resnorm_sq(lasso_work), res²_min))
    end

    mu_final = lasso_target_regparam!(lasso_work, δ²; atol = eps(T) * b²)
    return (; x = solution(lasso_work), mu = mu_final, chi2 = chi2_ratio(resnorm_sq(lasso_work), res²_min))
end

####
#### L-curve method for choosing the ℓ¹ regularization parameter
####

struct NNLSLCurveLassoRegProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W1, W2, S}
    A::TA # decay basis matrix
    b::Tb # decay curve data
    m::Int # number of rows of A
    n::Int # number of columns of A
    nnls_prob::W1 # unregularized NNLS problem, i.e. μ = 0
    lasso_work::W2 # ℓ¹-regularized solver workspace, warm-started across the μ-search
    nnls_prob_seed::S # source for the unregularized solve; see `NNLSUnregSource`
end
function NNLSLCurveLassoRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}, nnls_prob_seed::NNLSUnregSource{T} = nothing) where {T}
    m, n = size(A)
    return NNLSLCurveLassoRegProblem(A, b, m, n, NNLSProblem(A, b), NNLS.NNLSLassoWorkspace(A, b), nnls_prob_seed)
end

@inline solution(work::NNLSLCurveLassoRegProblem) = solution(work.lasso_work)
@inline ncomponents(work::NNLSLCurveLassoRegProblem) = ncomponents(work.lasso_work)

@doc raw"""
    lsqnonneg_lcurve_lasso(A::AbstractMatrix, b::AbstractVector; max_slope = $(LCURVE_SLOPE_MAX_DEFAULT))

Compute the ``\ell^1``-regularized nonnegative least-squares (NNLS) solution ``x_{\mu}`` of the problem:

```math
x_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu ||x||_1
```

where ``\mu`` is chosen at a corner of the ``\ell^1`` "L-curve"[1] ``\mu \mapsto (\log||Ax_\mu - b||_2^2, 2\log||x_\mu||_1)``, the first positive local maximum of its turning rate ``\omega = d\theta/d\log\mu`` that `max_slope` admits.

As for [`lsqnonneg_lcurve`](@ref), `max_slope` excludes corners in the near-vertical ``\mu \to 0`` tail. If no corner is found, ``\mu = 0`` and the unregularized solution is returned.

# Arguments

  - `A::AbstractMatrix`: Decay basis matrix
  - `b::AbstractVector`: Decay curve data
  - `max_slope::Real = $(LCURVE_SLOPE_MAX_DEFAULT)`: reject corners at which ``2 ||Ax_\mu - b||_2^2 / (\mu ||x_\mu||_1)`` exceeds `max_slope`. Pass `Inf` to accept any corner.

# Outputs

  - `x::AbstractVector`: Regularized NNLS solution
  - `mu::Real`: Resulting regularization parameter ``\mu``
  - `chi2::Real`: Resulting ratio ``||Ax_{\mu} - b||_2^2 / ||Ax_0 - b||_2^2`` of squared residual norms

# References

  1. J. Nasehi Tehrani, A. McEwan, C. Jin and A. van Schaik, "L1 regularization method in electrical impedance tomography by using the L1-curve (Pareto frontier curve)". Applied Mathematical Modelling, 36(3), 1095-1105, 2012, https://doi.org/10.1016/j.apm.2011.07.055.
"""
function lsqnonneg_lcurve_lasso(A::AbstractMatrix, b::AbstractVector; kwargs...)
    work = lsqnonneg_lcurve_lasso_work(A, b)
    return lsqnonneg_lcurve_lasso!(work; kwargs...)
end
lsqnonneg_lcurve_lasso_work(A::AbstractMatrix, b::AbstractVector, nnls_prob_seed = nothing) = NNLSLCurveLassoRegProblem(A, b, nnls_prob_seed)

function lsqnonneg_lcurve_lasso!(work::NNLSLCurveLassoRegProblem{T}; max_slope = LCURVE_SLOPE_MAX_DEFAULT, maxiters::Int = 64 * work.n + 64) where {T}
    (; lasso_work) = work
    @assert max_slope > 0 "max_slope must be positive; got $max_slope"

    (; μmax, R₀, N₀, b²) = lasso_baseline!(work)
    res²_min = R₀

    # A zero solution leaves no finite logarithmic curve
    (R₀ <= eps(T) * b² || N₀ <= 0) && return (; x = solution(lasso_work), mu = zero(T), chi2 = one(T))

    mu_final = lcurve_lasso_march!(lasso_work, μmax, T(max_slope), maxiters)

    # No admissible corner below μmax; return the unregularized solution
    mu_final == 0 && return (; x = NNLS.solve!(lasso_work, zero(T)), mu = zero(T), chi2 = one(T))

    x_final = NNLS.solve!(lasso_work, mu_final)
    return (; x = x_final, mu = mu_final, chi2 = chi2_ratio(resnorm_sq(lasso_work), res²_min))
end

# Corner of the ℓ¹ L-curve, found by walking the support segments upward from μ = 0, where `lasso_baseline!` has left the workspace solved.
# One solve per segment fixes ‖Ax-b‖², ‖x‖₁ and q there, making the geometry of `lcurve_geometry_lasso` closed-form across it.
# A stationary point of ω interior to the segment is a turning maximum of the path; one past the far knot leaves ω still rising, which `rising` carries into the next segment.
# One short of the near knot leaves ω falling across the whole segment, so that knot is a turning maximum only if ω rose into it and the curvature there is still positive.
# The maxima are located before the slope guard is applied, so `max_slope` rejects a corner near the vertical tail rather than moving it, as in `lcurve_corner`.
# Returns the selected μ, or zero when no segment below μmax carries an admissible maximum.
function lcurve_lasso_march!(lasso_work::NNLS.NNLSLassoWorkspace{T}, μmax::T, τ::T, maxiters::Int) where {T}
    μ, rising, nudge = zero(T), false, eps(T)^(3//4)

    for _ in 1:maxiters
        R, N₁ = resnorm_sq(lasso_work), seminorm(lasso_work)
        N₁ <= 0 && return zero(T)

        q, μ_end = NNLS.regparam_segment!(lasso_work, μ)
        q <= 0 && return error("The ℓ¹ path direction vanished on a nonempty support, where q = ‖R⁻ᵀ𝟙‖² is positive.")

        ν★ = lcurve_lasso_segment_turn(q, μ, R, N₁)
        if ν★ >= μ_end
            rising = true # ω is still rising where the segment ends
        else
            ν = ν★ > μ ? ν★ : μ
            turns = ν★ > μ || (rising && lcurve_geometry_lasso(R, N₁, q, μ)[1] > 0)
            turns && lcurve_lasso_segment_slope(q, μ, R, N₁, ν) <= τ && return ν
            rising = false
        end

        μ = max(μ_end * (1 + nudge), μ + nudge * μmax)
        μ >= μmax && return zero(T)
        NNLS.solve!(lasso_work, μ)
    end

    return error("The ℓ¹ L-curve was not walked to the end of the path within $maxiters support segments.")
end

# Stationary point of the turning rate on the segment containing the solution at μ.
# Writing R = ρ + qν²/4 and L = ℓ - qν/2 on the segment, the turning rate of `lcurve_geometry_lasso` is
#
#   ω(ν) = ν(2ρℓ - 2ρqν - qℓν²/2) / (q²ν⁴/2 - qℓν³ + (ℓ² + 2ρq)ν² + 4ρ²).
#
# ω(ν) is positive between its zeros at ν = 0 and at the positive root νᵤ of the quadratic factor in the numerator.
# The numerator N'D - ND' of ω'(ν) is therefore 8ρ³ℓ > 0 at ν = 0 and -(2ρqνᵤ + qℓνᵤ²)·D(νᵤ) < 0 at ν = νᵤ, bracketing its root.
# This root can be shown to be the unique maximum: in the variables z = qν/2ℓ and h = qρ/ℓ² the numerator is
#
#   P = 2z⁶ + 8hz⁵ - (12h+1)z⁴ + 4hz³ - (5h²+h)z² - 4h³z + h³.
#
# Its z-derivative -4h³ - 10zh² + 2zh(20z³-24z²+6z-1) + 4z³(3z²-1) is negative on 0 < z ≤ 1/4, where P runs from h³ > 0 to -(640h² + 80h + 7)/2048 < 0.
# On 1/4 ≤ z < 1/2, substituting h = (u+z²)/(1-2z) with u = h(1-2z) - z² > 0,
#
#   (1-2z)³P = (1-4z)u³ - 2z²(z+1)u² + z²(z-1)(32z⁴-48z³+32z²-7z+1)u + 2z⁴(z-1)³(8z²-4z+1)
#
# is a sum of nonpositive terms with the last strictly negative, and z ≥ 1/2 lies outside the positive-curvature branch entirely.
# Returns zero when ρ ≤ 0, where the positive-turning branch is empty.
function lcurve_lasso_segment_turn(q::T, μ::T, R::T, N₁::T) where {T}
    ρ, ℓ = R - q * μ^2 / 4, N₁ + q * μ / 2
    ρ <= 0 && return zero(T)

    # Derivative numerator N'D - ND', equal to (8ℓ⁷/q³)·P and thus has the same sign as P
    ρ², ℓ², ρq, ρℓ, qℓ = ρ^2, ℓ^2, ρ * q, ρ * ℓ, q * ℓ
    c₀, c₁, c₂, c₃ = 8 * ρ² * ρℓ, -16 * ρ² * ρq, -2 * ρℓ * (ℓ² + 5 * ρq), 4 * ρq * ℓ²
    c₄, c₅, c₆ = -(qℓ / 2) * (12 * ρq + ℓ²), 2 * ρ * q^3, q^3 * ℓ / 4
    dturn(ν) = (ν² = ν^2; muladd(ν²^2, muladd(c₆, ν², muladd(c₅, ν, c₄)), muladd(ν², muladd(c₃, ν, c₂), muladd(c₁, ν, c₀))))

    lo, hi = zero(T), 2 * ρℓ / (ρq + √(ρq * (ρq + ℓ²)))
    while lo < (mid = (lo + hi) / 2) < hi
        dturn(mid) > 0 ? (lo = mid) : (hi = mid)
    end

    return (lo + hi) / 2
end

# Tangent slope magnitude 2‖Ax_ν - b‖²/(ν‖x_ν‖₁) at ν on the segment solved at μ
@inline function lcurve_lasso_segment_slope(q::T, μ::T, R::T, N₁::T, ν::T) where {T}
    Rν, Nν = R + q * (ν^2 - μ^2) / 4, N₁ - q * (ν - μ) / 2
    return Nν <= 0 ? T(Inf) : 2 * Rν / (ν * Nν)
end

# Curvature κ and turning rate ω of the ℓ¹ log-log L-curve P(t) = (log R, 2log N₁) at t = logμ, from R = ‖Ax-b‖², N₁ = ‖x‖₁, q = 𝟙ᵀG_PP⁻¹𝟙.
# The curve is piecewise C²: R, N₁ and μ are continuous across a support change but q is not, so Ṗ = q(μ²/2R, -μ/N₁) jumps by in magnitude by a factor q₊/q₋ but its direction is unchanged.
# The curve is therefore tangent-continuous across a knot, while κ and ω are two-valued, and both strictly decreasing in q.
# On a fixed support, dR/dμ = μq/2 and dN₁/dμ = -q/2, so with a = Ṗ₁ = μ²q/(2R) and c = Ṗ₂ = -μq/N₁,
#   P̈₁ = 2a - a²,    P̈₂ = c - c²/2,    ω = (Ṗ₁P̈₂ - Ṗ₂P̈₁)/(a² + c²) = ac(a - c/2 - 1)/(a² + c²),    κ = ω/√(a² + c²).
# As μ → 0, κ tends to the positive plateau N₁²/(2qR) while ω vanishes, as for the Tikhonov curve; see `lcurve_geometry`.
# On a segment, R = ρ + qν²/4 and N₁ = ℓ - qν/2 with ρ ≥ 0 and ℓ > 0, so in the variables z = qν/2ℓ ∈ (0,1) and h = qρ/ℓ², positive curvature <=> h(1-2z) > z².
function lcurve_geometry_lasso(R::T, N₁::T, q::T, μ::T) where {T}
    a, c = μ^2 * q / (2 * R), -μ * q / N₁
    n² = a^2 + c^2
    ω = a * c * (a - c / 2 - 1) / n²
    return (ω / √n², ω)
end

####
#### Reginska (minimum-product) method for choosing the ℓ¹ regularization parameter
####

struct NNLSReginskaLassoRegProblem{T, TA <: AbstractMatrix{T}, Tb <: AbstractVector{T}, W1, W2, S}
    A::TA # decay basis matrix
    b::Tb # decay curve data
    m::Int # number of rows of A
    n::Int # number of columns of A
    nnls_prob::W1 # unregularized NNLS problem, i.e. μ = 0
    lasso_work::W2 # ℓ¹-regularized solver workspace, warm-started across the μ-search
    nnls_prob_seed::S # source for the unregularized solve; see `NNLSUnregSource`
end
function NNLSReginskaLassoRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}, nnls_prob_seed::NNLSUnregSource{T} = nothing) where {T}
    m, n = size(A)
    return NNLSReginskaLassoRegProblem(A, b, m, n, NNLSProblem(A, b), NNLS.NNLSLassoWorkspace(A, b), nnls_prob_seed)
end

@inline solution(work::NNLSReginskaLassoRegProblem) = solution(work.lasso_work)
@inline ncomponents(work::NNLSReginskaLassoRegProblem) = ncomponents(work.lasso_work)

@doc raw"""
    lsqnonneg_reginska_lasso(A::AbstractMatrix, b::AbstractVector)

Compute the ``\ell^1``-regularized nonnegative least-squares (NNLS) solution ``x_{\mu}`` of the problem:

```math
x_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; ||Ax - b||_2^2 + \mu ||x||_1
```

where ``\mu`` is the smallest positive local minimizer of an ``\ell^1`` analogue of Regińska's minimum-product criterion[1]:

```math
\Psi(\nu) = ||Ax_{\nu} - b||_2^2 \, ||x_{\nu}||_1.
```

This is the ``\ell^1`` counterpart of [`lsqnonneg_reginska`](@ref).
Stationarity of ``\Psi`` is equivalent to a log-log tangent slope of ``-1`` for the pair ``(||Ax_{\nu} - b||_2^2, ||x_{\nu}||_1)``, so the selected ``\mu`` is the balance point ``||Ax_{\mu} - b||_2^2 = \mu ||x_{\mu}||_1`` at which the two terms of the objective contribute equally.

# Arguments

  - `A::AbstractMatrix`: Decay basis matrix
  - `b::AbstractVector`: Decay curve data

# Outputs

  - `x::AbstractVector`: Regularized NNLS solution
  - `mu::Real`: Resulting regularization parameter ``\mu``
  - `chi2::Real`: Resulting ratio ``||Ax_{\mu} - b||_2^2 / ||Ax_0 - b||_2^2`` of squared residual norms

# References

  1. T. Regińska, "A Regularization Parameter in Discrete Ill-Posed Problems". SIAM Journal on Scientific Computing, 17(3), 740-749, 1996, https://doi.org/10.1137/S1064827593252672.
"""
function lsqnonneg_reginska_lasso(A::AbstractMatrix, b::AbstractVector; kwargs...)
    work = lsqnonneg_reginska_lasso_work(A, b)
    return lsqnonneg_reginska_lasso!(work; kwargs...)
end
lsqnonneg_reginska_lasso_work(A::AbstractMatrix, b::AbstractVector, nnls_prob_seed = nothing) = NNLSReginskaLassoRegProblem(A, b, nnls_prob_seed)

function lsqnonneg_reginska_lasso!(work::NNLSReginskaLassoRegProblem{T}; maxiters::Int = 8 * work.n + 16) where {T}
    (; lasso_work) = work

    (; μmax, R₀, N₀, b²) = lasso_baseline!(work) # ‖x_μ‖₁ = 0 from μmax onwards, where Ψ collapses to zero
    res²_min, seminrm_min = R₀, N₀

    # An exact unregularized fit makes the minimum-product criterion zero at μ = 0
    if res²_min <= eps(T) * b² || ncomponents(lasso_work) == 0
        return (; x = solution(lasso_work), mu = zero(T), chi2 = one(T))
    end

    # Ψ = res²·‖x‖₁ has dΨ/dμ = (d‖x‖₁/dμ)·φ(μ) with φ(μ) = res² - μ‖x‖₁ and d‖x‖₁/dμ ≤ 0, so the smallest local minimizer of Ψ is the leftmost downward crossing of φ, the balance point res² = μ‖x‖₁.
    # res² is nondecreasing and ‖x‖₁ nonincreasing, so φ(ν) ≥ res²(μ) - ν‖x_μ‖₁ for ν > μ, and one solve certifies φ > 0 up to Φ(μ) = res²/‖x‖₁.
    # The same monotonicities make Φ nondecreasing with the balance points as its fixed points, so every balance point satisfies μ ≥ Φ(0) and the search starts there.
    # Complementarity gives bᵀr = res² + μ‖x‖₁/2, which at a balance point is 3·res²/2, so Cauchy-Schwarz bounds res² ≤ 4‖b‖²/9 there and the first iterate above that bound proves none remains.
    μ = res²_min / seminrm_min
    res²_max = 4 * b² / 9
    μ >= μmax && return (; x = solution(lasso_work), mu = zero(T), chi2 = one(T)) # no balance point exists

    # Iterating Φ alone never brackets: φ > 0 means Φ(μ) > μ, and applying the nondecreasing Φ leaves φ ≥ 0 at every iterate.
    # What resolves this is that φ is piecewise quadratic with positive leading coefficient, where `NNLS.regparam_segment!` supplies both the quadratic and the interval of μ on which it holds.
    # The balance point is then a closed-form root as soon as an iterate lands in its interval, and no crossing pair inside one interval can be stepped over.
    mu_final, nudge = zero(T), √eps(T)
    μ_floor = μ * (1 - nudge)
    for _ in 1:maxiters
        NNLS.solve!(lasso_work, μ)
        res², seminrm = resnorm_sq(lasso_work), seminorm(lasso_work)
        q, μ_end = NNLS.regparam_segment!(lasso_work, μ)

        s = reginska_lasso_balance_root(res², seminrm, μ, q)
        if μ_floor - μ <= s <= μ_end - μ
            mu_final = μ + s
            break
        end

        μ_next = min(max(μ_end, res² / seminrm), μmax)
        (res² > res²_max || μ_next >= μmax) && return (; x = NNLS.solve!(lasso_work, zero(T)), mu = zero(T), chi2 = one(T)) # no balance point exists

        if μ_next > μ
            nudge = √eps(T)
            μ_floor, μ = μ_next * (1 - nudge), μ_next * (1 + nudge)
        else
            nudge = 2 * nudge
            μ = μ * (1 + nudge)
        end
    end
    mu_final == 0 && return error("Reginska's criterion was not resolved within $maxiters Lasso solves.")

    # The last solve need not be the one at the selected μ, so the selected solution is recomputed.
    # Two support events inside one nudge window leave a third segment between the endpoint and the solved point, across which the accepted quadratic was extrapolated.
    # The balance is re-tested on the support present at the selected μ and corrected by the quadratic there, and the correction is re-solved and re-tested in turn.
    x_final = NNLS.solve!(lasso_work, mu_final)
    balanced = false
    for _ in 1:4
        res², seminrm = resnorm_sq(lasso_work), seminorm(lasso_work)
        balanced = abs(res² - mu_final * seminrm) <= eps(T)^(3//4) * max(res², mu_final * seminrm)
        balanced && break
        q, _ = NNLS.regparam_segment!(lasso_work, mu_final)
        s = reginska_lasso_balance_root(res², seminrm, mu_final, q)
        isnan(s) && break
        mu_final += s
        x_final = NNLS.solve!(lasso_work, mu_final)
    end

    # The balance is what the criterion asserts, so the returned point is held to it rather than to the iteration having run
    if !(balanced || abs(resnorm_sq(lasso_work) - mu_final * seminorm(lasso_work)) <= eps(T)^(3//4) * max(resnorm_sq(lasso_work), mu_final * seminorm(lasso_work)))
        error("Reginska's balance ‖Ax-b‖² = μ‖x‖₁ was not attained at the selected μ = $mu_final.")
    end

    return (; x = x_final, mu = mu_final, chi2 = chi2_ratio(resnorm_sq(lasso_work), res²_min))
end

# Smallest root in s of the balance polynomial φ(μ + s) = (3q/4)s² + (qμ - ‖x‖₁)s + (res² - μ‖x‖₁), or `NaN` when it has none.
# Each branch is the form of the quadratic formula free of cancellation for the sign of the linear coefficient.
function reginska_lasso_balance_root(res²::T, seminrm::T, μ::T, q::T) where {T}
    α, β, γ = 3 * q / 4, q * μ - seminrm, res² - μ * seminrm
    disc = β^2 - 4 * α * γ
    disc < 0 && return T(NaN)
    return β < 0 ? 2 * γ / (√disc - β) : -(β + √disc) / (2 * α)
end
