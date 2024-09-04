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
solve!(work::NNLSProblem, args...; kwargs...) = solve!(work, work.A, work.b, args...; kwargs...)
# solve!(work::NNLSProblem, A::AbstractMatrix, b::AbstractVector, args...; kwargs...) = NNLS.nnls!(work.nnls_work, A, b, args...; kwargs...)

# The nnls algorithm selects candidate x[j] based on the largest negative gradient
# of ||Ax - b||, i.e. j = argmax_j w[j] where w = -A'(Ax - b) is the dual vector.
# In DECAES, the initial dual vector w_0 = A'b is sorted because A[i, j], b[j] >= 0
# and A[i, j+1] > A[i, j], and thus the last column of A will always be chosen first.
# Thence j = n and we can bypass the first iteration and initialize the gradient with
# x_0 = [0; x[n]], where x[n] >= 0 due to the nonnegativity of A, b.
#   NOTE: This will not fail even for generic A and b, it just forces NNLS to start
#         with column j = n. From there, it may remove the column if necessary.
function solve!(
    work::NNLSProblem{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T};
    kwargs...,
) where {T}
    m, n = size(A)
    C = work.nnls_work.A
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

    xj = zero(T)
    @inbounds @simd for i in 1:m
        xj += (A[i, n] / den) * b[i]
    end

    # w = -A'*(Ax - b)
    @inbounds @simd for i in 1:m
        z[i] = b[i] - A[i, n] * xj
    end

    @inbounds for j in 1:n-1
        wj = zero(T)
        @simd for i in 1:m
            Aij = A[i, j]
            wj += Aij * z[i]
            C[i, j] = Aij # initialize nnls workspace
        end
        w[j] = wj
    end
    @inbounds w[end] = 0
    @inbounds w[end] = all(<=(0), w)

    # Initialize nnls workspace
    @inbounds for i in 1:m
        f[i] = b[i]
        C[i, n] = A[i, n]
    end

    @inbounds for j in 1:n
        x[j] = 0
        idx[j] = j
    end

    return NNLS.unsafe_nnls!(work.nnls_work; kwargs..., init_dual = false)
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

    C = work.nnls_work.A
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
            Aij = A0[i, j]
            wj += Aij * z[i]
            C[i, j] = Aij # initialize nnls workspace
        end
        w[j] = wj
    end
    @inbounds w[end] = 0
    @inbounds w[end] = all(<=(0), w)

    # Initialize nnls workspace
    @inbounds for i in 1:m
        f[i] = b0[i]
        C[i, n] = A0[i, n]
    end

    @inbounds for j in 1:n
        for i in m+1:size(C, 1)
            C[i, j] = 0
        end
    end

    @inbounds for j in 1:n
        x[j] = 0
        f[m+j] = 0
        idx[j] = j
        diag[j] = false
    end

    return NNLS.unsafe_nnls!(work.nnls_work, μ; kwargs..., init_dual = false)
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
    b::Tb
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
    A::TA
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

#=
struct LSTikhonovRegProblem{
    T,
    TA <: AbstractMatrix{T},
    Tb <: AbstractVector{T},
    W,
}
    A::TA
    b::Tb
    m::Int
    n::Int
    ls_work::W
end
function LSTikhonovRegProblem(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    ls_work = NNLS.LSWorkspace__v5(A, b)
    return LSTikhonovRegProblem(A, b, m, n, ls_work)
end
=#

struct NNLSTikhonovRegProblem{
    T,
    TA <: AbstractMatrix{T},
    Tb <: AbstractVector{T},
    W <: NNLSProblem{T, <:TikhonovPaddedMatrix{T}, <:PaddedVector{T}},
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

function solve!(work::NNLSTikhonovRegProblem, μ::Real; kwargs...)
    # Set regularization parameter and solve NNLS problem
    regparam!(work, μ)
    solve!(work.nnls_prob, μ; kwargs...)
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
# Useful for optimization problems where the last function call may not be
# the optimium, but perhaps it was one or two calls previous and is still in the
# `NNLSTikhonovRegProblemCache` and a recomputation can be avoided.
#=
struct NNLSTikhonovRegProblemCache{N, W0, W <: NTuple{N}}
    ls_prob::W0
    cache::W
    idx::Base.RefValue{Int}
end
function NNLSTikhonovRegProblemCache(A::AbstractMatrix{T}, b::AbstractVector{T}, ::Val{N} = Val(8)) where {T, N}
    ls_prob = LSTikhonovRegProblem(A, b)
    cache = ntuple(_ -> NNLSTikhonovRegProblem(A, b), N)
    idx = Ref(1)
    return NNLSTikhonovRegProblemCache(ls_prob, cache, idx)
end
=#
struct NNLSTikhonovRegProblemCache{N, W <: NTuple{N}}
    cache::W
    idx::Base.RefValue{Int}
end
function NNLSTikhonovRegProblemCache(A::AbstractMatrix{T}, b::AbstractVector{T}, ::Val{N} = Val(8)) where {T, N}
    cache = ntuple(_ -> NNLSTikhonovRegProblem(A, b), N)
    idx = Ref(1)
    return NNLSTikhonovRegProblemCache(cache, idx)
end
reset_cache!(work::NNLSTikhonovRegProblemCache{N}) where {N} = foreach(w -> regparam!(w, NaN), work.cache)
Base.getindex(work::NNLSTikhonovRegProblemCache) = work.cache[get_cache_index(work)]

function next_cache_index!(work::NNLSTikhonovRegProblemCache{N}) where {N}
    for (i, w) in enumerate(work.cache)
        if isnan(regparam(w))
            set_cache_index!(work, i)
            return work.idx[]
        end
    end
    set_cache_index!(work, work.idx[] + 1)
    return work.idx[]
end
@inline get_cache_index(work::NNLSTikhonovRegProblemCache) = work.idx[]
@inline set_cache_index!(work::NNLSTikhonovRegProblemCache{N}, i) where {N} = (work.idx[] = mod1(i, N))

function solve!(work::NNLSTikhonovRegProblemCache, μ::Real)
    # Find index of cached workspace with μi nearest to μ
    @assert μ > 0 "Regularization parameter μ must be positive, got μ = $μ"
    T = typeof(regparam(work[]))
    μ = T(μ)

    emptycache = true
    imax, Δlogμmax = 0, T(Inf)
    for (i, μi) in enumerate(regparam.(work.cache))
        if !isnan(μi)
            emptycache = false
            imax, Δlogμ = i, μ == μi ? zero(T) : T(abs(log1p((μ - μi) / μi)))
            Δlogμ < Δlogμmax && ((imax, Δlogμmax) = (i, Δlogμ))
            Δlogμmax == 0 && break
        end
    end

    if emptycache || Δlogμmax > 0
        # No cached solves; solve from scratch
        next_cache_index!(work)
        solve!(work[], μ)
    else
        # Exact match; return cached solution
        set_cache_index!(work, imax)
    end

    return solution(work[])
    #=
    # Δlogμmax_thresh = zero(T)
    Δlogμmax_thresh = T(Inf) #T(Inf) #T(1e-6)
    if emptycache || Δlogμmax > Δlogμmax_thresh
        # @info "No cached solves; solve from scratch"
        next_cache_index!(work)
        solve!(work[], μ)
    elseif Δlogμmax == zero(T)
        # Exact match; return cached solution
        set_cache_index!(work, imax)
    else # Δlogμmax <= Δlogμmax_thresh
        # println(); @info "Nearby match; warm start with nearest cached solution"
        set_cache_index!(work, imax)
        (; A, b, nnls_prob) = work[]
        (; nnls_work) = nnls_prob
        (; ls_work) = work.ls_prob
        reload = nnls_work.nsetp[] != ls_work.nsetp[] || nnls_work.idx != ls_work.idx
        # NNLS.showall(; reload, μ, idx_nnls = nnls_work.idx', idx_ls = ls_work.idx')
        NNLS.ls!(ls_work, A, b, μ, nnls_work.idx, nnls_work.nsetp[]; reload)
        if NNLS.check_kkt!(ls_work, A, b, μ)
            # @info "Warm start successful"
            regparam!(work[], μ)
            copyto!(nnls_work.x, ls_work.x)
            nnls_work.rnorm[] = ls_work.rnorm[]
            # @show lsqnonneg_tikh(A, b, μ)'
            # @show solution(work[])'
            @show norm(lsqnonneg_tikh(A, b, μ)' - solution(work[])')
            # NNLS.showall(; x = ls_work.x', w = ls_work.w', rnorm = resnorm(work[]), rnorm′ = norm(A * solution(work[]) - b), Δrnorm = resnorm(work[]) - norm(A * solution(work[]) - b), idx = nnls_work.idx', nsetp = nnls_work.nsetp[])
            #=
            let
                # @info "Comparing with NNLS soln"
                next_cache_index!(work)
                solve!(work[], μ)
                # NNLS.showall(; x = solution(work[])', rnorm = resnorm(work[]), rnorm′ = norm(A * solution(work[]) - b), Δrnorm = resnorm(work[]) - norm(A * solution(work[]) - b), idx = nnls_work.idx', nsetp = nnls_work.nsetp[])
            end
            =#
        else
            # @info "Warm start failed; solve from scratch"
            next_cache_index!(work)
            solve!(work[], μ)
        end
    end
    =#

    #=
    if emptycache
        # No cached solves; solve from scratch
        next_cache_index!(work)
        solve!(work[], μ)
    elseif Δlogμmax == 0
        # Exact match; return cached solution
        set_cache_index!(work, imax)
    else # Δlogμmax > 0
        # Warm start with nearest cached solution
        set_cache_index!(work, imax)
        solve!(work[], μ; warm_start = true)
    end
    =#
end

#=
function test_NNLSTikhonovRegProblemCache(m = 8, n = 6, μ = 0.01, ::Val{N} = Val(5)) where {N}
    A, b = rand(m, n), rand(m)
    work0 = NNLSTikhonovRegProblem(A, b)
    work = NNLSTikhonovRegProblemCache(A, b, Val(N))
    @assert isapprox(solve!(work0, μ), solve!(work, μ); rtol = 1e-14, atol = 1e-14)
    @assert isapprox(solve!(work0, μ), solve!(work, μ); rtol = 1e-14, atol = 1e-14)
    @assert isapprox(solve!(work0, μ), solve!(work, μ); rtol = 1e-14, atol = 1e-14)
    for _ in 1:1, scale in exp10.(randn(100))
        # @assert isapprox(solve!(work0, scale * μ), solve!(work, scale * μ); rtol = 1e-14, atol = 1e-14)
        @assert isapprox(lsqnonneg_tikh(A, b, scale * μ), solve!(work, scale * μ); rtol = 1e-14, atol = 1e-14)
    end
    solve!(work, μ * exp(randn())) #(1 + randn() * 0.1))

    return nothing
end
=#

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
lsqnonneg_chi2_work(A::AbstractMatrix, b::AbstractVector) = NNLSChi2RegProblem(A, b)

function lsqnonneg_chi2!(work::NNLSChi2RegProblem{T}, chi2_target::T, legacy::Bool = false; method::Symbol = legacy ? :legacy : :brent) where {T}
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
            return resnorm_sq(work.nnls_prob_smooth_cache[])
        end
        if mu_final == 0
            x_final = x_unreg
        else
            x_final = solve!(work.nnls_prob_smooth_cache, mu_final)
        end

    elseif method === :bisect
        f = function (logμ)
            solve!(work.nnls_prob_smooth_cache, exp(logμ))
            return chi2_relerr!(work.nnls_prob_smooth_cache[], res²_target, logμ)
        end

        # Find bracketing interval containing root, then perform bisection search with slightly higher tolerance to not waste f evals
        a, b, fa, fb = bracket_root_monotonic(f, T(-4.0), T(1.0); dilate = T(1.5), mono = +1, maxiters = 6)

        if fa * fb < 0
            # Bracketing interval found
            a, fa, c, fc, b, fb = bisect_root(f, a, b, fa, fb; xatol = T(0.0), xrtol = T(0.0), ftol = T(1e-3) * (chi2_target - 1), maxiters = 100)

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
            return chi2_relerr!(work.nnls_prob_smooth_cache[], res²_target, logμ)
        end

        # Find bracketing interval containing root
        a, b, fa, fb = bracket_root_monotonic(f, T(-4.0), T(1.0); dilate = T(1.5), mono = +1, maxiters = 6)

        if fa * fb < 0
            # Find root using Brent's method
            logmu_final, relerr_final = brent_root(f, a, b, fa, fb; xatol = T(0.0), xrtol = T(0.0), ftol = T(1e-3) * (chi2_target - 1), maxiters = 100)
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
lsqnonneg_mdp_work(A::AbstractMatrix, b::AbstractVector) = NNLSMDPRegProblem(A, b)

function lsqnonneg_mdp!(work::NNLSMDPRegProblem{T}, δ::T) where {T}
    @assert δ > 0 "Residual norm δ must be a positive value, but got δ = $δ"

    # Non-regularized solution
    solve!(work.nnls_prob)
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

    # Prepare to solve
    reset_cache!(work.nnls_prob_smooth_cache)

    function f(logμ)
        solve!(work.nnls_prob_smooth_cache, exp(logμ))
        return resnorm_sq(work.nnls_prob_smooth_cache[]) - δ^2
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
lsqnonneg_lcurve_work(A::AbstractMatrix, b::AbstractVector) = NNLSLCurveRegProblem(A, b)

function lsqnonneg_lcurve!(work::NNLSLCurveRegProblem{T}; kwargs...) where {T}
    # Compute the regularization using the L-curve method
    reset_cache!(work.nnls_prob_smooth_cache)

    # A point on the L-curve is given by (ξ(μ), η(μ)) = (log||Ax-b||^2, log||x||^2)
    #   Note: Squaring the norms is convenient for computing gradients of (ξ(μ), η(μ));
    #         this scales the L-curve, but does not change μ* = argmax C(ξ(μ), η(μ)).
    function f_lcurve(logμ)
        solve!(work.nnls_prob_smooth_cache, exp(logμ))
        ξ = log(resnorm_sq(work.nnls_prob_smooth_cache[]))
        η = log(seminorm_sq(work.nnls_prob_smooth_cache[]))
        return SA{T}[ξ, η]
    end

    # Build cached function and solve
    f_lcurve_cached = CachedFunction(f_lcurve, empty!(work.lsqnonneg_lcurve_fun_cache))
    f = LCurveCornerCachedFunction(f_lcurve_cached, empty!.(work.lcurve_corner_caches)...)

    logmu_bounds = (T(-8), T(2))
    logmu_final = lcurve_corner(f, logmu_bounds...; kwargs...)

    # Return the final regularized solution
    mu_final = exp(logmu_final)
    x_final = solve!(work.nnls_prob_smooth_cache, mu_final)
    x_unreg = solve!(work.nnls_prob)
    chi2_final = resnorm_sq(work.nnls_prob_smooth_cache[]) / resnorm_sq(work.nnls_prob)

    #=
    @info "chi2" logmu_final mu_final
    @info "ratio" chi2_final
    x_final = copy(x_final) # these are views into buffers, need to copy
    x_unreg = copy(x_unreg) # these are views into buffers, need to copy
    let
        (; A, b) = work.nnls_prob
        A, b = copy(A), copy(b)

        ∇finitediff(f, t, h = √eps(one(t))) = (f(t .+ h) .- f(t .- h)) ./ 2h
        ∇²finitediff(f, t, h = ∛eps(one(t))) = (f(t .+ h) .- 2 .* f(t) .+ f(t .- h)) ./ h^2

        ∇logfinitediff(f, logt, h = √eps(one(logt))) = ∇finitediff(f, logt, h) ./ exp(logt)
        ∇²logfinitediff(f, logt, h = ∛eps(one(logt))) = (∇²finitediff(f, logt, h) - ∇finitediff(f, logt, h)) ./ exp(2 * logt)

        function finitediff_unequal_grid(y₋, y₊, h₋, h₊)
            # Finite difference approximation of the first derivative of f(x) using unequal grid spacing
            #   f'(x) = (f(x + h₊) - f(x - h₋)) / (h₋ + h₊) = (y₊ - y₋) / (h₋ + h₊)
            return (y₊ - y₋) / (h₋ + h₊)
        end

        function f_curvature(logμ)
            solve!(work.nnls_prob_smooth_cache, exp(logμ))
            return curvature(log, work.nnls_prob_smooth_cache[])
            # return curvature(identity, work.nnls_prob_smooth_cache[])
        end

        fired = Ref(false)
        function f_quasi_opt(logμ)
            # α = μ² --> α * ||dx/dα|| = μ² * ||dx/dμ / dα/dμ|| = μ² * ||dx/dμ / 2μ|| = μ/2 * ||dx/dμ||
            μ = exp(logμ)

            solve!(work.nnls_prob_smooth_cache, μ)
            cache = work.nnls_prob_smooth_cache[]
            ξ², η² = resnorm_sq(cache), seminorm_sq(cache)
            dlogξ²_dlogη² = -μ^2 * (η² / ξ²) # dx / dy
            solve!(work.nnls_prob_smooth_cache, exp(logμ - 0.3))
            cache = work.nnls_prob_smooth_cache[]
            ξ²₋, η²₋ = resnorm_sq(cache), seminorm_sq(cache)
            dlogξ²_dlogη²₋ = -exp(2 * (logμ - 0.3)) * (η²₋ / ξ²₋)
            solve!(work.nnls_prob_smooth_cache, exp(logμ + 0.3))
            cache = work.nnls_prob_smooth_cache[]
            ξ²₊, η²₊ = resnorm_sq(cache), seminorm_sq(cache)
            dlogξ²_dlogη²₊ = -exp(2 * (logμ + 0.3)) * (η²₊ / ξ²₊)

            return dlogξ²_dlogη²
            # return log(-inv(dlogξ²_dlogη²))

            if logμ < -5 || abs(dlogξ²_dlogη²) < 1e-2 || ξ²₋ >= ξ² || ξ² >= ξ²₊ || η²₋ <= η² || η² <= η²₊
                return zero(T)
            end

            d²logξ²_d²logη² = finitediff_unequal_grid(dlogξ²_dlogη²₋, dlogξ²_dlogη²₊, log(η²) - log(η²₋), log(η²₊) - log(η²))
            # return d²logξ²_d²logη²

            if !fired[] #&& -3 < logμ < -2 #&& !isfinite(d²logξ²_d²logη²)
                fired[] = true
                # @show μ logμ ξ² η² dlogξ²_dlogη² ξ²₋ η²₊ dlogξ²_dlogη²₋ ξ²₊ η²₊ dlogξ²_dlogη²₊ d²logξ²_d²logη²
                # for logμ in -8.0:-4.0
                #     μ = exp(logμ)
                #     x = solve!(work.nnls_prob_smooth_cache, μ)
                #     cache = work.nnls_prob_smooth_cache[]
                #     @show logμ μ log(gradient_temps(cache).xᵀB⁻¹x)
                #     NNLS.choleskyfactor(cache.nnls_prob.nnls_work, Val(:U)) |> copy |> display
                # end
            end
            # !isfinite(d²logξ²_d²logη²) && (d²logξ²_d²logη² = zero(T))

            return d²logξ²_d²logη² #/ sqrt(1 + (dlogξ²_dlogη²)^2)^3 # note: κ = y''/(1+y'^2)^(3/2) = -x''/(1+x'^2)^(3/2)

            # x = solve!(work.nnls_prob_smooth_cache, μ)
            # cache = work.nnls_prob_smooth_cache[]
            # (; A, b) = cache
            # B = A'A + μ^2 * I
            # return log(gradient_temps(cache).xᵀB⁻¹x)
            # return log(x' * (B \ x))
            # return x' * (B \ x)
            # return log(x' * (B \ x)) - log(gradient_temps(cache).xᵀB⁻¹x)
            # return log(gradient_temps(cache).xᵀB⁻¹x)
            # dx_dμ = -2 * μ * (B \ x)
            # dx_dμ = -2 * μ * (B \ (B \ A'b))
            # return log(μ * norm(dx_dμ))

            # solve!(work.nnls_prob_smooth_cache, exp(logμ))
            # dxdμ_norm = solution_gradnorm(work.nnls_prob_smooth_cache[])
            # return exp(logμ) * dxdμ_norm
        end

        function f_heuristic(logμ)
            work = NNLSGCVRegProblem(A, b)
            svdvals!(work)
            return log(gcv!(work, logμ))
            # return log(gcv_and_∇gcv!(work, logμ)[1])

            # solve!(work.nnls_prob_smooth_cache, exp(logμ))
            # cache = work.nnls_prob_smooth_cache[]
            # return cache.nnls_prob.nnls_work.nsetp[]

            # if logμ > -4
            #     x = solution(cache)
            #     fig = Main.Figure()
            #     ax = Main.Axis(fig[1, 1]; ylabel = "x", title = "logμ = $logμ")
            #     Main.scatterlines!(ax, x)
            #     display(fig)
            # end

            # solve!(work.nnls_prob_smooth_cache, exp(logμ))
            # cache = work.nnls_prob_smooth_cache[]
            # return log(resnorm(cache) * seminorm(cache))
            # return log(gcv!(NNLSGCVRegProblem(A, b), logμ))
            # return gcv_dof(A, exp(logμ))
            # solve!(work.nnls_prob_smooth_cache, exp(logμ))
            # return resnorm_sq(work.nnls_prob_smooth_cache[]) / gcv_dof(A, exp(logμ))^2
            # gcv, ∇gcv = gcv_and_∇gcv!(NNLSGCVRegProblem(A, b), logμ)
            # return log(gcv)
            # return exp(logμ) * ∇gcv / gcv # d/dlogμ log(gcv) = μ d/dμ log(gcv) = μ * ∇gcv / gcv
            # return ∇gcv_dof(A, exp(logμ))
            # return ∇resnorm_sq(cache)
            # return ∇logfinitediff(logμ, 1e-3) do _logμ
            #     solve!(work.nnls_prob_smooth_cache, exp(_logμ))
            #     return resnorm_sq(work.nnls_prob_smooth_cache[])
            # end
        end

        second = x -> x[2]
        third = x -> x[3]
        fourth = x -> x[4]
        logμ = range(logmu_bounds...; length = 128)#length = 2048)
        # logμ = range(-10, 2; length = 128)#length = 2048)

        logξη = f_lcurve.(logμ)
        logξ, logη = first.(logξη), second.(logξη)
        ξ, η = exp.(logξ), exp.(logη)

        sort!(pairs(f.point_cache); by = ((x, (P, C)),) -> x) #|> display
        unsplat = ((x, (P, C)),) -> (x, P[1], P[2], C)
        dropinf = tups -> filter(((logμ, logξ, logη, C),) -> !isinf(C), tups)
        unzip = tups -> (first.(tups), second.(tups), third.(tups), fourth.(tups))
        logμs, logξs, logηs, Cs = unzip(dropinf(unsplat.(pairs(f.point_cache))))
        ξs, ηs = exp.(logξs), exp.(logηs)

        # C = f_curvature.(logμ)
        # Cmax, iC = findmax(C)
        # _, iC = findmin(i -> abs(logμ[i] - logmu_final), 1:length(logμ))
        # Cmax = f_curvature(logmu_final)
        # logμC, logξC, logηC = logμ[iC], logξ[iC], logη[iC]
        _, iC = findmax(Cs)
        logμC, logξC, logηC, Cmax = logμs[iC], logξs[iC], logηs[iC], Cs[iC]

        qnorm = f_quasi_opt.(logμ)
        qmin, iQ = findmin(qnorm)
        logμQ, logξQ, logηQ = logμ[iQ], logξ[iQ], logη[iQ]
        CQmax = f_curvature(logμQ)

        REnorm = f_heuristic.(logμ)
        REmin, iQ = findmin(REnorm)
        logμRE, logξRE, logηRE = logμ[iQ], logξ[iQ], logη[iQ]
        CREmax = f_curvature(logμRE)

        fig = Main.Figure(; size = (1000, 750))

        ax = Main.Axis(fig[1, 1]; xlabel = "log||Ax-b||²", ylabel = "log||x||²", title = "L-curve (log-log)")
        Main.lines!(ax, logξ, logη)
        Main.scatter!(ax, logξs, logηs)
        Main.scatter!(ax, logξC, logηC; color = :red, markersize = 15)
        # Main.scatter!(ax, logξQ, logηQ; color = :green, marker = :diamond, markersize = 15)
        Main.scatter!(ax, logξRE, logηRE; color = :cyan, marker = :rect, markersize = 15)

        ax = Main.Axis(fig[1, 2]; xlabel = "||Ax-b||²", ylabel = "||x||²", title = "L-curve")
        Main.lines!(ax, ξ, η)
        Main.scatter!(ax, ξs, ηs)
        Main.scatter!(ax, exp(logξC), exp(logηC); color = :red, markersize = 15)
        # Main.scatter!(ax, exp(logξQ), exp(logηQ); color = :green, marker = :diamond, markersize = 15)
        # Main.scatter!(ax, exp(logξRE), exp(logηRE); color = :cyan, marker = :rect, markersize = 15)

        ax = Main.Axis(fig[1, 3]; xlabel = "logμ", ylabel = "μ * ||dx/dμ||", title = "Quasi-optimality")
        Main.lines!(ax, logμ, qnorm)
        Main.scatter!(ax, logμQ, qmin; color = :green, marker = :diamond, markersize = 15)

        ax = Main.Axis(fig[1, 4]; xlabel = "logμ", ylabel = "GCV", title = "Heuristic")
        Main.lines!(ax, logμ, REnorm)
        Main.scatter!(ax, logμRE, REmin; color = :cyan, marker = :rect, markersize = 15)

        ax = Main.Axis(fig[2, 1]; xlabel = "logμ", ylabel = "curvature")
        Main.scatterlines!(ax, logμs, Cs)
        # Main.scatter!(ax, logμC, Cmax; color = :red, markersize = 15)
        # Main.scatter!(ax, logμQ, CQmax; color = :green, marker = :diamond, markersize = 15)
        # Main.scatter!(ax, logμRE, CREmax; color = :cyan, marker = :rect, markersize = 15)

        ax = Main.Axis(fig[2, 2]; xlabel = "log||Ax-b||²", ylabel = "curvature")
        Main.scatterlines!(ax, logξs, Cs)
        # Main.scatter!(ax, logξC, Cmax; color = :red, markersize = 15)
        # Main.scatter!(ax, logξQ, CQmax; color = :green, marker = :diamond, markersize = 15)
        # Main.scatter!(ax, logξRE, CREmax; color = :cyan, marker = :rect, markersize = 15)

        ax = Main.Axis(fig[2, 3]; xlabel = "log||x||²", ylabel = "curvature")
        Main.scatterlines!(ax, logηs, Cs)
        # Main.scatter!(ax, logηC, Cmax; color = :red, markersize = 15)
        # Main.scatter!(ax, logηQ, CQmax; color = :green, marker = :diamond, markersize = 15)
        # Main.scatter!(ax, logηRE, CREmax; color = :cyan, marker = :rect, markersize = 15)

        ax = Main.Axis(fig[2, 4]; xlabel = "T2 bin", ylabel = "x")
        unitmax = x -> x ./ maximum(x)
        _cache = work.nnls_prob_smooth_cache[]
        Main.scatterlines!(ax, 1:length(x_unreg), unitmax(x_unreg); label = "unregularized")
        Main.scatterlines!(ax, 1:length(x_final), unitmax(x_final); label = "l-curve")
        Main.scatterlines!(ax, 1:length(x_final), unitmax(lsqnonneg_gcv(_cache.A, _cache.b).x); label = "gcv")

        ax = Main.Axis(fig[3, 1]; xlabel = "logμ", ylabel = "log||Ax-b||²")
        Main.lines!(ax, logμ, logξ)
        Main.scatter!(ax, logμs, logξs)
        Main.scatter!(ax, logμC, logξC; color = :red, markersize = 15)

        ax = Main.Axis(fig[3, 2]; xlabel = "logμ", ylabel = "log||x||²")
        Main.lines!(ax, logμ, logη)
        Main.scatter!(ax, logμs, logηs)
        Main.scatter!(ax, logμC, logηC; color = :red, markersize = 15)

        fig |> display
    end
    =#

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
function lcurve_corner(f::LCurveCornerCachedFunction{T}, xlow::T = -8.0, xhigh::T = 2.0; xtol = 1e-4, Ptol = 1e-4, Ctol = 1e-4, backtracking = true) where {T}
    # Initialize state
    state = initial_state(f, T(xlow), T(xhigh))

    # Note: tolerances are absolute because typically the L-curve is on a log-log scale, and atol on log-log is equivalent to rtol on linear-linear
    Ptopleft, Pbottomright = state.P⃗[1], state.P⃗[4]
    Ptol = T(Ptol) # convergence occurs when diameter of L-curve state is less than Ptol
    Ctol = T(Ctol) # note: *not* a tolerance on curvature, but on the minimum diameter of the L-curve state used to estimate curvature (see `Pfilter` below)

    # For very small regularization points on the L-curve may be extremely close, leading to
    # numerically unstable curvature estimates. Assign these points -Inf curvature.
    Pfilter = P -> min(norm(P - Ptopleft), norm(P - Pbottomright)) > T(Ctol)
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

    (x, (_, _)), _, _ = mapfindmax(T, ((x, (P, C)),) -> C, pairs(f.point_cache))
    # msg("Converged", state)

    return x
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
#=
function update_curvature!(f::LCurveCornerCachedFunction{T}, state::LCurveCornerState{T}, Pfilter = nothing) where {T}
    # Insert points into point cache
    (; x⃗, P⃗) = state
    @inbounds for i in 1:4
        x, P = x⃗[i], P⃗[i]
        C = Pfilter === nothing || Pfilter(P) ? T(NaN) : T(-Inf) # NaN => curvature needs to be computed, -Inf => point is excluded from search
        f.point_cache[x] = LCurveCornerPoint(P, C)
    end

    # Compute curvature for each point from nearest neighbours
    cache = pairs(f.point_cache) # AbstractVector view of (x, (P, C)) triples
    sort!(cache; by = ((x, (P, C)),) -> x)
    @inbounds for i in 2:length(cache)-1
        x, (P, C) = cache[i]
        isinf(C) && continue # point is excluded from search
        _, (P₋, _) = cache[i-1]
        _, (P₊, _) = cache[i+1]
        C = menger(P₋, P, P₊) # compute new curvature estimate (note: only neighbours to newly added point(s) actually need to be recomputed, but it's fast)
        cache[i] = (x, LCurveCornerPoint(P, C))
    end

    return nothing
end
=#

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
    @assert m >= n "Generalized Cross-Validation (GCV) regularization requires at least as many rows as columns, but got m = $m < n = $n"
    svd_work = SVDValsWorkspace(A) # workspace for computing singular values
    nnls_prob = NNLSProblem(A, b)
    nnls_prob_smooth_cache = NNLSTikhonovRegProblemCache(A, b)
    γ = svd_work.S # store reference to (generalized) singular values for convenience
    return NNLSGCVRegProblem(A, b, m, n, γ, svd_work, nnls_prob, nnls_prob_smooth_cache)
end

@inline solution(work::NNLSGCVRegProblem) = solution(work.nnls_prob_smooth_cache[])
@inline ncomponents(work::NNLSGCVRegProblem) = ncomponents(work.nnls_prob_smooth_cache[])
@inline LinearAlgebra.svdvals!(work::NNLSGCVRegProblem, A = work.A) = svdvals!(work.svd_work, A)

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
lsqnonneg_gcv_work(A::AbstractMatrix, b::AbstractVector) = NNLSGCVRegProblem(A, b)

function lsqnonneg_gcv!(work::NNLSGCVRegProblem{T}; method = :brent, init = -4.0, bounds = (-8.0, 2.0), rtol = 0.0, atol = 1e-4, maxiters = 20) where {T}
    # Find μ by minimizing the function G(μ) (GCV method)
    @assert bounds[1] < init < bounds[2] "Initial value must be within bounds"
    logμ₋, logμ₊ = T.(bounds)
    logμ₀ = T(init)

    # Precompute singular values for GCV computation
    svdvals!(work)

    # Non-zero lower bound for GCV to avoid log(0) in the objective function
    gcv_low = gcv_lower_bound(work)

    # Objective functions
    reset_cache!(work.nnls_prob_smooth_cache)
    function log𝒢(logμ)
        𝒢 = gcv!(work, logμ)
        𝒢 = max(𝒢, gcv_low)
        return log(𝒢)
    end
    function log𝒢_and_∇log𝒢(logμ)
        𝒢, ∇𝒢 = gcv_and_∇gcv!(work, logμ)
        𝒢 = max(𝒢, gcv_low)
        return log(𝒢), ∇𝒢 / 𝒢
    end

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
        opt.min_objective = (logμ, ∇logμ) -> @inbounds Float64(log𝒢(T(logμ[1])))
        minf, minx, ret   = NLopt.optimize(opt, Float64[logμ₀])
        logmu_final       = @inbounds T(minx[1])
        log𝒢_final        = T(minf)
    elseif method === :brent
        logmu_final, log𝒢_final = brent_minimize(log𝒢, logμ₋, logμ₊; xrtol = T(rtol), xatol = T(atol), maxiters)
        #TODO
        # log𝒢₋, ∇log𝒢₋ = log𝒢_and_∇log𝒢(logμ₋)
        # log𝒢₊, ∇log𝒢₊ = log𝒢_and_∇log𝒢(logμ₊)
        # logμ_bdry, log𝒢_bdry = log𝒢₋ < log𝒢₊ ? (logμ₋, log𝒢₋) : (logμ₊, log𝒢₊)
        # if ∇log𝒢₋ < 0 && ∇log𝒢₊ > 0
        #     logmu_final, log𝒢_final = brent_minimize(log𝒢, logμ₋, logμ₊; xrtol = T(rtol), xatol = T(atol), maxiters)
        # else
        #     logmu_final, log𝒢_final = logμ_bdry, log𝒢_bdry
        # end
        # if log𝒢_bdry < log𝒢_final
        #     logmu_final, log𝒢_final = logμ_bdry, log𝒢_bdry
        # end
    elseif method === :brent_newton
        log𝒢₋, ∇log𝒢₋ = log𝒢_and_∇log𝒢(logμ₋)
        log𝒢₊, ∇log𝒢₊ = log𝒢_and_∇log𝒢(logμ₊)
        logμ_bdry, log𝒢_bdry = log𝒢₋ < log𝒢₊ ? (logμ₋, log𝒢₋) : (logμ₊, log𝒢₊)
        if ∇log𝒢₋ < 0 && ∇log𝒢₊ > 0
            log𝒢₀, ∇log𝒢₀ = log𝒢_and_∇log𝒢(logμ₀)
            logmu_final, log𝒢_final = brent_newton_minimize(log𝒢_and_∇log𝒢, logμ₋, logμ₊, logμ₀, log𝒢₀, ∇log𝒢₀; xrtol = T(rtol), xatol = T(atol), maxiters)
        else
            logmu_final, log𝒢_final = logμ_bdry, log𝒢_bdry
        end
        if log𝒢_bdry < log𝒢_final
            logmu_final, log𝒢_final = logμ_bdry, log𝒢_bdry
        end
    else
        error("Unknown minimization method: $method")
    end

    # Return the final regularized solution
    mu_final = exp(logmu_final)
    x_final = solve!(work.nnls_prob_smooth_cache, mu_final)
    x_unreg = solve!(work.nnls_prob)
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
    #   NOTE: assumes `svdvals!(work)` has been called and that the singular values `work.γ` are ready
    (; m, n, γ) = work

    # Solve regularized NNLS problem
    μ = exp(logμ)
    solve!(work.nnls_prob_smooth_cache, μ)
    cache = work.nnls_prob_smooth_cache[]

    # Compute GCV
    res² = resnorm_sq(cache) # squared residual norm ||A * x(μ) - b||^2
    dof = gcv_dof(m, n, γ, μ) # degrees of freedom; γ are (generalized) singular values
    gcv = res² / dof^2

    return gcv
end

function gcv_and_∇gcv!(work::NNLSGCVRegProblem, logμ)
    # Unpack buffers
    #   NOTE: assumes `svdvals!(work)` has been called and that the singular values `work.γ` are ready
    (; m, n, γ) = work

    # Solve regularized NNLS problem
    μ = exp(logμ)
    solve!(work.nnls_prob_smooth_cache, μ)
    cache = work.nnls_prob_smooth_cache[]

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
    res² = resnorm_sq(work.nnls_prob_smooth_cache[])

    if extract_subproblem
        # Extract equivalent unconstrained least squares subproblem from NNLS problem
        # by extracting columns of A which correspond to nonzero components of x(μ)
        idx = NNLS.components(work.nnls_prob_smooth_cache[].nnls_prob.nnls_work)
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
