####
#### NNLS submodule
####

# This NNLS submodule is modified version of the corresponding NNLS module from
# the forked NonNegLeastSquares.jl repository:
#
#   https://github.com/jondeuce/NonNegLeastSquares.jl/blob/a122bf7acb498efcaf140b719133691e7c4cd03d/src/nnls.jl
#
# The original MIT licence from NonNegLeastSquares.jl is included below:
#
# -----------------------------------------------------------------------------
#
# The MIT License (MIT)
#
# Copyright (c) 2015 Alex Williams
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# -----------------------------------------------------------------------------

module NNLS

using Base.Cartesian: @nexprs
using LinearAlgebra: LinearAlgebra, Factorization, UpperTriangular, ldiv!, norm
using MuladdMacro: MuladdMacro, @muladd

export nnls, nnls!, load!
export NNLSWorkspace, NNLSGram, NormalEquation, NormalEquationCholesky

@muladd begin

"""
    NNLSWorkspace(A::AbstractMatrix{T}, b::AbstractVector{T})
    NNLSWorkspace(::Type{T}, m::Int, n::Int)

Preallocated workspace for the nonnegative least squares problem

```math
\\min_{x \\ge 0} ||Ax - b||_2
```

for an `m × n` matrix `A`, reused across solves so that repeated calls allocate nothing.
Pass it to [`nnls!`](@ref), then read the results with [`solution`](@ref), [`dual`](@ref), [`components`](@ref), [`ncomponents`](@ref) and [`residualnorm`](@ref).

The Tikhonov-regularized problem is solved by passing `A = [A₀; λI]` and `b = [b₀; 0]`, in which case the workspace must be sized for the padded system.
"""
struct NNLSWorkspace{T}
    A::Matrix{T}                # factor storage; A[1:nsetp, 1:nsetp] holds the upper triangular factor of the passive columns
    b::Vector{T}                # transformed right-hand side Q'b
    x::Vector{T}                # solution, indexed by original column
    w::Vector{T}                # dual, indexed by position; see `dual` for the original-column view
    zz::Vector{T}               # trial solution of the current round, doubling as the candidate column buffer
    idx::Vector{Int}            # column permutation; idx[1:nsetp] is the passive set, idx[nsetp+1:n] the active set
    invidx::Vector{Int}         # inverse permutation of idx
    diag::Vector{Int}           # activation row of each column's λ row; 0 means not yet activated
    b0::Vector{T}               # original right-hand side, the reference for residual and dual computations
    r::Vector{T}                # residual buffer r = b0 - A0[:, idx[1:nsetp]] * x₊
    H::Matrix{T}                # append-only panel of scaled Householder vector tails
    htau::Vector{T}             # Householder scalar factors
    hpos::Vector{Int}           # pivot row of each stored Householder
    hm1::Vector{Int}            # last row of each stored Householder
    hlen::Base.RefValue{Int}    # number of stored Householders
    gc::Vector{T}               # Givens cosines
    gs::Vector{T}               # Givens sines
    gi::Vector{Int}             # Givens row indices; rotation g acts on rows (gi[g]-1, gi[g])
    transforms::Vector{Int}     # transform order: +t = Householder t, -g = Givens g
    rnorm::Base.RefValue{T}     # residual norm at the solution
    mode::Base.RefValue{Int}    # termination status; 0 on success
    nsetp::Base.RefValue{Int}   # passive-set size
    solved::Base.RefValue{Bool} # see `issolved`
end

"""
    solution(work::NNLSWorkspace)

Solution `x` of the last solve, indexed by original column. Inactive components are exactly zero.
"""
@inline solution(work::NNLSWorkspace) = work.x

"""
    dual(work::NNLSWorkspace)

Dual vector `w = Aᵀ(b - Ax)` of the last solve, indexed by original column.
At a solution, `w ≤ 0` with `w = 0` on the solution indices; see [`components`](@ref).
"""
@inline dual(work::NNLSWorkspace) = @views work.w[work.invidx]

"""
    residualnorm(work::NNLSWorkspace)

Residual norm `||Ax - b||₂` at the solution of the last solve.
"""
@inline residualnorm(work::NNLSWorkspace) = work.rnorm[]

"""
    residual(work::NNLSWorkspace)

Residual `r = b₀ - A₀x` of the last solve, in original data coordinates.
Every solve leaves it current, so consumers that need the residual itself, and not only its norm, can read it instead of rebuilding `b₀ - A₀[:, P] x_P`.
For the padded Tikhonov convention only the leading `m₀` data rows are meaningful; the Tikhonov rows contribute `λ²||x₊||²` to the norm separately.
Note the sign: this is `b₀ - A₀x`, the negative of the fit residual reported by the output maps.
"""
@inline residual(work::NNLSWorkspace) = work.r

"""
    ncomponents(work::NNLSWorkspace)

Number of positive components in the solution of the last solve.
"""
@inline ncomponents(work::NNLSWorkspace) = work.nsetp[]

"""
    components(work::NNLSWorkspace)

Original column indices of the positive components in the solution of the last solve.
"""
@inline components(work::NNLSWorkspace) = @views work.idx[1:ncomponents(work)]

"""
    issolved(work::NNLSWorkspace)

Whether [`solution`](@ref), [`residual`](@ref), [`residualnorm`](@ref) and [`components`](@ref) hold the solution of the `A` and `b` most recently passed to a solve.
Every solve sets it. Callers that overwrite the `A` they solved against must clear it, since the workspace keeps no reference to `A` and cannot detect the write.
"""
@inline issolved(work::NNLSWorkspace) = work.solved[]

@inline positive_solution(work::NNLSWorkspace) = @views solution(work)[components(work)]
@inline positive_solution!(work::NNLSWorkspace, x::AbstractVector) = copyto!(x, positive_solution(work))
@inline choleskyfactor(work::NNLSWorkspace, ::Val{:U}) = @views UpperTriangular(work.A[1:ncomponents(work), 1:ncomponents(work)])
@inline choleskyfactor(work::NNLSWorkspace, ::Val{:L}) = choleskyfactor(work, Val(:U))'

function Base.show(io::IO, ::MIME"text/plain", work::NNLSWorkspace)
    (; A, b, x, w, zz, idx, invidx, diag, rnorm, mode, nsetp, solved) = work
    m, n = size(A)
    println(io, "NNLSWorkspace(m = $m, n = $n)")
    println(io, "  A        :: $(typeof(A)) size $(size(A))")
    println(io, "  b        :: $(typeof(b)) size $(size(b))")
    println(io, "  x        :: $(typeof(x)) size $(size(x))")
    println(io, "  w        :: $(typeof(w)) size $(size(w))")
    println(io, "  zz       :: $(typeof(zz)) size $(size(zz))")
    println(io, "  idx      :: $(typeof(idx)) size $(size(idx))")
    println(io, "  invidx   :: $(typeof(invidx)) size $(size(invidx))")
    println(io, "  diag     :: $(typeof(diag)) size $(size(diag))")
    println(io, "  rnorm[]  :: $(typeof(rnorm[])) = $(rnorm[])")
    println(io, "  mode[]   :: $(typeof(mode[])) = $(mode[])")
    println(io, "  nsetp[]  :: $(typeof(nsetp[])) = $(nsetp[])")
    println(io, "  solved[] :: $(typeof(solved[])) = $(solved[])")
    return nothing
end

function NNLSWorkspace(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    @assert size(b) == (m,)
    return NNLSWorkspace(T, m, n)
end

function NNLSWorkspace(::Type{T}, m::Int, n::Int) where {T}
    hcap = 2n + 8 # covers any realistic number of passive-set entries (n entries + re-entries)
    return NNLSWorkspace(
        zeros(T, m, n),       # A
        zeros(T, m),          # b
        zeros(T, n),          # x
        zeros(T, n),          # w
        zeros(T, m),          # zz
        zeros(Int, n),        # idx (Note: deliberately initialize to invalid permutation)
        zeros(Int, n),        # invidx
        zeros(Int, n),        # diag
        zeros(T, m),          # b0
        zeros(T, m),          # r
        zeros(T, m, hcap),    # H
        zeros(T, hcap),       # htau
        zeros(Int, hcap),     # hpos
        zeros(Int, hcap),     # hm1
        Ref(0),               # hlen
        sizehint!(T[], 4n),   # gc
        sizehint!(T[], 4n),   # gs
        sizehint!(Int[], 4n), # gi
        sizehint!(Int[], 4n), # transforms
        Ref(zero(T)),         # rnorm
        Ref(0),               # mode
        Ref(0),               # nsetp
        Ref(false),           # solved
    )
end

function Base.copy(w::NNLSWorkspace)
    return NNLSWorkspace(
        copy(w.A),
        copy(w.b),
        copy(w.x),
        copy(w.w),
        copy(w.zz),
        copy(w.idx),
        copy(w.invidx),
        copy(w.diag),
        copy(w.b0),
        copy(w.r),
        copy(w.H),
        copy(w.htau),
        copy(w.hpos),
        copy(w.hm1),
        Ref(w.hlen[]),
        copy(w.gc),
        copy(w.gs),
        copy(w.gi),
        copy(w.transforms),
        Ref(w.rnorm[]),
        Ref(w.mode[]),
        Ref(w.nsetp[]),
        Ref(w.solved[]),
    )
end

"""
    load!(work::NNLSWorkspace, A::AbstractMatrix, b::AbstractVector)

Copy the problem data `A` and `b` into `work`. The sizes must match those the workspace was constructed for.
"""
function load!(work::NNLSWorkspace{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    @assert size(A) == size(work.A)
    @assert size(b) == size(work.b)
    copyto!(work.A, A)
    copyto!(work.b, b)
    return work
end

# Load the data rows of b into the workspace, snapshot them into b0 as the reference for residual and dual computations, and compute the initial dual w = A[1:m, :]'b.
# The initial residual is exactly b0, so this is the round-0 dual and callers pass `init_dual = false` to the solver.
# A itself is not copied; the solver reads pristine column data directly from the caller's matrix.
# For the unregularized problem m = size(A, 1); for the Tikhonov-padded A = [A₀; λI] only the top m = M - N data rows contribute, since b = 0 on the padded rows.
function init_dual!(work::NNLSWorkspace{T}, A::AbstractMatrix{T}, b::AbstractVector{T}, m::Int = size(work.A, 1)) where {T}
    @assert size(A) == size(work.A)
    @assert size(b) == size(work.b)
    @assert 0 <= m <= size(work.A, 1)
    n = size(work.A, 2)
    @inbounds @simd ivdep for i in 1:m
        bi = b[i]
        work.b[i] = bi
        work.b0[i] = bi
    end
    w, b0 = work.w, work.b0
    @inbounds for j in 1:n
        sm = zero(T)
        @simd ivdep for i in 1:m
            sm = muladd(A[i, j], b0[i], sm)
        end
        w[j] = sm
    end
    return work
end

function checkargs(work::NNLSWorkspace)
    m, n = size(work.A)
    @assert size(work.b) == (m,)
    @assert size(work.x) == (n,)
    @assert size(work.w) == (n,)
    @assert size(work.zz) == (m,)
    @assert size(work.idx) == (n,)
    @assert size(work.invidx) == (n,)
    @assert size(work.b0) == (m,)
    @assert size(work.r) == (m,)
    @assert 0 <= work.rnorm[]
    @assert 0 <= work.mode[]
    @assert 0 <= work.nsetp[] <= min(m, n)
end

#### Cholesky factorization for the normal equation A'Ax = A'b

"""
    NormalEquationCholesky <: LinearAlgebra.Factorization

Cholesky factorization of the normal equations `AₚᵀAₚ`, where `Aₚ = A[:, components(work)]` holds the positive components of the last solve.
Obtained from `cholesky(NormalEquation(), work)` and usable with `ldiv!`. The factor is the triangular factor the solver already maintains, so nothing is refactorized.
"""
struct NormalEquationCholesky{T, W <: NNLSWorkspace{T}} <: Factorization{T}
    work::W
end
@inline Base.size(F::NormalEquationCholesky) = (n = size(F.work.A, 2); return (n, n))

function solve_triangular_system!(y, F::NormalEquationCholesky, ::Val{transp} = Val(false)) where {transp}
    solve_triangular_system!(y, F.work.A, F.work.nsetp[], Val(transp))
    return y
end

function LinearAlgebra.ldiv!(F::NormalEquationCholesky, x::AbstractVector)
    @assert length(x) == F.work.nsetp[]
    solve_triangular_system!(x, F, Val(true)) # x -> U'\x
    solve_triangular_system!(x, F, Val(false)) # U'\x -> U\(U'\x)
    return x
end
function LinearAlgebra.ldiv!(y::AbstractVector, F::NormalEquationCholesky, x::AbstractVector)
    @assert length(x) == length(y)
    copyto!(y, x)
    return ldiv!(F, y)
end
Base.:\(F::NormalEquationCholesky, x::AbstractVector) = ldiv!(F, copy(x))

"""
    NormalEquation

Singleton type indicating the normal-equations factorization of an [`NNLSWorkspace`](@ref); see [`NormalEquationCholesky`](@ref).
"""
struct NormalEquation end

LinearAlgebra.cholesky!(::NormalEquation, work::NNLSWorkspace) = NormalEquationCholesky(work)

"""
x = nnls(A, b; ...)

Solves non-negative least-squares problem by the active set method
of Lawson & Hanson (1974).

Optional arguments:

    - max_iter: maximum number of iterations (counts inner loop iterations)

References:

    - Lawson, C.L. and R.J. Hanson, Solving Least-Squares Problems
    - Prentice-Hall, Chapter 23, p. 161, 1974
"""
function nnls(A::AbstractMatrix{T}, b::AbstractVector{T}, args...; kwargs...) where {T}
    work = NNLSWorkspace(A, b)
    return nnls!(work, A, b, args...; kwargs...)
end

"""
    nnls!(work::NNLSWorkspace, A::AbstractMatrix, b::AbstractVector)
    nnls!(work::NNLSWorkspace, A::AbstractMatrix, b::AbstractVector, λ::Real)

Solve `min_{x ≥ 0} ||Ax - b||₂` in place, returning [`solution`](@ref)`(work)`.

The second form solves the Tikhonov-regularized problem `min_{x ≥ 0} ||A₀x - b₀||₂² + λ²||x||₂²` and requires `A = [A₀; λI]` and `b = [b₀; 0]`, i.e. `size(A, 1) > size(A, 2)`.
"""
function nnls!(
    work::NNLSWorkspace{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T};
    kwargs...,
) where {T}
    checkargs(work)
    init_dual!(work, A, b)
    init_nnls!(work)
    unsafe_nnls!(work, A; kwargs..., init_dual = false) # init_dual! preloaded the round-0 dual w = A'b
    return solution(work)
end

function nnls!(
    work::NNLSWorkspace{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    λ::T;
    kwargs...,
) where {T}
    size(A, 1) > size(A, 2) || throw(DimensionMismatch("A must be of the form [A₀; λ*I], got size(A) = $(size(A))"))
    checkargs(work)
    init_dual!(work, A, b, size(A, 1) - size(A, 2))
    init_nnls!(work, λ)
    unsafe_nnls!(work, A, λ; kwargs..., init_dual = false) # init_dual! preloaded the round-0 dual w = A₀'b₀
    return solution(work)
end

# Warm-started Tikhonov solve: seed the passive set with the columns idx0[1:nsetp0], typically `components(work)` cached from a solve at a nearby λ.
# Seeds enter without the positivity check, a feasibility pass drops the infeasible ones, and the standard algorithm then runs to convergence, so the result matches a cold solve in quality.
function nnls!(
    work::NNLSWorkspace{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    λ::T,
    idx0::AbstractVector{Int},
    nsetp0::Int;
    kwargs...,
) where {T}
    size(A, 1) > size(A, 2) || throw(DimensionMismatch("A must be of the form [A₀; λ*I], got size(A) = $(size(A))"))
    checkargs(work)
    n = size(work.A, 2)
    0 <= nsetp0 <= min(n, length(idx0)) || throw(ArgumentError("require 0 <= nsetp0 <= min(n, length(idx0))"))
    init_dual!(work, A, b, size(A, 1) - size(A, 2))
    init_nnls!(work, λ)
    @inbounds for t in 1:nsetp0
        j = idx0[t]
        1 <= j <= n || throw(ArgumentError("warm-start indices must be in 1:n"))
        work.hpos[t] = j # stash the seed; read back (before slot t is reused) in `unsafe_nnls!`
    end
    # With a nonempty seed the round-0 dual must be recomputed from the seeded residual; with an empty seed this is a cold solve and the dual preloaded by `init_dual!` is exact.
    unsafe_nnls!(work, A, λ; kwargs..., nwarm = nsetp0, init_dual = nsetp0 > 0)
    return solution(work)
end

# Warm-started unregularized solve: seed the passive set with the columns idx0[1:nsetp0], typically `components(work)` cached from a solve against a similar matrix.
# Same seeding protocol as the Tikhonov warm start above.
function nnls!(
    work::NNLSWorkspace{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    idx0::AbstractVector{Int},
    nsetp0::Int;
    kwargs...,
) where {T}
    checkargs(work)
    n = size(work.A, 2)
    0 <= nsetp0 <= min(n, length(idx0)) || throw(ArgumentError("require 0 <= nsetp0 <= min(n, length(idx0))"))
    init_dual!(work, A, b)
    init_nnls!(work)
    @inbounds for t in 1:nsetp0
        j = idx0[t]
        1 <= j <= n || throw(ArgumentError("warm-start indices must be in 1:n"))
        work.hpos[t] = j # stash the seed; read back (before slot t is reused) in `unsafe_nnls!`
    end
    # The preloaded round-0 dual w = A'b is exact for an empty seed; the seeding block invalidates it as soon as any seed enters.
    unsafe_nnls!(work, A; kwargs..., nwarm = nsetp0, init_dual = false)
    return solution(work)
end

# Construct the Householder transformation Q = I + u*(uᵀ)/b that annihilates x[2:end], overwriting x with u and returning the scalar factor tau.
# Adapted from the FORTRAN of Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory,
# published in "Solving Least Squares Problems", Prentice-Hall, 1974, revised February 1995 for the SIAM reprint.
function construct_householder!(x::AbstractVector{T}) where {T}
    if length(x) <= 1
        return zero(T)
    end

    @inbounds alpha = x[1]
    xnorm = norm(x)
    if xnorm == 0
        return zero(T)
    end

    beta = copysign(xnorm, alpha)
    alpha += beta

    @inbounds x[1] = -beta
    @inbounds @simd for i in 2:length(x)
        x[i] /= alpha
    end

    tau = alpha / beta
    return tau
end

# Apply the Householder transformation defined by u and tau to the vector c, in place.
# Adapted from the FORTRAN of Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory,
# published in "Solving Least Squares Problems", Prentice-Hall, 1974, revised February 1995 for the SIAM reprint.
function apply_householder!(
    c::AbstractVector{T},
    u::AbstractVector{T},
    tau::T,
) where {T}
    m = length(u)
    if m <= 1
        return nothing
    end

    @inbounds u1 = u[1]
    @inbounds u[1] = 1

    sm = zero(T)
    @inbounds @simd for i in 1:m
        sm = sm + c[i] * u[i]
    end

    sm *= -tau

    if sm != 0
        @inbounds @simd for i in 1:m
            c[i] = c[i] + sm * u[i]
        end
    end

    @inbounds u[1] = u1

    return nothing
end

# Apply the accumulated transforms to a fresh column c expressed in original row coordinates, i.e. compute c -> Q'c.
# The transforms are the Householder reflections from passive-set entries, interleaved with the Givens rotations from passive-set downdates.
function apply_transforms!(c::AbstractVector{T}, work::NNLSWorkspace{T}) where {T}
    (; H, htau, hpos, hm1, gc, gs, gi, transforms) = work
    @inbounds for op in transforms
        if op > 0 # Householder reflection: u[ip] = 1 implicit, scaled tail in H[ip+1:m1, op]
            ip, m1, tau = hpos[op], hm1[op], htau[op]
            sm = c[ip]
            @simd for i in ip+1:m1
                sm = sm + c[i] * H[i, op]
            end
            sm *= -tau
            c[ip] = c[ip] + sm
            @simd ivdep for i in ip+1:m1
                c[i] = c[i] + sm * H[i, op]
            end
        else # Givens rotation acting on rows (i-1, i)
            g = -op
            i, cc, ss = gi[g], gc[g], gs[g]
            c[i-1], c[i] = orthogonal_rotmatvec(cc, ss, c[i-1], c[i])
        end
    end
    return c
end

# Apply the single stored Householder reflection t to column `col` of C, which keeps staged candidates reduced as earlier block members enter.
@inline function apply_householder_to_col!(C::AbstractMatrix{T}, col::Int, work::NNLSWorkspace{T}, t::Int) where {T}
    (; H, htau, hpos, hm1) = work
    @inbounds begin
        ip, m1, tau = hpos[t], hm1[t], htau[t]
        sm = C[ip, col]
        @simd for i in ip+1:m1
            sm = sm + C[i, col] * H[i, t]
        end
        sm *= -tau
        C[ip, col] = C[ip, col] + sm
        @simd ivdep for i in ip+1:m1
            C[i, col] = C[i, col] + sm * H[i, t]
        end
    end
    return nothing
end

# Compute the dual, i.e. the negative gradient, for the active-set columns j = nsetp+1:n:
#   w[j] = A0[1:mdata, idx[j]]' * r,   r = b0 - A0[:, idx[1:nsetp]] * x₊,
# where x₊ = zz[1:nsetp] is the current passive-set solution and A0 is the caller's pristine matrix, indexed by original column.
# Q is orthogonal, so a_j'r equals the transformed-space dual of the classic algorithm.
# The padded rows of the Tikhonov problem contribute nothing, since x_j = 0 for set Z columns.
function compute_dual!(work::NNLSWorkspace{T}, A0::AbstractMatrix{T}, nsetp::Int, mdata::Int) where {T}
    (; w, zz, b0, idx, r) = work
    @inbounds @simd ivdep for i in 1:mdata
        r[i] = b0[i]
    end
    @inbounds for t in 1:nsetp
        xt = zz[t]
        jt = idx[t]
        @simd ivdep for i in 1:mdata
            r[i] = r[i] - xt * A0[i, jt]
        end
    end
    n = length(w)
    j = nsetp + 1
    while j + 3 <= n # blocks of 4 columns
        compute_dual_block!(w, A0, r, idx, mdata, j, Val(4))
        j += 4
    end
    if j + 1 <= n # remainder: 2/1 column blocks
        compute_dual_block!(w, A0, r, idx, mdata, j, Val(2))
        j += 2
    end
    if j <= n
        compute_dual_block!(w, A0, r, idx, mdata, j, Val(1))
    end
    return work
end

# Attempt to move a candidate column into the passive set at position ip = nsetp + 1.
# On input the scratch vector c, which is `work.zz`, holds the candidate column in original row coordinates over rows 1:m1; the caller zeroes inactive padded rows and places any λ entry.
# The candidate is reduced against the current Q via the transforms, a Householder reflection is built on rows ip:m1, and the classic entering-column checks follow: sufficient independence and positivity of the proposed new coefficient.
# On acceptance the reflection is appended to the log and applied to b, and the new U column is written at position ip. Pristine column data is never moved; kernels read it from the caller's matrix via `idx`.
# Returns tau >= 0 on acceptance, -1 on rejection, and -2 when the Householder panel is full, which the caller treats like iteration exhaustion.
function try_enter_column!(work::NNLSWorkspace{T}, c::AbstractVector{T}, ip::Int, m1::Int, check::Bool = true, reduce::Bool = true) where {T}
    (; A, b, H, htau, hpos, hm1, hlen, transforms) = work

    if reduce # skipped when the caller already reduced c against Q (block staging)
        apply_transforms!(c, work)
    end

    # Construct the Householder reflection of c on rows ip:m1
    @inbounds alpha = c[ip]
    xnorm = alpha * alpha
    bdotc = zero(T)
    @inbounds @simd for i in ip+1:m1
        ci = c[i]
        xnorm = xnorm + ci * ci
        bdotc = bdotc + b[i] * ci
    end
    xnorm = sqrt(xnorm)

    if xnorm == 0 # candidate column is numerically zero on the free rows
        return -one(T)
    end

    beta  = copysign(xnorm, alpha)
    alpha = alpha + beta
    tau   = alpha / beta

    sm = b[ip] + bdotc / alpha
    sm *= -tau

    A1 = -beta
    @inbounds b1 = b[ip] + sm

    if check && !(b1 / A1 > 0) # proposed new coefficient is not strictly positive
        return -one(T)
    end

    @inbounds if ip < m1
        # Append the reflection to the log and apply it to b
        t = hlen[] + 1
        if t > size(H, 2) # Householder panel is full; pathological, treated like iteration exhaustion by the caller
            return -2 * one(T)
        end
        inv_alpha = inv(alpha)
        @simd ivdep for i in ip+1:m1
            ui = c[i] * inv_alpha
            H[i, t] = ui
            b[i] = b[i] + sm * ui
        end
        b[ip] = b1
        htau[t], hpos[t], hm1[t] = tau, ip, m1
        hlen[] = t
        push!(transforms, t)
    else
        tau = zero(T) # single-row reflection acts as the identity (matches the classic algorithm)
    end

    # Write the new U column at position ip: reduced head, -beta on the diagonal or the reduced value itself in the trivial single-row case, then zeros below. The Householder tail lives in the H panel, not in A.
    @inbounds @simd ivdep for i in 1:ip-1
        A[i, ip] = c[i]
    end
    @inbounds A[ip, ip] = ip < m1 ? A1 : c[ip]
    M = size(A, 1)
    @inbounds @simd ivdep for i in ip+1:M
        A[i, ip] = zero(T)
    end

    return tau
end

# Dual components for a block of K active-set positions j:j+K-1.
# The pristine columns are gathered from A0 by original index via idx, and the K accumulator streams share each load of the residual r.
for K in (1, 2, 4)
    @eval @inline function compute_dual_block!(
        w::AbstractVector{T},
        A0::AbstractMatrix{T},
        r::AbstractVector{T},
        idx::AbstractVector{Int},
        m1::Int,
        j::Int,
        ::Val{$K},
    ) where {T}
        @muladd @inbounds begin
            @nexprs $K α -> j_α = idx[j+(α-1)]
            @nexprs $K α -> sm_α = zero(T)
            @simd for i in 1:m1
                ri = r[i]
                @nexprs $K α -> sm_α = sm_α + A0[i, j_α] * ri
            end
            @nexprs $K α -> w[j+(α-1)] = sm_α
        end
        return nothing
    end
end

# Compute the Givens rotation (c, s) with (c, s; -s, c) * (a, b) = (σ, 0) and σ = hypot(a, b).
# Adapted from the FORTRAN of Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory,
# published in "Solving Least Squares Problems", Prentice-Hall, 1974, revised February 1995 for the SIAM reprint.
@inline function orthogonal_rotmat(a::T, b::T) where {T}
    σ = hypot(a, b)
    c = a / σ
    s = b / σ
    return c, s, σ
end

@inline function orthogonal_rotmatvec(c::T, s::T, a::T, b::T) where {T}
    x = c * a + s * b
    y = -s * a + c * b
    return x, y
end

# Solve the triangular system U * x = z, or Uᵀ * x = z when `transp`, overwriting the right-hand side z with x.
# Adapted from the FORTRAN of Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory,
# published in "Solving Least Squares Problems", Prentice-Hall, 1974, revised February 1995 for the SIAM reprint.
function solve_triangular_system!(
    z::AbstractVector{T},
    A::AbstractMatrix{T},
    n::Int = size(A, 2),
    ::Val{transp} = Val(false),
) where {T, transp}
    if !transp
        # Solve the upper-triangular system Ux=b in-place where:
        #   U = A[1:n, 1:n]
        #   b = z[1:n]
        #   x = z[1:n] (i.e. RHS b is overwritten)
        @inbounds for j in n:-1:1
            zi = -z[j] / A[j, j]
            @simd ivdep for i in 1:j-1
                z[i] = z[i] + A[i, j] * zi
            end
            z[j] = -zi
        end
    else
        # Solve the lower-triangular system Lx=b in-place where:
        #   L = A[1:n, 1:n]' (i.e. transpose of U above)
        #   b = z[1:n]
        #   x = z[1:n] (i.e. RHS b is overwritten)
        @inbounds for j in 1:n
            z1 = z[j]
            @simd for l in 1:j-1
                z1 = z1 - A[l, j] * z[l]
            end
            z1 /= A[j, j]
            z[j] = z1
        end
    end
    return z
end

function largest_positive_dual(
    w::AbstractVector{T},
    j1::Int,
) where {T}
    wmax = zero(T)
    jmax = 0
    @inbounds @simd for j in j1:length(w)
        newmax = w[j] > wmax
        wmax = ifelse(newmax, w[j], wmax)
        jmax = ifelse(newmax, j, jmax)
    end
    return wmax, jmax
end

# Move the passive-set column at position imv back to the active set.
# Givens rotations restore the triangular structure of U; they are applied to the remaining passive-set columns and to b, and recorded in `transforms`.
# Columns imv+1:nsetp then shift left by one, mirrored in idx.
# Pristine data never lives in work.A, so nothing needs restoring there.
# Returns the new nsetp.
function downdate!(work::NNLSWorkspace{T}, imv::Int, nsetp::Int, mA::Int) where {T}
    (; A, b, idx, gc, gs, gi, transforms) = work
    @inbounds begin
        if imv != nsetp
            for i in imv+1:nsetp
                cc, ss, rr = orthogonal_rotmat(A[i-1, i], A[i, i])
                A[i-1, i] = rr
                A[i, i] = zero(T)

                # Apply procedure G2 (CC,SS,A(J-1,L),A(J,L)) to the remaining set P columns (set Z columns are unaffected)
                @simd for j in 1:i-1
                    A[i-1, j], A[i, j] = orthogonal_rotmatvec(cc, ss, A[i-1, j], A[i, j])
                end
                @simd for j in i+1:nsetp
                    A[i-1, j], A[i, j] = orthogonal_rotmatvec(cc, ss, A[i-1, j], A[i, j])
                end

                # Apply procedure G2 (CC,SS,B(J-1),B(J)) and log the rotation
                b[i-1], b[i] = orthogonal_rotmatvec(cc, ss, b[i-1], b[i])
                push!(gc, cc)
                push!(gs, ss)
                push!(gi, i)
                push!(transforms, -length(gc))
            end

            # Swap the U columns (set Z columns hold no data)
            for j in imv:nsetp-1
                @simd for i in 1:mA
                    A[i, j+1], A[i, j] = A[i, j], A[i, j+1]
                end
                idx[j], idx[j+1] = idx[j+1], idx[j]
            end
        end
    end
    return nsetp - 1
end

function init_nnls!(work::NNLSWorkspace{T}) where {T}
    checkargs(work)
    (; x, idx, invidx) = work

    fill!(x, zero(T))
    copyto!(idx, 1:length(idx))
    copyto!(invidx, 1:length(invidx))

    return work
end

function init_nnls!(work::NNLSWorkspace{T}, λ::T) where {T}
    checkargs(work)
    (; b, x, idx, invidx, diag) = work

    M, N = size(work.A)
    M > N || throw(DimensionMismatch("A must be of the form [A₀; λ*I], got size(A) = $(size(work.A))"))
    m, n = M - N, N

    # The padded rows A[m+1:M, :] are never read by `unsafe_nnls!(work, A₀, λ)`, since candidate columns are
    # materialized in a scratch buffer with their λ entry placed on the fly, so the bottom block is deliberately left untouched.
    @inbounds @simd for i in 1:N
        x[i] = zero(T)
        b[m+i] = zero(T)
        idx[i] = i
        invidx[i] = i
        diag[i] = 0
    end

    return work
end

"""
Algorithm NNLS: NONNEGATIVE LEAST SQUARES

The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 15, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.

GIVEN AN M BY N MATRIX, A, AND AN M-VECTOR, B, COMPUTE AN
N-VECTOR, X, THAT SOLVES THE LEAST SQUARES PROBLEM
A * X = B SUBJECT TO X .GE. 0
"""
function unsafe_nnls!(
    work::NNLSWorkspace{T},
    A0::AbstractMatrix{T}; # pristine problem data, read-only, indexed by original column (only rows 1:m are read)
    init_dual::Bool = true,
    max_iter::Int = 3 * size(work.A, 2),
    nwarm::Int = 0, # number of warm-start columns stashed in work.hpos[1:nwarm]
) where {T}
    (; A, b, x, w, zz, idx, invidx, b0, r, H, hpos, hlen, gc, gs, gi, transforms) = work
    m, n = size(A)

    copyto!(b0, b) # b holds the untransformed right-hand side at entry (for every caller); snapshot it for residual/dual computations
    hlen[] = 0
    empty!(gc)
    empty!(gs)
    empty!(gi)
    empty!(transforms)

    nsetp = 0
    iter = 0
    work.mode[] = 0
    terminated = false
    use_stale_w = !init_dual # init_dual = false: caller preloaded w for the first pivot selection
    rfresh = false # r = b0 - A0[:, idx[1:nsetp]]*x₊ is current for the final passive set (see the rnorm epilogue)

    # ******  WARM START  ******
    # Seed the passive set with the stashed columns, skipping the positivity check; a feasibility pass below drops any that come out non-positive. This mirrors the Tikhonov warm start without the λ rows.
    # Panel-QR seeding: all seed columns are staged upfront in spare columns of H and kept reduced column-parallel as each accepted seed's reflection is built, right-looking.
    # `apply_transforms!` is latency-bound, so replaying the whole transform log once per seed instead would dominate the cost of a warm-started solve.
    if nwarm > 0
        hcap = size(H, 2)
        sbase = hcap - nwarm # staging columns sbase+1:sbase+nwarm; reflections built during seeding occupy slots 1:nwarm <= sbase
        @inbounds for t in 1:nwarm # stage before any reflection exists (hpos[1:nwarm] still holds the stashed seeds)
            jorig = work.hpos[t]
            @simd ivdep for i in 1:m
                H[i, sbase+t] = A0[i, jorig]
            end
        end
        @inbounds for t in 1:nwarm
            nsetp >= m && break # all rows triangularized; remaining seeds cannot enter
            jorig = work.hpos[t] # slot t is intact: only reflections 1:nsetp (< t) have been written
            jmax = invidx[jorig] # invidx tracks positions during seeding (callers initialize it to the identity)
            jmax <= nsetp && continue # duplicate seed; already entered
            @simd ivdep for i in 1:m
                zz[i] = H[i, sbase+t]
            end
            hl0 = hlen[]
            tau = try_enter_column!(work, zz, nsetp + 1, m, false, false) # staged column is already reduced against Q
            tau == -2 && break # transforms full; continue with what we have
            tau < 0 && continue # numerically dependent; skip
            if hlen[] > hl0 # keep the still-staged seeds reduced against the new reflection
                for t2 in t+1:nwarm
                    apply_householder_to_col!(H, sbase + t2, work, hlen[])
                end
            end
            nsetp += 1
            idx[nsetp], idx[jmax] = idx[jmax], idx[nsetp]
            invidx[idx[nsetp]] = nsetp
            invidx[idx[jmax]] = jmax
            w[nsetp] = zero(T)
            use_stale_w = false # any preloaded dual is stale once the passive set is nonempty
        end
    end

    # Feasibility pass: solve on the seeded set and drop columns (most negative coefficient first) until the passive-set solution is strictly positive
    @inbounds while nsetp > 0
        @simd for i in 1:nsetp
            zz[i] = b[i]
        end
        solve_triangular_system!(zz, A, nsetp, Val(false))
        imv = 0
        zmin = zero(T)
        for i in 1:nsetp
            if zz[i] <= zmin
                imv, zmin = i, zz[i]
            end
        end
        if imv == 0 # all strictly positive
            for i in 1:nsetp
                x[idx[i]] = zz[i]
            end
            break
        end
        iter += 1
        if iter > max_iter
            work.mode[] = 1
            terminated = true
            break
        end
        x[idx[imv]] = zero(T)
        nsetp = downdate!(work, imv, nsetp, m)
    end

    # ******  MAIN LOOP BEGINS HERE  ******
    @inbounds while !terminated
        # QUIT IF ALL COEFFICIENTS ARE ALREADY IN THE SOLUTION.
        # OR IF M COLS OF A HAVE BEEN TRIANGULARIZED.
        if (nsetp >= n || nsetp >= m)
            terminated = true
            break
        end

        # COMPUTE COMPONENTS OF THE DUAL (NEGATIVE GRADIENT) VECTOR W() from the pristine set Z columns and the current passive-set residual, unless the caller preloaded w for the first round.
        if use_stale_w
            use_stale_w = false
        else
            compute_dual!(work, A0, nsetp, m)
            rfresh = true
        end

        while true
            # FIND LARGEST POSITIVE W(J).
            wmax, jmax = largest_positive_dual(w, nsetp + 1)

            # IF WMAX .LE. 0. GO TO TERMINATION.
            # THIS INDICATES SATISFACTION OF THE KUHN-TUCKER CONDITIONS.
            if wmax <= 0
                terminated = true
                break
            end

            # THE SIGN OF W(J) IS OK FOR J TO BE MOVED TO SET P.
            # BEGIN THE TRANSFORMATION AND CHECK NEW DIAGONAL ELEMENT TO AVOID
            # NEAR LINEAR DEPENDENCE.
            jorig = idx[jmax]
            @simd ivdep for i in 1:m
                zz[i] = A0[i, jorig]
            end
            tau = try_enter_column!(work, zz, nsetp + 1, m)
            if tau == -2
                work.mode[] = 1 # transform capacity exhausted; pathological
                terminated = true
                break
            elseif tau < 0
                # REJECT J AS A CANDIDATE TO BE MOVED FROM SET Z TO SET P.
                # SET W(J)=0., AND LOOP BACK TO TEST DUAL COEFFS AGAIN.
                w[jmax] = zero(T)
                continue
            end

            # THE INDEX J=INDEX(IZ) HAS BEEN SELECTED TO BE MOVED FROM
            # SET Z TO SET P. UPDATE INDICES AND SET W(J)=0.
            # NOTE: B updated in `try_enter_column!`
            nsetp += 1
            idx[nsetp], idx[jmax] = idx[jmax], idx[nsetp]
            w[nsetp] = zero(T)
            rfresh = false # b was transformed and the passive set changed
            break
        end

        if terminated
            break
        end

        # SOLVE THE TRIANGULAR SYSTEM.
        # STORE THE SOLUTION TEMPORARILY IN ZZ().
        @simd for i in 1:nsetp
            zz[i] = b[i]
        end
        solve_triangular_system!(zz, A, nsetp, Val(false))

        # ******  SECONDARY LOOP BEGINS HERE  ******
        while true
            iter += 1
            if iter > max_iter
                work.mode[] = 1
                terminated = true
                break
            end

            # SEE IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE.
            # IF NOT COMPUTE ALPHA.
            imv = nsetp
            alpha = T(2)
            for i in 1:nsetp
                if zz[i] <= 0
                    xi = x[idx[i]]
                    t = -xi / (zz[i] - xi)
                    if alpha > t
                        imv = i
                        alpha = t
                    end
                end
            end

            # IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE THEN ALPHA WILL
            # STILL = 2. IF SO EXIT FROM SECONDARY LOOP TO MAIN LOOP.
            if alpha == 2
                break
            end

            # OTHERWISE USE ALPHA WHICH WILL BE BETWEEN 0 AND 1 TO
            # INTERPOLATE BETWEEN THE OLD X AND THE NEW ZZ.
            for i in 1:nsetp
                ix = idx[i]
                x[ix] = x[ix] + alpha * (zz[i] - x[ix])
            end

            # MODIFY A AND B AND THE INDEX ARRAYS TO MOVE COEFFICIENT I
            # FROM SET P TO SET Z.
            while true
                x[idx[imv]] = zero(T)
                nsetp = downdate!(work, imv, nsetp, m)

                # SEE IF THE REMAINING COEFFS IN SET P ARE FEASIBLE. THEY SHOULD
                # BE BECAUSE OF THE WAY ALPHA WAS DETERMINED.
                # IF ANY ARE INFEASIBLE IT IS DUE TO ROUND-OFF ERROR. ANY
                # THAT ARE NONPOSITIVE WILL BE SET TO ZERO
                # AND MOVED FROM SET P TO SET Z.
                allfeasible = true
                for i in 1:nsetp
                    if x[idx[i]] <= 0
                        allfeasible = false
                        imv = i
                        break
                    end
                end
                if allfeasible
                    break
                end
            end

            # COPY B( ) INTO ZZ( ). THEN SOLVE AGAIN AND LOOP BACK.
            @simd for i in 1:nsetp
                zz[i] = b[i]
            end
            solve_triangular_system!(zz, A, nsetp, Val(false))
        end

        if terminated
            break
        end
        # ******  END OF SECONDARY LOOP  ******

        for i in 1:nsetp
            x[idx[i]] = zz[i]
        end
        # ALL NEW COEFFS ARE POSITIVE. LOOP BACK TO BEGINNING.
    end

    # ******  END OF MAIN LOOP  ******

    # Compute inverse permutation
    @inbounds for i in 1:n
        invidx[idx[i]] = i
    end

    # zz doubles as the candidate column buffer, so restore the passive-set solution into zz[1:nsetp]; this is bit-exact, since x was assigned from these values.
    @inbounds for i in 1:nsetp
        zz[i] = x[idx[i]]
    end
    if nsetp < m
        @inbounds @simd ivdep for i in nsetp+1:m
            zz[i] = b[i]
        end
    else
        fill!(w, zero(T))
    end

    # Compute the norm of the final residual from the untransformed residual r = b0 - A0[:, idx[1:nsetp]]*x₊.
    # The transformed free rows of b lose relative accuracy when a nearly dependent column enters, for instance a duplicated column, whereas r is accurate to working precision.
    # r is already current when termination followed the dual computation, and is recomputed otherwise. When all rows are triangularized the residual is zero by convention, matching the classic algorithm's empty free-row sum.
    sm = zero(T)
    if nsetp < m
        if !rfresh
            @inbounds @simd ivdep for i in 1:m
                r[i] = b0[i]
            end
            @inbounds for t in 1:nsetp
                xt = zz[t]
                jt = idx[t]
                @simd ivdep for i in 1:m
                    r[i] = r[i] - xt * A0[i, jt]
                end
            end
        end
        @inbounds @simd for i in 1:m
            sm = sm + r[i] * r[i]
        end
    else
        @inbounds @simd ivdep for i in 1:m # the residual is zero by convention, so leave `r` current for `residual`
            r[i] = zero(T)
        end
    end

    work.rnorm[] = sqrt(sm)
    work.nsetp[] = nsetp
    work.solved[] = true
    return work.x
end

function unsafe_nnls!(
    work::NNLSWorkspace{T},
    A0::AbstractMatrix{T}, # pristine problem data, read-only, indexed by original column (only the m0 data rows are read)
    λ::T;
    init_dual::Bool = true,
    max_iter::Int = 3 * size(work.A, 2),
    nwarm::Int = 0, # number of warm-start columns stashed in work.hpos[1:nwarm]
) where {T}
    (; A, b, x, w, zz, idx, invidx, diag, b0, r, H, hpos, hlen, gc, gs, gi, transforms) = work
    M, N = size(A)
    m, n = M - N, N
    m0 = m # number of data rows; rows m0+1:M are the (implicit) λI padding

    copyto!(b0, b) # b holds the untransformed right-hand side at entry (for every caller); snapshot it for residual/dual computations
    hlen[] = 0
    empty!(gc)
    empty!(gs)
    empty!(gi)
    empty!(transforms)

    nsetp = 0
    iter = 0
    work.mode[] = 0
    terminated = false
    use_stale_w = !init_dual # init_dual = false: caller preloaded w for the first pivot selection
    rfresh = false # r = b0 - A0[:, idx[1:nsetp]]*x₊ is current for the final passive set (see the rnorm epilogue)

    # ******  WARM START  ******
    # Seed the passive set with the stashed columns, skipping the positivity check; a feasibility pass below drops any that come out non-positive.
    # Panel-QR seeding (see the unregularized solver): seed columns are staged upfront and kept reduced column-parallel as each accepted seed's reflection is constructed, replacing the per-seed sequential log replay.
    # λ-entry placement: a pre-activated λ row, rj != 0, lies within the range of earlier reflections and must therefore be placed at staging time.
    # A fresh λ row is assigned at entry time, row m + 1, and lies strictly above the rows of every reflection built so far, so lazy placement commutes with the reduction.
    if nwarm > 0
        hcap = size(H, 2)
        sbase = hcap - nwarm # staging columns sbase+1:sbase+nwarm; reflections built during seeding occupy slots 1:nwarm <= sbase
        @inbounds for t in 1:nwarm # stage before any reflection exists (hpos[1:nwarm] still holds the stashed seeds)
            jorig = work.hpos[t]
            @simd ivdep for i in 1:m0
                H[i, sbase+t] = A0[i, jorig]
            end
            @simd ivdep for i in m0+1:M # padded rows participate in the panel reduction; clear stale data
                H[i, sbase+t] = zero(T)
            end
            rj = diag[jorig]
            if rj != 0
                H[rj, sbase+t] = λ
            end
        end
        @inbounds for t in 1:nwarm
            jorig = work.hpos[t] # slot t is intact: only reflections 1:nsetp (< t) have been written
            jmax = invidx[jorig] # invidx tracks positions during seeding (init_nnls! set it to the identity)
            jmax <= nsetp && continue # duplicate seed; already entered
            m1 = min(m + 1, M)
            @simd ivdep for i in 1:m1
                zz[i] = H[i, sbase+t]
            end
            if diag[jorig] == 0
                zz[m1] = λ # fresh λ row: untouched by every reflection built so far
            end
            hl0 = hlen[]
            tau = try_enter_column!(work, zz, nsetp + 1, m1, false, false) # staged column is already reduced against Q
            tau == -2 && break # transforms full; continue with what we have
            tau < 0 && continue # numerically dependent; skip
            if hlen[] > hl0 # keep the still-staged seeds reduced against the new reflection
                for t2 in t+1:nwarm
                    apply_householder_to_col!(H, sbase + t2, work, hlen[])
                end
            end
            if diag[idx[jmax]] == 0
                m += 1
                diag[idx[jmax]] = m
            end
            nsetp += 1
            idx[nsetp], idx[jmax] = idx[jmax], idx[nsetp]
            invidx[idx[nsetp]] = nsetp
            invidx[idx[jmax]] = jmax
            w[nsetp] = zero(T)
        end
    end

    # Feasibility pass: solve on the seeded set and drop columns (most negative coefficient first) until the passive-set solution is strictly positive
    @inbounds while nsetp > 0
        @simd for i in 1:nsetp
            zz[i] = b[i]
        end
        solve_triangular_system!(zz, A, nsetp, Val(false))
        imv = 0
        zmin = zero(T)
        for i in 1:nsetp
            if zz[i] <= zmin
                imv, zmin = i, zz[i]
            end
        end
        if imv == 0 # all strictly positive
            for i in 1:nsetp
                x[idx[i]] = zz[i]
            end
            break
        end
        iter += 1
        if iter > max_iter
            work.mode[] = 1
            terminated = true
            break
        end
        x[idx[imv]] = zero(T)
        nsetp = downdate!(work, imv, nsetp, m)
    end

    # ******  MAIN LOOP BEGINS HERE  ******
    @inbounds while !terminated
        # QUIT IF ALL COEFFICIENTS ARE ALREADY IN THE SOLUTION.
        if nsetp >= n
            terminated = true
            break
        end

        # COMPUTE COMPONENTS OF THE DUAL (NEGATIVE GRADIENT) VECTOR W() from the pristine set Z columns and the current passive-set residual.
        # Only the data rows contribute: a set Z column's λ row is inactive or its coefficient is zero, so the padded rows drop out.
        if use_stale_w # caller preloaded w for the first round
            use_stale_w = false
        elseif iszero(λ) && nsetp >= m0
            # The passive set spans the data rows and the padding is zero, so the dual vanishes identically.
            # In the transformed formulation the free rows are all exactly-zero padded rows; enforce the same exact zero here rather than computing residual roundoff noise.
            @simd ivdep for j in nsetp+1:n
                w[j] = zero(T)
            end
        else
            compute_dual!(work, A0, nsetp, m0)
            rfresh = true
        end

        while true
            # FIND LARGEST POSITIVE W(J).
            wmax, jmax = largest_positive_dual(w, nsetp + 1)

            # IF WMAX .LE. 0. GO TO TERMINATION.
            # THIS INDICATES SATISFACTION OF THE KUHN-TUCKER CONDITIONS.
            if wmax <= 0
                terminated = true
                break
            end

            # THE SIGN OF W(J) IS OK FOR J TO BE MOVED TO SET P.
            # BEGIN THE TRANSFORMATION AND CHECK NEW DIAGONAL ELEMENT TO AVOID
            # NEAR LINEAR DEPENDENCE.
            # Stage the candidate in original row coordinates and place its λ entry.
            # A pre-activated λ row lies within the range of earlier reflections, so it must be placed before the reduction;
            # a fresh λ row is assigned at row m + 1, strictly above the rows of every reflection built so far, so placing it here commutes with the reduction.
            jorig = idx[jmax]
            m1 = min(m + 1, M)
            @simd ivdep for i in 1:m0
                zz[i] = A0[i, jorig]
            end
            @simd ivdep for i in (m0+1):m1
                zz[i] = zero(T)
            end
            rj = diag[jorig]
            zz[rj > 0 ? rj : m1] = λ

            tau = try_enter_column!(work, zz, nsetp + 1, m1)
            if tau == -2
                work.mode[] = 1 # transform capacity exhausted; pathological
                terminated = true
                break
            elseif tau < 0
                # REJECT J AS A CANDIDATE TO BE MOVED FROM SET Z TO SET P.
                # SET W(J)=0., AND LOOP BACK TO TEST DUAL COEFFS AGAIN.
                w[jmax] = zero(T)
                continue
            end

            # THE INDEX J=INDEX(IZ) HAS BEEN SELECTED TO BE MOVED FROM
            # SET Z TO SET P. UPDATE INDICES AND SET W(J)=0.
            # NOTE: B updated in `try_enter_column!`
            if diag[jorig] == 0
                m += 1 # m < M guaranteed, since each column activates at most one padded row
                diag[jorig] = m
            end
            nsetp += 1
            idx[nsetp], idx[jmax] = idx[jmax], idx[nsetp]
            w[nsetp] = zero(T)
            rfresh = false # b was transformed and the passive set changed
            break
        end

        if terminated
            break
        end

        # SOLVE THE TRIANGULAR SYSTEM.
        # STORE THE SOLUTION TEMPORARILY IN ZZ().
        @simd for i in 1:nsetp
            zz[i] = b[i]
        end
        solve_triangular_system!(zz, A, nsetp, Val(false))

        # ******  SECONDARY LOOP BEGINS HERE  ******
        while true
            iter += 1
            if iter > max_iter
                work.mode[] = 1
                terminated = true
                break
            end

            # SEE IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE.
            # IF NOT COMPUTE ALPHA.
            imv = nsetp
            alpha = T(2)
            for i in 1:nsetp
                if zz[i] <= 0
                    xi = x[idx[i]]
                    t = -xi / (zz[i] - xi)
                    if alpha > t
                        imv = i
                        alpha = t
                    end
                end
            end

            # IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE THEN ALPHA WILL
            # STILL = 2. IF SO EXIT FROM SECONDARY LOOP TO MAIN LOOP.
            if alpha == 2
                break
            end

            # OTHERWISE USE ALPHA WHICH WILL BE BETWEEN 0 AND 1 TO
            # INTERPOLATE BETWEEN THE OLD X AND THE NEW ZZ.
            for i in 1:nsetp
                ix = idx[i]
                x[ix] = x[ix] + alpha * (zz[i] - x[ix])
            end

            # MODIFY A AND B AND THE INDEX ARRAYS TO MOVE COEFFICIENT I
            # FROM SET P TO SET Z.
            while true
                x[idx[imv]] = zero(T)
                nsetp = downdate!(work, imv, nsetp, m)

                # SEE IF THE REMAINING COEFFS IN SET P ARE FEASIBLE. THEY SHOULD
                # BE BECAUSE OF THE WAY ALPHA WAS DETERMINED.
                # IF ANY ARE INFEASIBLE IT IS DUE TO ROUND-OFF ERROR. ANY
                # THAT ARE NONPOSITIVE WILL BE SET TO ZERO
                # AND MOVED FROM SET P TO SET Z.
                allfeasible = true
                for i in 1:nsetp
                    if x[idx[i]] <= 0
                        allfeasible = false
                        imv = i
                        break
                    end
                end
                if allfeasible
                    break
                end
            end

            # COPY B( ) INTO ZZ( ). THEN SOLVE AGAIN AND LOOP BACK.
            @simd for i in 1:nsetp
                zz[i] = b[i]
            end
            solve_triangular_system!(zz, A, nsetp, Val(false))
        end

        if terminated
            break
        end
        # ******  END OF SECONDARY LOOP  ******

        for i in 1:nsetp
            x[idx[i]] = zz[i]
        end
        # ALL NEW COEFFS ARE POSITIVE. LOOP BACK TO BEGINNING.
    end

    # ******  END OF MAIN LOOP  ******

    # Compute inverse permutation
    @inbounds for i in 1:n
        invidx[idx[i]] = i
    end

    # zz doubles as the candidate column buffer, so restore the passive-set solution into zz[1:nsetp]; this is bit-exact, since x was assigned from these values.
    @inbounds for i in 1:nsetp
        zz[i] = x[idx[i]]
    end
    if nsetp < M
        @inbounds @simd ivdep for i in nsetp+1:M
            zz[i] = b[i]
        end
    else
        fill!(w, zero(T))
    end

    # Compute the norm of the final residual from the untransformed residual:
    #   ||[A₀; λI]x - [b₀; 0]||² = ||r||² + λ²||x₊||²,   r = b0 - A0[:, idx[1:nsetp]]*x₊.
    # For λ = 0 with the data rows exhausted the residual is zero by convention, matching the classic algorithm's exactly-zero free rows.
    sm = zero(T)
    if !(iszero(λ) && nsetp >= m0)
        if !rfresh
            @inbounds @simd ivdep for i in 1:m0
                r[i] = b0[i]
            end
            @inbounds for t in 1:nsetp
                xt = zz[t]
                jt = idx[t]
                @simd ivdep for i in 1:m0
                    r[i] = r[i] - xt * A0[i, jt]
                end
            end
        end
        @inbounds @simd for i in 1:m0
            sm = sm + r[i] * r[i]
        end
        λ² = λ * λ
        @inbounds @simd for i in 1:nsetp
            sm = sm + λ² * zz[i] * zz[i]
        end
    else
        @inbounds @simd ivdep for i in 1:m0 # the residual is zero by convention, so leave `r` current for `residual`
            r[i] = zero(T)
        end
    end

    work.rnorm[] = sqrt(sm)
    work.nsetp[] = nsetp
    work.solved[] = true
    return work.x
end

####
#### Gram matrix-based fast path for the Tikhonov-regularized problem
####

# The Gram matrix of the active columns is μ-independent, so evaluating the Tikhonov-regularized NNLS residual norm at a new μ with a warm active set costs one Cholesky factorization over the p active columns plus one KKT-verification GEMV, rather than a full QR rebuild.
# The KKT conditions are verified against the exact residual dual w = A'(b - A_P x), so accepted solutions are genuine NNLS solutions up to normal-equation roundoff. The κ(A_P)² amplification is tamed by the μ² shift of the Gram matrix, and the root tolerances are many orders coarser than the error that remains.
# Non-positive or tiny Cholesky pivots and an exhausted iteration budget both return NaN, on which the caller falls back to the exact QR solver.
struct NNLSGram{T}
    P::Vector{Int}           # active set (original column indices) in P[1:np]
    inP::Vector{Bool}        # membership mask of P[1:np]
    np::Base.RefValue{Int}   # active-set size
    GP::Matrix{T}            # n×n buffer; GP[1:np, 1:np] = Gram of the active columns (both triangles)
    L::Matrix{T}             # n×n buffer; upper Cholesky factor of GP + μ²I
    c::Vector{T}             # A'b
    cscale::Base.RefValue{T} # maximum(abs, c); scale for the dual tolerance
    xp::Vector{T}            # solution on the active set (length n buffer)
    r::Vector{T}             # residual b - A_P x (length m)
    w::Vector{T}             # dual A'r (length n)
    dinv::Vector{T}          # reciprocals of the Cholesky diagonal, so the O(p²) column updates multiply instead of divide
    y::Vector{T}             # scratch for the triangular solve in `inv_quadratic_form` (length n)
end

function NNLSGram(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    return NNLSGram(
        zeros(Int, n), fill(false, n), Ref(0),
        zeros(T, n, n), zeros(T, n, n),
        zeros(T, n), Ref(zero(T)),
        zeros(T, n), zeros(T, m), zeros(T, n), zeros(T, n), zeros(T, n),
    )
end

# Load the μ-independent right-hand side data c = A'b, once per problem
function load!(gp::NNLSGram{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    (; c) = gp
    m, n = size(A)
    cmax = zero(T)
    @inbounds for j in 1:n
        s = zero(T)
        @simd for i in 1:m
            s = muladd(A[i, j], b[i], s)
        end
        c[j] = s
        cmax = max(cmax, abs(s))
    end
    gp.cscale[] = cmax
    return gp
end

# Set the active set to idx0[1:np0] and (re)build its Gram block
function set_active!(gp::NNLSGram{T}, A::AbstractMatrix{T}, idx0::AbstractVector{Int}, np0::Int) where {T}
    (; P, inP, GP) = gp
    m = size(A, 1)
    fill!(inP, false)
    @inbounds for t in 1:np0
        j = idx0[t]
        P[t] = j
        inP[j] = true
    end
    @inbounds for t in 1:np0, s in 1:t
        g = coldot(A, P[s], P[t], m)
        GP[s, t] = g
        GP[t, s] = g
    end
    gp.np[] = np0
    return gp
end

@inline function add_active!(gp::NNLSGram{T}, A::AbstractMatrix{T}, j::Int) where {T}
    (; P, inP, GP, np) = gp
    m = size(A, 1)
    p = np[] + 1
    @inbounds begin
        P[p] = j
        inP[j] = true
        for s in 1:p
            g = coldot(A, P[s], j, m)
            GP[s, p] = g
            GP[p, s] = g
        end
    end
    np[] = p
    return gp
end

@inline function remove_active!(gp::NNLSGram, i::Int)
    (; P, inP, GP, np) = gp
    p = np[]
    @inbounds begin
        inP[P[i]] = false
        if i < p # swap-remove row/column i with the last
            P[i] = P[p]
            for s in 1:p
                GP[s, i] = GP[s, p]
            end
            for s in 1:p
                GP[i, s] = GP[p, s]
            end
            GP[i, i] = GP[p, p]
        end
    end
    np[] = p - 1
    return gp
end

@inline function coldot(A::AbstractMatrix{T}, ji::Int, jj::Int, m::Int) where {T}
    s = zero(T)
    @inbounds @simd for i in 1:m
        s = muladd(A[i, ji], A[i, jj], s)
    end
    return s
end

# ||x(μ)||² from the Gram path's active-set solution, valid immediately after a successful `solve!`, which leaves xp[1:np] holding the coefficients on P[1:np]
@inline function seminorm_sq(gp::NNLSGram{T}) where {T}
    s = zero(T)
    @inbounds @simd for i in 1:gp.np[]
        s = muladd(gp.xp[i], gp.xp[i], s)
    end
    return s
end

# xᵀB⁻¹x = ||L⁻ᵀx||² with B = A_PᵀA_P + μ²I = L'L, the quadratic form carrying the first-order geometry of the Tikhonov path: dR/dμ = 4μ³xᵀB⁻¹x and dN/dμ = -4μxᵀB⁻¹x.
# Valid immediately after a successful `solve!`, which leaves L[1:np, 1:np] current for the returned active set.
@inline function inv_quadratic_form(gp::NNLSGram{T}) where {T}
    (; L, xp, dinv, y, np) = gp
    s = zero(T)
    @inbounds for i in 1:np[]
        t = xp[i]
        @simd for k in 1:i-1
            t = t - L[k, i] * y[k]
        end
        yi = t * dinv[i]
        y[i] = yi
        s = muladd(yi, yi, s)
    end
    return s
end

# Digest of the active set, exclusive-or'd so that it does not depend on the order the columns are stored in. Equal digests mean equal sets up to a 2⁻¹²⁸ collision, possible only among columns past 128, the only ones without a bit of their own.
# Equal digests do not mean the set is constant between the two μ: a component can leave and re-enter, each passive coefficient being rational in μ² with several possible positive roots.
@inline function splitmix(x::UInt64)
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end
@inline column_digest(j::Int) = j <= 128 ? UInt128(1) << (j - 1) : UInt128(splitmix(UInt64(j))) << 64 | UInt128(splitmix(~UInt64(j)))

@inline function active_signature(gp::NNLSGram)
    s = zero(UInt128)
    @inbounds for i in 1:gp.np[]
        s ⊻= column_digest(gp.P[i])
    end
    return s
end

# Solve min ||Ax - b||² + μ²||x||² s.t. x ≥ 0 via active-set iteration on the cached Gram data, warm-started from the current active set.
# Returns the squared data residual ||Ax - b||², excluding the μ²||x||² penalty so as to match `resnorm_sq`, or NaN on failure, on which the caller falls back to the exact solver.
# On success the active set is left at the solution, warm-starting the next μ.
function solve!(gp::NNLSGram{T}, A::AbstractMatrix{T}, b::AbstractVector{T}, μ::T) where {T}
    (; P, inP, np, GP, L, c, xp, r, w, dinv) = gp
    m, n = size(A)
    μ² = μ * μ

    # Dual tolerance for accepting a KKT point, relative to the scale of A'b.
    # The objective is flat along the degenerate directions of near-collinear columns, so a dual violation of size δ admits an active set whose
    # objective is suboptimal only by O(δ²) but whose ‖Ax-b‖² and ‖x‖² individually move by O(δ).
    # Callers read those two separately and amplify the error: a transversal root-find by O(δ), a smooth minimization by O(√δ), a curvature maximization by more still.
    wtol = eps(T)^(3//4) * gp.cscale[]

    # Cholesky pivot guard: a length-p accumulated dot product carries relative error O(p·eps), so the threshold scales with the problem size.
    ϵpiv = 10 * n * eps(T)
    maxiter = 2 * n + 10
    iters = 0
    lvalid = 0 # leading columns of L that are current for this μ (adds append a column; drops invalidate from the removed column)

    @inbounds while true
        iters += 1
        iters > maxiter && return T(NaN)
        p = np[]

        # Upper Cholesky L'L = GP[1:p, 1:p] + μ²I, with a conditioning guard on the pivots.
        # Left-looking column-oriented: column jcol is computed from GP[:, jcol] and the previous columns of L only,
        # so recomputation can resume from the first column invalidated by an active-set change, μ being fixed for the duration of this call
        jcol = lvalid + 1
        while jcol <= p
            if jcol + 1 <= p
                # Two columns at a time: each pass over L[:, k] feeds both accumulators, halving the loads of the O(p³) update
                @simd ivdep for k in 1:jcol-1
                    L[k, jcol] = GP[k, jcol]
                    L[k, jcol+1] = GP[k, jcol+1]
                end

                for k in 1:jcol-1
                    a = L[k, jcol]
                    b2 = L[k, jcol+1]
                    @simd for k2 in 1:k-1
                        lk = L[k2, k]
                        a = a - lk * L[k2, jcol]
                        b2 = b2 - lk * L[k2, jcol+1]
                    end
                    dk = dinv[k]
                    L[k, jcol] = a * dk
                    L[k, jcol+1] = b2 * dk
                end

                s = GP[jcol, jcol] + μ²
                @simd for k in 1:jcol-1
                    s = s - L[k, jcol] * L[k, jcol]
                end

                s <= ϵpiv * (GP[jcol, jcol] + μ²) && return T(NaN)
                Ljj = sqrt(s)
                L[jcol, jcol] = Ljj
                dinv[jcol] = inv(Ljj)
                s2 = GP[jcol, jcol+1]
                @simd for k2 in 1:jcol-1
                    s2 = s2 - L[k2, jcol] * L[k2, jcol+1]
                end

                L[jcol, jcol+1] = s2 * dinv[jcol]
                s = GP[jcol+1, jcol+1] + μ²
                @simd for k in 1:jcol
                    s = s - L[k, jcol+1] * L[k, jcol+1]
                end

                s <= ϵpiv * (GP[jcol+1, jcol+1] + μ²) && return T(NaN)
                L[jcol+1, jcol+1] = sqrt(s)
                dinv[jcol+1] = inv(L[jcol+1, jcol+1])
                jcol += 2
            else
                for k in 1:jcol-1
                    s2 = GP[k, jcol]
                    @simd for k2 in 1:k-1
                        s2 = s2 - L[k2, k] * L[k2, jcol]
                    end
                    L[k, jcol] = s2 * dinv[k]
                end

                s = GP[jcol, jcol] + μ²
                @simd for k in 1:jcol-1
                    s = s - L[k, jcol] * L[k, jcol]
                end

                s <= ϵpiv * (GP[jcol, jcol] + μ²) && return T(NaN)
                L[jcol, jcol] = sqrt(s)
                dinv[jcol] = inv(L[jcol, jcol])
                jcol += 1
            end
        end
        lvalid = p

        # xp = (GP + μ²I) \ c_P via forward/back substitution
        for i in 1:p
            s = c[P[i]]
            @simd for k in 1:i-1
                s = s - L[k, i] * xp[k]
            end
            xp[i] = s * dinv[i]
        end

        for i in p:-1:1
            s = xp[i]
            @simd for k in i+1:p
                s = s - L[i, k] * xp[k]
            end
            xp[i] = s * dinv[i]
        end

        # Feasibility: drop the most negative coefficient, if any
        imv, xmin = 0, zero(T)
        for i in 1:p
            if xp[i] <= xmin
                imv, xmin = i, xp[i]
            end
        end

        if imv > 0
            remove_active!(gp, imv)
            lvalid = min(lvalid, imv - 1) # swap-remove invalidates the factor from column imv on
            continue
        end

        # Residual r = b - A_P x and squared objective
        @simd ivdep for i in 1:m
            r[i] = b[i]
        end

        res² = zero(T)
        tt = 1
        while tt + 1 <= p # two active columns per pass, halving the loads and stores of r
            x1, x2 = xp[tt], xp[tt+1]
            j1, j2 = P[tt], P[tt+1]
            @simd ivdep for i in 1:m
                r[i] = r[i] - x1 * A[i, j1] - x2 * A[i, j2]
            end
            tt += 2
        end

        if tt <= p
            xt = xp[tt]
            jt = P[tt]
            @simd ivdep for i in 1:m
                r[i] = r[i] - xt * A[i, jt]
            end
        end

        @simd for i in 1:m
            res² = muladd(r[i], r[i], res²)
        end

        # KKT dual w = A'r, computed only for the inactive columns, which are the only ones the scan below can select.
        # The active set is typically most of the basis, so this is far less work than a full A'r.
        @simd for jj in 1:n
            w[jj] = zero(T)
        end

        for jj in 1:n
            inP[jj] && continue
            s = zero(T)
            @simd for i in 1:m
                s = muladd(A[i, jj], r[i], s)
            end
            w[jj] = s
        end

        # w_Z must be non-positive (within tolerance)
        wmax, jmax = wtol, 0
        for j in 1:n
            wj = w[j]
            if wj > wmax && !inP[j]
                wmax, jmax = wj, j
            end
        end

        if jmax > 0
            # Enter the worst violator plus up to 3 further strong violators in one batch. An add costs a few Gram dots, whereas every feasibility round costs a residual build and a dual GEMV, and the feasibility drops above absorb any overshoot.
            p >= min(m, n) && return T(NaN) # cannot grow the active set further; fall back
            add_active!(gp, A, jmax)
            wthresh = max(wtol, T(0.2) * wmax)
            for _ in 1:3
                np[] >= min(m, n) && break
                wbest, jbest = wthresh, 0
                for j in 1:n
                    wj = w[j]
                    if wj > wbest && !inP[j]
                        wbest, jbest = wj, j
                    end
                end
                jbest == 0 && break
                add_active!(gp, A, jbest)
            end
            continue
        end

        return res²
    end
end

####
#### Gram matrix-based fast path for a fixed grid of unregularized problems
####

# Precomputed-Gram fast path for unregularized NNLS solves against a fixed set of bases.
# The bases depend only on the grid parameters, never on the signal b, so the Gram matrices Gᵢ = AᵢᵀAᵢ come from shared read-only storage supplied by the caller and this workspace holds only the mutable solve scratch.
# Each evaluation costs one GEMV c = Aᵢᵀb plus an active-set iteration entirely in Gram space: a resumable p×p Cholesky of G_P, feasibility drops, and the KKT dual w = c − G[:, P]x_P at n·p flops, replacing the m·n dual GEMV per round of the QR solver.
# Unlike `NNLSGram` there is no μ² diagonal shift to condition the normal equations, so a tiny or non-positive Cholesky pivot, a full active set, or an exhausted iteration budget returns `false` and the caller falls back to the exact QR solver.
struct NNLSGridGram{T}
    c::Vector{T}             # Aᵢᵀb for the current evaluation
    cscale::Base.RefValue{T} # maximum(abs, c); scale for the dual tolerance
    P::Vector{Int}           # active set (original column indices) in P[1:np]
    inP::Vector{Bool}        # membership mask of P[1:np]
    rejected::Vector{Bool}   # candidates rejected for the remainder of the current solve; see `solve!`
    np::Base.RefValue{Int}   # active-set size
    GP::Matrix{T}            # gathered active-set Gram block (both triangles)
    L::Matrix{T}             # upper Cholesky factor of GP
    dinv::Vector{T}          # reciprocals of the Cholesky diagonal, so the O(p²) column updates and the triangular solves multiply instead of divide
    xp::Vector{T}            # unconstrained-on-P solution of the current round
    xcur::Vector{T}          # feasible iterate aligned with P (Lawson-Hanson secondary-loop interpolation state)
    u::Vector{T}             # bordering scratch: u = L'⁻¹G_{P,j} for the entering-column pre-check / factor extension
    w::Vector{T}             # dual c − G[:, P]x_P
end

function NNLSGridGram(::Type{T}, n::Int) where {T}
    return NNLSGridGram(
        zeros(T, n), Ref(zero(T)), zeros(Int, n), fill(false, n), fill(false, n), Ref(0),
        zeros(T, n, n), zeros(T, n, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),
    )
end

# Load the signal-dependent right-hand side data c = Aᵀb, once per grid point per signal.
# Columns are processed in blocks of 4 sharing each load of b, cf. `compute_dual_block!`.
function load!(gp::NNLSGridGram{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    (; c) = gp
    m, n = size(A)
    j = 1
    @inbounds while j + 3 <= n
        s1 = s2 = s3 = s4 = zero(T)
        @simd for i in 1:m
            bi = b[i]
            s1 = muladd(A[i, j], bi, s1)
            s2 = muladd(A[i, j+1], bi, s2)
            s3 = muladd(A[i, j+2], bi, s3)
            s4 = muladd(A[i, j+3], bi, s4)
        end
        c[j], c[j+1], c[j+2], c[j+3] = s1, s2, s3, s4
        j += 4
    end
    @inbounds while j <= n
        s = zero(T)
        @simd for i in 1:m
            s = muladd(A[i, j], b[i], s)
        end
        c[j] = s
        j += 1
    end
    cmax = zero(T)
    @inbounds @simd for j in 1:n
        cmax = max(cmax, abs(c[j]))
    end
    gp.cscale[] = cmax
    return gp
end

# Set the active set to idx0[1:np0] and gather its Gram block
function set_active!(gp::NNLSGridGram{T}, G::AbstractMatrix{T}, idx0, np0::Int) where {T}
    (; P, inP, rejected, GP) = gp
    fill!(inP, false)
    fill!(rejected, false)
    @inbounds for t in 1:np0
        j = idx0[t]
        P[t] = j
        inP[j] = true
    end
    @inbounds for t in 1:np0, s in 1:t
        g = G[P[s], P[t]]
        GP[s, t] = g
        GP[t, s] = g
    end
    gp.np[] = np0
    return gp
end

@inline function add_active!(gp::NNLSGridGram{T}, G::AbstractMatrix{T}, j::Int) where {T}
    (; P, inP, GP, np) = gp
    p = np[] + 1
    @inbounds begin
        P[p] = j
        inP[j] = true
        @simd for s in 1:p
            g = G[P[s], j]
            GP[s, p] = g
            GP[p, s] = g
        end
    end
    np[] = p
    return gp
end

@inline function remove_active!(gp::NNLSGridGram, i::Int)
    (; P, inP, GP, np, xcur) = gp
    p = np[]
    @inbounds begin
        inP[P[i]] = false
        if i < p # swap-remove row/column i with the last (xcur rides along with P)
            P[i] = P[p]
            xcur[i] = xcur[p]
            @simd ivdep for s in 1:p
                GP[s, i] = GP[s, p]
            end
            @simd ivdep for s in 1:p
                GP[i, s] = GP[p, s]
            end
            GP[i, i] = GP[p, p]
        end
    end
    np[] = p - 1
    return gp
end

# Solve min ||Ax - b||² s.t. x ≥ 0 on the cached Gram data of one grid point, warm-started from the current active set.
# This is the Lawson-Hanson iteration in Gram (normal-equations) form, cf. FNNLS (Bro & de Jong 1997):
#   - entering candidates pass a bordering pre-check: with L'L = G_P and u = L'⁻¹G_{P,j}, the Schur complement d = Gⱼⱼ − ‖u‖² must exceed ϵ·Gⱼⱼ for numerical independence, the proposed new coefficient being wⱼ/d > 0. Accepting extends the Cholesky factor in place with u, so adds never refactorize;
#   - infeasible solves run the classic secondary-loop interpolation toward the last feasible iterate, so the objective strictly decreases and add→drop cycling cannot occur;
#   - candidates that fail the pre-check or come out infeasible immediately after entry stay rejected for the remainder of the solve, their attainable improvement being at the dual tolerance level. This mirrors the exact solver's `w[pos] = 0` handling.
# Returns `true` with the solution in xp[1:np] on P[1:np], or `false` when a conditioning or iteration guard trips, on which the caller falls back to the exact QR solver.
function solve!(gp::NNLSGridGram{T}, G::AbstractMatrix{T}, m::Int) where {T}
    (; c, P, inP, rejected, np, GP, L, dinv, xp, xcur, u, w) = gp
    n = size(G, 1)

    # Dual tolerance for accepting a KKT point, relative to the scale of Aᵀb.
    # The objective is flat along the degenerate directions of near-collinear columns, so a dual violation of size δ admits an active set whose
    # objective is suboptimal only by O(δ²) but whose loss ‖A x − b‖² moves by O(δ), and the surrogate search reads that loss directly.
    wtol = eps(T)^(3//4) * gp.cscale[]

    # Cholesky pivot guard: a length-p accumulated dot product carries relative error O(p·eps), so the threshold scales with the problem size.
    ϵpiv = 10 * n * eps(T)
    maxiter = 3 * n
    iters = 0
    lvalid = 0 # leading columns of L that are current; accepted adds extend the factor, drops invalidate from the removed column
    feasible = false # xcur[1:np] holds a feasible iterate; false until the first all-positive solve, so the seed feasibility pass drops most-negative coefficients without interpolation
    jentered = 0 # column entered by the most recent accept; rejected if the immediately-following solve drops it
    @inbounds while true
        iters += 1
        iters > maxiter && return false
        p = np[]

        # Upper Cholesky L'L = GP[1:p, 1:p] with a conditioning guard on the pivots.
        # Left-looking column-oriented, so recomputation resumes from the first column invalidated by an active-set change, and two columns at a time, so each pass over L[:, k] feeds both accumulators and halves the loads of the O(p³) update.
        jcol = lvalid + 1
        while jcol <= p
            if jcol + 1 <= p
                @simd ivdep for k in 1:jcol-1
                    L[k, jcol] = GP[k, jcol]
                    L[k, jcol+1] = GP[k, jcol+1]
                end
                for k in 1:jcol-1
                    a1 = L[k, jcol]
                    a2 = L[k, jcol+1]
                    @simd for k2 in 1:k-1
                        lk = L[k2, k]
                        a1 = a1 - lk * L[k2, jcol]
                        a2 = a2 - lk * L[k2, jcol+1]
                    end
                    dk = dinv[k]
                    L[k, jcol] = a1 * dk
                    L[k, jcol+1] = a2 * dk
                end
                s = GP[jcol, jcol]
                @simd for k in 1:jcol-1
                    s = s - L[k, jcol] * L[k, jcol]
                end
                s <= ϵpiv * GP[jcol, jcol] && return false
                Ljj = sqrt(s)
                L[jcol, jcol] = Ljj
                dinv[jcol] = inv(Ljj)
                s2 = GP[jcol, jcol+1]
                @simd for k2 in 1:jcol-1
                    s2 = s2 - L[k2, jcol] * L[k2, jcol+1]
                end
                L[jcol, jcol+1] = s2 * dinv[jcol]
                s = GP[jcol+1, jcol+1]
                @simd for k in 1:jcol
                    s = s - L[k, jcol+1] * L[k, jcol+1]
                end
                s <= ϵpiv * GP[jcol+1, jcol+1] && return false
                Ljj = sqrt(s)
                L[jcol+1, jcol+1] = Ljj
                dinv[jcol+1] = inv(Ljj)
                jcol += 2
            else
                for k in 1:jcol-1
                    s2 = GP[k, jcol]
                    @simd for k2 in 1:k-1
                        s2 = s2 - L[k2, k] * L[k2, jcol]
                    end
                    L[k, jcol] = s2 * dinv[k]
                end
                s = GP[jcol, jcol]
                @simd for k in 1:jcol-1
                    s = s - L[k, jcol] * L[k, jcol]
                end
                s <= ϵpiv * GP[jcol, jcol] && return false
                Ljj = sqrt(s)
                L[jcol, jcol] = Ljj
                dinv[jcol] = inv(Ljj)
                jcol += 1
            end
        end
        lvalid = p

        # xp = GP \ c_P via forward/back substitution
        for i in 1:p
            s = c[P[i]]
            @simd for k in 1:i-1
                s = s - L[k, i] * xp[k]
            end
            xp[i] = s * dinv[i]
        end
        for i in p:-1:1
            s = xp[i]
            @simd for k in i+1:p
                s = s - L[i, k] * xp[k]
            end
            xp[i] = s * dinv[i]
        end

        # Feasibility
        imv, xmin = 0, zero(T)
        for i in 1:p
            if xp[i] <= xmin
                imv, xmin = i, xp[i]
            end
        end
        if imv > 0
            if feasible
                # Secondary loop: interpolate the feasible iterate toward xp until the first coefficient hits zero and drop that column
                α = one(T)
                for i in 1:p
                    if xp[i] <= 0
                        t = xcur[i] / (xcur[i] - xp[i])
                        if t < α
                            α, imv = t, i
                        end
                    end
                end
                for i in 1:p
                    xcur[i] = xcur[i] + α * (xp[i] - xcur[i])
                end
                xcur[imv] = zero(T)
                P[imv] == jentered && (rejected[jentered] = true) # the entering column came straight back out, so its dual is marginal at the tolerance boundary; reject it for the rest of the solve
            end
            jentered = 0
            remove_active!(gp, imv)
            lvalid = min(lvalid, imv - 1) # swap-remove invalidates the factor from column imv on
            continue
        end

        # All positive: xp is the new feasible iterate
        for i in 1:p
            xcur[i] = xp[i]
        end
        feasible = true
        jentered = 0

        # KKT dual w = c − G[:, P]x_P at n·p flops on contiguous Gram columns, two active columns per pass so each load of w feeds both
        @simd ivdep for j in 1:n
            w[j] = c[j]
        end
        t = 1
        while t + 1 <= p
            x1, x2 = xp[t], xp[t+1]
            j1, j2 = P[t], P[t+1]
            @simd ivdep for j in 1:n
                w[j] = w[j] - x1 * G[j, j1] - x2 * G[j, j2]
            end
            t += 2
        end
        if t <= p
            xt = xp[t]
            jt = P[t]
            @simd ivdep for j in 1:n
                w[j] = w[j] - xt * G[j, jt]
            end
        end

        # w_Z must be non-positive (within tolerance); enter the worst violator passing the bordering pre-check
        while true
            wmax, jmax = wtol, 0
            for j in 1:n
                wj = w[j]
                if wj > wmax && !inP[j] && !rejected[j]
                    wmax, jmax = wj, j
                end
            end
            jmax == 0 && return true # KKT satisfied, up to rejected tolerance-level candidates
            p >= min(m, n) && return false # cannot grow the active set further; fall back

            # Bordering pre-check + in-place factor extension: u = L'⁻¹G_{P,j}, d = Gⱼⱼ − ‖u‖²
            d = G[jmax, jmax]
            for i in 1:p
                s = G[P[i], jmax]
                @simd for k in 1:i-1
                    s = s - L[k, i] * u[k]
                end
                s *= dinv[i]
                u[i] = s
                d = d - s * s
            end
            if d <= ϵpiv * G[jmax, jmax] # numerically dependent on the active set
                rejected[jmax] = true
                continue
            end
            add_active!(gp, G, jmax)
            pnew = np[]
            @simd ivdep for i in 1:p
                L[i, pnew] = u[i]
            end
            Ljj = sqrt(d)
            L[pnew, pnew] = Ljj
            dinv[pnew] = inv(Ljj)
            lvalid = pnew
            xcur[pnew] = zero(T)
            jentered = jmax
            break
        end
    end
end

end # @muladd

end # module NNLS
