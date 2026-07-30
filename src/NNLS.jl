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
export NNLSWorkspace, NormalEquation, NormalEquationCholesky

@muladd begin

struct NNLSWorkspace{T}
    A::Matrix{T}               # factor storage; A[1:nsetp, 1:nsetp] holds the upper triangular factor of the passive columns
    b::Vector{T}               # transformed right-hand side Q'b
    x::Vector{T}               # solution, indexed by original column
    w::Vector{T}               # dual, indexed by position; see `dual` for the original-column view
    zz::Vector{T}              # trial solution of the current round, doubling as the candidate column buffer
    idx::Vector{Int}           # column permutation; idx[1:nsetp] is the passive set, idx[nsetp+1:n] the active set
    invidx::Vector{Int}        # inverse permutation of idx
    diag::Vector{Int}          # activation row of each column's λ row; 0 means not yet activated
    b0::Vector{T}              # original right-hand side, the reference for residual and dual computations
    r::Vector{T}               # residual buffer r = b0 - A0[:, idx[1:nsetp]] * x₊
    H::Matrix{T}               # append-only panel of scaled Householder vector tails
    htau::Vector{T}            # Householder scalar factors
    hpos::Vector{Int}          # pivot row of each stored Householder
    hm1::Vector{Int}           # last row of each stored Householder
    hlen::Base.RefValue{Int}   # number of stored Householders
    gc::Vector{T}              # Givens cosines
    gs::Vector{T}              # Givens sines
    gi::Vector{Int}            # Givens row indices; rotation g acts on rows (gi[g]-1, gi[g])
    transforms::Vector{Int}    # transform order: +t = Householder t, -g = Givens g
    rnorm::Base.RefValue{T}    # residual norm at the solution
    mode::Base.RefValue{Int}   # termination status; 0 on success
    nsetp::Base.RefValue{Int}  # passive-set size
end
@inline solution(work::NNLSWorkspace) = work.x
@inline dual(work::NNLSWorkspace) = @views work.w[work.invidx]
@inline residualnorm(work::NNLSWorkspace) = work.rnorm[]
@inline ncomponents(work::NNLSWorkspace) = work.nsetp[]
@inline components(work::NNLSWorkspace) = @views work.idx[1:ncomponents(work)]
@inline positive_solution(work::NNLSWorkspace) = @views solution(work)[components(work)]
@inline positive_solution!(work::NNLSWorkspace, x::AbstractVector) = copyto!(x, positive_solution(work))
@inline choleskyfactor(work::NNLSWorkspace, ::Val{:U}) = @views UpperTriangular(work.A[1:ncomponents(work), 1:ncomponents(work)])
@inline choleskyfactor(work::NNLSWorkspace, ::Val{:L}) = choleskyfactor(work, Val(:U))'

function Base.show(io::IO, ::MIME"text/plain", work::NNLSWorkspace)
    (; A, b, x, w, zz, idx, invidx, diag, rnorm, mode, nsetp) = work
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
    return nothing
end

function NNLSWorkspace(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    @assert size(b) == (m,)
    return NNLSWorkspace(T, m, n)
end

function NNLSWorkspace(::Type{T}, m::Int, n::Int) where {T}
    hcap = 2n + 8 # covers any realistic number of passive-set entries and re-entries
    return NNLSWorkspace(
        zeros(T, m, n), # A
        zeros(T, m),    # b
        zeros(T, n),    # x
        zeros(T, n),    # w
        zeros(T, m),    # zz
        zeros(Int, n),  # idx (Note: deliberately initialize to invalid permutation)
        zeros(Int, n),  # invidx
        zeros(Int, n),  # diag
        zeros(T, m),    # b0
        zeros(T, m),    # r
        zeros(T, m, hcap),    # H
        zeros(T, hcap),       # htau
        zeros(Int, hcap),     # hpos
        zeros(Int, hcap),     # hm1
        Ref(0),               # hlen
        sizehint!(T[], 4n),   # gc
        sizehint!(T[], 4n),   # gs
        sizehint!(Int[], 4n), # gi
        sizehint!(Int[], 4n), # transforms
        Ref(zero(T)),   # rnorm
        Ref(0),         # mode
        Ref(0),         # nsetp
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
    )
end

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
# For the unregularized problem m = size(A, 1), while for the Tikhonov-padded problem A = [A₀; λI] only the top m = M - N data rows contribute, since x = 0 on the padded rows.
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

"""
CONSTRUCTION AND/OR APPLICATION OF A SINGLE
HOUSEHOLDER TRANSFORMATION Q = I + U*(U**T)/B

The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 12, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.
"""
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

"""
CONSTRUCTION AND/OR APPLICATION OF A SINGLE
HOUSEHOLDER TRANSFORMATION Q = I + U*(U**T)/B

The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 12, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.
"""
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
# On input c (= work.zz, used as scratch) holds the candidate column in original row coordinates (rows 1:m1; the caller zeroes inactive padded rows and places any λ entry).
# The candidate is reduced against the current Q, then a Householder reflection is constructed on rows ip:m1
# and the classic entering-column checks are performed, namely sufficient independence and positivity of the proposed new coefficient.
# On acceptance the reflection is recorded and applied to b, and the new U column is written at position ip.
# Pristine column data is never moved; kernels read it from the caller's matrix via `idx`.
# Returns tau >= 0 on acceptance and -1 on rejection.
function try_enter_column!(work::NNLSWorkspace{T}, c::AbstractVector{T}, ip::Int, m1::Int) where {T}
    (; A, b, H, htau, hpos, hm1, hlen, transforms) = work

    apply_transforms!(c, work)

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

    if !(b1 / A1 > 0) # proposed new coefficient is not strictly positive
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

    # Write the new U column at position ip: reduced head, -beta on the diagonal
    # (or the reduced value itself in the trivial single-row case), and zeros below (the Householder tail lives in the H panel, not in A)
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

"""
COMPUTE ORTHOGONAL ROTATION MATRIX
The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 12, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.

    COMPUTE MATRIX  (C, S) SO THAT (C, S)(A) = (SQRT(A**2+B**2))
                    (-S,C)         (-S,C)(B)   (   0          )
    COMPUTE SIG = SQRT(A**2+B**2)
        SIG IS COMPUTED LAST TO ALLOW FOR THE POSSIBILITY THAT
        SIG MAY BE IN THE SAME LOCATION AS A OR B .
"""
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

"""
The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 15, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.
"""
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
    A0::AbstractMatrix{T}; # pristine problem data, read-only, indexed by original column
    init_dual::Bool = true,
    max_iter::Int = 3 * size(work.A, 2),
) where {T}
    (; A, b, x, w, zz, idx, invidx, b0, H, hlen, gc, gs, gi, transforms) = work
    m, n = size(A)

    copyto!(b0, b) # b holds the untransformed right-hand side at entry, for every caller; snapshot it for residual and dual computations
    hlen[] = 0
    empty!(gc)
    empty!(gs)
    empty!(gi)
    empty!(transforms)

    nsetp = 0
    iter = 0
    work.mode[] = 0
    terminated = false
    use_stale_w = !init_dual # init_dual = false: the caller preloaded w for the first pivot selection

    # ******  MAIN LOOP BEGINS HERE  ******
    @inbounds while true
        # QUIT IF ALL COEFFICIENTS ARE ALREADY IN THE SOLUTION.
        # OR IF M COLS OF A HAVE BEEN TRIANGULARIZED.
        if (nsetp >= n || nsetp >= m)
            terminated = true
            break
        end

        # COMPUTE COMPONENTS OF THE DUAL (NEGATIVE GRADIENT) VECTOR W() from the pristine set Z columns
        # and the current passive-set residual, unless the caller preloaded w for the first round.
        if use_stale_w
            use_stale_w = false
        else
            compute_dual!(work, A0, nsetp, m)
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

    # zz doubles as the candidate scratch buffer, so restore the passive-set solution into zz[1:nsetp] (bit-exact: x was assigned from these values) and mirror the transformed b on the free rows
    @inbounds for i in 1:nsetp
        zz[i] = x[idx[i]]
    end

    # Compute the norm of the final residual vector
    sm = zero(T)
    if nsetp < m
        @inbounds @simd for i in nsetp+1:m
            bi = b[i]
            zz[i] = bi
            sm = sm + bi * bi
        end
    else
        fill!(w, zero(T))
    end

    work.rnorm[] = sqrt(sm)
    work.nsetp[] = nsetp
    return work.x
end

function unsafe_nnls!(
    work::NNLSWorkspace{T},
    A0::AbstractMatrix{T}, # pristine problem data, read-only, indexed by original column; only the m0 data rows are read
    λ::T;
    init_dual::Bool = true,
    max_iter::Int = 3 * size(work.A, 2),
) where {T}
    (; A, b, x, w, zz, idx, invidx, diag, b0, H, hlen, gc, gs, gi, transforms) = work
    M, N = size(A)
    m, n = M - N, N
    m0 = m # number of data rows; rows m0+1:M are the implicit λI padding

    copyto!(b0, b) # b holds the untransformed right-hand side at entry, for every caller; snapshot it for residual and dual computations
    hlen[] = 0
    empty!(gc)
    empty!(gs)
    empty!(gi)
    empty!(transforms)

    nsetp = 0
    iter = 0
    work.mode[] = 0
    terminated = false
    use_stale_w = !init_dual # init_dual = false: the caller preloaded w for the first pivot selection

    # ******  MAIN LOOP BEGINS HERE  ******
    @inbounds while true
        # QUIT IF ALL COEFFICIENTS ARE ALREADY IN THE SOLUTION.
        # OR IF M COLS OF A HAVE BEEN TRIANGULARIZED.
        if nsetp >= n
            terminated = true
            break
        end

        # COMPUTE COMPONENTS OF THE DUAL (NEGATIVE GRADIENT) VECTOR W() from the pristine set Z columns
        # and the current passive-set residual, unless the caller preloaded w for the first round.
        # Only the data rows contribute: a set Z column's λ row is inactive, or its coefficient is zero, so the padded rows drop out.
        if use_stale_w
            use_stale_w = false
        elseif iszero(λ) && nsetp >= m0
            # The passive set spans the data rows and the padding is zero, so the dual vanishes identically.
            # In the transformed formulation the free rows are all exactly-zero padded rows, so enforce the same exact zero
            # here rather than computing the roundoff noise of a residual that is mathematically zero.
            @simd ivdep for j in (nsetp+1):n
                w[j] = zero(T)
            end
        else
            compute_dual!(work, A0, nsetp, m0)
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

    # zz doubles as the candidate scratch buffer, so restore the passive-set solution into zz[1:nsetp] (bit-exact: x was assigned from these values) and mirror the transformed b on the free rows
    @inbounds for i in 1:nsetp
        zz[i] = x[idx[i]]
    end

    # Compute the norm of the final residual vector
    sm = zero(T)
    if nsetp < M
        @inbounds @simd for i in nsetp+1:M
            bi = b[i]
            zz[i] = bi
            sm = sm + bi * bi
        end
    else
        fill!(w, zero(T))
    end

    work.rnorm[] = sqrt(sm)
    work.nsetp[] = nsetp
    return work.x
end

end # @muladd

end # module NNLS
