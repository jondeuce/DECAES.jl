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
    A::Matrix{T}
    b::Vector{T}
    x::Vector{T}
    w::Vector{T}
    zz::Vector{T}
    idx::Vector{Int}
    invidx::Vector{Int}
    diag::Vector{Bool}
    rnorm::Base.RefValue{T}
    mode::Base.RefValue{Int}
    nsetp::Base.RefValue{Int}
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

function NNLSWorkspace(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    @assert size(b) == (m,)
    return NNLSWorkspace(T, m, n)
end

function NNLSWorkspace(::Type{T}, m::Int, n::Int) where {T}
    return NNLSWorkspace(
        zeros(T, m, n), # A
        zeros(T, m),    # b
        zeros(T, n),    # x
        zeros(T, n),    # w
        zeros(T, m),    # zz
        zeros(Int, n),  # idx (Note: deliberately initialize to invalid permutation)
        zeros(Int, n),  # invidx
        zeros(Bool, n), # diag
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

function checkargs(work::NNLSWorkspace)
    m, n = size(work.A)
    @assert size(work.b) == (m,)
    @assert size(work.x) == (n,)
    @assert size(work.w) == (n,)
    @assert size(work.zz) == (m,)
    @assert size(work.idx) == (n,)
    @assert size(work.invidx) == (n,)
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
    load!(work, A, b)
    init_nnls!(work)
    unsafe_nnls!(work)
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
    load!(work, A, b)
    init_nnls!(work, λ)
    unsafe_nnls!(work, λ)
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
@inline function construct_householder!(x::AbstractVector{T}) where {T}
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

function construct_apply_householder!(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    ip::Int,
    jp::Int,
    m::Int,
) where {T}
    if ip > m
        return zero(T)
    end

    # 1. Construct householder
    # 2. Check if column is sufficiently independent
    # 3. Check if proposed new value for x = A[ip, jp] / b[ip] > 0
    # 4. Update b
    # If good then tau >= 0, else tau < 0
    @inbounds alpha = A[ip, jp]

    xnorm = zero(T)
    @inbounds @simd for i in ip:m
        xnorm = xnorm + A[i, jp] * A[i, jp]
    end
    xnorm = sqrt(xnorm)

    if xnorm == 0
        return -one(T)
    end

    beta  = copysign(xnorm, alpha)
    alpha = alpha + beta
    tau   = alpha / beta

    sm = b[ip]
    @inbounds @simd for i in ip+1:m
        sm = sm + b[i] * (A[i, jp] / alpha)
    end
    sm *= -tau

    A1 = -beta
    @inbounds b1 = b[ip] + sm

    if b1 / A1 > 0
        # Good column, update b
        if ip < m
            if ip != jp
                # Swap columns
                @inbounds @simd for i in 1:ip-1
                    A[i, ip], A[i, jp] = A[i, jp], A[i, ip]
                end
                @inbounds b[ip] = b1
                @inbounds A[ip, ip], A[ip, jp] = A1, A[ip, ip]
                @inbounds @simd for i in ip+1:m
                    Aij = A[i, jp] / alpha
                    A[i, ip], A[i, jp] = Aij, A[i, ip]
                    b[i] = b[i] + sm * Aij
                end
            else
                @inbounds b[ip] = b1
                @inbounds A[ip, ip] = A1
                @inbounds @simd for i in ip+1:m
                    Aii = A[i, ip] / alpha
                    A[i, ip] = Aii
                    b[i] = b[i] + sm * Aii
                end
            end
        else
            tau = zero(T)
            if ip != jp
                @inbounds @simd for i in 1:m
                    A[i, ip], A[i, jp] = A[i, jp], A[i, ip]
                end
            end
        end
        return tau
    else
        return -one(T)
    end
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

function apply_householder_dual!(
    A::AbstractMatrix{T},
    w::AbstractVector{T},
    b::AbstractVector{T},
    tau::T,
    j1::Int,
    m1::Int,
) where {T}
    if j1 >= m1
        return nothing
    end

    @inbounds aii = A[j1, j1]
    @inbounds A[j1, j1] = 1

    n = size(A, 2)
    ntrunc = (j1 + 1) + 4 * ((n - (j1 + 1)) ÷ 4) - 1
    @inbounds for j in j1+1:4:ntrunc # unroll over columns
        @nexprs 4 α -> sm_α = zero(T)
        @simd for i in j1:m1
            ui = A[i, j1]
            @nexprs 4 α -> sm_α = sm_α + A[i, j+(α-1)] * ui
        end
        @nexprs 4 α -> sm_α *= -tau
        @nexprs 4 α -> w_α = zero(T)
        @nexprs 4 α -> A[j1, j+(α-1)] = A[j1, j+(α-1)] + sm_α
        @simd for i in j1+1:m1
            bi = b[i]
            ui = A[i, j1]
            @nexprs 4 α -> A_α = A[i, j+(α-1)] + sm_α * ui
            @nexprs 4 α -> w_α = w_α + A_α * bi
            @nexprs 4 α -> A[i, j+(α-1)] = A_α
        end
        @nexprs 4 α -> w[j+(α-1)] = w_α
    end

    if ntrunc != n # remainder loop
        @inbounds for j in ntrunc+1:n
            sm = zero(T)
            @simd for i in j1:m1
                sm = sm + A[i, j] * A[i, j1]
            end
            sm *= -tau
            wj = zero(T)
            A[j1, j] = A[j1, j] + sm
            @simd for i in j1+1:m1
                Aij = A[i, j] + sm * A[i, j1]
                wj = wj + Aij * b[i]
                A[i, j] = Aij
            end
            w[j] = wj
        end
    end

    @inbounds A[j1, j1] = aii

    return nothing
end

"""
COMPUTE COMPONENTS OF THE DUAL (NEGATIVE GRADIENT) VECTOR W().
"""
@inline function compute_dual!(
    w::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    j1::Int,
    m1::Int,
) where {T}
    n = size(A, 2)
    ntrunc = j1 + 4 * ((n - j1) ÷ 4) - 1
    @inbounds for j in j1:4:ntrunc # unroll over columns
        @nexprs 4 α -> sm_α = zero(T)
        @simd for i in j1:m1
            bi = b[i]
            @nexprs 4 α -> sm_α = sm_α + A[i, j+(α-1)] * bi
        end
        @nexprs 4 α -> w[j+(α-1)] = sm_α
    end

    if ntrunc != n # remainder loop
        @inbounds for j in ntrunc+1:n
            sm = zero(T)
            @simd for i in j1:m1
                sm = sm + A[i, j] * b[i]
            end
            w[j] = sm
        end
    end

    return w
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
@inline function solve_triangular_system!(
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
            @simd for i in 1:j-1
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

@inline function largest_positive_dual(
    w::AbstractVector{T},
    j1::Int,
) where {T}
    wmax = zero(T)
    jmax = 0
    @inbounds for j in j1:length(w)
        if w[j] > wmax
            wmax = w[j]
            jmax = j
        end
    end
    return wmax, jmax
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
    (; A, b, x, idx, invidx, diag) = work

    M, N = size(A)
    M > N || throw(DimensionMismatch("A must be of the form [A₀; λ*I], got size(A) = $(size(A))"))
    m, n = M - N, N

    @inbounds for j in 1:N
        @simd for i in m+1:M
            A[i, j] = 0
        end
    end

    @inbounds @simd for i in 1:N
        x[i] = zero(T)
        b[m+i] = zero(T)
        idx[i] = i
        invidx[i] = i
        diag[i] = false
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
    work::NNLSWorkspace{T};
    init_dual::Bool = true,
    max_iter::Int = 3 * size(work.A, 2),
) where {T}
    (; A, b, x, w, zz, idx, invidx) = work
    m, n = size(A)

    if init_dual
        fill!(w, zero(T))
        compute_dual!(w, A, b, 1, m)
    end

    nsetp = 0
    iter = 0
    work.mode[] = 0
    terminated = false

    # ******  MAIN LOOP BEGINS HERE  ******
    @inbounds while true
        # QUIT IF ALL COEFFICIENTS ARE ALREADY IN THE SOLUTION.
        # OR IF M COLS OF A HAVE BEEN TRIANGULARIZED.
        if (nsetp >= n || nsetp >= m)
            terminated = true
            break
        end

        jmax = nsetp
        tau = zero(T)
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
            tau = construct_apply_householder!(A, b, nsetp + 1, jmax, m)
            if tau >= 0
                break
            else
                # REJECT J AS A CANDIDATE TO BE MOVED FROM SET Z TO SET P.
                # RESTORE A(NPP1,J), SET W(J)=0., AND LOOP BACK TO TEST DUAL
                # COEFFS AGAIN.
                # NOTE: A(NPP1,J) restored in `construct_apply_householder!`
                w[jmax] = zero(T)
            end
        end

        if terminated
            break
        end

        # THE INDEX J=INDEX(IZ) HAS BEEN SELECTED TO BE MOVED FROM
        # SET Z TO SET P. UPDATE B, UPDATE INDICES, APPLY HOUSEHOLDER
        # TRANSFORMATIONS TO COLS IN NEW SET Z, ZERO SUBDIAGONAL ELTS IN
        # COL J, SET W(J)=0.
        # NOTE: B updated in `construct_apply_householder!`
        nsetp += 1
        idx[nsetp], idx[jmax] = idx[jmax], idx[nsetp]

        if nsetp < n
            apply_householder_dual!(A, w, b, tau, nsetp, m)
        end

        @simd for i in nsetp+1:m
            A[i, nsetp] = zero(T)
        end

        w[nsetp] = zero(T)

        # SOLVE THE TRIANGULAR SYSTEM.
        # STORE THE SOLUTION TEMPORARILY IN ZZ().
        @simd for i in 1:nsetp
            zz[i] = b[i]
        end
        solve_triangular_system!(zz, A, nsetp, Val(false))

        # ******  SECONDARY LOOP BEGINS HERE  ******
        dual_flag = false

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
            dual_flag = true

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

                if imv != nsetp
                    for i in imv+1:nsetp
                        cc, ss, rr = orthogonal_rotmat(A[i-1, i], A[i, i])
                        A[i-1, i] = rr
                        A[i, i] = zero(T)

                        # Apply procedure G2 (CC,SS,A(J-1,L),A(J,L))
                        @simd for j in 1:i-1
                            A[i-1, j], A[i, j] = orthogonal_rotmatvec(cc, ss, A[i-1, j], A[i, j])
                        end
                        @simd for j in i+1:n
                            A[i-1, j], A[i, j] = orthogonal_rotmatvec(cc, ss, A[i-1, j], A[i, j])
                        end

                        # Apply procedure G2 (CC,SS,B(J-1),B(J))
                        b[i-1], b[i] = orthogonal_rotmatvec(cc, ss, b[i-1], b[i])
                    end

                    # Swap columns
                    for j in imv:nsetp-1
                        @simd for i in 1:m
                            A[i, j+1], A[i, j] = A[i, j], A[i, j+1]
                        end
                        idx[j], idx[j+1] = idx[j+1], idx[j]
                    end
                end

                nsetp -= 1

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

        if dual_flag
            compute_dual!(w, A, b, nsetp + 1, m)
        end

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
    λ::T;
    init_dual::Bool = true,
    max_iter::Int = 3 * size(work.A, 2),
) where {T}
    (; A, b, x, w, zz, idx, invidx, diag) = work
    M, N = size(A)
    m, n = M - N, N

    if init_dual
        fill!(w, zero(T))
        compute_dual!(w, A, b, 1, m)
    end

    nsetp = 0
    iter = 0
    work.mode[] = 0
    terminated = false

    # ******  MAIN LOOP BEGINS HERE  ******
    @inbounds while true
        # QUIT IF ALL COEFFICIENTS ARE ALREADY IN THE SOLUTION.
        # OR IF M COLS OF A HAVE BEEN TRIANGULARIZED.
        if nsetp >= n
            terminated = true
            break
        end

        jmax = nsetp
        tau = zero(T)
        while true
            # FIND LARGEST POSITIVE W(J).
            wmax, jmax = largest_positive_dual(w, nsetp + 1)

            # IF WMAX .LE. 0. GO TO TERMINATION.
            # THIS INDICATES SATISFACTION OF THE KUHN-TUCKER CONDITIONS.
            if wmax <= 0
                terminated = true
                break
            end

            if !diag[idx[jmax]]
                A[m+1, jmax] = λ
            end

            # THE SIGN OF W(J) IS OK FOR J TO BE MOVED TO SET P.
            # BEGIN THE TRANSFORMATION AND CHECK NEW DIAGONAL ELEMENT TO AVOID
            # NEAR LINEAR DEPENDENCE.
            tau = construct_apply_householder!(A, b, nsetp + 1, jmax, min(m + 1, M))
            if tau >= 0
                break
            else
                # REJECT J AS A CANDIDATE TO BE MOVED FROM SET Z TO SET P.
                # RESTORE A(NPP1,J), SET W(J)=0., AND LOOP BACK TO TEST DUAL
                # COEFFS AGAIN.
                # NOTE: A(NPP1,J) restored in `construct_apply_householder!`
                w[jmax] = zero(T)
                if m < M
                    A[m+1, jmax] = zero(T)
                end
            end
        end

        if terminated
            break
        end

        # THE INDEX J=INDEX(IZ) HAS BEEN SELECTED TO BE MOVED FROM
        # SET Z TO SET P. UPDATE B, UPDATE INDICES, APPLY HOUSEHOLDER
        # TRANSFORMATIONS TO COLS IN NEW SET Z, ZERO SUBDIAGONAL ELTS IN
        # COL J, SET W(J)=0.
        # NOTE: B updated in `construct_apply_householder!`
        if !diag[idx[jmax]]
            m = min(m + 1, M)
            diag[idx[jmax]] = true
        end

        nsetp += 1
        idx[nsetp], idx[jmax] = idx[jmax], idx[nsetp]

        if nsetp < N
            apply_householder_dual!(A, w, b, tau, nsetp, m)
        end

        @simd for i in nsetp+1:m
            A[i, nsetp] = zero(T)
        end

        w[nsetp] = zero(T)

        # SOLVE THE TRIANGULAR SYSTEM.
        # STORE THE SOLUTION TEMPORARILY IN ZZ().
        @simd for i in 1:nsetp
            zz[i] = b[i]
        end
        solve_triangular_system!(zz, A, nsetp, Val(false))

        # ******  SECONDARY LOOP BEGINS HERE  ******
        dual_flag = false

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
            dual_flag = true

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

                if imv != nsetp
                    for i in imv+1:nsetp
                        cc, ss, rr = orthogonal_rotmat(A[i-1, i], A[i, i])
                        A[i-1, i] = rr
                        A[i, i] = zero(T)

                        # Apply procedure G2 (CC,SS,A(J-1,L),A(J,L))
                        @simd for j in 1:i-1
                            A[i-1, j], A[i, j] = orthogonal_rotmatvec(cc, ss, A[i-1, j], A[i, j])
                        end
                        @simd for j in i+1:n
                            A[i-1, j], A[i, j] = orthogonal_rotmatvec(cc, ss, A[i-1, j], A[i, j])
                        end

                        # Apply procedure G2 (CC,SS,B(J-1),B(J))
                        b[i-1], b[i] = orthogonal_rotmatvec(cc, ss, b[i-1], b[i])
                    end

                    # Swap columns
                    for j in imv:nsetp-1
                        @simd for i in 1:m
                            A[i, j+1], A[i, j] = A[i, j], A[i, j+1]
                        end
                        idx[j], idx[j+1] = idx[j+1], idx[j]
                    end
                end

                nsetp -= 1

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

        if dual_flag
            compute_dual!(w, A, b, nsetp + 1, m)
        end

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
