####
#### NNLS submodule
####

# This NNLS submodule is modified version of the corresponding NNLS module from
# the forked NonNegLeastSquares.jl repository:
#
#   https://github.com/jondeuce/NonNegLeastSquares.jl/blob/a122bf7acb498efcaf140b719133691e7c4cd03d/src/nnls.jl#L1
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

using ..DECAES: @acc
using LinearAlgebra
using MuladdMacro: @muladd
using UnsafeArrays: uview

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
    rnorm::Base.RefValue{T}
    mode::Base.RefValue{Int}
    nsetp::Base.RefValue{Int}
end
@inline solution(work::NNLSWorkspace) = work.x
@inline dual(work::NNLSWorkspace) = work.w
@inline residualnorm(work::NNLSWorkspace) = work.rnorm[]
@inline ncomponents(work::NNLSWorkspace) = work.nsetp[]
@inline components(work::NNLSWorkspace) = uview(work.idx, 1:ncomponents(work))
@inline positive_solution(work::NNLSWorkspace) = uview(solution(work), components(work))
@inline choleskyfactor(work::NNLSWorkspace, ::Val{:U}) = UpperTriangular(uview(work.A, 1:ncomponents(work), components(work)))
@inline choleskyfactor(work::NNLSWorkspace, ::Val{:L}) = choleskyfactor(work, Val(:U))'

function NNLSWorkspace{T}(m, n) where {T}
    NNLSWorkspace(
        zeros(T, m, n), # A
        zeros(T, m),    # b
        zeros(T, n),    # x
        zeros(T, n),    # w
        zeros(T, m),    # zz
        zeros(Int, n),  # idx
        Ref(zero(T)),   # rnorm
        Ref(0),         # mode
        Ref(0),         # nsetp
    )
end

function NNLSWorkspace(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    @assert size(b) == (m,)
    work = NNLSWorkspace{T}(m, n)
    load!(work, A, b)
    return work
end

function NNLSWorkspace(m::Int, n::Int, ::Type{T} = Float64) where {T}
    return NNLSWorkspace{T}(m, n)
end

function load!(work::NNLSWorkspace{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    @assert size(A) == size(work.A)
    @assert size(b) == size(work.b)
    copyto!(work.A, A)
    copyto!(work.b, b)
    return work
end

@noinline function checkargs(work::NNLSWorkspace)
    m, n = size(work.A)
    @assert size(work.b) == (m,)
    @assert size(work.x) == (n,)
    @assert size(work.w) == (n,)
    @assert size(work.zz) == (m,)
    @assert size(work.idx) == (n,)
end

struct NormalEquationCholesky{T, W <: NNLSWorkspace{T}} <: Factorization{T}
    work::W
end
@inline Base.size(F::NormalEquationCholesky) = (n = size(F.work.A, 2); return (n, n))

function solve_triangular_system!(y, F::NormalEquationCholesky, ::Val{transp} = Val(false)) where {transp}
    solve_triangular_system!(y, F.work.A, F.work.idx, F.work.nsetp[], Val(transp))
    return y
end

function LinearAlgebra.ldiv!(y::AbstractVector, F::NormalEquationCholesky, x::AbstractVector)
    @assert length(x) == length(y) == F.work.nsetp[]
    (y !== x) && copyto!(y, x)
    solve_triangular_system!(y, F, Val(true)) # y = U'\x
    solve_triangular_system!(y, F, Val(false)) # y = U\y = U\(U'\x)
    return y
end
LinearAlgebra.ldiv!(F::NormalEquationCholesky, x::AbstractVector) = ldiv!(x, F, x)
Base.:\(F::NormalEquationCholesky, x::AbstractVector) = ldiv!(copy(x), F, x)

struct NormalEquation end

LinearAlgebra.cholesky!(::NormalEquation, work::NNLSWorkspace) = NormalEquationCholesky(work)

"""
x = nnls(A, b; ...)

Solves non-negative least-squares problem by the active set method
of Lawson & Hanson (1974).

Optional arguments:
    max_iter: maximum number of iterations (counts inner loop iterations)

References:
    Lawson, C.L. and R.J. Hanson, Solving Least-Squares Problems,
    Prentice-Hall, Chapter 23, p. 161, 1974.
"""
function nnls(
        A,
        b::AbstractVector{T};
        max_iter::Int = 3*size(A, 2)
    ) where {T}
    work = NNLSWorkspace(A, b)
    nnls!(work, max_iter)
    return solution(work)
end

function nnls!(
        work::NNLSWorkspace{T},
        A::AbstractMatrix{T},
        b::AbstractVector{T},
        max_iter::Int = 3*size(A, 2)
    ) where {T}
    load!(work, A, b)
    nnls!(work, max_iter)
    return solution(work)
end

"""
CONSTRUCTION AND/OR APPLICATION OF A SINGLE
HOUSEHOLDER TRANSFORMATION..     Q = I + U*(U**T)/B

The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 12, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.
"""
function construct_householder!(u::AbstractVector{T}, up::T) where {T}
    if length(u) <= 1
        return up
    end

    sm = zero(T)
    @acc for i in eachindex(u)
        sm = sm + u[i] * u[i]
    end
    cl = sqrt(sm)

    @inbounds u1 = u[1]
    cl = ifelse(u1 > 0, -cl, cl)
    up = u1 - cl
    @inbounds u[1] = cl

    return up
end

"""
CONSTRUCTION AND/OR APPLICATION OF A SINGLE
HOUSEHOLDER TRANSFORMATION..     Q = I + U*(U**T)/B

The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 12, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.
"""
function apply_householder!(u::AbstractVector{T}, up::T, c::AbstractVector{T}) where {T}
    m = length(u)
    if m <= 1
        return nothing
    end

    @inbounds u1 = u[1]
    @assert abs(u1) > 0
    up_u1 = up * u1
    if up_u1 >= 0
        return nothing
    end

    @inbounds c1 = c[1]
    sm = c1 * up
    @acc for i in 2:m
        sm = sm + c[i] * u[i]
    end

    if sm != 0
        sm /= up_u1
        @inbounds c[1] = c[1] + sm * up
        @acc for i in 2:m
            c[i] = c[i] + sm * u[i]
        end
    end
end

function apply_householder!(A::AbstractMatrix{T}, up::T, idx::AbstractVector{Int}, nsetp::Int, jup::Int) where {T}
    m, n = size(A)
    if m - nsetp <= 0
        return nothing
    end

    @inbounds u1 = A[nsetp, jup]
    @assert abs(u1) > 0
    up_u1 = up * u1
    if up_u1 >= 0
        return nothing
    end

    @inbounds A[nsetp, jup] = up
    @inbounds for ip in nsetp+1:n
        j = idx[ip]
        sm = zero(T)
        @acc for l in nsetp:m
            sm = sm + A[l, j] * A[l, jup]
        end
        @inbounds if sm != 0
            sm /= up_u1
            @acc for l in nsetp:m
                A[l, j] = A[l, j] + sm * A[l, jup]
            end
        end
    end
    @inbounds A[nsetp, jup] = u1

    return nothing
end

"""
   COMPUTE ORTHOGONAL ROTATION MATRIX..
The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 12, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.

   COMPUTE.. MATRIX   (C, S) SO THAT (C, S)(A) = (SQRT(A**2+B**2))
                      (-S,C)         (-S,C)(B)   (   0          )
   COMPUTE SIG = SQRT(A**2+B**2)
      SIG IS COMPUTED LAST TO ALLOW FOR THE POSSIBILITY THAT
      SIG MAY BE IN THE SAME LOCATION AS A OR B .
"""
@inline function orthogonal_rotmat(a::T, b::T) where {T}
    if abs(a) > abs(b)
        xr = b / a
        yr = sqrt(1 + xr * xr)
        c = inv(yr) * sign(a)
        s = c * xr
        sig = abs(a) * yr
    elseif b != 0
        xr = a / b
        yr = sqrt(1 + xr * xr)
        s = inv(yr) * sign(b)
        c = s * xr
        sig = abs(b) * yr
    else
        sig = zero(T)
        c = zero(T)
        s = one(T)
    end
    return c, s, sig
end

"""
The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 15, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.
"""
function solve_triangular_system!(zz::AbstractVector{T}, A::AbstractMatrix{T}, idx::AbstractVector{Int}, nsetp::Int, ::Val{transp} = Val(false)) where {T, transp}
    if nsetp <= 0
        return zz
    end

    if !transp
        # Solve the upper-triangular system Ux=b in-place where:
        #   U = A[1:nsetp, idx[1:nstep]]
        #   b = zz[1:nsetp]
        #   x = zz[1:nsetp] (i.e. RHS b is overwritten)
        @inbounds j = idx[nsetp]
        @inbounds zz[nsetp] /= A[nsetp, j]
        @inbounds for ip in nsetp-1:-1:1
            zz1 = zz[ip + 1]
            @acc for l in 1:ip
                zz[l] = zz[l] - A[l, j] * zz1
            end
            j = idx[ip]
            zz[ip] /= A[ip, j]
        end
    else
        # Solve the lower-triangular system Lx=b in-place where:
        #   L = A[1:nsetp, idx[1:nstep]]' (i.e. transpose of U above)
        #   b = zz[1:nsetp]
        #   x = zz[1:nsetp] (i.e. RHS b is overwritten)
        @inbounds j = idx[1]
        @inbounds zz[1] /= A[1, j]
        @inbounds for ip in 2:nsetp
            j = idx[ip]
            zz1 = zz[ip]
            @acc for l in 1:ip-1
                zz1 = zz1 - A[l, j] * zz[l]
            end
            zz1 /= A[ip, j]
            zz[ip] = zz1
        end
    end

    return zz
end

function largest_positive_dual(w::AbstractVector{T}, idx::AbstractVector{Int}, nsetp::Int) where {T}
    n = length(w)
    wmax = zero(T)
    i_wmax = 0
    @inbounds for ip in nsetp+1:n
        j = idx[ip]
        if w[j] > wmax
            wmax = w[j]
            i_wmax = ip
        end
    end
    wmax, i_wmax
end

"""
Algorithm NNLS: NONNEGATIVE LEAST SQUARES

The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 15, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.

GIVEN AN M BY N MATRIX, A, AND AN M-VECTOR, B,  COMPUTE AN
N-VECTOR, X, THAT SOLVES THE LEAST SQUARES PROBLEM
                 A * X = B  SUBJECT TO X .GE. 0
"""
function nnls!(
        work::NNLSWorkspace{T},
        max_iter::Int = 3*size(work.A, 2)
    ) where {T}

    checkargs(work)
    (; A, b, x, w, zz, idx) = work
    m, n = size(A)

    iter = 0
    nsetp = 0
    up = zero(T)
    fill!(x, zero(T))
    copyto!(idx, 1:n)

    work.mode[] = 1
    terminated = false

    # ******  MAIN LOOP BEGINS HERE  ******
    @inbounds while true
        local i_curr, j_curr, i_maxdual, j_maxdual

        # QUIT IF ALL COEFFICIENTS ARE ALREADY IN THE SOLUTION.
        # OR IF M COLS OF A HAVE BEEN TRIANGULARIZED.
        if (nsetp + 1 > n || nsetp >= m)
            terminated = true
            break
        end

        # COMPUTE COMPONENTS OF THE DUAL (NEGATIVE GRADIENT) VECTOR W().
        @inbounds for ip in nsetp+1:n
            j = idx[ip]
            sm = zero(T)
            @acc for l in nsetp+1:m
                sm = sm + A[l, j] * b[l]
            end
            w[j] = sm
        end

        @inbounds while true
            # FIND LARGEST POSITIVE W(J).
            wmax, i_wmax = largest_positive_dual(w, idx, nsetp)

            # IF WMAX .LE. 0. GO TO TERMINATION.
            # THIS INDICATES SATISFACTION OF THE KUHN-TUCKER CONDITIONS.
            if wmax <= 0
                terminated = true
                break
            end

            i_maxdual = i_wmax
            j_maxdual = idx[i_maxdual]

            # THE SIGN OF W(J) IS OK FOR J TO BE MOVED TO SET P.
            # BEGIN THE TRANSFORMATION AND CHECK NEW DIAGONAL ELEMENT TO AVOID
            # NEAR LINEAR DEPENDENCE.
            Asave = A[nsetp + 1, j_maxdual]
            up = construct_householder!(uview(A, nsetp+1:m, j_maxdual), up)

            if abs(A[nsetp + 1, j_maxdual]) > 0
                # COL J IS SUFFICIENTLY INDEPENDENT.  COPY B INTO ZZ, UPDATE ZZ
                # AND SOLVE FOR ZTEST ( = PROPOSED NEW VALUE FOR X(J) ).
                copyto!(zz, b)
                apply_householder!(uview(A, nsetp+1:m, j_maxdual), up, uview(zz, nsetp+1:m))
                ztest = zz[nsetp + 1] / A[nsetp + 1, j_maxdual]

                # SEE IF ZTEST IS POSITIVE
                if ztest > 0
                    break
                end
            end

            # REJECT J AS A CANDIDATE TO BE MOVED FROM SET Z TO SET P.
            # RESTORE A(NPP1,J), SET W(J)=0., AND LOOP BACK TO TEST DUAL
            # COEFFS AGAIN.
            A[nsetp + 1, j_maxdual] = Asave
            w[j_maxdual] = zero(T)
        end
        if terminated
            break
        end

        # THE INDEX  J=INDEX(IZ)  HAS BEEN SELECTED TO BE MOVED FROM
        # SET Z TO SET P.    UPDATE B,  UPDATE INDICES,  APPLY HOUSEHOLDER
        # TRANSFORMATIONS TO COLS IN NEW SET Z,  ZERO SUBDIAGONAL ELTS IN
        # COL J,  SET W(J)=0.
        copyto!(b, zz)

        idx[i_maxdual] = idx[nsetp + 1]
        idx[nsetp + 1] = j_maxdual
        nsetp += 1

        if nsetp + 1 <= n
            apply_householder!(A, up, idx, nsetp, j_maxdual)
        end

        if nsetp != m
            @acc for l in nsetp+1:m
                A[l, j_maxdual] = zero(T)
            end
        end

        w[j_maxdual] = zero(T)

        # SOLVE THE TRIANGULAR SYSTEM.
        # STORE THE SOLUTION TEMPORARILY IN ZZ().
        solve_triangular_system!(zz, A, idx, nsetp)
        @inbounds if nsetp > 0
            i_curr = idx[1]
        end

        # ******  SECONDARY LOOP BEGINS HERE ******
        #
        # ITERATION COUNTER.
        @inbounds while true
            iter += 1
            if iter > max_iter
                work.mode[] = 3
                terminated = true
                break
            end

            # SEE IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE.
            # IF NOT COMPUTE ALPHA.
            alpha = convert(T, 2)
            @inbounds for ip in 1:nsetp
                j = idx[ip]
                if zz[ip] <= 0
                    t = -x[j] / (zz[ip] - x[j])
                    if alpha > t
                        alpha = t
                        i_curr = ip
                    end
                end
            end

            # IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE THEN ALPHA WILL
            # STILL = 2.    IF SO EXIT FROM SECONDARY LOOP TO MAIN LOOP.
            if alpha == 2
                break
            end

            # OTHERWISE USE ALPHA WHICH WILL BE BETWEEN 0 AND 1 TO
            # INTERPOLATE BETWEEN THE OLD X AND THE NEW ZZ.
            @inbounds for ip in 1:nsetp
                j = idx[ip]
                x[j] = x[j] + alpha * (zz[ip] - x[j])
            end

            # MODIFY A AND B AND THE INDEX ARRAYS TO MOVE COEFFICIENT I
            # FROM SET P TO SET Z.
            j_curr = idx[i_curr]
            @inbounds while true
                x[j_curr] = zero(T)

                if i_curr != nsetp
                    i_curr += 1
                    @inbounds for ip in i_curr:nsetp
                        j = idx[ip]
                        idx[ip-1] = j

                        cc, ss, sig = orthogonal_rotmat(A[ip-1, j], A[ip, j])
                        A[ip-1, j] = sig
                        A[ip, j]   = zero(T)

                        # Apply procedure G2 (CC,SS,A(J-1,L),A(J,L))
                        @simd ivdep for l in 1:j-1
                            tmp = A[ip-1, l]
                            A[ip-1, l] =  cc * tmp + ss * A[ip, l]
                            A[ip,   l] = -ss * tmp + cc * A[ip, l]
                        end

                        @simd ivdep for l in j+1:n
                            tmp = A[ip-1, l]
                            A[ip-1, l] =  cc * tmp + ss * A[ip, l]
                            A[ip,   l] = -ss * tmp + cc * A[ip, l]
                        end

                        # Apply procedure G2 (CC,SS,B(J-1),B(J))
                        tmp = b[ip-1]
                        b[ip-1] =  cc * tmp + ss * b[ip]
                        b[ip]   = -ss * tmp + cc * b[ip]
                    end
                end

                nsetp -= 1
                idx[nsetp + 1] = j_curr

                # SEE IF THE REMAINING COEFFS IN SET P ARE FEASIBLE.  THEY SHOULD
                # BE BECAUSE OF THE WAY ALPHA WAS DETERMINED.
                # IF ANY ARE INFEASIBLE IT IS DUE TO ROUND-OFF ERROR.  ANY
                # THAT ARE NONPOSITIVE WILL BE SET TO ZERO
                # AND MOVED FROM SET P TO SET Z.
                allfeasible = true
                @inbounds for ip in 1:nsetp
                    j_curr = idx[ip]
                    if x[j_curr] <= 0
                        allfeasible = false
                        i_curr = ip
                        break
                    end
                end
                if allfeasible
                    break
                end
            end

            # COPY B( ) INTO ZZ( ).  THEN SOLVE AGAIN AND LOOP BACK.
            copyto!(zz, b)
            solve_triangular_system!(zz, A, idx, nsetp)
            @inbounds if nsetp > 0
                i_curr = idx[1]
            end
        end
        if terminated
            break
        end
        # ******  END OF SECONDARY LOOP  ******

        @inbounds for ip in 1:nsetp
            x[idx[ip]] = zz[ip]
        end
        # ALL NEW COEFFS ARE POSITIVE.  LOOP BACK TO BEGINNING.
    end

    # ******  END OF MAIN LOOP  ******
    # COME TO HERE FOR TERMINATION.
    # COMPUTE THE NORM OF THE FINAL RESIDUAL VECTOR.

    sm = zero(T)
    if nsetp < m
        @acc for ip in nsetp+1:m
            bi = b[ip]
            zz[ip] = bi
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

end # module
