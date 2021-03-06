####
#### NNLS submodule
####

# This NNLS submodule has been copied in directly from the forked NonNegLeastSquares.jl repository:
# 
#   https://github.com/jondeuce/NonNegLeastSquares.jl/blob/master/src/NonNegLeastSquares.jl
# 
# This is a temporary measure to get DECAES up and running.
# Depending on the unregistered package at the above link repeatedly leads to the following error:
# 
#   ERROR: Unsatisfiable requirements detected for package NonNegLeastSquares [e5310699]:
#    NonNegLeastSquares [e5310699] log:
#    ├─NonNegLeastSquares [e5310699] has no known versions!
#    └─restricted to versions * by DECAES [4d91e7b4] — no versions left
#      └─DECAES [4d91e7b4] log:
#        ├─possible versions are: 0.1.0 or uninstalled
#        └─DECAES [4d91e7b4] is fixed to version 0.1.0
# 
# Due to the fact that this error is possibly related to a know LibGit2 issue:
# 
#   https://github.com/JuliaLang/julia/issues/33111
# 
# I have resorted to simply copying the required module into this package.
# Fortunately, it is completely self contained in a single file, and therefore
# is easily included in this project.

module NNLS

using LinearAlgebra
using UnsafeArrays: @uviews, uviews, uview
# using LoopVectorization: @avx

export nnls,
       nnls!,
       NNLSWorkspace,
       load!

"""
CONSTRUCTION AND/OR APPLICATION OF A SINGLE
HOUSEHOLDER TRANSFORMATION..     Q = I + U*(U**T)/B

The original version of this code was developed by
Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
1973 JUN 12, and published in the book
"SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
Revised FEB 1995 to accompany reprinting of the book by SIAM.
"""
function construct_householder!(u::AbstractVector{T}, up::T)::T where {T}
    m = length(u)
    if m <= 1
        return up
    end

    sm = zero(T)
    @inbounds @simd for i in 1:m #@avx
        sm += u[i]^2
    end
    cl = sqrt(sm)

    @inbounds u1 = u[1]
    cl = ifelse(u1 > 0, -cl, cl)
    result = u1 - cl
    @inbounds u[1] = cl

    return result
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
        return
    end

    @inbounds u1 = u[1]
    cl = abs(u1)
    @assert cl > 0
    b = up * u1
    if b >= 0
        return
    end

    @inbounds c1 = c[1]
    sm = c1 * up
    @inbounds @simd for i in 2:m #@avx
        sm += c[i] * u[i]
    end

    if sm != 0
        sm /= b
        @inbounds c[1] += sm * up
        @inbounds @simd ivdep for i in 2:m #@avx
            c[i] += sm * u[i]
        end
    end
end

function apply_householder!(A::AbstractMatrix{T}, up, cols, nsetp, j) where {T}
    m, n = size(A)
    if m - nsetp <= 0
        return
    end

    @inbounds u1 = A[nsetp,j]
    cl = abs(u1)
    @assert cl > 0
    b = up * u1
    if b >= 0
        return
    end

    @inbounds A[nsetp,j] = up
    @inbounds for jj in cols
        sm = zero(T)
        @inbounds @simd for k in nsetp:m #@avx
            sm += A[k,jj] * A[k,j]
        end
        @inbounds if sm != 0
            sm /= b
            @inbounds @simd ivdep for i in nsetp:m #@avx
                A[i,jj] += sm * A[i,j]
            end
        end
    end
    @inbounds A[nsetp,j] = u1

    return
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
function orthogonal_rotmat(a::T, b::T)::Tuple{T, T, T} where {T}
    if abs(a) > abs(b)
        xr = b / a
        yr = sqrt(1 + xr^2)
        c = inv(yr) * sign(a)
        s = c * xr
        sig = abs(a) * yr
    elseif b != 0
        xr = a / b
        yr = sqrt(1 + xr^2)
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
function solve_triangular_system!(zz, A, idx, nsetp, jj)
    if nsetp <= 0
        return jj
    end
    ip = nsetp
    @inbounds jj = idx[ip]
    @inbounds zz[ip] /= A[ip, jj]
    @inbounds for ip in nsetp-1:-1:1
        zz1 = zz[ip + 1]
        @inbounds @simd ivdep for ii in 1:ip #@avx
            zz[ii] -= A[ii, jj] * zz1
        end
        jj = idx[ip]
        zz[ip] /= A[ip, jj]
    end
    return jj
end

mutable struct NNLSWorkspace{T, I <: Integer}
    QA::Matrix{T}
    Qb::Vector{T}
    x::Vector{T}
    w::Vector{T}
    zz::Vector{T}
    idx::Vector{I}
    rnorm::T
    mode::I
    nsetp::I
end

function NNLSWorkspace{T,I}(m, n) where {T, I <: Integer}
    NNLSWorkspace{T,I}(
        zeros(T, m, n), # A
        zeros(T, m),    # b
        zeros(T, n),    # x
        zeros(T, n),    # w
        zeros(T, m),    # zz
        zeros(I, n),    # idx
        zero(T),        # rnorm
        zero(I),        # mode
        zero(I),        # nsetp
    )
end

function Base.resize!(work::NNLSWorkspace{T}, m::Integer, n::Integer) where {T}
    work.QA = zeros(T, m, n)
    work.Qb = zeros(T, m)
    resize!(work.x, n)
    resize!(work.w, n)
    resize!(work.zz, m)
    resize!(work.idx, n)
end

function load!(work::NNLSWorkspace{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    @assert size(b) == (m,)
    if size(work.QA, 1) != m || size(work.QA, 2) != n
        resize!(work, m, n)
    end
    copyto!(work.QA, A)
    copyto!(work.Qb, b)
    work
end

function NNLSWorkspace(
        m::Integer,
        n::Integer,
        eltype::Type{T} = Float64,
        indextype::Type{I} = Int
    ) where {T,I}
    NNLSWorkspace{T, I}(m, n)
end

function NNLSWorkspace(A::AbstractMatrix{T}, b::AbstractVector{T}, indextype::Type{I} = Int) where {T,I}
    m, n = size(A)
    @assert size(b) == (m,)
    work = NNLSWorkspace{T, I}(m, n)
    load!(work, A, b)
    work
end

"""
Views in Julia no longer allocate memory since v1.5, but unsafe views
provided by the UnsafeArrays package are still faster.

Note: will segfault if parent array is garbage collected. If this is
a possibility, use `UnsafeArrays.@uviews(parent, I...)` instead.
"""
@inline function fastview(parent::DenseArray, start_ind::Integer, len::Integer)
    uview(parent, start_ind:(start_ind + len - 1)) # uview checks for isbitstype(T) internally
end

@noinline function checkargs(work::NNLSWorkspace)
    m, n = size(work.QA)
    @assert size(work.Qb) == (m,)
    @assert size(work.x) == (n,)
    @assert size(work.w) == (n,)
    @assert size(work.zz) == (m,)
    @assert size(work.idx) == (n,)
end

function largest_positive_dual(
        w::AbstractVector{T},
        idx::AbstractVector{TI},
        range
    ) where {T, TI}
    wmax = zero(T)
    izmax = 0
    @inbounds for i in range
        j = idx[i]
        if w[j] > wmax
            wmax = w[j]
            izmax = i
        end
    end
    wmax, izmax
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
        work::NNLSWorkspace{T, TI},
        max_iter::Integer = 3*size(work.QA, 2)
    ) where {T, TI}

    checkargs(work)

    A = work.QA
    b = work.Qb
    x = work.x
    w = work.w
    zz = work.zz
    idx = work.idx
    factor = T(0.01)
    work.mode = 1

    m = size(A, 1)
    n = size(A, 2)

    iter = 0
    fill!(x, zero(T))
    copyto!(idx, 1:n)

    iz2 = n
    iz1 = 1
    iz = 0
    j = 0
    jj = 0
    nsetp = 0
    up = zero(T)

    terminated = false

    # ******  MAIN LOOP BEGINS HERE  ******
    @inbounds while true
        # QUIT IF ALL COEFFICIENTS ARE ALREADY IN THE SOLUTION.
        # OR IF M COLS OF A HAVE BEEN TRIANGULARIZED.
        if (iz1 > iz2 || nsetp >= m)
            terminated = true
            break
        end

        # COMPUTE COMPONENTS OF THE DUAL (NEGATIVE GRADIENT) VECTOR W().
        @inbounds for i in iz1:iz2
            idxi = idx[i]
            sm = zero(T)
            @inbounds @simd for l in (nsetp + 1):m #@avx
                sm += A[l, idxi] * b[l]
            end
            w[idxi] = sm
        end

        @inbounds while true
            # FIND LARGEST POSITIVE W(J).
            wmax, izmax = largest_positive_dual(w, idx, iz1:iz2)

            # IF WMAX .LE. 0. GO TO TERMINATION.
            # THIS INDICATES SATISFACTION OF THE KUHN-TUCKER CONDITIONS.
            if wmax <= 0
                terminated = true
                break
            end

            iz = izmax
            j = idx[iz]

            # THE SIGN OF W(J) IS OK FOR J TO BE MOVED TO SET P.
            # BEGIN THE TRANSFORMATION AND CHECK NEW DIAGONAL ELEMENT TO AVOID
            # NEAR LINEAR DEPENDENCE.
            Asave = A[nsetp + 1, j]
            up = construct_householder!(uview(A, nsetp+1:m, j), up)
            if abs(A[nsetp + 1, j]) > 0
                # COL J IS SUFFICIENTLY INDEPENDENT.  COPY B INTO ZZ, UPDATE ZZ
                # AND SOLVE FOR ZTEST ( = PROPOSED NEW VALUE FOR X(J) ).
                # println("copying b into zz")
                copyto!(zz, b)
                apply_householder!(uview(A, nsetp+1:m, j), up, uview(zz, nsetp+1:m))
                ztest = zz[nsetp + 1] / A[nsetp + 1, j]

                # SEE IF ZTEST IS POSITIVE
                if ztest > 0
                    break
                end
            end

            # REJECT J AS A CANDIDATE TO BE MOVED FROM SET Z TO SET P.
            # RESTORE A(NPP1,J), SET W(J)=0., AND LOOP BACK TO TEST DUAL
            # COEFFS AGAIN.
            A[nsetp + 1, j] = Asave
            w[j] = zero(T)
        end
        if terminated
            break
        end

        # THE INDEX  J=INDEX(IZ)  HAS BEEN SELECTED TO BE MOVED FROM
        # SET Z TO SET P.    UPDATE B,  UPDATE INDICES,  APPLY HOUSEHOLDER
        # TRANSFORMATIONS TO COLS IN NEW SET Z,  ZERO SUBDIAGONAL ELTS IN
        # COL J,  SET W(J)=0.
        copyto!(b, zz)

        idx[iz] = idx[iz1]
        idx[iz1] = j
        iz1 += 1
        nsetp += 1

        if iz1 <= iz2
            apply_householder!(A, up, uview(idx, iz1:iz2), nsetp, j)
        end

        if nsetp != m
            @inbounds @simd ivdep for l in (nsetp + 1):m #@avx
                A[l, j] = zero(T)
            end
        end

        w[j] = zero(T)

        # SOLVE THE TRIANGULAR SYSTEM.
        # STORE THE SOLUTION TEMPORARILY IN ZZ().
        jj = solve_triangular_system!(zz, A, idx, nsetp, jj)

        # ******  SECONDARY LOOP BEGINS HERE ******
        #
        # ITERATION COUNTER.
        @inbounds while true
            iter += 1
            if iter > max_iter
                work.mode = 3
                terminated = true
                # println("NNLS quitting on iteration count")
                break
            end

            # SEE IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE.
            # IF NOT COMPUTE ALPHA.
            alpha = convert(T, 2)
            @inbounds for ip in 1:nsetp
                l = idx[ip]
                if zz[ip] <= 0
                    t = -x[l] / (zz[ip] - x[l])
                    if alpha > t
                        alpha = t
                        jj = ip
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
            @inbounds @simd for ip in 1:nsetp
                l = idx[ip]
                x[l] = x[l] + alpha * (zz[ip] - x[l])
            end

            # MODIFY A AND B AND THE INDEX ARRAYS TO MOVE COEFFICIENT I
            # FROM SET P TO SET Z.
            i = idx[jj]
            @inbounds while true
                x[i] = zero(T)

                if jj != nsetp
                    jj += 1
                    @inbounds for jji in jj:nsetp
                        ii = idx[jji]
                        idx[jji - 1] = ii
                        cc, ss, sig = orthogonal_rotmat(A[jji - 1, ii], A[jji, ii])
                        A[jji - 1, ii] = sig
                        A[jji, ii] = zero(T)

                        # Apply procedure G2 (CC,SS,A(J-1,L),A(J,L))
                        @inbounds @simd for l in 1:ii-1
                            tmp = A[jji-1, l]
                            A[jji-1, l] = cc * tmp + ss * A[jji, l]
                            A[jji, l] = -ss * tmp + cc * A[jji, l]
                        end

                        @inbounds @simd for l in ii+1:n
                            tmp = A[jji-1, l]
                            A[jji-1, l] = cc * tmp + ss * A[jji, l]
                            A[jji, l] = -ss * tmp + cc * A[jji, l]
                        end

                        # Apply procedure G2 (CC,SS,B(J-1),B(J))
                        tmp = b[jji - 1]
                        b[jji - 1] = cc * tmp + ss * b[jji]
                        b[jji] = -ss * tmp + cc * b[jji]
                    end
                end

                nsetp -= 1
                iz1 -= 1
                idx[iz1] = i

                # SEE IF THE REMAINING COEFFS IN SET P ARE FEASIBLE.  THEY SHOULD
                # BE BECAUSE OF THE WAY ALPHA WAS DETERMINED.
                # IF ANY ARE INFEASIBLE IT IS DUE TO ROUND-OFF ERROR.  ANY
                # THAT ARE NONPOSITIVE WILL BE SET TO ZERO
                # AND MOVED FROM SET P TO SET Z.
                allfeasible = true
                @inbounds for jji in 1:nsetp
                    i = idx[jji]
                    if x[i] <= 0
                        allfeasible = false
                        jj = jji
                        break
                    end
                end
                if allfeasible
                    break
                end
            end

            # COPY B( ) INTO ZZ( ).  THEN SOLVE AGAIN AND LOOP BACK.
            copyto!(zz, b)
            jj = solve_triangular_system!(zz, A, idx, nsetp, jj)
        end
        if terminated
            break
        end
        # ******  END OF SECONDARY LOOP  ******

        @inbounds @simd for i in 1:nsetp
            x[idx[i]] = zz[i]
        end
        # ALL NEW COEFFS ARE POSITIVE.  LOOP BACK TO BEGINNING.
    end

    # ******  END OF MAIN LOOP  ******
    # COME TO HERE FOR TERMINATION.
    # COMPUTE THE NORM OF THE FINAL RESIDUAL VECTOR.

    sm = zero(T)
    if nsetp < m
        @inbounds @simd for i in (nsetp + 1):m #@avx
            sm += b[i]^2
        end
    else
        fill!(w, zero(T))
    end
    work.rnorm = sqrt(sm)
    work.nsetp = nsetp
    return work.x
end

function nnls!(
        work::NNLSWorkspace{T},
        A::AbstractMatrix{T},
        b::AbstractVector{T},
        max_iter::Integer = 3*size(A, 2)
    ) where {T}
    load!(work, A, b)
    nnls!(work, max_iter)
    work.x
end

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
        max_iter::Integer = 3*size(A, 2)
    ) where {T}
    work = NNLSWorkspace(A, b)
    nnls!(work, max_iter)
    work.x
end

end # module
