#=
AUTO-GENERATED FILE - DO NOT EDIT

This file is derived from the following fork of the NormalHermiteSplines.jl package:

    https://github.com/jondeuce/NormalHermiteSplines.jl#03e4b80fd1d44a8f2352f47bb3a6f379b52d8f49

As it is not possible to depend on a package fork, the above module is included here verbatim.

The `LICENSE.md` file contents from the original repository follows:

################################################################################

MIT License

Copyright (c) 2021 Igor Kohanovsky

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#

module NormalHermiteSplines

export prepare, construct, interpolate
export evaluate, evaluate!, evaluate_one, evaluate_gradient, evaluate_derivative
export NormalSpline, RK_H0, RK_H1, RK_H2
export get_epsilon, estimate_epsilon, get_cond, estimate_cond, estimate_accuracy

using LinearAlgebra
using StaticArrays
using UnPack
using UnsafeArrays

const AbstractArrOfSVecs{n,T,N} = AbstractArray{SVector{n,T},N}
const AbstractVecOfSVecs{n,T} = AbstractVector{SVector{n,T}}
const VecOfSVecs{n,T} = Vector{SVector{n,T}}

@inline svectors(x::AbstractMatrix{T}) where {T} = reinterpret(reshape, SVector{size(x,1),T}, x)
@inline svectors(x::AbstractVector{T}) where {T} = reinterpret(SVector{1,T}, x)

####
#### ReproducingKernels.jl
####

abstract type ReproducingKernel end
abstract type ReproducingKernel_0 <: ReproducingKernel end
abstract type ReproducingKernel_1 <: ReproducingKernel_0 end
abstract type ReproducingKernel_2 <: ReproducingKernel_1 end

@doc raw"
`struct RK_H0{T} <: ReproducingKernel_0`

Defines a type of reproducing kernel of Bessel Potential space ``H^{n/2 + 1/2}_ε (R^n)`` ('Basic Matérn kernel'):
```math
V(\eta , \xi, \varepsilon) = \exp (-\varepsilon |\xi - \eta|) \, .
```
# Fields
- `ε::T`: 'scaling parameter' from the Bessel Potential space definition,
           it may be omitted in the struct constructor otherwise it must be greater than zero
"
struct RK_H0{T} <: ReproducingKernel_0
    ε::T
    RK_H0() = new{Float64}(0.0)
    RK_H0(ε) = (@assert ε > 0; new{typeof(float(ε))}(float(ε)))
    RK_H0{T}(ε) where {T} = (@assert ε > 0; new{T}(T(ε)))
end
Base.eltype(::RK_H0{T}) where {T} = T

@doc raw"
`struct RK_H1{T} <: ReproducingKernel_1`

Defines a type of reproducing kernel of Bessel Potential space ``H^{n/2 + 3/2}_ε (R^n)`` ('Linear Matérn kernel'):
```math
V(\eta , \xi, \varepsilon) = \exp (-\varepsilon |\xi - \eta|)
             (1 + \varepsilon |\xi  - \eta|) \, .
```
# Fields
- `ε::T`: 'scaling parameter' from the Bessel Potential space definition,
           it may be omitted in the struct constructor otherwise it must be greater than zero
"
struct RK_H1{T} <: ReproducingKernel_1
    ε::T
    RK_H1() = new{Float64}(0.0)
    RK_H1(ε) = (@assert ε > 0; new{typeof(float(ε))}(float(ε)))
    RK_H1{T}(ε) where {T} = (@assert ε > 0; new{T}(T(ε)))
end
Base.eltype(::RK_H1{T}) where {T} = T

@doc raw"
`struct RK_H2{T} <: ReproducingKernel_2`

Defines a type of reproducing kernel of Bessel Potential space ``H^{n/2 + 5/2}_ε (R^n)`` ('Quadratic Matérn kernel'):
```math
V(\eta , \xi, \varepsilon) = \exp (-\varepsilon |\xi - \eta|)
             (3 + 3\varepsilon |\xi  - \eta| + \varepsilon ^2 |\xi - \eta| ^2 ) \, .
```
# Fields
- `ε::T`: 'scaling parameter' from the Bessel Potential space definition,
           it may be omitted in the struct constructor otherwise it must be greater than zero
"
struct RK_H2{T} <: ReproducingKernel_2
    ε::T
    RK_H2() = new{Float64}(0.0)
    RK_H2(ε) = (@assert ε > 0; new{typeof(float(ε))}(float(ε)))
    RK_H2{T}(ε) where {T} = (@assert ε > 0; new{T}(T(ε)))
end
Base.eltype(::RK_H2{T}) where {T} = T

@inline function _rk(kernel::RK_H2, η::SVector, ξ::SVector)
    x = kernel.ε * norm(η - ξ)
    return (3 + x * (3 + x)) * exp(-x)
end

@inline function _rk(kernel::RK_H1, η::SVector, ξ::SVector)
    x = kernel.ε * norm(η - ξ)
    return (1 + x) * exp(-x)
end

@inline function _rk(kernel::RK_H0, η::SVector, ξ::SVector)
    x = kernel.ε * norm(η - ξ)
    return exp(-x)
end

@inline function _∂rk_∂e(kernel::RK_H2, η::SVector, ξ::SVector, e::SVector)
    t = η - ξ
    x = kernel.ε * norm(t)
    return kernel.ε^2 * exp(-x) * (1 + x) * (t ⋅ e)
end

@inline function _∂rk_∂e(kernel::RK_H1, η::SVector, ξ::SVector, e::SVector)
    t = η - ξ
    x = kernel.ε * norm(t)
    return kernel.ε^2 * exp(-x) * (t ⋅ e)
end

@inline function _∂rk_∂η(kernel::RK_H2, η::SVector, ξ::SVector)
    t = η - ξ
    x = kernel.ε * norm(t)
    return -kernel.ε^2 * exp(-x) * (1 + x) * t
end

@inline function _∂rk_∂η(kernel::RK_H1, η::SVector, ξ::SVector)
    t = η - ξ
    x = kernel.ε * norm(t)
    return -kernel.ε^2 * exp(-x) * t
end

@inline function _∂rk_∂η(kernel::RK_H0, η::SVector, ξ::SVector)
    # Note: Derivative of spline built with reproducing kernel RK_H0 does not exist at the spline nodes
    t    = η - ξ
    tnrm = norm(t)
    x    = kernel.ε * tnrm
    ∇    = kernel.ε * exp(-x)
    t    = ifelse(x > eps(typeof(x)), t, zeros(t))
    ∇   *= ifelse(x > eps(typeof(x)), inv(tnrm), one(tnrm))
    ∇   *= t
end

@inline function _∂²rk_∂²e(kernel::RK_H2, η::SVector{n}, ξ::SVector{n}, êη::SVector{n}, êξ::SVector{n}) where {n}
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = norm(t)
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    ∇²    = ((1 + x) * ε²e⁻ˣ) * (êξ ⋅ êη)
    ∇²   -= (ε² * ε²e⁻ˣ) * (êξ ⋅ t) * (t ⋅ êη)
end

@inline function _∂²rk_∂²e(kernel::RK_H1, η::SVector{n}, ξ::SVector{n}, êη::SVector{n}, êξ::SVector{n}) where {n}
    # Note: Second derivative of spline built with reproducing kernel RK_H1 does not exist at the spline nodes
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = norm(t)
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    S     = SMatrix{n,n,typeof(x)}
    ∇²    = ε²e⁻ˣ * (êξ ⋅ êη)
    ∇²   -= ifelse(x > eps(typeof(x)), (ε * ε²e⁻ˣ / tnrm) * (êξ ⋅ t) * (t ⋅ êη), zero(∇²))
end

@inline function _∂²rk_∂η∂ξ(kernel::RK_H2, η::SVector{n}, ξ::SVector{n}) where {n}
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = norm(t)
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    S     = SMatrix{n,n,typeof(x)}
    ∇²    = S(((1 + x) * ε²e⁻ˣ) * I)
    ∇²   -= ((ε² * ε²e⁻ˣ) * t) * t'
end

@inline function _∂²rk_∂η∂ξ(kernel::RK_H1, η::SVector{n}, ξ::SVector{n}) where {n}
    # Note: Second derivative of spline built with reproducing kernel RK_H1 does not exist at the spline nodes
    ε     = kernel.ε
    ε²    = ε * ε
    t     = η - ξ
    tnrm  = norm(t)
    x     = ε * tnrm
    ε²e⁻ˣ = ε² * exp(-x)
    S     = SMatrix{n,n,typeof(x)}
    ∇²    = S(ε²e⁻ˣ * I)
    ∇²   -= ifelse(x > eps(typeof(x)), ((ε * ε²e⁻ˣ / tnrm) * t) * t', zeros(S))
end

####
#### GramMatrix.jl
####

function _gram!(
        A::AbstractMatrix,
        nodes::AbstractVecOfSVecs,
        kernel::ReproducingKernel_0,
    )
    n₁ = length(nodes)
    @inbounds for j in 1:n₁
        for i in j:n₁
            A[i,j] = _rk(kernel, nodes[i], nodes[j])
        end
    end
    return Hermitian(A, :L)
end
function _gram(
        nodes::AbstractVecOfSVecs,
        kernel::ReproducingKernel_0,
    )
    n₁ = length(nodes)
    T  = eltype(eltype(nodes))
    _gram!(zeros(T, n₁, n₁), nodes, kernel)
end

function _gram!(
        A::AbstractMatrix,
        nodes::AbstractVecOfSVecs{n},
        d_nodes::AbstractVecOfSVecs{n},
        d_dirs::AbstractVecOfSVecs{n},
        kernel::ReproducingKernel_1,
    ) where {n}
    n₁  = length(nodes)
    n₂  = length(d_nodes)
    A11 = A
    A21 = uview(A, n₁+1 : n₁+n₂, 1 : n₁)
    A22 = uview(A, n₁+1 : n₁+n₂, n₁+1 : n₁+n₂)

    @inbounds for j in 1:n₁
        # Top-left block (n₁ x n₁)
        for i in j:n₁
            A11[i,j] = _rk(kernel, nodes[i], nodes[j])
        end
        # Bottom-left block (n₂ x n₁)
        for i in 1:n₂
            A21[i,j] = _∂rk_∂e(kernel, nodes[j], d_nodes[i], d_dirs[i])
        end
    end

    # Bottom-right block (n₂ x n₂)
    ε² = kernel.ε^2
    @inbounds for j in 1:n₂
        A22[j, j] = ε²
        for i in j+1:n₂
            A22[i,j] = _∂²rk_∂²e(kernel, d_nodes[j], d_nodes[i], d_dirs[j], d_dirs[i])
        end
    end

    return Hermitian(A, :L)
end
function _gram(
        nodes::AbstractVecOfSVecs{n},
        d_nodes::AbstractVecOfSVecs{n},
        d_dirs::AbstractVecOfSVecs{n},
        kernel::ReproducingKernel_1,
    ) where {n}
    n₁ = length(nodes)
    n₂ = length(d_nodes)
    T  = promote_type(eltype(eltype(nodes)), eltype(eltype(d_nodes)), eltype(eltype(d_dirs)))
    _gram!(zeros(T, n₁+n₂, n₁+n₂), nodes, d_nodes, d_dirs, kernel)
end

"""
    _insert_factor_column(C::LinearAlgebra.Cholesky{T,Matrix{T}}, v::AbstractVector{T}) where {T}

Update the Cholesky factorization `C` as if a row and column were inserted
into the underlying matrix `A`. Specifically, let `C = cholesky(A)` and

    Ã = [A  d]
        [dᵀ γ]

where `v = [d; γ]`.

The corresponding updated cholesky factorization is:

    L̃ = [L   ]
        [eᵀ α]

where `e = L⁻¹d`, `α = √τ`, and `τ = γ - e⋅e > 0`.
If `τ ≤ 0` then `Ã` is not positive definite.

See:
    https://igorkohan.github.io/NormalHermiteSplines.jl/dev/Normal-Splines-Method/#Algorithms-for-updating-Cholesky-factorization
"""
function _insert_factor_column(
        C::LinearAlgebra.Cholesky{T,Matrix{T}},
        v::AbstractVector{T},
    ) where {T}
    @assert size(C,1) == size(C,2) == length(v)-1
    m = size(C,1)
    d = @views v[1:m]
    γ = @inbounds v[m+1]
    e = C.L\d
    τ = max(γ - e⋅e, zero(T))
    α = √τ
    U = UpperTriangular([C.U e; e' α])
    return Cholesky(U, :U, 0)
end

####
#### Splines.jl
####

abstract type AbstractNormalSpline{n,T,RK} end

Base.ndims(::AbstractNormalSpline{n,T,RK}) where {n,T,RK} = n
Base.eltype(::AbstractNormalSpline{n,T,RK}) where {n,T,RK} = T

@inline _get_kernel(spl::AbstractNormalSpline)    = spl._kernel
@inline _get_nodes(spl::AbstractNormalSpline)     = spl._nodes
@inline _get_values(spl::AbstractNormalSpline)    = spl._values
@inline _get_d_nodes(spl::AbstractNormalSpline)   = spl._d_nodes
@inline _get_d_dirs(spl::AbstractNormalSpline)    = spl._d_dirs
@inline _get_d_values(spl::AbstractNormalSpline)  = spl._d_values
@inline _get_mu(spl::AbstractNormalSpline)        = spl._mu
@inline _get_gram(spl::AbstractNormalSpline)      = spl._gram
@inline _get_chol(spl::AbstractNormalSpline)      = spl._chol
@inline _get_cond(spl::AbstractNormalSpline)      = spl._cond
@inline _get_min_bound(spl::AbstractNormalSpline) = spl._min_bound
@inline _get_max_bound(spl::AbstractNormalSpline) = spl._max_bound
@inline _get_scale(spl::AbstractNormalSpline)     = spl._scale

@doc raw"
`struct NormalSpline{n, T <: Real, RK <: ReproducingKernel_0} <: AbstractNormalSpline{n,T,RK}`

Define a structure containing full information of a normal spline
# Fields
- `_kernel`: a reproducing kernel spline was built with
- `_scale`: factor of transforming the original node locations into unit hypercube
- `_nodes`: transformed function value nodes
- `_values`: function values at interpolation nodes
- `_d_nodes`: transformed function directional derivative nodes
- `_d_dirs`: normalized derivative directions
- `_d_values`: function directional derivative values
- `_min_bound`: minimal bounds of the original node locations area
- `_gram`: Gram matrix of the problem
- `_chol`: Cholesky factorization of the Gram matrix
- `_mu`: spline coefficients
- `_cond`: estimation of the Gram matrix condition number
"
struct NormalSpline{n, T <: Real, RK <: ReproducingKernel_0} <: AbstractNormalSpline{n,T,RK}
    _kernel::RK
    _nodes::VecOfSVecs{n,T}
    _values::Vector{T}
    _d_nodes::VecOfSVecs{n,T}
    _d_dirs::VecOfSVecs{n,T}
    _d_values::Vector{T}
    _mu::Vector{T}
    _gram::Hermitian{T, Matrix{T}}
    _chol::Cholesky{T, Matrix{T}}
    _cond::T
    _min_bound::SVector{n,T}
    _max_bound::SVector{n,T}
    _scale::T
end

####
#### Utils.jl
####

@inbounds function _normalize(point::SVector{n}, min_bound::SVector{n}, max_bound::SVector{n}, scale::Real) where {n}
    return (point .- min_bound) ./ scale
    # return clamp.((point .- min_bound) ./ scale, 0, 1) #TODO: clamp nodes? roughly equivalent to nearest neighbour extrapolation
end
@inbounds function _normalize(spl::AbstractNormalSpline{n}, point::SVector{n}) where {n}
    return _normalize(point, _get_min_bound(spl), _get_max_bound(spl), _get_scale(spl))
end

@inbounds function _unnormalize(point::SVector{n}, min_bound::SVector{n}, max_bound::SVector{n}, scale::Real) where {n}
    return min_bound .+ scale .* point
    # return clamp.(min_bound .+ scale .* point, min_bound, max_bound) #TODO: clamp nodes? roughly equivalent to nearest neighbour extrapolation
end
@inbounds function _unnormalize(spl::AbstractNormalSpline{n}, point::SVector{n}) where {n}
    return _unnormalize(point, _get_min_bound(spl), _get_max_bound(spl), _get_scale(spl))
end

function _normalization_scaling(nodes::AbstractVecOfSVecs)
    min_bound = reduce((x, y) -> min.(x, y), nodes; init = fill(+Inf, eltype(nodes)))
    max_bound = reduce((x, y) -> max.(x, y), nodes; init = fill(-Inf, eltype(nodes)))
    scale = maximum(max_bound .- min_bound)
    return min_bound, max_bound, scale
end

function _normalization_scaling(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs)
    min_bound = min.(reduce((x, y) -> min.(x, y), nodes; init = fill(+Inf, eltype(nodes))), reduce((x, y) -> min.(x, y), d_nodes; init = fill(+Inf, eltype(d_nodes))))
    max_bound = max.(reduce((x, y) -> max.(x, y), nodes; init = fill(-Inf, eltype(nodes))), reduce((x, y) -> max.(x, y), d_nodes; init = fill(-Inf, eltype(d_nodes))))
    scale = maximum(max_bound .- min_bound)
    return min_bound, max_bound, scale
end

function _estimate_accuracy(spl::AbstractNormalSpline{n,T,RK}) where {n,T,RK <: ReproducingKernel_0}
    vmax = maximum(abs, _get_values(spl))
    rmae = zero(T)
    @inbounds for i in 1:length(_get_nodes(spl))
        point = _unnormalize(spl, _get_nodes(spl)[i])
        σ     = _evaluate(spl, point)
        rmae  = max(rmae, abs(_get_values(spl)[i] - σ))
    end
    if vmax > 0
        rmae /= vmax
    end
    rmae = max(rmae, eps(T))
    res  = -floor(log10(rmae)) - 1
    res  = max(res, 0)
    return trunc(Int, res)
end

function _pairwise_sum_norms(nodes::AbstractVecOfSVecs{n,T}) where {n,T}
    ℓ = zero(T)
    @inbounds for i in 1:length(nodes), j in i:length(nodes)
        ℓ += norm(nodes[i] .- nodes[j])
    end
    return ℓ
end

function _pairwise_sum_norms_weighted(nodes::AbstractVecOfSVecs{n,T}, d_nodes::AbstractVecOfSVecs{n,T}, w_d_nodes::T) where {n,T}
    ℓ = zero(T)
    @inbounds for i in 1:length(nodes), j in 1:length(d_nodes)
        ℓ += norm(nodes[i] .- w_d_nodes .* d_nodes[j])
    end
    return ℓ
end

@inline _ε_factor(::RK_H0, ε::T) where {T} = one(T)
@inline _ε_factor(::RK_H1, ε::T) where {T} = T(3)/2
@inline _ε_factor(::RK_H2, ε::T) where {T} = T(2)

@inline _ε_factor_d(::RK_H0, ε::T) where {T} = one(T)
@inline _ε_factor_d(::RK_H1, ε::T) where {T} = T(2)
@inline _ε_factor_d(::RK_H2, ε::T) where {T} = T(5)/2

function _estimate_ε(k::ReproducingKernel_0, nodes)
    ε  = _estimate_ε(nodes)
    ε *= _ε_factor(k, ε)
    k  = typeof(k)(ε)
end

function _estimate_ε(k::ReproducingKernel_0, nodes, d_nodes)
    ε  = _estimate_ε(nodes, d_nodes)
    ε *= _ε_factor_d(k, ε)
    k  = typeof(k)(ε)
end

function _estimate_ε(nodes::AbstractVecOfSVecs{n,T}) where {n,T}
    n₁ = length(nodes)
    ε  = _pairwise_sum_norms(nodes)
    return ε > 0 ? ε * T(n)^T(inv(n)) / T(n₁)^(T(5) / 3) : one(T)
end

function _estimate_ε(nodes::AbstractVecOfSVecs{n,T}, d_nodes::AbstractVecOfSVecs{n,T}, w_d_nodes::T=T(0.1)) where {n,T}
    n₁ = length(nodes)
    n₂ = length(d_nodes)
    ε  = _pairwise_sum_norms(nodes) + _pairwise_sum_norms_weighted(nodes, d_nodes, w_d_nodes) + w_d_nodes * _pairwise_sum_norms(d_nodes)
    return ε > 0 ? ε * T(n)^T(inv(n)) / T(n₁ + n₂)^(T(5) / 3) : one(T)
end

function _estimate_epsilon(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    min_bound, max_bound, scale = _normalization_scaling(nodes)
    nodes = _normalize.(nodes, (min_bound,), (max_bound,), scale)
    ε     = _estimate_ε(nodes)
    ε    *= _ε_factor(kernel, ε)
    return ε
end

function _estimate_epsilon(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_1)
    min_bound, max_bound, scale = _normalization_scaling(nodes, d_nodes)
    nodes   = _normalize.(nodes, (min_bound,), (max_bound,), scale)
    d_nodes = _normalize.(d_nodes, (min_bound,), (max_bound,), scale)
    ε       = _estimate_ε(nodes, d_nodes)
    ε      *= _ε_factor_d(kernel, ε)
    return ε
end

function _get_gram(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    min_bound, max_bound, scale = _normalization_scaling(nodes)
    nodes = _normalize.(nodes, (min_bound,), (max_bound,), scale)
    if kernel.ε == 0
        kernel = _estimate_ε(kernel, nodes)
    end
    return _gram(nodes, kernel)
end

function _get_gram(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs, d_dirs::AbstractVecOfSVecs, kernel::ReproducingKernel_1)
    min_bound, max_bound, scale = _normalization_scaling(nodes, d_nodes)
    nodes   = _normalize.(nodes, (min_bound,), (max_bound,), scale)
    d_nodes = _normalize.(d_nodes, (min_bound,), (max_bound,), scale)
    d_dirs  = d_dirs ./ norm.(d_dirs)
    if kernel.ε == 0
        kernel = _estimate_ε(kernel, nodes, d_nodes)
    end
    return _gram(nodes, d_nodes, d_dirs, kernel)
end

function _get_cond(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    T = promote_type(eltype(kernel), eltype(eltype(nodes)))
    gram = _get_gram(nodes, kernel)
    cond = zero(T)
    try
        evs = svdvals!(gram)
        maxevs = maximum(evs)
        minevs = minimum(evs)
        if minevs > 0
            cond = maxevs / minevs
            cond = T(10)^floor(log10(cond))
        end
    catch
    end
    return cond
end

function _get_cond(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs, d_dirs::AbstractVecOfSVecs, kernel::ReproducingKernel_1)
    _get_cond(nodes, kernel)
end

# ```
# Get estimation of the Gram matrix condition number
# Brás, C.P., Hager, W.W. & Júdice, J.J. An investigation of feasible descent algorithms for estimating the condition number of a matrix. TOP 20, 791–809 (2012).
# https://link.springer.com/article/10.1007/s11750-010-0161-9
# ```
function _estimate_cond(
        gram::AbstractMatrix{T},
        chol::LinearAlgebra.Cholesky{T,Matrix{T}},
        nit = 3,
    ) where {T}
    normgram = norm(gram, 1)
    n = size(gram, 1)
    x = fill(inv(T(n)), n)
    z = Vector{T}(undef, n)
    gamma = zero(T)
    for _ in 1:nit
        z = ldiv!(z, chol, x)
        gamma = zero(T)
        @inbounds for i in 1:n
            gamma += abs(z[i])
            z[i] = sign(z[i])
        end
        z = ldiv!(z, chol, copy(z))
        zx = z ⋅ x
        idx = 1
        @inbounds for i in 1:n
            z[i] = abs(z[i])
            if z[i] > z[idx]
                idx = i
            end
        end
        @inbounds begin
            if z[idx] <= zx
                break
            end
            x .= 0
            x[idx] = 1
        end
    end
    cond = T(10)^floor(log10(normgram * gamma))
    return cond
end

####
#### Interpolate.jl
####

function _prepare(nodes::AbstractVecOfSVecs{n,T}, kernel::ReproducingKernel_0) where {n,T}
    # Normalize nodes out-of-place to avoid aliasing
    min_bound, max_bound, scale = _normalization_scaling(nodes)
    nodes = _normalize.(nodes, (min_bound,), (max_bound,), scale)

    if kernel.ε == 0
        kernel = _estimate_ε(kernel, nodes)
    end

    values   = zeros(T, 0)
    d_nodes  = zeros(SVector{n,T}, 0)
    d_dirs   = zeros(SVector{n,T}, 0)
    d_values = zeros(T, 0)
    mu       = zeros(T, 0)
    gram     = _gram(nodes, kernel)
    chol     = cholesky(gram)
    cond     = _estimate_cond(gram, chol)

    return NormalSpline(kernel, nodes, values, d_nodes, d_dirs, d_values, mu, gram, chol, cond, min_bound, max_bound, scale)
end

function _construct!(
        spl::NormalSpline{n,T,RK},
        values::AbstractVector{T},
    ) where {n, T, RK <: ReproducingKernel_0}
    n₁ = length(values)
    length(_get_nodes(spl)) != n₁ && error("Number of data values ($n₁) does not correspond to the number of nodes $(length(_get_nodes(spl))).")
    size(_get_chol(spl)) != (n₁, n₁) && error("Number of data values ($n₁) does not correspond to the size of the Gram matrix ($(size(_get_chol(spl)))).")

    # Resize buffers
    resize!(_get_values(spl), n₁)
    empty!(_get_d_nodes(spl))
    empty!(_get_d_dirs(spl))
    empty!(_get_d_values(spl))
    resize!(_get_mu(spl), n₁)

    # Copy values to avoid aliasing
    _get_values(spl) .= values

    # Compute spline coefficients
    ldiv!(_get_mu(spl), _get_chol(spl), _get_values(spl))

    return spl
end

function _prepare(nodes::AbstractVecOfSVecs{n,T}, d_nodes::AbstractVecOfSVecs{n,T}, d_dirs::AbstractVecOfSVecs{n,T}, kernel::ReproducingKernel_1) where {n,T}
    # Normalize inputs out-of-place to avoid aliasing
    min_bound, max_bound, scale = _normalization_scaling(nodes, d_nodes)
    nodes   = _normalize.(nodes, (min_bound,), (max_bound,), scale)
    d_nodes = _normalize.(d_nodes, (min_bound,), (max_bound,), scale)
    d_dirs  = d_dirs ./ norm.(d_dirs)

    if kernel.ε == 0
        kernel = _estimate_ε(kernel, nodes, d_nodes)
    end

    values   = zeros(T, 0)
    d_values = zeros(T, 0)
    mu       = zeros(T, 0)
    gram     = _gram(nodes, d_nodes, d_dirs, kernel)
    chol     = cholesky(gram)
    cond     = _estimate_cond(gram, chol)

    return NormalSpline(kernel, nodes, values, d_nodes, d_dirs, d_values, mu, gram, chol, cond, min_bound, max_bound, scale)
end

function _construct!(
        spl::NormalSpline{n,T,RK},
        values::AbstractVector{T},
        d_values::AbstractVector{T},
    ) where {n, T, RK <: ReproducingKernel_0}
    n₁ = length(values)
    n₂ = length(d_values)
    length(_get_nodes(spl)) != n₁ && error("Number of data values ($n₁) does not correspond to the number of nodes $(length(_get_nodes(spl))).")
    length(_get_d_nodes(spl)) != n₂ && error("Number of derivative values ($n₂) does not correspond to the number of derivative nodes.")
    size(_get_chol(spl)) != (n₁+n₂, n₁+n₂) && error("Number of data and derivative values ($(n₁+n₂)) do not correspond to the size of the Gram matrix ($(size(_get_chol(spl)))).")

    # Resize buffers
    resize!(_get_values(spl), n₁)
    resize!(_get_d_nodes(spl), n₂)
    resize!(_get_d_dirs(spl), n₂)
    resize!(_get_d_values(spl), n₂)
    resize!(_get_mu(spl), n₁+n₂)

    # Copy values to avoid aliasing
    _get_values(spl) .= values

    # Nodes scaled down by `scale` -> directional derivative scaled up by `scale`; allocate new array to avoid aliasing
    _get_d_values(spl) .= _get_scale(spl) .* d_values

    # Compute spline coefficients and construct spline
    ldiv!(_get_mu(spl), _get_chol(spl), [_get_values(spl); _get_d_values(spl)])

    return spl
end

#### Evaluation

@inline function _evaluate!(
        vals::AbstractArray{<:Any,N},
        spl::AbstractNormalSpline{n,<:Any,<:ReproducingKernel_0},
        points::AbstractArrOfSVecs{n,<:Any,N},
    ) where {n, N}
    @inbounds for i in 1:length(points)
        vals[i] = _evaluate(spl, points[i])
    end
    return vals
end

@inline function _evaluate(
        spl::AbstractNormalSpline{n,<:Any,RK},
        x::SVector{n},
    ) where {n, RK <: ReproducingKernel_0}
    kernel, nodes, d_nodes, d_dirs, mu =
        _get_kernel(spl), _get_nodes(spl), _get_d_nodes(spl), _get_d_dirs(spl), _get_mu(spl)
    n₁ = length(nodes)
    n₂ = length(d_nodes)
    x  = _normalize(spl, x)
    v  = zero(promote_type(eltype(spl), eltype(kernel), eltype(x)))
    @inbounds for i in 1:n₁
        v += mu[i] * _rk(kernel, x, nodes[i])
    end
    if RK <: ReproducingKernel_1
        @inbounds for i in 1:n₂
            v += mu[i+n₁] * _∂rk_∂e(kernel, x, d_nodes[i], d_dirs[i])
        end
    end
    return v
end

@inline function _evaluate_gradient(
        spl::AbstractNormalSpline{n,<:Any,RK},
        x::SVector{n},
    ) where {n, RK <: ReproducingKernel_0}
    kernel, nodes, d_nodes, d_dirs, mu, scale =
        _get_kernel(spl), _get_nodes(spl), _get_d_nodes(spl), _get_d_dirs(spl), _get_mu(spl), _get_scale(spl)
    n₁ = length(nodes)
    n₂ = length(d_nodes)
    x  = _normalize(spl, x)
    ∇  = zero(SVector{n,promote_type(eltype(spl), eltype(kernel), eltype(x))})
    @inbounds for i in 1:n₁
        ∇ += mu[i] * _∂rk_∂η(kernel, x, nodes[i])
    end
    if RK <: ReproducingKernel_1
        @inbounds for i in 1:n₂
            ∇² = _∂²rk_∂η∂ξ(kernel, x, d_nodes[i])
            ∇ += mu[i+n₁] * (∇² * d_dirs[i])
        end
    end
    return ∇ ./ scale
end

####
#### Public API
####

#### ReproducingKernel_0

"""
`prepare(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Prepare the spline by constructing and factoring a Gram matrix of the interpolation problem.
Initialize the `NormalSpline` object.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
           `n₁` is the number of function value nodes. It means that each column in the matrix defines one node.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the partly initialized `NormalSpline` object that must be passed to `construct` function
        in order to complete the spline initialization.
"""
@inline function prepare(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return prepare(svectors(nodes), kernel)
end
@inline function prepare(nodes::AbstractVecOfSVecs{n,T}, kernel::RK = RK_H0()) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _prepare(nodes, kernel)
end

"""
`construct(spline::AbstractNormalSpline{n,T,RK}, values::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Construct the spline by calculating its coefficients and completely initializing the `NormalSpline` object.
# Arguments
- `spline`: the partly initialized `NormalSpline` object returned by `prepare` function.
- `values`: function values at `nodes` nodes.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function
        to interpolate the data to required points.
"""
@inline function construct(spline::AbstractNormalSpline{n,T,RK}, values::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _construct!(deepcopy(spline), values)
end

"""
`interpolate(nodes::AbstractMatrix{T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Prepare and construct the spline.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
           and `n₁` is the number of function value nodes.
           It means that each column in the matrix defines one node.
- `values`: function values at `nodes` nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
@inline function interpolate(nodes::AbstractMatrix{T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return interpolate(svectors(nodes), values, kernel)
end
@inline function interpolate(nodes::AbstractVecOfSVecs{n,T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {n, T <: Real, RK <: ReproducingKernel_0}
    spline = _prepare(nodes, kernel)
    return _construct!(spline, values)
end

"""
`evaluate(spline::AbstractNormalSpline{n,T,RK}, points::AbstractMatrix{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Evaluate the spline values at the locations defined in `points`.

# Arguments
- `spline: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `points`: locations at which spline values are evaluating.
            This should be an `n×m` matrix, where `n` is dimension of the sampled space
            and `m` is the number of locations where spline values are evaluating.
            It means that each column in the matrix defines one location.

Return: `Vector{T}` of the spline values at the locations defined in `points`.
"""
@inline function evaluate(spline::AbstractNormalSpline{n,T,RK}, points::AbstractMatrix{T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return evaluate(spline, svectors(points))
end
@inline function evaluate(spline::AbstractNormalSpline{n,T,RK}, points::AbstractArrOfSVecs{n,T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return evaluate!(zeros(T, size(points)), spline, points)
end
@inline function evaluate!(spline_values::AbstractArray{T,N}, spline::AbstractNormalSpline{n,T,RK}, points::AbstractArrOfSVecs{n,T,N}) where {n, T <: Real, N, RK <: ReproducingKernel_0}
    return _evaluate!(spline_values, spline, points)
end
@inline function evaluate(spline::AbstractNormalSpline{n,T,RK}, point::SVector{n,T}) where {n, T <: Real, N, RK <: ReproducingKernel_0}
    return _evaluate(spline, point)
end

"""
`evaluate_one(spline::AbstractNormalSpline{n,T,RK}, point::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Evaluate the spline value at the `point` location.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `point`: location at which spline value is evaluating.
           This should be a vector of size `n`, where `n` is dimension of the sampled space.

Return: the spline value at the location defined in `point`.
"""
@inline function evaluate_one(spline::AbstractNormalSpline{n,T,RK}, point::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return evaluate_one(spline, SVector{n,T}(ntuple(i -> point[i], n)))
end
@inline function evaluate_one(spline::AbstractNormalSpline{n,T,RK}, point::SVector{n,T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return evaluate(spline, point)
end

"""
`evaluate_gradient(spline::AbstractNormalSpline{n,T,RK}, point::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Evaluate gradient of the spline at the location defined in `point`.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `point`: location at which gradient value is evaluating.
           This should be a vector of size `n`, where `n` is dimension of the sampled space.

Note: Gradient of spline built with reproducing kernel RK_H0 does not exist at the spline nodes.

Return: `Vector{T}` - gradient of the spline at the location defined in `point`.
"""
@inline function evaluate_gradient(spline::AbstractNormalSpline{n,T,RK}, point::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return evaluate_gradient(spline, SVector{n,T}(ntuple(i -> point[i], n)))
end
@inline function evaluate_gradient(spline::AbstractNormalSpline{n,T,RK}, point::SVector{n,T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _evaluate_gradient(spline, point)
end

#### ReproducingKernel_1

"""
`prepare(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, d_dirs::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Prepare the spline by constructing and factoring a Gram matrix of the interpolation problem.
Initialize the `NormalSpline` object.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
           `n₁` is the number of function value nodes.
            It means that each column in the matrix defines one node.
- `d_nodes`: The function directional derivatives nodes.
             This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
             `n₂` is the number of function directional derivative nodes.
- `d_dirs`: Directions of the function directional derivatives.
        This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
        `n₂` is the number of function directional derivative nodes.
        It means that each column in the matrix defines one direction of the function directional derivative.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the partly initialized `NormalSpline` object that must be passed to `construct` function
        in order to complete the spline initialization.
"""
@inline function prepare(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, d_dirs::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return prepare(svectors(nodes), svectors(d_nodes), svectors(d_dirs), kernel)
end
@inline function prepare(nodes::AbstractVecOfSVecs{n,T}, d_nodes::AbstractVecOfSVecs{n,T}, d_dirs::AbstractVecOfSVecs{n,T}, kernel::RK = RK_H1()) where {n, T <: Real, RK <: ReproducingKernel_1}
    return _prepare(nodes, d_nodes, d_dirs, kernel)
end

"""
`construct(spline::AbstractNormalSpline{n,T,RK}, values::AbstractVector{T}, d_values::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_1}`

Construct the spline by calculating its coefficients and completely initializing the `NormalSpline` object.
# Arguments
- `spline`: the partly initialized `NormalSpline` object returned by `prepare` function.
- `values`: function values at `nodes` nodes.
- `d_values`: function directional derivative values at `d_nodes` nodes.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function
        to interpolate the data to required points.
"""
@inline function construct(spline::AbstractNormalSpline{n,T,RK}, values::AbstractVector{T}, d_values::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_1}
    return _construct!(deepcopy(spline), values, d_values)
end

"""
`interpolate(nodes::AbstractMatrix{T}, values::AbstractVector{T}, d_nodes::AbstractMatrix{T}, d_dirs::AbstractMatrix{T}, d_values::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Prepare and construct the spline.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
           and `n₁` is the number of function value nodes.
           It means that each column in the matrix defines one node.
- `values`: function values at `nodes` nodes.
- `d_nodes`: The function directional derivative nodes.
            This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
            `n₂` is the number of function directional derivative nodes.
- `d_dirs`: Directions of the function directional derivatives.
       This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
       `n₂` is the number of function directional derivative nodes.
       It means that each column in the matrix defines one direction of the function directional derivative.
- `d_values`: function directional derivative values at `d_nodes` nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
@inline function interpolate(nodes::AbstractMatrix{T}, values::AbstractVector{T}, d_nodes::AbstractMatrix{T}, d_dirs::AbstractMatrix{T}, d_values::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return interpolate(svectors(nodes), values, svectors(d_nodes), svectors(d_dirs), d_values, kernel)
end
@inline function interpolate(nodes::AbstractVecOfSVecs{n,T}, values::AbstractVector{T}, d_nodes::AbstractVecOfSVecs{n,T}, d_dirs::AbstractVecOfSVecs{n,T}, d_values::AbstractVector{T}, kernel::RK = RK_H1()) where {n, T <: Real, RK <: ReproducingKernel_1}
    spline = _prepare(nodes, d_nodes, d_dirs, kernel)
    return _construct!(spline, values, d_values)
end

#### ReproducingKernel_0 (1-dimensional case)

"""
`prepare(nodes::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Prepare the 1D spline by constructing and factoring a Gram matrix of the interpolation problem.
Initialize the `NormalSpline` object.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n₁` vector where `n₁` is the number of function value nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the partly initialized `NormalSpline` object that must be passed to `construct` function
        in order to complete the spline initialization.
"""
@inline function prepare(nodes::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return prepare(svectors(nodes), kernel)
end

"""
`interpolate(nodes::AbstractVector{T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Prepare and construct the 1D spline.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n₁` vector where `n₁` is the number of function value nodes.
- `values`: function values at `n₁` interpolation nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
@inline function interpolate(nodes::AbstractVector{T}, values::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    interpolate(svectors(nodes), values, kernel)
end

"""
`evaluate(spline::AbstractNormalSpline{n,T,RK}, points::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Evaluate the 1D spline values/value at the `points` locations.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `points`: locations at which spline values are evaluating.
            This should be a vector of size `m` where `m` is the number of evaluating points.

Return: spline value at the `point` location.
"""
@inline function evaluate(spline::AbstractNormalSpline{n,T,RK}, points::AbstractVector{T}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return evaluate(spline, svectors(points))
end

"""
`evaluate_one(spline::AbstractNormalSpline{1,T,RK}, point::T) where {T <: Real, RK <: ReproducingKernel_0}`

Evaluate the 1D spline value at the `point` location.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `point`: location at which spline value is evaluating.

Return: spline value at the `point` location.
"""
@inline function evaluate_one(spline::AbstractNormalSpline{1,T,RK}, point::T) where {T <: Real, RK <: ReproducingKernel_0}
    return evaluate(spline, SVector{1,T}((point,)))
end

"""
`evaluate_derivative(spline::AbstractNormalSpline{1,T,RK}, point::T) where {T <: Real, RK <: ReproducingKernel_0}`

Evaluate the 1D spline derivative at the `point` location.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `point`: location at which spline derivative is evaluating.

Note: Derivative of spline built with reproducing kernel RK_H0 does not exist at the spline nodes.

Return: the spline derivative value at the `point` location.
"""
@inline function evaluate_derivative(spline::AbstractNormalSpline{1,T,RK}, point::T) where {T <: Real, RK <: ReproducingKernel_0}
    return evaluate_gradient(spline, SVector{1,T}((point,)))[1]
end

#### ReproducingKernel_1 (1-dimensional case)

"""
`prepare(nodes::AbstractVector{T}, d_nodes::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Prepare the 1D interpolating normal spline by constructing and factoring a Gram matrix of the problem.
Initialize the `NormalSpline` object.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n₁` vector where `n₁` is the number of function value nodes.
- `d_nodes`: The function derivatives nodes.
             This should be an `n₂` vector where `n₂` is the number of function derivatives nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the partly initialized `NormalSpline` object that must be passed to `construct` function
        in order to complete the spline initialization.
"""
@inline function prepare(nodes::AbstractVector{T}, d_nodes::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    d_dirs = fill(ones(SVector{1,T}), length(d_nodes))
    return prepare(svectors(nodes), svectors(d_nodes), d_dirs, kernel)
end

"""
`interpolate(nodes::AbstractVector{T}, values::AbstractVector{T}, d_nodes::AbstractVector{T}, d_values::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Prepare and construct the 1D interpolating normal spline.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n₁` vector where `n₁` is the number of function value nodes.
- `values`: function values at `nodes` nodes.
- `d_nodes`: The function derivatives nodes.
             This should be an `n₂` vector where `n₂` is the number of function derivatives nodes.
- `d_values`: function derivative values at `d_nodes` nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
@inline function interpolate(nodes::AbstractVector{T}, values::AbstractVector{T}, d_nodes::AbstractVector{T}, d_values::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    d_dirs = fill(ones(SVector{1,T}), length(d_nodes))
    return interpolate(svectors(nodes), values, svectors(d_nodes), d_dirs, d_values, kernel)
end

#### Utils for general case

"""
`get_epsilon(spline::AbstractNormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Get the 'scaling parameter' of Bessel Potential space the spline was built in.
# Arguments
- `spline`: the `NormalSpline` object returned by `prepare`, `construct` or `interpolate` function.

Return: `ε` - the 'scaling parameter'.
"""
@inline function get_epsilon(spline::AbstractNormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _get_kernel(spline).ε
end

"""
`estimate_epsilon(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Get the estimation of the 'scaling parameter' of Bessel Potential space the spline being built in.
It coincides with the result returned by `get_epsilon` function.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
           and `n₁` is the number of function value nodes.
           It means that each column in the matrix defines one node.
- `kernel`: reproducing kernel of Bessel potential space the normal spline will be constructed in.
           It must be a struct object of the following type:
             `RK_H0` if the spline is constructing as a continuous function,
             `RK_H1` if the spline is constructing as a differentiable function,
             `RK_H2` if the spline is constructing as a twice differentiable function.
Return: estimation of `ε`.
"""
@inline function estimate_epsilon(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return estimate_epsilon(svectors(nodes), kernel)
end
@inline function estimate_epsilon(nodes::AbstractVecOfSVecs{n,T}, kernel::RK = RK_H0()) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _estimate_epsilon(nodes, kernel)
end

"""
`estimate_epsilon(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Get an the estimation of the 'scaling parameter' of Bessel Potential space the spline being built in.
It coincides with the result returned by `get_epsilon` function.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
           and `n₁` is the number of function value nodes.
           It means that each column in the matrix defines one node.
- `d_nodes`: The function directional derivative nodes.
           This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
           `n₂` is the number of function directional derivative nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline will be constructed in.
            It must be a struct object of the following type:
            `RK_H1` if the spline is constructing as a differentiable function,
            `RK_H2` if the spline is constructing as a twice differentiable function.

Return: estimation of `ε`.
"""
@inline function estimate_epsilon(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return estimate_epsilon(svectors(nodes), svectors(d_nodes), kernel)
end
@inline function estimate_epsilon(nodes::AbstractVecOfSVecs{n,T}, d_nodes::AbstractVecOfSVecs{n,T}, kernel::RK = RK_H1()) where {n, T <: Real, RK <: ReproducingKernel_1}
    return _estimate_epsilon(nodes, d_nodes, kernel)
end

"""
`estimate_cond(spline::AbstractNormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Get an estimation of the Gram matrix condition number. It needs the `spline` object is prepared and requires O(N^2) operations.
(C. Brás, W. Hager, J. Júdice, An investigation of feasible descent algorithms for estimating the condition number of a matrix. TOP Vol.20, No.3, 2012.)
# Arguments
- `spline`: the `NormalSpline` object returned by `prepare`, `construct` or `interpolate` function.

Return: an estimation of the Gram matrix condition number.
"""
@inline function estimate_cond(spline::AbstractNormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel}
    return _get_cond(spline)
end

"""
`estimate_accuracy(spline::AbstractNormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel_0}`

Assess accuracy of interpolation results by analyzing residuals.
# Arguments
- `spline`: the `NormalSpline` object returned by `construct` or `interpolate` function.

Return: an estimation of the number of significant digits in the interpolation result.
"""
@inline function estimate_accuracy(spline::AbstractNormalSpline{n,T,RK}) where {n, T <: Real, RK <: ReproducingKernel_0}
    return _estimate_accuracy(spline)
end

"""
`get_cond(nodes::AbstractMatrix{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Get a value of the Gram matrix spectral condition number. It is obtained by means of the matrix SVD decomposition and requires ``O(N^3)`` operations.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
           `n₁` is the number of function value nodes. It means that each column in the matrix defines one node.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: a value of the Gram matrix spectral condition number.
"""
@inline function get_cond(nodes::AbstractMatrix{T}, kernel::RK) where {T <: Real, RK <: ReproducingKernel_0}
    return get_cond(svectors(nodes), kernel)
end
@inline function get_cond(nodes::AbstractVecOfSVecs{n,T}, kernel::RK) where {n, T <: Real, RK <: ReproducingKernel_0}
    return get_cond(nodes, kernel)
end

"""
`get_cond(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, d_dirs::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Get a value of the Gram matrix spectral condition number. It is obtained by means of the matrix SVD decomposition and requires ``O(N^3)`` operations.
# Arguments
- `nodes`: The function value nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
           `n₁` is the number of function value nodes.
            It means that each column in the matrix defines one node.
- `d_nodes`: The function directional derivatives nodes.
             This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
             `n₂` is the number of function directional derivative nodes.
- `d_dirs`: Directions of the function directional derivatives.
        This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
        `n₂` is the number of function directional derivative nodes.
        It means that each column in the matrix defines one direction of the function directional derivative.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: a value of the Gram matrix spectral condition number.
"""
@inline function get_cond(nodes::AbstractMatrix{T}, d_nodes::AbstractMatrix{T}, d_dirs::AbstractMatrix{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return get_cond(svectors(nodes), svectors(d_nodes), svectors(d_dirs), kernel)
end
@inline function get_cond(nodes::AbstractVecOfSVecs{n,T}, d_nodes::AbstractVecOfSVecs{n,T}, d_dirs::AbstractVecOfSVecs{n,T}, kernel::RK = RK_H1()) where {n, T <: Real, RK <: ReproducingKernel_1}
    return get_cond(nodes, d_nodes, d_dirs, kernel)
end

#### Utils for 1-dimensional case

"""
`estimate_epsilon(nodes::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}`

Get an the estimation of the 'scaling parameter' of Bessel Potential space the 1D spline is being built in.
It coincides with the result returned by `get_epsilon` function.
# Arguments
- `nodes`: The function value nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: estimation of `ε`.
"""
@inline function estimate_epsilon(nodes::AbstractVector{T}, kernel::RK = RK_H0()) where {T <: Real, RK <: ReproducingKernel_0}
    return estimate_epsilon(svectors(nodes), kernel)
end

"""
`estimate_epsilon(nodes::AbstractVector{T}, d_nodes::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}`

Get an the estimation of the 'scaling parameter' of Bessel Potential space the 1D spline is being built in.
It coincides with the result returned by `get_epsilon` function.
# Arguments
- `nodes`: The function value nodes.
- `d_nodes`: The function derivative nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: estimation of `ε`.
"""
@inline function estimate_epsilon(nodes::AbstractVector{T}, d_nodes::AbstractVector{T}, kernel::RK = RK_H1()) where {T <: Real, RK <: ReproducingKernel_1}
    return estimate_epsilon(svectors(nodes), svectors(d_nodes), kernel)
end

end # module NormalHermiteSplines
