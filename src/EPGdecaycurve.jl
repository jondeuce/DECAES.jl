#=
Notation shared by the kernels in this file, which all evaluate the odd-state shifted MPSV recursion Mₙ = S²𝒜ₙ Mₙ₋₁ of the CPMG extended phase graph,
where 𝒜ₙ = E Rₙ E is the relaxation-rotation-relaxation core of pulse n, and S² is the double phase state shift:
E₁, E₂          relaxation factors over TE/2
Ω = (F⁺, F⁻, Z) an odd phase state; F⁺, F⁻, Z also name the state components and, with a subscript, the ping-pong buffers holding them (₁ = written, ₂ = read)
n               pulse and echo index; pulse n has angle αₙ = κₙ α, with κₙ independent of α
j, L            state index, and the number L = min(n, ETL - n + 1) of input states that can reach the readout by echo ETL
x, y, z         F⁺ + F⁻, F⁺ - F⁻, Z
a, b, c, d      coefficients of 𝒜ₙ: E₂²/2, a cos(αₙ), E₁E₂ sin(αₙ), E₁² cos(αₙ); c′ = -c/2, and a⁺, a⁻ = a + b, a - b = E₂² cos²(αₙ/2), E₂² sin²(αₙ/2)
𝒜⁺, 𝒜⁻, 𝒜ᶻ      rows of 𝒜ₙ, so that (Ω⁺, Ω⁻, Ωᶻ) = (𝒜⁺⋅Ω, 𝒜⁻⋅Ω, 𝒜ᶻ⋅Ω) = (a x + u, a x - u, d z + c′ y) with u = b y + c z
h               unit-excitation response, equal to the readout F⁺[1] after each refocusing pulse
m₀              excitation amplitude sin(α/2); the decay curve is |s| with s = m₀ h and σ = sgn(s), so that ∂A = σṡ and ∂²A = σs̈
l, W            lane index, and the number W = EPG_BATCH_WIDTH of lanes of the batched kernels, one T2 time per lane
k, Â            cosine series degree and coefficients, h = Σₖ Âₖ cos kα
A, ∂A, ∂²A      a decay curve, or decay basis with one curve per T2 column, and its first two α-derivatives
ḣ, ḧ            first and second α-derivatives of h, and likewise ṁ₀, Ḟ⁺, ẋ, ...
=#

####
#### Helper functions
####

# Rotation matrix R_x(α) = R(α, 0) for rotating the complex magnetization vector representation (M⁺, M⁻, Mz) about the x-axis by angle α (radians), where M⁺ = Mx + iMy and M⁻ = Mx - iMy.
# Equivalently, rotates the magnetization phase state vector (MPSV) of Fourier coefficients (Fₖ⁺, Fₖ⁻, Zₖ) = (Fₖ, F̄₋ₖ, Zₖ).
@inline element_flipmat(α::T) where {T} = SA{Complex{T}}[
    cos(α / 2)^2 sin(α / 2)^2 -im*sin(α);
    sin(α / 2)^2 cos(α / 2)^2 im*sin(α);
    -im*sin(α)/2 im*sin(α)/2 cos(α)]

# Rotation matrix R(α, φ) for rotating the MPSV by angle α (radians) about an axis having an angle φ (radians) with the x-axis.
@inline element_flipmat(α::T, φ::T) where {T} = SA{Complex{T}}[
    cos(α / 2)^2 cis(2φ)*sin(α / 2)^2 -im*cis(φ)*sin(α);
    cis(-2φ)*sin(α / 2)^2 cos(α / 2)^2 im*cis(-φ)*sin(α);
    -im*cis(-φ)*sin(α)/2 im*cis(φ)*sin(α)/2 cos(α)]

# Real representation for the rotation matrix R(α, 90) for the Carr-Purcell (CP/anti-CPMG) pulse sequence following excitation 90_y.
@inline anti_cpmg_flipmat(α::T) where {T} = SA{T}[
    cos(α / 2)^2 -sin(α / 2)^2 sin(α);
    -sin(α / 2)^2 cos(α / 2)^2 sin(α);
    -sin(α)/2 -sin(α)/2 cos(α)]

# Real representation for the rotation matrix R_x(α) for the Carr-Purcell-Meiboom-Gill (CPMG) pulse sequence following excitation 90_y and change of variables Mz = i * Mz′.
@inline cpmg_flipmat(α::T) where {T} = SA{T}[
    cos(α / 2)^2 sin(α / 2)^2 sin(α);
    sin(α / 2)^2 cos(α / 2)^2 -sin(α);
    -sin(α)/2 sin(α)/2 cos(α)]

####
#### EPG parameterizations
####

struct EPGOptions{T, ETL}
    ETL::ETL
    α::T
    TE::T
    T2::T
    T1::T
    β::T
    EPGOptions{T, ETL}(etl, α, TE, T2, T1, β) where {T, ETL} = new{T, ETL}(ETL(etl), T(α), T(TE), T(T2), T(T1), T(β))
    function EPGOptions(etl, α, TE, T2, T1, β)
        α, TE, T2, T1, β = promote(float(α), float(TE), float(T2), float(T1), float(β))
        return new{typeof(α), typeof(etl)}(etl, α, TE, T2, T1, β)
    end
end
@inline Base.Tuple(θ::EPGOptions) = (θ.ETL, θ.α, θ.TE, θ.T2, θ.T1, θ.β)
@inline Base.NamedTuple(θ::EPGOptions) = NamedTuple{(:ETL, :α, :TE, :T2, :T1, :β)}(Tuple(θ))
@inline EPGOptions(θ::NamedTuple{(:ETL, :α, :TE, :T2, :T1, :β)}) = EPGOptions(Tuple(θ)...)

@inline Base.eltype(::EPGOptions{T}) where {T} = T
@inline echotrainlength(θ::EPGOptions) = θ.ETL
@inline B1correctedflipangle(θ::EPGOptions{T}, n::Int) where {T} = ifelse(n == 0, θ.α / 2, ifelse(n == 1, θ.α, θ.α * θ.β / T(π))) # B1-corrected pulse sequence: α/2, α, αβ/π, αβ/π, ...
@inline echotime(θ::EPGOptions) = θ.TE
@inline T2time(θ::EPGOptions) = θ.T2
@inline T1time(θ::EPGOptions) = θ.T1

struct EPGConstantFlipAngleOptions{T, ETL}
    ETL::ETL
    α::T
    TE::T
    T2::T
    T1::T
    EPGConstantFlipAngleOptions{T, ETL}(etl, α, TE, T2, T1) where {T, ETL} = new{T, ETL}(ETL(etl), T(α), T(TE), T(T2), T(T1))
    function EPGConstantFlipAngleOptions(etl, α, TE, T2, T1)
        α, TE, T2, T1 = promote(float(α), float(TE), float(T2), float(T1))
        return new{typeof(α), typeof(etl)}(etl, α, TE, T2, T1)
    end
end
@inline Base.Tuple(θ::EPGConstantFlipAngleOptions) = (θ.ETL, θ.α, θ.TE, θ.T2, θ.T1)
@inline Base.NamedTuple(θ::EPGConstantFlipAngleOptions) = NamedTuple{(:ETL, :α, :TE, :T2, :T1)}(Tuple(θ))
@inline EPGConstantFlipAngleOptions(θ::NamedTuple{(:ETL, :α, :TE, :T2, :T1)}) = EPGConstantFlipAngleOptions(Tuple(θ)...)

@inline Base.eltype(::EPGConstantFlipAngleOptions{T}) where {T} = T
@inline echotrainlength(θ::EPGConstantFlipAngleOptions) = θ.ETL
@inline B1correctedflipangle(θ::EPGConstantFlipAngleOptions{T}, n::Int) where {T} = ifelse(n == 0, θ.α / 2, θ.α) # B1-corrected pulse sequence: α/2, α, α, α, ...
@inline echotime(θ::EPGConstantFlipAngleOptions) = θ.TE
@inline T2time(θ::EPGConstantFlipAngleOptions) = θ.T2
@inline T1time(θ::EPGConstantFlipAngleOptions) = θ.T1

struct EPGIncreasingFlipAnglesOptions{T, ETL}
    ETL::ETL
    α::T
    α1::T
    α2::T
    TE::T
    T2::T
    T1::T
    EPGIncreasingFlipAnglesOptions{T, ETL}(etl, α, α1, α2, TE, T2, T1) where {T, ETL} = new{T, ETL}(ETL(etl), T(α), T(α1), T(α2), T(TE), T(T2), T(T1))
    function EPGIncreasingFlipAnglesOptions(etl, α, α1, α2, TE, T2, T1)
        α, α1, α2, TE, T2, T1 = promote(float(α), float(α1), float(α2), float(TE), float(T2), float(T1))
        return new{typeof(α), typeof(etl)}(etl, α, α1, α2, TE, T2, T1)
    end
end
@inline Base.Tuple(θ::EPGIncreasingFlipAnglesOptions) = (θ.ETL, θ.α, θ.α1, θ.α2, θ.TE, θ.T2, θ.T1)
@inline Base.NamedTuple(θ::EPGIncreasingFlipAnglesOptions) = NamedTuple{(:ETL, :α, :α1, :α2, :TE, :T2, :T1)}(Tuple(θ))
@inline EPGIncreasingFlipAnglesOptions(θ::NamedTuple{(:ETL, :α, :α1, :α2, :TE, :T2, :T1)}) = EPGIncreasingFlipAnglesOptions(Tuple(θ)...)

@inline Base.eltype(::EPGIncreasingFlipAnglesOptions{T}) where {T} = T
@inline echotrainlength(θ::EPGIncreasingFlipAnglesOptions) = θ.ETL
@inline B1correctedflipangle(θ::EPGIncreasingFlipAnglesOptions{T}, n::Int) where {T} = ifelse(n == 0, θ.α / 2, ifelse(n == 1, θ.α * θ.α1 / T(π), ifelse(n == 2, θ.α * θ.α2 / T(π), θ.α))) # B1-corrected pulse sequence: α/2, αα₁/π, αα₂/π, α, α, ...
@inline echotime(θ::EPGIncreasingFlipAnglesOptions) = θ.TE
@inline T2time(θ::EPGIncreasingFlipAnglesOptions) = θ.T2
@inline T1time(θ::EPGIncreasingFlipAnglesOptions) = θ.T1

const EPGParameterization{T, ETL} = Union{
    EPGOptions{T, ETL},
    EPGConstantFlipAngleOptions{T, ETL},
    EPGIncreasingFlipAnglesOptions{T, ETL},
}

#### Destructuring/restructuring to/from vectors

struct SymbolVector{Fs} <: AbstractVector{Symbol}
    fields::Val{Fs}
end
@inline Base.Tuple(::SymbolVector{Fs}) where {Fs} = Fs
@inline Base.length(::SymbolVector{Fs}) where {Fs} = length(Fs)
@inline Base.size(::SymbolVector{Fs}) where {Fs} = (length(Fs),)
@inline Base.getindex(::SymbolVector{Fs}, i::Int) where {Fs} = Fs[i]

@generated function constructorof(::Type{T}) where {T}
    return :($(getfield(parentmodule(T), nameof(T))))
end
@inline constructorof(θ) = constructorof(typeof(θ))

@inline restructure(θ, xs::Tuple) = constructorof(θ)(xs...)
@inline restructure(θ, xs::NamedTuple{Fs}) where {Fs} = restructure(θ, Tuple(xs), Val(Fs))

@generated function restructure(θ, xs, ::Val{Fs}) where {Fs}
    idxmap = NamedTuple{Fs}(ntuple(i -> i, length(Fs)))
    vals = [F ∈ Fs ? :(@inbounds(xs[$(idxmap[F])])) : :(getfield(θ, $(QuoteNode(F)))) for F in fieldsof(θ)]
    return :(Base.@_inline_meta; $restructure(θ, tuple($(vals...))))
end
@inline restructure(θ, xs, Fs::SymbolVector) = restructure(θ, xs, Fs.fields)

@generated function destructure(θ, ::Val{Fs}) where {Fs}
    vals = [:(convert(eltype(θ), getfield(θ, $(QuoteNode(F))))) for F in Fs]
    return :(Base.@_inline_meta; $SVector{$(length(Fs)), eltype(θ)}(tuple($(vals...))))
end
@inline destructure(θ, Fs::SymbolVector) = destructure(θ, Fs.fields)

####
#### Abstract Interface
####

abstract type AbstractEPGWorkspace{T, ETL} end

@inline Base.eltype(::AbstractEPGWorkspace{T}) where {T} = T
@inline echotrainlength(work::AbstractEPGWorkspace{T}) where {T} = work.ETL
@inline decaycurve(work::AbstractEPGWorkspace) = work.A

"""
    EPGdecaycurve_work(θ::EPGParameterization)
    EPGdecaycurve_work(T::Type, ETL::Int)
    EPGdecaycurve_work(T::Type, ::Val{ETL})

Allocate a reusable workspace for [`EPGdecaycurve!`](@ref) with element type `T` and echo train length `ETL`.
Passing `ETL` as a `Val` selects the workspace specialized for a statically known echo train length.

See also:

  - [`EPGdecaycurve!`](@ref)
"""
@inline EPGdecaycurve_work(θ::EPGParameterization{T, ETL}) where {T, ETL} = default_cache(θ)
@inline EPGdecaycurve_work(::Type{T}, ETL::Int) where {T} = EPGWork_ReIm_DualVector_Split_Dynamic(T, ETL) # default for dynamic `ETL`
@inline EPGdecaycurve_work(::Type{T}, ::Val{ETL}) where {T, ETL} = EPGWork_ReIm_DualMVector_Split(T, Val(ETL)) # default for static `ETL`

# Workspaces for computing decay bases over a T2 grid
@inline EPGdecaybasis_work(θ::EPGParameterization{T}) where {T} = EPGWork_ReIm_Tangent_Batched_Split_Dynamic(T, echotrainlength(θ))
@inline EPGdecaybasis_work(θ::EPGConstantFlipAngleOptions{T}) where {T} = EPGWork_ReIm_Batched_Split_Dynamic(T, echotrainlength(θ))

# Default fastest cache types and builders for each EPGParameterization type

@inline default_cache_type(θ::EPGOptions, ::Type{T} = eltype(θ), etl::ETL = θ.ETL) where {T, ETL} = EPGWork_ReIm_DualVector_Split_Dynamic{T, ETL, Vector{SVector{3, T}}, Vector{T}}
@inline default_cache_type(θ::EPGConstantFlipAngleOptions, ::Type{T} = eltype(θ), etl::ETL = θ.ETL) where {T, ETL} = EPGWork_ReIm_DualFlat_Split_Dynamic{T, ETL, Vector{T}, Vector{T}}
@inline default_cache_type(θ::EPGIncreasingFlipAnglesOptions, ::Type{T} = eltype(θ), etl::ETL = θ.ETL) where {T, ETL} = EPGWork_Basic_Cplx{T, ETL, Vector{SVector{3, Complex{T}}}, Vector{T}}

@inline default_cache(θ::EPGOptions, ::Type{T} = eltype(θ), etl::ETL = θ.ETL) where {T, ETL} = EPGWork_ReIm_DualVector_Split_Dynamic(T, etl)
@inline default_cache(θ::EPGConstantFlipAngleOptions, ::Type{T} = eltype(θ), etl::ETL = θ.ETL) where {T, ETL} = EPGWork_ReIm_DualFlat_Split_Dynamic(T, etl)
@inline default_cache(θ::EPGIncreasingFlipAnglesOptions, ::Type{T} = eltype(θ), etl::ETL = θ.ETL) where {T, ETL} = EPGWork_Basic_Cplx(T, etl)

"""
    EPGdecaycurve(ETL::Int, α::Real, TE::Real, T2::Real, T1::Real, β::Real)

Computes the normalized echo decay curve for a multi-spin echo sequence
using the extended phase graph algorithm using the given input parameters.

The sequence of flip angles used is slight generalization of the standard
90 degree excitation pulse followed by 180 degree pulse train.
Here, the sequence used is `A*90, A*180, A*β, A*β, ...` where `A = α/180`
accounts for B1 inhomogeneities. Equivalently, the pulse sequence can
be written as `α/2, α, α * (β/180), α * (β/180), ...`.
Note that if `α = β = 180`, we recover the standard `90, 180, 180, ...`
pulse sequence.

# Arguments

  - `ETL::Int`:   echo train length, i.e. number of echos
  - `α::Real`:    angle of refocusing pulses (Units: degrees)
  - `TE::Real`:   inter-echo time (Units: time, must match `T1` and `T2`)
  - `T2::Real`:   transverse relaxation time (Units: time, must match `TE`)
  - `T1::Real`:   longitudinal relaxation time (Units: time, must match `TE`)
  - `β::Real`:    refocusing pulse control angle (Units: degrees)

# Outputs

  - `decay_curve::AbstractVector`: normalized echo decay curve with length `ETL`

!!! note "Units of time"

    The decay curve depends on `TE`, `T2`, and `T1` only through the ratios `TE/T2` and `TE/T1`, so no particular unit of time is assumed; the three need only use the same units.

The four-argument method omits `β`, fixing the refocusing control angle at 180 degrees, which is the standard CPMG sequence.
"""
@inline EPGdecaycurve(ETL, α::Real, TE::Real, T2::Real, T1::Real, β::Real) = EPGdecaycurve(EPGOptions((; ETL, α = deg2rad(α), TE, T2, T1, β = deg2rad(β)))) # the arguments are degrees; the parameterization stores radians
@inline EPGdecaycurve(ETL, α::Real, TE::Real, T2::Real, T1::Real) = EPGdecaycurve(EPGConstantFlipAngleOptions((; ETL, α = deg2rad(α), TE, T2, T1))) # the arguments are degrees; the parameterization stores radians
@inline EPGdecaycurve(θ::EPGParameterization{T}) where {T} = EPGdecaycurve!(EPGdecaycurve_work(θ), θ)

"""
    EPGdecaycurve!(work::AbstractEPGWorkspace, θ::EPGParameterization)
    EPGdecaycurve!(decay_curve::AbstractVector, work::AbstractEPGWorkspace, θ::EPGParameterization)

In-place form of [`EPGdecaycurve`](@ref), reusing the workspace `work` allocated by [`EPGdecaycurve_work`](@ref).
The two-argument method writes into and returns the decay curve held by `work`; the three-argument method writes into `decay_curve`.

See also:

  - [`EPGdecaycurve`](@ref)
  - [`EPGdecaycurve_work`](@ref)
"""
@inline EPGdecaycurve!(work::AbstractEPGWorkspace{T}, θ::EPGParameterization{T}) where {T} = EPGdecaycurve!(decaycurve(work), work, θ)
@inline EPGdecaycurve!(A::AbstractVector{T}, work::AbstractEPGWorkspace{T}, θ::EPGParameterization{T}) where {T} = epg_decay_curve!(A, work, θ)

####
#### Jacobian utilities (currently hardcoded for `EPGWork_ReIm_DualVector_Split_Dynamic`)
####

struct EPGWorkCacheDict{T, ETL, Tθ <: EPGParameterization{T, ETL}} <: AbstractDict{DataType, Any}
    θ::Tθ
    dict::Dict{DataType, AbstractEPGWorkspace{<:Any, ETL}}
end
EPGWorkCacheDict(θ::EPGParameterization{<:Any, ETL}) where {ETL} = EPGWorkCacheDict(θ, Dict{DataType, AbstractEPGWorkspace{<:Any, ETL}}())

@inline Base.keys(caches::EPGWorkCacheDict) = Base.keys(caches.dict)
@inline Base.values(caches::EPGWorkCacheDict) = Base.values(caches.dict)
@inline Base.length(caches::EPGWorkCacheDict) = Base.length(caches.dict)
@inline Base.iterate(caches::EPGWorkCacheDict, state...) = Base.iterate(caches.dict, state...)

@inline function Base.getindex(caches::EPGWorkCacheDict{ETL}, ::Type{T}) where {ETL, T}
    R = default_cache_type(caches.θ, T)
    get!(caches.dict, T) do
        return default_cache(caches.θ, T)
    end::R
end

struct EPGFunctor{T, ETL, Fs, Tθ <: EPGParameterization{T, ETL}, TC <: EPGWorkCacheDict{T, ETL, Tθ}}
    θ::Tθ
    fields::SymbolVector{Fs}
    caches::TC
end
EPGFunctor(θ::EPGParameterization, fields::SymbolVector) = EPGFunctor(θ, fields, EPGWorkCacheDict(θ))
EPGFunctor(θ::EPGParameterization, fields::Val) = EPGFunctor(θ, SymbolVector(fields))
EPGFunctor(f!::EPGFunctor, θ::EPGParameterization) = EPGFunctor(θ, f!.fields, f!.caches)

@inline parameters(f!::EPGFunctor) = f!.θ
@inline optfields(f!::EPGFunctor) = f!.fields

function (f!::EPGFunctor)(y::AbstractVector{D}, epg_work::AbstractEPGWorkspace{D}, x::AbstractVector{D}) where {D}
    θ = restructure(parameters(f!), x, optfields(f!))
    return EPGdecaycurve!(y, epg_work, θ)
end
(f!::EPGFunctor)(y::AbstractVector{D}, x::AbstractVector{D}) where {D} = f!(y, f!.caches[D], x)

struct EPGJacobianFunctor{T, ETL, Fs, F <: EPGFunctor{T, ETL, Fs}, R <: DiffResults.DiffResult, C <: ForwardDiff.JacobianConfig}
    f!::F
    res::R
    cfg::C
end
function EPGJacobianFunctor(θ::EPGParameterization{T}, fields::SymbolVector) where {T}
    ETL, N = echotrainlength(θ), length(fields)
    f! = EPGFunctor(θ, fields)
    res = DiffResults.JacobianResult(zeros(T, ETL), zeros(T, N))
    cfg = ForwardDiff.JacobianConfig(f!, zeros(T, ETL), zeros(T, N), ForwardDiff.Chunk{N}())
    return EPGJacobianFunctor(f!, res, cfg)
end
EPGJacobianFunctor(θ::EPGParameterization, fields::Val) = EPGJacobianFunctor(θ, SymbolVector(fields))

@inline parameters(j!::EPGJacobianFunctor) = parameters(j!.f!)
@inline optfields(j!::EPGJacobianFunctor) = optfields(j!.f!)

function (j!::EPGJacobianFunctor{T})(J::Union{AbstractMatrix, DiffResults.DiffResult}, y::AbstractVector{T}, θ::EPGParameterization{T}) where {T}
    (; f!, cfg) = j!
    f! = EPGFunctor(f!, θ)
    x = destructure(parameters(f!), optfields(f!))
    ForwardDiff.jacobian!(J, f!, y, x, cfg)
    return J isa AbstractMatrix ? J : DiffResults.jacobian(J)
end
(j!::EPGJacobianFunctor{T})(y::AbstractVector{T}, θ::EPGParameterization{T}) where {T} = j!(j!.res, y, θ)

####
#### EPGWork_Basic_Cplx
####

struct EPGWork_Basic_Cplx{T, ETL, MPSVType <: AbstractVector{SVector{3, Complex{T}}}, AType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV::MPSVType
    A::AType
end
function EPGWork_Basic_Cplx(::Type{T}, ETL::Int) where {T}
    MSPV = zeros(SVector{3, Complex{T}}, ETL)
    A = zeros(T, ETL)
    return EPGWork_Basic_Cplx(ETL, MSPV, A)
end

# Compute a basis function under the extended phase graph algorithm. The magnetization phase state vector (MPSV) is
# successively modified by applying relaxation for TE/2, then a refocusing pulse as described by Hennig (1988),
# then transitioning phase states as given by Hennig (1988) but corrected by Jones (1997), and a finally relaxing for TE/2.
# See the appendix in Prasloski (2012) for details:
#    https://doi.org/10.1002/mrm.23157

function epg_decay_curve!(A::AbstractVector{T}, work::EPGWork_Basic_Cplx{T}, θ::EPGOptions{T}) where {T}
    ETL = length(A)

    # Unpack workspace
    (; MPSV) = work
    αₑₓ = B1correctedflipangle(θ, 0)
    α₁ = B1correctedflipangle(θ, 1)
    α = B1correctedflipangle(θ, 2)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = SA{Complex{T}} # alias

    # Precompute compute element flip matrices and other intermediate variables
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    E = SA{T}[E₂, E₂, E₁]
    R₁ = element_flipmat(α₁)
    Rₙ = element_flipmat(α)

    # Initialize magnetization phase state vector (MPSV)
    @inbounds for j in 1:ETL
        MPSV[j] = V[0, 0, 0]
    end
    @inbounds MPSV[1] = V[sin(αₑₓ), 0, 0] # initial magnetization in F1 state

    @inbounds for n in 1:ETL
        # Relaxation for TE/2, followed by flip matrix
        R = n == 1 ? R₁ : Rₙ
        for j in 1:ETL
            MPSV[j] = R * (E .* MPSV[j])
        end

        # Transition between phase states
        Ωⱼ, Ωⱼ₊₁ = MPSV[1], MPSV[2]
        MPSV[1] = V[Ωⱼ[2], Ωⱼ₊₁[2], Ωⱼ[3]] # (F⁺₁, F⁺₁*, Z₁)⁺ = (F⁺₁*, F⁺₂*, Z₁)
        for j in 2:(ETL-1)
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV[j+1]
            MPSV[j] = V[Ωⱼ₋₁[1], Ωⱼ₊₁[2], Ωⱼ[3]] # (Fᵢ, Fᵢ*, Zᵢ)⁺ = (Fᵢ₋₁, Fᵢ₊₁*, Zᵢ)
        end
        Ωⱼ₋₁, Ωⱼ = Ωⱼ, Ωⱼ₊₁
        MPSV[ETL] = V[Ωⱼ₋₁[1], 0, Ωⱼ[3]] # (Fₙ, Fₙ*, Zₙ)⁺ = (Fₙ₋₁, 0, Zₙ)

        # Relaxation for TE/2
        for j in 1:ETL
            MPSV[j] = E .* MPSV[j] # Relaxation for TE/2
        end
        A[n] = abs(MPSV[1][1]) # first echo amplitude
    end

    return A
end

function epg_decay_curve!(A::AbstractVector{T}, work::EPGWork_Basic_Cplx{T}, θ::EPGParameterization{T}) where {T}
    ETL = length(A)

    # Unpack workspace
    (; MPSV) = work
    αₑₓ = B1correctedflipangle(θ, 0)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = SA{Complex{T}} # alias

    # Precompute compute element flip matrices and other intermediate variables
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    E = SA{T}[E₂, E₂, E₁]

    # Initialize magnetization phase state vector (MPSV)
    @inbounds for j in 1:ETL
        MPSV[j] = V[0, 0, 0]
    end
    @inbounds MPSV[1] = V[sin(αₑₓ), 0, 0] # initial magnetization in F1 state

    @inbounds for n in 1:ETL
        # Relaxation for TE/2, followed by flip matrix
        R = element_flipmat(B1correctedflipangle(θ, n))
        for j in 1:ETL
            MPSV[j] = R * (E .* MPSV[j])
        end

        # Transition between phase states
        Ωⱼ, Ωⱼ₊₁ = MPSV[1], MPSV[2]
        MPSV[1] = V[Ωⱼ[2], Ωⱼ₊₁[2], Ωⱼ[3]] # (F⁺₁, F⁺₁*, Z₁)⁺ = (F⁺₁*, F⁺₂*, Z₁)
        for j in 2:(ETL-1)
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV[j+1]
            MPSV[j] = V[Ωⱼ₋₁[1], Ωⱼ₊₁[2], Ωⱼ[3]] # (Fᵢ, Fᵢ*, Zᵢ)⁺ = (Fᵢ₋₁, Fᵢ₊₁*, Zᵢ)
        end
        Ωⱼ₋₁, Ωⱼ = Ωⱼ, Ωⱼ₊₁
        MPSV[ETL] = V[Ωⱼ₋₁[1], 0, Ωⱼ[3]] # (Fₙ, Fₙ*, Zₙ)⁺ = (Fₙ₋₁, 0, Zₙ)

        # Relaxation for TE/2
        for j in 1:ETL
            MPSV[j] = E .* MPSV[j] # Relaxation for TE/2
        end
        A[n] = abs(MPSV[1][1]) # first echo amplitude
    end

    return A
end

####
#### EPGWork_ReIm
####

struct EPGWork_ReIm{T, ETL, MPSVType <: AbstractVector{SVector{3, T}}, AType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV::MPSVType
    A::AType
end
function EPGWork_ReIm(::Type{T}, ETL::Int) where {T}
    MSPV = zeros(SVector{3, T}, ETL)
    A = zeros(T, ETL)
    return EPGWork_ReIm(ETL, MSPV, A)
end

function epg_decay_curve!(A::AbstractVector{T}, work::EPGWork_ReIm{T}, θ::EPGOptions{T}) where {T}
    ETL = length(A)

    # Unpack workspace
    (; MPSV) = work
    α₁ = B1correctedflipangle(θ, 1)
    α = B1correctedflipangle(θ, 2)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = SA{T} # alias

    # Precompute intermediate variables
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sin½α₁, cos½α₁ = sincos(α₁ / 2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2 * sin½α₁ * cos½α₁
    sinα, cosα = sincos(α)
    cos²½α = (1 + cosα) / 2
    sin²½α = 1 - cos²½α
    a⁺₁, a⁻₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    a⁺, a⁻, c, d = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
    𝒜⁺, 𝒜⁻, 𝒜ᶻ = V[a⁺, a⁻, c], V[a⁻, a⁺, -c], V[-c/2, c/2, d]

    # Initialize magnetization phase state vector (MPSV), pulling n=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Ω₁′ = V[a⁻₁*m₀, 0, -c₁*m₀/2]
        A[1] = abs(Ω₁′[1])
        MPSV[1] = Ω₁′
        MPSV[2] = V[a⁺₁*m₀, 0, 0]
        MPSV[3] = V[0, 0, 0]
    end

    @inbounds for n in 2:(ETL-1)
        # j = 1, initialize and update `A`
        Ωⱼ, Ωⱼ₊₁ = MPSV[1], MPSV[2]
        Ω₁′ = V[𝒜⁻⋅Ωⱼ, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ]
        A[n] = abs(Ω₁′[1])
        MPSV[1] = Ω₁′

        # inner loop
        jup = min(n, ETL - n)
        for j in 2:jup
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV[j+1]
            MPSV[j] = V[𝒜⁺⋅Ωⱼ₋₁, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ]
        end

        # cleanup for next iteration
        if n == jup
            Ωⱼ₋₁ = Ωⱼ
            MPSV[n+1] = V[𝒜⁺⋅Ωⱼ₋₁, 0, 0]
            MPSV[n+2] = V[0, 0, 0]
        end
    end

    @inbounds A[ETL] = abs(𝒜⁻ ⋅ MPSV[1])

    return A
end

####
#### EPGWork_ReIm_Generated
####

#=
struct EPGWork_ReIm_Generated{T, ETL, MPSVType <: AbstractVector{SVector{3, T}}, AType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV::MPSVType
    A::AType
end
function EPGWork_ReIm_Generated(::Type{T}, ::Val{ETL}) where {T, ETL}
    MSPV = SizedVector{ETL, SVector{3, T}}(undef)
    A    = SizedVector{ETL, T}(undef)
    return EPGWork_ReIm_Generated(Val(ETL), MSPV, A)
end
EPGWork_ReIm_Generated(::Type{T}, ETL::Int) where {T} = EPGWork_ReIm_Generated(T, Val(ETL))

function epg_decay_curve_impl!(A::Type{AType}, work::Type{W}, θ::Type{O}) where {T, ETL, AType <: AbstractVector{T}, W <: EPGWork_ReIm_Generated{T, Val{ETL}}, O <: EPGOptions{T, Val{ETL}}}
    MPSV(n::Int) = Symbol(:MPSV, n)
    quote
        # Unpack workspace
        α₁ = B1correctedflipangle(θ, 1)
        α = B1correctedflipangle(θ, 2)
        TE = echotime(θ)
        T2 = T2time(θ)
        T1 = T1time(θ)

        # Precompute intermediate variables
        V                = SA{$T} # alias
        E₁, E₂           = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
        sin½α₁, cos½α₁   = sincos(α₁ / 2)
        sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
        sinα₁            = 2 * sin½α₁ * cos½α₁
        sinα, cosα     = sincos(α)
        cos²½α          = (1 + cosα) / 2
        sin²½α          = 1 - cos²½α
        a⁺₁, a⁻₁, c₁       = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
        a⁺, a⁻, c, d   = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
        𝒜⁺, 𝒜⁻, 𝒜ᶻ         = V[a⁺, a⁻, c], V[a⁻, a⁺, -c], V[-c/2, c/2, d]

        # Initialize MPSV vector elements
        $([
            :($(MPSV(n)) = zero(SVector{3, $T}))
            for n in 1:ETL
        ]...)

        # Initialize magnetization phase state vector (MPSV), pulling n=1 iteration out of loop
        @inbounds begin
            m₀         = sin½α₁ # since αₑₓ = ½α₁
            Ω₁′        = V[a⁻₁*m₀, 0, -c₁*m₀/2]
            A[1]       = abs(Ω₁′[1])
            $(MPSV(1)) = Ω₁′
            $(MPSV(2)) = V[a⁺₁*m₀, 0, 0]
        end

        # Main loop
        $([
            quote
                # Initialize and update `A` (j = 1)
                @inbounds begin
                    Ωⱼ, Ωⱼ₊₁   = $(MPSV(1)), $(MPSV(2))
                    Ω₁′        = V[𝒜⁻⋅Ωⱼ, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ]
                    A[$n]      = abs(Ω₁′[1])
                    $(MPSV(1)) = Ω₁′
                end

                # Inner loop
                $([
                    quote
                        (Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁) = (Ωⱼ, Ωⱼ₊₁, $(MPSV(j + 1)))
                        $(MPSV(j))       = V[𝒜⁺⋅Ωⱼ₋₁, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ]
                    end
                    for j in 2:min(n, ETL - n)+1
                ]...)
            end
            for n in 2:ETL-1
        ]...)

        # Last echo
        @inbounds A[$ETL] = abs(𝒜⁻ ⋅ $(MPSV(1)))

        return A
    end
end

@generated function epg_decay_curve!(A::AbstractVector{T}, work::EPGWork_ReIm_Generated{T}, θ::EPGOptions{T}) where {T}
    return epg_decay_curve_impl!(A, work, θ)
end
=#

####
#### EPGWork_ReIm_DualVector
####

struct EPGWork_ReIm_DualVector{T, ETL, MPSVType <: AbstractVector{SVector{3, T}}, AType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    A::AType
end
function EPGWork_ReIm_DualVector(::Type{T}, ETL::Int) where {T}
    MPSV₁ = zeros(SVector{3, T}, ETL)
    MPSV₂ = zeros(SVector{3, T}, ETL)
    A = zeros(T, ETL)
    return EPGWork_ReIm_DualVector(ETL, MPSV₁, MPSV₂, A)
end

function epg_decay_curve!(A::AbstractVector{T}, work::EPGWork_ReIm_DualVector{T}, θ::EPGOptions{T}) where {T}
    ETL = length(A)

    # Unpack workspace
    (; MPSV₁, MPSV₂) = work
    α₁ = B1correctedflipangle(θ, 1)
    α = B1correctedflipangle(θ, 2)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = SA{T} # alias

    # Precompute intermediate variables
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sin½α₁, cos½α₁ = sincos(α₁ / 2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2 * sin½α₁ * cos½α₁
    sinα, cosα = sincos(α)
    cos²½α = (1 + cosα) / 2
    sin²½α = 1 - cos²½α
    a⁺₁, a⁻₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    a⁺, a⁻, c, d = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
    𝒜⁺, 𝒜⁻, 𝒜ᶻ = V[a⁺, a⁻, c], V[a⁻, a⁺, -c], V[-c/2, c/2, d]

    # Initialize magnetization phase state vector (MPSV), pulling n=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Ω₁′ = V[a⁻₁*m₀, 0, -c₁*m₀/2]
        A[1] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        MPSV₁[2] = V[a⁺₁*m₀, 0, 0]
        MPSV₁[3] = V[0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in 2:(ETL-1)
        # j = 1, initialize and update `A`
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2]
        Ω₁′ = V[𝒜⁻⋅Ωⱼ, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ]
        A[n] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′

        # inner loop
        jup = min(n, ETL - n)
        @simd for j in 2:jup
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[𝒜⁺⋅Ωⱼ₋₁, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ]
        end

        # cleanup for next iteration
        if n == jup
            Ωⱼ₋₁ = Ωⱼ
            MPSV₁[n+1] = V[𝒜⁺⋅Ωⱼ₋₁, 0, 0]
            MPSV₁[n+2] = V[0, 0, 0]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds A[ETL] = abs(𝒜⁻ ⋅ MPSV₂[1])

    return A
end

####
#### EPGWork_ReIm_DualVector_Split
####

struct EPGWork_ReIm_DualVector_Split{T, ETL, MPSVType <: AbstractVector{SVector{3, T}}, AType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    A::AType
end
function EPGWork_ReIm_DualVector_Split(::Type{T}, ETL::Int) where {T}
    MPSV₁ = zeros(SVector{3, T}, ETL)
    MPSV₂ = zeros(SVector{3, T}, ETL)
    A = zeros(T, ETL)
    return EPGWork_ReIm_DualVector_Split(ETL, MPSV₁, MPSV₂, A)
end

function epg_decay_curve!(A::AbstractVector{T}, work::EPGWork_ReIm_DualVector_Split{T}, θ::EPGOptions{T}) where {T}
    ETL = length(A)

    # Unpack workspace
    (; MPSV₁, MPSV₂) = work
    α₁ = B1correctedflipangle(θ, 1)
    α = B1correctedflipangle(θ, 2)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = SA{T} # alias

    # Precompute intermediate variables
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sin½α₁, cos½α₁ = sincos(α₁ / 2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2 * sin½α₁ * cos½α₁
    sinα, cosα = sincos(α)
    cos²½α = (1 + cosα) / 2
    sin²½α = 1 - cos²½α
    a⁺₁, a⁻₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    a⁺, a⁻, c, d = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
    𝒜⁺, 𝒜⁻, 𝒜ᶻ = V[a⁺, a⁻, c], V[a⁻, a⁺, -c], V[-c/2, c/2, d]

    # Initialize magnetization phase state vector (MPSV), pulling n=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Ω₁′ = V[a⁻₁*m₀, 0, -c₁*m₀/2]
        A[1] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        MPSV₁[2] = V[a⁺₁*m₀, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in 2:(ETL÷2)
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `A`
        Ω₁′ = V[𝒜⁻⋅Ωⱼ, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ]
        A[n] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:(n-1)
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[𝒜⁺⋅Ωⱼ₋₁, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ]
        end
        MPSV₁[n] = V[𝒜⁺⋅Ωⱼ, 0, 𝒜ᶻ⋅Ωⱼ₊₁]
        MPSV₁[n+1] = V[𝒜⁺⋅Ωⱼ₊₁, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in (ETL÷2+1):(ETL-1)
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `A`
        Ω₁′ = V[𝒜⁻⋅Ωⱼ, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ]
        A[n] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:(ETL-n)
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[𝒜⁺⋅Ωⱼ₋₁, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds A[ETL] = abs(𝒜⁻ ⋅ MPSV₂[1])

    return A
end

####
#### EPGWork_ReIm_DualVector_Split_Dynamic
####

struct EPGWork_ReIm_DualVector_Split_Dynamic{T, ETL, MPSVType <: AbstractVector{SVector{3, T}}, AType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    A::AType
end
function EPGWork_ReIm_DualVector_Split_Dynamic(::Type{T}, ETL::Int) where {T}
    MPSV₁ = zeros(SVector{3, T}, ETL)
    MPSV₂ = zeros(SVector{3, T}, ETL)
    A = zeros(T, ETL)
    return EPGWork_ReIm_DualVector_Split_Dynamic(ETL, MPSV₁, MPSV₂, A)
end

function epg_decay_curve!(A::AbstractVector, work::EPGWork_ReIm_DualVector_Split_Dynamic{T}, θ::EPGOptions{T}) where {T}
    V = SA{T} # alias
    ETL = length(A)

    # Unpack workspace
    (; MPSV₁, MPSV₂) = work

    α₁ = B1correctedflipangle(θ, 1)
    α = B1correctedflipangle(θ, 2)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)

    # Precompute intermediate variables
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sin½α₁, cos½α₁ = sincos(α₁ / 2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2 * sin½α₁ * cos½α₁
    sinα, cosα = sincos(α)
    cos²½α = (1 + cosα) / 2
    sin²½α = 1 - cos²½α
    a⁺₁, a⁻₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    a⁺, a⁻, c, d = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
    𝒜⁺, 𝒜⁻, 𝒜ᶻ = V[a⁺, a⁻, c], V[a⁻, a⁺, -c], V[-c/2, c/2, d]

    @inbounds begin
        # n = 1 iteration
        # Initialize magnetization phase state vector (MPSV)
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Ω₁ = V[a⁻₁*m₀, 0, -c₁*m₀/2]
        MPSV₁[1] = Ω₁
        MPSV₁[2] = V[a⁺₁*m₀, 0, 0]

        A[1] = abs(Ω₁[1])
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁

        # n = 2 iteration
        Ω₁, Ω₂ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `A`
        Ω⁺₁, Ω⁻₁, Ωᶻ₁ = 𝒜⁺ ⋅ Ω₁, 𝒜⁻ ⋅ Ω₁, 𝒜ᶻ ⋅ Ω₁
        Ω⁺₂, Ω⁻₂, Ωᶻ₂ = 𝒜⁺ ⋅ Ω₂, 𝒜⁻ ⋅ Ω₂, 𝒜ᶻ ⋅ Ω₂

        MPSV₁[1] = V[Ω⁻₁, Ω⁻₂, Ωᶻ₁]
        MPSV₁[2] = V[Ω⁺₁, 0, Ωᶻ₂]
        MPSV₁[3] = V[Ω⁺₂, 0, 0]

        A[2] = abs(Ω⁻₁)
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in 3:(ETL÷2)
        Ω₁, Ω₂, Ω₃ = MPSV₂[1], MPSV₂[2], MPSV₂[3] # j = 1, initialize and update `A`
        Ω⁺₁, Ω⁻₁, Ωᶻ₁ = 𝒜⁺ ⋅ Ω₁, 𝒜⁻ ⋅ Ω₁, 𝒜ᶻ ⋅ Ω₁
        Ω⁺₂, Ω⁻₂, Ωᶻ₂ = 𝒜⁺ ⋅ Ω₂, 𝒜⁻ ⋅ Ω₂, 𝒜ᶻ ⋅ Ω₂
        Ω⁺₃, Ω⁻₃, Ωᶻ₃ = 𝒜⁺ ⋅ Ω₃, 𝒜⁻ ⋅ Ω₃, 𝒜ᶻ ⋅ Ω₃

        MPSV₁[1] = V[Ω⁻₁, Ω⁻₂, Ωᶻ₁]
        MPSV₁[2] = V[Ω⁺₁, Ω⁻₃, Ωᶻ₂]

        for j in 3:(n-1)
            Ω⁺₁, Ω⁺₂, Ωᶻ₂ = Ω⁺₂, Ω⁺₃, Ωᶻ₃
            Ω₃ = MPSV₂[j+1]
            Ω⁺₃, Ω⁻₃, Ωᶻ₃ = 𝒜⁺ ⋅ Ω₃, 𝒜⁻ ⋅ Ω₃, 𝒜ᶻ ⋅ Ω₃
            MPSV₁[j] = V[Ω⁺₁, Ω⁻₃, Ωᶻ₂]
        end

        MPSV₁[n] = V[Ω⁺₂, 0, Ωᶻ₃]
        MPSV₁[n+1] = V[Ω⁺₃, 0, 0]

        A[n] = abs(Ω⁻₁)
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in (ETL÷2+1):(ETL-1)
        Ω₁, Ω₂, Ω₃ = MPSV₂[1], MPSV₂[2], MPSV₂[3] # j = 1, initialize and update `A`
        Ω⁺₁, Ω⁻₁, Ωᶻ₁ = 𝒜⁺ ⋅ Ω₁, 𝒜⁻ ⋅ Ω₁, 𝒜ᶻ ⋅ Ω₁
        Ω⁺₂, Ω⁻₂, Ωᶻ₂ = 𝒜⁺ ⋅ Ω₂, 𝒜⁻ ⋅ Ω₂, 𝒜ᶻ ⋅ Ω₂
        Ω⁺₃, Ω⁻₃, Ωᶻ₃ = 𝒜⁺ ⋅ Ω₃, 𝒜⁻ ⋅ Ω₃, 𝒜ᶻ ⋅ Ω₃

        MPSV₁[1] = V[Ω⁻₁, Ω⁻₂, Ωᶻ₁]
        MPSV₁[2] = V[Ω⁺₁, Ω⁻₃, Ωᶻ₂]

        for j in 3:(ETL-n)
            Ω⁺₁, Ω⁺₂, Ωᶻ₂ = Ω⁺₂, Ω⁺₃, Ωᶻ₃
            Ω₃ = MPSV₂[j+1]
            Ω⁺₃, Ω⁻₃, Ωᶻ₃ = 𝒜⁺ ⋅ Ω₃, 𝒜⁻ ⋅ Ω₃, 𝒜ᶻ ⋅ Ω₃
            MPSV₁[j] = V[Ω⁺₁, Ω⁻₃, Ωᶻ₂]
        end

        A[n] = abs(Ω⁻₁)
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds A[ETL] = abs(𝒜⁻ ⋅ MPSV₂[1])

    return A
end

function epg_decay_curve!(A::AbstractVector, work::EPGWork_ReIm_DualVector_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    epg_impulse_response!(A, work, θ)

    # Scale impulse response by initial magnetization and take absolute value
    m₀ = sin(B1correctedflipangle(θ, 0))
    @simd ivdep for n in eachindex(A)
        A[n] = abs(m₀ * A[n])
    end

    return A
end

function epg_impulse_response!(A::AbstractVector{T}, work::EPGWork_ReIm_DualVector_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    ETL = length(A)
    (; MPSV₁, MPSV₂) = work

    α = θ.α
    TE, T2, T1 = echotime(θ), T2time(θ), T1time(θ)
    V = SA{T} # alias

    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sinα, cosα = sincos(α)
    E₂²half, E₁E₂, E₁² = (E₂ * E₂) / 2, E₁ * E₂, E₁ * E₁
    a, b, c, d = E₂²half, E₂²half * cosα, E₁E₂ * sinα, E₁² * cosα
    c′ = -c / 2

    @inbounds begin
        Ω₁ = V[a-b, zero(T), c′]
        MPSV₁[1] = Ω₁
        MPSV₁[2] = V[a+b, zero(T), zero(T)]

        A[1] = Ω₁[1]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁

        for n in 2:(ETL÷2)
            F⁺, F⁻, Z = MPSV₂[1]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            Ω⁻₂, Ω⁺₂, Ωᶻ₂ = muladd(-c, Z, x′ - y′), muladd(c, Z, x′ + y′), muladd(c′, y, d * Z)

            F⁺, F⁻, Z = MPSV₂[2]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            Ω⁻₃, Ω⁺₃, Ωᶻ₃ = muladd(-c, Z, x′ - y′), muladd(c, Z, x′ + y′), muladd(c′, y, d * Z)

            MPSV₁[1] = V[Ω⁻₂, Ω⁻₃, Ωᶻ₂]
            A[n] = Ω⁻₂
            Ω⁺₁, Ω⁺₂, Ωᶻ₂ = Ω⁺₂, Ω⁺₃, Ωᶻ₃

            @simd ivdep for j in 2:(n-1)
                F⁺, F⁻, Z = MPSV₂[j+1]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a * x, b * y
                Ω⁺₃, Ω⁻₃, Ωᶻ₃ = muladd(c, Z, x′ + y′), muladd(-c, Z, x′ - y′), muladd(c′, y, d * Z)
                MPSV₁[j] = V[Ω⁺₁, Ω⁻₃, Ωᶻ₂]
                Ω⁺₁, Ω⁺₂, Ωᶻ₂ = Ω⁺₂, Ω⁺₃, Ωᶻ₃
            end

            MPSV₁[n], MPSV₁[n+1] = V[Ω⁺₁, zero(T), Ωᶻ₂], V[Ω⁺₂, zero(T), zero(T)]
            MPSV₁, MPSV₂ = MPSV₂, MPSV₁
        end

        for n in (ETL÷2+1):(ETL-1)
            F⁺, F⁻, Z = MPSV₂[1]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            Ω⁻₂, Ω⁺₂, Ωᶻ₂ = muladd(-c, Z, x′ - y′), muladd(c, Z, x′ + y′), muladd(c′, y, d * Z)

            F⁺, F⁻, Z = MPSV₂[2]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            Ω⁻₃, Ω⁺₃, Ωᶻ₃ = muladd(-c, Z, x′ - y′), muladd(c, Z, x′ + y′), muladd(c′, y, d * Z)

            MPSV₁[1] = V[Ω⁻₂, Ω⁻₃, Ωᶻ₂]
            A[n] = Ω⁻₂
            Ω⁺₁, Ω⁺₂, Ωᶻ₂ = Ω⁺₂, Ω⁺₃, Ωᶻ₃

            @simd ivdep for j in 2:(ETL-n)
                F⁺, F⁻, Z = MPSV₂[j+1]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a * x, b * y
                Ω⁺₃, Ω⁻₃, Ωᶻ₃ = muladd(c, Z, x′ + y′), muladd(-c, Z, x′ - y′), muladd(c′, y, d * Z)
                MPSV₁[j] = V[Ω⁺₁, Ω⁻₃, Ωᶻ₂]
                Ω⁺₁, Ω⁺₂, Ωᶻ₂ = Ω⁺₂, Ω⁺₃, Ωᶻ₃
            end

            MPSV₁[ETL-n+1] = V[Ω⁺₁, zero(T), Ωᶻ₂]
            MPSV₁, MPSV₂ = MPSV₂, MPSV₁
        end

        F⁺, F⁻, Z = MPSV₂[1]
        x, y = F⁺ + F⁻, F⁺ - F⁻
        A[ETL] = muladd(-c, Z, muladd(a, x, -b * y))
    end

    return A
end

####
#### EPGWork_ReIm_DualFlat_Split_Dynamic
####

struct EPGWork_ReIm_DualFlat_Split_Dynamic{T, ETL, MPSVType <: AbstractVector{T}, AType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    A::AType
end

function EPGWork_ReIm_DualFlat_Split_Dynamic(::Type{T}, ETL::Int) where {T}
    MPSV₁ = zeros(T, 3 * ETL)
    MPSV₂ = zeros(T, 3 * ETL)
    A = zeros(T, ETL)
    return EPGWork_ReIm_DualFlat_Split_Dynamic(ETL, MPSV₁, MPSV₂, A)
end

function epg_decay_curve!(A::AbstractVector, work::EPGWork_ReIm_DualFlat_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    epg_impulse_response!(A, work, θ)

    # Scale impulse response by initial magnetization and take absolute value
    m₀ = sin(B1correctedflipangle(θ, 0))
    @simd ivdep for n in eachindex(A)
        A[n] = abs(m₀ * A[n])
    end

    return A
end

function epg_impulse_response!(A::AbstractVector{T}, work::EPGWork_ReIm_DualFlat_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    Base.require_one_based_indexing(A)
    ETL = length(A)

    (; MPSV₁, MPSV₂) = work
    length(MPSV₁) == length(MPSV₂) == 3 * ETL || throw(DimensionMismatch("workspace holds $(length(MPSV₁) ÷ 3) echoes, but the decay curve has $ETL"))

    α = θ.α
    TE, T2, T1 = echotime(θ), T2time(θ), T1time(θ)

    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sinα, cosα = sincos(α)
    E₂²half, E₁E₂, E₁² = (E₂ * E₂) / 2, E₁ * E₂, E₁ * E₁
    a, b, c, d = E₂²half, E₂²half * cosα, E₁E₂ * sinα, E₁² * cosα
    c′ = -c / 2
    ΔF⁻, ΔZ = ETL, 2 * ETL

    @inbounds begin
        A[1] = a - b
        MPSV₁[1], MPSV₁[1+ΔF⁻], MPSV₁[1+ΔZ] = a - b, zero(T), c′
        MPSV₁[2], MPSV₁[2+ΔF⁻], MPSV₁[2+ΔZ] = a + b, zero(T), zero(T)
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁

        for n in 2:(ETL÷2)
            F⁺, F⁻, Z = MPSV₂[1], MPSV₂[1+ΔF⁻], MPSV₂[1+ΔZ]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            A[n] = MPSV₁[1] = muladd(-c, Z, x′ - y′)
            MPSV₁[2] = muladd(c, Z, x′ + y′)
            MPSV₁[1+ΔZ] = muladd(c′, y, d * Z)

            @simd ivdep for j in 2:(n-1)
                F⁺, F⁻, Z = MPSV₂[j], MPSV₂[j+ΔF⁻], MPSV₂[j+ΔZ]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a * x, b * y
                MPSV₁[j+1] = muladd(c, Z, x′ + y′)
                MPSV₁[j-1+ΔF⁻] = muladd(-c, Z, x′ - y′)
                MPSV₁[j+ΔZ] = muladd(c′, y, d * Z)
            end

            F⁺, F⁻, Z = MPSV₂[n], MPSV₂[n+ΔF⁻], MPSV₂[n+ΔZ]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            MPSV₁[n+1] = muladd(c, Z, x′ + y′)
            MPSV₁[n-1+ΔF⁻] = muladd(-c, Z, x′ - y′)
            MPSV₁[n+ΔZ] = muladd(c′, y, d * Z)

            MPSV₁[n+ΔF⁻] = zero(T)
            MPSV₁[n+1+ΔF⁻] = zero(T)
            MPSV₁[n+1+ΔZ] = zero(T)

            MPSV₁, MPSV₂ = MPSV₂, MPSV₁
        end

        for n in (ETL÷2+1):(ETL-1)
            F⁺, F⁻, Z = MPSV₂[1], MPSV₂[1+ΔF⁻], MPSV₂[1+ΔZ]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            A[n] = MPSV₁[1] = muladd(-c, Z, x′ - y′)
            MPSV₁[2] = muladd(c, Z, x′ + y′)
            MPSV₁[1+ΔZ] = muladd(c′, y, d * Z)

            @simd ivdep for j in 2:(ETL-n+1)
                F⁺, F⁻, Z = MPSV₂[j], MPSV₂[j+ΔF⁻], MPSV₂[j+ΔZ]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a * x, b * y
                MPSV₁[j+1] = muladd(c, Z, x′ + y′)
                MPSV₁[j-1+ΔF⁻] = muladd(-c, Z, x′ - y′)
                MPSV₁[j+ΔZ] = muladd(c′, y, d * Z)
            end

            MPSV₁, MPSV₂ = MPSV₂, MPSV₁
        end

        F⁺, F⁻, Z = MPSV₂[1], MPSV₂[1+ΔF⁻], MPSV₂[1+ΔZ]
        x, y = F⁺ + F⁻, F⁺ - F⁻
        A[ETL] = muladd(-c, Z, muladd(a, x, -b * y))
    end

    return A
end

####
#### EPGWork_ReIm_DualTuple_Split_Dynamic
####

struct EPGWork_ReIm_DualTuple_Split_Dynamic{T, ETL, MPSVType <: AbstractVector{T}, AType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::NTuple{3, MPSVType}
    MPSV₂::NTuple{3, MPSVType}
    A::AType
end

function EPGWork_ReIm_DualTuple_Split_Dynamic(::Type{T}, ETL::Int) where {T}
    MPSV₁ = (zeros(T, ETL), zeros(T, ETL), zeros(T, ETL))
    MPSV₂ = (zeros(T, ETL), zeros(T, ETL), zeros(T, ETL))
    A = zeros(T, ETL)
    return EPGWork_ReIm_DualTuple_Split_Dynamic(ETL, MPSV₁, MPSV₂, A)
end

function epg_decay_curve!(A::AbstractVector, work::EPGWork_ReIm_DualTuple_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    epg_impulse_response!(A, work, θ)

    # Scale impulse response by initial magnetization and take absolute value
    m₀ = sin(B1correctedflipangle(θ, 0))
    @simd ivdep for n in eachindex(A)
        A[n] = abs(m₀ * A[n])
    end

    return A
end

function epg_impulse_response!(A::AbstractVector{T}, work::EPGWork_ReIm_DualTuple_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    Base.require_one_based_indexing(A)
    ETL = length(A)

    (; MPSV₁, MPSV₂) = work
    (F⁺₁, F⁻₁, Z₁), (F⁺₂, F⁻₂, Z₂) = MPSV₁, MPSV₂
    length(F⁺₁) == length(F⁻₁) == length(Z₁) == length(F⁺₂) == length(F⁻₂) == length(Z₂) == ETL || throw(DimensionMismatch("workspace holds $(length(F⁺₁)) echoes, but the decay curve has $ETL"))

    α = θ.α
    TE, T2, T1 = echotime(θ), T2time(θ), T1time(θ)

    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sinα, cosα = sincos(α)
    E₂²half, E₁E₂, E₁² = (E₂ * E₂) / 2, E₁ * E₂, E₁ * E₁
    a, b, c, d = E₂²half, E₂²half * cosα, E₁E₂ * sinα, E₁² * cosα
    c′ = -c / 2

    @inbounds begin
        A[1] = a - b
        F⁺₁[1], F⁻₁[1], Z₁[1] = a - b, zero(T), c′
        F⁺₁[2], F⁻₁[2], Z₁[2] = a + b, zero(T), zero(T)
        (F⁺₁, F⁻₁, Z₁), (F⁺₂, F⁻₂, Z₂) = (F⁺₂, F⁻₂, Z₂), (F⁺₁, F⁻₁, Z₁)

        for n in 2:(ETL÷2)
            F⁺, F⁻, Z = F⁺₂[1], F⁻₂[1], Z₂[1]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            A[n] = F⁺₁[1] = muladd(-c, Z, x′ - y′)
            F⁺₁[2] = muladd(c, Z, x′ + y′)
            Z₁[1] = muladd(c′, y, d * Z)

            @simd ivdep for j in 2:(n-1)
                F⁺, F⁻, Z = F⁺₂[j], F⁻₂[j], Z₂[j]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a * x, b * y
                F⁺₁[j+1] = muladd(c, Z, x′ + y′)
                F⁻₁[j-1] = muladd(-c, Z, x′ - y′)
                Z₁[j] = muladd(c′, y, d * Z)
            end

            F⁺, F⁻, Z = F⁺₂[n], F⁻₂[n], Z₂[n]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            F⁺₁[n+1] = muladd(c, Z, x′ + y′)
            F⁻₁[n-1] = muladd(-c, Z, x′ - y′)
            Z₁[n] = muladd(c′, y, d * Z)

            F⁻₁[n] = zero(T)
            F⁻₁[n+1] = zero(T)
            Z₁[n+1] = zero(T)

            (F⁺₁, F⁻₁, Z₁), (F⁺₂, F⁻₂, Z₂) = (F⁺₂, F⁻₂, Z₂), (F⁺₁, F⁻₁, Z₁)
        end

        for n in (ETL÷2+1):(ETL-1)
            F⁺, F⁻, Z = F⁺₂[1], F⁻₂[1], Z₂[1]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            A[n] = F⁺₁[1] = muladd(-c, Z, x′ - y′)
            F⁺₁[2] = muladd(c, Z, x′ + y′)
            Z₁[1] = muladd(c′, y, d * Z)

            @simd ivdep for j in 2:(ETL-n+1)
                F⁺, F⁻, Z = F⁺₂[j], F⁻₂[j], Z₂[j]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a * x, b * y
                F⁺₁[j+1] = muladd(c, Z, x′ + y′)
                F⁻₁[j-1] = muladd(-c, Z, x′ - y′)
                Z₁[j] = muladd(c′, y, d * Z)
            end

            (F⁺₁, F⁻₁, Z₁), (F⁺₂, F⁻₂, Z₂) = (F⁺₂, F⁻₂, Z₂), (F⁺₁, F⁻₁, Z₁)
        end

        F⁺, F⁻, Z = F⁺₂[1], F⁻₂[1], Z₂[1]
        x, y = F⁺ + F⁻, F⁺ - F⁻
        A[ETL] = muladd(-c, Z, muladd(a, x, -b * y))
    end

    return A
end

####
#### EPGWork_ReIm_Batched_Split_Dynamic
####

const EPG_BATCH_WIDTH = 8

# Batched constant-flip-angle kernel computing up to `EPG_BATCH_WIDTH` decay curves simultaneously, one T2 per lane, with α, TE and T1 shared.
# The EPG recursion is identical to `EPGWork_ReIm_DualTuple_Split_Dynamic`, but every state update is vectorized across the lane dimension.
# That dimension is unit-stride and shift-free, since the k ± 1 shifts move whole lane blocks,
# unlike the single-curve kernels whose k-loops have short trip counts and shifted stores.
# Intended for computing decay bases over a T2 grid; the single-curve `epg_decay_curve!` method duplicates one T2 across all lanes.
struct EPGWork_ReIm_Batched_Split_Dynamic{T, ETL, MType <: AbstractMatrix{T}, VType <: AbstractVector{T}, AType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL   # echo train length
    F⁺₁::MType  # F⁺ states of the current half-step (EPG_BATCH_WIDTH × ETL, lane-major)
    F⁻₁::MType  # F⁻ states of the current half-step
    Z₁::MType  # Z states of the current half-step
    F⁺₂::MType  # F⁺ states of the previous half-step; the two triples are swapped each step
    F⁻₂::MType  # F⁻ states of the previous half-step
    Z₂::MType  # Z states of the previous half-step
    h::MType   # lane-major unit-excitation responses (EPG_BATCH_WIDTH × ETL)
    a::VType   # per-lane coefficients; see `epg_setup_lanes!`
    b::VType   # per-lane a·cos α
    c::VType   # per-lane E₁E₂·sin α
    c′::VType  # per-lane -c/2
    d::VType   # per-lane E₁²·cos α
    A::AType # single-curve decay curve output; see `epg_decay_curve!`
end

function EPGWork_ReIm_Batched_Split_Dynamic(::Type{T}, ETL::Int) where {T}
    W = EPG_BATCH_WIDTH
    F⁺₁, F⁻₁, Z₁ = zeros(T, W, ETL), zeros(T, W, ETL), zeros(T, W, ETL)
    F⁺₂, F⁻₂, Z₂ = zeros(T, W, ETL), zeros(T, W, ETL), zeros(T, W, ETL)
    h = zeros(T, W, ETL)
    a, b, c, c′, d = (zeros(T, W) for _ in 1:5)
    A = zeros(T, ETL)
    return EPGWork_ReIm_Batched_Split_Dynamic(ETL, F⁺₁, F⁻₁, Z₁, F⁺₂, F⁻₂, Z₂, h, a, b, c, c′, d, A)
end

# Load per-lane relaxation and rotation coefficients: lane l gets T2 = T2s[min(l0 + l - 1, lmax)].
# Out-of-range lanes repeat the last T2, and their results are discarded by the caller.
function epg_setup_lanes!(work::EPGWork_ReIm_Batched_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}, T2s::AbstractVector, l0::Int, lmax::Int) where {T}
    (; a, b, c, c′, d) = work
    α = θ.α
    TE, T1 = echotime(θ), T1time(θ)
    E₁ = exp(-(TE / 2) / T1)
    sinα, cosα = sincos(α)
    E₁² = E₁ * E₁
    @inbounds for l in 1:EPG_BATCH_WIDTH
        T2 = T(T2s[min(l0 + l - 1, lmax)])
        E₂ = exp(-(TE / 2) / T2)
        a[l] = (E₂ * E₂) / 2
        b[l] = a[l] * cosα
        c[l] = E₁ * E₂ * sinα
        c′[l] = -c[l] / 2
        d[l] = E₁² * cosα
    end
    return work
end

function epg_impulse_response_batched!(work::EPGWork_ReIm_Batched_Split_Dynamic{T}) where {T}
    (; F⁺₁, F⁻₁, Z₁, F⁺₂, F⁻₂, Z₂, h) = work
    ETL = echotrainlength(work)
    W = EPG_BATCH_WIDTH

    (; a, b, c, c′, d) = work

    @inbounds begin
        @simd ivdep for l in 1:W
            h[l, 1] = a[l] - b[l]
            F⁺₁[l, 1], F⁻₁[l, 1], Z₁[l, 1] = a[l] - b[l], zero(T), c′[l]
            F⁺₁[l, 2], F⁻₁[l, 2], Z₁[l, 2] = a[l] + b[l], zero(T), zero(T)
        end
        (F⁺₁, F⁻₁, Z₁), (F⁺₂, F⁻₂, Z₂) = (F⁺₂, F⁻₂, Z₂), (F⁺₁, F⁻₁, Z₁)

        for n in 2:(ETL÷2)
            @simd ivdep for l in 1:W
                F⁺, F⁻, Z = F⁺₂[l, 1], F⁻₂[l, 1], Z₂[l, 1]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a[l] * x, b[l] * y
                Ω⁻ = muladd(-c[l], Z, x′ - y′)
                h[l, n] = Ω⁻
                F⁺₁[l, 1] = Ω⁻
                F⁺₁[l, 2] = muladd(c[l], Z, x′ + y′)
                Z₁[l, 1] = muladd(c′[l], y, d[l] * Z)
            end

            for j in 2:(n-1)
                @simd ivdep for l in 1:W
                    F⁺, F⁻, Z = F⁺₂[l, j], F⁻₂[l, j], Z₂[l, j]
                    x, y = F⁺ + F⁻, F⁺ - F⁻
                    x′, y′ = a[l] * x, b[l] * y
                    F⁺₁[l, j+1] = muladd(c[l], Z, x′ + y′)
                    F⁻₁[l, j-1] = muladd(-c[l], Z, x′ - y′)
                    Z₁[l, j] = muladd(c′[l], y, d[l] * Z)
                end
            end

            @simd ivdep for l in 1:W
                F⁺, F⁻, Z = F⁺₂[l, n], F⁻₂[l, n], Z₂[l, n]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a[l] * x, b[l] * y
                F⁺₁[l, n+1] = muladd(c[l], Z, x′ + y′)
                F⁻₁[l, n-1] = muladd(-c[l], Z, x′ - y′)
                Z₁[l, n] = muladd(c′[l], y, d[l] * Z)
                F⁻₁[l, n] = zero(T)
                F⁻₁[l, n+1] = zero(T)
                Z₁[l, n+1] = zero(T)
            end

            (F⁺₁, F⁻₁, Z₁), (F⁺₂, F⁻₂, Z₂) = (F⁺₂, F⁻₂, Z₂), (F⁺₁, F⁻₁, Z₁)
        end

        for n in (ETL÷2+1):(ETL-1)
            @simd ivdep for l in 1:W
                F⁺, F⁻, Z = F⁺₂[l, 1], F⁻₂[l, 1], Z₂[l, 1]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a[l] * x, b[l] * y
                Ω⁻ = muladd(-c[l], Z, x′ - y′)
                h[l, n] = Ω⁻
                F⁺₁[l, 1] = Ω⁻
                F⁺₁[l, 2] = muladd(c[l], Z, x′ + y′)
                Z₁[l, 1] = muladd(c′[l], y, d[l] * Z)
            end

            for j in 2:(ETL-n+1)
                @simd ivdep for l in 1:W
                    F⁺, F⁻, Z = F⁺₂[l, j], F⁻₂[l, j], Z₂[l, j]
                    x, y = F⁺ + F⁻, F⁺ - F⁻
                    x′, y′ = a[l] * x, b[l] * y
                    F⁺₁[l, j+1] = muladd(c[l], Z, x′ + y′)
                    F⁻₁[l, j-1] = muladd(-c[l], Z, x′ - y′)
                    Z₁[l, j] = muladd(c′[l], y, d[l] * Z)
                end
            end

            (F⁺₁, F⁻₁, Z₁), (F⁺₂, F⁻₂, Z₂) = (F⁺₂, F⁻₂, Z₂), (F⁺₁, F⁻₁, Z₁)
        end

        @simd ivdep for l in 1:W
            F⁺, F⁻, Z = F⁺₂[l, 1], F⁻₂[l, 1], Z₂[l, 1]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            h[l, ETL] = muladd(-c[l], Z, muladd(a[l], x, -b[l] * y))
        end
    end

    return work
end

function epg_decay_curve!(A::AbstractVector, work::EPGWork_ReIm_Batched_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    ETL = echotrainlength(work)
    epg_setup_lanes!(work, θ, SA[T2time(θ)], 1, 1) # duplicate the single T2 across all lanes
    epg_impulse_response_batched!(work)

    # Scale impulse response by initial magnetization and take absolute value
    m₀ = sin(B1correctedflipangle(θ, 0))
    (; h) = work
    @inbounds @simd ivdep for n in 1:ETL
        A[n] = abs(m₀ * h[1, n])
    end

    return A
end

function epg_decay_basis!(A::AbstractMatrix{T}, decay_curve_work::EPGWork_ReIm_Batched_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}, T2_times::AbstractVector) where {T}
    # Compute the NNLS basis over T2 space in lane-batched chunks
    ETL, nT2 = size(A)
    W = EPG_BATCH_WIDTH
    m₀ = sin(B1correctedflipangle(θ, 0))
    (; h) = decay_curve_work
    @inbounds for j0 in 1:W:nT2
        epg_setup_lanes!(decay_curve_work, θ, T2_times, j0, nT2)
        epg_impulse_response_batched!(decay_curve_work)
        for l in 1:min(W, nT2-j0+1)
            @simd ivdep for n in 1:ETL
                A[n, j0+l-1] = abs(m₀ * h[l, n])
            end
        end
    end
    return A
end

####
#### EPGWork_ReIm_DualMVector_Split
####

struct EPGWork_ReIm_DualMVector_Split{T, ETL, MPSVType <: AbstractVector{SVector{3, T}}, AType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    A::AType
end
function EPGWork_ReIm_DualMVector_Split(::Type{T}, ::Val{ETL}) where {T, ETL}
    MPSV₁ = MVector{ETL, SVector{3, T}}(undef)
    MPSV₂ = MVector{ETL, SVector{3, T}}(undef)
    A = MVector{ETL, T}(undef)
    return EPGWork_ReIm_DualMVector_Split(Val(ETL), MPSV₁, MPSV₂, A)
end
EPGWork_ReIm_DualMVector_Split(::Type{T}, ETL::Int) where {T} = EPGWork_ReIm_DualMVector_Split(T, Val(ETL))

function epg_decay_curve!(A::AbstractVector{T}, work::EPGWork_ReIm_DualMVector_Split{T, Val{ETL}}, θ::EPGOptions{T, Val{ETL}}) where {T, ETL}
    # Unpack workspace
    (; MPSV₁, MPSV₂) = work
    α₁ = B1correctedflipangle(θ, 1)
    α = B1correctedflipangle(θ, 2)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = SA{T} # alias

    # Precompute intermediate variables
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sin½α₁, cos½α₁ = sincos(α₁ / 2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2 * sin½α₁ * cos½α₁
    sinα, cosα = sincos(α)
    cos²½α = (1 + cosα) / 2
    sin²½α = 1 - cos²½α
    a⁺₁, a⁻₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    a⁺, a⁻, c, d = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
    𝒜⁺, 𝒜⁻, 𝒜ᶻ = V[a⁺, a⁻, c], V[a⁻, a⁺, -c], V[-c/2, c/2, d]

    # Initialize magnetization phase state vector (MPSV), pulling n=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Ω₁′ = V[a⁻₁*m₀, 0, -c₁*m₀/2]
        A[1] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        MPSV₁[2] = V[a⁺₁*m₀, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in 2:(ETL÷2)
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `A`
        Ω₁′ = V[𝒜⁻⋅Ωⱼ, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ]
        A[n] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:(n-1)
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[𝒜⁺⋅Ωⱼ₋₁, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ]
        end
        MPSV₁[n] = V[𝒜⁺⋅Ωⱼ, 0, 𝒜ᶻ⋅Ωⱼ₊₁]
        MPSV₁[n+1] = V[𝒜⁺⋅Ωⱼ₊₁, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in (ETL÷2+1):(ETL-1)
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `A`
        Ω₁′ = V[𝒜⁻⋅Ωⱼ, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ]
        A[n] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:(ETL-n)
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[𝒜⁺⋅Ωⱼ₋₁, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds A[ETL] = abs(𝒜⁻ ⋅ MPSV₂[1])

    return A
end

####
#### EPGWork_ReIm_DualPaddedMVector_Vec_Split
####

#=
struct EPGWork_ReIm_DualPaddedMVector_Vec_Split{T, ETL, MPSVType <: AbstractVector{Vec{4, T}}, AType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    A::AType
end
function EPGWork_ReIm_DualPaddedMVector_Vec_Split(::Type{T}, ::Val{ETL}) where {T, ETL}
    MPSV₁ = MVector{ETL, Vec{4, T}}(undef)
    MPSV₂ = MVector{ETL, Vec{4, T}}(undef)
    A     = MVector{ETL, T}(undef)
    return EPGWork_ReIm_DualPaddedMVector_Vec_Split(Val(ETL), MPSV₁, MPSV₂, A)
end
EPGWork_ReIm_DualPaddedMVector_Vec_Split(::Type{T}, ETL::Int) where {T} = EPGWork_ReIm_DualPaddedMVector_Vec_Split(T, Val(ETL))

function epg_decay_curve!(A::AbstractVector{T}, work::EPGWork_ReIm_DualPaddedMVector_Vec_Split{T, Val{ETL}}, θ::EPGOptions{T, Val{ETL}}) where {T, ETL}
    # Unpack workspace
    (; MPSV₁, MPSV₂) = work
    α₁ = B1correctedflipangle(θ, 1)
    α = B1correctedflipangle(θ, 2)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = Vec{4, T} # alias

    # Precompute intermediate variables
    E₁, E₂           = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sin½α₁, cos½α₁   = sincos(α₁ / 2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁            = 2 * sin½α₁ * cos½α₁
    sinα, cosα     = sincos(α)
    cos²½α          = (1 + cosα) / 2
    sin²½α          = 1 - cos²½α
    a⁺₁, a⁻₁, c₁       = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    a⁺, a⁻, c, d   = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
    𝒜⁺, 𝒜⁻, 𝒜ᶻ         = V((a⁺, a⁻, c, 0)), V((a⁻, a⁺, -c, 0)), V((-c / 2, c / 2, d, 0))

    # Initialize magnetization phase state vector (MPSV), pulling n=1 iteration out of loop
    @inbounds begin
        m₀           = sin½α₁ # since αₑₓ = ½α₁
        Ω₁′          = V((a⁻₁ * m₀, 0, -c₁ * m₀ / 2, 0))
        A[1]         = abs(Ω₁′[1])
        MPSV₁[1]     = Ω₁′
        MPSV₁[2]     = V((a⁺₁ * m₀, 0, 0, 0))
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in 2:ETL÷2
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `A`
        Ω₁′      = V((sum(𝒜⁻ * Ωⱼ), sum(𝒜⁻ * Ωⱼ₊₁), sum(𝒜ᶻ * Ωⱼ), 0))
        A[n]     = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:n-1
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V((sum(𝒜⁺ * Ωⱼ₋₁), sum(𝒜⁻ * Ωⱼ₊₁), sum(𝒜ᶻ * Ωⱼ), 0))
        end
        MPSV₁[n]     = V((sum(𝒜⁺ * Ωⱼ), 0, sum(𝒜ᶻ * Ωⱼ₊₁), 0))
        MPSV₁[n+1]   = V((sum(𝒜⁺ * Ωⱼ₊₁), 0, 0, 0))
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in ETL÷2+1:ETL-1
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `A`
        Ω₁′      = V((sum(𝒜⁻ * Ωⱼ), sum(𝒜⁻ * Ωⱼ₊₁), sum(𝒜ᶻ * Ωⱼ), 0))
        A[n]     = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:ETL-n
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V((sum(𝒜⁺ * Ωⱼ₋₁), sum(𝒜⁻ * Ωⱼ₊₁), sum(𝒜ᶻ * Ωⱼ), 0))
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds A[ETL] = abs(sum(𝒜⁻ * MPSV₂[1]))

    return A
end
=#

####
#### EPGWork_ReIm_DualPaddedVector_Split
####

struct EPGWork_ReIm_DualPaddedVector_Split{T, ETL, MPSVType <: AbstractVector{SVector{4, T}}, AType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    A::AType
end
function EPGWork_ReIm_DualPaddedVector_Split(::Type{T}, ETL::Int) where {T}
    MPSV₁ = zeros(SVector{4, T}, ETL)
    MPSV₂ = zeros(SVector{4, T}, ETL)
    A = zeros(T, ETL)
    return EPGWork_ReIm_DualPaddedVector_Split(ETL, MPSV₁, MPSV₂, A)
end

function epg_decay_curve!(A::AbstractVector{T}, work::EPGWork_ReIm_DualPaddedVector_Split{T}, θ::EPGOptions{T}) where {T}
    ETL = length(A)

    # Unpack workspace
    (; MPSV₁, MPSV₂) = work
    α₁ = B1correctedflipangle(θ, 1)
    α = B1correctedflipangle(θ, 2)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = SA{T} # alias

    # Precompute intermediate variables
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sin½α₁, cos½α₁ = sincos(α₁ / 2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2 * sin½α₁ * cos½α₁
    sinα, cosα = sincos(α)
    cos²½α = (1 + cosα) / 2
    sin²½α = 1 - cos²½α
    a⁺₁, a⁻₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    a⁺, a⁻, c, d = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
    𝒜⁺, 𝒜⁻, 𝒜ᶻ = V[a⁺, a⁻, c, 0], V[a⁻, a⁺, -c, 0], V[-c/2, c/2, d, 0]

    # Initialize magnetization phase state vector (MPSV), pulling n=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Ω₁′ = V[a⁻₁*m₀, 0, -c₁*m₀/2, 0]
        A[1] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        MPSV₁[2] = V[a⁺₁*m₀, 0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in 2:(ETL÷2)
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `A`
        Ω₁′ = V[𝒜⁻⋅Ωⱼ, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ, 0]
        A[n] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:(n-1)
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[𝒜⁺⋅Ωⱼ₋₁, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ, 0]
        end
        MPSV₁[n] = V[𝒜⁺⋅Ωⱼ, 0, 𝒜ᶻ⋅Ωⱼ₊₁, 0]
        MPSV₁[n+1] = V[𝒜⁺⋅Ωⱼ₊₁, 0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in (ETL÷2+1):(ETL-1)
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `A`
        Ω₁′ = V[𝒜⁻⋅Ωⱼ, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ, 0]
        A[n] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:(ETL-n)
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[𝒜⁺⋅Ωⱼ₋₁, 𝒜⁻⋅Ωⱼ₊₁, 𝒜ᶻ⋅Ωⱼ, 0]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds A[ETL] = abs(𝒜⁻ ⋅ MPSV₂[1])

    return A
end

####
#### EPGWork_Vec
####

# Flip matrix and relaxation matrix steps are combined into one loop, and SIMD.jl `Vec` types are used instead of `Complex`.
# As this function is called many times during T2mapSEcorr, the micro-optimizations are worth the loss of code readability.
# See `EPGWork_Basic_Cplx` for a more readable, mathematically identicaly implementation.

#=
struct EPGWork_Vec{T, ETL, MPSVType <: AbstractVector{Vec{2, T}}, AType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV::MPSVType
    A::AType
end
function EPGWork_Vec(::Type{T}, ETL::Int) where {T}
    MSPV = zeros(Vec{2, T}, 3 * ETL)
    A = zeros(T, ETL)
    return EPGWork_Vec(ETL, MSPV, A)
end

function epg_decay_curve!(A::AbstractVector{T}, work::EPGWork_Vec{T}, θ::EPGOptions{T}) where {T}
    ETL = length(A)

    ###########################
    # Setup
    (; MPSV) = work
    α₁ = B1correctedflipangle(θ, 1)
    α = B1correctedflipangle(θ, 2)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)

    @inbounds begin
        # Initialize magnetization phase state vector (MPSV)
        E₂², E₁²  = exp(-TE / T2), exp(-TE / T1)
        E₂ = exp(-(TE / 2) / T2)
        m₀      = E₂ * sin(α₁ / 2) # initial population; since αₑₓ = α₁/2
        M1x     = m₀ * cos(α₁ / 2)^2   # M1x, M1y, M1z are elements resulting from first refocusing pulse applied to [m₀, 0, 0]
        M1y     = m₀ - M1x           # M1y = m₀ * sin(α₁/2)^2 = m₀ - m₀ * cos(α₁/2)^2 = m₀ - M1x
        M1z     = -m₀ * sin(α₁) / 2     # Note: this is the imaginary part
        A[1]    = E₂ * abs(M1y)  # first echo amplitude

        # Apply first relaxation matrix iteration on non-zero states
        MPSV[1] = Vec((E₂² * M1y, zero(T)))
        MPSV[2] = zero(Vec{2, T})
        MPSV[3] = Vec((zero(T), E₁² * M1z))
        MPSV[4] = Vec((E₂² * M1x, zero(T)))

        # Extract matrix elements + initialize temporaries
        r₁, r₂, r₃, r₄, r₅ = sin(α), cos(α), sin(α / 2)^2, cos(α / 2)^2, sin(α) / 2 # independent rotation elements rᵢ; Erᵢ and E½rᵢ carry the relaxation over TE and TE/2
        Er₁, Er₂, Er₃, Er₄, Er₅ = E₂² * r₁, E₁² * r₂, E₂² * r₃, E₂² * r₄, E₁² * r₅
        E½r₁, E½r₃, E½r₄         = E₂ * r₁, E₂ * r₃, E₂ * r₄
        Er₁F, Er₅F, E½r₁F      = Vec((-Er₁, Er₁)), Vec((-Er₅, Er₅)), Vec((-E½r₁, E½r₁))
        Mz3                = MPSV[3]
    end

    @inbounds for n in 2:ETL-1
        ###########################
        # Unroll first flipmat/relaxmat iteration
        Vx, Vy  = MPSV[1], MPSV[2]
        E½r₁z     = shufflevector(E½r₁F * Mz3, Val((1, 0)))
        Mz2     = muladd(E½r₃, Vx, muladd(E½r₄, Vy, -E½r₁z)) # flipmat: 2 -> A
        Mz4     = muladd(Er₄, Vx, muladd(Er₃, Vy, E₂ * E½r₁z)) # relaxmat: 1 -> 4, save in buffer
        A[n]    = √(sum(Mz2 * Mz2)) # decay curve coefficient
        MPSV[1] = E₂ * Mz2 # relaxmat: 2 -> 1
        Er₅xy    = shufflevector(Er₅F * (Vx - Vy), Val((1, 0)))
        Mz3     = muladd(Er₂, Mz3, Er₅xy) # relaxmat: 3 -> 3, save in buffer

        ###########################
        # flipmat + relaxmat loop
        for j in 4:3:3*min(n - 1, ETL)
            Vx, Vy, Vz = MPSV[j], MPSV[j+1], MPSV[j+2]
            Er₁z        = shufflevector(Er₁F * Vz, Val((1, 0)))
            MPSV[j]    = Mz4 # relaxmat: assign forward, j -> j+3
            Mz4        = muladd(Er₄, Vx, muladd(Er₃, Vy, Er₁z))
            MPSV[j-2]  = muladd(Er₃, Vx, muladd(Er₄, Vy, -Er₁z)) # relaxmat: assign backwards, j+1 -> j+1-3
            Er₅xy       = shufflevector(Er₅F * (Vx - Vy), Val((1, 0)))
            MPSV[j+2]  = muladd(Er₂, Vz, Er₅xy) # relaxmat: j+2 -> j+2
        end

        ###########################
        # cleanup + zero next elements
        j         = 3i - 2
        Vx        = MPSV[j]
        MPSV[j]   = Mz4 # relaxmat: assign forward, j -> j+3
        MPSV[j-2] = Er₃ * Vx # relaxmat: assign backwards, j+1 -> j+1-3
        MPSV[j+2] = shufflevector(Er₅F * Vx, Val((1, 0))) # relaxmat: j+2 -> j+2
        MPSV[j+3] = Er₄ * Vx # relaxmat: assign forward, j -> j+3
        MPSV[j+1] = Vec((zero(T), zero(T))) # relaxmat: assign backwards, j+1 -> j+1-3
        MPSV[j+5] = Vec((zero(T), zero(T))) # relaxmat: j+2 -> j+2
    end

    ###########################
    # decay curve coefficient
    @inbounds begin
        E½r₁z     = shufflevector(E½r₁F * Mz3, Val((1, 0)))
        Mz2     = muladd(E½r₃, MPSV[1], muladd(E½r₄, MPSV[2], -E½r₁z)) # last iteration of flipmat unrolled
        A[end] = √(sum(Mz2 * Mz2))
    end

    return A
end
=#

####
#### Analytic flip-angle derivatives
####

# α-derivatives of the recursion Mₙ = S²𝒜ₙ Mₙ₋₁:
#   Ṁₙ = S²(𝒜ₙ Ṁₙ₋₁ + 𝒜̇ₙ Mₙ₋₁),   M̈ₙ = S²(𝒜ₙ M̈ₙ₋₁ + 2𝒜̇ₙ Ṁₙ₋₁ + 𝒜̈ₙ Mₙ₋₁),   Ṁ₀ = M̈₀ = 0,
# where M₀ = e₁ is independent of α, a is constant, (ḃ, ċ, ḋ) = κₙ(-a sin αₙ, E₁E₂ cos αₙ, -E₁² sin αₙ), and (b̈, c̈, d̈) = -κₙ²(b, c, d).
@inline flipangle_scale(θ::EPGOptions{T}, n::Int) where {T} = ifelse(n == 1, one(T), θ.β / T(π))
@inline flipangle_scale(θ::EPGConstantFlipAngleOptions{T}, n::Int) where {T} = one(T)
@inline flipangle_scale(θ::EPGIncreasingFlipAnglesOptions{T}, n::Int) where {T} = ifelse(n == 1, θ.α1 / T(π), ifelse(n == 2, θ.α2 / T(π), one(T)))

# Coefficients (a, b, c, d) for pulse angle αₙ = κα and their first two α-derivatives
@inline epg_tangent_coeffs(α::T, κ::T, E₁::T, E₂::T) where {T} = epg_tangent_coeffs(sincos(κ * α), κ, E₁, E₂)
@inline function epg_tangent_coeffs((sinαₙ, cosαₙ)::NTuple{2, T}, κ::T, E₁::T, E₂::T) where {T}
    a = E₂ * E₂ / 2
    b, c, d = a * cosαₙ, E₁ * E₂ * sinαₙ, E₁ * E₁ * cosαₙ
    ḃ, ċ, ḋ = -κ * a * sinαₙ, κ * E₁ * E₂ * cosαₙ, -κ * E₁ * E₁ * sinαₙ
    b̈, c̈, d̈ = -κ^2 * b, -κ^2 * c, -κ^2 * d
    return a, b, c, d, ḃ, ċ, ḋ, b̈, c̈, d̈
end

# Whether κₙ may differ from κₙ₋₁; κₙ = κ₃ for n ≥ 3
@inline epg_tangent_recoeff(::EPGParameterization, n::Int) = 2 <= n <= 3
@inline epg_tangent_recoeff(::EPGConstantFlipAngleOptions, ::Int) = false

####
#### EPGWork_ReIm_Tangent_DualTuple_Split_Dynamic
####

# Decay curve and its α-derivatives for any `EPGParameterization`. F⁻[j] is stored at index j + 1, so that Ω⁻ of state 1 lands in F⁻[0] and is then copied to F⁺[1].
struct EPGWork_ReIm_Tangent_DualTuple_Split_Dynamic{T, ETL, MPSVType <: AbstractVector{T}, AType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::NTuple{9, MPSVType} # (F⁺, F⁻, Z, Ḟ⁺, Ḟ⁻, Ż, F̈⁺, F̈⁻, Z̈)
    MPSV₂::NTuple{9, MPSVType}
    A::AType
end
function EPGWork_ReIm_Tangent_DualTuple_Split_Dynamic(::Type{T}, ETL::Int) where {T}
    MPSV₁ = ntuple(_ -> zeros(T, ETL ÷ 2 + 2), 9)
    MPSV₂ = ntuple(_ -> zeros(T, ETL ÷ 2 + 2), 9)
    A = zeros(T, ETL)
    return EPGWork_ReIm_Tangent_DualTuple_Split_Dynamic(ETL, MPSV₁, MPSV₂, A)
end

epg_decay_curve!(A::AbstractVector{T}, work::EPGWork_ReIm_Tangent_DualTuple_Split_Dynamic{T}, θ::EPGParameterization{T}) where {T} = epg_decay_curve_∂α!(A, nothing, nothing, work, θ)

# A, ∂A, ∂²A with respect to `θ.α` (Units: radians). Any output may be `nothing`; only the derivatives up to the highest requested order are propagated.
function epg_decay_curve_∂α!(A::Union{Nothing, AbstractVector{T}}, ∂A::Union{Nothing, AbstractVector{T}}, ∂²A::Union{Nothing, AbstractVector{T}}, work::EPGWork_ReIm_Tangent_DualTuple_Split_Dynamic{T}, θ::EPGParameterization{T}) where {T}
    order = ∂²A !== nothing ? 2 : ∂A !== nothing ? 1 : 0 # a constant, resolved from the output types
    ETL = echotrainlength(θ)
    echotrainlength(work) == ETL || throw(DimensionMismatch("workspace holds $(echotrainlength(work)) echoes, but the decay curve has $ETL"))
    A === nothing || length(A) == ETL || throw(DimensionMismatch("decay curve has length $(length(A)), expected $ETL"))
    ∂A === nothing || length(∂A) == ETL || throw(DimensionMismatch("decay curve derivative has length $(length(∂A)), expected $ETL"))
    ∂²A === nothing || length(∂²A) == ETL || throw(DimensionMismatch("decay curve second derivative has length $(length(∂²A)), expected $ETL"))

    (; MPSV₁, MPSV₂) = work
    α, TE, T2, T1 = θ.α, echotime(θ), T2time(θ), T1time(θ)
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    m₀, ṁ₀, m̈₀ = sin(α / 2), cos(α / 2) / 2, -sin(α / 2) / 4
    a, b, c, d, ḃ, ċ, ḋ, b̈, c̈, d̈ = epg_tangent_coeffs(α, flipangle_scale(θ, 1), E₁, E₂)

    @inbounds begin
        # M₀ = e₁, with vanishing tangents
        (F⁺₂, F⁻₂, Z₂, Ḟ⁺₂, Ḟ⁻₂, Ż₂, F̈⁺₂, F̈⁻₂, Z̈₂) = MPSV₂
        F⁺₂[1], F⁻₂[2], Z₂[1] = one(T), zero(T), zero(T)
        order >= 1 && (Ḟ⁺₂[1] = Ḟ⁻₂[2] = Ż₂[1] = zero(T))
        order >= 2 && (F̈⁺₂[1] = F̈⁻₂[2] = Z̈₂[1] = zero(T))

        for n in 1:ETL
            epg_tangent_recoeff(θ, n) && ((a, b, c, d, ḃ, ċ, ḋ, b̈, c̈, d̈) = epg_tangent_coeffs(α, flipangle_scale(θ, n), E₁, E₂))
            c′, ċ′, c̈′ = -c / 2, -ċ / 2, -c̈ / 2
            (F⁺₁, F⁻₁, Z₁, Ḟ⁺₁, Ḟ⁻₁, Ż₁, F̈⁺₁, F̈⁻₁, Z̈₁), (F⁺₂, F⁻₂, Z₂, Ḟ⁺₂, Ḟ⁻₂, Ż₂, F̈⁺₂, F̈⁻₂, Z̈₂) = MPSV₁, MPSV₂
            L = min(n, ETL - n + 1)

            # Apply 𝒜ₙ, then shift by S²: F⁺[j+1] ← Ω⁺, F⁻[j-1] ← Ω⁻, Z[j] ← Ωᶻ
            @simd ivdep for j in 1:L
                x, y, z = F⁺₂[j] + F⁻₂[j+1], F⁺₂[j] - F⁻₂[j+1], Z₂[j]
                u = muladd(b, y, c * z)
                F⁺₁[j+1], F⁻₁[j], Z₁[j] = muladd(a, x, u), muladd(a, x, -u), muladd(d, z, c′ * y)
                if order >= 1 # 𝒜ₙ Ṁ + 𝒜̇ₙ M
                    ẋ, ẏ, ż = Ḟ⁺₂[j] + Ḟ⁻₂[j+1], Ḟ⁺₂[j] - Ḟ⁻₂[j+1], Ż₂[j]
                    u̇ = muladd(b, ẏ, muladd(c, ż, muladd(ḃ, y, ċ * z)))
                    Ḟ⁺₁[j+1], Ḟ⁻₁[j], Ż₁[j] = muladd(a, ẋ, u̇), muladd(a, ẋ, -u̇), muladd(d, ż, muladd(c′, ẏ, muladd(ḋ, z, ċ′ * y)))
                end
                if order >= 2 # 𝒜ₙ M̈ + 2𝒜̇ₙ Ṁ + 𝒜̈ₙ M
                    ẍ, ÿ, z̈ = F̈⁺₂[j] + F̈⁻₂[j+1], F̈⁺₂[j] - F̈⁻₂[j+1], Z̈₂[j]
                    ü = muladd(b, ÿ, muladd(c, z̈, muladd(2ḃ, ẏ, muladd(2ċ, ż, muladd(b̈, y, c̈ * z)))))
                    F̈⁺₁[j+1], F̈⁻₁[j], Z̈₁[j] = muladd(a, ẍ, ü), muladd(a, ẍ, -ü), muladd(d, z̈, muladd(c′, ÿ, muladd(2ḋ, ż, muladd(2ċ′, ẏ, muladd(d̈, z, c̈′ * y)))))
                end
            end

            # Ω⁻ of state 1 becomes F⁺[1]; while L grows, the next step also reads F⁻[n], F⁻[n+1] and Z[n+1]
            F⁺₁[1] = F⁻₁[1]
            order >= 1 && (Ḟ⁺₁[1] = Ḟ⁻₁[1])
            order >= 2 && (F̈⁺₁[1] = F̈⁻₁[1])
            if n <= ETL ÷ 2
                F⁻₁[n+1] = F⁻₁[n+2] = Z₁[n+1] = zero(T)
                order >= 1 && (Ḟ⁻₁[n+1] = Ḟ⁻₁[n+2] = Ż₁[n+1] = zero(T))
                order >= 2 && (F̈⁻₁[n+1] = F̈⁻₁[n+2] = Z̈₁[n+1] = zero(T))
            end

            h, ḣ, ḧ = F⁺₁[1], Ḟ⁺₁[1], F̈⁺₁[1]
            s = m₀ * h
            σ = ifelse(signbit(s), -one(T), one(T))
            A === nothing || (A[n] = abs(s))
            ∂A === nothing || (∂A[n] = σ * muladd(ṁ₀, h, m₀ * ḣ))
            ∂²A === nothing || (∂²A[n] = σ * muladd(m̈₀, h, muladd(2ṁ₀, ḣ, m₀ * ḧ)))
            MPSV₁, MPSV₂ = MPSV₂, MPSV₁
        end
    end

    return A
end

function epg_decay_curve_∂α(θ::EPGParameterization{T}) where {T}
    work = EPGWork_ReIm_Tangent_DualTuple_Split_Dynamic(T, echotrainlength(θ))
    ∂A = zeros(T, echotrainlength(θ))
    epg_decay_curve_∂α!(work.A, ∂A, nothing, work, θ)
    return work.A, ∂A
end

function EPGdecaycurve(θ::EPGParameterization{V}, α::Dual{Tg, V}) where {Tg, V} # plain-valued `θ`, with `α` supplying the partials
    A, ∂A = epg_decay_curve_∂α(θ)
    return Dual{Tg}.(A, deg2rad.(∂A) .* Ref(ForwardDiff.partials(α))) # `α` is in degrees, the kernel's derivative is per radian
end
EPGdecaycurve(ETL::Int, α::Dual{Tg, V}, TE::V, T2::V, T1::V, β::V) where {Tg, V <: AbstractFloat} = EPGdecaycurve(EPGOptions((; ETL, α = deg2rad(ForwardDiff.value(α)), TE, T2, T1, β = deg2rad(β))), α)
EPGdecaycurve(ETL::Int, α::Dual{Tg, V}, TE::V, T2::V, T1::V) where {Tg, V <: AbstractFloat} = EPGdecaycurve(EPGConstantFlipAngleOptions((; ETL, α = deg2rad(ForwardDiff.value(α)), TE, T2, T1)), α)

####
#### EPGWork_ReIm_Tangent_Batched_Split_Dynamic
####

# `EPGWork_ReIm_Tangent_DualTuple_Split_Dynamic` for EPG_BATCH_WIDTH T2 times at once, one per lane
struct EPGWork_ReIm_Tangent_Batched_Split_Dynamic{T, ETL, VType <: AbstractVector{T}, AType <: AbstractVector{T}, S <: EPGWork_ReIm_Tangent_DualTuple_Split_Dynamic{T, ETL}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::VType # (F⁺, F⁻, Z, Ḟ⁺, Ḟ⁻, Ż, F̈⁺, F̈⁻, Z̈) of each state and lane, W × 9 × (ETL ÷ 2 + 2), with F⁻ offset as in the single-curve kernel
    MPSV₂::VType
    E₂::VType # E₂ of each lane
    C::VType # coefficients of each lane, W × 10; see `epg_setup_tangent_lanes!`
    decay_curve_work::S
    A::AType
end
function EPGWork_ReIm_Tangent_Batched_Split_Dynamic(::Type{T}, ETL::Int) where {T}
    W = EPG_BATCH_WIDTH
    MPSV₁, MPSV₂ = zeros(T, W * 9 * (ETL ÷ 2 + 2)), zeros(T, W * 9 * (ETL ÷ 2 + 2))
    E₂, C = zeros(T, W), zeros(T, W * 10)
    decay_curve_work = EPGWork_ReIm_Tangent_DualTuple_Split_Dynamic(T, ETL)
    A = zeros(T, ETL)
    return EPGWork_ReIm_Tangent_Batched_Split_Dynamic(ETL, MPSV₁, MPSV₂, E₂, C, decay_curve_work, A)
end

epg_decay_curve!(A::AbstractVector{T}, work::EPGWork_ReIm_Tangent_Batched_Split_Dynamic{T}, θ::EPGParameterization{T}) where {T} = epg_decay_basis_∂α!(A, nothing, nothing, work, θ, SA[T2time(θ)])
epg_decay_basis!(A::AbstractMatrix{T}, work::EPGWork_ReIm_Tangent_Batched_Split_Dynamic{T}, θ::EPGParameterization{T}, T2_times::AbstractVector) where {T} = epg_decay_basis_∂α!(A, nothing, nothing, work, θ, T2_times)

# Coefficients (a, b, c, c′, ḃ, ċ, ċ′, b̈, c̈, c̈′) of pulse n in each lane; returns (d, ḋ, d̈), which are the same in every lane
function epg_setup_tangent_lanes!(work::EPGWork_ReIm_Tangent_Batched_Split_Dynamic{T}, θ::EPGParameterization{T}, n::Int, E₁::T) where {T}
    (; E₂, C) = work
    κ = flipangle_scale(θ, n)
    sincosαₙ = sincos(κ * θ.α)
    d = ḋ = d̈ = zero(T)
    @inbounds for l in 1:EPG_BATCH_WIDTH
        a, b, c, d, ḃ, ċ, ḋ, b̈, c̈, d̈ = epg_tangent_coeffs(sincosαₙ, κ, E₁, E₂[l])
        C[epg_lane(l, 1)], C[epg_lane(l, 2)], C[epg_lane(l, 3)], C[epg_lane(l, 4)] = a, b, c, -c / 2
        C[epg_lane(l, 5)], C[epg_lane(l, 6)], C[epg_lane(l, 7)] = ḃ, ċ, -ċ / 2
        C[epg_lane(l, 8)], C[epg_lane(l, 9)], C[epg_lane(l, 10)] = b̈, c̈, -c̈ / 2
    end
    return d, ḋ, d̈
end

# A, ∂A, ∂²A over the T2 times `T2_times[cols]`, output column `col` holding `T2_times[cols[col]]`; see `epg_decay_curve_∂α!`
function epg_decay_basis_∂α!(A::Union{Nothing, AbstractVecOrMat{T}}, ∂A::Union{Nothing, AbstractVecOrMat{T}}, ∂²A::Union{Nothing, AbstractVecOrMat{T}}, work::EPGWork_ReIm_Tangent_Batched_Split_Dynamic{T}, θ::EPGParameterization{T}, T2_times::AbstractVector, cols::AbstractVector{Int} = eachindex(T2_times)) where {T}
    order = ∂²A !== nothing ? 2 : ∂A !== nothing ? 1 : 0 # a constant, resolved from the output types
    ETL, W, ncols = echotrainlength(θ), EPG_BATCH_WIDTH, length(cols)
    echotrainlength(work) == ETL || throw(DimensionMismatch("workspace holds $(echotrainlength(work)) echoes, but the decay basis has $ETL"))
    A === nothing || size(A, 1) == ETL && size(A, 2) == ncols || throw(DimensionMismatch("decay basis has size $(size(A)), expected ($ETL, $ncols)"))
    ∂A === nothing || size(∂A, 1) == ETL && size(∂A, 2) == ncols || throw(DimensionMismatch("decay basis derivative has size $(size(∂A)), expected ($ETL, $ncols)"))
    ∂²A === nothing || size(∂²A, 1) == ETL && size(∂²A, 2) == ncols || throw(DimensionMismatch("decay basis second derivative has size $(size(∂²A)), expected ($ETL, $ncols)"))

    (; MPSV₁, MPSV₂, E₂, C) = work
    F⁺, F⁻, Z, Ḟ⁺, Ḟ⁻, Ż, F̈⁺, F̈⁻, Z̈ = 1, 2, 3, 4, 5, 6, 7, 8, 9 # state slots
    α, TE = θ.α, echotime(θ)
    E₁ = exp(-(TE / 2) / T1time(θ))
    m₀, ṁ₀, m̈₀ = sin(α / 2), cos(α / 2) / 2, -sin(α / 2) / 4

    @inbounds for col₀ in 1:W:ncols
        if ncols - col₀ + 1 < 4
            @views for col in col₀:ncols
                θcol = restructure(θ, (; T2 = T(T2_times[cols[col]])))
                epg_decay_curve_∂α!(A === nothing ? nothing : A[:, col], ∂A === nothing ? nothing : ∂A[:, col], ∂²A === nothing ? nothing : ∂²A[:, col], work.decay_curve_work, θcol)
            end
            break
        end

        # Trailing lanes repeat the last column and are not written out
        for l in 1:W
            E₂[l] = exp(-(TE / 2) / T(T2_times[cols[min(col₀ + l - 1, ncols)]]))
        end
        d, ḋ, d̈ = epg_setup_tangent_lanes!(work, θ, 1, E₁)

        # M₀ = e₁, with zero tangents
        @simd ivdep for l in 1:W
            MPSV₂[epg_lane(l, F⁺, 1)], MPSV₂[epg_lane(l, F⁻, 2)], MPSV₂[epg_lane(l, Z, 1)] = one(T), zero(T), zero(T)
            order >= 1 && (MPSV₂[epg_lane(l, Ḟ⁺, 1)] = MPSV₂[epg_lane(l, Ḟ⁻, 2)] = MPSV₂[epg_lane(l, Ż, 1)] = zero(T))
            order >= 2 && (MPSV₂[epg_lane(l, F̈⁺, 1)] = MPSV₂[epg_lane(l, F̈⁻, 2)] = MPSV₂[epg_lane(l, Z̈, 1)] = zero(T))
        end

        for n in 1:ETL
            epg_tangent_recoeff(θ, n) && ((d, ḋ, d̈) = epg_setup_tangent_lanes!(work, θ, n, E₁))
            L = min(n, ETL - n + 1)

            # Apply 𝒜ₙ, then shift by S², as in `epg_decay_curve_∂α!`
            for j in 1:L
                @simd ivdep for l in 1:W
                    a, b, c, c′ = C[epg_lane(l, 1)], C[epg_lane(l, 2)], C[epg_lane(l, 3)], C[epg_lane(l, 4)]
                    x, y, z = MPSV₂[epg_lane(l, F⁺, j)] + MPSV₂[epg_lane(l, F⁻, j + 1)], MPSV₂[epg_lane(l, F⁺, j)] - MPSV₂[epg_lane(l, F⁻, j + 1)], MPSV₂[epg_lane(l, Z, j)]
                    u = muladd(b, y, c * z)
                    MPSV₁[epg_lane(l, F⁺, j + 1)], MPSV₁[epg_lane(l, F⁻, j)], MPSV₁[epg_lane(l, Z, j)] = muladd(a, x, u), muladd(a, x, -u), muladd(d, z, c′ * y)

                    if order >= 1
                        ḃ, ċ, ċ′ = C[epg_lane(l, 5)], C[epg_lane(l, 6)], C[epg_lane(l, 7)]
                        ẋ, ẏ, ż = MPSV₂[epg_lane(l, Ḟ⁺, j)] + MPSV₂[epg_lane(l, Ḟ⁻, j + 1)], MPSV₂[epg_lane(l, Ḟ⁺, j)] - MPSV₂[epg_lane(l, Ḟ⁻, j + 1)], MPSV₂[epg_lane(l, Ż, j)]
                        u̇ = muladd(b, ẏ, muladd(c, ż, muladd(ḃ, y, ċ * z)))
                        MPSV₁[epg_lane(l, Ḟ⁺, j + 1)], MPSV₁[epg_lane(l, Ḟ⁻, j)], MPSV₁[epg_lane(l, Ż, j)] = muladd(a, ẋ, u̇), muladd(a, ẋ, -u̇), muladd(d, ż, muladd(c′, ẏ, muladd(ḋ, z, ċ′ * y)))

                        if order >= 2
                            b̈, c̈, c̈′ = C[epg_lane(l, 8)], C[epg_lane(l, 9)], C[epg_lane(l, 10)]
                            ẍ, ÿ, z̈ = MPSV₂[epg_lane(l, F̈⁺, j)] + MPSV₂[epg_lane(l, F̈⁻, j + 1)], MPSV₂[epg_lane(l, F̈⁺, j)] - MPSV₂[epg_lane(l, F̈⁻, j + 1)], MPSV₂[epg_lane(l, Z̈, j)]
                            ü = muladd(b, ÿ, muladd(c, z̈, muladd(2ḃ, ẏ, muladd(2ċ, ż, muladd(b̈, y, c̈ * z)))))
                            MPSV₁[epg_lane(l, F̈⁺, j + 1)], MPSV₁[epg_lane(l, F̈⁻, j)], MPSV₁[epg_lane(l, Z̈, j)] = muladd(a, ẍ, ü), muladd(a, ẍ, -ü), muladd(d, z̈, muladd(c′, ÿ, muladd(2ḋ, ż, muladd(2ċ′, ẏ, muladd(d̈, z, c̈′ * y)))))
                        end
                    end
                end
            end

            @simd ivdep for l in 1:W
                MPSV₁[epg_lane(l, F⁺, 1)] = MPSV₁[epg_lane(l, F⁻, 1)]
                order >= 1 && (MPSV₁[epg_lane(l, Ḟ⁺, 1)] = MPSV₁[epg_lane(l, Ḟ⁻, 1)])
                order >= 2 && (MPSV₁[epg_lane(l, F̈⁺, 1)] = MPSV₁[epg_lane(l, F̈⁻, 1)])
            end
            if n <= ETL ÷ 2
                @simd ivdep for l in 1:W
                    MPSV₁[epg_lane(l, F⁻, n + 1)] = MPSV₁[epg_lane(l, F⁻, n + 2)] = MPSV₁[epg_lane(l, Z, n + 1)] = zero(T)
                    order >= 1 && (MPSV₁[epg_lane(l, Ḟ⁻, n + 1)] = MPSV₁[epg_lane(l, Ḟ⁻, n + 2)] = MPSV₁[epg_lane(l, Ż, n + 1)] = zero(T))
                    order >= 2 && (MPSV₁[epg_lane(l, F̈⁻, n + 1)] = MPSV₁[epg_lane(l, F̈⁻, n + 2)] = MPSV₁[epg_lane(l, Z̈, n + 1)] = zero(T))
                end
            end

            for l in 1:min(W, ncols-col₀+1)
                col = col₀ + l - 1
                h, ḣ, ḧ = MPSV₁[epg_lane(l, F⁺, 1)], MPSV₁[epg_lane(l, Ḟ⁺, 1)], MPSV₁[epg_lane(l, F̈⁺, 1)]
                s = m₀ * h
                σ = ifelse(signbit(s), -one(T), one(T))
                A === nothing || (A[n, col] = abs(s))
                ∂A === nothing || (∂A[n, col] = σ * muladd(ṁ₀, h, m₀ * ḣ))
                ∂²A === nothing || (∂²A[n, col] = σ * muladd(m̈₀, h, muladd(2ṁ₀, ḣ, m₀ * ḧ)))
            end
            MPSV₁, MPSV₂ = MPSV₂, MPSV₁
        end
    end

    return A
end

# Linear index of lane l of slot s, and of slot s of state j
@inline epg_lane(l::Int, s::Int) = l + EPG_BATCH_WIDTH * (s - 1)
@inline epg_lane(l::Int, s::Int, j::Int) = l + EPG_BATCH_WIDTH * ((s - 1) + 9 * (j - 1))

####
#### Exact cosine-series representation of the constant-flip-angle EPG basis
####

# For a constant flip angle train, (b, d) ∝ cos α and c ∝ sin α, so by induction F⁺ and F⁻ are cosine and Z sine polynomials in α, each gaining at most one degree per pulse.
# Hence, A[i, j] = |m₀ Σₖ Â[i, k+1, j] cos kα| over k = 0, …, i, where Â is constant w.r.t. α.
struct EPGCosineSeriesBasis{T}
    ETL::Int # echo train length, and also the cosine series degree
    nT2::Int # number of T2 times, i.e. basis columns
    coeffs::Vector{T} # Â packed per column into blocks of 4 echoes, each interleaved over k = 0, …, min(4block, ETL) and zero-padded
    cosk::Vector{T} # cosk[k+1] = cos kα
    ksink::Vector{T} # ksink[k+1] = k sin kα
    k²cosk::Vector{T} # k²cosk[k+1] = k² cos kα
end

function EPGCosineSeriesBasis(θ::EPGConstantFlipAngleOptions{T}, T2_times::AbstractVector) where {T}
    ETL, nT2 = echotrainlength(θ), length(T2_times)
    Â = epg_cosine_coefficients(θ, T2_times)

    coeffs = zeros(T, 4 * nT2 * sum(block -> min(4block, ETL) + 1, 1:cld(ETL, 4)))
    idx = 0
    for col in 1:nT2, block in 1:cld(ETL, 4), k in 0:min(4block, ETL), r in 1:4
        i = 4 * (block - 1) + r
        coeffs[idx+=1] = i <= ETL && k <= i ? Â[i, k+1, col] : zero(T)
    end

    return EPGCosineSeriesBasis{T}(ETL, nT2, coeffs, zeros(T, ETL + 1), zeros(T, ETL + 1), zeros(T, ETL + 1))
end

# Per-thread copy sharing the read-only coefficients, with its own feature buffers for `cosine_features!` to overwrite
EPGCosineSeriesBasis(decay_basis_work::EPGCosineSeriesBasis{T}) where {T} = EPGCosineSeriesBasis{T}(decay_basis_work.ETL, decay_basis_work.nT2, decay_basis_work.coeffs, similar(decay_basis_work.cosk), similar(decay_basis_work.ksink), similar(decay_basis_work.k²cosk))

# Cosine coefficients Â[i, k+1, col] of hᵢ for the T2 time `T2_times[col]`, by the recursion on the coefficients of F⁺, F⁻ and Z as opposed to their values
function epg_cosine_coefficients(θ::EPGConstantFlipAngleOptions{T}, T2_times::AbstractVector) where {T}
    ETL, nT2, W = echotrainlength(θ), length(T2_times), EPG_BATCH_WIDTH
    TE, T1 = echotime(θ), T1time(θ)
    E₁ = exp(-(TE / 2) / T1)
    Â = zeros(T, ETL, ETL + 1, nT2)
    Âₗ, a, E₁E₂ = zeros(T, W, ETL, ETL + 1), zeros(T, W), zeros(T, W)
    MPSV = ntuple(_ -> zeros(T, W, ETL + 3, ETL + 2), 6)

    for col₀ in 1:W:nT2
        for l in 1:W
            E₂ = exp(-(TE / 2) / T(T2_times[min(col₀ + l - 1, nT2)]))
            a[l], E₁E₂[l] = E₂ * E₂ / 2, E₁ * E₂
        end

        epg_cosine_coefficients!(Âₗ, MPSV, a, E₁E₂, E₁ * E₁)

        @inbounds for l in 1:min(W, nT2-col₀+1), k in 1:(ETL+1), i in 1:ETL
            Â[i, k, col₀+l-1] = Âₗ[l, i, k]
        end
    end

    return Â
end

# Cosine coefficients for a batch of W T2 times, one per lane. F⁺[l, k+2, j] holds the degree-k coefficient of state j in lane l, and is zero for k = -1 and k > n.
# With subscripts ∓ denoting degree k ∓ 1, the products at degree k are
#   cos α y = ((1 + δₖ₁)y₋ + y₊)/2,   sin α y = ((1 + δₖ₁)y₋ - y₊)/2,   cos α z = (z₋ + z₊)/2,   sin α z = (z₊ - z₋)/2,
# where the degree-0 term of y reaches degree 1 through both cos(±α), and Z has no degree-0 term.
function epg_cosine_coefficients!(Â::Array{T, 3}, MPSV::NTuple{6, Array{T, 3}}, a::Vector{T}, E₁E₂::Vector{T}, E₁²::T) where {T}
    W, ETL = EPG_BATCH_WIDTH, size(Â, 2) # a compile-time lane count
    F⁺₂, F⁻₂, Z₂, F⁺₁, F⁻₁, Z₁ = MPSV
    fill!(Â, zero(T))
    foreach(S -> fill!(S, zero(T)), MPSV)

    # Pulse 1 from M₀ = e₁: h₁ = F⁺[1] = a - b, F⁺[2] = a + b, Z[1] = c′
    @inbounds @simd ivdep for l in 1:W
        Â[l, 1, 1], Â[l, 1, 2] = a[l], -a[l]
        F⁺₂[l, 2, 1], F⁺₂[l, 3, 1] = a[l], -a[l]
        F⁺₂[l, 2, 2], F⁺₂[l, 3, 2] = a[l], a[l]
        Z₂[l, 3, 1] = -E₁E₂[l] / 2
    end

    @inbounds for n in 2:ETL
        L = min(n, ETL - n + 1)
        for j in 1:L
            F⁻dst, jdst = j == 1 ? (F⁺₁, 1) : (F⁻₁, j - 1) # Ω⁻ of state 1 becomes F⁺[1]
            for k in 0:n
                r = k + 2
                δₖ₁ = k == 1 ? one(T) : zero(T)
                𝟙ₖ = k == 0 ? zero(T) : one(T)
                @simd ivdep for l in 1:W
                    x = F⁺₂[l, r, j] + F⁻₂[l, r, j]
                    y₋ = F⁺₂[l, r-1, j] - F⁻₂[l, r-1, j]
                    y₋ = muladd(δₖ₁, y₋, y₋)
                    y₊ = F⁺₂[l, r+1, j] - F⁻₂[l, r+1, j]
                    z₋, z₊ = Z₂[l, r-1, j], Z₂[l, r+1, j]
                    ax = a[l] * x
                    u = muladd(a[l] / 2, y₋ + y₊, (E₁E₂[l] / 2) * (z₊ - z₋)) # b y + c z
                    F⁺₁[l, r, j+1] = ax + u
                    F⁻dst[l, r, jdst] = ax - u
                    Z₁[l, r, j] = 𝟙ₖ * muladd(-E₁E₂[l] / 4, y₋ - y₊, (E₁² / 2) * (z₋ + z₊)) # d z + c′ y
                end
            end
        end

        # While L grows, the next step reads F⁻[L], F⁻[L+1] and Z[L+1]
        @views fill!(F⁻₁[:, :, L:(L+1)], zero(T))
        @views fill!(Z₁[:, :, L+1], zero(T))
        for k in 0:n
            @simd ivdep for l in 1:W
                Â[l, n, k+1] = F⁺₁[l, k+2, 1]
            end
        end

        F⁺₁, F⁻₁, Z₁, F⁺₂, F⁻₂, Z₂ = F⁺₂, F⁻₂, Z₂, F⁺₁, F⁻₁, Z₁
    end

    return Â
end

# Compute cos kα, k sin kα and k² cos kα for k = 0, …, ETL, by repeated rotation through α, renormalizing onto the unit circle
function cosine_features!(decay_basis_work::EPGCosineSeriesBasis{T}, α::T) where {T}
    (; ETL, cosk, ksink, k²cosk) = decay_basis_work
    sinα, cosα = sin(α), cos(α)
    sinkα, coskα = zero(T), one(T)
    cosk[1], ksink[1], k²cosk[1] = coskα, zero(T), zero(T)

    @inbounds for k in 1:ETL
        sinkα, coskα = muladd(sinα, coskα, cosα * sinkα), muladd(-sinα, sinkα, cosα * coskα)
        scale = (3 - muladd(sinkα, sinkα, coskα * coskα)) / 2 # one Newton step for 1/√(sin² + cos²)
        sinkα, coskα = scale * sinkα, scale * coskα
        cosk[k+1] = coskα
        ksink[k+1] = k * sinkα
        k²cosk[k+1] = k^2 * coskα
    end

    return decay_basis_work
end

function epg_decay_basis!(A::Matrix{T}, decay_basis_work::EPGCosineSeriesBasis{T}, α::T) where {T}
    (; coeffs, cosk) = decay_basis_work
    ETL, nT2 = size(A)
    cosine_features!(decay_basis_work, α)

    # Four columns per pass; a trailing group repeats the last column
    V = SVector{4, T}
    blocks = reinterpret(V, coeffs)
    cs = sum(block -> min(4block, ETL) + 1, 1:cld(ETL, 4)) # coefficient blocks per column
    m₀ = sin(α / 2)

    @inbounds for col in 1:4:nT2
        col₁, col₂, col₃, col₄ = col, min(col + 1, nT2), min(col + 2, nT2), min(col + 3, nT2)
        o₁, o₂, o₃, o₄ = (col₁ - 1) * cs, (col₂ - 1) * cs, (col₃ - 1) * cs, (col₄ - 1) * cs
        oblock = 0 # offset of `block` within a column

        for block in 1:cld(ETL, 4)
            i₀ = 4 * (block - 1)
            K = min(i₀ + 4, ETL) + 1
            h₁ = h₂ = h₃ = h₄ = zero(V)
            for k in 1:K
                cosₖ = cosk[k]
                h₁ = muladd.(blocks[o₁+oblock+k], cosₖ, h₁)
                h₂ = muladd.(blocks[o₂+oblock+k], cosₖ, h₂)
                h₃ = muladd.(blocks[o₃+oblock+k], cosₖ, h₃)
                h₄ = muladd.(blocks[o₄+oblock+k], cosₖ, h₄)
            end
            oblock += K

            for r in 1:min(4, ETL-i₀)
                A[i₀+r, col₁] = abs(m₀ * h₁[r])
                A[i₀+r, col₂] = abs(m₀ * h₂[r])
                A[i₀+r, col₃] = abs(m₀ * h₃[r])
                A[i₀+r, col₄] = abs(m₀ * h₄[r])
            end
        end
    end

    return A
end

# Basis columns `js` of A, ∂A, ∂²A into output columns `cols`, from h = Σₖ Âₖ cos kα, ḣ = -Σₖ k Âₖ sin kα and ḧ = -Σₖ k² Âₖ cos kα.
# `A` may be `nothing`. Requires up-to-date precomputed `cosine_features!` at α.
@inline function epg_decay_basis_∂α_cols!(A::Union{Nothing, AbstractVecOrMat{T}}, ∂A::AbstractVecOrMat{T}, ∂²A::AbstractVecOrMat{T}, decay_basis_work::EPGCosineSeriesBasis{T}, α::T, cols::NTuple{N, Int}, js::NTuple{N, Int}) where {T, N}
    (; ETL, coeffs, cosk, ksink, k²cosk) = decay_basis_work
    m₀, ṁ₀, m̈₀ = sin(α / 2), cos(α / 2) / 2, -sin(α / 2) / 4
    V = SVector{4, T}
    blocks = reinterpret(V, coeffs)
    cs = sum(block -> min(4block, ETL) + 1, 1:cld(ETL, 4)) # coefficient blocks per column
    os = map(j -> (j - 1) * cs, js) # offset of each column
    oblock = 0 # offset of `block` within a column

    @inbounds for block in 1:cld(ETL, 4)
        i₀ = 4 * (block - 1)
        K = min(i₀ + 4, ETL) + 1

        # Σₖ Âₖ cos kα, Σₖ k Âₖ sin kα, Σₖ k² Âₖ cos kα of each column
        Σs = ntuple(_ -> (zero(V), zero(V), zero(V)), Val(N))
        for k in 1:K
            cosₖ, ksinₖ, k²cosₖ = cosk[k], ksink[k], k²cosk[k]
            oblockₖ = oblock + k # fresh binding; otherwise, the closure would box `oblock`
            Σs = map(Σs, os) do (Σcos, Σksin, Σk²cos), o
                @inline
                Âₖ = @inbounds blocks[o+oblockₖ]
                return muladd.(Âₖ, cosₖ, Σcos), muladd.(Âₖ, ksinₖ, Σksin), muladd.(Âₖ, k²cosₖ, Σk²cos)
            end
        end
        oblock += K

        foreach(Σs, cols) do (Σcos, Σksin, Σk²cos), col
            @inline
            h, ḣ, ḧ = Σcos, -Σksin, -Σk²cos
            s = m₀ .* h
            σ = ifelse.(signbit.(s), -one(T), one(T))
            ∂s, ∂²s = σ .* muladd.(ṁ₀, h, m₀ .* ḣ), σ .* muladd.(m̈₀, h, muladd.(2ṁ₀, ḣ, m₀ .* ḧ))
            @inbounds for r in 1:min(4, ETL-i₀)
                A === nothing || (A[i₀+r, col] = abs(s[r]))
                ∂A[i₀+r, col] = ∂s[r]
                ∂²A[i₀+r, col] = ∂²s[r]
            end
        end
    end

    return A, ∂A, ∂²A
end

# Basis column j of A, ∂A, ∂²A; see `epg_decay_basis_∂α_cols!`
epg_decay_basis_∂α_col!(A::Union{Nothing, AbstractVector{T}}, ∂A::AbstractVector{T}, ∂²A::AbstractVector{T}, decay_basis_work::EPGCosineSeriesBasis{T}, α::T, j::Int) where {T} = epg_decay_basis_∂α_cols!(A, ∂A, ∂²A, decay_basis_work, α, (1,), (j,))

# Basis columns `cols` of A, ∂A, ∂²A, output column `col` holding basis column `cols[col]`; see `epg_decay_basis_∂α_cols!`
function epg_decay_basis_∂α!(A::Union{Nothing, AbstractMatrix{T}}, ∂A::AbstractMatrix{T}, ∂²A::AbstractMatrix{T}, decay_basis_work::EPGCosineSeriesBasis{T}, α::T, cols::AbstractVector{Int}) where {T}
    col, ncols = 1, length(cols)
    @inbounds while col + 3 <= ncols
        epg_decay_basis_∂α_cols!(A, ∂A, ∂²A, decay_basis_work, α, (col, col + 1, col + 2, col + 3), (cols[col], cols[col+1], cols[col+2], cols[col+3]))
        col += 4
    end
    @inbounds if col + 2 <= ncols
        epg_decay_basis_∂α_cols!(A, ∂A, ∂²A, decay_basis_work, α, (col, col + 1, col + 2), (cols[col], cols[col+1], cols[col+2]))
    elseif col + 1 <= ncols
        epg_decay_basis_∂α_cols!(A, ∂A, ∂²A, decay_basis_work, α, (col, col + 1), (cols[col], cols[col+1]))
    elseif col <= ncols
        epg_decay_basis_∂α_cols!(A, ∂A, ∂²A, decay_basis_work, α, (col,), (cols[col],))
    end
    return A
end
