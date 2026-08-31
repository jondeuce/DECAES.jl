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
####
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
@inline decaycurve(work::AbstractEPGWorkspace) = work.dc

@inline EPGdecaycurve_work(θ::EPGParameterization{T, ETL}) where {T, ETL} = default_cache(θ)
@inline EPGdecaycurve_work(::Type{T}, ETL::Int) where {T} = EPGWork_ReIm_DualVector_Split_Dynamic(T, ETL) # default for dynamic `ETL`
@inline EPGdecaycurve_work(::Type{T}, ::Val{ETL}) where {T, ETL} = EPGWork_ReIm_DualMVector_Split(T, Val(ETL)) # default for static `ETL`

# Workspace used for computing decay bases over a T2 grid.
# Constant-flip-angle bases use the lane-batched kernel; see `epg_decay_basis!`.
@inline EPGdecaybasis_work(θ::EPGParameterization{T}) where {T} = EPGdecaycurve_work(θ)
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

Computes the normalized echo decay curve for a multi spin echo sequence
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
@inline EPGdecaycurve!(work::AbstractEPGWorkspace{T}, θ::EPGParameterization{T}) where {T} = EPGdecaycurve!(decaycurve(work), work, θ)
@inline EPGdecaycurve!(dc::AbstractVector{T}, work::AbstractEPGWorkspace{T}, θ::EPGParameterization{T}) where {T} = epg_decay_curve!(dc, work, θ)

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

struct EPGWork_Basic_Cplx{T, ETL, MPSVType <: AbstractVector{SVector{3, Complex{T}}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV::MPSVType
    dc::DCType
end
function EPGWork_Basic_Cplx(::Type{T}, ETL::Int) where {T}
    MSPV = zeros(SVector{3, Complex{T}}, ETL)
    dc = zeros(T, ETL)
    return EPGWork_Basic_Cplx(ETL, MSPV, dc)
end

# Compute a basis function under the extended phase graph algorithm. The magnetization phase state vector (MPSV) is
# successively modified by applying relaxation for TE/2, then a refocusing pulse as described by Hennig (1988),
# then transitioning phase states as given by Hennig (1988) but corrected by Jones (1997), and a finally relaxing for TE/2.
# See the appendix in Prasloski (2012) for details:
#    https://doi.org/10.1002/mrm.23157

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_Basic_Cplx{T}, θ::EPGOptions{T}) where {T}
    ETL = length(dc)

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
    E1, E2 = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    E = SA{T}[E2, E2, E1]
    R₁ = element_flipmat(α₁)
    R₂₊ = element_flipmat(α)

    # Initialize magnetization phase state vector (MPSV)
    @inbounds for j in 1:ETL
        MPSV[j] = V[0, 0, 0]
    end
    @inbounds MPSV[1] = V[sin(αₑₓ), 0, 0] # initial magnetization in F1 state

    @inbounds for n in 1:ETL
        # Relaxation for TE/2, followed by flip matrix
        R = n == 1 ? R₁ : R₂₊
        for j in 1:ETL
            MPSV[j] = R * (E .* MPSV[j])
        end

        # Transition between phase states
        Ωⱼ, Ωⱼ₊₁ = MPSV[1], MPSV[2]
        MPSV[1] = V[Ωⱼ[2], Ωⱼ₊₁[2], Ωⱼ[3]] # (F⁺₁, F⁺₁*, Z₁)⁺ = (F⁺₁*, F⁺₂*, Z₁)
        for j in 2:ETL-1
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV[j+1]
            MPSV[j] = V[Ωⱼ₋₁[1], Ωⱼ₊₁[2], Ωⱼ[3]] # (Fᵢ, Fᵢ*, Zᵢ)⁺ = (Fᵢ₋₁, Fᵢ₊₁*, Zᵢ)
        end
        Ωⱼ₋₁, Ωⱼ = Ωⱼ, Ωⱼ₊₁
        MPSV[ETL] = V[Ωⱼ₋₁[1], 0, Ωⱼ[3]] # (Fₙ, Fₙ*, Zₙ)⁺ = (Fₙ₋₁, 0, Zₙ)

        # Relaxation for TE/2
        for j in 1:ETL
            MPSV[j] = E .* MPSV[j] # Relaxation for TE/2
        end
        dc[n] = abs(MPSV[1][1]) # first echo amplitude
    end

    return dc
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_Basic_Cplx{T}, θ::EPGParameterization{T}) where {T}
    ETL = length(dc)

    # Unpack workspace
    (; MPSV) = work
    αₑₓ = B1correctedflipangle(θ, 0)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = SA{Complex{T}} # alias

    # Precompute compute element flip matrices and other intermediate variables
    E1, E2 = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    E = SA{T}[E2, E2, E1]

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
        for j in 2:ETL-1
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV[j+1]
            MPSV[j] = V[Ωⱼ₋₁[1], Ωⱼ₊₁[2], Ωⱼ[3]] # (Fᵢ, Fᵢ*, Zᵢ)⁺ = (Fᵢ₋₁, Fᵢ₊₁*, Zᵢ)
        end
        Ωⱼ₋₁, Ωⱼ = Ωⱼ, Ωⱼ₊₁
        MPSV[ETL] = V[Ωⱼ₋₁[1], 0, Ωⱼ[3]] # (Fₙ, Fₙ*, Zₙ)⁺ = (Fₙ₋₁, 0, Zₙ)

        # Relaxation for TE/2
        for j in 1:ETL
            MPSV[j] = E .* MPSV[j] # Relaxation for TE/2
        end
        dc[n] = abs(MPSV[1][1]) # first echo amplitude
    end

    return dc
end

####
#### EPGWork_ReIm
####

struct EPGWork_ReIm{T, ETL, MPSVType <: AbstractVector{SVector{3, T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV::MPSVType
    dc::DCType
end
function EPGWork_ReIm(::Type{T}, ETL::Int) where {T}
    MSPV = zeros(SVector{3, T}, ETL)
    dc = zeros(T, ETL)
    return EPGWork_ReIm(ETL, MSPV, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm{T}, θ::EPGOptions{T}) where {T}
    ETL = length(dc)

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
    a₁, b₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    a, b, c, d = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
    F⁺, F⁻, Z = V[a, b, c], V[b, a, -c], V[-c/2, c/2, d]

    # Initialize magnetization phase state vector (MPSV), pulling n=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Ω₁′ = V[b₁*m₀, 0, -c₁*m₀/2]
        dc[1] = abs(Ω₁′[1])
        MPSV[1] = Ω₁′
        MPSV[2] = V[a₁*m₀, 0, 0]
        MPSV[3] = V[0, 0, 0]
    end

    @inbounds for n in 2:ETL-1
        # j = 1, initialize and update `dc`
        Ωⱼ, Ωⱼ₊₁ = MPSV[1], MPSV[2]
        Ω₁′ = V[F⁻⋅Ωⱼ, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ]
        dc[n] = abs(Ω₁′[1])
        MPSV[1] = Ω₁′

        # inner loop
        jup = min(n, ETL - n)
        for j in 2:jup
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV[j+1]
            MPSV[j] = V[F⁺⋅Ωⱼ₋₁, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ]
        end

        # cleanup for next iteration
        if n == jup
            Ωⱼ₋₁ = Ωⱼ
            MPSV[n+1] = V[F⁺⋅Ωⱼ₋₁, 0, 0]
            MPSV[n+2] = V[0, 0, 0]
        end
    end

    @inbounds dc[ETL] = abs(F⁻ ⋅ MPSV[1])

    return dc
end

####
#### EPGWork_ReIm_Generated
####

#=
struct EPGWork_ReIm_Generated{T, ETL, MPSVType <: AbstractVector{SVector{3, T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV::MPSVType
    dc::DCType
end
function EPGWork_ReIm_Generated(::Type{T}, ::Val{ETL}) where {T, ETL}
    MSPV = SizedVector{ETL, SVector{3, T}}(undef)
    dc   = SizedVector{ETL, T}(undef)
    return EPGWork_ReIm_Generated(Val(ETL), MSPV, dc)
end
EPGWork_ReIm_Generated(::Type{T}, ETL::Int) where {T} = EPGWork_ReIm_Generated(T, Val(ETL))

function epg_decay_curve_impl!(dc::Type{A}, work::Type{W}, θ::Type{O}) where {T, ETL, A <: AbstractVector{T}, W <: EPGWork_ReIm_Generated{T, Val{ETL}}, O <: EPGOptions{T, Val{ETL}}}
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
        a₁, b₁, c₁       = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
        a, b, c, d   = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
        F⁺, F⁻, Z         = V[a, b, c], V[b, a, -c], V[-c/2, c/2, d]

        # Initialize MPSV vector elements
        $([
            :($(MPSV(n)) = zero(SVector{3, $T}))
            for n in 1:ETL
        ]...)

        # Initialize magnetization phase state vector (MPSV), pulling n=1 iteration out of loop
        @inbounds begin
            m₀         = sin½α₁ # since αₑₓ = ½α₁
            Ω₁′        = V[b₁*m₀, 0, -c₁*m₀/2]
            dc[1]      = abs(Ω₁′[1])
            $(MPSV(1)) = Ω₁′
            $(MPSV(2)) = V[a₁*m₀, 0, 0]
        end

        # Main loop
        $([
            quote
                # Initialize and update `dc` (j = 1)
                @inbounds begin
                    Ωⱼ, Ωⱼ₊₁   = $(MPSV(1)), $(MPSV(2))
                    Ω₁′        = V[F⁻⋅Ωⱼ, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ]
                    dc[$n]     = abs(Ω₁′[1])
                    $(MPSV(1)) = Ω₁′
                end

                # Inner loop
                $([
                    quote
                        (Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁) = (Ωⱼ, Ωⱼ₊₁, $(MPSV(j + 1)))
                        $(MPSV(j))       = V[F⁺⋅Ωⱼ₋₁, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ]
                    end
                    for j in 2:min(n, ETL - n)+1
                ]...)
            end
            for n in 2:ETL-1
        ]...)

        # Last echo
        @inbounds dc[$ETL] = abs(F⁻ ⋅ $(MPSV(1)))

        return dc
    end
end

@generated function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_Generated{T}, θ::EPGOptions{T}) where {T}
    return epg_decay_curve_impl!(dc, work, θ)
end
=#

####
#### EPGWork_ReIm_DualVector
####

struct EPGWork_ReIm_DualVector{T, ETL, MPSVType <: AbstractVector{SVector{3, T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualVector(::Type{T}, ETL::Int) where {T}
    MPSV₁ = zeros(SVector{3, T}, ETL)
    MPSV₂ = zeros(SVector{3, T}, ETL)
    dc = zeros(T, ETL)
    return EPGWork_ReIm_DualVector(ETL, MPSV₁, MPSV₂, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualVector{T}, θ::EPGOptions{T}) where {T}
    ETL = length(dc)

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
    a₁, b₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    a, b, c, d = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
    F⁺, F⁻, Z = V[a, b, c], V[b, a, -c], V[-c/2, c/2, d]

    # Initialize magnetization phase state vector (MPSV), pulling n=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Ω₁′ = V[b₁*m₀, 0, -c₁*m₀/2]
        dc[1] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        MPSV₁[2] = V[a₁*m₀, 0, 0]
        MPSV₁[3] = V[0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in 2:ETL-1
        # j = 1, initialize and update `dc`
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2]
        Ω₁′ = V[F⁻⋅Ωⱼ, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ]
        dc[n] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′

        # inner loop
        jup = min(n, ETL - n)
        @simd for j in 2:jup
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⁺⋅Ωⱼ₋₁, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ]
        end

        # cleanup for next iteration
        if n == jup
            Ωⱼ₋₁ = Ωⱼ
            MPSV₁[n+1] = V[F⁺⋅Ωⱼ₋₁, 0, 0]
            MPSV₁[n+2] = V[0, 0, 0]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F⁻ ⋅ MPSV₂[1])

    return dc
end

####
#### EPGWork_ReIm_DualVector_Split
####

struct EPGWork_ReIm_DualVector_Split{T, ETL, MPSVType <: AbstractVector{SVector{3, T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualVector_Split(::Type{T}, ETL::Int) where {T}
    MPSV₁ = zeros(SVector{3, T}, ETL)
    MPSV₂ = zeros(SVector{3, T}, ETL)
    dc = zeros(T, ETL)
    return EPGWork_ReIm_DualVector_Split(ETL, MPSV₁, MPSV₂, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualVector_Split{T}, θ::EPGOptions{T}) where {T}
    ETL = length(dc)

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
    a₁, b₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    a, b, c, d = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
    F⁺, F⁻, Z = V[a, b, c], V[b, a, -c], V[-c/2, c/2, d]

    # Initialize magnetization phase state vector (MPSV), pulling n=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Ω₁′ = V[b₁*m₀, 0, -c₁*m₀/2]
        dc[1] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        MPSV₁[2] = V[a₁*m₀, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in 2:ETL÷2
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Ω₁′ = V[F⁻⋅Ωⱼ, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ]
        dc[n] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:n-1
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⁺⋅Ωⱼ₋₁, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ]
        end
        MPSV₁[n] = V[F⁺⋅Ωⱼ, 0, Z⋅Ωⱼ₊₁]
        MPSV₁[n+1] = V[F⁺⋅Ωⱼ₊₁, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in ETL÷2+1:ETL-1
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Ω₁′ = V[F⁻⋅Ωⱼ, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ]
        dc[n] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:ETL-n
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⁺⋅Ωⱼ₋₁, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F⁻ ⋅ MPSV₂[1])

    return dc
end

####
#### EPGWork_ReIm_DualVector_Split_Dynamic
####

struct EPGWork_ReIm_DualVector_Split_Dynamic{T, ETL, MPSVType <: AbstractVector{SVector{3, T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualVector_Split_Dynamic(::Type{T}, ETL::Int) where {T}
    MPSV₁ = zeros(SVector{3, T}, ETL)
    MPSV₂ = zeros(SVector{3, T}, ETL)
    dc = zeros(T, ETL)
    return EPGWork_ReIm_DualVector_Split_Dynamic(ETL, MPSV₁, MPSV₂, dc)
end

function epg_decay_curve!(dc::AbstractVector, work::EPGWork_ReIm_DualVector_Split_Dynamic{T}, θ::EPGOptions{T}) where {T}
    V = SA{T} # alias
    ETL = length(dc)

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
    a₁, b₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    a, b, c, d = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
    F⁺, F⁻, Z = V[a, b, c], V[b, a, -c], V[-c/2, c/2, d]

    @inbounds begin
        # n = 1 iteration
        # Initialize magnetization phase state vector (MPSV)
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Ω₁ = V[b₁*m₀, 0, -c₁*m₀/2]
        MPSV₁[1] = Ω₁
        MPSV₁[2] = V[a₁*m₀, 0, 0]

        dc[1] = abs(Ω₁[1])
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁

        # n = 2 iteration
        Ω₁, Ω₂ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        FΩ₁, F⁻Ω₁, ZΩ₁ = F⁺ ⋅ Ω₁, F⁻ ⋅ Ω₁, Z ⋅ Ω₁
        FΩ₂, F⁻Ω₂, ZΩ₂ = F⁺ ⋅ Ω₂, F⁻ ⋅ Ω₂, Z ⋅ Ω₂

        MPSV₁[1] = V[F⁻Ω₁, F⁻Ω₂, ZΩ₁]
        MPSV₁[2] = V[FΩ₁, 0, ZΩ₂]
        MPSV₁[3] = V[FΩ₂, 0, 0]

        dc[2] = abs(F⁻Ω₁)
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in 3:ETL÷2
        Ω₁, Ω₂, Ω₃ = MPSV₂[1], MPSV₂[2], MPSV₂[3] # j = 1, initialize and update `dc`
        FΩ₁, F⁻Ω₁, ZΩ₁ = F⁺ ⋅ Ω₁, F⁻ ⋅ Ω₁, Z ⋅ Ω₁
        FΩ₂, F⁻Ω₂, ZΩ₂ = F⁺ ⋅ Ω₂, F⁻ ⋅ Ω₂, Z ⋅ Ω₂
        FΩ₃, F⁻Ω₃, ZΩ₃ = F⁺ ⋅ Ω₃, F⁻ ⋅ Ω₃, Z ⋅ Ω₃

        MPSV₁[1] = V[F⁻Ω₁, F⁻Ω₂, ZΩ₁]
        MPSV₁[2] = V[FΩ₁, F⁻Ω₃, ZΩ₂]

        for j in 3:n-1
            FΩ₁, FΩ₂, ZΩ₂ = FΩ₂, FΩ₃, ZΩ₃
            Ω₃ = MPSV₂[j+1]
            FΩ₃, F⁻Ω₃, ZΩ₃ = F⁺ ⋅ Ω₃, F⁻ ⋅ Ω₃, Z ⋅ Ω₃
            MPSV₁[j] = V[FΩ₁, F⁻Ω₃, ZΩ₂]
        end

        MPSV₁[n] = V[FΩ₂, 0, ZΩ₃]
        MPSV₁[n+1] = V[FΩ₃, 0, 0]

        dc[n] = abs(F⁻Ω₁)
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in ETL÷2+1:ETL-1
        Ω₁, Ω₂, Ω₃ = MPSV₂[1], MPSV₂[2], MPSV₂[3] # j = 1, initialize and update `dc`
        FΩ₁, F⁻Ω₁, ZΩ₁ = F⁺ ⋅ Ω₁, F⁻ ⋅ Ω₁, Z ⋅ Ω₁
        FΩ₂, F⁻Ω₂, ZΩ₂ = F⁺ ⋅ Ω₂, F⁻ ⋅ Ω₂, Z ⋅ Ω₂
        FΩ₃, F⁻Ω₃, ZΩ₃ = F⁺ ⋅ Ω₃, F⁻ ⋅ Ω₃, Z ⋅ Ω₃

        MPSV₁[1] = V[F⁻Ω₁, F⁻Ω₂, ZΩ₁]
        MPSV₁[2] = V[FΩ₁, F⁻Ω₃, ZΩ₂]

        for j in 3:ETL-n
            FΩ₁, FΩ₂, ZΩ₂ = FΩ₂, FΩ₃, ZΩ₃
            Ω₃ = MPSV₂[j+1]
            FΩ₃, F⁻Ω₃, ZΩ₃ = F⁺ ⋅ Ω₃, F⁻ ⋅ Ω₃, Z ⋅ Ω₃
            MPSV₁[j] = V[FΩ₁, F⁻Ω₃, ZΩ₂]
        end

        dc[n] = abs(F⁻Ω₁)
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F⁻ ⋅ MPSV₂[1])

    return dc
end

function epg_decay_curve!(dc::AbstractVector, work::EPGWork_ReIm_DualVector_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    epg_impulse_response!(dc, work, θ)

    # Scale impulse response by initial magnetization and take absolute value
    m₀ = sin(B1correctedflipangle(θ, 0)) # B1-corrected excitation angle
    @simd ivdep for n in eachindex(dc)
        dc[n] = abs(m₀ * dc[n])
    end

    return dc
end

function epg_impulse_response!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualVector_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    ETL = length(dc)
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

        dc[1] = Ω₁[1]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁

        for n in 2:ETL÷2
            F⁺, F⁻, Z = MPSV₂[1]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            F⁻Ω₂, FΩ₂, ZΩ₂ = muladd(-c, Z, x′ - y′), muladd(c, Z, x′ + y′), muladd(c′, y, d * Z)

            F⁺, F⁻, Z = MPSV₂[2]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            F⁻Ω₃, FΩ₃, ZΩ₃ = muladd(-c, Z, x′ - y′), muladd(c, Z, x′ + y′), muladd(c′, y, d * Z)

            MPSV₁[1] = V[F⁻Ω₂, F⁻Ω₃, ZΩ₂]
            dc[n] = F⁻Ω₂
            FΩ₁, FΩ₂, ZΩ₂ = FΩ₂, FΩ₃, ZΩ₃

            @simd ivdep for j in 2:n-1
                F⁺, F⁻, Z = MPSV₂[j+1]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a * x, b * y
                FΩ₃, F⁻Ω₃, ZΩ₃ = muladd(c, Z, x′ + y′), muladd(-c, Z, x′ - y′), muladd(c′, y, d * Z)
                MPSV₁[j] = V[FΩ₁, F⁻Ω₃, ZΩ₂]
                FΩ₁, FΩ₂, ZΩ₂ = FΩ₂, FΩ₃, ZΩ₃
            end

            MPSV₁[n], MPSV₁[n+1] = V[FΩ₁, zero(T), ZΩ₂], V[FΩ₂, zero(T), zero(T)]
            MPSV₁, MPSV₂ = MPSV₂, MPSV₁
        end

        for n in ETL÷2+1:ETL-1
            F⁺, F⁻, Z = MPSV₂[1]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            F⁻Ω₂, FΩ₂, ZΩ₂ = muladd(-c, Z, x′ - y′), muladd(c, Z, x′ + y′), muladd(c′, y, d * Z)

            F⁺, F⁻, Z = MPSV₂[2]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            F⁻Ω₃, FΩ₃, ZΩ₃ = muladd(-c, Z, x′ - y′), muladd(c, Z, x′ + y′), muladd(c′, y, d * Z)

            MPSV₁[1] = V[F⁻Ω₂, F⁻Ω₃, ZΩ₂]
            dc[n] = F⁻Ω₂
            FΩ₁, FΩ₂, ZΩ₂ = FΩ₂, FΩ₃, ZΩ₃

            @simd ivdep for j in 2:ETL-n
                F⁺, F⁻, Z = MPSV₂[j+1]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a * x, b * y
                FΩ₃, F⁻Ω₃, ZΩ₃ = muladd(c, Z, x′ + y′), muladd(-c, Z, x′ - y′), muladd(c′, y, d * Z)
                MPSV₁[j] = V[FΩ₁, F⁻Ω₃, ZΩ₂]
                FΩ₁, FΩ₂, ZΩ₂ = FΩ₂, FΩ₃, ZΩ₃
            end

            MPSV₁[ETL-n+1] = V[FΩ₁, zero(T), ZΩ₂]
            MPSV₁, MPSV₂ = MPSV₂, MPSV₁
        end

        F⁺, F⁻, Z = MPSV₂[1]
        x, y = F⁺ + F⁻, F⁺ - F⁻
        dc[ETL] = muladd(-c, Z, muladd(a, x, -b * y))
    end

    return dc
end

####
#### EPGWork_ReIm_DualFlat_Split_Dynamic
####

struct EPGWork_ReIm_DualFlat_Split_Dynamic{T, ETL, MPSVType <: AbstractVector{T}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end

function EPGWork_ReIm_DualFlat_Split_Dynamic(::Type{T}, ETL::Int) where {T}
    MPSV₁ = zeros(T, 3 * ETL)
    MPSV₂ = zeros(T, 3 * ETL)
    dc = zeros(T, ETL)
    return EPGWork_ReIm_DualFlat_Split_Dynamic(ETL, MPSV₁, MPSV₂, dc)
end

function epg_decay_curve!(dc::AbstractVector, work::EPGWork_ReIm_DualFlat_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    epg_impulse_response!(dc, work, θ)

    # Scale impulse response by initial magnetization and take absolute value
    m₀ = sin(B1correctedflipangle(θ, 0)) # B1-corrected excitation angle
    @simd ivdep for n in eachindex(dc)
        dc[n] = abs(m₀ * dc[n])
    end

    return dc
end

function epg_impulse_response!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualFlat_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    Base.require_one_based_indexing(dc)
    ETL = length(dc)

    (; MPSV₁, MPSV₂) = work
    @assert length(MPSV₁) == length(MPSV₂) == 3 * ETL "Dimension mismatch"

    α = θ.α
    TE, T2, T1 = echotime(θ), T2time(θ), T1time(θ)

    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sinα, cosα = sincos(α)
    E₂²half, E₁E₂, E₁² = (E₂ * E₂) / 2, E₁ * E₂, E₁ * E₁
    a, b, c, d = E₂²half, E₂²half * cosα, E₁E₂ * sinα, E₁² * cosα
    c′ = -c / 2
    Δy, Δz = ETL, 2 * ETL

    @inbounds begin
        dc[1] = a - b
        MPSV₁[1], MPSV₁[1+Δy], MPSV₁[1+Δz] = a - b, zero(T), c′
        MPSV₁[2], MPSV₁[2+Δy], MPSV₁[2+Δz] = a + b, zero(T), zero(T)
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁

        for n in 2:ETL÷2
            F⁺, F⁻, Z = MPSV₂[1], MPSV₂[1+Δy], MPSV₂[1+Δz]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            dc[n] = MPSV₁[1] = muladd(-c, Z, x′ - y′)
            MPSV₁[2] = muladd(c, Z, x′ + y′)
            MPSV₁[1+Δz] = muladd(c′, y, d * Z)

            @simd ivdep for j in 2:n-1
                F⁺, F⁻, Z = MPSV₂[j], MPSV₂[j+Δy], MPSV₂[j+Δz]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a * x, b * y
                MPSV₁[j+1] = muladd(c, Z, x′ + y′)
                MPSV₁[j-1+Δy] = muladd(-c, Z, x′ - y′)
                MPSV₁[j+Δz] = muladd(c′, y, d * Z)
            end

            F⁺, F⁻, Z = MPSV₂[n], MPSV₂[n+Δy], MPSV₂[n+Δz]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            MPSV₁[n+1] = muladd(c, Z, x′ + y′)
            MPSV₁[n-1+Δy] = muladd(-c, Z, x′ - y′)
            MPSV₁[n+Δz] = muladd(c′, y, d * Z)

            MPSV₁[n+Δy] = zero(T)
            MPSV₁[n+1+Δy] = zero(T)
            MPSV₁[n+1+Δz] = zero(T)

            MPSV₁, MPSV₂ = MPSV₂, MPSV₁
        end

        for n in ETL÷2+1:ETL-1
            F⁺, F⁻, Z = MPSV₂[1], MPSV₂[1+Δy], MPSV₂[1+Δz]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            dc[n] = MPSV₁[1] = muladd(-c, Z, x′ - y′)
            MPSV₁[2] = muladd(c, Z, x′ + y′)
            MPSV₁[1+Δz] = muladd(c′, y, d * Z)

            @simd ivdep for j in 2:ETL-n+1
                F⁺, F⁻, Z = MPSV₂[j], MPSV₂[j+Δy], MPSV₂[j+Δz]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a * x, b * y
                MPSV₁[j+1] = muladd(c, Z, x′ + y′)
                MPSV₁[j-1+Δy] = muladd(-c, Z, x′ - y′)
                MPSV₁[j+Δz] = muladd(c′, y, d * Z)
            end

            MPSV₁, MPSV₂ = MPSV₂, MPSV₁
        end

        F⁺, F⁻, Z = MPSV₂[1], MPSV₂[1+Δy], MPSV₂[1+Δz]
        x, y = F⁺ + F⁻, F⁺ - F⁻
        dc[ETL] = muladd(-c, Z, muladd(a, x, -b * y))
    end

    return dc
end

####
#### EPGWork_ReIm_DualTuple_Split_Dynamic
####

struct EPGWork_ReIm_DualTuple_Split_Dynamic{T, ETL, MPSVType <: AbstractVector{T}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::NTuple{3, MPSVType}
    MPSV₂::NTuple{3, MPSVType}
    dc::DCType
end

function EPGWork_ReIm_DualTuple_Split_Dynamic(::Type{T}, ETL::Int) where {T}
    MPSV₁ = (zeros(T, ETL), zeros(T, ETL), zeros(T, ETL))
    MPSV₂ = (zeros(T, ETL), zeros(T, ETL), zeros(T, ETL))
    dc = zeros(T, ETL)
    return EPGWork_ReIm_DualTuple_Split_Dynamic(ETL, MPSV₁, MPSV₂, dc)
end

function epg_decay_curve!(dc::AbstractVector, work::EPGWork_ReIm_DualTuple_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    epg_impulse_response!(dc, work, θ)

    # Scale impulse response by initial magnetization and take absolute value
    m₀ = sin(B1correctedflipangle(θ, 0)) # B1-corrected excitation angle
    @simd ivdep for n in eachindex(dc)
        dc[n] = abs(m₀ * dc[n])
    end

    return dc
end

function epg_impulse_response!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualTuple_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    Base.require_one_based_indexing(dc)
    ETL = length(dc)

    (; MPSV₁, MPSV₂) = work
    (MPSVx₁, MPSVy₁, MPSVz₁), (MPSVx₂, MPSVy₂, MPSVz₂) = MPSV₁, MPSV₂
    @assert length(MPSVx₁) == length(MPSVy₁) == length(MPSVz₁) == ETL "Dimension mismatch"
    @assert length(MPSVx₂) == length(MPSVy₂) == length(MPSVz₂) == ETL "Dimension mismatch"

    α = θ.α
    TE, T2, T1 = echotime(θ), T2time(θ), T1time(θ)

    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sinα, cosα = sincos(α)
    E₂²half, E₁E₂, E₁² = (E₂ * E₂) / 2, E₁ * E₂, E₁ * E₁
    a, b, c, d = E₂²half, E₂²half * cosα, E₁E₂ * sinα, E₁² * cosα
    c′ = -c / 2

    @inbounds begin
        dc[1] = a - b
        MPSVx₁[1], MPSVy₁[1], MPSVz₁[1] = a - b, zero(T), c′
        MPSVx₁[2], MPSVy₁[2], MPSVz₁[2] = a + b, zero(T), zero(T)
        (MPSVx₁, MPSVy₁, MPSVz₁), (MPSVx₂, MPSVy₂, MPSVz₂) = (MPSVx₂, MPSVy₂, MPSVz₂), (MPSVx₁, MPSVy₁, MPSVz₁)

        for n in 2:ETL÷2
            F⁺, F⁻, Z = MPSVx₂[1], MPSVy₂[1], MPSVz₂[1]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            dc[n] = MPSVx₁[1] = muladd(-c, Z, x′ - y′)
            MPSVx₁[2] = muladd(c, Z, x′ + y′)
            MPSVz₁[1] = muladd(c′, y, d * Z)

            @simd ivdep for j in 2:n-1
                F⁺, F⁻, Z = MPSVx₂[j], MPSVy₂[j], MPSVz₂[j]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a * x, b * y
                MPSVx₁[j+1] = muladd(c, Z, x′ + y′)
                MPSVy₁[j-1] = muladd(-c, Z, x′ - y′)
                MPSVz₁[j] = muladd(c′, y, d * Z)
            end

            F⁺, F⁻, Z = MPSVx₂[n], MPSVy₂[n], MPSVz₂[n]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            MPSVx₁[n+1] = muladd(c, Z, x′ + y′)
            MPSVy₁[n-1] = muladd(-c, Z, x′ - y′)
            MPSVz₁[n] = muladd(c′, y, d * Z)

            MPSVy₁[n] = zero(T)
            MPSVy₁[n+1] = zero(T)
            MPSVz₁[n+1] = zero(T)

            (MPSVx₁, MPSVy₁, MPSVz₁), (MPSVx₂, MPSVy₂, MPSVz₂) = (MPSVx₂, MPSVy₂, MPSVz₂), (MPSVx₁, MPSVy₁, MPSVz₁)
        end

        for n in ETL÷2+1:ETL-1
            F⁺, F⁻, Z = MPSVx₂[1], MPSVy₂[1], MPSVz₂[1]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            x′, y′ = a * x, b * y
            dc[n] = MPSVx₁[1] = muladd(-c, Z, x′ - y′)
            MPSVx₁[2] = muladd(c, Z, x′ + y′)
            MPSVz₁[1] = muladd(c′, y, d * Z)

            @simd ivdep for j in 2:ETL-n+1
                F⁺, F⁻, Z = MPSVx₂[j], MPSVy₂[j], MPSVz₂[j]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a * x, b * y
                MPSVx₁[j+1] = muladd(c, Z, x′ + y′)
                MPSVy₁[j-1] = muladd(-c, Z, x′ - y′)
                MPSVz₁[j] = muladd(c′, y, d * Z)
            end

            (MPSVx₁, MPSVy₁, MPSVz₁), (MPSVx₂, MPSVy₂, MPSVz₂) = (MPSVx₂, MPSVy₂, MPSVz₂), (MPSVx₁, MPSVy₁, MPSVz₁)
        end

        F⁺, F⁻, Z = MPSVx₂[1], MPSVy₂[1], MPSVz₂[1]
        x, y = F⁺ + F⁻, F⁺ - F⁻
        dc[ETL] = muladd(-c, Z, muladd(a, x, -b * y))
    end

    return dc
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
struct EPGWork_ReIm_Batched_Split_Dynamic{T, ETL, MType <: AbstractMatrix{T}, VType <: AbstractVector{T}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL   # echo train length
    X₁::MType  # F⁺ states of the current half-step (EPG_BATCH_WIDTH × ETL, lane-major)
    Y₁::MType  # F⁻ states of the current half-step
    Z₁::MType  # Z states of the current half-step
    X₂::MType  # F⁺ states of the previous half-step; the two triples are swapped each step
    Y₂::MType  # F⁻ states of the previous half-step
    Z₂::MType  # Z states of the previous half-step
    dcb::MType # lane-major impulse responses (EPG_BATCH_WIDTH × ETL)
    a::VType   # per-lane coefficients; see `epg_setup_lanes!`
    b::VType   # per-lane a·cos α
    c::VType   # per-lane E₁E₂·sin α
    c′::VType  # per-lane -c/2
    d::VType   # per-lane E₁²·cos α
    dc::DCType # single-curve decay curve output; see `epg_decay_curve!`
end

function EPGWork_ReIm_Batched_Split_Dynamic(::Type{T}, ETL::Int) where {T}
    W = EPG_BATCH_WIDTH
    X₁, Y₁, Z₁ = zeros(T, W, ETL), zeros(T, W, ETL), zeros(T, W, ETL)
    X₂, Y₂, Z₂ = zeros(T, W, ETL), zeros(T, W, ETL), zeros(T, W, ETL)
    dcb = zeros(T, W, ETL)
    a, b, c, c′, d = (zeros(T, W) for _ in 1:5)
    dc = zeros(T, ETL)
    return EPGWork_ReIm_Batched_Split_Dynamic(ETL, X₁, Y₁, Z₁, X₂, Y₂, Z₂, dcb, a, b, c, c′, d, dc)
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
    (; X₁, Y₁, Z₁, X₂, Y₂, Z₂, dcb) = work
    ETL = echotrainlength(work)
    W = EPG_BATCH_WIDTH

    (; a, b, c, c′, d) = work

    @inbounds begin
        @simd ivdep for l in 1:W
            dcb[l, 1] = a[l] - b[l]
            X₁[l, 1], Y₁[l, 1], Z₁[l, 1] = a[l] - b[l], zero(T), c′[l]
            X₁[l, 2], Y₁[l, 2], Z₁[l, 2] = a[l] + b[l], zero(T), zero(T)
        end
        (X₁, Y₁, Z₁), (X₂, Y₂, Z₂) = (X₂, Y₂, Z₂), (X₁, Y₁, Z₁)

        for n in 2:ETL÷2
            @simd ivdep for l in 1:W
                F⁺, F⁻, Z = X₂[l, 1], Y₂[l, 1], Z₂[l, 1]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a[l] * x, b[l] * y
                v = muladd(-c[l], Z, x′ - y′)
                dcb[l, n] = v
                X₁[l, 1] = v
                X₁[l, 2] = muladd(c[l], Z, x′ + y′)
                Z₁[l, 1] = muladd(c′[l], y, d[l] * Z)
            end

            for j in 2:n-1
                @simd ivdep for l in 1:W
                    F⁺, F⁻, Z = X₂[l, j], Y₂[l, j], Z₂[l, j]
                    x, y = F⁺ + F⁻, F⁺ - F⁻
                    x′, y′ = a[l] * x, b[l] * y
                    X₁[l, j+1] = muladd(c[l], Z, x′ + y′)
                    Y₁[l, j-1] = muladd(-c[l], Z, x′ - y′)
                    Z₁[l, j] = muladd(c′[l], y, d[l] * Z)
                end
            end

            @simd ivdep for l in 1:W
                F⁺, F⁻, Z = X₂[l, n], Y₂[l, n], Z₂[l, n]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a[l] * x, b[l] * y
                X₁[l, n+1] = muladd(c[l], Z, x′ + y′)
                Y₁[l, n-1] = muladd(-c[l], Z, x′ - y′)
                Z₁[l, n] = muladd(c′[l], y, d[l] * Z)
                Y₁[l, n] = zero(T)
                Y₁[l, n+1] = zero(T)
                Z₁[l, n+1] = zero(T)
            end

            (X₁, Y₁, Z₁), (X₂, Y₂, Z₂) = (X₂, Y₂, Z₂), (X₁, Y₁, Z₁)
        end

        for n in ETL÷2+1:ETL-1
            @simd ivdep for l in 1:W
                F⁺, F⁻, Z = X₂[l, 1], Y₂[l, 1], Z₂[l, 1]
                x, y = F⁺ + F⁻, F⁺ - F⁻
                x′, y′ = a[l] * x, b[l] * y
                v = muladd(-c[l], Z, x′ - y′)
                dcb[l, n] = v
                X₁[l, 1] = v
                X₁[l, 2] = muladd(c[l], Z, x′ + y′)
                Z₁[l, 1] = muladd(c′[l], y, d[l] * Z)
            end

            for j in 2:ETL-n+1
                @simd ivdep for l in 1:W
                    F⁺, F⁻, Z = X₂[l, j], Y₂[l, j], Z₂[l, j]
                    x, y = F⁺ + F⁻, F⁺ - F⁻
                    x′, y′ = a[l] * x, b[l] * y
                    X₁[l, j+1] = muladd(c[l], Z, x′ + y′)
                    Y₁[l, j-1] = muladd(-c[l], Z, x′ - y′)
                    Z₁[l, j] = muladd(c′[l], y, d[l] * Z)
                end
            end

            (X₁, Y₁, Z₁), (X₂, Y₂, Z₂) = (X₂, Y₂, Z₂), (X₁, Y₁, Z₁)
        end

        @simd ivdep for l in 1:W
            F⁺, F⁻, Z = X₂[l, 1], Y₂[l, 1], Z₂[l, 1]
            x, y = F⁺ + F⁻, F⁺ - F⁻
            dcb[l, ETL] = muladd(-c[l], Z, muladd(a[l], x, -b[l] * y))
        end
    end

    return work
end

function epg_decay_curve!(dc::AbstractVector, work::EPGWork_ReIm_Batched_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    ETL = echotrainlength(work)
    epg_setup_lanes!(work, θ, SA[T2time(θ)], 1, 1) # duplicate the single T2 across all lanes
    epg_impulse_response_batched!(work)

    # Scale impulse response by initial magnetization and take absolute value
    m₀ = sin(B1correctedflipangle(θ, 0)) # B1-corrected excitation angle
    (; dcb) = work
    @inbounds @simd ivdep for n in 1:ETL
        dc[n] = abs(m₀ * dcb[1, n])
    end

    return dc
end

function epg_decay_basis!(decay_basis::AbstractMatrix{T}, decay_curve_work::EPGWork_ReIm_Batched_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}, T2_times::AbstractVector) where {T}
    # Compute the NNLS basis over T2 space in lane-batched chunks
    ETL, nT2 = size(decay_basis)
    W = EPG_BATCH_WIDTH
    m₀ = sin(B1correctedflipangle(θ, 0)) # B1-corrected excitation angle
    (; dcb) = decay_curve_work
    @inbounds for j0 in 1:W:nT2
        epg_setup_lanes!(decay_curve_work, θ, T2_times, j0, nT2)
        epg_impulse_response_batched!(decay_curve_work)
        for l in 1:min(W, nT2-j0+1)
            @simd ivdep for n in 1:ETL
                decay_basis[n, j0+l-1] = abs(m₀ * dcb[l, n])
            end
        end
    end
    return decay_basis
end

####
#### EPGWork_ReIm_DualMVector_Split
####

struct EPGWork_ReIm_DualMVector_Split{T, ETL, MPSVType <: AbstractVector{SVector{3, T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualMVector_Split(::Type{T}, ::Val{ETL}) where {T, ETL}
    MPSV₁ = MVector{ETL, SVector{3, T}}(undef)
    MPSV₂ = MVector{ETL, SVector{3, T}}(undef)
    dc = MVector{ETL, T}(undef)
    return EPGWork_ReIm_DualMVector_Split(Val(ETL), MPSV₁, MPSV₂, dc)
end
EPGWork_ReIm_DualMVector_Split(::Type{T}, ETL::Int) where {T} = EPGWork_ReIm_DualMVector_Split(T, Val(ETL))

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualMVector_Split{T, Val{ETL}}, θ::EPGOptions{T, Val{ETL}}) where {T, ETL}
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
    a₁, b₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    a, b, c, d = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
    F⁺, F⁻, Z = V[a, b, c], V[b, a, -c], V[-c/2, c/2, d]

    # Initialize magnetization phase state vector (MPSV), pulling n=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Ω₁′ = V[b₁*m₀, 0, -c₁*m₀/2]
        dc[1] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        MPSV₁[2] = V[a₁*m₀, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in 2:ETL÷2
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Ω₁′ = V[F⁻⋅Ωⱼ, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ]
        dc[n] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:n-1
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⁺⋅Ωⱼ₋₁, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ]
        end
        MPSV₁[n] = V[F⁺⋅Ωⱼ, 0, Z⋅Ωⱼ₊₁]
        MPSV₁[n+1] = V[F⁺⋅Ωⱼ₊₁, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in ETL÷2+1:ETL-1
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Ω₁′ = V[F⁻⋅Ωⱼ, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ]
        dc[n] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:ETL-n
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⁺⋅Ωⱼ₋₁, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F⁻ ⋅ MPSV₂[1])

    return dc
end

####
#### EPGWork_ReIm_DualPaddedMVector_Vec_Split
####

#=
struct EPGWork_ReIm_DualPaddedMVector_Vec_Split{T, ETL, MPSVType <: AbstractVector{Vec{4, T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualPaddedMVector_Vec_Split(::Type{T}, ::Val{ETL}) where {T, ETL}
    MPSV₁ = MVector{ETL, Vec{4, T}}(undef)
    MPSV₂ = MVector{ETL, Vec{4, T}}(undef)
    dc    = MVector{ETL, T}(undef)
    return EPGWork_ReIm_DualPaddedMVector_Vec_Split(Val(ETL), MPSV₁, MPSV₂, dc)
end
EPGWork_ReIm_DualPaddedMVector_Vec_Split(::Type{T}, ETL::Int) where {T} = EPGWork_ReIm_DualPaddedMVector_Vec_Split(T, Val(ETL))

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualPaddedMVector_Vec_Split{T, Val{ETL}}, θ::EPGOptions{T, Val{ETL}}) where {T, ETL}
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
    a₁, b₁, c₁       = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    a, b, c, d   = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
    F⁺, F⁻, Z         = V((a, b, c, 0)), V((b, a, -c, 0)), V((-c / 2, c / 2, d, 0))

    # Initialize magnetization phase state vector (MPSV), pulling n=1 iteration out of loop
    @inbounds begin
        m₀           = sin½α₁ # since αₑₓ = ½α₁
        Ω₁′          = V((b₁ * m₀, 0, -c₁ * m₀ / 2, 0))
        dc[1]        = abs(Ω₁′[1])
        MPSV₁[1]     = Ω₁′
        MPSV₁[2]     = V((a₁ * m₀, 0, 0, 0))
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in 2:ETL÷2
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Ω₁′      = V((sum(F⁻ * Ωⱼ), sum(F⁻ * Ωⱼ₊₁), sum(Z * Ωⱼ), 0))
        dc[n]    = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:n-1
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V((sum(F⁺ * Ωⱼ₋₁), sum(F⁻ * Ωⱼ₊₁), sum(Z * Ωⱼ), 0))
        end
        MPSV₁[n]     = V((sum(F⁺ * Ωⱼ), 0, sum(Z * Ωⱼ₊₁), 0))
        MPSV₁[n+1]   = V((sum(F⁺ * Ωⱼ₊₁), 0, 0, 0))
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in ETL÷2+1:ETL-1
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Ω₁′      = V((sum(F⁻ * Ωⱼ), sum(F⁻ * Ωⱼ₊₁), sum(Z * Ωⱼ), 0))
        dc[n]    = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:ETL-n
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V((sum(F⁺ * Ωⱼ₋₁), sum(F⁻ * Ωⱼ₊₁), sum(Z * Ωⱼ), 0))
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(sum(F⁻ * MPSV₂[1]))

    return dc
end
=#

####
#### EPGWork_ReIm_DualPaddedVector_Split
####

struct EPGWork_ReIm_DualPaddedVector_Split{T, ETL, MPSVType <: AbstractVector{SVector{4, T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualPaddedVector_Split(::Type{T}, ETL::Int) where {T}
    MPSV₁ = zeros(SVector{4, T}, ETL)
    MPSV₂ = zeros(SVector{4, T}, ETL)
    dc = zeros(T, ETL)
    return EPGWork_ReIm_DualPaddedVector_Split(ETL, MPSV₁, MPSV₂, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualPaddedVector_Split{T}, θ::EPGOptions{T}) where {T}
    ETL = length(dc)

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
    a₁, b₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    a, b, c, d = E₂^2 * cos²½α, E₂^2 * sin²½α, E₁ * E₂ * sinα, E₁^2 * cosα
    F⁺, F⁻, Z = V[a, b, c, 0], V[b, a, -c, 0], V[-c/2, c/2, d, 0]

    # Initialize magnetization phase state vector (MPSV), pulling n=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Ω₁′ = V[b₁*m₀, 0, -c₁*m₀/2, 0]
        dc[1] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        MPSV₁[2] = V[a₁*m₀, 0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in 2:ETL÷2
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Ω₁′ = V[F⁻⋅Ωⱼ, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ, 0]
        dc[n] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:n-1
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⁺⋅Ωⱼ₋₁, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ, 0]
        end
        MPSV₁[n] = V[F⁺⋅Ωⱼ, 0, Z⋅Ωⱼ₊₁, 0]
        MPSV₁[n+1] = V[F⁺⋅Ωⱼ₊₁, 0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for n in ETL÷2+1:ETL-1
        Ωⱼ, Ωⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Ω₁′ = V[F⁻⋅Ωⱼ, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ, 0]
        dc[n] = abs(Ω₁′[1])
        MPSV₁[1] = Ω₁′
        @simd for j in 2:ETL-n
            Ωⱼ₋₁, Ωⱼ, Ωⱼ₊₁ = Ωⱼ, Ωⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⁺⋅Ωⱼ₋₁, F⁻⋅Ωⱼ₊₁, Z⋅Ωⱼ, 0]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F⁻ ⋅ MPSV₂[1])

    return dc
end

####
#### EPGWork_Vec
####

# Flip matrix and relaxation matrix steps are combined into one loop, and SIMD.jl `Vec` types are used instead of `Complex`.
# As this function is called many times during T2mapSEcorr, the micro-optimizations are worth the loss of code readability.
# See `EPGWork_Basic_Cplx` for a more readable, mathematically identicaly implementation.

#=
struct EPGWork_Vec{T, ETL, MPSVType <: AbstractVector{Vec{2, T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, ETL}
    ETL::ETL
    MPSV::MPSVType
    dc::DCType
end
function EPGWork_Vec(::Type{T}, ETL::Int) where {T}
    MSPV = zeros(Vec{2, T}, 3 * ETL)
    dc = zeros(T, ETL)
    return EPGWork_Vec(ETL, MSPV, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_Vec{T}, θ::EPGOptions{T}) where {T}
    ETL = length(dc)

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
        E2, E1  = exp(-TE / T2), exp(-TE / T1)
        E2_half = exp(-(TE / 2) / T2)
        m₀      = E2_half * sin(α₁ / 2) # initial population; since αₑₓ = α₁/2
        M1x     = m₀ * cos(α₁ / 2)^2   # M1x, M1y, M1z are elements resulting from first refocusing pulse applied to [m₀, 0, 0]
        M1y     = m₀ - M1x           # M1y = m₀ * sin(α₁/2)^2 = m₀ - m₀ * cos(α₁/2)^2 = m₀ - M1x
        M1z     = -m₀ * sin(α₁) / 2     # Note: this is the imaginary part
        dc[1]   = E2_half * abs(M1y)  # first echo amplitude

        # Apply first relaxation matrix iteration on non-zero states
        MPSV[1] = Vec((E2 * M1y, zero(T)))
        MPSV[2] = zero(Vec{2, T})
        MPSV[3] = Vec((zero(T), E1 * M1z))
        MPSV[4] = Vec((E2 * M1x, zero(T)))

        # Extract matrix elements + initialize temporaries
        a1, a2, a3, a4, a5 = sin(α), cos(α), sin(α / 2)^2, cos(α / 2)^2, sin(α) / 2 # independent elements of T2mat
        b1, b2, b3, b4, b5 = E2 * a1, E1 * a2, E2 * a3, E2 * a4, E1 * a5
        c1, c3, c4         = E2_half * a1, E2_half * a3, E2_half * a4
        b1F, b5F, c1F      = Vec((-b1, b1)), Vec((-b5, b5)), Vec((-c1, c1))
        Mz3                = MPSV[3]
    end

    @inbounds for n in 2:ETL-1
        ###########################
        # Unroll first flipmat/relaxmat iteration
        Vx, Vy  = MPSV[1], MPSV[2]
        c1z     = shufflevector(c1F * Mz3, Val((1, 0)))
        Mz2     = muladd(c3, Vx, muladd(c4, Vy, -c1z)) # flipmat: 2 -> dc
        Mz4     = muladd(b4, Vx, muladd(b3, Vy, E2_half * c1z)) # relaxmat: 1 -> 4, save in buffer
        dc[n]   = √(sum(Mz2 * Mz2)) # decay curve coefficient
        MPSV[1] = E2_half * Mz2 # relaxmat: 2 -> 1
        b5xy    = shufflevector(b5F * (Vx - Vy), Val((1, 0)))
        Mz3     = muladd(b2, Mz3, b5xy) # relaxmat: 3 -> 3, save in buffer

        ###########################
        # flipmat + relaxmat loop
        for j in 4:3:3*min(n - 1, ETL)
            Vx, Vy, Vz = MPSV[j], MPSV[j+1], MPSV[j+2]
            b1z        = shufflevector(b1F * Vz, Val((1, 0)))
            MPSV[j]    = Mz4 # relaxmat: assign forward, j -> j+3
            Mz4        = muladd(b4, Vx, muladd(b3, Vy, b1z))
            MPSV[j-2]  = muladd(b3, Vx, muladd(b4, Vy, -b1z)) # relaxmat: assign backwards, j+1 -> j+1-3
            b5xy       = shufflevector(b5F * (Vx - Vy), Val((1, 0)))
            MPSV[j+2]  = muladd(b2, Vz, b5xy) # relaxmat: j+2 -> j+2
        end

        ###########################
        # cleanup + zero next elements
        j         = 3i - 2
        Vx        = MPSV[j]
        MPSV[j]   = Mz4 # relaxmat: assign forward, j -> j+3
        MPSV[j-2] = b3 * Vx # relaxmat: assign backwards, j+1 -> j+1-3
        MPSV[j+2] = shufflevector(b5F * Vx, Val((1, 0))) # relaxmat: j+2 -> j+2
        MPSV[j+3] = b4 * Vx # relaxmat: assign forward, j -> j+3
        MPSV[j+1] = Vec((zero(T), zero(T))) # relaxmat: assign backwards, j+1 -> j+1-3
        MPSV[j+5] = Vec((zero(T), zero(T))) # relaxmat: j+2 -> j+2
    end

    ###########################
    # decay curve coefficient
    @inbounds begin
        c1z     = shufflevector(c1F * Mz3, Val((1, 0)))
        Mz2     = muladd(c3, MPSV[1], muladd(c4, MPSV[2], -c1z)) # last iteration of flipmat unrolled
        dc[end] = √(sum(Mz2 * Mz2))
    end

    return dc
end
=#

####
#### Exact cosine-series representation of the constant-flip-angle EPG basis
####

# Exact cosine-series form of the constant-flip-angle basis, used to evaluate A(α) at arbitrary α without re-running the EPG recursion.
# When every pulse is proportional to α, i.e. RefConAngle == 180, the signed unit-excitation impulse response of echo i is an even trigonometric polynomial of degree ≤ i in α.
# All α-dependence enters the recursion affinely via cos α and sin α, each echo adds at most one degree, and transverse↔longitudinal transfers carry sin α factors in pairs.
# Hence A_ij(α) = |sin(α/2) Σ_{k=0}^{i} Â_ijk cos(kα)| exactly, with coefficients from a cosine-Vandermonde solve on nTE+1 impulse-response samples at α_m = πm/nTE; the Vandermonde has DCT-I structure and condition number ≈ 1.5.
struct EPGCosineSeriesBasis{T}
    ETL::Int # echo train length, and also the cosine series degree
    nT2::Int # number of T2 times, i.e. basis columns
    coeffs::Vector{T} # triangular cosine coefficients stored in 4-row blocks; for each column j and echo block b, the k = 0..min(4b, ETL) runs have the block's 4 rows interleaved per k and zero-padded, so evaluation is one 4-wide broadcast-FMA per harmonic.
    c::Vector{T} # cosine feature buffer c[k+1] = cos(kα)
    sn::Vector{T} # k-weighted sine feature buffer sn[k+1] = k·sin(kα)
    cn::Vector{T} # k²-weighted cosine feature buffer cn[k+1] = k²·cos(kα)
end

function EPGCosineSeriesBasis(θ::EPGConstantFlipAngleOptions{T}, T2_times::AbstractVector) where {T}
    ETL, nT2, W = echotrainlength(θ), length(T2_times), EPG_BATCH_WIDTH
    N = ETL # cosine series degree; echo i has degree ≤ i ≤ N, so N+1 samples interpolate exactly
    work = EPGWork_ReIm_Batched_Split_Dynamic(T, ETL)
    S = zeros(T, ETL, nT2, N + 1)

    for m in 0:N
        θm = restructure(θ, (; α = T(π) * m / N))
        for j0 in 1:W:nT2
            epg_setup_lanes!(work, θm, T2_times, j0, nT2)
            epg_impulse_response_batched!(work)
            for l in 1:min(W, nT2-j0+1), i in 1:ETL
                S[i, j0+l-1, m+1] = work.dcb[l, i]
            end
        end
    end

    C = T[cospi(T(k * m) / N) for m in 0:N, k in 0:N]
    Â = reshape(permutedims(C \ permutedims(reshape(S, ETL * nT2, N + 1))), ETL, nT2, N + 1)
    nblk = cld(ETL, 4)
    coeffs = zeros(T, 4 * nT2 * sum(b -> min(4b, ETL) + 1, 1:nblk))
    p = 0
    for j in 1:nT2, b in 1:nblk, k in 0:min(4b, ETL), r in 1:4
        i = 4 * (b - 1) + r
        coeffs[p+=1] = (i <= ETL && k <= i) ? Â[i, j, k+1] : zero(T)
    end

    return EPGCosineSeriesBasis{T}(ETL, nT2, coeffs, zeros(T, N + 1), zeros(T, N + 1), zeros(T, N + 1))
end

# Sibling sharing the read-only coefficient tensor with fresh feature buffers.
# The coefficients depend only on the sequence parameters; all thread buffers share one tensor, while the mutable feature buffers stay thread-local.
EPGCosineSeriesBasis(decay_basis_work::EPGCosineSeriesBasis{T}) where {T} = EPGCosineSeriesBasis{T}(decay_basis_work.ETL, decay_basis_work.nT2, decay_basis_work.coeffs, similar(decay_basis_work.c), similar(decay_basis_work.sn), similar(decay_basis_work.cn))

# Fill c[k+1] = cos(kα), sn[k+1] = k·sin(kα), and cn[k+1] = k²·cos(kα) through coupled rotations by α, normalized after each rotation to suppress unit-circle drift.
# The rotation keeps sin²+cos² within roundoff of one, so the normalization is one Newton step for the inverse square root rather than a square root and a division.
function cosine_features!(decay_basis_work::EPGCosineSeriesBasis{T}, α::T) where {T}
    (; ETL, c, sn, cn) = decay_basis_work
    sinα, cosα = sin(α), cos(α)
    sinkα, coskα = zero(T), one(T)
    c[1], sn[1], cn[1] = coskα, zero(T), zero(T)

    @inbounds for k in 1:ETL
        sinkα, coskα = muladd(sinα, coskα, cosα * sinkα), muladd(-sinα, sinkα, cosα * coskα)
        scale = (3 - muladd(sinkα, sinkα, coskα * coskα)) / 2
        sinkα, coskα = scale * sinkα, scale * coskα
        c[k+1] = coskα
        sn[k+1] = k * sinkα
        cn[k+1] = k^2 * coskα
    end

    return decay_basis_work
end

function epg_decay_basis!(decay_basis::Matrix{T}, decay_basis_work::EPGCosineSeriesBasis{T}, α::T) where {T}
    (; coeffs, c) = decay_basis_work
    ETL, nT2 = size(decay_basis)
    cosine_features!(decay_basis_work, α)

    # Evaluate two columns per pass, reusing each cosine feature.
    colstride = 4 * sum(b -> min(4b, ETL) + 1, 1:cld(ETL, 4))
    s = sin(α / 2)
    p = 0
    @inbounds for j in 1:2:nT2
        j2 = min(j + 1, nT2)
        o1, o2, Δ = (j - 1) * ETL, (j2 - 1) * ETL, (j2 - j) * colstride

        for b in 1:cld(ETL, 4)
            i0 = 4 * (b - 1)
            L = min(i0 + 4, ETL) + 1
            a1 = a2 = a3 = a4 = e1 = e2 = e3 = e4 = zero(T)

            @simd ivdep for k in 1:L
                cₖ = c[k]
                q = p + 4 * (k - 1)
                a1 = muladd(coeffs[q+1], cₖ, a1)
                a2 = muladd(coeffs[q+2], cₖ, a2)
                a3 = muladd(coeffs[q+3], cₖ, a3)
                a4 = muladd(coeffs[q+4], cₖ, a4)
                e1 = muladd(coeffs[q+Δ+1], cₖ, e1)
                e2 = muladd(coeffs[q+Δ+2], cₖ, e2)
                e3 = muladd(coeffs[q+Δ+3], cₖ, e3)
                e4 = muladd(coeffs[q+Δ+4], cₖ, e4)
            end

            p += 4L
            if i0 + 4 <= ETL
                decay_basis[o1+i0+1], decay_basis[o1+i0+2], decay_basis[o1+i0+3], decay_basis[o1+i0+4] = abs(s * a1), abs(s * a2), abs(s * a3), abs(s * a4)
                decay_basis[o2+i0+1], decay_basis[o2+i0+2], decay_basis[o2+i0+3], decay_basis[o2+i0+4] = abs(s * e1), abs(s * e2), abs(s * e3), abs(s * e4)
            else
                nr = ETL - i0
                nr >= 1 && (decay_basis[o1+i0+1] = abs(s * a1); decay_basis[o2+i0+1] = abs(s * e1))
                nr >= 2 && (decay_basis[o1+i0+2] = abs(s * a2); decay_basis[o2+i0+2] = abs(s * e2))
                nr >= 3 && (decay_basis[o1+i0+3] = abs(s * a3); decay_basis[o2+i0+3] = abs(s * e3))
            end
        end

        p += Δ
    end

    return decay_basis
end

# Fill column j of A(α) into `Acol` and its first two α-derivatives into `dAcol` and `ddAcol` (all length ETL) from the cosine series.
# A_ij = |s·a_ij| with s = sin(α/2) and a_ij = Σₖ Â_ijk cos(kα), for α in radians. Writing σ = sign(a_ij), within a sign cell A_ij = σ·s·a_ij and so
#   ∂A_ij/∂α = σ·(s′·a + s·a′),   ∂²A_ij/∂α² = σ·(s″·a + 2·s′·a′ + s·a″),
# with s′ = cos(α/2)/2, s″ = −s/4, a′ = −Σₖ k·Â_ijk sin(kα), and a″ = −Σₖ k²·Â_ijk cos(kα).
# Requires `c`, `sn`, and `cn` current at α through `cosine_features!`.
# One column is O(ETL²/2), so building only the few support columns is far cheaper than a full basis.
function epg_decay_basis_∂α_col!(Acol::AbstractVector{T}, dAcol::AbstractVector{T}, ddAcol::AbstractVector{T}, decay_basis_work::EPGCosineSeriesBasis{T}, α::T, j::Int) where {T}
    (; ETL, coeffs, c, sn, cn) = decay_basis_work
    nblk = cld(ETL, 4)
    s, ds = sin(α / 2), cos(α / 2) / 2
    dds = -s / 4
    p = 4 * (j - 1) * sum(b -> min(4b, ETL) + 1, 1:nblk)

    @inbounds for b in 1:nblk
        i0 = 4 * (b - 1)
        L = min(i0 + 4, ETL) + 1

        a1 = a2 = a3 = a4 = zero(T)
        d1 = d2 = d3 = d4 = zero(T)
        e1 = e2 = e3 = e4 = zero(T)
        @simd ivdep for k in 1:L
            cₖ, sₖ, nₖ = c[k], sn[k], cn[k]
            q = p + 4 * (k - 1)
            a1 = muladd(coeffs[q+1], cₖ, a1)
            a2 = muladd(coeffs[q+2], cₖ, a2)
            a3 = muladd(coeffs[q+3], cₖ, a3)
            a4 = muladd(coeffs[q+4], cₖ, a4)
            d1 = muladd(coeffs[q+1], sₖ, d1)
            d2 = muladd(coeffs[q+2], sₖ, d2)
            d3 = muladd(coeffs[q+3], sₖ, d3)
            d4 = muladd(coeffs[q+4], sₖ, d4)
            e1 = muladd(coeffs[q+1], nₖ, e1)
            e2 = muladd(coeffs[q+2], nₖ, e2)
            e3 = muladd(coeffs[q+3], nₖ, e3)
            e4 = muladd(coeffs[q+4], nₖ, e4)
        end

        p += 4L
        for (r, a, d, e) in ((1, a1, d1, e1), (2, a2, d2, e2), (3, a3, d3, e3), (4, a4, d4, e4))
            i = i0 + r
            i <= ETL || break
            a′ = -d
            a″ = -e
            dv = muladd(ds, a, s * a′)
            ddv = muladd(dds, a, muladd(2 * ds, a′, s * a″))
            neg = a < 0
            Acol[i] = abs(s * a)
            dAcol[i] = ifelse(neg, -dv, dv)
            ddAcol[i] = ifelse(neg, -ddv, ddv)
        end
    end

    return Acol, dAcol, ddAcol
end
