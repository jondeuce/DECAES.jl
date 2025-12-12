####
#### Helper functions
####

# Element matrix for refocusing pulse with angle α (in degrees); acts on the magnetization state vector (MPSV)
@inline element_flipmat(α::T) where {T} = SA{Complex{T}}[
    cosd(α / 2)^2 sind(α / 2)^2 -im*sind(α);
    sind(α / 2)^2 cosd(α / 2)^2 im*sind(α);
    -im*sind(α)/2 im*sind(α)/2 cosd(α)]

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
@inline B1correction(θ::EPGOptions{T}) where {T} = T(θ.α / 180) # Multiplicative FA correction: A = α/180
@inline flipangle(θ::EPGOptions{T}, i::Int) where {T} = ifelse(i == 0, T(90), ifelse(i == 1, T(180), θ.β)) # Pulse sequence: 90, 180, β, β, ...
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
@inline B1correction(θ::EPGConstantFlipAngleOptions{T}) where {T} = T(θ.α / 180) # Multiplicative FA correction: A = α/180
@inline flipangle(θ::EPGConstantFlipAngleOptions{T}, i::Int) where {T} = ifelse(i == 0, T(90), T(180)) # Pulse sequence: 90, 180, 180, 180, ...
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
@inline B1correction(θ::EPGIncreasingFlipAnglesOptions{T}) where {T} = T(θ.α / 180) # Multiplicative FA correction: A = α/180
@inline flipangle(θ::EPGIncreasingFlipAnglesOptions{T}, i::Int) where {T} = ifelse(i == 0, T(90), ifelse(i == 1, θ.α1, ifelse(i == 2, θ.α2, T(180)))) # Pulse sequence: 90, α1, α2, 180, 180, ...
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
@inline Base.getindex(::SymbolVector{Fs}, i::Int) where {Fs} = Fs[i] #TODO uncomment

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

@inline EPGdecaycurve_work(θ::EPGParameterization{T}) where {T} = EPGdecaycurve_work(T, θ.ETL)
@inline EPGdecaycurve_work(::Type{T}, ETL::Int) where {T} = EPGWork_ReIm_DualVector_Split_Dynamic(T, ETL) # default for dynamic `ETL`
@inline EPGdecaycurve_work(::Type{T}, ::Val{ETL}) where {T, ETL} = EPGWork_ReIm_DualMVector_Split(T, Val(ETL)) # default for static `ETL`

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
  - `TE::Real`:   inter-echo time (Units: seconds)
  - `T2::Real`:   transverse relaxation time (Units: seconds)
  - `T1::Real`:   longitudinal relaxation time (Units: seconds)
  - `β::Real`:    value of Refocusing Pulse Control Angle (Units: degrees)

# Outputs

  - `decay_curve::AbstractVector`: normalized echo decay curve with length `ETL`
"""
@inline EPGdecaycurve(ETL, α::Real, TE::Real, T2::Real, T1::Real, β::Real) = EPGdecaycurve(EPGOptions((; ETL, α, TE, T2, T1, β)))
@inline EPGdecaycurve(θ::EPGParameterization{T}) where {T} = EPGdecaycurve!(EPGdecaycurve_work(θ), θ)
@inline EPGdecaycurve!(work::AbstractEPGWorkspace{T}, θ::EPGParameterization{T}) where {T} = EPGdecaycurve!(decaycurve(work), work, θ)
@inline EPGdecaycurve!(dc::AbstractVector{T}, work::AbstractEPGWorkspace{T}, θ::EPGParameterization{T}) where {T} = epg_decay_curve!(dc, work, θ)

####
#### Jacobian utilities (currently hardcoded for `EPGWork_ReIm_DualVector_Split_Dynamic`)
####

struct EPGWorkCacheDict{ETL} <: AbstractDict{DataType, Any}
    ETL::ETL
    dict::Dict{DataType, Any}
end
EPGWorkCacheDict(ETL) = EPGWorkCacheDict(ETL, Dict{DataType, Any}())

@inline Base.keys(caches::EPGWorkCacheDict) = Base.keys(caches.dict)
@inline Base.values(caches::EPGWorkCacheDict) = Base.values(caches.dict)
@inline Base.length(caches::EPGWorkCacheDict) = Base.length(caches.dict)
@inline Base.iterate(caches::EPGWorkCacheDict, state...) = Base.iterate(caches.dict, state...)

@inline function Base.getindex(caches::EPGWorkCacheDict{ETL}, ::Type{T}) where {ETL, T}
    R = EPGWork_ReIm_DualVector_Split_Dynamic{T, ETL, Vector{SVector{3, T}}, Vector{T}}
    get!(caches.dict, T) do
        return EPGWork_ReIm_DualVector_Split_Dynamic(T, caches.ETL)::R
    end::R
end

struct EPGFunctor{T, ETL, Fs, TC <: EPGWorkCacheDict{ETL}, Tθ <: EPGParameterization{T, ETL}}
    θ::Tθ
    fields::SymbolVector{Fs}
    caches::TC
end
EPGFunctor(θ::EPGParameterization, fields::SymbolVector) = EPGFunctor(θ, fields, EPGWorkCacheDict(echotrainlength(θ)))
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
    A = B1correction(θ)
    αₑₓ = A * 90
    α₁ = A * 180
    αᵢ = A * θ.β
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = SA{Complex{T}} # alias

    # Precompute compute element flip matrices and other intermediate variables
    E1, E2 = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    E = SA{T}[E2, E2, E1]
    R₁ = element_flipmat(α₁)
    Rᵢ = element_flipmat(αᵢ)

    # Initialize magnetization phase state vector (MPSV)
    @inbounds for j in 1:ETL
        MPSV[j] = V[0, 0, 0]
    end
    @inbounds MPSV[1] = V[sind(αₑₓ), 0, 0] # initial magnetization in F1 state

    @inbounds for i in 1:ETL
        # Relaxation for TE/2, followed by flip matrix
        R = i == 1 ? R₁ : Rᵢ
        for j in 1:ETL
            MPSV[j] = R * (E .* MPSV[j])
        end

        # Transition between phase states
        Mᵢ, Mᵢ₊₁ = MPSV[1], MPSV[2]
        MPSV[1] = V[Mᵢ[2], Mᵢ₊₁[2], Mᵢ[3]] # (F₁, F₁*, Z₁)⁺ = (F₁*, F₂*, Z₁)
        for j in 2:ETL-1
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV[j+1]
            MPSV[j] = V[Mᵢ₋₁[1], Mᵢ₊₁[2], Mᵢ[3]] # (Fᵢ, Fᵢ*, Zᵢ)⁺ = (Fᵢ₋₁, Fᵢ₊₁*, Zᵢ)
        end
        Mᵢ₋₁, Mᵢ = Mᵢ, Mᵢ₊₁
        MPSV[ETL] = V[Mᵢ₋₁[1], 0, Mᵢ[3]] # (Fₙ, Fₙ*, Zₙ)⁺ = (Fₙ₋₁, 0, Zₙ)

        # Relaxation for TE/2
        for j in 1:ETL
            MPSV[j] = E .* MPSV[j] # Relaxation for TE/2
        end
        dc[i] = abs(MPSV[1][1]) # first echo amplitude
    end

    return dc
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_Basic_Cplx{T}, θ::EPGIncreasingFlipAnglesOptions{T}) where {T}
    ETL = length(dc)

    # Unpack workspace
    (; MPSV) = work
    A = B1correction(θ)
    αₑₓ = A * flipangle(θ, 0)
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
    @inbounds MPSV[1] = V[sind(αₑₓ), 0, 0] # initial magnetization in F1 state

    @inbounds for i in 1:ETL
        # Relaxation for TE/2, followed by flip matrix
        R = element_flipmat(A * flipangle(θ, i))
        for j in 1:ETL
            MPSV[j] = R * (E .* MPSV[j])
        end

        # Transition between phase states
        Mᵢ, Mᵢ₊₁ = MPSV[1], MPSV[2]
        MPSV[1] = V[Mᵢ[2], Mᵢ₊₁[2], Mᵢ[3]] # (F₁, F₁*, Z₁)⁺ = (F₁*, F₂*, Z₁)
        for j in 2:ETL-1
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV[j+1]
            MPSV[j] = V[Mᵢ₋₁[1], Mᵢ₊₁[2], Mᵢ[3]] # (Fᵢ, Fᵢ*, Zᵢ)⁺ = (Fᵢ₋₁, Fᵢ₊₁*, Zᵢ)
        end
        Mᵢ₋₁, Mᵢ = Mᵢ, Mᵢ₊₁
        MPSV[ETL] = V[Mᵢ₋₁[1], 0, Mᵢ[3]] # (Fₙ, Fₙ*, Zₙ)⁺ = (Fₙ₋₁, 0, Zₙ)

        # Relaxation for TE/2
        for j in 1:ETL
            MPSV[j] = E .* MPSV[j] # Relaxation for TE/2
        end
        dc[i] = abs(MPSV[1][1]) # first echo amplitude
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
    A = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = SA{T} # alias

    # Precompute intermediate variables
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sin½α₁, cos½α₁ = sincos(α₁ / 2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2 * sin½α₁ * cos½α₁
    sinαᵢ, cosαᵢ = sincos(αᵢ)
    cos²½αᵢ = (1 + cosαᵢ) / 2
    sin²½αᵢ = 1 - cos²½αᵢ
    a₁, b₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    aᵢ, bᵢ, cᵢ, dᵢ = E₂^2 * cos²½αᵢ, E₂^2 * sin²½αᵢ, E₁ * E₂ * sinαᵢ, E₁^2 * cosαᵢ
    F, F̄, Z = V[aᵢ, bᵢ, cᵢ], V[bᵢ, aᵢ, -cᵢ], V[-cᵢ/2, cᵢ/2, dᵢ]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Mᵢ⁺ = V[b₁*m₀, 0, -c₁*m₀/2]
        dc[1] = abs(Mᵢ⁺[1])
        MPSV[1] = Mᵢ⁺
        MPSV[2] = V[a₁*m₀, 0, 0]
        MPSV[3] = V[0, 0, 0]
    end

    @inbounds for i in 2:ETL-1
        # j = 1, initialize and update `dc`
        Mᵢ, Mᵢ₊₁ = MPSV[1], MPSV[2]
        Mᵢ⁺ = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i] = abs(Mᵢ⁺[1])
        MPSV[1] = Mᵢ⁺

        # inner loop
        jup = min(i, ETL - i)
        for j in 2:jup
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV[j+1]
            MPSV[j] = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        end

        # cleanup for next iteration
        if i == jup
            Mᵢ₋₁ = Mᵢ
            MPSV[i+1] = V[F⋅Mᵢ₋₁, 0, 0]
            MPSV[i+2] = V[0, 0, 0]
        end
    end

    @inbounds dc[ETL] = abs(F̄ ⋅ MPSV[1])

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
    MPSV(i::Int) = Symbol(:MPSV, i)
    quote
        # Unpack workspace
        A  = B1correction(θ)
        α₁ = deg2rad(A * 180)
        αᵢ = deg2rad(A * θ.β)
        TE = echotime(θ)
        T2 = T2time(θ)
        T1 = T1time(θ)

        # Precompute intermediate variables
        V                = SA{$T} # alias
        E₁, E₂           = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
        sin½α₁, cos½α₁   = sincos(α₁ / 2)
        sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
        sinα₁            = 2 * sin½α₁ * cos½α₁
        sinαᵢ, cosαᵢ     = sincos(αᵢ)
        cos²½αᵢ          = (1 + cosαᵢ) / 2
        sin²½αᵢ          = 1 - cos²½αᵢ
        a₁, b₁, c₁       = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
        aᵢ, bᵢ, cᵢ, dᵢ   = E₂^2 * cos²½αᵢ, E₂^2 * sin²½αᵢ, E₁ * E₂ * sinαᵢ, E₁^2 * cosαᵢ
        F, F̄, Z         = V[aᵢ, bᵢ, cᵢ], V[bᵢ, aᵢ, -cᵢ], V[-cᵢ/2, cᵢ/2, dᵢ]

        # Initialize MPSV vector elements
        $([
            :($(MPSV(i)) = zero(SVector{3, $T}))
            for i in 1:ETL
        ]...)

        # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
        @inbounds begin
            m₀         = sin½α₁ # since αₑₓ = ½α₁
            Mᵢ⁺        = V[b₁*m₀, 0, -c₁*m₀/2]
            dc[1]      = abs(Mᵢ⁺[1])
            $(MPSV(1)) = Mᵢ⁺
            $(MPSV(2)) = V[a₁*m₀, 0, 0]
        end

        # Main loop
        $([
            quote
                # Initialize and update `dc` (j = 1)
                @inbounds begin
                    Mᵢ, Mᵢ₊₁   = $(MPSV(1)), $(MPSV(2))
                    Mᵢ⁺        = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
                    dc[$i]     = abs(Mᵢ⁺[1])
                    $(MPSV(1)) = Mᵢ⁺
                end

                # Inner loop
                $([
                    quote
                        (Mᵢ₋₁, Mᵢ, Mᵢ₊₁) = (Mᵢ, Mᵢ₊₁, $(MPSV(j + 1)))
                        $(MPSV(j))       = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
                    end
                    for j in 2:min(i, ETL - i)+1
                ]...)
            end
            for i in 2:ETL-1
        ]...)

        # Last echo
        @inbounds dc[$ETL] = abs(F̄ ⋅ $(MPSV(1)))

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
    A = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = SA{T} # alias

    # Precompute intermediate variables
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sin½α₁, cos½α₁ = sincos(α₁ / 2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2 * sin½α₁ * cos½α₁
    sinαᵢ, cosαᵢ = sincos(αᵢ)
    cos²½αᵢ = (1 + cosαᵢ) / 2
    sin²½αᵢ = 1 - cos²½αᵢ
    a₁, b₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    aᵢ, bᵢ, cᵢ, dᵢ = E₂^2 * cos²½αᵢ, E₂^2 * sin²½αᵢ, E₁ * E₂ * sinαᵢ, E₁^2 * cosαᵢ
    F, F̄, Z = V[aᵢ, bᵢ, cᵢ], V[bᵢ, aᵢ, -cᵢ], V[-cᵢ/2, cᵢ/2, dᵢ]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Mᵢ⁺ = V[b₁*m₀, 0, -c₁*m₀/2]
        dc[1] = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        MPSV₁[2] = V[a₁*m₀, 0, 0]
        MPSV₁[3] = V[0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 2:ETL-1
        # j = 1, initialize and update `dc`
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2]
        Mᵢ⁺ = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i] = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺

        # inner loop
        jup = min(i, ETL - i)
        @simd for j in 2:jup
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        end

        # cleanup for next iteration
        if i == jup
            Mᵢ₋₁ = Mᵢ
            MPSV₁[i+1] = V[F⋅Mᵢ₋₁, 0, 0]
            MPSV₁[i+2] = V[0, 0, 0]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F̄ ⋅ MPSV₂[1])

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
    A = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = SA{T} # alias

    # Precompute intermediate variables
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sin½α₁, cos½α₁ = sincos(α₁ / 2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2 * sin½α₁ * cos½α₁
    sinαᵢ, cosαᵢ = sincos(αᵢ)
    cos²½αᵢ = (1 + cosαᵢ) / 2
    sin²½αᵢ = 1 - cos²½αᵢ
    a₁, b₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    aᵢ, bᵢ, cᵢ, dᵢ = E₂^2 * cos²½αᵢ, E₂^2 * sin²½αᵢ, E₁ * E₂ * sinαᵢ, E₁^2 * cosαᵢ
    F, F̄, Z = V[aᵢ, bᵢ, cᵢ], V[bᵢ, aᵢ, -cᵢ], V[-cᵢ/2, cᵢ/2, dᵢ]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Mᵢ⁺ = V[b₁*m₀, 0, -c₁*m₀/2]
        dc[1] = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        MPSV₁[2] = V[a₁*m₀, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 2:ETL÷2
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺ = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i] = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:i-1
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        end
        MPSV₁[i] = V[F⋅Mᵢ, 0, Z⋅Mᵢ₊₁]
        MPSV₁[i+1] = V[F⋅Mᵢ₊₁, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in ETL÷2+1:ETL-1
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺ = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i] = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:ETL-i
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F̄ ⋅ MPSV₂[1])

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

    A = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)

    # Precompute intermediate variables
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sin½α₁, cos½α₁ = sincos(α₁ / 2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2 * sin½α₁ * cos½α₁
    sinαᵢ, cosαᵢ = sincos(αᵢ)
    cos²½αᵢ = (1 + cosαᵢ) / 2
    sin²½αᵢ = 1 - cos²½αᵢ
    a₁, b₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    aᵢ, bᵢ, cᵢ, dᵢ = E₂^2 * cos²½αᵢ, E₂^2 * sin²½αᵢ, E₁ * E₂ * sinαᵢ, E₁^2 * cosαᵢ
    F, F̄, Z = V[aᵢ, bᵢ, cᵢ], V[bᵢ, aᵢ, -cᵢ], V[-cᵢ/2, cᵢ/2, dᵢ]

    @inbounds begin
        # i = 1 iteration
        # Initialize magnetization phase state vector (MPSV)
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        M₀ = V[b₁*m₀, 0, -c₁*m₀/2]
        MPSV₁[1] = M₀
        MPSV₁[2] = V[a₁*m₀, 0, 0]

        dc[1] = abs(M₀[1])
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁

        # i = 2 iteration
        M₀, M₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        FM₀, F̄M₀, ZM₀ = F ⋅ M₀, F̄ ⋅ M₀, Z ⋅ M₀
        FM₁, F̄M₁, ZM₁ = F ⋅ M₁, F̄ ⋅ M₁, Z ⋅ M₁

        MPSV₁[1] = V[F̄M₀, F̄M₁, ZM₀]
        MPSV₁[2] = V[FM₀, 0, ZM₁]
        MPSV₁[3] = V[FM₁, 0, 0]

        dc[2] = abs(F̄M₀)
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 3:ETL÷2
        M₀, M₁, M₂ = MPSV₂[1], MPSV₂[2], MPSV₂[3] # j = 1, initialize and update `dc`
        FM₀, F̄M₀, ZM₀ = F ⋅ M₀, F̄ ⋅ M₀, Z ⋅ M₀
        FM₁, F̄M₁, ZM₁ = F ⋅ M₁, F̄ ⋅ M₁, Z ⋅ M₁
        FM₂, F̄M₂, ZM₂ = F ⋅ M₂, F̄ ⋅ M₂, Z ⋅ M₂

        MPSV₁[1] = V[F̄M₀, F̄M₁, ZM₀]
        MPSV₁[2] = V[FM₀, F̄M₂, ZM₁]

        for j in 3:i-1
            FM₀, FM₁, ZM₁ = FM₁, FM₂, ZM₂
            M₂ = MPSV₂[j+1]
            FM₂, F̄M₂, ZM₂ = F ⋅ M₂, F̄ ⋅ M₂, Z ⋅ M₂
            MPSV₁[j] = V[FM₀, F̄M₂, ZM₁]
        end

        MPSV₁[i] = V[FM₁, 0, ZM₂]
        MPSV₁[i+1] = V[FM₂, 0, 0]

        dc[i] = abs(F̄M₀)
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in ETL÷2+1:ETL-1
        M₀, M₁, M₂ = MPSV₂[1], MPSV₂[2], MPSV₂[3] # j = 1, initialize and update `dc`
        FM₀, F̄M₀, ZM₀ = F ⋅ M₀, F̄ ⋅ M₀, Z ⋅ M₀
        FM₁, F̄M₁, ZM₁ = F ⋅ M₁, F̄ ⋅ M₁, Z ⋅ M₁
        FM₂, F̄M₂, ZM₂ = F ⋅ M₂, F̄ ⋅ M₂, Z ⋅ M₂

        MPSV₁[1] = V[F̄M₀, F̄M₁, ZM₀]
        MPSV₁[2] = V[FM₀, F̄M₂, ZM₁]

        for j in 3:ETL-i
            FM₀, FM₁, ZM₁ = FM₁, FM₂, ZM₂
            M₂ = MPSV₂[j+1]
            FM₂, F̄M₂, ZM₂ = F ⋅ M₂, F̄ ⋅ M₂, Z ⋅ M₂
            MPSV₁[j] = V[FM₀, F̄M₂, ZM₁]
        end

        dc[i] = abs(F̄M₀)
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F̄ ⋅ MPSV₂[1])

    return dc
end

function epg_decay_curve!(dc::AbstractVector, work::EPGWork_ReIm_DualVector_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    epg_impulse_response!(dc, work, θ)

    # Scale impulse response by initial magnetization and take absolute value
    m₀ = sind(θ.α / 2)
    @simd ivdep for i in eachindex(dc)
        dc[i] = abs(m₀ * dc[i])
    end

    return dc
end

function epg_impulse_response!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualVector_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    ETL = length(dc)
    (; MPSV₁, MPSV₂) = work

    α = deg2rad(θ.α)
    TE, T2, T1 = echotime(θ), T2time(θ), T1time(θ)
    V = SA{T} # alias

    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sinα, cosα = sincos(α)
    E₂²half, E₁E₂, E₁² = (E₂ * E₂) / 2, E₁ * E₂, E₁ * E₁
    a, b, c, d = E₂²half, E₂²half * cosα, E₁E₂ * sinα, E₁² * cosα
    c′ = -c / 2

    @inbounds begin
        M₀ = V[a-b, zero(T), c′]
        MPSV₁[1] = M₀
        MPSV₁[2] = V[a+b, zero(T), zero(T)]

        dc[1] = M₀[1]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁

        for i in 2:ETL÷2
            F, F̄, Z = MPSV₂[1]
            C, S = F + F̄, F - F̄
            C′, S′ = a * C, b * S
            F̄M₁, FM₁, ZM₁ = muladd(-c, Z, C′ - S′), muladd(c, Z, C′ + S′), muladd(c′, S, d * Z)

            F, F̄, Z = MPSV₂[2]
            C, S = F + F̄, F - F̄
            C′, S′ = a * C, b * S
            F̄M₂, FM₂, ZM₂ = muladd(-c, Z, C′ - S′), muladd(c, Z, C′ + S′), muladd(c′, S, d * Z)

            MPSV₁[1] = V[F̄M₁, F̄M₂, ZM₁]
            dc[i] = F̄M₁
            FM₀, FM₁, ZM₁ = FM₁, FM₂, ZM₂

            @simd ivdep for k in 2:i-1
                F, F̄, Z = MPSV₂[k+1]
                C, S = F + F̄, F - F̄
                C′, S′ = a * C, b * S
                FM₂, F̄M₂, ZM₂ = muladd(c, Z, C′ + S′), muladd(-c, Z, C′ - S′), muladd(c′, S, d * Z)
                MPSV₁[k] = V[FM₀, F̄M₂, ZM₁]
                FM₀, FM₁, ZM₁ = FM₁, FM₂, ZM₂
            end

            MPSV₁[i], MPSV₁[i+1] = V[FM₀, zero(T), ZM₁], V[FM₁, zero(T), zero(T)]
            MPSV₁, MPSV₂ = MPSV₂, MPSV₁
        end

        for i in ETL÷2+1:ETL-1
            F, F̄, Z = MPSV₂[1]
            C, S = F + F̄, F - F̄
            C′, S′ = a * C, b * S
            F̄M₁, FM₁, ZM₁ = muladd(-c, Z, C′ - S′), muladd(c, Z, C′ + S′), muladd(c′, S, d * Z)

            F, F̄, Z = MPSV₂[2]
            C, S = F + F̄, F - F̄
            C′, S′ = a * C, b * S
            F̄M₂, FM₂, ZM₂ = muladd(-c, Z, C′ - S′), muladd(c, Z, C′ + S′), muladd(c′, S, d * Z)

            MPSV₁[1] = V[F̄M₁, F̄M₂, ZM₁]
            dc[i] = F̄M₁
            FM₀, FM₁, ZM₁ = FM₁, FM₂, ZM₂

            @simd ivdep for k in 2:ETL-i
                F, F̄, Z = MPSV₂[k+1]
                C, S = F + F̄, F - F̄
                C′, S′ = a * C, b * S
                FM₂, F̄M₂, ZM₂ = muladd(c, Z, C′ + S′), muladd(-c, Z, C′ - S′), muladd(c′, S, d * Z)
                MPSV₁[k] = V[FM₀, F̄M₂, ZM₁]
                FM₀, FM₁, ZM₁ = FM₁, FM₂, ZM₂
            end

            MPSV₁[ETL-i+1] = V[FM₀, zero(T), ZM₁]
            MPSV₁, MPSV₂ = MPSV₂, MPSV₁
        end

        F, F̄, Z = MPSV₂[1]
        C, S = F + F̄, F - F̄
        dc[ETL] = muladd(-c, Z, muladd(a, C, -b * S))
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
    m₀ = sind(θ.α / 2)
    @simd ivdep for i in eachindex(dc)
        dc[i] = abs(m₀ * dc[i])
    end

    return dc
end

function epg_impulse_response!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualFlat_Split_Dynamic{T}, θ::EPGConstantFlipAngleOptions{T}) where {T}
    Base.require_one_based_indexing(dc)
    ETL = length(dc)

    (; MPSV₁, MPSV₂) = work
    @assert length(MPSV₁) == length(MPSV₂) == 3 * ETL "Dimension mismatch"

    α = deg2rad(θ.α)
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

        for i in 2:ETL÷2
            F, F̄, Z = MPSV₂[1], MPSV₂[1+Δy], MPSV₂[1+Δz]
            C, S = F + F̄, F - F̄
            C′, S′ = a * C, b * S
            dc[i] = MPSV₁[1] = muladd(-c, Z, C′ - S′)
            MPSV₁[2] = muladd(c, Z, C′ + S′)
            MPSV₁[1+Δz] = muladd(c′, S, d * Z)

            @simd ivdep for k in 2:i-1
                F, F̄, Z = MPSV₂[k], MPSV₂[k+Δy], MPSV₂[k+Δz]
                C, S = F + F̄, F - F̄
                C′, S′ = a * C, b * S
                MPSV₁[k+1] = muladd(c, Z, C′ + S′)
                MPSV₁[k-1+Δy] = muladd(-c, Z, C′ - S′)
                MPSV₁[k+Δz] = muladd(c′, S, d * Z)
            end

            F, F̄, Z = MPSV₂[i], MPSV₂[i+Δy], MPSV₂[i+Δz]
            C, S = F + F̄, F - F̄
            C′, S′ = a * C, b * S
            MPSV₁[i+1] = muladd(c, Z, C′ + S′)
            MPSV₁[i-1+Δy] = muladd(-c, Z, C′ - S′)
            MPSV₁[i+Δz] = muladd(c′, S, d * Z)

            MPSV₁[i+Δy] = zero(T)
            MPSV₁[i+1+Δy] = zero(T)
            MPSV₁[i+1+Δz] = zero(T)

            MPSV₁, MPSV₂ = MPSV₂, MPSV₁
        end

        for i in ETL÷2+1:ETL-1
            F, F̄, Z = MPSV₂[1], MPSV₂[1+Δy], MPSV₂[1+Δz]
            C, S = F + F̄, F - F̄
            C′, S′ = a * C, b * S
            dc[i] = MPSV₁[1] = muladd(-c, Z, C′ - S′)
            MPSV₁[2] = muladd(c, Z, C′ + S′)
            MPSV₁[1+Δz] = muladd(c′, S, d * Z)

            @simd ivdep for k in 2:ETL-i+1
                F, F̄, Z = MPSV₂[k], MPSV₂[k+Δy], MPSV₂[k+Δz]
                C, S = F + F̄, F - F̄
                C′, S′ = a * C, b * S
                MPSV₁[k+1] = muladd(c, Z, C′ + S′)
                MPSV₁[k-1+Δy] = muladd(-c, Z, C′ - S′)
                MPSV₁[k+Δz] = muladd(c′, S, d * Z)
            end

            MPSV₁, MPSV₂ = MPSV₂, MPSV₁
        end

        F, F̄, Z = MPSV₂[1], MPSV₂[1+Δy], MPSV₂[1+Δz]
        C, S = F + F̄, F - F̄
        dc[ETL] = muladd(-c, Z, muladd(a, C, -b * S))
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
    m₀ = sind(θ.α / 2)
    @simd ivdep for i in eachindex(dc)
        dc[i] = abs(m₀ * dc[i])
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

    α = deg2rad(θ.α)
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

        for i in 2:ETL÷2
            F, F̄, Z = MPSVx₂[1], MPSVy₂[1], MPSVz₂[1]
            C, S = F + F̄, F - F̄
            C′, S′ = a * C, b * S
            dc[i] = MPSVx₁[1] = muladd(-c, Z, C′ - S′)
            MPSVx₁[2] = muladd(c, Z, C′ + S′)
            MPSVz₁[1] = muladd(c′, S, d * Z)

            @simd ivdep for k in 2:i-1
                F, F̄, Z = MPSVx₂[k], MPSVy₂[k], MPSVz₂[k]
                C, S = F + F̄, F - F̄
                C′, S′ = a * C, b * S
                MPSVx₁[k+1] = muladd(c, Z, C′ + S′)
                MPSVy₁[k-1] = muladd(-c, Z, C′ - S′)
                MPSVz₁[k] = muladd(c′, S, d * Z)
            end

            F, F̄, Z = MPSVx₂[i], MPSVy₂[i], MPSVz₂[i]
            C, S = F + F̄, F - F̄
            C′, S′ = a * C, b * S
            MPSVx₁[i+1] = muladd(c, Z, C′ + S′)
            MPSVy₁[i-1] = muladd(-c, Z, C′ - S′)
            MPSVz₁[i] = muladd(c′, S, d * Z)

            MPSVy₁[i] = zero(T)
            MPSVy₁[i+1] = zero(T)
            MPSVz₁[i+1] = zero(T)

            (MPSVx₁, MPSVy₁, MPSVz₁), (MPSVx₂, MPSVy₂, MPSVz₂) = (MPSVx₂, MPSVy₂, MPSVz₂), (MPSVx₁, MPSVy₁, MPSVz₁)
        end

        for i in ETL÷2+1:ETL-1
            F, F̄, Z = MPSVx₂[1], MPSVy₂[1], MPSVz₂[1]
            C, S = F + F̄, F - F̄
            C′, S′ = a * C, b * S
            dc[i] = MPSVx₁[1] = muladd(-c, Z, C′ - S′)
            MPSVx₁[2] = muladd(c, Z, C′ + S′)
            MPSVz₁[1] = muladd(c′, S, d * Z)

            @simd ivdep for k in 2:ETL-i+1
                F, F̄, Z = MPSVx₂[k], MPSVy₂[k], MPSVz₂[k]
                C, S = F + F̄, F - F̄
                C′, S′ = a * C, b * S
                MPSVx₁[k+1] = muladd(c, Z, C′ + S′)
                MPSVy₁[k-1] = muladd(-c, Z, C′ - S′)
                MPSVz₁[k] = muladd(c′, S, d * Z)
            end

            (MPSVx₁, MPSVy₁, MPSVz₁), (MPSVx₂, MPSVy₂, MPSVz₂) = (MPSVx₂, MPSVy₂, MPSVz₂), (MPSVx₁, MPSVy₁, MPSVz₁)
        end

        F, F̄, Z = MPSVx₂[1], MPSVy₂[1], MPSVz₂[1]
        C, S = F + F̄, F - F̄
        dc[ETL] = muladd(-c, Z, muladd(a, C, -b * S))
    end

    return dc
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
    A = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = SA{T} # alias

    # Precompute intermediate variables
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sin½α₁, cos½α₁ = sincos(α₁ / 2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2 * sin½α₁ * cos½α₁
    sinαᵢ, cosαᵢ = sincos(αᵢ)
    cos²½αᵢ = (1 + cosαᵢ) / 2
    sin²½αᵢ = 1 - cos²½αᵢ
    a₁, b₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    aᵢ, bᵢ, cᵢ, dᵢ = E₂^2 * cos²½αᵢ, E₂^2 * sin²½αᵢ, E₁ * E₂ * sinαᵢ, E₁^2 * cosαᵢ
    F, F̄, Z = V[aᵢ, bᵢ, cᵢ], V[bᵢ, aᵢ, -cᵢ], V[-cᵢ/2, cᵢ/2, dᵢ]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Mᵢ⁺ = V[b₁*m₀, 0, -c₁*m₀/2]
        dc[1] = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        MPSV₁[2] = V[a₁*m₀, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 2:ETL÷2
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺ = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i] = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:i-1
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        end
        MPSV₁[i] = V[F⋅Mᵢ, 0, Z⋅Mᵢ₊₁]
        MPSV₁[i+1] = V[F⋅Mᵢ₊₁, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in ETL÷2+1:ETL-1
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺ = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i] = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:ETL-i
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F̄ ⋅ MPSV₂[1])

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
    A = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = Vec{4, T} # alias

    # Precompute intermediate variables
    E₁, E₂           = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sin½α₁, cos½α₁   = sincos(α₁ / 2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁            = 2 * sin½α₁ * cos½α₁
    sinαᵢ, cosαᵢ     = sincos(αᵢ)
    cos²½αᵢ          = (1 + cosαᵢ) / 2
    sin²½αᵢ          = 1 - cos²½αᵢ
    a₁, b₁, c₁       = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    aᵢ, bᵢ, cᵢ, dᵢ   = E₂^2 * cos²½αᵢ, E₂^2 * sin²½αᵢ, E₁ * E₂ * sinαᵢ, E₁^2 * cosαᵢ
    F, F̄, Z         = V((aᵢ, bᵢ, cᵢ, 0)), V((bᵢ, aᵢ, -cᵢ, 0)), V((-cᵢ / 2, cᵢ / 2, dᵢ, 0))

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀           = sin½α₁ # since αₑₓ = ½α₁
        Mᵢ⁺          = V((b₁ * m₀, 0, -c₁ * m₀ / 2, 0))
        dc[1]        = abs(Mᵢ⁺[1])
        MPSV₁[1]     = Mᵢ⁺
        MPSV₁[2]     = V((a₁ * m₀, 0, 0, 0))
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 2:ETL÷2
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺      = V((sum(F̄ * Mᵢ), sum(F̄ * Mᵢ₊₁), sum(Z * Mᵢ), 0))
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:i-1
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V((sum(F * Mᵢ₋₁), sum(F̄ * Mᵢ₊₁), sum(Z * Mᵢ), 0))
        end
        MPSV₁[i]     = V((sum(F * Mᵢ), 0, sum(Z * Mᵢ₊₁), 0))
        MPSV₁[i+1]   = V((sum(F * Mᵢ₊₁), 0, 0, 0))
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in ETL÷2+1:ETL-1
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺      = V((sum(F̄ * Mᵢ), sum(F̄ * Mᵢ₊₁), sum(Z * Mᵢ), 0))
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:ETL-i
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V((sum(F * Mᵢ₋₁), sum(F̄ * Mᵢ₊₁), sum(Z * Mᵢ), 0))
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(sum(F̄ * MPSV₂[1]))

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
    A = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V = SA{T} # alias

    # Precompute intermediate variables
    E₁, E₂ = exp(-(TE / 2) / T1), exp(-(TE / 2) / T2)
    sin½α₁, cos½α₁ = sincos(α₁ / 2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2 * sin½α₁ * cos½α₁
    sinαᵢ, cosαᵢ = sincos(αᵢ)
    cos²½αᵢ = (1 + cosαᵢ) / 2
    sin²½αᵢ = 1 - cos²½αᵢ
    a₁, b₁, c₁ = E₂^2 * cos²½α₁, E₂^2 * sin²½α₁, E₁ * E₂ * sinα₁
    aᵢ, bᵢ, cᵢ, dᵢ = E₂^2 * cos²½αᵢ, E₂^2 * sin²½αᵢ, E₁ * E₂ * sinαᵢ, E₁^2 * cosαᵢ
    F, F̄, Z = V[aᵢ, bᵢ, cᵢ, 0], V[bᵢ, aᵢ, -cᵢ, 0], V[-cᵢ/2, cᵢ/2, dᵢ, 0]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁ # since αₑₓ = ½α₁
        Mᵢ⁺ = V[b₁*m₀, 0, -c₁*m₀/2, 0]
        dc[1] = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        MPSV₁[2] = V[a₁*m₀, 0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 2:ETL÷2
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺ = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ, 0]
        dc[i] = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:i-1
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ, 0]
        end
        MPSV₁[i] = V[F⋅Mᵢ, 0, Z⋅Mᵢ₊₁, 0]
        MPSV₁[i+1] = V[F⋅Mᵢ₊₁, 0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in ETL÷2+1:ETL-1
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺ = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ, 0]
        dc[i] = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:ETL-i
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ, 0]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F̄ ⋅ MPSV₂[1])

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
    A = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
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
        a1, a2, a3, a4, a5 = sin(αᵢ), cos(αᵢ), sin(αᵢ / 2)^2, cos(αᵢ / 2)^2, sin(αᵢ) / 2 # independent elements of T2mat
        b1, b2, b3, b4, b5 = E2 * a1, E1 * a2, E2 * a3, E2 * a4, E1 * a5
        c1, c3, c4         = E2_half * a1, E2_half * a3, E2_half * a4
        b1F, b5F, c1F      = Vec((-b1, b1)), Vec((-b5, b5)), Vec((-c1, c1))
        Mz3                = MPSV[3]
    end

    @inbounds for i in 2:ETL-1
        ###########################
        # Unroll first flipmat/relaxmat iteration
        Vx, Vy  = MPSV[1], MPSV[2]
        c1z     = shufflevector(c1F * Mz3, Val((1, 0)))
        Mz2     = muladd(c3, Vx, muladd(c4, Vy, -c1z)) # flipmat: 2 -> dc
        Mz4     = muladd(b4, Vx, muladd(b3, Vy, E2_half * c1z)) # relaxmat: 1 -> 4, save in buffer
        dc[i]   = √(sum(Mz2 * Mz2)) # decay curve coefficient
        MPSV[1] = E2_half * Mz2 # relaxmat: 2 -> 1
        b5xy    = shufflevector(b5F * (Vx - Vy), Val((1, 0)))
        Mz3     = muladd(b2, Mz3, b5xy) # relaxmat: 3 -> 3, save in buffer

        ###########################
        # flipmat + relaxmat loop
        for j in 4:3:3*min(i - 1, ETL)
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
