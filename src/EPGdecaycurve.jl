####
#### Helper functions
####

# Element matrix for refocusing pulse with angle α (in degrees); acts on the magnetization state vector (MPSV)
@inline element_flipmat(α::T) where {T} = SA{Complex{T}}[
        cosd(α/2)^2    sind(α/2)^2 -im*sind(α);
        sind(α/2)^2    cosd(α/2)^2  im*sind(α);
    -im*sind(α)/2   im*sind(α)/2       cosd(α)]

####
####
####

####
# EPG parameterization interface
# For each parameterization, define:
#   - struct MyParameterization{T,ETL} <: FieldVector{N,T}
#   - restructure(::MyParameterization, ::NTuple{N,T})
#   - parameter getters

struct EPGOptions{T,ETL} <: FieldVector{5,T}
    α::T
    TE::T
    T2::T
    T1::T
    β::T
end
Base.NamedTuple(θ::EPGOptions) = NamedTuple{(:α, :TE, :T2, :T1, :β)}(Tuple(θ))
@inline EPGOptions(xs::NamedTuple{(:α, :TE, :T2, :T1, :β)}, ::Val{ETL}, ::Type{T} = floattype(xs)) where {T,ETL} = EPGOptions{T,ETL}(Tuple(xs))
@inline restructure(::EPGOptions{<:Any, ETL}, xs::NTuple{5,T}) where {T,ETL} = EPGOptions{T,ETL}(xs)

@inline Base.eltype(::EPGOptions{T}) where {T} = T
@inline echotrainlength(::EPGOptions{T,ETL}) where {T,ETL} = ETL
@inline B1correction(θ::EPGOptions{T}) where {T} = T(θ.α / 180) # Multiplicative FA correction: A = α/180
@inline flipangle(θ::EPGOptions{T}, i::Int) where {T} = ifelse(i == 0, T(90), ifelse(i == 1, T(180), θ.β)) # Pulse sequence: 90, 180, β, β, ...
@inline echotime(θ::EPGOptions{T}) where {T} = θ.TE
@inline T2time(θ::EPGOptions{T}) where {T} = θ.T2
@inline T1time(θ::EPGOptions{T}) where {T} = θ.T1

struct EPGIncreasingFlipAnglesOptions{T,ETL} <: FieldVector{6,T}
    α::T
    α1::T
    α2::T
    TE::T
    T2::T
    T1::T
end
Base.NamedTuple(θ::EPGIncreasingFlipAnglesOptions) = NamedTuple{(:α, :α1, :α2, :TE, :T2, :T1)}(Tuple(θ))
@inline EPGIncreasingFlipAnglesOptions(xs::NamedTuple{(:α, :α1, :α2, :TE, :T2, :T1)}, ::Val{ETL}, ::Type{T} = floattype(xs)) where {T,ETL} = EPGIncreasingFlipAnglesOptions{T,ETL}(Tuple(xs))
@inline restructure(::EPGIncreasingFlipAnglesOptions{<:Any, ETL}, xs::NTuple{6,T}) where {T,ETL} = EPGIncreasingFlipAnglesOptions{T,ETL}(xs)

@inline Base.eltype(::EPGIncreasingFlipAnglesOptions{T}) where {T} = T
@inline echotrainlength(::EPGIncreasingFlipAnglesOptions{T,ETL}) where {T,ETL} = ETL
@inline B1correction(θ::EPGIncreasingFlipAnglesOptions{T}) where {T} = T(θ.α / 180) # Multiplicative FA correction: A = α/180
@inline flipangle(θ::EPGIncreasingFlipAnglesOptions{T}, i::Int) where {T} = ifelse(i == 0, T(90), ifelse(i == 1, θ.α1, ifelse(i == 2, θ.α2, T(180)))) # Pulse sequence: 90, α1, α2, 180, 180, ...
@inline echotime(θ::EPGIncreasingFlipAnglesOptions{T}) where {T} = θ.TE
@inline T2time(θ::EPGIncreasingFlipAnglesOptions{T}) where {T} = θ.T2
@inline T1time(θ::EPGIncreasingFlipAnglesOptions{T}) where {T} = θ.T1

const EPGParameterization{T,ETL} = Union{
    EPGOptions{T,ETL},
    EPGIncreasingFlipAnglesOptions{T,ETL},
}

#### Destructuring/restructuring to/from vectors

@generated function destructure(θ, ::Val{Fs}) where {Fs}
    vals = [:(getproperty(θ, $(QuoteNode(F)))) for F in Fs]
    :(Base.@_inline_meta; SVector{$(length(Fs)), $(eltype(θ))}(tuple($(vals...))))
end
destructure(θ, Fs::NTuple{N,Symbol}) where {N} = SVector{N, eltype(θ)}(map(F -> getproperty(θ, F), Fs))

@generated function restructure(θ, x::AbstractVector{T}, ::Val{Fs}) where {T, Fs}
    idxmap = NamedTuple{Fs}(ntuple(i -> i, length(Fs)))
    vals   = [F ∈ Fs ? :(@inbounds(x[$(getproperty(idxmap, F))])) : :($T(getproperty(θ, $(QuoteNode(F))))) for F in fieldsof(θ)]
    :(Base.@_inline_meta; restructure(θ, tuple($(vals...))))
end
function restructure(θ, x::AbstractVector{T}, Fs::NTuple{N,Symbol}) where {T, N}
    fields = fieldsof(typeof(θ))
    vals   = ntuple(length(θ)) do i
        @inbounds for j in 1:N
            (Fs[j] == fields[i]) && return x[j]
        end
        @inbounds T(getproperty(θ, fields[i]))
    end
    return restructure(θ, vals)
end
restructure(θ, x::NamedTuple{Fs}) where {Fs} = restructure(θ, SVector(Tuple(x)), Val(Fs))

####
#### Abstract Interface
####

abstract type AbstractEPGWorkspace{T,ETL} end

@inline Base.eltype(::AbstractEPGWorkspace{T}) where {T} = T
@inline echotrainlength(::AbstractEPGWorkspace{T,ETL}) where {T,ETL} = ETL
@inline mpsv(work::AbstractEPGWorkspace) = work.MPSV
@inline decaycurve(work::AbstractEPGWorkspace) = work.dc

@inline EPGdecaycurve_work(::EPGParameterization{T,ETL}) where {T,ETL} = EPGdecaycurve_work(T, ETL)
@inline EPGdecaycurve_work(::Type{T}, ETL::Int) where {T} = EPGWork_ReIm_DualMVector_Split(T, ETL) # fallback
@inline EPGdecaycurve_work(::Type{T}, ETL::Int) where {T <: FloatingTypes} = EPGWork_ReIm_DualMVector_Split(T, ETL) # default for T <: SIMD.FloatingTypes

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
@inline EPGdecaycurve(ETL::Int, α::Real, TE::Real, T2::Real, T1::Real, β::Real) = EPGdecaycurve(EPGOptions{floattype((α, TE, T2, T1, β)), ETL}((α, TE, T2, T1, β)))
@inline EPGdecaycurve(θ::EPGParameterization{T,ETL}) where {T,ETL} = EPGdecaycurve!(EPGdecaycurve_work(θ), θ)
@inline EPGdecaycurve!(work::AbstractEPGWorkspace{T,ETL}, args::Real...) where {T,ETL} = EPGdecaycurve!(decaycurve(work), work, EPGParameterization{T,ETL}(args...))
@inline EPGdecaycurve!(work::AbstractEPGWorkspace{T,ETL}, θ::EPGParameterization{T,ETL}) where {T,ETL} = EPGdecaycurve!(decaycurve(work), work, θ)
@inline EPGdecaycurve!(dc::AbstractVector{T}, work::AbstractEPGWorkspace{T,ETL}, θ::EPGParameterization{T,ETL}) where {T,ETL} = epg_decay_curve!(dc, work, θ)

####
#### Jacobian utilities (currently hardcoded for `EPGWork_ReIm_DualMVector_Split`)
####

struct EPGWorkCacheDict{ETL} <: AbstractDict{DataType, Any}
    dict::Dict{DataType, Any}
    EPGWorkCacheDict{ETL}() where {ETL} = new{ETL}(Dict{DataType, Any}())
end
@inline Base.keys(caches::EPGWorkCacheDict) = Base.keys(caches.dict)
@inline Base.values(caches::EPGWorkCacheDict) = Base.values(caches.dict)
@inline Base.length(caches::EPGWorkCacheDict) = Base.length(caches.dict)
@inline Base.iterate(caches::EPGWorkCacheDict, state...) = Base.iterate(caches.dict, state...)

@inline function Base.getindex(caches::EPGWorkCacheDict{ETL}, ::Type{T}) where {T,ETL}
    R = cachetype(caches, T)
    get!(caches.dict, T) do
        EPGWork_ReIm_DualMVector_Split(T, ETL)::R
    end::R
end
@inline cachetype(::EPGWorkCacheDict{ETL}, ::Type{T}) where {T,ETL} = EPGWork_ReIm_DualMVector_Split{T, ETL, MVector{ETL, SVector{3, T}}, MVector{ETL, T}}

#=
struct EPGWorkDualCache{T, D}
    work::T
    dual_work::D
end

function EPGWorkDualCache(::Type{T}, ::Val{ETL}, ::Val{chunk_size}) where {T, ETL, chunk_size}
    D = ForwardDiff.Dual{nothing, T, chunk_size}
    work = EPGWork_ReIm_DualVector_Split(T, ETL)
    dual_work = EPGWork_ReIm_DualVector_Split(D, ETL)
    EPGWorkDualCache(work, dual_work)
end

getindex(c::EPGWorkDualCache, ::Type{T}) where {T} = c.work
getindex(c::EPGWorkDualCache, ::Type{D}) where {D <: ForwardDiff.Dual} = remake(c.dual_work, D)

remake(x::Array, ::Type{T}) where {T} = x
remake(x::Array, ::Type{D}) where {D <: ForwardDiff.Dual} = reinterpret(D, x)

@inline function remake(work::EPGWork_ReIm_DualVector_Split{<:Any, ETL}, ::Type{D}) where {D <: ForwardDiff.Dual, ETL}
    MPSV₁ = reinterpret(SVector{3,D}, work.MPSV₁.data)
    MPSV₁ = SizedVector{ETL,SVector{3,D},typeof(MPSV₁)}(MPSV₁)
    MPSV₂ = reinterpret(SVector{3,D}, work.MPSV₂.data)
    MPSV₂ = SizedVector{ETL,SVector{3,D},typeof(MPSV₂)}(MPSV₂)
    dc    = reinterpret(D, work.dc.data)
    dc    = SizedVector{ETL,D,typeof(dc)}(dc)
    EPGWork_ReIm_DualVector_Split{D,ETL,typeof(MPSV₁),typeof(dc)}(MPSV₁, MPSV₂, dc)
end
=#

struct EPGFunctor{T, ETL, Fs, Tθ <: EPGParameterization{T,ETL}}
    caches::EPGWorkCacheDict{ETL}
    θ::Tθ
end
EPGFunctor(θ::EPGParameterization{T,ETL}, Fs::NTuple{N,Symbol}) where {T,ETL,N} = EPGFunctor{T,ETL,Fs,typeof(θ)}(EPGWorkCacheDict{ETL}(), θ)
EPGFunctor(f!::EPGFunctor{T,ETL,Fs}, θ::EPGParameterization{T,ETL}) where {T,ETL,Fs} = EPGFunctor{T,ETL,Fs,typeof(θ)}(f!.caches, θ)

@inline parameters(f!::EPGFunctor) = f!.θ
@inline optfields(::EPGFunctor{T,ETL,Fs}) where {T,ETL,Fs} = Val(Fs)

function (f!::EPGFunctor)(y::AbstractVector{D}, epg_work::AbstractEPGWorkspace{D,ETL}, x::AbstractVector{D}) where {D,ETL}
    θ = restructure(parameters(f!), x, optfields(f!))
    EPGdecaycurve!(y, epg_work, θ)
end
(f!::EPGFunctor)(x::AbstractVector{D}) where {D} = f!(decaycurve(f!.caches[D]), f!.caches[D], x)
(f!::EPGFunctor)(y::AbstractVector{D}, x::AbstractVector{D}) where {D} = f!(y, f!.caches[D], x)

struct EPGJacobianFunctor{T, ETL, Fs, F <: EPGFunctor{T,ETL,Fs}, R <: DiffResults.DiffResult, C <: ForwardDiff.JacobianConfig}
    f!::F
    res::R
    cfg::C
end
function EPGJacobianFunctor(θ::EPGParameterization{T,ETL}, Fs::NTuple{N,Symbol}) where {T,ETL,N}
    f!  = EPGFunctor(θ, Fs)
    res = DiffResults.JacobianResult(zeros(T, ETL), zeros(T, N))
    cfg = ForwardDiff.JacobianConfig(f!, zeros(T, ETL), zeros(T, N), ForwardDiff.Chunk(N))
    return EPGJacobianFunctor(f!, res, cfg)
end

@inline parameters(j!::EPGJacobianFunctor) = parameters(j!.f!)
@inline optfields(j!::EPGJacobianFunctor) = optfields(j!.f!)

function (j!::EPGJacobianFunctor{T,ETL})(J::Union{AbstractMatrix, DiffResults.DiffResult}, y::AbstractVector{T}, θ::EPGParameterization{T,ETL}) where {T,ETL}
    (; f!, cfg) = j!
    f! = EPGFunctor(f!, θ)
    x  = destructure(parameters(f!), optfields(f!))
    ForwardDiff.jacobian!(J, f!, y, x, cfg)
    return J isa AbstractMatrix ? J : DiffResults.jacobian(J)
end
(j!::EPGJacobianFunctor{T,ETL})(y::AbstractVector{T}, θ::EPGParameterization{T,ETL}) where {T,ETL} = j!(j!.res, y, θ)

####
#### EPGWork_Basic_Cplx
####

struct EPGWork_Basic_Cplx{T, ETL, MPSVType <: AbstractVector{SVector{3,Complex{T}}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV::MPSVType
    dc::DCType
end
function EPGWork_Basic_Cplx(T, ETL::Int)
    MSPV = SizedVector{ETL,SVector{3,Complex{T}}}(undef)
    dc   = SizedVector{ETL,T}(undef)
    EPGWork_Basic_Cplx{T,ETL,typeof(MSPV),typeof(dc)}(MSPV, dc)
end

# Compute a basis function under the extended phase graph algorithm. The magnetization phase state vector (MPSV) is
# successively modified by applying relaxation for TE/2, then a refocusing pulse as described by Hennig (1988),
# then transitioning phase states as given by Hennig (1988) but corrected by Jones (1997), and a finally relaxing for TE/2.
# See the appendix in Prasloski (2012) for details:
#    https://doi.org/10.1002/mrm.23157

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_Basic_Cplx{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    (; MPSV) = work
    A   = B1correction(θ)
    αₑₓ = A * 90
    α₁  = A * 180
    αᵢ  = A * θ.β
    TE  = echotime(θ)
    T2  = T2time(θ)
    T1  = T1time(θ)
    V   = SA{Complex{T}} # alias

    # Precompute compute element flip matrices and other intermediate variables
    E1, E2 = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
    E      = SA{T}[E2, E2, E1]
    R₁     = element_flipmat(α₁)
    Rᵢ     = element_flipmat(αᵢ)

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
        MPSV[1]  = V[Mᵢ[2], Mᵢ₊₁[2], Mᵢ[3]] # (F₁, F₁*, Z₁)⁺ = (F₁*, F₂*, Z₁)
        for j in 2:ETL-1
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV[j+1]
            MPSV[j]        = V[Mᵢ₋₁[1], Mᵢ₊₁[2], Mᵢ[3]] # (Fᵢ, Fᵢ*, Zᵢ)⁺ = (Fᵢ₋₁, Fᵢ₊₁*, Zᵢ)
        end
        Mᵢ₋₁, Mᵢ  = Mᵢ, Mᵢ₊₁
        MPSV[ETL] = V[Mᵢ₋₁[1], 0, Mᵢ[3]] # (Fₙ, Fₙ*, Zₙ)⁺ = (Fₙ₋₁, 0, Zₙ)

        # Relaxation for TE/2
        for j in 1:ETL
            MPSV[j] = E .* MPSV[j] # Relaxation for TE/2
        end
        dc[i] = abs(MPSV[1][1]) # first echo amplitude
    end

    return dc
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_Basic_Cplx{T,ETL}, θ::EPGIncreasingFlipAnglesOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    (; MPSV) = work
    A   = B1correction(θ)
    αₑₓ = A * flipangle(θ, 0)
    TE  = echotime(θ)
    T2  = T2time(θ)
    T1  = T1time(θ)
    V   = SA{Complex{T}} # alias

    # Precompute compute element flip matrices and other intermediate variables
    E1, E2 = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
    E      = SA{T}[E2, E2, E1]

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
        MPSV[1]  = V[Mᵢ[2], Mᵢ₊₁[2], Mᵢ[3]] # (F₁, F₁*, Z₁)⁺ = (F₁*, F₂*, Z₁)
        for j in 2:ETL-1
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV[j+1]
            MPSV[j]        = V[Mᵢ₋₁[1], Mᵢ₊₁[2], Mᵢ[3]] # (Fᵢ, Fᵢ*, Zᵢ)⁺ = (Fᵢ₋₁, Fᵢ₊₁*, Zᵢ)
        end
        Mᵢ₋₁, Mᵢ  = Mᵢ, Mᵢ₊₁
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

struct EPGWork_ReIm{T, ETL, MPSVType <: AbstractVector{SVector{3,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV::MPSVType
    dc::DCType
end
function EPGWork_ReIm(T, ETL::Int)
    MSPV = SizedVector{ETL,SVector{3,T}}(undef)
    dc   = SizedVector{ETL,T}(undef)
    EPGWork_ReIm{T,ETL,typeof(MSPV),typeof(dc)}(MSPV, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    (; MPSV) = work
    A  = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V  = SA{T} # alias

    # Precompute intermediate variables
    E₁, E₂           = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
    sin½α₁, cos½α₁   = sincos(α₁/2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁            = 2*sin½α₁*cos½α₁
    sinαᵢ, cosαᵢ     = sincos(αᵢ)
    cos²½αᵢ          = (1+cosαᵢ)/2
    sin²½αᵢ          = 1-cos²½αᵢ
    a₁, b₁, c₁       = E₂^2*cos²½α₁, E₂^2*sin²½α₁, E₁*E₂*sinα₁
    aᵢ, bᵢ, cᵢ, dᵢ   = E₂^2*cos²½αᵢ, E₂^2*sin²½αᵢ, E₁*E₂*sinαᵢ, E₁^2*cosαᵢ
    F, F̄, Z          = V[aᵢ, bᵢ, cᵢ], V[bᵢ, aᵢ, -cᵢ], V[-cᵢ/2, cᵢ/2, dᵢ]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀      = sin½α₁ # since αₑₓ = ½α₁
        Mᵢ⁺     = V[b₁*m₀, 0, -c₁*m₀/2]
        dc[1]   = abs(Mᵢ⁺[1])
        MPSV[1] = Mᵢ⁺
        MPSV[2] = V[a₁*m₀, 0, 0]
        MPSV[3] = V[0, 0, 0]
    end

    @inbounds for i in 2:ETL-1
        # j = 1, initialize and update `dc`
        Mᵢ, Mᵢ₊₁ = MPSV[1], MPSV[2]
        Mᵢ⁺      = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV[1]  = Mᵢ⁺

        # inner loop
        jup = min(i, ETL-i)
        @simd for j in 2:jup
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV[j+1]
            MPSV[j]        = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        end

        # cleanup for next iteration
        if i == jup
            Mᵢ₋₁      = Mᵢ
            MPSV[i+1] = V[F⋅Mᵢ₋₁, 0, 0]
            MPSV[i+2] = V[0, 0, 0]
        end
    end

    @inbounds dc[ETL] = abs(F̄⋅MPSV[1])

    return dc
end

####
#### EPGWork_ReIm_Generated
####

struct EPGWork_ReIm_Generated{T, ETL, MPSVType <: AbstractVector{SVector{3,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV::MPSVType
    dc::DCType
end
function EPGWork_ReIm_Generated(T, ETL::Int)
    MSPV = SizedVector{ETL,SVector{3,T}}(undef)
    dc   = SizedVector{ETL,T}(undef)
    EPGWork_ReIm_Generated{T,ETL,typeof(MSPV),typeof(dc)}(MSPV, dc)
end

function epg_decay_curve_impl!(dc::Type{A}, work::Type{W}, θ::Type{O}) where {T, ETL, A<:AbstractVector{T}, W<:EPGWork_ReIm_Generated{T,ETL}, O<:EPGOptions{T,ETL}}
    MPSV(i::Int) = Symbol(:MPSV, i)
    quote
        # Unpack workspace
        A = B1correction(θ)
        α₁  = deg2rad(A * 180)
        αᵢ  = deg2rad(A * θ.β)
        TE  = echotime(θ)
        T2  = T2time(θ)
        T1  = T1time(θ)

        # Precompute intermediate variables
        # α                = deg2rad(α)
        # α₁, αᵢ           = α, α*β/180
        V                = SA{$T} # alias
        E₁, E₂           = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
        sin½α₁,  cos½α₁  = sincos(α₁/2)
        sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
        sinα₁            = 2*sin½α₁*cos½α₁
        sinαᵢ, cosαᵢ     = sincos(αᵢ)
        cos²½αᵢ          = (1+cosαᵢ)/2
        sin²½αᵢ          = 1-cos²½αᵢ
        a₁, b₁, c₁       = E₂^2*cos²½α₁, E₂^2*sin²½α₁, E₁*E₂*sinα₁
        aᵢ, bᵢ, cᵢ, dᵢ   = E₂^2*cos²½αᵢ, E₂^2*sin²½αᵢ, E₁*E₂*sinαᵢ, E₁^2*cosαᵢ
        F, F̄, Z          = V[aᵢ, bᵢ, cᵢ], V[bᵢ, aᵢ, -cᵢ], V[-cᵢ/2, cᵢ/2, dᵢ]

        # Initialize MPSV vector elements
        $([
            :($(MPSV(i)) = zero(SVector{3,$T}))
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
                        (Mᵢ₋₁, Mᵢ, Mᵢ₊₁) = (Mᵢ, Mᵢ₊₁, $(MPSV(j+1)))
                        $(MPSV(j))       = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
                    end
                    for j in 2:min(i, ETL-i)+1
                ]...)
            end
            for i in 2:ETL-1
        ]...)

        # Last echo
        @inbounds dc[$ETL] = abs(F̄⋅$(MPSV(1)))

        return dc
    end
end

@generated function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_Generated{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    return epg_decay_curve_impl!(dc, work, θ)
end

####
#### EPGWork_ReIm_DualVector
####

struct EPGWork_ReIm_DualVector{T, ETL, MPSVType <: AbstractVector{SVector{3,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualVector(T, ETL::Int)
    MPSV₁ = SizedVector{ETL,SVector{3,T}}(undef)
    MPSV₂ = SizedVector{ETL,SVector{3,T}}(undef)
    dc    = SizedVector{ETL,T}(undef)
    EPGWork_ReIm_DualVector{T,ETL,typeof(MPSV₁),typeof(dc)}(MPSV₁, MPSV₂, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualVector{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    (; MPSV₁, MPSV₂) = work
    A  = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V  = SA{T} # alias

    # Precompute intermediate variables
    # α                = deg2rad(α)
    # α₁, αᵢ           = α, α*β/180
    E₁, E₂           = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
    sin½α₁, cos½α₁   = sincos(α₁/2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁            = 2*sin½α₁*cos½α₁
    sinαᵢ, cosαᵢ     = sincos(αᵢ)
    cos²½αᵢ          = (1 + cosαᵢ)/2
    sin²½αᵢ          = 1-cos²½αᵢ
    a₁, b₁, c₁       = E₂^2*cos²½α₁, E₂^2*sin²½α₁, E₁*E₂*sinα₁
    aᵢ, bᵢ, cᵢ, dᵢ   = E₂^2*cos²½αᵢ, E₂^2*sin²½αᵢ, E₁*E₂*sinαᵢ, E₁^2*cosαᵢ
    F, F̄, Z          = V[aᵢ, bᵢ, cᵢ], V[bᵢ, aᵢ, -cᵢ], V[-cᵢ/2, cᵢ/2, dᵢ]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀           = sin½α₁ # since αₑₓ = ½α₁
        Mᵢ⁺          = V[b₁*m₀, 0, -c₁*m₀/2]
        dc[1]        = abs(Mᵢ⁺[1])
        MPSV₁[1]     = Mᵢ⁺
        MPSV₁[2]     = V[a₁*m₀, 0, 0]
        MPSV₁[3]     = V[0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 2:ETL-1
        # j = 1, initialize and update `dc`
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2]
        Mᵢ⁺      = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺

        # inner loop
        jup = min(i, ETL-i)
        @simd for j in 2:jup
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        end

        # cleanup for next iteration
        if i == jup
            Mᵢ₋₁       = Mᵢ
            MPSV₁[i+1] = V[F⋅Mᵢ₋₁, 0, 0]
            MPSV₁[i+2] = V[0, 0, 0]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F̄⋅MPSV₂[1])

    return dc
end

####
#### EPGWork_ReIm_DualVector_Split
####

struct EPGWork_ReIm_DualVector_Split{T, ETL, MPSVType <: AbstractVector{SVector{3,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualVector_Split(T, ETL::Int)
    MPSV₁ = SizedVector{ETL,SVector{3,T}}(undef)
    MPSV₂ = SizedVector{ETL,SVector{3,T}}(undef)
    dc    = SizedVector{ETL,T}(undef)
    EPGWork_ReIm_DualVector_Split{T,ETL,typeof(MPSV₁),typeof(dc)}(MPSV₁, MPSV₂, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualVector_Split{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    (; MPSV₁, MPSV₂) = work
    A  = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V  = SA{T} # alias

    # Precompute intermediate variables
    # α                = deg2rad(α)
    # α₁, αᵢ           = α, α*β/180
    E₁, E₂           = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
    sin½α₁, cos½α₁   = sincos(α₁/2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁            = 2*sin½α₁*cos½α₁
    sinαᵢ, cosαᵢ     = sincos(αᵢ)
    cos²½αᵢ          = (1+cosαᵢ)/2
    sin²½αᵢ          = 1-cos²½αᵢ
    a₁, b₁, c₁       = E₂^2*cos²½α₁, E₂^2*sin²½α₁, E₁*E₂*sinα₁
    aᵢ, bᵢ, cᵢ, dᵢ   = E₂^2*cos²½αᵢ, E₂^2*sin²½αᵢ, E₁*E₂*sinαᵢ, E₁^2*cosαᵢ
    F, F̄, Z          = V[aᵢ, bᵢ, cᵢ], V[bᵢ, aᵢ, -cᵢ], V[-cᵢ/2, cᵢ/2, dᵢ]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀           = sin½α₁ # since αₑₓ = ½α₁
        Mᵢ⁺          = V[b₁*m₀, 0, -c₁*m₀/2]
        dc[1]        = abs(Mᵢ⁺[1])
        MPSV₁[1]     = Mᵢ⁺
        MPSV₁[2]     = V[a₁*m₀, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 2:ETL÷2
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺      = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:i-1
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        end
        MPSV₁[i]     = V[F⋅Mᵢ, 0, Z⋅Mᵢ₊₁]
        MPSV₁[i+1]   = V[F⋅Mᵢ₊₁, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in ETL÷2+1:ETL-1
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺      = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:ETL-i
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F̄⋅MPSV₂[1])

    return dc
end

####
#### EPGWork_ReIm_DualVector_Split_Dynamic
####

struct EPGWork_ReIm_DualVector_Split_Dynamic{T, MPSVType <: AbstractVector{SVector{3,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T, Nothing}
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualVector_Split_Dynamic(::Type{T}, ETL::Int) where {T}
    MPSV₁ = zeros(SVector{3,T}, ETL)
    MPSV₂ = zeros(SVector{3,T}, ETL)
    dc    = zeros(T, ETL)
    EPGWork_ReIm_DualVector_Split_Dynamic{T, typeof(MPSV₁), typeof(dc)}(MPSV₁, MPSV₂, dc)
end

function DECAES.epg_decay_curve!(dc::AbstractVector, work::EPGWork_ReIm_DualVector_Split_Dynamic{T}, θ::EPGOptions{T}) where {T}
    ETL = length(dc)

    # Unpack workspace
    (; MPSV₁, MPSV₂) = work
    A  = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V  = SA{T} # alias

    # Precompute intermediate variables
    E₁, E₂           = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
    sin½α₁, cos½α₁   = sincos(α₁/2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁            = 2*sin½α₁*cos½α₁
    sinαᵢ, cosαᵢ     = sincos(αᵢ)
    cos²½αᵢ          = (1+cosαᵢ)/2
    sin²½αᵢ          = 1-cos²½αᵢ
    a₁, b₁, c₁       = E₂^2*cos²½α₁, E₂^2*sin²½α₁, E₁*E₂*sinα₁
    aᵢ, bᵢ, cᵢ, dᵢ   = E₂^2*cos²½αᵢ, E₂^2*sin²½αᵢ, E₁*E₂*sinαᵢ, E₁^2*cosαᵢ
    F, F̄, Z          = V[aᵢ, bᵢ, cᵢ], V[bᵢ, aᵢ, -cᵢ], V[-cᵢ/2, cᵢ/2, dᵢ]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀           = sin½α₁ # since αₑₓ = ½α₁
        Mᵢ⁺          = V[b₁*m₀, 0, -c₁*m₀/2]
        dc[1]        = abs(Mᵢ⁺[1])
        MPSV₁[1]     = Mᵢ⁺
        MPSV₁[2]     = V[a₁*m₀, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 2:ETL÷2
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺      = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:i-1
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        end
        MPSV₁[i]     = V[F⋅Mᵢ, 0, Z⋅Mᵢ₊₁]
        MPSV₁[i+1]   = V[F⋅Mᵢ₊₁, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in ETL÷2+1:ETL-1
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺      = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:ETL-i
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F̄⋅MPSV₂[1])

    return dc
end

####
#### EPGWork_ReIm_DualMVector_Split
####

struct EPGWork_ReIm_DualMVector_Split{T, ETL, MPSVType <: AbstractVector{SVector{3,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualMVector_Split(T, ETL::Int)
    MPSV₁ = MVector{ETL,SVector{3,T}}(undef)
    MPSV₂ = MVector{ETL,SVector{3,T}}(undef)
    dc    = MVector{ETL,T}(undef)
    EPGWork_ReIm_DualMVector_Split{T,ETL,typeof(MPSV₁),typeof(dc)}(MPSV₁, MPSV₂, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualMVector_Split{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    (; MPSV₁, MPSV₂) = work
    A  = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V  = SA{T} # alias

    # Precompute intermediate variables
    # α                = deg2rad(α)
    # α₁, αᵢ           = α, α*β/180
    E₁, E₂           = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
    sin½α₁, cos½α₁   = sincos(α₁/2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁            = 2*sin½α₁*cos½α₁
    sinαᵢ, cosαᵢ     = sincos(αᵢ)
    cos²½αᵢ          = (1+cosαᵢ)/2
    sin²½αᵢ          = 1-cos²½αᵢ
    a₁, b₁, c₁       = E₂^2*cos²½α₁, E₂^2*sin²½α₁, E₁*E₂*sinα₁
    aᵢ, bᵢ, cᵢ, dᵢ   = E₂^2*cos²½αᵢ, E₂^2*sin²½αᵢ, E₁*E₂*sinαᵢ, E₁^2*cosαᵢ
    F, F̄, Z          = V[aᵢ, bᵢ, cᵢ], V[bᵢ, aᵢ, -cᵢ], V[-cᵢ/2, cᵢ/2, dᵢ]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀           = sin½α₁ # since αₑₓ = ½α₁
        Mᵢ⁺          = V[b₁*m₀, 0, -c₁*m₀/2]
        dc[1]        = abs(Mᵢ⁺[1])
        MPSV₁[1]     = Mᵢ⁺
        MPSV₁[2]     = V[a₁*m₀, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 2:ETL÷2
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺      = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:i-1
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        end
        MPSV₁[i]     = V[F⋅Mᵢ, 0, Z⋅Mᵢ₊₁]
        MPSV₁[i+1]   = V[F⋅Mᵢ₊₁, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in ETL÷2+1:ETL-1
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺      = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:ETL-i
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F̄⋅MPSV₂[1])

    return dc
end

####
#### EPGWork_ReIm_DualPaddedMVector_Vec_Split
####

struct EPGWork_ReIm_DualPaddedMVector_Vec_Split{T, ETL, MPSVType <: AbstractVector{Vec{4,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualPaddedMVector_Vec_Split(T, ETL::Int)
    MPSV₁ = MVector{ETL,Vec{4,T}}(undef)
    MPSV₂ = MVector{ETL,Vec{4,T}}(undef)
    dc    = MVector{ETL,T}(undef)
    EPGWork_ReIm_DualPaddedMVector_Vec_Split{T,ETL,typeof(MPSV₁),typeof(dc)}(MPSV₁, MPSV₂, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualPaddedMVector_Vec_Split{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    (; MPSV₁, MPSV₂) = work
    A  = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V  = Vec{4,T} # alias

    # Precompute intermediate variables
    # α                = deg2rad(α)
    # α₁, αᵢ           = α, α*β/180
    E₁, E₂           = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
    sin½α₁, cos½α₁   = sincos(α₁/2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁            = 2*sin½α₁*cos½α₁
    sinαᵢ, cosαᵢ     = sincos(αᵢ)
    cos²½αᵢ          = (1+cosαᵢ)/2
    sin²½αᵢ          = 1-cos²½αᵢ
    a₁, b₁, c₁       = E₂^2*cos²½α₁, E₂^2*sin²½α₁, E₁*E₂*sinα₁
    aᵢ, bᵢ, cᵢ, dᵢ   = E₂^2*cos²½αᵢ, E₂^2*sin²½αᵢ, E₁*E₂*sinαᵢ, E₁^2*cosαᵢ
    F, F̄, Z          = V((aᵢ, bᵢ, cᵢ, 0)), V((bᵢ, aᵢ, -cᵢ, 0)), V((-cᵢ/2, cᵢ/2, dᵢ, 0))

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀           = sin½α₁ # since αₑₓ = ½α₁
        Mᵢ⁺          = V((b₁*m₀, 0, -c₁*m₀/2, 0))
        dc[1]        = abs(Mᵢ⁺[1])
        MPSV₁[1]     = Mᵢ⁺
        MPSV₁[2]     = V((a₁*m₀, 0, 0, 0))
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 2:ETL÷2
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺      = V((sum(F̄*Mᵢ), sum(F̄*Mᵢ₊₁), sum(Z*Mᵢ), 0))
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:i-1
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V((sum(F*Mᵢ₋₁), sum(F̄*Mᵢ₊₁), sum(Z*Mᵢ), 0))
        end
        MPSV₁[i]     = V((sum(F*Mᵢ), 0, sum(Z*Mᵢ₊₁), 0))
        MPSV₁[i+1]   = V((sum(F*Mᵢ₊₁), 0, 0, 0))
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in ETL÷2+1:ETL-1
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺      = V((sum(F̄*Mᵢ), sum(F̄*Mᵢ₊₁), sum(Z*Mᵢ), 0))
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:ETL-i
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V((sum(F*Mᵢ₋₁), sum(F̄*Mᵢ₊₁), sum(Z*Mᵢ), 0))
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(sum(F̄*MPSV₂[1]))

    return dc
end

####
#### EPGWork_ReIm_DualPaddedVector_Split
####

struct EPGWork_ReIm_DualPaddedVector_Split{T, ETL, MPSVType <: AbstractVector{SVector{4,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualPaddedVector_Split(T, ETL::Int)
    MPSV₁ = SizedVector{ETL,SVector{4,T}}(undef)
    MPSV₂ = SizedVector{ETL,SVector{4,T}}(undef)
    dc    = SizedVector{ETL,T}(undef)
    EPGWork_ReIm_DualPaddedVector_Split{T,ETL,typeof(MPSV₁),typeof(dc)}(MPSV₁, MPSV₂, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualPaddedVector_Split{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    (; MPSV₁, MPSV₂) = work
    A  = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)
    V  = SA{T} # alias

    # Precompute intermediate variables
    # α                = deg2rad(α)
    # α₁, αᵢ           = α, α*β/180
    E₁, E₂           = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
    sin½α₁, cos½α₁   = sincos(α₁/2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁            = 2*sin½α₁*cos½α₁
    sinαᵢ, cosαᵢ     = sincos(αᵢ)
    cos²½αᵢ          = (1+cosαᵢ)/2
    sin²½αᵢ          = 1-cos²½αᵢ
    a₁, b₁, c₁       = E₂^2*cos²½α₁, E₂^2*sin²½α₁, E₁*E₂*sinα₁
    aᵢ, bᵢ, cᵢ, dᵢ   = E₂^2*cos²½αᵢ, E₂^2*sin²½αᵢ, E₁*E₂*sinαᵢ, E₁^2*cosαᵢ
    F, F̄, Z          = V[aᵢ, bᵢ, cᵢ, 0], V[bᵢ, aᵢ, -cᵢ, 0], V[-cᵢ/2, cᵢ/2, dᵢ, 0]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀           = sin½α₁ # since αₑₓ = ½α₁
        Mᵢ⁺          = V[b₁*m₀, 0, -c₁*m₀/2, 0]
        dc[1]        = abs(Mᵢ⁺[1])
        MPSV₁[1]     = Mᵢ⁺
        MPSV₁[2]     = V[a₁*m₀, 0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 2:ETL÷2
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺      = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ, 0]
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:i-1
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ, 0]
        end
        MPSV₁[i]     = V[F⋅Mᵢ, 0, Z⋅Mᵢ₊₁, 0]
        MPSV₁[i+1]   = V[F⋅Mᵢ₊₁, 0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in ETL÷2+1:ETL-1
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `dc`
        Mᵢ⁺      = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ, 0]
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        @simd for j in 2:ETL-i
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
            MPSV₁[j]       = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ, 0]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds dc[ETL] = abs(F̄⋅MPSV₂[1])

    return dc
end

####
#### EPGWork_Vec
####

# Flip matrix and relaxation matrix steps are combined into one loop, and SIMD.jl `Vec` types are used instead of `Complex`.
# As this function is called many times during T2mapSEcorr, the micro-optimizations are worth the loss of code readability.
# See `EPGWork_Basic_Cplx` for a more readable, mathematically identicaly implementation.

struct EPGWork_Vec{T, ETL, MPSVType <: AbstractVector{Vec{2,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV::MPSVType
    dc::DCType
end
function EPGWork_Vec(T, ETL::Int)
    MSPV = SizedVector{3*ETL,Vec{2,T}}(undef)
    dc = SizedVector{ETL,T}(undef)
    EPGWork_Vec{T,ETL,typeof(MSPV),typeof(dc)}(MSPV, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_Vec{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    ###########################
    # Setup
    (; MPSV) = work
    A  = B1correction(θ)
    α₁ = deg2rad(A * 180)
    αᵢ = deg2rad(A * θ.β)
    TE = echotime(θ)
    T2 = T2time(θ)
    T1 = T1time(θ)

    @inbounds begin
        # Initialize magnetization phase state vector (MPSV)
        E2, E1  = exp(-TE/T2), exp(-TE/T1)
        E2_half = exp(-(TE/2)/T2)
        m₀      = E2_half * sin(α₁/2) # initial population; since αₑₓ = α₁/2
        M1x     =  m₀ * cos(α₁/2)^2   # M1x, M1y, M1z are elements resulting from first refocusing pulse applied to [m₀, 0, 0]
        M1y     =  m₀ - M1x           # M1y = m₀ * sin(α₁/2)^2 = m₀ - m₀ * cos(α₁/2)^2 = m₀ - M1x
        M1z     = -m₀ * sin(α₁)/2     # Note: this is the imaginary part
        dc[1]   = E2_half * abs(M1y)  # first echo amplitude

        # Apply first relaxation matrix iteration on non-zero states
        MPSV[1] = Vec((E2 * M1y, zero(T)))
        MPSV[2] = zero(Vec{2,T})
        MPSV[3] = Vec((zero(T), E1 * M1z))
        MPSV[4] = Vec((E2 * M1x, zero(T)))

        # Extract matrix elements + initialize temporaries
        a1, a2, a3, a4, a5 = sin(αᵢ), cos(αᵢ), sin(αᵢ/2)^2, cos(αᵢ/2)^2, sin(αᵢ)/2 # independent elements of T2mat
        b1, b2, b3, b4, b5 = E2*a1, E1*a2, E2*a3, E2*a4, E1*a5
        c1, c3, c4         = E2_half*a1, E2_half*a3, E2_half*a4
        b1F, b5F, c1F      = Vec((-b1, b1)), Vec((-b5, b5)), Vec((-c1, c1))
        Mz3                = MPSV[3]
    end

    @inbounds for i = 2:ETL-1
        ###########################
        # Unroll first flipmat/relaxmat iteration
        Vx, Vy  = MPSV[1], MPSV[2]
        c1z     = shufflevector(c1F * Mz3, Val((1, 0)))
        Mz2     = muladd(c3, Vx, muladd(c4, Vy, -c1z)) # flipmat: 2 -> dc
        Mz4     = muladd(b4, Vx, muladd(b3, Vy, E2_half * c1z)) # relaxmat: 1 -> 4, save in buffer
        dc[i]   = sqrt(sum(Mz2 * Mz2)) # decay curve coefficient
        MPSV[1] = E2_half * Mz2 # relaxmat: 2 -> 1
        b5xy    = shufflevector(b5F * (Vx - Vy), Val((1, 0)))
        Mz3     = muladd(b2, Mz3, b5xy) # relaxmat: 3 -> 3, save in buffer

        ###########################
        # flipmat + relaxmat loop
        @inbounds for j in 4:3:3*min(i-1, ETL)
            Vx, Vy, Vz = MPSV[j], MPSV[j+1], MPSV[j+2]
            b1z        = shufflevector(b1F * Vz, Val((1, 0)))
            MPSV[j  ]  = Mz4 # relaxmat: assign forward, j -> j+3
            Mz4        = muladd(b4, Vx, muladd(b3, Vy,  b1z))
            MPSV[j-2]  = muladd(b3, Vx, muladd(b4, Vy, -b1z)) # relaxmat: assign backwards, j+1 -> j+1-3
            b5xy       = shufflevector(b5F * (Vx - Vy), Val((1, 0)))
            MPSV[j+2]  = muladd(b2, Vz, b5xy) # relaxmat: j+2 -> j+2
        end

        ###########################
        # cleanup + zero next elements
        j         = 3i-2
        Vx        = MPSV[j]
        MPSV[j  ] = Mz4 # relaxmat: assign forward, j -> j+3
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
        dc[end] = sqrt(sum(Mz2 * Mz2))
    end

    return dc
end

####
#### Algorithm list
####

const EPG_Algorithms = Any[
    EPGWork_Basic_Cplx,
    EPGWork_Vec,
    EPGWork_ReIm,
    EPGWork_ReIm_DualVector,
    EPGWork_ReIm_DualVector_Split,
    EPGWork_ReIm_DualVector_Split_Dynamic,
    EPGWork_ReIm_DualMVector_Split,
    EPGWork_ReIm_DualPaddedMVector_Vec_Split,
    EPGWork_ReIm_DualPaddedVector_Split,
    EPGWork_ReIm_Generated,
]
