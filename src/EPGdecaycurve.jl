####
#### Helper functions
####

# Element matrix for refocusing pulse with angle α (in degrees); acts on the magnetization state vector (MPSV)
@inline element_flipmat(α::T) where {T} = SA{Complex{T}}[
        cosd(α/2)^2    sind(α/2)^2 -im*sind(α);
        sind(α/2)^2    cosd(α/2)^2  im*sind(α);
    -im*sind(α)/2   im*sind(α)/2       cosd(α)]

####
#### Abstract Interface
####

abstract type AbstractEPGWorkspace{T,ETL} end

@inline Base.eltype(::AbstractEPGWorkspace{T}) where {T} = T
@inline echotrainlength(::AbstractEPGWorkspace{T,ETL}) where {T,ETL} = ETL
@inline get_mpsv(work::AbstractEPGWorkspace) = work.MPSV
@inline get_decaycurve(work::AbstractEPGWorkspace) = work.dc

struct EPGOptions{T,ETL} <: FieldVector{5,T}
    α::T
    TE::T
    T2::T
    T1::T
    β::T
end

@inline function EPGOptions(ETL::Int, α::Real, TE::Real, T2::Real, T1::Real, β::Real)
    T = float(promote_type(typeof(α), typeof(TE), typeof(T2), typeof(T1), typeof(β)))
    EPGOptions{T,ETL}(α, TE, T2, T1, β)
end
@inline EPGOptions(::EPGOptions{T,ETL}, α::Real, TE::Real, T2::Real, T1::Real, β::Real) where {T,ETL} = EPGOptions{T,ETL}(α, TE, T2, T1, β)
@inline EPGOptions(::AbstractEPGWorkspace{T,ETL}, α::Real, TE::Real, T2::Real, T1::Real, β::Real) where {T,ETL} = EPGOptions{T,ETL}(α, TE, T2, T1, β)

@inline function EPGOptions(θ::EPGOptions{T,ETL}, xs::NamedTuple) where {T,ETL}
    θ = setproperties!!(NamedTuple(θ), xs)
    EPGOptions{T,ETL}(Tuple(θ)...)
end
Base.NamedTuple(θ::EPGOptions{T}) where {T} = NamedTuple{(:α,:TE,:T2,:T1,:β), NTuple{5,T}}(Tuple(θ))

@inline EPGdecaycurve_work(::EPGOptions{T,ETL}) where {T,ETL} = EPGdecaycurve_work(T, ETL)
@inline EPGdecaycurve_work(::Type{T}, ETL::Int) where {T} = EPGWork_ReIm_DualMVector_Split(T, ETL) # fallback
@inline EPGdecaycurve_work(::Type{T}, ETL::Int) where {T <: FloatingTypes} = EPGWork_ReIm_DualMVector_Split(T, ETL) # default for T <: SIMD.FloatingTypes

"""
    EPGdecaycurve(ETL::Int, α::Real, TE::Real, T2::Real, T1::Real, β::Real)

Computes the normalized echo decay curve for a MR spin echo sequence
using the extended phase graph algorithm using the given input parameters.

# Arguments
- `ETL::Int`:         echo train length, i.e. number of echos
- `α::Real`: angle of refocusing pulses (Units: degrees)
- `TE::Real`:         inter-echo time (Units: seconds)
- `T2::Real`:         transverse relaxation time (Units: seconds)
- `T1::Real`:         longitudinal relaxation time (Units: seconds)
- `β::Real`:     value of Refocusing Pulse Control Angle (Units: degrees)

# Outputs
- `decay_curve::AbstractVector`: normalized echo decay curve with length `ETL`
"""
@inline EPGdecaycurve(ETL::Int, args::Real...) = EPGdecaycurve(EPGOptions(ETL, args...))
@inline EPGdecaycurve(θ::EPGOptions{T,ETL}) where {T,ETL} = EPGdecaycurve!(EPGdecaycurve_work(θ), θ)
@inline EPGdecaycurve!(work::AbstractEPGWorkspace{T,ETL}, args::Real...) where {T,ETL} = EPGdecaycurve!(get_decaycurve(work), work, EPGOptions{T,ETL}(args...))
@inline EPGdecaycurve!(work::AbstractEPGWorkspace{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL} = EPGdecaycurve!(get_decaycurve(work), work, θ)
@inline EPGdecaycurve!(dc::AbstractVector{T}, work::AbstractEPGWorkspace{T,ETL}, args::Real...) where {T,ETL} = EPGdecaycurve!(dc, work, EPGOptions{T,ETL}(args...))
@inline EPGdecaycurve!(dc::AbstractVector{T}, work::AbstractEPGWorkspace{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL} = @timeit_debug TIMER() "EPG decay curve" epg_decay_curve!(dc, work, θ)

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

struct EPGFunctor{T,ETL,N,Fs}
    caches::EPGWorkCacheDict{ETL}
    θ::EPGOptions{T,ETL}
end
EPGFunctor(θ::EPGOptions{T,ETL}, Fs::NTuple{N,Symbol}) where {T,ETL,N} = EPGFunctor{T,ETL,N,Fs}(EPGWorkCacheDict{ETL}(), θ)
EPGFunctor(f!::EPGFunctor{T,ETL,N,Fs}, θ::EPGOptions{T,ETL}) where {T,ETL,N,Fs} = EPGFunctor{T,ETL,N,Fs}(f!.caches, θ)

@generated function destructure(::EPGFunctor{<:Any,ETL,N,Fs}, θ::EPGOptions{D,ETL}) where {D,ETL,N,Fs}
    vals = [:(getproperty(θ, $(QuoteNode(F)))) for F in Fs]
    :(Base.@_inline_meta; SVector{$N,$D}(tuple($(vals...))))
end

@generated function restructure(f!::EPGFunctor{<:Any,ETL,N,Fs}, x::AbstractVector{D}) where {D,ETL,N,Fs}
    idxmap = NamedTuple{Fs}(1:N)
    vals   = [F ∈ Fs ? :(x[$(getproperty(idxmap, F))]) : :(getproperty(f!.θ, $(QuoteNode(F)))) for F in fieldnames(EPGOptions)]
    :(Base.@_inline_meta; EPGOptions{$D,$ETL}(tuple($(vals...))))
end

function (f!::EPGFunctor)(y::AbstractVector{D}, epg_work::AbstractEPGWorkspace{D,ETL}, x::AbstractVector{D}) where {D,ETL}
    θ = restructure(f!, x)
    DECAES.EPGdecaycurve!(y, epg_work, θ)
end
(f!::EPGFunctor)(x::AbstractVector{D}) where {D} = f!(get_decaycurve(f!.caches[D]), f!.caches[D], x)
(f!::EPGFunctor)(y::AbstractVector{D}, x::AbstractVector{D}) where {D} = f!(y, f!.caches[D], x)

struct EPGJacobianFunctor{T, ETL, N, Fs, R <: DiffResults.DiffResult, C <: ForwardDiff.JacobianConfig}
    f!::EPGFunctor{T,ETL,N,Fs}
    res::R
    cfg::C
end
function EPGJacobianFunctor(θ::EPGOptions{T,ETL}, Fs::NTuple{N,Symbol}) where {T,ETL,N}
    f!  = EPGFunctor(θ, Fs)
    res = DiffResults.JacobianResult(zeros(T, ETL), zeros(T, N))
    cfg = ForwardDiff.JacobianConfig(f!, zeros(T, ETL), zeros(T, N), ForwardDiff.Chunk(N))
    return EPGJacobianFunctor(f!, res, cfg)
end

destructure(j!::EPGJacobianFunctor{<:Any,ETL}, θ::EPGOptions{D,ETL}) where {D,ETL} = destructure(j!.f!, θ)
restructure(j!::EPGJacobianFunctor{<:Any,ETL}, x::AbstractVector{D}) where {D,ETL} = restructure(j!.f!, x)

function (j!::EPGJacobianFunctor{T,ETL})(J::Union{AbstractMatrix, DiffResults.DiffResult}, y::AbstractVector{T}, θ::EPGOptions{T,ETL}) where {T,ETL}
    @unpack f!, cfg = j!
    f! = EPGFunctor(f!, θ)
    x  = destructure(f!, θ)
    ForwardDiff.jacobian!(J, f!, y, x, cfg)
    return J isa AbstractMatrix ? J : DiffResults.jacobian(J)
end
(j!::EPGJacobianFunctor{T,ETL})(y::AbstractVector{T}, θ::EPGOptions{T,ETL}) where {T,ETL} = j!(j!.res, y, θ)

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
    @unpack MPSV = work
    @unpack α, TE, T2, T1, β = θ
    V = SA{Complex{T}} # alias
    
    # Precompute compute element flip matrices and other intermediate variables
    αₑₓ    = α/2
    α₁, αᵢ = α, α * (β / 180)
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
    @unpack MPSV = work
    @unpack α, TE, T2, T1, β = θ
    V = SA{T} # alias

    # Precompute intermediate variables
    α                = deg2rad(α)
    α₁, αᵢ           = α, α*β/180
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
        m₀      = sin½α₁
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
    mpsv(i::Int) = Symbol(:MPSV, i)
    quote
        # Unpack workspace
        @unpack α, TE, T2, T1, β = θ

        # Precompute intermediate variables
        α                = deg2rad(α)
        α₁, αᵢ           = α, α*β/180
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
            :($(mpsv(i)) = zero(SVector{3,$T}))
            for i in 1:ETL
        ]...)

        # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
        @inbounds begin
            m₀         = sin½α₁
            Mᵢ⁺        = V[b₁*m₀, 0, -c₁*m₀/2]
            dc[1]      = abs(Mᵢ⁺[1])
            $(mpsv(1)) = Mᵢ⁺
            $(mpsv(2)) = V[a₁*m₀, 0, 0]
        end

        # Main loop
        $([
            quote
                # Initialize and update `dc` (j = 1)
                @inbounds begin
                    Mᵢ, Mᵢ₊₁   = $(mpsv(1)), $(mpsv(2))
                    Mᵢ⁺        = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
                    dc[$i]     = abs(Mᵢ⁺[1])
                    $(mpsv(1)) = Mᵢ⁺
                end

                # Inner loop
                $([
                    quote
                        (Mᵢ₋₁, Mᵢ, Mᵢ₊₁) = (Mᵢ, Mᵢ₊₁, $(mpsv(j+1)))
                        $(mpsv(j))       = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
                    end
                    for j in 2:min(i, ETL-i)+1
                ]...)
            end
            for i in 2:ETL-1
        ]...)

        # Last echo
        @inbounds dc[$ETL] = abs(F̄⋅$(mpsv(1)))

        return dc
    end
end

@generated function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_Generated{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    return epg_decay_curve_impl!(dc, work, θ)
end

####
#### EPGWork_ReIm_DualCache
####

struct EPGWork_ReIm_DualCache{T, ETL, MPSVType <: AbstractVector{SVector{3,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualCache(T, ETL::Int)
    mpsv₁ = SizedVector{ETL,SVector{3,T}}(undef)
    mpsv₂ = SizedVector{ETL,SVector{3,T}}(undef)
    dc    = SizedVector{ETL,T}(undef)
    EPGWork_ReIm_DualCache{T,ETL,typeof(mpsv₁),typeof(dc)}(mpsv₁, mpsv₂, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualCache{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    @unpack MPSV₁, MPSV₂ = work
    @unpack α, TE, T2, T1, β = θ
    V = SA{T} # alias

    # Precompute intermediate variables
    α                = deg2rad(α)
    α₁, αᵢ           = α, α*β/180
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
        m₀           = sin½α₁
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
#### EPGWork_ReIm_DualCache_Split
####

struct EPGWork_ReIm_DualCache_Split{T, ETL, MPSVType <: AbstractVector{SVector{3,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualCache_Split(T, ETL::Int)
    mpsv₁ = SizedVector{ETL,SVector{3,T}}(undef)
    mpsv₂ = SizedVector{ETL,SVector{3,T}}(undef)
    dc    = SizedVector{ETL,T}(undef)
    EPGWork_ReIm_DualCache_Split{T,ETL,typeof(mpsv₁),typeof(dc)}(mpsv₁, mpsv₂, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualCache_Split{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    @unpack MPSV₁, MPSV₂ = work
    @unpack α, TE, T2, T1, β = θ
    V = SA{T} # alias

    # Precompute intermediate variables
    α                = deg2rad(α)
    α₁, αᵢ           = α, α*β/180
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
        m₀           = sin½α₁
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
        MPSV₁[i]     = V[F⋅Mᵢ, 0, Z⋅Mᵢ₊₁] # j = i
        MPSV₁[i+1]   = V[F⋅Mᵢ₊₁, 0, 0] # j = i + 1
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

    #= Cleaner, but slightly slower
    @inbounds for i in 2:ETL-1
        # j = 1, initialize and update `dc`
        Mᵢ, Mᵢ₊₁ = MPSV₂[1], MPSV₂[2]
        Mᵢ⁺      = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        dc[i]    = abs(Mᵢ⁺[1])
        MPSV₁[1] = Mᵢ⁺
        if i <= ETL÷2
            # inner loop
            @simd for j in 2:i-1
                Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
                MPSV₁[j]       = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
            end
            MPSV₁[i]   = V[F⋅Mᵢ, 0, Z⋅Mᵢ₊₁] # j = i
            MPSV₁[i+1] = V[F⋅Mᵢ₊₁, 0, 0] # j = i + 1
        else
            # inner loop
            @simd for j in 2:ETL-i
                Mᵢ₋₁, Mᵢ, Mᵢ₊₁ = Mᵢ, Mᵢ₊₁, MPSV₂[j+1]
                MPSV₁[j]       = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
            end
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end
    =#

    @inbounds dc[ETL] = abs(F̄⋅MPSV₂[1])

    return dc
end

####
#### EPGWork_ReIm_DualCache_Unrolled
####

struct EPGWork_ReIm_DualCache_Unrolled{T, ETL, MPSVType <: AbstractVector{SVector{3,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    dc::DCType
end
function EPGWork_ReIm_DualCache_Unrolled(T, ETL::Int)
    mpsv₁ = SizedVector{ETL,SVector{3,T}}(undef)
    mpsv₂ = SizedVector{ETL,SVector{3,T}}(undef)
    dc    = SizedVector{ETL,T}(undef)
    EPGWork_ReIm_DualCache_Unrolled{T,ETL,typeof(mpsv₁),typeof(dc)}(mpsv₁, mpsv₂, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualCache_Unrolled{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    @unpack MPSV₁, MPSV₂ = work
    @unpack α, TE, T2, T1, β = θ
    V = SA{T} # alias

    # Precompute intermediate variables
    α                = deg2rad(α)
    α₁, αᵢ           = α, α*β/180
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
        m₀  = sin½α₁
        M₁  = V[b₁*m₀, 0, -c₁*m₀/2]
        M₂  = V[a₁*m₀, 0, 0]
        M₁⁺ = V[F̄⋅M₁, F̄⋅M₂, Z⋅M₁] # V[F̄⋅Mᵢ,   F̄⋅Mᵢ₊₁, Z⋅Mᵢ], j=1
        M₂⁺ = V[F⋅M₁, 0, Z⋅M₂]    # V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ], j=2, M₃ = 0
        M₃⁺ = V[F⋅M₂, 0, 0]       # V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ], j=3, M₃ = M₄ = 0

        dc[1]        = abs(M₁[1])
        dc[2]        = abs(M₁⁺[1])
        MPSV₁[1]     = M₁⁺
        MPSV₁[2]     = M₂⁺
        MPSV₁[3]     = M₃⁺
        MPSV₁[4]     = V[0, 0, 0]
        MPSV₁[5]     = V[0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 3:ETL-1
        # j = 1, j = 2, and update `dc`
        Mᵢ, Mᵢ₊₁, Mᵢ₊₂ = MPSV₂[1], MPSV₂[2], MPSV₂[3]
        Mᵢ⁺            = V[F̄⋅Mᵢ, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
        Mᵢ₊₁⁺          = V[F⋅Mᵢ, F̄⋅Mᵢ₊₂, Z⋅Mᵢ₊₁]
        dc[i]          = abs(Mᵢ⁺[1])
        MPSV₁[1]       = Mᵢ⁺
        MPSV₁[2]       = Mᵢ₊₁⁺

        # inner loop
        jup = min(i, ETL-i)
        @simd for j in 3:2:jup#-isodd(jup)
            Mᵢ₋₁, Mᵢ, Mᵢ₊₁, Mᵢ₊₂ = Mᵢ₊₁, Mᵢ₊₂, MPSV₂[j+1], MPSV₂[j+2]
            MPSV₁[j]             = V[F⋅Mᵢ₋₁, F̄⋅Mᵢ₊₁, Z⋅Mᵢ]
            MPSV₁[j+1]           = V[F⋅Mᵢ,   F̄⋅Mᵢ₊₂, Z⋅Mᵢ₊₁]
        end

        # cleanup for next iteration
        if i == jup
            if iseven(jup)
                Mᵢ₋₁ = Mᵢ₊₁
                MPSV₁[jup+1] = V[F⋅Mᵢ₋₁, 0, 0]
                MPSV₁[jup+2] = V[0, 0, 0]
                MPSV₁[jup+3] = V[0, 0, 0]
            else
                # inner loop: for j in 3:2:jup
                # MPSV₁[jup+1] = V[F⋅Mᵢ, 0, 0] # (with or without)
                MPSV₁[jup+2] = V[0, 0, 0]

                # # (not working) inner loop: for j in 3:2:jup-isodd
                # Mᵢ₋₁, Mᵢ = Mᵢ₊₁, Mᵢ₊₂
                # MPSV₁[jup]   = V[F⋅Mᵢ₋₁, 0, Z⋅Mᵢ]
                # MPSV₁[jup+1] = V[F⋅Mᵢ, 0, 0]
                # MPSV₁[jup+2] = V[0, 0, 0]
                # MPSV₁[jup+3] = V[0, 0, 0]
            end
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
    mpsv₁ = MVector{ETL,SVector{3,T}}(undef)
    mpsv₂ = MVector{ETL,SVector{3,T}}(undef)
    dc    = MVector{ETL,T}(undef)
    EPGWork_ReIm_DualMVector_Split{T,ETL,typeof(mpsv₁),typeof(dc)}(mpsv₁, mpsv₂, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_ReIm_DualMVector_Split{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    @unpack MPSV₁, MPSV₂ = work
    @unpack α, TE, T2, T1, β = θ
    V = SA{T} # alias

    # Precompute intermediate variables
    α                = deg2rad(α)
    α₁, αᵢ           = α, α*β/180
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
        m₀           = sin½α₁
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
        MPSV₁[i]     = V[F⋅Mᵢ, 0, Z⋅Mᵢ₊₁] # j = i
        MPSV₁[i+1]   = V[F⋅Mᵢ₊₁, 0, 0] # j = i + 1
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
    @unpack MPSV = work
    @unpack α, TE, T2, T1, β = θ

    @inbounds begin
        # Initialize magnetization phase state vector (MPSV)
        E2, E1  = exp(-TE/T2), exp(-TE/T1)
        E2_half = exp(-(TE/2)/T2)
        α1      = α # T1 flip angle
        α2      = α * (β/180) # T2 flip angle
        m0      = E2_half * sind(α1/2) # initial population
        M1x     =  m0 * cosd(α1/2)^2 # M1x, M1y, M1z are elements resulting from first refocusing pulse applied to [m0, 0, 0]
        M1y     =  m0 - M1x          # M1y = m0 * sind(α1/2)^2 = m0 - m0 * cosd(α1/2)^2 = m0 - M1x
        M1z     = -m0 * sind(α1)/2   # Note: this is the imaginary part
        dc[1]   = E2_half * abs(M1y) # first echo amplitude

        # Apply first relaxation matrix iteration on non-zero states
        MPSV[1] = Vec((E2 * M1y, zero(T)))
        MPSV[2] = zero(Vec{2,T})
        MPSV[3] = Vec((zero(T), E1 * M1z))
        MPSV[4] = Vec((E2 * M1x, zero(T)))

        # Extract matrix elements + initialize temporaries
        a1, a2, a3, a4, a5 = sind(α2), cosd(α2), sind(α2/2)^2, cosd(α2/2)^2, sind(α2)/2 # independent elements of T2mat
        b1, b2, b3, b4, b5 = E2*a1, E1*a2, E2*a3, E2*a4, E1*a5
        c1, c3, c4         = E2_half*a1, E2_half*a3, E2_half*a4
        b1F, b5F, c1F      = Vec((-b1, b1)), Vec((-b5, b5)), Vec((-c1, c1))
        Mz3                = MPSV[3]
    end

    @inbounds for i = 2:ETL-1
        ###########################
        # Unroll first flipmat/relaxmat iteration
        Vx, Vy  = MPSV[1], MPSV[2]
        c1z     = shufflevector(c1F * Mz3, Val((1,0)))
        Mz2     = muladd(c3, Vx, muladd(c4, Vy, -c1z)) # flipmat: 2 -> dc
        Mz4     = muladd(b4, Vx, muladd(b3, Vy, E2_half * c1z)) # relaxmat: 1 -> 4, save in buffer
        dc[i]   = sqrt(sum(Mz2 * Mz2)) # decay curve coefficient
        MPSV[1] = E2_half * Mz2 # relaxmat: 2 -> 1
        b5xy    = shufflevector(b5F * (Vx - Vy), Val((1,0)))
        Mz3     = muladd(b2, Mz3, b5xy) # relaxmat: 3 -> 3, save in buffer

        ###########################
        # flipmat + relaxmat loop
        @inbounds for j in 4:3:3*min(i-1, ETL)
            Vx, Vy, Vz = MPSV[j], MPSV[j+1], MPSV[j+2]
            b1z        = shufflevector(b1F * Vz, Val((1,0)))
            MPSV[j  ]  = Mz4 # relaxmat: assign forward, j -> j+3
            Mz4        = muladd(b4, Vx, muladd(b3, Vy,  b1z))
            MPSV[j-2]  = muladd(b3, Vx, muladd(b4, Vy, -b1z)) # relaxmat: assign backwards, j+1 -> j+1-3
            b5xy       = shufflevector(b5F * (Vx - Vy), Val((1,0)))
            MPSV[j+2]  = muladd(b2, Vz, b5xy) # relaxmat: j+2 -> j+2
        end

        ###########################
        # cleanup + zero next elements
        j         = 3i-2
        Vx        = MPSV[j]
        MPSV[j  ] = Mz4 # relaxmat: assign forward, j -> j+3
        MPSV[j-2] = b3 * Vx # relaxmat: assign backwards, j+1 -> j+1-3
        MPSV[j+2] = shufflevector(b5F * Vx, Val((1,0))) # relaxmat: j+2 -> j+2
        MPSV[j+3] = b4 * Vx # relaxmat: assign forward, j -> j+3
        MPSV[j+1] = Vec((zero(T), zero(T))) # relaxmat: assign backwards, j+1 -> j+1-3
        MPSV[j+5] = Vec((zero(T), zero(T))) # relaxmat: j+2 -> j+2
    end

    ###########################
    # decay curve coefficient
    @inbounds begin
        c1z     = shufflevector(c1F * Mz3, Val((1,0)))
        Mz2     = muladd(c3, MPSV[1], muladd(c4, MPSV[2], -c1z)) # last iteration of flipmat unrolled
        dc[end] = sqrt(sum(Mz2 * Mz2))
    end

    return dc
end

####
#### EPGWork_Cplx
####

struct EPGWork_Cplx{T, ETL, MPSVType <: AbstractVector{Complex{T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV::MPSVType
    dc::DCType
end
function EPGWork_Cplx(T, ETL::Int)
    MSPV = SizedVector{3*ETL,Complex{T}}(undef)
    dc   = SizedVector{ETL,T}(undef)
    EPGWork_Cplx{T,ETL,typeof(MSPV),typeof(dc)}(MSPV, dc)
end

function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_Cplx{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    @unpack MPSV = work
    @unpack α, TE, T2, T1, β = θ

    # Precompute compute element flip matrices and other intermediate variables
    @inbounds begin
        E2, E1  = exp(-TE/T2), exp(-TE/T1)
        E2_half = exp(-(TE/2)/T2)
        T1mat   = element_flipmat(α)
        T2mat   = element_flipmat(α * (β/180))

        # Independent elements of T2mat
        s_α     = -imag(T2mat[1,3]) # sind(α)
        c_α     =  real(T2mat[3,3]) # cosd(α)
        s_½α_sq =  real(T2mat[2,1]) # sind(α/2)^2
        c_½α_sq =  real(T2mat[1,1]) # cosd(α/2)^2
        s_α_½   = -imag(T2mat[3,1]) # sind(α)/2

        # Initialize magnetization phase state vector (MPSV)
        m0      = E2_half * sind(α/2) # initial population
        M0      = SA{T}[m0, 0, 0] # initial magnetization in F1 state
        M1      = T1mat * M0 # apply first refocusing pulse
        dc[1]   = E2_half * sqrt(abs2(M1[2])) # first echo amplitude

        # Apply first relaxation matrix iteration on non-zero states
        MPSV[1] = E2 * M1[2]
        MPSV[2] = 0
        MPSV[3] = E1 * M1[3]
        MPSV[4] = E2 * M1[1]
        MPSV[5] = 0
        MPSV[6] = 0
    end

    @inbounds for i = 2:ETL
        # Perform the flip for all states
        @simd for j in 1:3:3i-2
            Vmx, Vmy, Vmz = MPSV[j], MPSV[j+1], MPSV[j+2]
            ms_α_Vtmp     = s_α * mul_im(Vmz)
            s_α_½_Vtmp    = s_α_½ * mul_im(Vmy - Vmx)
            MPSV[j]       = muladd(c_½α_sq, Vmx, muladd(s_½α_sq, Vmy, -ms_α_Vtmp))
            MPSV[j+1]     = muladd(s_½α_sq, Vmx, muladd(c_½α_sq, Vmy,  ms_α_Vtmp))
            MPSV[j+2]     = muladd(c_α, Vmz, s_α_½_Vtmp)
        end

        # Zero out next elements
        if i+1 < ETL
            j         = 3i+1
            MPSV[j]   = 0
            MPSV[j+1] = 0
            MPSV[j+2] = 0
        end

        # Record the magnitude of the population of F1* as the echo amplitude, allowing for relaxation
        dc[i] = E2_half * sqrt(abs2(MPSV[2]))

        # Allow time evolution of magnetization between pulses
        if (i < ETL)
            # Basic relaxation matrix loop:
            mprev   = MPSV[1]
            MPSV[1] = E2 * MPSV[2] # F1* --> F1
            @simd for j in 2:3:3i-1
                m1, m2, m3 = MPSV[j+1], MPSV[j+2], MPSV[j+3]
                mtmp  = m2
                m0    = E2 * m3     # F(n)* --> F(n-1)*
                m1   *= E1          # Z(n)  --> Z(n)
                m2    = E2 * mprev  # F(n)  --> F(n+1)
                mprev = mtmp
                MPSV[j], MPSV[j+1], MPSV[j+2] = m0, m1, m2
            end
        end
    end

    return dc
end

####
#### EPGWork_Cplx_Vec_Unrolled
####

struct EPGWork_Cplx_Vec_Unrolled{T, ETL, MPSVType <: AbstractVector{Complex{T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV::MPSVType
    dc::DCType
end
function EPGWork_Cplx_Vec_Unrolled(T, ETL::Int)
    MSPV = SizedVector{3*ETL,Complex{T}}(undef)
    dc   = SizedVector{ETL,T}(undef)
    EPGWork_Cplx_Vec_Unrolled{T,ETL,typeof(MSPV),typeof(dc)}(MSPV, dc)
end

@inline function epg_decay_curve!(dc::AbstractVector{T}, work::EPGWork_Cplx_Vec_Unrolled{T,ETL}, θ::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    @unpack MPSV = work
    @unpack α, TE, T2, T1, β = θ

    # Precompute compute element flip matrices and other intermediate variables
    @inbounds begin
        E2, E1  = exp(-TE/T2), exp(-TE/T1)
        E2_half = exp(-(TE/2)/T2)
        T1mat   = element_flipmat(α)
        T2mat   = element_flipmat(α * (β/180))

        # Independent elements of T2mat
        s_α     = -imag(T2mat[1,3]) # sind(α)
        c_α     =  real(T2mat[3,3]) # cosd(α)
        s_½α_sq =  real(T2mat[2,1]) # sind(α/2)^2
        c_½α_sq =  real(T2mat[1,1]) # cosd(α/2)^2
        s_α_½   = -imag(T2mat[3,1]) # sind(α)/2

        # Initialize magnetization phase state vector (MPSV)
        m0      = E2_half * sind(α/2) # initial population
        M0      = SA{T}[m0, 0, 0] # initial magnetization in F1 state
        M1      = T1mat * M0 # apply first refocusing pulse
        dc[1]   = E2_half * sqrt(abs2(M1[2])) # first echo amplitude

        # Apply first relaxation matrix iteration on non-zero states
        MPSV[1] = E2 * M1[2]
        MPSV[2] = 0
        MPSV[3] = E1 * M1[3]
        MPSV[4] = E2 * M1[1]
        MPSV[5] = 0
        MPSV[6] = 0
    end

    @inbounds for i = 2:ETL
        # Twice loop-unrolled flip-matrix loop
        Vs_α   = Vec((-s_α, s_α, -s_α, s_α))
        Vs_α_½ = Vec((-s_α_½, s_α_½, -s_α_½, s_α_½))
        @simd for j in 1:6:3i-5 # 1+(0:3*(n-2))
            Vmx        = Vec((reim(MPSV[j  ])..., reim(MPSV[j+3])...))
            Vmy        = Vec((reim(MPSV[j+1])..., reim(MPSV[j+4])...))
            Vmz        = Vec((reim(MPSV[j+2])..., reim(MPSV[j+5])...))
            s_α_Vtmp   = shufflevector(Vs_α * Vmz, Val((1,0,3,2)))
            s_α_½_Vtmp = shufflevector(Vs_α_½ * (Vmx - Vmy), Val((1,0,3,2)))
            VMx        = muladd(c_½α_sq, Vmx, muladd(s_½α_sq, Vmy,  s_α_Vtmp))
            VMy        = muladd(s_½α_sq, Vmx, muladd(c_½α_sq, Vmy, -s_α_Vtmp))
            VMz        = muladd(c_α, Vmz, s_α_½_Vtmp)
            MPSV[j  ]  = Complex(VMx[1], VMx[2])
            MPSV[j+1]  = Complex(VMy[1], VMy[2])
            MPSV[j+2]  = Complex(VMz[1], VMz[2])
            MPSV[j+3]  = Complex(VMx[3], VMx[4])
            MPSV[j+4]  = Complex(VMy[3], VMy[4])
            MPSV[j+5]  = Complex(VMz[3], VMz[4])
        end
        if isodd(i)
            j = 3i-2
            Vmx, Vmy, Vmz = MPSV[j], MPSV[j+1], MPSV[j+2]
            ms_α_Vtmp     = s_α * mul_im(Vmz)
            s_α_½_Vtmp    = s_α_½ * mul_im(Vmy - Vmx)
            MPSV[j]       = muladd(c_½α_sq, Vmx, muladd(s_½α_sq, Vmy, -ms_α_Vtmp))
            MPSV[j+1]     = muladd(s_½α_sq, Vmx, muladd(c_½α_sq, Vmy,  ms_α_Vtmp))
            MPSV[j+2]     = muladd(c_α, Vmz, s_α_½_Vtmp)
        end

        # Zero out next elements
        if i+1 < ETL
            j         = 3i+1
            MPSV[j]   = 0
            MPSV[j+1] = 0
            MPSV[j+2] = 0
        end

        # Record the magnitude of the population of F1* as the echo amplitude, allowing for relaxation
        dc[i] = E2_half * sqrt(abs2(MPSV[2]))

        if (i < ETL)
            # Twice loop-unrolled relaxmat loop
            mprev   = MPSV[1]
            MPSV[1] = E2 * MPSV[2] # F1* --> F1
            @simd for j in 2:6:3i-4
                m1, m2, m3, m4, m5, m6 = MPSV[j+1], MPSV[j+2], MPSV[j+3], MPSV[j+4], MPSV[j+5], MPSV[j+6]
                m0    = E2 * m3    # F(n)* --> F(n-1)*
                m1   *= E1         # Z(n)  --> Z(n)
                mtmp2 = m2
                m2    = E2 * mprev # F(n)  --> F(n+1)
                m3    = E2 * m6    # F(n)* --> F(n-1)*
                m4   *= E1         # Z(n)  --> Z(n)
                mtmp5 = m5
                m5    = E2 * mtmp2 # F(n)  --> F(n+1)
                mprev = mtmp5
                MPSV[j], MPSV[j+1], MPSV[j+2], MPSV[j+3], MPSV[j+4], MPSV[j+5] = m0, m1, m2, m3, m4, m5
            end
            if isodd(i)
                j   = 3i-1
                m1, m2, m3 = MPSV[j+1], MPSV[j+2], MPSV[j+3]
                m0  = E2 * m3      # F(n)* --> F(n-1)*
                m1 *= E1           # Z(n)  --> Z(n)
                m2  = E2 * mprev   # F(n)  --> F(n+1)
                MPSV[j], MPSV[j+1], MPSV[j+2] = m0, m1, m2
            end
        end
    end

    return dc
end

####
#### Algorithm list
####

const EPGWork_List = [
    EPGWork_Basic_Cplx,
    EPGWork_Cplx,
    if VERSION >= v"1.6-"
        # TODO: SIMD.jl broken on v1.6.0-beta1
        []
    else
        [EPGWork_Vec, EPGWork_Cplx_Vec_Unrolled]
    end...,
    EPGWork_ReIm,
    # TODO: generated function approach is extremely slow to test and provides a small speedup only for ETL <= 16
    # EPGWork_ReIm_Generated,
    EPGWork_ReIm_DualCache,
    EPGWork_ReIm_DualCache_Split,
    EPGWork_ReIm_DualCache_Unrolled,
    EPGWork_ReIm_DualMVector_Split,
]
