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

struct EPGOptions{T,ETL}
    flip_angle::T
    TE::T
    T2::T
    T1::T
    refcon::T
end
@inline function EPGOptions(ETL::Int, flip_angle::Real, TE::Real, T2::Real, T1::Real, refcon::Real)
    T = float(promote_type(typeof(flip_angle), typeof(TE), typeof(T2), typeof(T1), typeof(refcon)))
    EPGOptions{T,ETL}(flip_angle, TE, T2, T1, refcon)
end

@inline EPGdecaycurve_work(::EPGOptions{T,ETL}) where {T,ETL} = EPGdecaycurve_work(T, ETL)
@inline EPGdecaycurve_work(::Type{T}, ETL::Int) where {T} = EPGWork_ReIm_DualCache_Split(T, ETL) # fallback
@inline EPGdecaycurve_work(::Type{T}, ETL::Int) where {T <: FloatingTypes} = EPGWork_ReIm_DualCache_Split(T, ETL) # default for T <: SIMD.FloatingTypes

"""
    EPGdecaycurve(ETL::Int, flip_angle::Real, TE::Real, T2::Real, T1::Real, refcon::Real)

Computes the normalized echo decay curve for a MR spin echo sequence
using the extended phase graph algorithm using the given input parameters.

# Arguments
- `ETL::Int`:         echo train length, i.e. number of echos
- `flip_angle::Real`: angle of refocusing pulses (Units: degrees)
- `TE::Real`:         inter-echo time (Units: seconds)
- `T2::Real`:         transverse relaxation time (Units: seconds)
- `T1::Real`:         longitudinal relaxation time (Units: seconds)
- `refcon::Real`:     value of Refocusing Pulse Control Angle (Units: degrees)

# Outputs
- `decay_curve::AbstractVector`: normalized echo decay curve with length `ETL`
"""
@inline EPGdecaycurve(ETL::Int, args::Real...) = EPGdecaycurve(EPGOptions(ETL, args...))
@inline EPGdecaycurve(o::EPGOptions{T,ETL}) where {T,ETL} = EPGdecaycurve!(EPGdecaycurve_work(o), o)
@inline EPGdecaycurve!(work::AbstractEPGWorkspace{T,ETL}, args::Real...) where {T,ETL} = EPGdecaycurve!(work.decay_curve, work, EPGOptions{T,ETL}(args...))
@inline EPGdecaycurve!(work::AbstractEPGWorkspace{T,ETL}, o::EPGOptions{T,ETL}) where {T,ETL} = EPGdecaycurve!(work.decay_curve, work, o)
@inline EPGdecaycurve!(decay_curve::AbstractVector{T}, work::AbstractEPGWorkspace{T,ETL}, args::Real...) where {T,ETL} = EPGdecaycurve!(decay_curve, work, EPGOptions{T,ETL}(args...))
@inline EPGdecaycurve!(decay_curve::AbstractVector{T}, work::AbstractEPGWorkspace{T,ETL}, o::EPGOptions{T,ETL}) where {T,ETL} = @timeit_debug TIMER() "EPG decay curve" epg_decay_curve!(decay_curve, work, o)

####
#### EPGWork_Basic_Cplx
####

struct EPGWork_Basic_Cplx{T, ETL, MPSVType <: AbstractVector{SVector{3,Complex{T}}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV::MPSVType
    decay_curve::DCType
end
function EPGWork_Basic_Cplx(T, ETL::Int)
    mpsv = SizedVector{ETL,SVector{3,Complex{T}}}(undef)
    dc = SizedVector{ETL,T}(undef)
    EPGWork_Basic_Cplx{T,ETL,typeof(mpsv),typeof(dc)}(mpsv, dc)
end

# Compute a basis function under the extended phase graph algorithm. The magnetization phase state vector (MPSV) is
# successively modified by applying relaxation for TE/2, then a refocusing pulse as described by Hennig (1988),
# then transitioning phase states as given by Hennig (1988) but corrected by Jones (1997), and a finally relaxing for TE/2.
# See the appendix in Prasloski (2012) for details:
#    https://doi.org/10.1002/mrm.23157

function epg_decay_curve!(decay_curve::AbstractVector{T}, work::EPGWork_Basic_Cplx{T,ETL}, o::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    @unpack MPSV = work
    @unpack flip_angle, TE, T2, T1, refcon = o
    αₑₓ, α₁, αⱼ = flip_angle/2, flip_angle, flip_angle * refcon / 180
    V = SA{Complex{T}} # alias

    # Precompute compute element flip matrices and other intermediate variables
    E1, E2 = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
    E  = SA{T}[E2, E2, E1]
    R₁ = element_flipmat(α₁)
    Rⱼ = element_flipmat(αⱼ)

    # Initialize magnetization phase state vector (MPSV)
    @inbounds for j in 1:ETL
        MPSV[j] = V[0, 0, 0]
    end
    @inbounds MPSV[1] = V[sind(αₑₓ), 0, 0] # initial magnetization in F1 state

    @inbounds for i in 1:ETL
        R = i == 1 ? R₁ : Rⱼ
        jend = min(i+1, ETL)
        for j in 1:jend
            MPSV[j] = R * (E .* MPSV[j]) # Relaxation for TE/2 and apply flip matrix
        end
        Mlast = MPSV[1]
        MPSV[1] = V[MPSV[1][2], MPSV[2][2], MPSV[1][3]] # (F₁, F₁*, Z₁)⁺ = (F₁*, F₂*, Z₁)
        for j in 2:i
            Mlast, MPSV[j] = MPSV[j], V[Mlast[1], MPSV[j+1][2], MPSV[j][3]] # (Fⱼ, Fⱼ*, Zⱼ)⁺ = (Fⱼ₋₁, Fⱼ₊₁*, Zⱼ)
        end
        MPSV[jend] = V[Mlast[1], 0, MPSV[jend][3]] # (Fₙ, Fₙ*, Zₙ)⁺ = (Fₙ₋₁, 0, Zₙ)
        for j in 1:jend
            MPSV[j] = E .* MPSV[j] # Relaxation for TE/2
        end
        decay_curve[i] = abs(MPSV[1][1]) # first echo amplitude
    end

    return decay_curve
end

####
#### EPGWork_ReIm
####

struct EPGWork_ReIm{T, ETL, MPSVType <: AbstractVector{SVector{3,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV::MPSVType
    decay_curve::DCType
end
function EPGWork_ReIm(T, ETL::Int)
    mpsv = SizedVector{ETL,SVector{3,T}}(undef)
    dc = SizedVector{ETL,T}(undef)
    EPGWork_ReIm{T,ETL,typeof(mpsv),typeof(dc)}(mpsv, dc)
end

function epg_decay_curve!(decay_curve::AbstractVector{T}, work::EPGWork_ReIm{T,ETL}, o::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    @unpack MPSV = work
    @unpack flip_angle, TE, T2, T1, refcon = o
    α = deg2rad(flip_angle)
    α₁, αⱼ = α, α*refcon/180

    # Precompute intermediate variables
    V = SA{T} # alias
    E₁, E₂ = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
    sin½α₁, cos½α₁ = sincos(α₁/2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2*sin½α₁*cos½α₁
    sinαⱼ, cosαⱼ = sincos(αⱼ)
    cos²½αⱼ = (1+cosαⱼ)/2
    sin²½αⱼ = 1-cos²½αⱼ
    a₁, b₁, c₁     = E₂^2*cos²½α₁, E₂^2*sin²½α₁, E₁*E₂*sinα₁
    aⱼ, bⱼ, cⱼ, dⱼ = E₂^2*cos²½αⱼ, E₂^2*sin²½αⱼ, E₁*E₂*sinαⱼ, E₁^2*cosαⱼ
    F, F̄, Z = V[aⱼ, bⱼ, cⱼ], V[bⱼ, aⱼ, -cⱼ], V[-cⱼ/2, cⱼ/2, dⱼ]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁
        Mⱼ⁺ = V[b₁*m₀, 0, -c₁*m₀/2]
        decay_curve[1] = abs(Mⱼ⁺[1])
        MPSV[1] = Mⱼ⁺
        MPSV[2] = V[a₁*m₀, 0, 0]
        MPSV[3] = V[0, 0, 0]
    end

    @inbounds for i in 2:ETL-1
        # j = 1, initialize and update `decay_curve`
        Mⱼ, Mⱼ₊₁ = MPSV[1], MPSV[2]
        Mⱼ⁺ = V[F̄⋅Mⱼ, F̄⋅Mⱼ₊₁, Z⋅Mⱼ]
        decay_curve[i] = abs(Mⱼ⁺[1])
        MPSV[1] = Mⱼ⁺

        # inner loop
        jup = min(i, ETL-i)
        @simd for j in 2:jup
            Mⱼ₋₁, Mⱼ, Mⱼ₊₁ = Mⱼ, Mⱼ₊₁, MPSV[j+1]
            MPSV[j] = V[F⋅Mⱼ₋₁, F̄⋅Mⱼ₊₁, Z⋅Mⱼ]
        end

        # cleanup for next iteration
        if i == jup
            Mⱼ₋₁ = Mⱼ
            MPSV[i+1] = V[F⋅Mⱼ₋₁, 0, 0]
            MPSV[i+2] = V[0, 0, 0]
        end
    end

    @inbounds decay_curve[ETL] = abs(F̄⋅MPSV[1])

    return decay_curve
end

####
#### EPGWork_ReIm_DualCache
####

struct EPGWork_ReIm_DualCache{T, ETL, MPSVType <: AbstractVector{SVector{3,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    decay_curve::DCType
end
function EPGWork_ReIm_DualCache(T, ETL::Int)
    mpsv₁ = SizedVector{ETL,SVector{3,T}}(undef)
    mpsv₂ = SizedVector{ETL,SVector{3,T}}(undef)
    dc = SizedVector{ETL,T}(undef)
    EPGWork_ReIm_DualCache{T,ETL,typeof(mpsv₁),typeof(dc)}(mpsv₁, mpsv₂, dc)
end

function epg_decay_curve!(decay_curve::AbstractVector{T}, work::EPGWork_ReIm_DualCache{T,ETL}, o::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    @unpack MPSV₁, MPSV₂ = work
    @unpack flip_angle, TE, T2, T1, refcon = o
    α = deg2rad(flip_angle)
    α₁, αⱼ = α, α*refcon/180

    # Precompute intermediate variables
    V = SA{T} # alias
    E₁, E₂ = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
    sin½α₁, cos½α₁ = sincos(α₁/2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2*sin½α₁*cos½α₁
    sinαⱼ, cosαⱼ = sincos(αⱼ)
    cos²½αⱼ = (1+cosαⱼ)/2
    sin²½αⱼ = 1-cos²½αⱼ
    a₁, b₁, c₁     = E₂^2*cos²½α₁, E₂^2*sin²½α₁, E₁*E₂*sinα₁
    aⱼ, bⱼ, cⱼ, dⱼ = E₂^2*cos²½αⱼ, E₂^2*sin²½αⱼ, E₁*E₂*sinαⱼ, E₁^2*cosαⱼ
    F, F̄, Z = V[aⱼ, bⱼ, cⱼ], V[bⱼ, aⱼ, -cⱼ], V[-cⱼ/2, cⱼ/2, dⱼ]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁
        Mⱼ⁺ = V[b₁*m₀, 0, -c₁*m₀/2]
        decay_curve[1] = abs(Mⱼ⁺[1])
        MPSV₁[1] = Mⱼ⁺
        MPSV₁[2] = V[a₁*m₀, 0, 0]
        MPSV₁[3] = V[0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 2:ETL-1
        # j = 1, initialize and update `decay_curve`
        Mⱼ, Mⱼ₊₁ = MPSV₂[1], MPSV₂[2]
        Mⱼ⁺ = V[F̄⋅Mⱼ, F̄⋅Mⱼ₊₁, Z⋅Mⱼ]
        decay_curve[i] = abs(Mⱼ⁺[1])
        MPSV₁[1] = Mⱼ⁺

        # inner loop
        jup = min(i, ETL-i)
        @simd for j in 2:jup
            Mⱼ₋₁, Mⱼ, Mⱼ₊₁ = Mⱼ, Mⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⋅Mⱼ₋₁, F̄⋅Mⱼ₊₁, Z⋅Mⱼ]
        end

        # cleanup for next iteration
        if i == jup
            Mⱼ₋₁ = Mⱼ
            MPSV₁[i+1] = V[F⋅Mⱼ₋₁, 0, 0]
            MPSV₁[i+2] = V[0, 0, 0]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds decay_curve[ETL] = abs(F̄⋅MPSV₂[1])

    return decay_curve
end

####
#### EPGWork_ReIm_DualCache_Split
####

struct EPGWork_ReIm_DualCache_Split{T, ETL, MPSVType <: AbstractVector{SVector{3,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    decay_curve::DCType
end
function EPGWork_ReIm_DualCache_Split(T, ETL::Int)
    mpsv₁ = SizedVector{ETL,SVector{3,T}}(undef)
    mpsv₂ = SizedVector{ETL,SVector{3,T}}(undef)
    dc = SizedVector{ETL,T}(undef)
    EPGWork_ReIm_DualCache_Split{T,ETL,typeof(mpsv₁),typeof(dc)}(mpsv₁, mpsv₂, dc)
end

function epg_decay_curve!(decay_curve::AbstractVector{T}, work::EPGWork_ReIm_DualCache_Split{T,ETL}, o::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    @unpack MPSV₁, MPSV₂ = work
    @unpack flip_angle, TE, T2, T1, refcon = o
    α = deg2rad(flip_angle)
    α₁, αⱼ = α, α*refcon/180

    # Precompute intermediate variables
    V = SA{T} # alias
    E₁, E₂ = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
    sin½α₁, cos½α₁ = sincos(α₁/2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2*sin½α₁*cos½α₁
    sinαⱼ, cosαⱼ = sincos(αⱼ)
    cos²½αⱼ = (1+cosαⱼ)/2
    sin²½αⱼ = 1-cos²½αⱼ
    a₁, b₁, c₁     = E₂^2*cos²½α₁, E₂^2*sin²½α₁, E₁*E₂*sinα₁
    aⱼ, bⱼ, cⱼ, dⱼ = E₂^2*cos²½αⱼ, E₂^2*sin²½αⱼ, E₁*E₂*sinαⱼ, E₁^2*cosαⱼ
    F, F̄, Z = V[aⱼ, bⱼ, cⱼ], V[bⱼ, aⱼ, -cⱼ], V[-cⱼ/2, cⱼ/2, dⱼ]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁
        Mⱼ⁺ = V[b₁*m₀, 0, -c₁*m₀/2]
        decay_curve[1] = abs(Mⱼ⁺[1])
        MPSV₁[1] = Mⱼ⁺
        MPSV₁[2] = V[a₁*m₀, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 2:ETL÷2
        Mⱼ, Mⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `decay_curve`
        Mⱼ⁺ = V[F̄⋅Mⱼ, F̄⋅Mⱼ₊₁, Z⋅Mⱼ]
        decay_curve[i] = abs(Mⱼ⁺[1])
        MPSV₁[1] = Mⱼ⁺
        @simd for j in 2:i-1
            Mⱼ₋₁, Mⱼ, Mⱼ₊₁ = Mⱼ, Mⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⋅Mⱼ₋₁, F̄⋅Mⱼ₊₁, Z⋅Mⱼ]
        end
        MPSV₁[i]   = V[F⋅Mⱼ, 0, Z⋅Mⱼ₊₁] # j = i
        MPSV₁[i+1] = V[F⋅Mⱼ₊₁, 0, 0] # j = i + 1
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in ETL÷2+1:ETL-1
        Mⱼ, Mⱼ₊₁ = MPSV₂[1], MPSV₂[2] # j = 1, initialize and update `decay_curve`
        Mⱼ⁺ = V[F̄⋅Mⱼ, F̄⋅Mⱼ₊₁, Z⋅Mⱼ]
        decay_curve[i] = abs(Mⱼ⁺[1])
        MPSV₁[1] = Mⱼ⁺
        @simd for j in 2:ETL-i
            Mⱼ₋₁, Mⱼ, Mⱼ₊₁ = Mⱼ, Mⱼ₊₁, MPSV₂[j+1]
            MPSV₁[j] = V[F⋅Mⱼ₋₁, F̄⋅Mⱼ₊₁, Z⋅Mⱼ]
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    #= Cleaner, but slightly slower
    @inbounds for i in 2:ETL-1
        # j = 1, initialize and update `decay_curve`
        Mⱼ, Mⱼ₊₁ = MPSV₂[1], MPSV₂[2]
        Mⱼ⁺ = V[F̄⋅Mⱼ, F̄⋅Mⱼ₊₁, Z⋅Mⱼ]
        decay_curve[i] = abs(Mⱼ⁺[1])
        MPSV₁[1] = Mⱼ⁺
        if i <= ETL÷2
            # inner loop
            @simd for j in 2:i-1
                Mⱼ₋₁, Mⱼ, Mⱼ₊₁ = Mⱼ, Mⱼ₊₁, MPSV₂[j+1]
                MPSV₁[j] = V[F⋅Mⱼ₋₁, F̄⋅Mⱼ₊₁, Z⋅Mⱼ]
            end
            MPSV₁[i]   = V[F⋅Mⱼ, 0, Z⋅Mⱼ₊₁] # j = i
            MPSV₁[i+1] = V[F⋅Mⱼ₊₁, 0, 0] # j = i + 1
        else
            # inner loop
            @simd for j in 2:ETL-i
                Mⱼ₋₁, Mⱼ, Mⱼ₊₁ = Mⱼ, Mⱼ₊₁, MPSV₂[j+1]
                MPSV₁[j] = V[F⋅Mⱼ₋₁, F̄⋅Mⱼ₊₁, Z⋅Mⱼ]
            end
        end
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end
    =#

    @inbounds decay_curve[ETL] = abs(F̄⋅MPSV₂[1])

    return decay_curve
end

####
#### EPGWork_ReIm_DualCache_Unrolled
####

struct EPGWork_ReIm_DualCache_Unrolled{T, ETL, MPSVType <: AbstractVector{SVector{3,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV₁::MPSVType
    MPSV₂::MPSVType
    decay_curve::DCType
end
function EPGWork_ReIm_DualCache_Unrolled(T, ETL::Int)
    mpsv₁ = SizedVector{ETL,SVector{3,T}}(undef)
    mpsv₂ = SizedVector{ETL,SVector{3,T}}(undef)
    dc = SizedVector{ETL,T}(undef)
    EPGWork_ReIm_DualCache_Unrolled{T,ETL,typeof(mpsv₁),typeof(dc)}(mpsv₁, mpsv₂, dc)
end

function epg_decay_curve!(decay_curve::AbstractVector{T}, work::EPGWork_ReIm_DualCache_Unrolled{T,ETL}, o::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    @unpack MPSV₁, MPSV₂ = work
    @unpack flip_angle, TE, T2, T1, refcon = o
    α = deg2rad(flip_angle)
    α₁, αⱼ = α, α*refcon/180

    # Precompute intermediate variables
    V = SA{T} # alias
    E₁, E₂ = exp(-(TE/2)/T1), exp(-(TE/2)/T2)
    sin½α₁, cos½α₁ = sincos(α₁/2)
    sin²½α₁, cos²½α₁ = sin½α₁^2, cos½α₁^2
    sinα₁ = 2*sin½α₁*cos½α₁
    sinαⱼ, cosαⱼ = sincos(αⱼ)
    cos²½αⱼ = (1+cosαⱼ)/2
    sin²½αⱼ = 1-cos²½αⱼ
    a₁, b₁, c₁     = E₂^2*cos²½α₁, E₂^2*sin²½α₁, E₁*E₂*sinα₁
    aⱼ, bⱼ, cⱼ, dⱼ = E₂^2*cos²½αⱼ, E₂^2*sin²½αⱼ, E₁*E₂*sinαⱼ, E₁^2*cosαⱼ
    F, F̄, Z = V[aⱼ, bⱼ, cⱼ], V[bⱼ, aⱼ, -cⱼ], V[-cⱼ/2, cⱼ/2, dⱼ]

    # Initialize magnetization phase state vector (MPSV), pulling i=1 iteration out of loop
    @inbounds begin
        m₀ = sin½α₁
        M₁  = V[b₁*m₀, 0, -c₁*m₀/2]
        M₂  = V[a₁*m₀, 0, 0]
        M₁⁺ = V[F̄⋅M₁, F̄⋅M₂, Z⋅M₁] # V[F̄⋅Mⱼ,   F̄⋅Mⱼ₊₁, Z⋅Mⱼ], j=1
        M₂⁺ = V[F⋅M₁, 0, Z⋅M₂]    # V[F⋅Mⱼ₋₁, F̄⋅Mⱼ₊₁, Z⋅Mⱼ], j=2, M₃ = 0
        M₃⁺ = V[F⋅M₂, 0, 0]       # V[F⋅Mⱼ₋₁, F̄⋅Mⱼ₊₁, Z⋅Mⱼ], j=3, M₃ = M₄ = 0

        decay_curve[1] = abs(M₁[1])
        decay_curve[2] = abs(M₁⁺[1])
        MPSV₁[1] = M₁⁺
        MPSV₁[2] = M₂⁺
        MPSV₁[3] = M₃⁺
        MPSV₁[4] = V[0, 0, 0]
        MPSV₁[5] = V[0, 0, 0]
        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds for i in 3:ETL-1
        # j = 1, j = 2, and update `decay_curve`
        Mⱼ, Mⱼ₊₁, Mⱼ₊₂ = MPSV₂[1], MPSV₂[2], MPSV₂[3]
        Mⱼ⁺   = V[F̄⋅Mⱼ, F̄⋅Mⱼ₊₁, Z⋅Mⱼ]
        Mⱼ₊₁⁺ = V[F⋅Mⱼ, F̄⋅Mⱼ₊₂, Z⋅Mⱼ₊₁]
        decay_curve[i] = abs(Mⱼ⁺[1])
        MPSV₁[1] = Mⱼ⁺
        MPSV₁[2] = Mⱼ₊₁⁺

        # inner loop
        jup = min(i, ETL-i)
        @simd for j in 3:2:jup#-isodd(jup)
            Mⱼ₋₁, Mⱼ, Mⱼ₊₁, Mⱼ₊₂ = Mⱼ₊₁, Mⱼ₊₂, MPSV₂[j+1], MPSV₂[j+2]
            MPSV₁[j]   = V[F⋅Mⱼ₋₁, F̄⋅Mⱼ₊₁, Z⋅Mⱼ]
            MPSV₁[j+1] = V[F⋅Mⱼ,   F̄⋅Mⱼ₊₂, Z⋅Mⱼ₊₁]
        end

        # cleanup for next iteration
        if i == jup
            if iseven(jup)
                Mⱼ₋₁ = Mⱼ₊₁
                MPSV₁[jup+1] = V[F⋅Mⱼ₋₁, 0, 0]
                MPSV₁[jup+2] = V[0, 0, 0]
                MPSV₁[jup+3] = V[0, 0, 0]
            else
                # inner loop: for j in 3:2:jup
                # MPSV₁[jup+1] = V[F⋅Mⱼ, 0, 0] # (with or without)
                MPSV₁[jup+2] = V[0, 0, 0]

                # # (not working) inner loop: for j in 3:2:jup-isodd
                # Mⱼ₋₁, Mⱼ = Mⱼ₊₁, Mⱼ₊₂
                # MPSV₁[jup]   = V[F⋅Mⱼ₋₁, 0, Z⋅Mⱼ]
                # MPSV₁[jup+1] = V[F⋅Mⱼ, 0, 0]
                # MPSV₁[jup+2] = V[0, 0, 0]
                # MPSV₁[jup+3] = V[0, 0, 0]
            end
        end

        MPSV₁, MPSV₂ = MPSV₂, MPSV₁
    end

    @inbounds decay_curve[ETL] = abs(F̄⋅MPSV₂[1])

    return decay_curve
end

####
#### EPGWork_Vec
####

# Flip matrix and relaxation matrix steps are combined into one loop, and SIMD.jl `Vec` types are used instead of `Complex`.
# As this function is called many times during T2mapSEcorr, the micro-optimizations are worth the loss of code readability.
# See `EPGWork_Basic_Cplx` for a more readable, mathematically identicaly implementation.

struct EPGWork_Vec{T, ETL, MPSVType <: AbstractVector{Vec{2,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV::MPSVType
    decay_curve::DCType
end
function EPGWork_Vec(T, ETL::Int)
    mpsv = SizedVector{3*ETL,Vec{2,T}}(undef)
    dc = SizedVector{ETL,T}(undef)
    EPGWork_Vec{T,ETL,typeof(mpsv),typeof(dc)}(mpsv, dc)
end

function epg_decay_curve!(decay_curve::AbstractVector{T}, work::EPGWork_Vec{T,ETL}, o::EPGOptions{T,ETL}) where {T,ETL}
    ###########################
    # Setup
    @unpack MPSV = work
    @unpack flip_angle, TE, T2, T1, refcon = o

    # Initialize magnetization phase state vector (MPSV)
    E2, E1, E2_half = exp(-TE/T2), exp(-TE/T1), exp(-(TE/2)/T2)
    α1  = flip_angle # T1 flip angle
    α2  = flip_angle * (refcon/180) # T2 flip angle
    m0  = E2_half * sind(α1/2) # initial population
    M1x =  m0 * cosd(α1/2)^2 # M1x, M1y, M1z are elements resulting from first refocusing pulse applied to [m0, 0, 0]
    M1y =  m0 - M1x          # M1y = m0 * sind(α1/2)^2 = m0 - m0 * cosd(α1/2)^2 = m0 - M1x
    M1z = -m0 * sind(α1)/2   # Note: this is the imaginary part
    @inbounds decay_curve[1] = E2_half * abs(M1y) # first echo amplitude

    # Apply first relaxation matrix iteration on non-zero states
    @inbounds begin
        MPSV[1] = Vec((E2 * M1y, zero(T)))
        MPSV[2] = zero(Vec{2,T})
        MPSV[3] = Vec((zero(T), E1 * M1z))
        MPSV[4] = Vec((E2 * M1x, zero(T)))
    end

    ###########################
    # Extract matrix elements + initialize temporaries
    a1, a2, a3, a4, a5 = sind(α2), cosd(α2), sind(α2/2)^2, cosd(α2/2)^2, sind(α2)/2 # independent elements of T2mat
    b1, b2, b3, b4, b5 = E2*a1, E1*a2, E2*a3, E2*a4, E1*a5
    c1, c3, c4 = E2_half*a1, E2_half*a3, E2_half*a4
    b1F, b5F, c1F = Vec((-b1, b1)), Vec((-b5, b5)), Vec((-c1, c1))
    @inbounds Mz3 = MPSV[3]

    @inbounds for i = 2:ETL-1
        ###########################
        # Unroll first flipmat/relaxmat iteration
        Vx, Vy = MPSV[1], MPSV[2]
        c1z = shufflevector(c1F * Mz3, Val((1,0)))
        Mz2 = muladd(c3, Vx, muladd(c4, Vy, -c1z)) # flipmat: 2 -> decay_curve
        Mz4 = muladd(b4, Vx, muladd(b3, Vy, E2_half * c1z)) # relaxmat: 1 -> 4, save in buffer

        ###########################
        # decay_curve curve coefficient
        decay_curve[i] = sqrt(sum(Mz2 * Mz2))

        # Unrolled flipmat/relaxmat iteration
        MPSV[1] = E2_half * Mz2 # relaxmat: 2 -> 1
        b5xy  = shufflevector(b5F * (Vx - Vy), Val((1,0)))
        Mz3   = muladd(b2, Mz3, b5xy) # relaxmat: 3 -> 3, save in buffer

        ###########################
        # flipmat + relaxmat loop
        @inbounds for j in 4:3:3*min(i-1, ETL)
            Vx, Vy, Vz = MPSV[j], MPSV[j+1], MPSV[j+2]
            b1z     = shufflevector(b1F * Vz, Val((1,0)))
            MPSV[j  ] = Mz4 # relaxmat: assign forward, j -> j+3
            Mz4     = muladd(b4, Vx, muladd(b3, Vy,  b1z))
            MPSV[j-2] = muladd(b3, Vx, muladd(b4, Vy, -b1z)) # relaxmat: assign backwards, j+1 -> j+1-3
            b5xy    = shufflevector(b5F * (Vx - Vy), Val((1,0)))
            MPSV[j+2] = muladd(b2, Vz, b5xy) # relaxmat: j+2 -> j+2
        end

        ###########################
        # cleanup + zero next elements
        j  = 3i-2
        Vx = MPSV[j]
        MPSV[j  ] = Mz4 # relaxmat: assign forward, j -> j+3
        MPSV[j-2] = b3 * Vx # relaxmat: assign backwards, j+1 -> j+1-3
        MPSV[j+2] = shufflevector(b5F * Vx, Val((1,0))) # relaxmat: j+2 -> j+2
        MPSV[j+3] = b4 * Vx # relaxmat: assign forward, j -> j+3
        MPSV[j+1] = Vec((zero(T), zero(T))) # relaxmat: assign backwards, j+1 -> j+1-3
        MPSV[j+5] = Vec((zero(T), zero(T))) # relaxmat: j+2 -> j+2
    end

    ###########################
    # decay_curve curve coefficient
    c1z = shufflevector(c1F * Mz3, Val((1,0)))
    @inbounds Mz2 = muladd(c3, MPSV[1], muladd(c4, MPSV[2], -c1z)) # last iteration of flipmat unrolled
    @inbounds decay_curve[end] = sqrt(sum(Mz2 * Mz2))

    return decay_curve
end

####
#### EPGWork_Cplx
####

struct EPGWork_Cplx{T, ETL, MPSVType <: AbstractVector{Complex{T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV::MPSVType
    decay_curve::DCType
end
function EPGWork_Cplx(T, ETL::Int)
    mpsv = SizedVector{3*ETL,Complex{T}}(undef)
    dc = SizedVector{ETL,T}(undef)
    EPGWork_Cplx{T,ETL,typeof(mpsv),typeof(dc)}(mpsv, dc)
end

function epg_decay_curve!(decay_curve::AbstractVector{T}, work::EPGWork_Cplx{T,ETL}, o::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    @unpack MPSV = work
    @unpack flip_angle, TE, T2, T1, refcon = o

    # Precompute compute element flip matrices and other intermediate variables
    E2, E1, E2_half = exp(-TE/T2), exp(-TE/T1), exp(-(TE/2)/T2)
    T1mat = element_flipmat(flip_angle)
    T2mat = element_flipmat(flip_angle * (refcon/180))
    @inbounds T2mat_elements = ( # independent elements of T2mat
       -imag(T2mat[1,3]), # sind(α)
        real(T2mat[3,3]), # cosd(α)
        real(T2mat[2,1]), # sind(α/2)^2
        real(T2mat[1,1]), # cosd(α/2)^2
       -imag(T2mat[3,1]), # sind(α)/2
    )

    # Initialize magnetization phase state vector (MPSV)
    m0 = E2_half * sind(flip_angle/2) # initial population
    M0 = SA{T}[m0, 0, 0] # initial magnetization in F1 state
    M1 = T1mat * M0 # apply first refocusing pulse
    @inbounds decay_curve[1] = E2_half * sqrt(abs2(M1[2])) # first echo amplitude

    # Apply first relaxation matrix iteration on non-zero states
    @inbounds begin
        MPSV[1] = E2 * M1[2]
        MPSV[2] = 0
        MPSV[3] = E1 * M1[3]
        MPSV[4] = E2 * M1[1]
        MPSV[5] = 0
        MPSV[6] = 0
    end

    s_α, c_α, s_½α_sq, c_½α_sq, s_α_½ = T2mat_elements
    @inbounds for i = 2:ETL
        # Perform the flip for all states
        @inbounds @simd for j in 1:3:3*i-2
            Vmx, Vmy, Vmz = MPSV[j], MPSV[j+1], MPSV[j+2]
            ms_α_Vtmp  = s_α * mul_im(Vmz)
            s_α_½_Vtmp = s_α_½ * mul_im(Vmy - Vmx)
            MPSV[j]      = muladd(c_½α_sq, Vmx, muladd(s_½α_sq, Vmy, -ms_α_Vtmp))
            MPSV[j+1]    = muladd(s_½α_sq, Vmx, muladd(c_½α_sq, Vmy,  ms_α_Vtmp))
            MPSV[j+2]    = muladd(c_α, Vmz, s_α_½_Vtmp)
        end

        # Zero out next elements
        if i+1 < ETL
            j = 3*i+1
            @inbounds MPSV[j]   = 0
            @inbounds MPSV[j+1] = 0
            @inbounds MPSV[j+2] = 0
        end

        # Record the magnitude of the population of F1* as the echo amplitude, allowing for relaxation
        decay_curve[i] = E2_half * sqrt(abs2(MPSV[2]))

        # Allow time evolution of magnetization between pulses
        if (i < ETL)
            # Basic relaxation matrix loop:
            @inbounds mprev = MPSV[1]
            @inbounds MPSV[1] = E2 * MPSV[2] # F1* --> F1
            @inbounds @simd for j in 2:3:3i-1
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

    return decay_curve
end

####
#### EPGWork_Cplx_Vec_Unrolled
####

struct EPGWork_Cplx_Vec_Unrolled{T, ETL, MPSVType <: AbstractVector{Complex{T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    MPSV::MPSVType
    decay_curve::DCType
end
function EPGWork_Cplx_Vec_Unrolled(T, ETL::Int)
    mpsv = SizedVector{3*ETL,Complex{T}}(undef)
    dc = SizedVector{ETL,T}(undef)
    EPGWork_Cplx_Vec_Unrolled{T,ETL,typeof(mpsv),typeof(dc)}(mpsv, dc)
end

@inline function epg_decay_curve!(decay_curve::AbstractVector{T}, work::EPGWork_Cplx_Vec_Unrolled{T,ETL}, o::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    @unpack MPSV = work
    @unpack flip_angle, TE, T2, T1, refcon = o

    # Precompute compute element flip matrices and other intermediate variables
    E2, E1, E2_half = exp(-TE/T2), exp(-TE/T1), exp(-(TE/2)/T2)
    T1mat = element_flipmat(flip_angle)
    T2mat = element_flipmat(flip_angle * (refcon/180))
    @inbounds T2mat_elements = ( # independent elements of T2mat
       -imag(T2mat[1,3]), # sind(α)
        real(T2mat[3,3]), # cosd(α)
        real(T2mat[2,1]), # sind(α/2)^2
        real(T2mat[1,1]), # cosd(α/2)^2
       -imag(T2mat[3,1]), # sind(α)/2
    )

    # Initialize magnetization phase state vector (MPSV)
    m0 = E2_half * sind(flip_angle/2) # initial population
    M0 = SA{T}[m0, 0, 0] # initial magnetization in F1 state
    M1 = T1mat * M0 # apply first refocusing pulse
    @inbounds decay_curve[1] = E2_half * sqrt(abs2(M1[2])) # first echo amplitude

    # Apply first relaxation matrix iteration on non-zero states
    @inbounds begin
        MPSV[1] = E2 * M1[2]
        MPSV[2] = 0
        MPSV[3] = E1 * M1[3]
        MPSV[4] = E2 * M1[1]
        MPSV[5] = 0
        MPSV[6] = 0
    end

    @inbounds for i = 2:ETL
        # Twice loop-unrolled flip-matrix loop
        s_α, c_α, s_½α_sq, c_½α_sq, s_α_½ = T2mat_elements
        Vs_α = Vec((-s_α, s_α, -s_α, s_α))
        Vs_α_½ = Vec((-s_α_½, s_α_½, -s_α_½, s_α_½))
        @inbounds @simd for j in 1:6:3*i-5 # 1+(0:3*(n-2))
            Vmx = Vec((reim(MPSV[j  ])..., reim(MPSV[j+3])...))
            Vmy = Vec((reim(MPSV[j+1])..., reim(MPSV[j+4])...))
            Vmz = Vec((reim(MPSV[j+2])..., reim(MPSV[j+5])...))
            s_α_Vtmp   = shufflevector(Vs_α * Vmz, Val((1,0,3,2)))
            s_α_½_Vtmp = shufflevector(Vs_α_½ * (Vmx - Vmy), Val((1,0,3,2)))
            VMx        = muladd(c_½α_sq, Vmx, muladd(s_½α_sq, Vmy,  s_α_Vtmp))
            VMy        = muladd(s_½α_sq, Vmx, muladd(c_½α_sq, Vmy, -s_α_Vtmp))
            VMz        = muladd(c_α, Vmz, s_α_½_Vtmp)
            MPSV[j  ]    = Complex(VMx[1], VMx[2])
            MPSV[j+1]    = Complex(VMy[1], VMy[2])
            MPSV[j+2]    = Complex(VMz[1], VMz[2])
            MPSV[j+3]    = Complex(VMx[3], VMx[4])
            MPSV[j+4]    = Complex(VMy[3], VMy[4])
            MPSV[j+5]    = Complex(VMz[3], VMz[4])
        end
        if isodd(i)
            @inbounds begin
                j = 3*i-2
                Vmx, Vmy, Vmz = MPSV[j], MPSV[j+1], MPSV[j+2]
                ms_α_Vtmp  = s_α * mul_im(Vmz)
                s_α_½_Vtmp = s_α_½ * mul_im(Vmy - Vmx)
                MPSV[j]      = muladd(c_½α_sq, Vmx, muladd(s_½α_sq, Vmy, -ms_α_Vtmp))
                MPSV[j+1]    = muladd(s_½α_sq, Vmx, muladd(c_½α_sq, Vmy,  ms_α_Vtmp))
                MPSV[j+2]    = muladd(c_α, Vmz, s_α_½_Vtmp)
            end
        end

        # Zero out next elements
        if i+1 < ETL
            j = 3*i+1
            @inbounds MPSV[j]   = 0
            @inbounds MPSV[j+1] = 0
            @inbounds MPSV[j+2] = 0
        end

        # Record the magnitude of the population of F1* as the echo amplitude, allowing for relaxation
        decay_curve[i] = E2_half * sqrt(abs2(MPSV[2]))

        if (i < ETL)
            # Twice loop-unrolled relaxmat loop
            @inbounds mprev = MPSV[1]
            @inbounds MPSV[1] = E2 * MPSV[2] # F1* --> F1
            @inbounds @simd for j in 2:6:3i-4
                m1, m2, m3, m4, m5, m6 = MPSV[j+1], MPSV[j+2], MPSV[j+3], MPSV[j+4], MPSV[j+5], MPSV[j+6]
                m0    = E2 * m3     # F(n)* --> F(n-1)*
                m1   *= E1          # Z(n)  --> Z(n)
                mtmp2 = m2
                m2    = E2 * mprev  # F(n)  --> F(n+1)
                m3    = E2 * m6     # F(n)* --> F(n-1)*
                m4   *= E1          # Z(n)  --> Z(n)
                mtmp5 = m5
                m5    = E2 * mtmp2  # F(n)  --> F(n+1)
                mprev = mtmp5
                MPSV[j], MPSV[j+1], MPSV[j+2], MPSV[j+3], MPSV[j+4], MPSV[j+5] = m0, m1, m2, m3, m4, m5
            end
            if isodd(i)
                @inbounds begin
                    j = 3i-1
                    m1, m2, m3 = MPSV[j+1], MPSV[j+2], MPSV[j+3]
                    m0    = E2 * m3     # F(n)* --> F(n-1)*
                    m1   *= E1          # Z(n)  --> Z(n)
                    m2    = E2 * mprev  # F(n)  --> F(n+1)
                    MPSV[j], MPSV[j+1], MPSV[j+2] = m0, m1, m2
                end
            end
        end
    end

    return decay_curve
end

####
#### Algorithm list
####

const EPGWork_List = [
    EPGWork_Basic_Cplx,
    EPGWork_Vec,
    EPGWork_Cplx,
    EPGWork_Cplx_Vec_Unrolled,
    EPGWork_ReIm,
    EPGWork_ReIm_DualCache,
    EPGWork_ReIm_DualCache_Split,
    EPGWork_ReIm_DualCache_Unrolled,
]
