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

EPGdecaycurve_work(::Type{T}, ETL::Int) where {T} = EPGWork_Cplx(T, ETL::Int) # fallback
EPGdecaycurve_work(::Type{T}, ETL::Int) where {T <: FloatingTypes} = EPGWork_Vec(T, ETL::Int)
# EPGdecaycurve_work(::Type{T}, ETL::Int) where {T <: FloatingTypes} = EPGWork_Cplx_Vec_Unrolled(T, ETL::Int)

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
function EPGdecaycurve(ETL::Int, flip_angle::Real, TE::Real, T2::Real, T1::Real, refcon::Real)
    T = float(promote_type(typeof(flip_angle), typeof(TE), typeof(T2), typeof(T1), typeof(refcon)))
    EPGdecaycurve!(EPGdecaycurve_work(T, ETL), EPGOptions{T,ETL}(flip_angle, TE, T2, T1, refcon))
end
EPGdecaycurve!(work::AbstractEPGWorkspace{T,ETL}, o::EPGOptions{T,ETL}) where {T,ETL} = EPGdecaycurve!(work.decay_curve, work, o)
EPGdecaycurve!(work::AbstractEPGWorkspace{T,ETL}, flip_angle::Real, TE::Real, T2::Real, T1::Real, refcon::Real) where {T,ETL} = EPGdecaycurve!(work.decay_curve, work, EPGOptions{T,ETL}(flip_angle, TE, T2, T1, refcon))
EPGdecaycurve!(decay_curve::AbstractVector{T}, work::AbstractEPGWorkspace{T,ETL}, flip_angle::Real, TE::Real, T2::Real, T1::Real, refcon::Real) where {T,ETL} = EPGdecaycurve!(decay_curve, work, EPGOptions{T,ETL}(flip_angle, TE, T2, T1, refcon))

function EPGdecaycurve!(decay_curve::AbstractVector{T}, work::AbstractEPGWorkspace{T,ETL}, o::EPGOptions{T,ETL}) where {T,ETL}

    @timeit_debug TIMER() "Flip-Relax Sequence" begin
        # Initialize decay_curve, workspace, etc. and return precomputed values for flipmat_relaxmat_action!
        precomp = epg_setup!(decay_curve, work, o)

        # Perform flip-relax sequence
        flipmat_relaxmat_action!(decay_curve, work, precomp)
    end

    return decay_curve
end

####
#### Helper functions
####

# Element matrix for refocusing pulse with angle α (in degrees); acts on the magnetization state vector (MPSV)
@inline element_flipmat(α::T) where {T} = SA{Complex{T}}[
        cosd(α/2)^2    sind(α/2)^2 -im*sind(α);
        sind(α/2)^2    cosd(α/2)^2  im*sind(α);
    -im*sind(α)/2   im*sind(α)/2       cosd(α)]

####
#### EPGWork_Vec
####

# Optimized version which combines flip matrix and
# relaxation matrix functions into one loop.
# Combining these loops and using explicit SIMD.jl `Vec`
# vector types instead of `Complex` decreases execution time
# by a factor of ~2.
# As this function is called many times during T2mapSEcorr,
# the micro-optimizations are worth the loss of code readability.
# See the `EPGWork_Cplx` section below for a more readable
# implementation which produces exactly the same result.

struct EPGWork_Vec{T, ETL, MzType<:AbstractVector{Vec{2,T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    Mz::MzType
    decay_curve::DCType
    function EPGWork_Vec(T, ETL::Int)
        Mz = SizedVector{3*ETL,Vec{2,T}}(zeros(Vec{2,T}, 3*ETL))
        dc = SizedVector{ETL,T}(zeros(T, ETL))
        new{T,ETL,typeof(Mz),typeof(dc)}(Mz, dc)
    end
end

@inline function epg_setup!(decay_curve::AbstractVector{T}, work::EPGWork_Vec{T,ETL}, o::EPGOptions{T,ETL}) where {T,ETL}
    # Unpack workspace
    @unpack Mz = work
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
        Mz[1] = Vec((E2 * M1y, zero(T)))
        Mz[2] = zero(Vec{2,T})
        Mz[3] = Vec((zero(T), E1 * M1z))
        Mz[4] = Vec((E2 * M1x, zero(T)))
    end

    return @ntuple(α2, E1, E2_half, E2)
end

@inline function flipmat_relaxmat_action!(
        decay_curve    :: AbstractVector{T},
        work           :: EPGWork_Vec{T,ETL},
        precomp,
    ) where {T,ETL}

    ###########################
    # Extract matrix elements + initialize temporaries
    @unpack α2, E1, E2_half, E2 = precomp
    a1, a2, a3, a4, a5 = sind(α2), cosd(α2), sind(α2/2)^2, cosd(α2/2)^2, sind(α2)/2 # independent elements of T2mat
    b1, b2, b3, b4, b5 = E2*a1, E1*a2, E2*a3, E2*a4, E1*a5
    c1, c3, c4 = E2_half*a1, E2_half*a3, E2_half*a4
    b1F, b5F, c1F = Vec((-b1, b1)), Vec((-b5, b5)), Vec((-c1, c1))
    @unpack Mz = work
    @inbounds Mz3 = Mz[3]

    @inbounds for i = 2:ETL-1
        ###########################
        # Unroll first flipmat/relaxmat iteration
        Vx, Vy = Mz[1], Mz[2]
        c1z = shufflevector(c1F * Mz3, Val((1,0)))
        Mz2 = muladd(c3, Vx, muladd(c4, Vy, -c1z)) # flipmat: 2 -> decay_curve
        Mz4 = muladd(b4, Vx, muladd(b3, Vy, E2_half * c1z)) # relaxmat: 1 -> 4, save in buffer

        ###########################
        # decay_curve curve coefficient
        decay_curve[i] = sqrt(sum(Mz2 * Mz2))

        # Unrolled flipmat/relaxmat iteration
        Mz[1] = E2_half * Mz2 # relaxmat: 2 -> 1
        b5xy  = shufflevector(b5F * (Vx - Vy), Val((1,0)))
        Mz3   = muladd(b2, Mz3, b5xy) # relaxmat: 3 -> 3, save in buffer

        ###########################
        # flipmat + relaxmat loop
        @inbounds for j in 4:3:3*min(i-1, ETL)
            Vx, Vy, Vz = Mz[j], Mz[j+1], Mz[j+2]
            b1z     = shufflevector(b1F * Vz, Val((1,0)))
            Mz[j  ] = Mz4 # relaxmat: assign forward, j -> j+3
            Mz4     = muladd(b4, Vx, muladd(b3, Vy,  b1z))
            Mz[j-2] = muladd(b3, Vx, muladd(b4, Vy, -b1z)) # relaxmat: assign backwards, j+1 -> j+1-3
            b5xy    = shufflevector(b5F * (Vx - Vy), Val((1,0)))
            Mz[j+2] = muladd(b2, Vz, b5xy) # relaxmat: j+2 -> j+2
        end

        ###########################
        # cleanup + zero next elements
        j  = 3i-2
        Vx = Mz[j]
        Mz[j  ] = Mz4 # relaxmat: assign forward, j -> j+3
        Mz[j-2] = b3 * Vx # relaxmat: assign backwards, j+1 -> j+1-3
        Mz[j+2] = shufflevector(b5F * Vx, Val((1,0))) # relaxmat: j+2 -> j+2
        Mz[j+3] = b4 * Vx # relaxmat: assign forward, j -> j+3
        Mz[j+1] = Vec((zero(T), zero(T))) # relaxmat: assign backwards, j+1 -> j+1-3
        Mz[j+5] = Vec((zero(T), zero(T))) # relaxmat: j+2 -> j+2
    end

    ###########################
    # decay_curve curve coefficient
    c1z = shufflevector(c1F * Mz3, Val((1,0)))
    @inbounds Mz2 = muladd(c3, Mz[1], muladd(c4, Mz[2], -c1z)) # last iteration of flipmat unrolled
    @inbounds decay_curve[end] = sqrt(sum(Mz2 * Mz2))

    return decay_curve
end

####
#### EPGWork_Cplx and EPGWork_Cplx_Vec_Unrolled
####

struct EPGWork_Cplx{T, ETL, MzType<:AbstractVector{Complex{T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    Mz::MzType
    decay_curve::DCType
    function EPGWork_Cplx(T, ETL::Int)
        Mz = SizedVector{3*ETL,Complex{T}}(zeros(Complex{T}, 3*ETL))
        dc = SizedVector{ETL,T}(zeros(T, ETL))
        new{T,ETL,typeof(Mz),typeof(dc)}(Mz, dc)
    end
end

struct EPGWork_Cplx_Vec_Unrolled{T, ETL, MzType<:AbstractVector{Complex{T}}, DCType <: AbstractVector{T}} <: AbstractEPGWorkspace{T,ETL}
    Mz::MzType
    decay_curve::DCType
    function EPGWork_Cplx_Vec_Unrolled(T, ETL::Int)
        Mz = SizedVector{3*ETL,Complex{T}}(zeros(Complex{T}, 3*ETL))
        dc = SizedVector{ETL,T}(zeros(T, ETL))
        new{T,ETL,typeof(Mz),typeof(dc)}(Mz, dc)
    end
end

@inline function epg_setup!(
        decay_curve::AbstractVector{T},
        work::Union{EPGWork_Cplx{T,ETL}, EPGWork_Cplx_Vec_Unrolled{T,ETL}},
        o::EPGOptions{T,ETL},
    ) where {T,ETL}
    # Unpack workspace
    @unpack Mz = work
    @unpack flip_angle, TE, T2, T1, refcon = o

    # Precompute compute element flip matrices and other intermediate variables
    E2, E1, E2_half = exp(-TE/T2), exp(-TE/T1), exp(-(TE/2)/T2)
    T1mat = element_flipmat(flip_angle)
    T2mat = element_flipmat(flip_angle * (refcon/180))
    @inbounds T2mat_elements = ( # independent elements of T2mat
       -imag(T2mat[1,3]), # sin(α)
        real(T2mat[3,3]), # cos(α)
        real(T2mat[2,1]), # sin(α/2)^2
        real(T2mat[1,1]), # cos(α/2)^2
       -imag(T2mat[3,1]), # sin(α)/2
    )

    # Initialize magnetization phase state vector (MPSV)
    m0 = E2_half * sind(flip_angle/2) # initial population
    M0 = SA{T}[m0, 0, 0] # initial magnetization in F1 state
    M1 = T1mat * M0 # apply first refocusing pulse
    @inbounds decay_curve[1] = E2_half * sqrt(abs2(M1[2])) # first echo amplitude

    # Apply first relaxation matrix iteration on non-zero states
    @inbounds begin
        Mz[1] = E2 * M1[2]
        Mz[2] = 0
        Mz[3] = E1 * M1[3]
        Mz[4] = E2 * M1[1]
        Mz[5] = 0
        Mz[6] = 0
    end

    return @ntuple(T2mat_elements, E1, E2_half, E2)
end

@inline function flipmat_relaxmat_action!(
        decay_curve::AbstractVector{T},
        work::Union{EPGWork_Cplx{T,ETL}, EPGWork_Cplx_Vec_Unrolled{T,ETL}},
        precomp,
    ) where {T,ETL}

    @unpack Mz = work
    @unpack E2_half = precomp

    @inbounds for i = 2:ETL
        # Perform the flip for all states
        flipmat_action!(work, precomp, i)

        # Record the magnitude of the population of F1* as the echo amplitude, allowing for relaxation
        decay_curve[i] = E2_half * sqrt(abs2(Mz[2]))

        # Allow time evolution of magnetization between pulses
        (i < ETL) && relaxmat_action!(work, precomp, min(i+1, ETL))
    end

    return decay_curve
end

#### EPGWork_Cplx

@inline function flipmat_action!(work::EPGWork_Cplx{T,ETL}, precomp, nStates::Int) where {T,ETL}
    @unpack Mz = work
    @unpack T2mat_elements = precomp
    s_α, c_α, s_½α_sq, c_½α_sq, s_α_½ = T2mat_elements

    @inbounds @simd for j in 1:3:3*nStates-2
        Vmx, Vmy, Vmz = Mz[j], Mz[j+1], Mz[j+2]
        ms_α_Vtmp  = s_α * mul_im(Vmz)
        s_α_½_Vtmp = s_α_½ * mul_im(Vmy - Vmx)
        Mz[j]      = muladd(c_½α_sq, Vmx, muladd(s_½α_sq, Vmy, -ms_α_Vtmp))
        Mz[j+1]    = muladd(s_½α_sq, Vmx, muladd(c_½α_sq, Vmy,  ms_α_Vtmp))
        Mz[j+2]    = muladd(c_α, Vmz, s_α_½_Vtmp)
    end

    # Zero out next elements
    if nStates+1 < ETL
        j = 3*nStates+1
        @inbounds Mz[j]   = 0
        @inbounds Mz[j+1] = 0
        @inbounds Mz[j+2] = 0
    end

    return Mz
end

@inline function relaxmat_action!(work::EPGWork_Cplx{T,ETL}, precomp, nStates::Int) where {T,ETL}
    @unpack Mz = work
    @unpack E2, E1 = precomp

    # Basic relaxation matrix loop:
    @inbounds mprev = Mz[1]
    @inbounds Mz[1] = E2 * Mz[2] # F1* --> F1
    @inbounds @simd for j in 2:3:3*nStates-4
        m1, m2, m3 = Mz[j+1], Mz[j+2], Mz[j+3]
        mtmp  = m2
        m0    = E2 * m3     # F(n)* --> F(n-1)*
        m1   *= E1          # Z(n)  --> Z(n)
        m2    = E2 * mprev  # F(n)  --> F(n+1)
        mprev = mtmp
        Mz[j], Mz[j+1], Mz[j+2] = m0, m1, m2
    end

    return Mz
end

#### EPGWork_Cplx_Vec_Unrolled

@inline function flipmat_action!(work::EPGWork_Cplx_Vec_Unrolled{T,ETL}, precomp, nStates::Int) where {T,ETL}
    @unpack Mz = work
    @unpack T2mat_elements = precomp
    s_α, c_α, s_½α_sq, c_½α_sq, s_α_½ = T2mat_elements

    # Twice loop-unrolled flip-matrix loop
    s_α, c_α, s_½α_sq, c_½α_sq, s_α_½ = T2mat_elements
    Vs_α = Vec((-s_α, s_α, -s_α, s_α))
    Vs_α_½ = Vec((-s_α_½, s_α_½, -s_α_½, s_α_½))
    @inbounds @simd for j in 1:6:3*nStates-5 # 1+(0:3*(n-2))
        Vmx = Vec((reim(Mz[j  ])..., reim(Mz[j+3])...))
        Vmy = Vec((reim(Mz[j+1])..., reim(Mz[j+4])...))
        Vmz = Vec((reim(Mz[j+2])..., reim(Mz[j+5])...))
        s_α_Vtmp   = shufflevector(Vs_α * Vmz, Val((1,0,3,2)))
        s_α_½_Vtmp = shufflevector(Vs_α_½ * (Vmx - Vmy), Val((1,0,3,2)))
        VMx        = muladd(c_½α_sq, Vmx, muladd(s_½α_sq, Vmy,  s_α_Vtmp))
        VMy        = muladd(s_½α_sq, Vmx, muladd(c_½α_sq, Vmy, -s_α_Vtmp))
        VMz        = muladd(c_α, Vmz, s_α_½_Vtmp)
        Mz[j  ]    = Complex(VMx[1], VMx[2])
        Mz[j+1]    = Complex(VMy[1], VMy[2])
        Mz[j+2]    = Complex(VMz[1], VMz[2])
        Mz[j+3]    = Complex(VMx[3], VMx[4])
        Mz[j+4]    = Complex(VMy[3], VMy[4])
        Mz[j+5]    = Complex(VMz[3], VMz[4])
    end
    if isodd(nStates)
        @inbounds begin
            j = 3*nStates-2
            Vmx, Vmy, Vmz = Mz[j], Mz[j+1], Mz[j+2]
            ms_α_Vtmp  = s_α * mul_im(Vmz)
            s_α_½_Vtmp = s_α_½ * mul_im(Vmy - Vmx)
            Mz[j]      = muladd(c_½α_sq, Vmx, muladd(s_½α_sq, Vmy, -ms_α_Vtmp))
            Mz[j+1]    = muladd(s_½α_sq, Vmx, muladd(c_½α_sq, Vmy,  ms_α_Vtmp))
            Mz[j+2]    = muladd(c_α, Vmz, s_α_½_Vtmp)
        end
    end

    # Zero out next elements
    if nStates+1 < ETL
        j = 3*nStates+1
        @inbounds Mz[j]   = 0
        @inbounds Mz[j+1] = 0
        @inbounds Mz[j+2] = 0
    end

    return Mz
end

@inline function relaxmat_action!(work::EPGWork_Cplx_Vec_Unrolled{T,ETL}, precomp, nStates::Int) where {T,ETL}
    @unpack Mz = work
    @unpack E2, E1 = precomp

    # Twice loop-unrolled version of above loop
    @inbounds mprev = Mz[1]
    @inbounds Mz[1] = E2 * Mz[2] # F1* --> F1
    @inbounds @simd for j in 2:6:3*nStates-7
        m1, m2, m3, m4, m5, m6 = Mz[j+1], Mz[j+2], Mz[j+3], Mz[j+4], Mz[j+5], Mz[j+6]
        m0    = E2 * m3     # F(n)* --> F(n-1)*
        m1   *= E1          # Z(n)  --> Z(n)
        mtmp2 = m2
        m2    = E2 * mprev  # F(n)  --> F(n+1)
        m3    = E2 * m6     # F(n)* --> F(n-1)*
        m4   *= E1          # Z(n)  --> Z(n)
        mtmp5 = m5
        m5    = E2 * mtmp2  # F(n)  --> F(n+1)
        mprev = mtmp5
        Mz[j], Mz[j+1], Mz[j+2], Mz[j+3], Mz[j+4], Mz[j+5] = m0, m1, m2, m3, m4, m5
    end
    if iseven(nStates)
        @inbounds begin
            j = 3*nStates-4
            m1, m2, m3 = Mz[j+1], Mz[j+2], Mz[j+3]
            m0    = E2 * m3     # F(n)* --> F(n-1)*
            m1   *= E1          # Z(n)  --> Z(n)
            m2    = E2 * mprev  # F(n)  --> F(n+1)
            Mz[j], Mz[j+1], Mz[j+2] = m0, m1, m2
        end
    end

    return Mz
end

# Computing the action of the transition matrix that describes the effect of the refocusing pulse
# on the magnetization phase state vector, as given by Hennig (1988), but corrected by Jones (1997):
# 
#   for each complex magnetization vector M
#       M <-- T2mat * M
#   end for
# 
# where T2mat is complex flip matrix from `element_flipmat`

# Basic loop: load in complex SVector, do multiplication, assign to Mz
# @inbounds @simd for j in 1:3:3*nStates-2
#     m = SVector{3,Complex{T}}((Mz[j], Mz[j+1], Mz[j+2]))
#     m = T2mat * m
#     Mz[j], Mz[j+1], Mz[j+2] = m
# end

# Computes the action of the relaxation matrix that describes the time evolution of the
# magnetization phase state vector after each refocusing pulse as described by Hennig (1988)
