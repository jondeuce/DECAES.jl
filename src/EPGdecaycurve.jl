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
EPGdecaycurve(ETL::Int, flip_angle::T, TE::T, T2::T, T1::T, refcon::T) where {T} =
    EPGdecaycurve!(EPGdecaycurve_work(T, ETL), ETL, flip_angle, TE, T2, T1, refcon)

function EPGdecaycurve_work(T, ETL)
    Mz = SizedVector{3*ETL}(zeros(Vec{2,T}, 3*ETL))
    decay_curve = SizedVector{ETL}(zeros(T, ETL))
    return @ntuple(Mz, decay_curve)
end

function EPGdecaycurve!(work, ETL::Int, flip_angle::T, TE::T, T2::T, T1::T, refcon::T) where {T}
    # Unpack workspace
    @unpack Mz, decay_curve = work
    @assert ETL > 2 && length(Mz) == 3*ETL && length(decay_curve) == ETL
    Mz .= Ref(zero(Vec{2,T}))

    # Precompute compute element flip matrices and other intermediate variables
    E2, E1, E2_half = exp(-TE/T2), exp(-TE/T1), exp(-(TE/2)/T2)
    T1mat = element_flipmat(flip_angle)
    T2mat = element_flipmat(flip_angle * (refcon/180))
    @inbounds T2mat_elements = ( # independent elements of T2mat
        s_α     = -imag(T2mat[1,3]), # sin(α)
        c_α     =  real(T2mat[3,3]), # cos(α)
        s_½α_sq =  real(T2mat[2,1]), # sin(α/2)^2
        c_½α_sq =  real(T2mat[1,1]), # cos(α/2)^2
        s_α_½   = -imag(T2mat[3,1]), # sin(α)/2
    )

    # Initialize magnetization phase state vector (MPSV)
    m0 = E2_half * sind(flip_angle/2) # initial population
    M0 = SA{T}[m0, 0, 0] # initial magnetization in F1 state
    M1 = T1mat * M0 # apply first refocusing pulse
    @inbounds decay_curve[1] = E2_half * sqrt(abs2(M1[2])) # first echo amplitude

    # Apply first relaxation matrix iteration on non-zero states
    @inbounds Mz[1] = E2 * Vec(reim(M1[2]))
    @inbounds Mz[2] = zero(Vec{2,T})
    @inbounds Mz[3] = E1 * Vec(reim(M1[3]))
    @inbounds Mz[4] = E2 * Vec(reim(M1[1]))

    # Perform flip-relax sequence ETL-1 times
    @timeit_debug TIMER "Flip-Relax Sequence" begin
        flipmat_relaxmat_action!(decay_curve, Mz, T2mat_elements, E1, E2_half, E2)
    end

    return decay_curve
end

# Element matrix for the effect of the refocusing pulse with angle
# α (in degrees) on the magnetization state vector
element_flipmat(α::T) where {T} = SA{Complex{T}}[
        cosd(α/2)^2    sind(α/2)^2 -im*sind(α);
        sind(α/2)^2    cosd(α/2)^2  im*sind(α);
    -im*sind(α)/2   im*sind(α)/2       cosd(α)]

####
#### flipmat_relaxmat_action!
####

# Optimized function which combines flip matrix and
# relaxation matrix functions into one loop.
# Combining these loop and using explicit SIMD.jl `Vec`
# vector types instead of `Complex` values drastically
# lowers execution time.
# Since this function is called many times during T2mapSEcorr,
# the micro-optimizations are worth it despite the loss of
# code readability. See the `flipmat_relaxmat_action_slow!`
# section below for a more readable implementation which
# produces exactly the same result
@inline function flipmat_relaxmat_action!(
        decay_curve :: SizedArray{Tuple{ETL}, T},
        Mz          :: SizedArray{Tuple{ETL3}, Vec{2,T}},
        T2mat_elements,
        E1,
        E2_half,
        E2
    ) where {ETL,ETL3,T}
    # @assert 3*ETL == ETL3

    # Extract matrix elements, initialize buffers
    a1, a2, a3, a4, a5 = T2mat_elements
    @inbounds Mz3 = Mz[3]

    @inbounds for i = 2:ETL-1
        # Unroll first flipmat/relaxmat iteration to get Mz[2] for decay_curve
        Vx, Vy = Mz[1], Mz[2]
        a1z = a1 * Vec((Mz3[2], -Mz3[1]))
        Mz4 = E2 * (a4 * Vx + a3 * Vy + a1z) # relaxmat: 1 -> 4, save in buffer
        Mz2 = a3 * Vx + a4 * Vy - a1z # flipmat: 2 -> decay_curve

        ###########################
        # decay_curve curve coefficient
        decay_curve[i] = E2_half * sqrt(sum(Mz2 * Mz2))

        # Unrolled flipmat/relaxmat iteration
        Mz[1] = E2 * Mz2 # relaxmat: 2 -> 1
        Mz3   = E1 * (a2 * Mz3 + a5 * Vec((Vx[2] - Vy[2], Vy[1] - Vx[1]))) # relaxmat: 3 -> 3, save in buffer

        ###########################
        # flipmat + relaxmat loop
        @inbounds for j in 4:3:3*min(i+1, ETL)
            Vx, Vy, Vz = Mz[j], Mz[j+1], Mz[j+2]
            a1z     = a1 * Vec((Vz[2], -Vz[1]))
            tmp     = E2 * (a4 * Vx + a3 * Vy + a1z)
            Mz[j  ] = Mz4 # relaxmat: assign forward, j -> j+3
            Mz4     = tmp
            Mz[j-2] = E2 * (a3 * Vx + a4 * Vy - a1z) # relaxmat: assign backwards, j+1 -> j+1-3
            Mz[j+2] = E1 * (a2 * Vz + a5 * Vec((Vx[2] - Vy[2], Vy[1] - Vx[1]))) # relaxmat: j+2 -> j+2
        end
    end

    ###########################
    # decay_curve curve coefficient
    @inbounds V = a3 * Mz[1] + a4 * Mz[2] + a1 * Vec((-Mz3[2], Mz3[1])) # last iteration of flipmat unrolled
    @inbounds decay_curve[end] = E2_half * sqrt(sum(V * V))

    return decay_curve
end

####
#### flipmat_relaxmat_action_slow!
####

function flipmat_relaxmat_action_slow!(decay_curve, Mz, ETL, T2mat_elements, E1, E2_half, E2)
    @inbounds for i = 2:ETL
        # Perform the flip for all states
        flipmat_action!(Mz, min(i+1, ETL), T2mat_elements)

        # Record the magnitude of the population of F1* as the echo amplitude
        # and allow for relaxation
        decay_curve[i] = E2_half * sqrt(abs2(Mz[2]))

        # Allow time evolution of magnetization between pulses
        relaxmat_action!(Mz, min(i+1, ETL), E2, E1)
    end
    return decay_curve
end

# Computes the action of the transition matrix that describes the effect of the refocusing pulse
# on the magnetization phase state vector, as given by Hennig (1988), but corrected by Jones (1997)
# 
# The below implementation is heavily micro-optimized, but is exactly equivilant to the
# following pseudo-code loop:
# 
#   for each complex magnetization vector M
#       M <-- T2mat * M
#   end for
# 
# where T2mat is complex flip matrix from `element_flipmat`
@inline function flipmat_action!(Mz::AbstractVector{Complex{T}}, nStates, T2mat_elements) where {T}
    @assert 3*nStates <= length(Mz)

    # # Basic loop: load in complex SVector, do multiplication, assign to Mz
    # @inbounds @simd for j in 1:3:3*nStates
    #     m = SVector{3,Complex{T}}((Mz[j], Mz[j+1], Mz[j+2]))
    #     m = T2mat * m
    #     Mz[j], Mz[j+1], Mz[j+2] = m
    # end
    
    # Optimized loop: use explicity SIMD vector units from SIMD.jl
    # to ensure full use of vector instructions and specialize on structure
    # of `element_flipmat`
    s_α, c_α, s_½α_sq, c_½α_sq, s_α_½ = T2mat_elements
    @inbounds @simd for j in 1:3:3*nStates
        Vmx = Vec(reim(Mz[j  ]))
        Vmy = Vec(reim(Mz[j+1]))
        Vmz = Vec(reim(Mz[j+2]))
        s_α_Vtmp   = s_α * Vec((Vmz[2], -Vmz[1]))
        s_α_½_Vtmp = s_α_½ * Vec((Vmx[2] - Vmy[2], Vmy[1] - Vmx[1]))
        Vm         = muladd(c_½α_sq, Vmx, muladd(s_½α_sq, Vmy,  s_α_Vtmp))
        Mz[j]      = Complex(Vm[1], Vm[2])
        Vm         = muladd(s_½α_sq, Vmx, muladd(c_½α_sq, Vmy, -s_α_Vtmp))
        Mz[j+1]    = Complex(Vm[1], Vm[2])
        Vm         = muladd(c_α, Vmz, s_α_½_Vtmp)
        Mz[j+2]    = Complex(Vm[1], Vm[2])
    end

    # # Twice loop-unrolled version of above loop:
    # #     NOTE: currently not working
    # s_α, c_α, s_½α_sq, c_½α_sq, s_α_½ = T2mat_elements
    # @inbounds @simd for j in 1:6:3*nStates
    #     Vmx = Vec((reim(Mz[j  ])..., reim(Mz[j+3])...))
    #     Vmy = Vec((reim(Mz[j+1])..., reim(Mz[j+4])...))
    #     Vmz = Vec((reim(Mz[j+2])..., reim(Mz[j+5])...))
    #     s_α_Vtmp   = s_α * Vec((Vmz[2], -Vmz[1], Vmz[4], -Vmz[3]))
    #     s_α_½_Vtmp = s_α_½ * Vec((Vmx[2] - Vmy[2], Vmy[1] - Vmx[1], Vmx[4] - Vmy[4], Vmy[3] - Vmx[3]))
    #     VMx        = muladd(c_½α_sq, Vmx, muladd(s_½α_sq, Vmy,  s_α_Vtmp))
    #     VMy        = muladd(s_½α_sq, Vmx, muladd(c_½α_sq, Vmy, -s_α_Vtmp))
    #     VMz        = muladd(c_α, Vmz, s_α_½_Vtmp)
    #     Mz[j  ]    = Complex(VMx[1], VMx[2])
    #     Mz[j+1]    = Complex(VMy[1], VMy[2])
    #     Mz[j+2]    = Complex(VMz[1], VMz[2])
    #     Mz[j+3]    = Complex(VMx[3], VMx[4])
    #     Mz[j+4]    = Complex(VMy[3], VMy[4])
    #     Mz[j+5]    = Complex(VMz[3], VMz[4])
    # end

    return Mz
end

# Computes the action of the relaxation matrix that describes the time evolution of the
# magnetization phase state vector after each refocusing pulse as described by Hennig (1988)
@inline function relaxmat_action!(Mz::AbstractVector{Complex{T}}, nStates, E2, E1) where {T}
    @assert 3*nStates <= length(Mz)

    # Basic relaxation matrix loop:
    @inbounds mprev = Mz[1]
    @inbounds Mz[1] = E2 * Mz[2] # F1* --> F1
    @inbounds @simd for j in 2:3:3*(nStates-1)
        m1, m2, m3 = Mz[j+1], Mz[j+2], Mz[j+3]
        mtmp  = m2
        m0    = E2 * m3     # F(n)* --> F(n-1)*
        m1   *= E1          # Z(n)  --> Z(n)
        m2    = E2 * mprev  # F(n)  --> F(n+1)
        mprev = mtmp
        Mz[j], Mz[j+1], Mz[j+2] = m0, m1, m2
    end
    @inbounds Mz[end-1]  = 0
    @inbounds Mz[end  ] *= E1

    # # Twice loop-unrolled version of above loop:
    # #     NOTE: currently not working
    # @inbounds mprev = Mz[1]
    # @inbounds Mz[1] = E2 * Mz[2] # F1* --> F1
    # @inbounds @simd for j in 2:6:6*(nStates÷2-1)
    #     m1, m2, m3, m4, m5, m6 = Mz[j+1], Mz[j+2], Mz[j+3], Mz[j+4], Mz[j+5], Mz[j+6]
    #     m0    = E2 * m3     # F(n)* --> F(n-1)*
    #     m1   *= E1          # Z(n)  --> Z(n)
    #     mtmp2 = m2
    #     m2    = E2 * mprev  # F(n)  --> F(n+1)
    #     m3    = E2 * m6     # F(n)* --> F(n-1)*
    #     m4   *= E1          # Z(n)  --> Z(n)
    #     mtmp5 = m5
    #     m5    = E2 * mtmp2  # F(n)  --> F(n+1)
    #     mprev = mtmp5
    #     Mz[j], Mz[j+1], Mz[j+2], Mz[j+3], Mz[j+4], Mz[j+5] = m0, m1, m2, m3, m4, m5
    # end
    # @inbounds if iseven(length(Mz)÷3)
    #     # Last state handled manually, if necessary
    #     Mz[end-4]  = E2 * Mz[end-1] # F(n)* --> F(n-1)*
    #     Mz[end-3] *= E1             # Z(n)  --> Z(n)
    #     Mz[end-2]  = E2 * mprev     # F(n)  --> F(n+1)
    # end
    # @inbounds Mz[end-1]  = 0
    # @inbounds Mz[end  ] *= E1

    return Mz
end
