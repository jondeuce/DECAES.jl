"""
    T2mapOptions(; kwargs...)
    T2mapOptions(image::Array{T,4}; kwargs...) where {T}

Options structure for [`T2mapSEcorr`](@ref).
This struct collects keyword arguments passed to `T2mapSEcorr`, performs checks on parameter types and values, and assigns default values to unspecified parameters.

# Arguments

    $TYPEDFIELDS

!!! note
    The 5D array that is saved when `SaveNNLSBasis` is set to `true` has dimensions `MatrixSize x nTE x nT2`, and therefore is typically extremely large.
    If the flip angle is fixed via `SetFlipAngle`, however, this is not an issue as only the unique `nTE x nT2` 2D basis matrix is saved.

See also:
* [`T2mapSEcorr`](@ref)
"""
@with_kw_noshow struct T2mapOptions{T<:Real}
    "Perform T2-mapping using legacy algorithms."
    legacy::Bool = false

    "Perform T2-mapping using multiple threads."
    threaded::Bool = Threads.nthreads() > 1

    "Size of first 3 dimensions of input 4D image. This argument has no default, but is inferred automatically as `size(image)[1:3]` when calling `T2mapSEcorr(image; kwargs...)`."
    MatrixSize::NTuple{3,Int}
    @assert all(MatrixSize .>= 1)

    "Number of echoes in input signal. This argument is has no default, but is inferred automatically as `size(image, 4)` when calling `T2mapSEcorr(image; kwargs...)`."
    nTE::Int
    @assert nTE >= 4

    "Interecho spacing (Units: seconds). This argument has no default."
    TE::T # seconds
    @assert TE > 0.0

    "Number of T2 times to estimate in the multi-exponential analysis. This argument has no default."
    nT2::Int
    @assert nT2 >= 2

    "Tuple of min and max T2 values (Units: seconds). This argument has no default."
    T2Range::NTuple{2,T} # seconds
    @assert 0.0 < T2Range[1] < T2Range[2]

    # "Variable interecho spacing"
    # vTEparam::Union{Tuple{T,T,Int}, Nothing} = nothing
    # @assert vTEparam === nothing || begin
    #     error("Variable interecho spacing is not yet supported")
    #     TE1, TE2, nTE1 = vTEparam
    #     0.0 < TE1 < TE2 && nTE1 < nTE && round(Int, TE2/TE1) ≈ TE2/TE1
    # end

    "Assumed value of T1 (Units: seconds)."
    T1::T = 1.0 # seconds
    @assert T1 > 0.0

    "First echo intensity cutoff for empty voxels."
    Threshold::T = !legacy ? 0.0 : 200.0
    @assert Threshold >= 0.0

    "Minimum refocusing angle for flip angle optimization (Units: degrees)."
    MinRefAngle::T = 50.0 # degrees
    @assert 0.0 <= MinRefAngle <= 180.0

    "During flip angle optimization, goodness of fit is checked for up to `nRefAngles` angles in the range `[MinRefAngle, 180]`. The optimal angle is then determined through interpolation from these samples."
    nRefAngles::Int = !legacy ? 32 : 8
    @assert nRefAngles >= 2

    "Initial number of angles to check during flip angle optimization before refinement near likely optima. Setting `nRefAnglesMin` equal to `nRefAngles` forces all angles to be checked."
    nRefAnglesMin::Int = !legacy ? min(5, nRefAngles) : nRefAngles
    @assert 2 <= nRefAnglesMin <= nRefAngles

    "Regularization routine to use. One of \"none\", \"chi2\", \"gcv\", or \"lcurve\", representing no regularization, `Chi2Factor`-based Tikhonov regularization, Generalized Cross-Validation method, or L-Curve based regularization, respectively."
    Reg::String
    @assert Reg ∈ ("none", "chi2", "gcv", "lcurve")

    "Constraint on ``\\chi^2`` used for regularization when `Reg == \"chi2\"`."
    Chi2Factor::Union{T,Nothing} = nothing
    @assert (Reg == "chi2" && Chi2Factor !== nothing && Chi2Factor > 1.0) || Reg != "chi2"

    "Refocusing pulse control angle (Units: degrees)."
    RefConAngle::Union{T,Nothing} = 180.0 # degrees
    @assert RefConAngle === nothing || 0.0 < RefConAngle <= 180.0

    "Instead of optimizing flip angle, use `SetFlipAngle` for all voxels (Units: degrees)."
    SetFlipAngle::Union{T,Nothing} = nothing
    @assert SetFlipAngle === nothing || 0.0 < SetFlipAngle <= 180.0

    "Boolean flag to include a 3D array of the ``\\ell^2``-norms of the residuals from the NNLS fits in the output maps dictionary."
    SaveResidualNorm::Bool = false

    "Boolean flag to include a 4D array of the time domain decay curves resulting from the NNLS fits in the output maps dictionary."
    SaveDecayCurve::Bool = false

    "Boolean flag to include 3D arrays of the regularization parameters ``\\mu`` and resulting ``\\chi^2``-factors in the output maps dictionary."
    SaveRegParam::Bool = false

    "Boolean flag to include a 5D (or 2D if `SetFlipAngle` is used) array of NNLS basis matrices in the output maps dictionary."
    SaveNNLSBasis::Bool = SetFlipAngle !== nothing

    "Suppress printing to the console."
    Silent::Bool = false
end
T2mapOptions(args...; kwargs...) = T2mapOptions{Float64}(args...; kwargs...)
T2mapOptions(image::Array{T,4}; kwargs...) where {T} = T2mapOptions{T}(; kwargs..., MatrixSize = size(image)[1:3], nTE = size(image)[4])

t2_times(o::T2mapOptions{T}) where {T} = logrange(o.T2Range..., o.nT2)
flip_angles(o::T2mapOptions{T}) where {T} = o.SetFlipAngle === nothing ? collect(range(o.MinRefAngle, T(180); length = o.nRefAngles)) : T[o.SetFlipAngle]
refcon_angles(o::T2mapOptions{T}) where {T} = o.RefConAngle === nothing ? collect(range(o.MinRefAngle, T(180); length = o.nRefAngles)) : T[o.RefConAngle]

function _show_string(o::T2mapOptions)
    io = IOBuffer()
    print(io, "T2-distribution analysis settings:")
    fields = fieldsof(typeof(o), Vector)
    fields = fields[sortperm(uppercase.(string.(fields)))] # sort alphabetically, ignoring case
    padlen = 1 + maximum(f -> length(string(f)), fields)
    for f in fields
        (f == :vTEparam) && continue # skip
        print(io, "\n$(lpad(rpad(f, padlen), padlen + 4)): $(getfield(o,f))")
    end
    return String(take!(io))
end
Base.show(io::IO, ::MIME"text/plain", o::T2mapOptions) = print(io, _show_string(o))

"""
    T2partOptions(; kwargs...)
    T2partOptions(t2dist::Array{T,4}; kwargs...) where {T}

Options structure for [`T2partSEcorr`](@ref).
This struct collects keyword arguments passed to `T2partSEcorr`, performs checks on parameter types and values, and assigns default values to unspecified parameters.

# Arguments

    $TYPEDFIELDS

See also:
* [`T2partSEcorr`](@ref)
"""
@with_kw_noshow struct T2partOptions{T<:Real}
    "Calculate T2-parts using legacy algorithms."
    legacy::Bool = false

    "Perform T2-parts using multiple threads."
    threaded::Bool = Threads.nthreads() > 1

    "Size of first 3 dimensions of input 4D T2 distribution. This argument is has no default, but is inferred automatically as `size(t2dist)[1:3]` when calling `T2partSEcorr(t2dist; kwargs...)`."
    MatrixSize::NTuple{3,Int}
    @assert all(MatrixSize .>= 1)

    "Number of T2 times to use. This argument has no default."
    nT2::Int
    @assert nT2 >= 2

    "Tuple of min and max T2 values (Units: seconds). This argument has no default."
    T2Range::NTuple{2,T} # seconds
    @assert 0.0 < T2Range[1] < T2Range[2]

    "Tuple of min and max T2 values of the short peak window (Units: seconds). This argument has no default."
    SPWin::NTuple{2,T} # seconds
    @assert SPWin[1] < SPWin[2]

    "Tuple of min and max T2 values of the middle peak window (Units: seconds). This argument has no default."
    MPWin::NTuple{2,T} # seconds
    @assert MPWin[1] < MPWin[2]

    "Apply sigmoidal weighting to the upper limit of the short peak window in order to smooth the hard small peak window cutoff time. `Sigmoid` is the delta-T2 parameter, which is the distance in seconds on either side of the `SPWin` upper limit where the sigmoid curve reaches 10% and 90% (Units: seconds)."
    Sigmoid::Union{T,Nothing} = nothing
    @assert Sigmoid === nothing || Sigmoid > 0

    "Suppress printing to the console."
    Silent::Bool = false
end
T2partOptions(args...; kwargs...) = T2partOptions{Float64}(args...; kwargs...)
T2partOptions(t2dist::Array{T,4}; kwargs...) where {T} = T2partOptions{T}(; kwargs..., MatrixSize = size(t2dist)[1:3], nT2 = size(t2dist)[4])

function _show_string(o::T2partOptions)
    io = IOBuffer()
    print(io, "T2-parts analysis settings:")
    fields = fieldsof(typeof(o), Vector)
    fields = fields[sortperm(uppercase.(string.(fields)))] # sort alphabetically
    padlen = 1 + maximum(f -> length(string(f)), fields)
    for f in fields
        print(io, "\n$(lpad(rpad(f, padlen), padlen + 4)): $(getfield(o,f))")
    end
    return String(take!(io))
end
Base.show(io::IO, ::MIME"text/plain", o::T2partOptions) = print(io, _show_string(o))
