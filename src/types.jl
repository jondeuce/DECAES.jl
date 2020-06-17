"""
    T2mapOptions(; <keyword arguments>)

Options structure for [`T2mapSEcorr`](@ref).
This struct collects keyword arguments passed to `T2mapSEcorr`, performs checks on parameter types and values, and assigns default values to unspecified parameters.

# Arguments
- `MatrixSize`:       size of first 3 dimensions of input 4D image. This argument is has no default, but is inferred automatically as `size(image)[1:3]` when calling `T2mapSEcorr(image; kwargs...)`
- `nTE`:              number of echoes in input signal. This argument is has no default, but is inferred automatically as `size(image, 4)` when calling `T2mapSEcorr(image; kwargs...)`
- `TE`:               interecho spacing (Default: `10e-3`, Units: seconds)
- `T1`:               assumed value of T1 (Default: `1.0`, Units: seconds)
- `Threshold`:        first echo intensity cutoff for empty voxels (Default: `200.0`)
- `Chi2Factor`:       constraint on ``\\chi^2`` used for regularization when `Reg == "chi2"` (Default: `1.02`)
- `nT2`:              number of T2 times to use (Default: `40`)
- `T2Range`:          min and max T2 values (Default: `(10e-3, 2.0)`, Units: seconds)
- `RefConAngle`:      refocusing pulse control angle (Default: `180.0`, Units: degrees)
- `MinRefAngle`:      minimum refocusing angle for flip angle optimization (Default: `50.0`, Units: degrees)
- `nRefAngles`:       during flip angle optimization, goodness of fit is checked for up to `nRefAngles` angles in the range `[MinRefAngle, 180]`. The optimal angle is then determined through interpolation from these samples (Default: `32`)
- `nRefAnglesMin`:    initial number of angles to check during flip angle optimization before refinement near likely optima; `nRefAnglesMin == nRefAngles` forces all angles to be checked (Default: `5`)
- `Reg`:              regularization routine to use:
    - `"no"`:         no regularization of solution
    - `"chi2"`:       use `Chi2Factor` based regularization (Default)
    - `"lcurve"`:     use L-Curve based regularization
- `SetFlipAngle`:     instead of optimizing flip angle, use this flip angle for all voxels (Default: `nothing`, Units: degrees)
- `SaveResidualNorm`: `true`/`false` option to include a 3D array of the ``\\ell^2``-norms of the residuals from the NNLS fits in the output maps dictionary (Default: `false`)
- `SaveDecayCurve`:   `true`/`false` option to include a 4D array of the time domain decay curves resulting from the NNLS fits in the output maps dictionary (Default: `false`)
- `SaveRegParam`:     `true`/`false` option to include 3D arrays of the regularization parameters ``\\mu`` and resulting ``\\chi^2``-factors in the output maps dictionary (Default: `false`)
- `SaveNNLSBasis`:    `true`/`false` option to include a 4D array of NNLS basis matrices in the output maps dictionary (Default: `false`)
!!! note
    The 4D array that is saved when `SaveNNLSBasis` is set to `true` has dimensions `MatrixSize x nTE x nT2`,
    and therefore is typically extremely large; by default, it is `nT2 = 40` times the size of the input image.
- `Silent`:         suppress printing to the console (Default: `false`)

See also:
* [`T2mapSEcorr`](@ref)
"""
@with_kw_noshow struct T2mapOptions{T} @deftype T
    legacy::Bool = false

    MatrixSize::NTuple{3,Int}
    @assert all(MatrixSize .>= 1)

    nTE::Int
    @assert nTE >= 4

    TE = 0.010 # seconds
    @assert 0.0 < TE

    vTEparam::Union{Tuple{T,T,Int}, Nothing} = nothing
    @assert isnothing(vTEparam) || begin
        TE1, TE2, nTE1 = vTEparam
        0.0 < TE1 < TE2 && nTE1 < nTE && round(Int, TE2/TE1) ≈ TE2/TE1
    end
    @assert isnothing(vTEparam) || error("Variable TE is not implemented")

    T1 = 1.0 # seconds
    @assert 0.0 < T1

    Threshold = 200.0 # magnitude signal intensity
    @assert Threshold >= 0.0

    Chi2Factor = 1.02
    @assert Chi2Factor > 1

    nT2::Int = 40
    @assert 10 <= nT2 <= 120

    T2Range::NTuple{2,T} = !legacy ? (0.01, 2.0) : (0.015, 2.0) # seconds
    @assert 0.0 < T2Range[1] < T2Range[2]

    RefConAngle = 180.0 # degrees
    @assert 0.0 <= RefConAngle <= 180.0

    MinRefAngle = 50.0 # degrees
    @assert 0.0 <= MinRefAngle <= 180.0

    nRefAngles::Int = !legacy ? 32 : 8
    @assert nRefAngles >= 2

    nRefAnglesMin::Int = !legacy ? min(5, nRefAngles) : nRefAngles
    @assert 2 <= nRefAnglesMin <= nRefAngles

    Reg::String = "chi2"
    @assert Reg ∈ ("no", "chi2", "lcurve")

    SetFlipAngle::Union{T,Nothing} = nothing
    @assert isnothing(SetFlipAngle) || 0.0 < SetFlipAngle <= 180.0

    SaveResidualNorm::Bool = false

    SaveDecayCurve::Bool = false

    SaveRegParam::Bool = false

    SaveNNLSBasis::Bool = false

    Silent::Bool = false
end
T2mapOptions(args...; kwargs...) = T2mapOptions{Float64}(args...; kwargs...)
T2mapOptions(image::Array{T,4}; kwargs...) where {T} = T2mapOptions{T}(;MatrixSize = size(image)[1:3], nTE = size(image)[4], kwargs...)

function _show_string(o::T2mapOptions)
    io = IOBuffer()
    print(io, "T2-Distribution analysis settings:")
    fields = fieldnames(typeof(o))
    fields = fields[sortperm(uppercase.(string.([fields...])))] # sort alphabetically, ignoring case
    padlen = 1 + maximum(f -> length(string(f)), fields)
    for f in fields
        (f == :legacy || f == :vTEparam) && continue # skip
        print(io, "\n$(lpad(rpad(f, padlen), padlen + 4)): $(getfield(o,f))")
    end
    return String(take!(io))
end
# Base.show(io::IO, o::T2mapOptions) = print(io, _show_string(o))
Base.show(io::IO, ::MIME"text/plain", o::T2mapOptions) = print(io, _show_string(o))

"""
    T2partOptions(; <keyword arguments>)

Options structure for [`T2partSEcorr`](@ref).
This struct collects keyword arguments passed to `T2partSEcorr`, performs checks on parameter types and values, and assigns default values to unspecified parameters.

# Arguments
- `MatrixSize`: size of first 3 dimensions of input 4D T2 distribution. This argument is has no default, but is inferred automatically as `size(T2distribution)[1:3]` when calling `T2partSEcorr(T2distribution; kwargs...)`
- `nT2`:        number of T2 values in distribution. This argument is has no default, but is inferred automatically as `size(T2distribution, 4)` when calling `T2partSEcorr(T2distribution; kwargs...)`
- `T2Range`:    min and max T2 values of distribution (Default: `(10e-3, 2.0)`, Units: seconds)
- `SPWin`:      min and max T2 values of the short peak window (Default: `(10e-3, 25e-3)`, Units: seconds)
- `MPWin`:      min and max T2 values of the middle peak window (Default: `(25e-3, 200e-3)`, Units: seconds)
- `Sigmoid`:    apply sigmoidal weighting to the upper limit of the short peak window in order to smooth the hard small pool window cutoff time.
                `Sigmoid` is the delta-T2 parameter, which is the distance in seconds on either side of the `SPWin` upper limit where the sigmoid curve reaches 10% and 90% (Default: `nothing`, Units: seconds)
- `Silent`:     suppress printing to the console (Default: `false`)

See also:
* [`T2partSEcorr`](@ref)
"""
@with_kw_noshow struct T2partOptions{T} @deftype T
    legacy::Bool = false

    MatrixSize::NTuple{3,Int}
    @assert all(MatrixSize .>= 1)

    nT2::Int
    @assert nT2 > 1

    T2Range::NTuple{2,T} = !legacy ? (10e-3, 2.0) : (15e-3, 2.0) # seconds
    @assert 0.0 < T2Range[1] < T2Range[2]

    SPWin::NTuple{2,T} = !legacy ? (10e-3, 25e-3) : (14e-3, 40e-3) # seconds
    @assert SPWin[1] < SPWin[2]

    MPWin::NTuple{2,T} = !legacy ? (25e-3, 200e-3) : (40e-3, 200e-3) # seconds
    @assert MPWin[1] < MPWin[2]

    Sigmoid::Union{T,Nothing} = nothing
    @assert isnothing(Sigmoid) || Sigmoid > 0

    Silent::Bool = false
end
T2partOptions(args...; kwargs...) = T2partOptions{Float64}(args...; kwargs...)
T2partOptions(image::Array{T,4}; kwargs...) where {T} = T2partOptions{T}(;MatrixSize = size(image)[1:3], nT2 = size(image)[4], kwargs...)

function _show_string(o::T2partOptions)
    io = IOBuffer()
    print(io, "T2-part analysis settings:")
    fields = fieldnames(typeof(o))
    fields = fields[sortperm(uppercase.(string.([fields...])))] # sort alphabetically
    padlen = 1 + maximum(f -> length(string(f)), fields)
    for f in fields
        (f == :legacy) && continue
        print(io, "\n$(lpad(rpad(f, padlen), padlen + 4)): $(getfield(o,f))")
    end
    return String(take!(io))
end
# Base.show(io::IO, o::T2partOptions) = print(io, _show_string(o))
Base.show(io::IO, ::MIME"text/plain", o::T2partOptions) = print(io, _show_string(o))
