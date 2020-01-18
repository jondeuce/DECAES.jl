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
    @assert vTEparam isa Nothing || begin
        TE1 = vTEparam[1]; TE2 = vTEparam[2]; nTE1 = vTEparam[3]
        TE1 < TE2 && nTE1 < nTE && TE1 * round(Int, TE2/TE1) ≈ TE2
    end
    @assert vTEparam === nothing || error("Variable TE is not implemented")

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
    @assert SetFlipAngle isa Nothing || 0.0 < SetFlipAngle <= 180.0

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
    T2mapSEcorr(image; <keyword arguments>)
    T2mapSEcorr(image, opts::T2mapOptions)

Uses nonnegative least squares (NNLS) to compute T2 distributions in the presence of stimulated echos by optimizing the refocusing pulse flip angle.
Records parameter maps and T2 distributions for further partitioning.

# Arguments
- `image`: 4D array with intensity data as `(row, column, slice, echo)`
- A series of optional keyword argument settings which will be used to construct a [`T2mapOptions`](@ref) struct internally, or a [`T2mapOptions`](@ref) struct directly

# Outputs
- `maps`: dictionary containing 3D, 4D, or 5D parameter maps with the following fields:
    - `"gdn"`:        3D (MatrixSize)             General density = sum(T2distribution)
    - `"ggm"`:        3D (MatrixSize)             General geometric mean
    - `"gva"`:        3D (MatrixSize)             General variance
    - `"fnr"`:        3D (MatrixSize)             Fit to noise ratio = gdn / sqrt(sum(residuals.^2) / (nTE-1))
    - `"snr"`:        3D (MatrixSize)             Signal to noise ratio = maximum(signal) / std(residuals)
    - `"alpha"`:      3D (MatrixSize)             Refocusing pulse flip angle
    - `"resnorm"`:    3D (MatrixSize)             (optional) ``\\ell^2``-norm of NNLS fit residuals
    - `"decaycurve"`: 4D (MatrixSize x nTE)       (optional) Decay curve resulting from NNLS fit
    - `"mu"`:         3D (MatrixSize)             (optional) Regularization parameter from NNLS fit
    - `"chi2factor"`: 3D (MatrixSize)             (optional) ``\\chi^2`` increase factor from NNLS fit
    - `"decaybasis"`: 5D (MatrixSize x nTE x nT2) (optional) Decay basis from EPGdecaycurve
- `distributions`: 4D (MatrixSize x nT2) array with data as `(row, column, slice, T2 amplitude)` containing T2 distributions.

# Examples
```julia-repl
julia> image = DECAES.mock_image(MatrixSize = (100,100,1), nTE = 32); # mock image with size 100x100x1x32

julia> maps, dist = T2mapSEcorr(image; TE = 10e-3, Silent = true); # compute the T2-maps and T2-distribution

julia> maps
Dict{String,Array{Float64,N} where N} with 6 entries:
  "gdn"   => [10052.7 10117.6 … 10030.9 10290.4; 10110.2 10193.6 … 9953.01 10085.3; … ; 1004…
  "alpha" => [176.666 177.116 … 177.557 176.503; 177.455 177.83 … 177.558 177.639; … ; 176.6…
  "snr"   => [541.095 636.746 … 492.519 787.512; 503.934 592.39 … 455.082 509.539; … ; 448.9…
  "fnr"   => [677.631 807.875 … 626.881 1012.96; 625.444 764.197 … 571.902 653.781; … ; 569.…
  "gva"   => [0.280518 0.307561 … 0.372818 0.423089; 0.330033 0.377154 … 0.218693 0.260413; …
  "ggm"   => [0.0494424 0.0456093 … 0.0467535 0.0454226; 0.0480455 0.0444683 … 0.0473485 0.0…
```

See also:
* [`T2partSEcorr`](@ref)
* [`lsqnonneg`](@ref)
* [`lsqnonneg_reg`](@ref)
* [`lsqnonneg_lcurve`](@ref)
* [`EPGdecaycurve`](@ref)
"""
function T2mapSEcorr(image::Array{T,4}; kwargs...) where {T}
    reset_timer!(TIMER)
    out = @timeit_debug TIMER "T2mapSEcorr" begin
        T2mapSEcorr(image, T2mapOptions(image; kwargs...))
    end
    if timeit_debug_enabled()
        println("\n"); show(TIMER); println("\n")
    end
    return out
end

function T2mapSEcorr(image::Array{T,4}, opts::T2mapOptions{T}) where {T}
    # =========================================================================
    # Initialize output data structures and thread-local buffers
    # =========================================================================
    @assert size(image) == (opts.MatrixSize..., opts.nTE)

    # Print settings to terminal
    if !opts.Silent
        @info _show_string(opts)
    end
    LEGACY[] = opts.legacy

    maps = Dict{String, Array{T}}()
    maps["gdn"]    = fill(T(NaN), opts.MatrixSize...)
    maps["ggm"]    = fill(T(NaN), opts.MatrixSize...)
    maps["gva"]    = fill(T(NaN), opts.MatrixSize...)
    maps["fnr"]    = fill(T(NaN), opts.MatrixSize...)
    maps["snr"]    = fill(T(NaN), opts.MatrixSize...)
    maps["alpha"]  = fill(T(NaN), opts.MatrixSize...)
    opts.SaveResidualNorm && (maps["resnorm"]    = fill(T(NaN), opts.MatrixSize...))
    opts.SaveDecayCurve   && (maps["decaycurve"] = fill(T(NaN), opts.MatrixSize..., opts.nTE))
    opts.SaveRegParam     && (maps["mu"]         = fill(T(NaN), opts.MatrixSize...))
    opts.SaveRegParam     && (maps["chi2factor"] = fill(T(NaN), opts.MatrixSize...))
    opts.SaveNNLSBasis    && (maps["decaybasis"] = fill(T(NaN), opts.MatrixSize..., opts.nTE, opts.nT2))

    distributions  = fill(T(NaN), opts.MatrixSize..., opts.nT2)

    thread_buffers = [thread_buffer_maker(image, opts) for _ in 1:Threads.nthreads()]
    global_buffer  = global_buffer_maker(image, opts)

    # =========================================================================
    # Find the basis matrices for each flip angle
    # =========================================================================
    @timeit_debug TIMER "Initialization" begin
        for i in 1:length(thread_buffers)
            init_epg_decay_basis!(thread_buffers[i], opts)
        end
    end

    # =========================================================================
    # Process all pixels
    # =========================================================================
    LinearAlgebra.BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads

    # Run analysis in parallel
    global_buffer.loop_start_time[] = tic()
    indices = filter(I -> image[I,1] >= opts.Threshold, CartesianIndices(opts.MatrixSize))
    tforeach(indices) do I
        thread_buffer = thread_buffers[Threads.threadid()]
        voxelwise_T2_distribution!(thread_buffer, maps, distributions, image, opts, I)
        update_progress!(global_buffer, thread_buffers, opts)
    end

    LinearAlgebra.BLAS.set_num_threads(Threads.nthreads()) # Reset BLAS threads
    LEGACY[] = false

    return @ntuple(maps, distributions)
end

# =========================================================
# Main loop function
# =========================================================
function voxelwise_T2_distribution!(thread_buffer, maps, distributions, image, opts::T2mapOptions, I::CartesianIndex)
    # Skip low signal voxels
    @inbounds if image[I,1] < opts.Threshold
        return nothing
    end

    # Extract decay curve from the voxel
    @inbounds for j in 1:opts.nTE
        thread_buffer.decay_data[j] = image[I,j]
    end

    # Find optimum flip angle
    if opts.SetFlipAngle === nothing
        @timeit_debug TIMER "Optimize Flip Angle" begin
            optimize_flip_angle!(thread_buffer, opts)
        end
    end

    # Fit decay basis using optimized alpha
    if opts.SetFlipAngle === nothing
        @timeit_debug TIMER "Compute Final NNLS Basis" begin
            epg_decay_basis!(thread_buffer, opts)
        end
    end

    # Calculate T2 distribution and map parameters
    @timeit_debug TIMER "Calculate T2 Dist" begin
        fit_T2_distribution!(thread_buffer, opts)
    end

    # Save loop results to outputs
    save_results!(thread_buffer, maps, distributions, opts, I)

    return nothing
end

# =========================================================
# Flip angle optimization
# =========================================================
function optimize_flip_angle_work(o::T2mapOptions{T}) where {T}
    work = (
        nnls_work  = lsqnonneg_work(zeros(T, o.nTE, o.nT2), zeros(T, o.nTE)),
        chi2_alpha = zeros(T, o.nRefAngles),
        decay_pred = zeros(T, o.nTE),
        residuals  = zeros(T, o.nTE),
    )
    return work
end

function optimize_flip_angle!(thread_buffer, o::T2mapOptions)
    @unpack flip_angle_work, decay_basis_set, flip_angles, decay_data, T2_times = thread_buffer
    @unpack nnls_work, chi2_alpha, decay_pred, residuals = flip_angle_work
    @unpack alpha_opt, chi2_alpha_opt = thread_buffer

    function chi2_alpha_fun(flip_angles, i)
        # First argument `flip_angles` has been used implicitly in creating `decay_basis_set` already
        lsqnonneg!(nnls_work, decay_basis_set[i], decay_data)
        T2_dist_nnls = nnls_work.x
        mul!(decay_pred, decay_basis_set[i], T2_dist_nnls)
        residuals .= decay_data .- decay_pred
        chi2 = sum(abs2, residuals)
        return chi2
    end

    @timeit_debug TIMER "Surrogate Spline Opt" begin
        # Find the minimum chi-squared and the corresponding angle
        alpha_opt[], chi2_alpha_opt[] = surrogate_spline_opt(chi2_alpha_fun, flip_angles, o.nRefAnglesMin)
    end

    return nothing
end

# =========================================================
# EPG decay curve fitting
# =========================================================
function epg_decay_basis_work(o::T2mapOptions{T}) where {T}
    return o.vTEparam === nothing ?
        EPGdecaycurve_work(T, o.nTE) :
        EPGdecaycurve_vTE_work(T, o.vTEparam...)
end

function init_epg_decay_basis!(thread_buffer, o::T2mapOptions)
    @unpack decay_basis_work, decay_basis_set, decay_basis, flip_angles, T2_times = thread_buffer

    if o.SetFlipAngle === nothing
        # Loop to compute basis for each angle
        @inbounds for i = 1:o.nRefAngles
            epg_decay_basis!(decay_basis_work, decay_basis_set[i], flip_angles[i], T2_times, o)
        end
    else
        # Compute basis for fixed flip angle `SetFlipAngle`
        epg_decay_basis!(decay_basis_work, decay_basis, o.SetFlipAngle, T2_times, o)
    end

    return nothing
end

function epg_decay_basis!(thread_buffer, o::T2mapOptions)
    @unpack decay_basis_work, decay_basis, alpha_opt, T2_times = thread_buffer
    epg_decay_basis!(decay_basis_work, decay_basis, alpha_opt[], T2_times, o)
    return nothing
end

function epg_decay_basis!(decay_basis_work, decay_basis, flip_angle, T2_times, o::T2mapOptions)
    @unpack decay_curve = decay_basis_work

    # Compute the NNLS basis over T2 space
    @inbounds for j = 1:o.nT2
        @timeit_debug TIMER "EPGdecaycurve!" begin
            if o.vTEparam !== nothing
                EPGdecaycurve_vTE!(decay_basis_work, o.nTE, flip_angle, o.vTEparam..., T2_times[j], o.T1, o.RefConAngle)
            else
                EPGdecaycurve!(decay_basis_work, o.nTE, flip_angle, o.TE, T2_times[j], o.T1, o.RefConAngle)
            end
        end
        for i = 1:o.nTE
            decay_basis[i,j] = decay_curve[i]
        end
    end

    return nothing
end

# =========================================================
# T2-distribution fitting
# =========================================================
function T2_distribution_work(o::T2mapOptions{T}) where {T}
    decay_basis_buffer = zeros(T, o.nTE, o.nT2)
    decay_data_buffer  = zeros(T, o.nTE)

    work = if o.Reg == "no"
        # Fit T2 distribution using unregularized NNLS
        lsqnonneg_work(decay_basis_buffer, decay_data_buffer)
    elseif o.Reg == "chi2"
        # Fit T2 distribution using chi2 based regularized NNLS
        lsqnonneg_reg_work(decay_basis_buffer, decay_data_buffer)
    elseif o.Reg == "lcurve"
        # Fit T2 distribution using lcurve based regularization
        lsqnonneg_lcurve_work(decay_basis_buffer, decay_data_buffer)
    end

    return work
end

function fit_T2_distribution!(thread_buffer, o::T2mapOptions{T}) where {T}
    @unpack T2_dist_work, decay_basis, decay_data = thread_buffer
    @unpack T2_dist, mu_opt, chi2fact_opt = thread_buffer

    if o.Reg == "no"
        # Fit T2 distribution using unregularized NNLS
        lsqnonneg!(T2_dist_work, decay_basis, decay_data)
        T2_dist       .= T2_dist_work.x
        mu_opt[]       = T(NaN)
        chi2fact_opt[] = one(T)
    elseif o.Reg == "chi2"
        # Fit T2 distribution using chi2 based regularized NNLS
        lsqnonneg_reg!(T2_dist_work, decay_basis, decay_data, o.Chi2Factor)
        T2_dist       .= T2_dist_work.x
        mu_opt[]       = T2_dist_work.mu_opt[]
        chi2fact_opt[] = T2_dist_work.chi2fact_opt[]
    elseif o.Reg == "lcurve"
        # Fit T2 distribution using lcurve based regularization
        lsqnonneg_lcurve!(T2_dist_work, decay_basis, decay_data)
        T2_dist       .= T2_dist_work.x
        mu_opt[]       = T2_dist_work.mu_opt[]
        chi2fact_opt[] = T2_dist_work.chi2fact_opt[]
    end

    return nothing
end

# =========================================================
# Save thread local results to output maps
# =========================================================
function save_results!(thread_buffer, maps, distributions, o::T2mapOptions, I::CartesianIndex)
    @unpack T2_dist, T2_times, logT2_times, alpha_opt, mu_opt, chi2fact_opt, decay_data, decay_basis = thread_buffer
    @unpack decay_calc, residuals, gva_buf, curr_count = thread_buffer

    # Compute and save parameters of distribution
    @inbounds begin
        @unpack gdn, ggm, gva, fnr, snr, alpha = maps
        mul!(decay_calc, decay_basis, T2_dist)
        residuals .= decay_calc .- decay_data
        gdn[I] = sum(T2_dist)
        ggm[I] = exp(dot(T2_dist, logT2_times) / sum(T2_dist))
        gva_buf .= (logT2_times .- log(ggm[I])).^2 .* T2_dist
        gva[I] = exp(sum(gva_buf) / sum(T2_dist)) - 1
        fnr[I] = sum(T2_dist) / sqrt(sum(abs2, residuals)/(o.nTE-1))
        snr[I] = maximum(decay_data) / std(residuals)
        alpha[I] = alpha_opt[]
    end

    # Save distribution
    @inbounds for j in 1:o.nT2
        distributions[I,j] = T2_dist[j]
    end

    # Optionally save regularization parameters
    if o.SaveRegParam
        @unpack mu, chi2factor = maps
        @inbounds mu[I], chi2factor[I] = mu_opt[], chi2fact_opt[]
    end

    # Optionally save ℓ²-norm of residuals
    if o.SaveResidualNorm
        @unpack resnorm = maps
        @inbounds resnorm[I] = sqrt(sum(abs2, residuals))
    end

    # Optionally save signal decay curve from fit
    if o.SaveDecayCurve
        @unpack decaycurve = maps
        @inbounds for j in 1:o.nTE
            decaycurve[I,j] = decay_calc[j]
        end
    end

    # Optionally save NNLS basis
    if o.SaveNNLSBasis
        @unpack decaybasis = maps
        @inbounds for J in CartesianIndices((o.nTE, o.nT2))
            decaybasis[I,J] = decay_basis[J]
        end
    end

    # Increment progress counter
    curr_count[] += 1

    return nothing
end

# =========================================================
# Utility functions
# =========================================================
function thread_buffer_maker(image::Array{T,4}, o::T2mapOptions{T}) where {T}
    buffer = (
        curr_count       = Ref(0),
        total_count      = sum(≥(o.Threshold), @views(image[:,:,:,1])),
        T2_times         = logrange(o.T2Range..., o.nT2),
        logT2_times      = log.(logrange(o.T2Range..., o.nT2)),
        flip_angles      = range(o.MinRefAngle, T(180); length = o.nRefAngles),
        decay_basis_set  = [zeros(T, o.nTE, o.nT2) for _ in 1:o.nRefAngles],
        decay_basis      = zeros(T, o.nTE, o.nT2),
        decay_data       = zeros(T, o.nTE),
        decay_calc       = zeros(T, o.nTE),
        residuals        = zeros(T, o.nTE),
        decay_basis_work = epg_decay_basis_work(o),
        flip_angle_work  = optimize_flip_angle_work(o),
        T2_dist_work     = T2_distribution_work(o),
        alpha_opt        = Ref(ifelse(o.SetFlipAngle === nothing, T(NaN), o.SetFlipAngle)),
        chi2_alpha_opt   = Ref(T(NaN)),
        T2_dist          = zeros(T, o.nT2),
        gva_buf          = zeros(T, o.nT2),
        mu_opt           = Ref(T(NaN)),
        chi2fact_opt     = Ref(T(NaN)),
    )
    return buffer
end

function global_buffer_maker(image::Array{T,4}, o::T2mapOptions{T}) where {T}
    buffer = (
        global_lock     = Threads.SpinLock(),
        loop_start_time = Ref(0.0),
        last_print_time = Ref(-Inf),
        done_printing   = Ref(false),
        trigger_print   = Threads.Atomic{Bool}(false),
        print_message   = Ref(""),
        total_count     = sum(≥(o.Threshold), @views(image[:,:,:,1])),
        running_stats   = RunningLinReg{Float64}(),
    )
    return buffer
end

function update_progress!(global_buffer, thread_buffers, o::T2mapOptions)
    @unpack global_lock, loop_start_time, last_print_time, done_printing = global_buffer
    @unpack running_stats, total_count, trigger_print, print_message = global_buffer

    if !o.Silent && !done_printing[]
        # Update global buffer every `print_interval` seconds
        print_interval = 15.0
        lock(global_lock) do
            time_elapsed = toc(loop_start_time[])
            print_now = time_elapsed - last_print_time[] ≥ print_interval
            curr_count = sum(buf -> buf.curr_count[], thread_buffers)
            finished = curr_count == total_count

            if !done_printing[] && (finished || print_now)
                last_print_time[] = time_elapsed
                done_printing[] = finished

                # Update completion time estimate
                push!(running_stats, curr_count, time_elapsed)
                est_complete = running_stats(total_count)
                est_remaining = done_printing[] ? 0.0 :
                    time_elapsed < print_interval || length(running_stats) < 2 ? NaN :
                    max(est_complete - time_elapsed, 0.0)

                # Set print message, and trigger printing below
                print_message[] = join([
                    "[" * lpad(floor(Int, 100 * (curr_count / total_count)), 3) * "%]",
                    "Elapsed Time: " * pretty_time(time_elapsed),
                    "Estimated Time Remaining: " * pretty_time(est_remaining),
                ], " -- ")
                trigger_print[] = true
            end
        end

        # Print message outside of spin lock to avoid blocking I/O
        if Threads.atomic_cas!(trigger_print, true, false)
            @info print_message[]
        end
    end

    return nothing
end
