"""
    T2mapSEcorr(image; <keyword arguments>)
    T2mapSEcorr(image, opts::T2mapOptions)

Uses nonnegative least squares (NNLS) to compute T2 distributions in the presence of stimulated echos by optimizing the refocusing pulse flip angle.
Records parameter maps and T2 distributions for further partitioning.

# Arguments
- `image`: 4D array with intensity data as `(row, column, slice, echo)`
- A series of optional keyword argument settings which will be used to construct a [`T2mapOptions`](@ref) struct internally, or a [`T2mapOptions`](@ref) struct directly

# Outputs
- `maps`: dictionary containing parameter maps with the following fields:
    - **Default Fields**
        - `"echotimes"`     Echo times of time signal (length `nTE` 1D array)
        - `"t2times"`       T2 times corresponding to T2-distributions (length `nT2` 1D array)
        - `"refangleset"`   Refocusing angles used during flip angle optimization (length `nRefAngles` 1D array by default; scalar if `SetFlipAngle` is used)
        - `"decaybasisset"` Decay basis sets corresponding to `"refangleset"` (`nTE x nT2 x nRefAngles` 3D array by default; `nTE x nT2` 2D array if `SetFlipAngle` is used)
        - `"gdn"`:          Map of general density = sum(T2distribution) (`MatrixSize` 3D array)
        - `"ggm"`:          Map of general geometric mean of T2-distribution (`MatrixSize` 3D array)
        - `"gva"`:          Map of general variance (`MatrixSize` 3D array)
        - `"fnr"`:          Map of fit to noise ratio = gdn / sqrt(sum(residuals.^2) / (nTE-1)) (`MatrixSize` 3D array)
        - `"snr"`:          Map of signal to noise ratio = maximum(signal) / std(residuals) (`MatrixSize` 3D array)
        - `"alpha"`:        Map of optimized refocusing pulse flip angle (`MatrixSize` 3D array)
    - **Optional Fields**
        - `"resnorm"`:      ``\\ell^2``-norm of NNLS fit residuals; see `SaveResidualNorm` option (`MatrixSize` 3D array)
        - `"decaycurve"`:   Signal decay curve resulting from NNLS fit; see `SaveDecayCurve` option (`MatrixSize x nTE` 4D array)
        - `"mu"`:           Regularization parameter used during from NNLS fit; see `SaveRegParam` option (`MatrixSize` 3D array)
        - `"chi2factor"`:   ``\\chi^2`` increase factor relative to unregularized NNLS fit; see `SaveRegParam` option (`MatrixSize` 3D array)
        - `"decaybasis"`:   Decay bases resulting from flip angle optimization; see `SaveNNLSBasis` option (`MatrixSize x nTE x nT2` 5D array, or `nTE x nT2` 2D array if `SetFlipAngle` is used)
- `distributions`: T2-distribution array with data as `(row, column, slice, T2 amplitude)` (`MatrixSize x nT2` 4D array)

# Examples
```julia-repl
julia> image = DECAES.mock_image(MatrixSize = (100,100,1), nTE = 32); # mock image with size 100x100x1x32

julia> maps, dist = T2mapSEcorr(image; TE = 10e-3, Silent = true); # compute the T2-maps and T2-distribution

julia> maps
Dict{String,Array{Float64,N} where N} with 10 entries:
  "echotimes"     => [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1  …  0.23, …
  "t2times"       => [0.01, 0.0114551, 0.013122, 0.0150315, 0.0172188, 0.0197244, 0.022594…
  "refangleset"   => [50.0, 54.1935, 58.3871, 62.5806, 66.7742, 70.9677, 75.1613, 79.3548,…
  "gdn"           => [1.86624e5 1.1901e5 … 1.3624e5 1.72367e5; 1.49169e5 1.24536e5 … 1.623…
  "fnr"           => [479.252 451.153 … 492.337 522.685; 328.645 440.903 … 473.648 416.366…
  "alpha"         => [165.011 164.049 … 163.291 164.945; 164.849 166.599 … 163.674 165.679…
  "gva"           => [0.485815 0.401323 … 0.340727 0.436184; 0.321887 0.317154 … 0.278256 …
  "ggm"           => [0.0494864 0.0506017 … 0.0518207 0.0501371; 0.0530299 0.052983 … 0.05…
  "snr"           => [368.732 349.29 … 384.417 406.59; 258.914 349.542 … 373.792 331.304; …
  "decaybasisset" => [0.0277684 0.0315296 … 0.0750511 0.0751058; 0.0469882 0.0536334 … 0.1…
```

See also:
* [`T2partSEcorr`](@ref)
* [`lsqnonneg`](@ref)
* [`lsqnonneg_reg`](@ref)
* [`lsqnonneg_lcurve`](@ref)
* [`EPGdecaycurve`](@ref)
"""
function T2mapSEcorr(image::Array{T,4}; kwargs...) where {T}
    map(reset_timer!, THREAD_LOCAL_TIMERS)
    out = @timeit_debug TIMER() "T2mapSEcorr" begin
        T2mapSEcorr(image, T2mapOptions(image; kwargs...))
    end
    if timeit_debug_enabled()
        display(THREAD_LOCAL_TIMERS[1]) #TODO
        display(TimerOutputs.flatten(THREAD_LOCAL_TIMERS[1])) #TODO
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

    maps = Dict{String, Any}()
    distributions  = fill(T(NaN), opts.MatrixSize..., opts.nT2)
    thread_buffers = [thread_buffer_maker(image, opts) for _ in 1:Threads.nthreads()]
    global_buffer  = global_buffer_maker(image, opts)

    # =========================================================================
    # Initialization
    # =========================================================================
    @timeit_debug TIMER() "Initialization" begin
        for i in 1:length(thread_buffers)
            # Compute lookup table of EPG decay bases
            init_epg_decay_basis!(thread_buffers[i], opts)
        end
        # Initialize output maps
        init_output_t2maps!(thread_buffers[1], maps, opts)
    end

    # =========================================================================
    # Process all pixels
    # =========================================================================
    LinearAlgebra.BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads

    # Run analysis in parallel
    global_buffer.loop_start_time[] = tic()
    indices = filter(I -> image[I,1] > opts.Threshold, CartesianIndices(opts.MatrixSize))
    tforeach(indices; blocksize = 64) do I
        thread_buffer = thread_buffers[Threads.threadid()]
        voxelwise_T2_distribution!(thread_buffer, maps, distributions, image, opts, I)
        update_progress!(global_buffer, thread_buffers, opts)
    end

    LinearAlgebra.BLAS.set_num_threads(Threads.nthreads()) # Reset BLAS threads
    LEGACY[] = false

    return @ntuple(maps, distributions)
end

function init_output_t2maps!(thread_buffer, maps, opts::T2mapOptions{T}) where {T}
    @unpack T2_times, flip_angles, decay_basis, decay_basis_set = thread_buffer

    # Misc. processing parameters
    maps["echotimes"] = convert(Vector{T}, copy(opts.TE .* (1:opts.nTE))) #TODO update if vTEparam is implemented
    maps["t2times"]   = convert(Vector{T}, copy(T2_times))

    if isnothing(opts.SetFlipAngle)
        maps["refangleset"]   = convert(Vector{T}, copy(flip_angles))
        maps["decaybasisset"] = convert(Array{T,3}, reshape(reduce(hcat, decay_basis_set), size(decay_basis_set[1])..., length(decay_basis_set)))
    else
        maps["refangleset"]   = T(opts.SetFlipAngle)
        maps["decaybasisset"] = convert(Matrix{T}, copy(decay_basis))
    end

    # Default output maps
    maps["gdn"] = fill(T(NaN), opts.MatrixSize...)
    maps["ggm"] = fill(T(NaN), opts.MatrixSize...)
    maps["gva"] = fill(T(NaN), opts.MatrixSize...)
    maps["fnr"] = fill(T(NaN), opts.MatrixSize...)
    maps["snr"] = fill(T(NaN), opts.MatrixSize...)
    maps["alpha"] = fill(T(NaN), opts.MatrixSize...)

    # Optional output maps
    if opts.SaveResidualNorm
        maps["resnorm"] = fill(T(NaN), opts.MatrixSize...)
    end

    if opts.SaveDecayCurve
        maps["decaycurve"] = fill(T(NaN), opts.MatrixSize..., opts.nTE)
    end

    if opts.SaveRegParam
        maps["mu"] = fill(T(NaN), opts.MatrixSize...)
        maps["chi2factor"] = fill(T(NaN), opts.MatrixSize...)
    end

    if opts.SaveNNLSBasis
        if isnothing(opts.SetFlipAngle)
            maps["decaybasis"] = fill(T(NaN), opts.MatrixSize..., opts.nTE, opts.nT2) # unique decay basis set for each voxel
        else
            maps["decaybasis"] = convert(Matrix{T}, copy(decay_basis)) # single decay basis set used for all voxels
        end
    end

    return nothing
end

# =========================================================
# Main loop function
# =========================================================
function voxelwise_T2_distribution!(thread_buffer, maps, distributions, image, opts::T2mapOptions, I::CartesianIndex)
    # Skip low signal voxels
    @inbounds if !(image[I,1] > opts.Threshold)
        return nothing
    end

    # Extract decay curve from the voxel
    @inbounds for j in 1:opts.nTE
        thread_buffer.decay_data[j] = image[I,j]
    end

    # Find optimum flip angle
    if isnothing(opts.SetFlipAngle)
        @timeit_debug TIMER() "Optimize Flip Angle" begin
            optimize_flip_angle!(thread_buffer, opts)
        end
    end

    # Fit decay basis using optimized alpha
    if isnothing(opts.SetFlipAngle)
        @timeit_debug TIMER() "Compute Final NNLS Basis" begin
            epg_decay_basis!(thread_buffer, opts)
        end
    end

    # Calculate T2 distribution and map parameters
    @timeit_debug TIMER() "Calculate T2 Dist" begin
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
        @timeit_debug TIMER() "lsqnonneg!" begin
            lsqnonneg!(nnls_work, decay_basis_set[i], decay_data)
        end
        T2_dist_nnls = nnls_work.x
        mul!(decay_pred, decay_basis_set[i], T2_dist_nnls)
        residuals .= decay_data .- decay_pred
        chi2 = sum(abs2, residuals)
        return chi2
    end

    @timeit_debug TIMER() "Surrogate Spline Opt" begin
        # Find the minimum chi-squared and the corresponding angle
        alpha_opt[], chi2_alpha_opt[] = surrogate_spline_opt(chi2_alpha_fun, flip_angles, o.nRefAnglesMin)
    end

    return nothing
end

# =========================================================
# EPG decay curve fitting
# =========================================================
function epg_decay_curve_work(o::T2mapOptions{T}) where {T}
    return EPGdecaycurve_work(T, o.nTE)
    # return EPGdecaycurve_vTE_work(T, o.vTEparam...) #TODO update if vTEparam is implemented
end

function init_epg_decay_basis!(thread_buffer, o::T2mapOptions)
    @unpack decay_curve_work, decay_basis_set, decay_basis, flip_angles, T2_times = thread_buffer

    if isnothing(o.SetFlipAngle)
        # Loop to compute basis for each angle
        @inbounds for i in 1:o.nRefAngles
            epg_decay_basis!(decay_curve_work, decay_basis_set[i], flip_angles[i], T2_times, o)
        end
    else
        # Compute basis for fixed flip angle `SetFlipAngle`
        epg_decay_basis!(decay_curve_work, decay_basis, o.SetFlipAngle, T2_times, o)
    end

    return nothing
end

function epg_decay_basis!(thread_buffer, o::T2mapOptions)
    @unpack decay_curve_work, decay_basis, alpha_opt, T2_times = thread_buffer
    epg_decay_basis!(decay_curve_work, decay_basis, alpha_opt[], T2_times, o)
    return nothing
end

function epg_decay_basis!(decay_curve_work, decay_basis::AbstractMatrix{T}, flip_angle::T, T2_times::AbstractVector{T}, o::T2mapOptions{T}) where {T}

    # Compute the NNLS basis over T2 space
    @timeit_debug TIMER() "EPGdecaycurve!" begin
        @inbounds for j in 1:o.nT2
            decay_curve = NNLS.fastview(decay_basis, 1+(j-1)*o.nTE, o.nTE)
            EPGdecaycurve!(decay_curve, decay_curve_work, flip_angle, o.TE, T2_times[j], o.T1, o.RefConAngle)
            # EPGdecaycurve_vTE!(decay_curve_work, o.nTE, flip_angle, o.vTEparam..., T2_times[j], o.T1, o.RefConAngle) #TODO update if vTEparam is implemented
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
    if o.SaveNNLSBasis && isnothing(o.SetFlipAngle)
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
        total_count      = sum(>(o.Threshold), @views(image[:,:,:,1])),
        T2_times         = logrange(o.T2Range..., o.nT2),
        logT2_times      = log.(logrange(o.T2Range..., o.nT2)),
        flip_angles      = range(o.MinRefAngle, T(180); length = o.nRefAngles),
        decay_basis_set  = [zeros(T, o.nTE, o.nT2) for _ in 1:o.nRefAngles],
        decay_basis      = zeros(T, o.nTE, o.nT2),
        decay_data       = zeros(T, o.nTE),
        decay_calc       = zeros(T, o.nTE),
        residuals        = zeros(T, o.nTE),
        decay_curve_work = epg_decay_curve_work(o),
        flip_angle_work  = optimize_flip_angle_work(o),
        T2_dist_work     = T2_distribution_work(o),
        alpha_opt        = Ref(ifelse(isnothing(o.SetFlipAngle), T(NaN), o.SetFlipAngle)),
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
        total_count     = sum(>(o.Threshold), @views(image[:,:,:,1])),
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
