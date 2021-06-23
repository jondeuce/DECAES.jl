"""
    T2mapSEcorr([io = stderr,] image::Array{T,4}; <keyword arguments>)
    T2mapSEcorr([io = stderr,] image::Array{T,4}, opts::T2mapOptions{T})

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

julia> maps, dist = T2mapSEcorr(image; TE = 10e-3, nT2 = 40, T2Range = (10e-3, 2.0), Reg = "lcurve", Silent = true); # compute the T2-maps and T2-distribution

julia> maps
Dict{String, Any} with 10 entries:
  "echotimes"     => [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1  …  0.23, …
  "t2times"       => [0.01, 0.0114551, 0.013122, 0.0150315, 0.0172188, 0.0197244, 0.022594…
  "refangleset"   => [50.0, 54.1935, 58.3871, 62.5806, 66.7742, 70.9677, 75.1613, 79.3548,…
  "gdn"           => [185326.0 1.26487e5 … 1.85322e5 1.025e5; 1.48254e5 1.91871e5 … 1.5159…
  "fnr"           => [460.929 398.485 … 415.599 428.055; 316.05 340.206 … 372.7 410.487; ……
  "alpha"         => [165.348 164.657 … 163.031 166.735; 165.212 164.994 … 165.6 165.094; …
  "gva"           => [0.438218 0.330969 … 0.367929 0.350691; 0.287478 0.307593 … 0.3892 0.…
  "ggm"           => [0.0501681 0.0518976 … 0.0509003 0.0519268; 0.0535984 0.0528403 … 0.0…
  "snr"           => [357.763 317.439 … 322.451 340.88; 250.674 275.185 … 294.411 324.464;…
  "decaybasisset" => [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0…
```

See also:
* [`T2partSEcorr`](@ref)
* [`lsqnonneg`](@ref)
* [`lsqnonneg_chi2`](@ref)
* [`lsqnonneg_gcv`](@ref)
* [`lsqnonneg_lcurve`](@ref)
* [`EPGdecaycurve`](@ref)
"""
T2mapSEcorr(image::Array{T,4}; kwargs...) where {T} = T2mapSEcorr(stderr, image; kwargs...)
T2mapSEcorr(io::IO, image::Array{T,4}; kwargs...) where {T} = T2mapSEcorr(io, image, T2mapOptions(image; kwargs...))
T2mapSEcorr(image::Array{T,4}, opts::T2mapOptions{T}) where {T} = T2mapSEcorr(stderr, image, opts)
T2mapSEcorr(io::IO, image::Array{T,4}, opts::T2mapOptions{T}) where {T} = @timeit_debug TIMER() "T2mapSEcorr" T2mapSEcorr_(io, image, opts)

function T2mapSEcorr_(io::IO, image::Array{T,4}, opts::T2mapOptions{T}) where {T}
    # =========================================================================
    # Initialize output data structures and thread-local buffers
    # =========================================================================
    @assert size(image) == (opts.MatrixSize..., opts.nTE)

    # Print settings to terminal
    !opts.Silent && printbody(io, _show_string(opts))
    LEGACY[] = opts.legacy

    # =========================================================================
    # Initialization
    # =========================================================================
    @timeit_debug TIMER() "Initialization" begin
        # Initialize output maps and distributions
        maps = init_output_t2maps(image, opts)
        distributions = init_output_t2distributions(image, opts)

        # Initialize reusable temporary buffers
        thread_buffers = [thread_buffer_maker(image, opts) for _ in 1:Threads.nthreads()]
        for i in 1:length(thread_buffers)
            # Compute lookup table of EPG decay bases
            init_epg_decay_basis!(thread_buffers[i], opts)
        end
    end

    # =========================================================================
    # Process all pixels
    # =========================================================================
    LinearAlgebra.BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads

    # Run analysis in parallel
    indices = filter(I -> image[I,1] > opts.Threshold, CartesianIndices(opts.MatrixSize))
    blocksize = 64
    if opts.Silent
        tforeach(indices; blocksize = blocksize) do I
            thread_buffer = thread_buffers[Threads.threadid()]
            voxelwise_T2_distribution!(thread_buffer, maps, distributions, image, opts, I)
        end
    else
        bigblocks = Iterators.partition(indices, 8 * Threads.nthreads() * blocksize) .|> copy
        progmeter = DECAESProgress(io, length(bigblocks), "Computing T2-Distribution: "; dt = 5.0)
        for bigblock in bigblocks
            tforeach(bigblock; blocksize = blocksize) do I
                thread_buffer = thread_buffers[Threads.threadid()]
                voxelwise_T2_distribution!(thread_buffer, maps, distributions, image, opts, I)
            end
            next!(progmeter)
        end
    end

    LinearAlgebra.BLAS.set_num_threads(Threads.nthreads()) # Reset BLAS threads
    LEGACY[] = false

    return (; maps, distributions)
end

function init_output_t2distributions(image::Array{T,4}, opts::T2mapOptions{T}) where {T}
    distributions = fill(T(NaN), opts.MatrixSize..., opts.nT2)
    return distributions
end

function init_output_t2maps(image::Array{T,4}, opts::T2mapOptions{T}) where {T}
    maps = Dict{String, Any}()
    init_output_t2maps!(thread_buffer_maker(image, opts), maps, opts)
    return maps
end

function init_output_t2maps!(thread_buffer, maps, opts::T2mapOptions{T}) where {T}
    @unpack T2_times, flip_angles, decay_basis, decay_basis_set = thread_buffer

    # Misc. processing parameters
    maps["echotimes"] = convert(Vector{T}, copy(opts.TE .* (1:opts.nTE))) #TODO update if vTEparam is implemented
    maps["t2times"]   = convert(Vector{T}, copy(T2_times))

    if opts.SetFlipAngle === nothing
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
        if opts.SetFlipAngle === nothing
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
    @inbounds @simd for j in 1:opts.nTE
        thread_buffer.decay_data[j] = image[I,j]
    end

    # Find optimum flip angle
    if opts.SetFlipAngle === nothing
        @timeit_debug TIMER() "Optimize Flip Angle" begin
            optimize_flip_angle!(thread_buffer, opts)
        end
    end

    # Fit decay basis using optimized alpha
    if opts.SetFlipAngle === nothing
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
            solve!(nnls_work, decay_basis_set[i], decay_data)
            return chi2(nnls_work)
        end
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

    if o.SetFlipAngle === nothing
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
            # decay_curve = @views decay_basis[:,j] # @views no longer allocates since v1.5
            decay_curve = uview(decay_basis, :, j) # `UnsafeArrays.uview` is still a bit faster than `Base.view`
            EPGdecaycurve!(decay_curve, decay_curve_work, flip_angle, o.TE, T2_times[j], o.T1, o.RefConAngle)
            # EPGdecaycurve_vTE!(decay_curve_work, o.nTE, flip_angle, o.vTEparam..., T2_times[j], o.T1, o.RefConAngle) #TODO update if vTEparam is implemented
        end
    end

    return nothing
end

# =========================================================
# T2-distribution fitting
# =========================================================
function T2_distribution_work(decay_basis, decay_data, o::T2mapOptions{T}) where {T}
    work = if o.Reg == "none"
        # Fit T2 distribution using unregularized NNLS
        lsqnonneg_work(decay_basis, decay_data)
    elseif o.Reg == "chi2"
        # Fit T2 distribution using chi2-based regularized NNLS
        lsqnonneg_chi2_work(decay_basis, decay_data)
    elseif o.Reg == "gcv"
        # Fit T2 distribution using GCV-based regularized NNLS
        lsqnonneg_gcv_work(decay_basis, decay_data)
    elseif o.Reg == "lcurve"
        # Fit T2 distribution using L-curve-based regularized NNLS
        lsqnonneg_lcurve_work(decay_basis, decay_data)
    end

    return work
end

function fit_T2_distribution!(thread_buffer, o::T2mapOptions{T}) where {T}
    @unpack T2_dist_work, decay_basis, decay_data = thread_buffer
    @unpack T2_dist, mu_opt, chi2fact_opt = thread_buffer

    @unpack x, mu, chi2factor = if o.Reg == "none"
        # Fit T2 distribution using unregularized NNLS
        (; x = lsqnonneg!(T2_dist_work), mu = zero(T), chi2factor = one(T))
    elseif o.Reg == "chi2"
        # Fit T2 distribution using chi2 based regularized NNLS
        lsqnonneg_chi2!(T2_dist_work, o.Chi2Factor)
    elseif o.Reg == "gcv"
        # Fit T2 distribution using gcv based regularization
        lsqnonneg_gcv!(T2_dist_work)
    elseif o.Reg == "lcurve"
        # Fit T2 distribution using lcurve based regularization
        lsqnonneg_lcurve!(T2_dist_work)
    end

    copyto!(T2_dist, x)
    mu_opt[] = mu
    chi2fact_opt[] = chi2factor

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
    @inbounds @simd for j in 1:o.nT2
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
        @inbounds @simd for j in 1:o.nTE
            decaycurve[I,j] = decay_calc[j]
        end
    end

    # Optionally save NNLS basis
    if o.SaveNNLSBasis && o.SetFlipAngle === nothing
        @unpack decaybasis = maps
        @inbounds @simd for J in CartesianIndices((o.nTE, o.nT2))
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
    decay_basis = zeros(T, o.nTE, o.nT2)
    decay_data  = zeros(T, o.nTE)
    buffer = (
        curr_count       = Ref(0),
        total_count      = sum(>(o.Threshold), @views(image[:,:,:,1])),
        T2_times         = logrange(o.T2Range..., o.nT2),
        logT2_times      = log.(logrange(o.T2Range..., o.nT2)),
        flip_angles      = range(o.MinRefAngle, T(180); length = o.nRefAngles),
        decay_basis_set  = [zeros(T, o.nTE, o.nT2) for _ in 1:o.nRefAngles],
        decay_basis      = decay_basis,
        decay_data       = decay_data,
        decay_calc       = zeros(T, o.nTE),
        residuals        = zeros(T, o.nTE),
        decay_curve_work = epg_decay_curve_work(o),
        flip_angle_work  = optimize_flip_angle_work(o),
        T2_dist_work     = T2_distribution_work(decay_basis, decay_data, o),
        alpha_opt        = Ref(o.SetFlipAngle === nothing ? T(NaN) : o.SetFlipAngle),
        chi2_alpha_opt   = Ref(T(NaN)),
        T2_dist          = zeros(T, o.nT2),
        gva_buf          = zeros(T, o.nT2),
        mu_opt           = Ref(T(NaN)),
        chi2fact_opt     = Ref(T(NaN)),
    )
    return buffer
end
