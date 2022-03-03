"""
    T2partSEcorr(T2distributions::Array{T,4}; <keyword arguments>)
    T2partSEcorr(T2distributions::Array{T,4}, opts::T2partOptions{T})

Analyzes T2 distributions produced by [`T2mapSEcorr`](@ref) to produce data maps of a series of parameters.

# Arguments
- `T2distributions`: 4D array with data as `(row, column, slice, T2 amplitude)`
- A series of optional keyword argument settings which will be used to construct a [`T2partOptions`](@ref) struct internally, or a [`T2partOptions`](@ref) struct directly

# Ouputs
- `maps`: a dictionary containing the following 3D data maps as fields:
    - `"sfr"`: small pool fraction, e.g. myelin water fraction (`MatrixSize` 3D array)
    - `"sgm"`: small pool geometric mean T2 (`MatrixSize` 3D array)
    - `"mfr"`: medium pool fraction, e.g. intra/extracellular water fraction (`MatrixSize` 3D array)
    - `"mgm"`: medium pool geometric mean T2 (`MatrixSize` 3D array)

# Examples
```julia-repl
julia> dist = DECAES.mock_T2_dist(MatrixSize = (100,100,1), nT2 = 40); # mock distribution with size 100x100x1x40

julia> maps = T2partSEcorr(dist; T2Range = (10e-3, 2.0), SPWin = (10e-3, 25e-3), MPWin = (25e-3, 200e-3), Silent = true); # compute T2-parts maps

julia> maps
Dict{String, Any} with 4 entries:
  "sgm" => [0.0120849 0.0103704 … 0.01 0.0114649; 0.0107563 0.0110531 … 0.0…
  "mfr" => [0.907028 0.869309 … 0.854094 0.898737; 0.889522 0.855133 … 0.86…
  "sfr" => [0.0929724 0.130691 … 0.145906 0.101263; 0.110478 0.144867 … 0.1…
  "mgm" => [0.0608452 0.0623255 … 0.063191 0.0613871; 0.0621637 0.0632261 ……
```

See also:
* [`T2mapSEcorr`](@ref)
"""
T2partSEcorr(T2distributions::Array{T,4}; kwargs...) where {T} = T2partSEcorr(T2distributions, T2partOptions(T2distributions; kwargs...))

function T2partSEcorr(T2distributions::Array{T,4}, opts::T2partOptions{T}) where {T}
    @assert size(T2distributions) == (opts.MatrixSize..., opts.nT2)

    # Print settings to terminal
    !opts.Silent && @info show_string(opts)

    # Initial output
    maps = init_output_t2parts(T2distributions, opts)

    # For each worker in the worker pool, allocate a separete thread-local buffer, then run the work function `work!`
    function with_thread_buffer(work!)
        thread_buffer = thread_buffer_maker(opts)
        work!(thread_buffer)
    end

    # Run T2-Part analysis
    LinearAlgebra.BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads

    indices = CartesianIndices(opts.MatrixSize)
    indices_blocks = split_indices(length(indices), default_blocksize())
    workerpool(with_thread_buffer, indices_blocks; ntasks = opts.Threaded ? Threads.nthreads() : 1) do inds, thread_buffer
        @inbounds for j in inds
            voxelwise_T2_parts!(thread_buffer, maps, T2distributions, opts, indices[j])
        end
    end

    LinearAlgebra.BLAS.set_num_threads(Threads.nthreads()) # Reset BLAS threads

    return maps
end

function init_output_t2parts(T2distributions::Array{T,4}, opts::T2partOptions{T}) where {T}
    maps = Dict{String,Any}()
    maps["sfr"] = fill(T(NaN), opts.MatrixSize...)
    maps["sgm"] = fill(T(NaN), opts.MatrixSize...)
    maps["mfr"] = fill(T(NaN), opts.MatrixSize...)
    maps["mgm"] = fill(T(NaN), opts.MatrixSize...)
    return maps
end

# =========================================================
# Save thread local results to output maps
# =========================================================
function voxelwise_T2_parts!(thread_buffer, maps, T2distributions, o::T2partOptions{T}, I) where {T}
    @unpack dist, T2_times, sp_range, mp_range, logT2_times_sp, logT2_times_mp, weights = thread_buffer

    # Return nothing if distribution contains NaN entries
    @inbounds for j in 1:o.nT2
        isnan(T2distributions[I, j]) && return nothing
    end

    # Load in voxel T2 distribution
    @inbounds @simd for j in 1:o.nT2
        dist[j] = T2distributions[I, j]
    end

    # Precompute sums and dot products over small pool, medium pool, and entire ranges
    Σ_dist, Σ_dist_sp, Σ_dist_mp = zero(T), zero(T), zero(T)
    dot_sp, dot_mp = zero(T), zero(T)
    @inbounds @simd for j in 1:length(dist)
        Σ_dist += dist[j]
    end
    @inbounds @simd for j in 1:length(sp_range)
        sp = sp_range[j]
        dot_sp    += dist[sp] * logT2_times_sp[j]
        Σ_dist_sp += dist[sp]
    end
    @inbounds @simd for j in 1:length(mp_range)
        mp = mp_range[j]
        dot_mp    += dist[mp] * logT2_times_mp[j]
        Σ_dist_mp += dist[mp]
    end

    # Compute T2 distribution parts
    if Σ_dist > 0
        @inbounds maps["sfr"][I] = o.Sigmoid !== nothing ? dot(dist, weights) / Σ_dist : Σ_dist_sp / Σ_dist
        @inbounds maps["mfr"][I] = Σ_dist_mp / Σ_dist
    end
    if Σ_dist_sp > 0
        @inbounds maps["sgm"][I] = exp(dot_sp / Σ_dist_sp)
    end
    if Σ_dist_mp > 0
        @inbounds maps["mgm"][I] = exp(dot_mp / Σ_dist_mp)
    end

    return nothing
end

# =========================================================
# Utility functions
# =========================================================
function thread_buffer_maker(o::T2partOptions{T}) where {T}
    dist           = zeros(T, o.nT2)
    T2_times       = logrange(o.T2Range..., o.nT2)
    sp_range       = findfirst(>=(o.SPWin[1]), T2_times) : findlast(<=(o.SPWin[2]), T2_times)
    mp_range       = findfirst(>=(o.MPWin[1]), T2_times) : findlast(<=(o.MPWin[2]), T2_times)
    logT2_times    = log.(T2_times)
    logT2_times_sp = logT2_times[sp_range]
    logT2_times_mp = logT2_times[mp_range]
    weights        = sigmoid_weights(o)
    return (; dist, T2_times, sp_range, mp_range, logT2_times_sp, logT2_times_mp, weights)
end

function sigmoid_weights(o::T2partOptions{T}) where {T}
    if o.Sigmoid !== nothing
        # Curve reaches 50% at T2_50perc and is (k and 1-k)*100 percent at T2_50perc +/- T2_kperc  
        k, T2_kperc, T2_50perc = T(0.1), o.Sigmoid, o.SPWin[2]
        sigma = abs(T2_kperc / (sqrt(T(2)) * erfinv(2*k-1)))
        (x -> x <= eps(T) ? zero(T) : x).(normccdf.((logrange(o.T2Range..., o.nT2) .- T2_50perc) ./ sigma))
    else
        nothing
    end
end
