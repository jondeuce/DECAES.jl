"""
    T2partSEcorr([io = stderr,] T2distributions::Array{T,4}; <keyword arguments>)
    T2partSEcorr([io = stderr,] T2distributions::Array{T,4}, opts::T2partOptions{T})

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

julia> maps # MWF is contained in maps["sfr"]
Dict{String, Any} with 4 entries:
  "sgm" => [0.0105638 0.0128673 … 0.0120685 0.0117591; 0.0108218 0.0139536 … 0.0118711 0.011…
  "mfr" => [0.879499 0.889248 … 0.886412 0.897167; 0.927041 0.887472 … 0.872513 0.89026; … ;…
  "sfr" => [0.120501 0.110752 … 0.113588 0.102833; 0.0729592 0.112528 … 0.127487 0.10974; … …
  "mgm" => [0.0621055 0.0617419 … 0.0612094 0.0615634; 0.0607907 0.0625588 … 0.0629157 0.062…
```

See also:
* [`T2mapSEcorr`](@ref)
"""
T2partSEcorr(T2distributions::Array{T,4}; kwargs...) where {T} = T2partSEcorr(stderr, T2distributions; kwargs...)
T2partSEcorr(io::IO, T2distributions::Array{T,4}; kwargs...) where {T} = T2partSEcorr(io, T2distributions, T2partOptions(T2distributions; kwargs...))
T2partSEcorr(T2distributions::Array{T,4}, opts::T2partOptions{T}) where {T} = T2partSEcorr(stderr, T2distributions, opts)
T2partSEcorr(io::IO, T2distributions::Array{T,4}, opts::T2partOptions{T}) where {T} = @timeit_debug TIMER() "T2partSEcorr" T2partSEcorr_(io, T2distributions, opts)

function T2partSEcorr_(io::IO, T2distributions::Array{T,4}, opts::T2partOptions{T}) where {T}
    @assert size(T2distributions) == (opts.MatrixSize..., opts.nT2)

    # Print settings to terminal
    !opts.Silent && printbody(io, _show_string(opts))
    LEGACY[] = opts.legacy

    # Initial output
    maps = init_output_t2parts(T2distributions, opts)
    thread_buffers = [thread_buffer_maker(opts) for _ in 1:Threads.nthreads()]

    # Run T2-Part analysis
    LinearAlgebra.BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads
    @inbounds Threads.@threads for I in CartesianIndices(opts.MatrixSize)
        thread_buffer = thread_buffers[Threads.threadid()]
        voxelwise_T2_parts!(thread_buffer, maps, T2distributions, opts, I)
    end
    LinearAlgebra.BLAS.set_num_threads(Threads.nthreads()) # Reset BLAS threads
    LEGACY[] = false

    return maps
end

function init_output_t2parts(T2distributions::Array{T,4}, opts::T2partOptions{T}) where {T}
    maps = Dict{String, Any}()
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
