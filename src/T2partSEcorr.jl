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
- `Sigmoid`:    apply sigmoidal weighting to the upper limit of the short peak window.
                `Sigmoid` is the delta-T2 parameter, which is the distance in seconds on either side of the `SPWin` upper limit where the sigmoid curve reaches 10% and 90% (Default: `nothing`, Units: seconds)
- `Silent`:     suppress printing to the console (Default: `false`)

See also:
* [`T2partSEcorr`](@ref)
"""
@with_kw struct T2partOptions{T} @deftype T
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
    @assert Sigmoid isa Nothing || Sigmoid > 0.0

    Silent::Bool = false
end
T2partOptions(args...; kwargs...) = T2partOptions{Float64}(args...; kwargs...)

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
Base.show(io::IO, ::MIME"text/plain", o::T2partOptions) = show(io, _show_string(o))

"""
    T2partSEcorr(T2distributions; <keyword arguments>)
    T2partSEcorr(T2distributions, opts::T2partOptions)

Analyzes T2 distributions produced by [`T2mapSEcorr`](@ref) to produce data maps of a series of parameters.

# Arguments
- `T2distributions`: 4D array with data as `(row, column, slice, T2 amplitude)`
- A series of optional keyword argument settings which will be used to construct a [`T2partOptions`](@ref) struct internally, or a [`T2partOptions`](@ref) struct directly

# Ouputs
- `maps`: a dictionary containing the following 3D data maps as fields:
    - `"sfr"`: small pool fraction, e.g. myelin water fraction
    - `"sgm"`: small pool geometric mean T2
    - `"mfr"`: medium pool fraction, e.g. intra/extracellular water fraction
    - `"mgm"`: medium pool geometric mean T2

# Examples
```julia-repl
julia> dist = DECAES.mock_T2_dist(MatrixSize = (100,100,1), nT2 = 40); # mock distribution with size 100x100x1x40

julia> maps = T2partSEcorr(dist; Silent = true); # compute T2-parts maps

julia> maps # MWF is contained in maps["sfr"]
Dict{String,Array{Float64,3}} with 4 entries:
  "sgm" => [0.0159777 0.0156194 … 0.0149169 0.0121455; 0.015296 0.0143854 … 0.018459 0.01627…
  "mfr" => [0.852735 0.814759 … 0.808621 0.859088; 0.830943 0.804878 … 0.836248 0.816681; … …
  "sfr" => [0.147265 0.185241 … 0.191379 0.140912; 0.169057 0.195122 … 0.163752 0.183319; … …
  "mgm" => [0.0600928 0.0581919 … 0.0612683 0.0563942; 0.0606434 0.0584615 … 0.0569397 0.054…
```

See also:
* [`T2mapSEcorr`](@ref)
"""
function T2partSEcorr(T2distributions::Array{T,4}; kwargs...) where {T}
    reset_timer!(TIMER)
    out = @timeit_debug TIMER "T2partSEcorr" begin
        T2partSEcorr(T2distributions, T2partOptions{T}(;
            MatrixSize = size(T2distributions)[1:3],
            nT2 = size(T2distributions, 4),
            kwargs...
        ))
    end
    if timeit_debug_enabled()
        println("\n"); show(TIMER); println("\n")
    end
    return out
end

function T2partSEcorr(T2distributions::Array{T,4}, opts::T2partOptions{T}) where {T}
    @assert size(T2distributions) == (opts.MatrixSize..., opts.nT2)

    # Print settings to terminal
    if !opts.Silent
        @info _show_string(opts)
    end
    LEGACY[] = opts.legacy

    # Initial output
    maps = Dict{String, Array{T,3}}()
    maps["sfr"] = fill(T(NaN), opts.MatrixSize...)
    maps["sgm"] = fill(T(NaN), opts.MatrixSize...)
    maps["mfr"] = fill(T(NaN), opts.MatrixSize...)
    maps["mgm"] = fill(T(NaN), opts.MatrixSize...)
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

# =========================================================
# Save thread local results to output maps
# =========================================================
function voxelwise_T2_parts!(thread_buffer, maps, T2distributions, o::T2partOptions{T}, I) where {T}
    @unpack dist, T2_times, sp_range, mp_range, logT2_times_sp, logT2_times_mp, weights = thread_buffer

    # Load in voxel T2 distribution, or return nothing for distributions containing NaN entries
    @inbounds for j in 1:o.nT2
        T2 = T2distributions[I, j]
        if isnan(T2)
            return nothing
        else
            dist[j] = T2
        end
    end

    # Precompute sums and dot products over small pool, medium pool, and entire ranges
    Σ_dist, Σ_dist_sp, Σ_dist_mp = zero(T), zero(T), zero(T)
    dot_sp, dot_mp = zero(T), zero(T)
    @inbounds @simd for j in 1:length(dist)
        Σ_dist += dist[j]
    end
    @inbounds for (j,sp) in enumerate(sp_range)
        dot_sp    += dist[sp] * logT2_times_sp[j]
        Σ_dist_sp += dist[sp]
    end
    @inbounds for (j,mp) in enumerate(mp_range)
        dot_mp    += dist[mp] * logT2_times_mp[j]
        Σ_dist_mp += dist[mp]
    end

    # Compute T2 distribution parts
    @inbounds begin
        maps["sfr"][I] = !(o.Sigmoid === nothing) ?
            dot(dist, weights) / Σ_dist : # Use sigmoidal weighting function
            Σ_dist_sp / Σ_dist
        maps["sgm"][I] = exp(dot_sp / Σ_dist_sp)
        maps["mfr"][I] = Σ_dist_mp / Σ_dist
        maps["mgm"][I] = exp(dot_mp / Σ_dist_mp)
    end

    return nothing
end

# =========================================================
# Utility functions
# =========================================================
function thread_buffer_maker(o::T2partOptions{T}) where {T}
    dist           = zeros(T, o.nT2)
    T2_times       = logrange(o.T2Range..., o.nT2)
    sp_range       = findfirst(t -> t >= o.SPWin[1], T2_times) : findlast(t -> t <= o.SPWin[2], T2_times)
    mp_range       = findfirst(t -> t >= o.MPWin[1], T2_times) : findlast(t -> t <= o.MPWin[2], T2_times)
    logT2_times    = log.(T2_times)
    logT2_times_sp = logT2_times[sp_range]
    logT2_times_mp = logT2_times[mp_range]
    weights        = sigmoid_weights(o)
    return @ntuple(dist, T2_times, sp_range, mp_range, logT2_times_sp, logT2_times_mp, weights)
end

function sigmoid_weights(o::T2partOptions{T}) where {T}
    if !(o.Sigmoid === nothing)
        # Curve reaches 50% at T2_50perc and is (k and 1-k)*100 percent at T2_50perc +/- T2_kperc  
        k, T2_kperc, T2_50perc = T(0.1), o.Sigmoid, o.SPWin[2]
        sigma = abs(T2_kperc / (sqrt(T(2)) * erfinv(2*k-1)))
        normccdf.((logrange(o.T2Range..., o.nT2) .- T2_50perc) ./ sigma)
    else
        nothing
    end
end
