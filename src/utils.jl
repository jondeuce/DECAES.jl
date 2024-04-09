####
#### Miscellaneous utils
####

ndigits(x::Int) = x == 0 ? 1 : floor(Int, log10(abs(x))) + 1

logrange(a::Real, b::Real, len::Int) = (r = exp.(range(log(a), log(b); length = len)); r[begin] = a; r[end] = b; return r)

@inline normcdf(x::T) where {T} = erfc(-x / √(T(2))) / 2 # Cumulative distribution for normal distribution
@inline normccdf(x::T) where {T} = erfc(x / √(T(2))) / 2 # Compliment of normcdf, i.e. 1 - normcdf(x)

@inline strictsign(x::Real) = ifelse(signbit(x), -one(x), one(x))

@inline lt_nan(x, y) = ifelse(isnan(x), typemax(typeof(x)), x) < ifelse(isnan(y), typemax(typeof(y)), y) # <(x, y), treating NaN as Inf

@inline basisvector(::Type{SVector{D, T}}, i::Int) where {D, T} = SVector{D, T}(ntuple(d -> T(d == i), D))

function meshgrid(::Type{T}, iters...) where {T}
    A = [T(xs) for xs in Iterators.product(iters...)]
    return reshape(A, length.(iters)...)
end
meshgrid(iters...) = meshgrid(Tuple, iters...)

@inline function SplitCartesianIndices(sz::NTuple{N, Int}, ::Val{M}) where {N, M}
    @assert 0 <= M <= N
    sz1 = sz[1:M]
    sz2 = sz[M+1:N]
    return CartesianIndices(sz1), CartesianIndices(sz2)
end
@inline SplitCartesianIndices(x::AbstractArray{<:Any, N}, ::Val{M}) where {N, M} = SplitCartesianIndices(size(x), Val(M))

@generated function fieldsof(::Type{T}, ::Type{C} = Tuple) where {T, C}
    fields = fieldnames(T) # fieldnames(T) allocates; hoist to generated function
    return C <: Tuple ?
           :($fields) : # default to returning tuple of field symbols
           :($(C(Symbol[fields...]))) # call container constructor on vector of symbols
end

@inline floattype(xs::Tuple) = float(promote_type(map(typeof, xs)...))
@inline floattype(xs::NamedTuple) = floattype(Tuple(xs))

function insertsorted!(x, val, len = length(x))
    len = min(len, length(x))
    @inbounds for i in 1:len-1
        x[i], val = minmax(x[i], val)
    end
    @inbounds x[len] = val
    return x
end

####
#### Linear algebra utils
####

function with_singlethreaded_blas(f)
    nblasthreads = LinearAlgebra.BLAS.get_num_threads()
    try
        LinearAlgebra.BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads
        f()
    finally
        LinearAlgebra.BLAS.set_num_threads(nblasthreads) # Reset BLAS threads
    end
end

#### Fast SVD values by preallocating workspace and calling LAPACK directly

const BlasRealFloat = Union{Float32, Float64}

struct SVDValsWorkspace{elty <: BlasRealFloat}
    job::Char
    m::Int
    n::Int
    A::Matrix{elty}
    S::Vector{elty}
    U::Matrix{elty}
    VT::Matrix{elty}
    work::Vector{elty}
    lwork::Base.RefValue{BlasInt}
    iwork::Vector{BlasInt}
    info::Base.RefValue{BlasInt}
end
function SVDValsWorkspace(A::AbstractMatrix{elty}) where {elty <: BlasRealFloat}
    job   = 'N'
    m, n  = size(A)
    minmn = min(m, n)
    S     = similar(A, elty, minmn)
    U     = similar(A, elty, (m, 0))
    VT    = similar(A, elty, (n, 0))
    work  = Vector{elty}(undef, 1)
    lwork = Ref{BlasInt}(-1)
    iwork = Vector{BlasInt}(undef, 8 * minmn)
    info  = Ref{BlasInt}()
    return SVDValsWorkspace(job, m, n, copy(A), S, U, VT, work, lwork, iwork, info)
end

function LinearAlgebra.svdvals!(work::SVDValsWorkspace, A::AbstractMatrix)
    copyto!(work.A, A)
    return LinearAlgebra.svdvals!(work)
end

# See: https://github.com/JuliaLang/julia/blob/64de065a183ac70bb049f7f9e30d790f8845dd2b/stdlib/LinearAlgebra/src/lapack.jl#L1590
# (GE) general matrices eigenvalue-eigenvector and singular value decompositions
for (gesdd, elty) in ((:dgesdd_, :Float64), (:sgesdd_, :Float32))
    #    SUBROUTINE DGESDD( JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, IWORK, INFO )
    #*     .. Scalar Arguments ..
    #      CHARACTER          JOBZ
    #      INTEGER            INFO, LDA, LDU, LDVT, LWORK, M, N
    #*     .. Array Arguments ..
    #      INTEGER            IWORK( * )
    #      DOUBLE PRECISION   A( LDA, * ), S( * ), U( LDU, * ), VT( LDVT, * ), WORK( * )
    @eval function LinearAlgebra.svdvals!(work::SVDValsWorkspace{$elty})
        (; job, m, n, A, S, U, VT, work, lwork, iwork, info) = work
        Base.require_one_based_indexing(A)
        # lwork[] = BlasInt(-1) # uncomment this line to query lwork every call
        for i in 1:2
            i == 1 && lwork[] != BlasInt(-1) && continue # uncomment this line to skip lwork query after first call
            ccall((@blasfunc($gesdd), libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                    Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{BlasInt}, Ptr{BlasInt}, Clong),
                job, m, n, A, max(1, stride(A, 2)), S, U, max(1, stride(U, 2)), VT, max(1, stride(VT, 2)),
                work, lwork[], iwork, info, 1)
            chklapackerror(info[])
            if i == 1
                # Work around issue with truncated Float32 representation of lwork in `sgesdd` by using `nextfloat`.
                # See: https://github.com/scipy/scipy/issues/5401
                lwork[] = round(BlasInt, nextfloat(work[1])) # lwork returned in work[1]
                resize!(work, lwork[])
            end
        end
        return S
    end
end

####
#### Dynamically sized caches
####

struct GrowableCache{K, V, C}
    keys::Vector{K}
    values::Vector{V}
    length::Base.RefValue{Int}
    cmp::C
    function GrowableCache{K, V}(bufsize::Int = 0, cmp = ==) where {K, V}
        keys = Vector{K}(undef, bufsize)
        values = Vector{V}(undef, bufsize)
        return new{K, V, typeof(cmp)}(keys, values, Ref(0), cmp)
    end
end
@inline Base.keys(c::GrowableCache) = view(c.keys, 1:c.length[])
@inline Base.values(c::GrowableCache) = view(c.values, 1:c.length[])
@inline Base.length(c::GrowableCache) = c.length[]
@inline Base.empty!(c::GrowableCache) = (c.length[] = 0; c)
@inline Base.isempty(c::GrowableCache) = c.length[] == 0
@inline Base.pairs(c::GrowableCache) = GrowableCachePairs(c)
@inline Base.iterate(c::GrowableCache, i = 0) = length(c) <= i ? nothing : @inbounds(((c.keys[i+1], c.values[i+1]), i + 1))
@inline Base.getindex(c::GrowableCache, x) = @inbounds c.values[findfirst(c, x)]

@inline function Base.setindex!(c::GrowableCache, v, x)
    ind = findfirst(c, x)
    if ind == 0
        push!(c, (x, v))
    else
        @inbounds c.values[ind] = v
    end
    return v
end

function Base.findfirst(c::GrowableCache, x)
    ind = 0
    @inbounds for i in 1:c.length[]
        if c.cmp(x, c.keys[i])
            ind = i
            break
        end
    end
    return ind
end

function Base.push!(c::GrowableCache, (x, v))
    ind = c.length[] += 1
    @inbounds if ind <= length(c.keys)
        c.keys[ind] = x
        c.values[ind] = v
    else
        push!(c.keys, x)
        push!(c.values, v)
    end
    return c
end

function Base.get!(f, c::GrowableCache, x)
    ind = findfirst(c, x)
    @inbounds if ind > 0
        v = c.values[ind]
    else
        v = f(x)
        push!(c, (x, v))
    end
    return v
end

function Base.pushfirst!(c::GrowableCache, (x, v))
    isempty(c) && return push!(c, (x, v))
    @inbounds for i in 1:c.length[]
        c.keys[i], x = x, c.keys[i]
        c.values[i], v = v, c.values[i]
    end
    return push!(c, (x, v))
end

struct GrowableCachePairs{K, V, C} <: AbstractVector{Tuple{K, V}}
    cache::GrowableCache{K, V, C}
end
@inline Base.IndexStyle(::GrowableCachePairs) = IndexLinear()
@inline Base.size(c::GrowableCachePairs) = (length(c.cache),)
@inline Base.length(c::GrowableCachePairs) = length(c.cache)
@inline Base.empty!(c::GrowableCachePairs) = (empty!(c.cache); c)
@inline Base.isempty(c::GrowableCachePairs) = isempty(c.cache)
@inline Base.push!(c::GrowableCachePairs, xv) = push!(c.cache, xv)
@inline Base.pushfirst!(c::GrowableCachePairs, xv) = pushfirst!(c.cache, xv)
Base.@propagate_inbounds Base.getindex(c::GrowableCachePairs, i::Int) = (c.cache.keys[i], c.cache.values[i])
Base.@propagate_inbounds Base.setindex!(c::GrowableCachePairs, (x, v), i::Int) = (c.cache.keys[i] = x; c.cache.values[i] = v; (x, v))

struct CachedFunction{K, V, C <: GrowableCache{K, V}, F}
    f::F
    cache::C
end
CachedFunction{K, V}(f, args...) where {K, V} = CachedFunction(f, GrowableCache{K, V}(args...))
@inline (f::CachedFunction)(x) = get!(f.f, f.cache, x)
@inline Base.empty!(f::CachedFunction) = (empty!(f.cache); f)

struct MappedArray{K, V, N, A <: AbstractArray{K, N}, F} <: AbstractArray{V, N}
    f::F
    x::A
    MappedArray(f, x::AbstractArray) = MappedArray{eltype(x)}(f, x)
    MappedArray{V}(f::F, x::A) where {K, V, N, A <: AbstractArray{K, N}, F} = new{K, V, N, A, F}(f, x)
end
@inline Base.IndexStyle(::MappedArray) = IndexLinear()
@inline Base.size(m::MappedArray) = size(m.x)

Base.@propagate_inbounds Base.getindex(m::MappedArray, i::Int) = m.f(m.x[i])
Base.setindex!(::MappedArray, v, i...) = error("MappedArray's are read only")

function mapfind(finder, m::MappedArray)
    y, i = finder(m)
    x = @inbounds m.x[i]
    return x, y, i
end
@inline mapfindmax(m::MappedArray) = mapfind(findmax, m)
@inline mapfindmin(m::MappedArray) = mapfind(findmin, m)
@inline mapfindmax(::Type{V}, f, xs::AbstractArray) where {V} = mapfindmax(MappedArray{V}(f, xs))
@inline mapfindmin(::Type{V}, f, xs::AbstractArray) where {V} = mapfindmin(MappedArray{V}(f, xs))

####
#### Timing utils
####

tic() = time()
toc(t) = tic() - t

function hour_min_sec(t)
    hour = floor(Int, t / 3600)
    min = floor(Int, (t - 3600 * hour) / 60)
    sec = floor(Int, t - 3600 * hour - 60 * min)
    return (; hour, min, sec)
end

function pretty_time(t)
    if isnan(t) || isinf(t)
        "--h:--m:--s"
    else
        hh, mm, ss = hour_min_sec(t)
        lpad(hh, 2, "0") * "h:" * lpad(mm, 2, "0") * "m:" * lpad(ss, 2, "0") * "s"
    end
end

@with_kw struct DECAESProgress
    progress_meter::Progress
    io_buffer::IOBuffer
    io_lock::ReentrantLock = Threads.ReentrantLock()
    last_msg::Base.RefValue{String} = Ref("")
end

function DECAESProgress(n::Int, desc::AbstractString = ""; kwargs...)
    io_buffer = IOBuffer()
    return DECAESProgress(;
        progress_meter = Progress(n; dt = 0.0, desc = desc, color = :cyan, output = io_buffer, barglyphs = BarGlyphs("[=> ]"), kwargs...),
        io_buffer = io_buffer,
    )
end

ProgressMeter.next!(p::DECAESProgress) = (ProgressMeter.next!(p.progress_meter); maybe_print!(p))
ProgressMeter.finish!(p::DECAESProgress) = (ProgressMeter.finish!(p.progress_meter); maybe_print!(p))
ProgressMeter.update!(p::DECAESProgress, counter) = (ProgressMeter.update!(p.progress_meter, counter); maybe_print!(p))

function maybe_print!(p::DECAESProgress)
    # Internal `progress_meter` prints to the IOBuffer `p.io_buffer`; check this buffer for new messages
    msg = String(take!(p.io_buffer)) # note: take!(::IOBuffer) is threadsafe
    if !isempty(msg)
        # Format message
        msg = replace(msg, "\r" => "", "\u1b[K" => "", "\u1b[A" => "")

        # Update last message
        last_msg = lock(p.io_lock) do
            last_msg, p.last_msg[] = p.last_msg[], msg
            return last_msg
        end

        # Print progress message and flush
        @info msg
        flush(stderr)
    end
    return nothing
end

# Macro for timing arbitrary code snippet and printing time
macro showtime(msg, ex)
    quote
        @info $(esc(msg))
        local t = time()
        local val = $(esc(ex))
        local t = time() - t
        @info "Done ($(round(t; digits = 2)) seconds)"
        val
    end
end

####
#### Threading utils
####

# Threaded `foreach` construct, borrowing implementation from ThreadTools.jl:
#
#   https://github.com/baggepinnen/ThreadTools.jl/blob/55aaf2bbe735e52cefaad143e7614d4f00e312b0/src/ThreadTools.jl#L57
#
# Updated according to suggestions from the folks at DataFrames.jl:
#
#   https://github.com/jondeuce/DECAES.jl/issues/37
function tforeach(work!, allocate, x::AbstractArray; blocksize::Int = default_blocksize())
    nt = Threads.nthreads()
    len = length(x)
    if nt > 1 && len > blocksize
        @sync for p in split_indices(len, blocksize)
            Threads.@spawn allocate() do resource
                @simd for i in p
                    work!(x[i], resource)
                end
            end
        end
    else
        allocate() do resource
            @simd for i in eachindex(x)
                work!(x[i], resource)
            end
        end
    end
    return nothing
end
tforeach(f, x::AbstractArray; kwargs...) = tforeach((x, r) -> f(x), g -> g(nothing), x; kwargs...)

default_blocksize() = 64

# Worker pool for allocating thread-local resources. This is a more robust alternative to
# the (now anti-)pattern of allocating vectors of thread-local storage buffers and indexing them
# by `threadid()`; this is no longer guaranteed to work in v1.7+, as tasks are now allowed to
# migrate across threads. Instead, here we allocate a local `resource` via the `allocate` function
# argument at `@spawn`-time, obviating the need to tie the resource to the `threadid()`.
#
#   See: https://juliafolds.github.io/data-parallelism/tutorials/concurrency-patterns/#worker_pool

function workerpool(work!, allocate, inputs::Channel; ninputs::Int, ntasks::Int = Threads.nthreads(), verbose::Bool = false)
    function consumer(callback = () -> nothing)
        allocate() do resource
            for input in inputs
                work!(input, resource)
                callback()
            end
        end
    end

    if !verbose
        if ntasks == 1
            consumer()
        else
            @sync for _ in 1:ntasks
                Threads.@spawn consumer()
            end
        end
    else
        @sync begin
            counter = Threads.Atomic{Int}(0)
            for _ in 1:ntasks-1
                Threads.@spawn consumer() do
                    return counter[] += 1
                end
            end

            dt = 5.0
            last_time = Ref(time())
            progmeter = DECAESProgress(ninputs)
            consumer() do
                counter[] += 1
                if (new_time = time()) > last_time[] + dt
                    ProgressMeter.update!(progmeter, counter[])
                    last_time[] = new_time
                end
            end
            ProgressMeter.finish!(progmeter)
        end
    end
end

function workerpool(work!, allocate, inputs, args...; kwargs...)
    ch = Channel{eltype(inputs)}(length(inputs))
    for inds in inputs
        put!(ch, inds)
    end
    close(ch)
    return workerpool(work!, allocate, ch, args...; ninputs = length(inputs), kwargs...)
end

function split_indices(len::Int, basesize::Int)
    len′ = Int64(len) # Avoid overflow on 32-bit machines
    np = max(1, div(len′, basesize))
    return [Int(1 + ((i - 1) * len′) ÷ np):Int((i * len′) ÷ np) for i in 1:np]
end

####
#### Logging
####

# https://github.com/JuliaLogging/LoggingExtras.jl/issues/15
function TimestampLogger(logger, date_format = "yyyy-mm-dd HH:MM:SS")
    return TransformerLogger(logger) do log
        return merge(log, (; message = "$(Dates.format(Dates.now(), date_format)) $(log.message)"))
    end
end

function tee_capture(f; logfile = tempname(), suppress_terminal = false, suppress_logfile = false)
    logger =
        suppress_terminal && suppress_logfile ? ConsoleLogger(devnull) :
        suppress_logfile ? ConsoleLogger(stderr) :
        suppress_terminal ? TimestampLogger(FileLogger(logfile)) :
        TeeLogger(ConsoleLogger(stderr), TimestampLogger(FileLogger(logfile)))
    logger = LevelOverrideLogger(LoggingExtras.Info, logger) # suppress debug messages
    with_logger(logger) do
        return f()
    end
end

# https://discourse.julialang.org/t/redirect-stdout-and-stderr/13424/3
function redirect_to_files(f, outfile, errfile)
    open(outfile, "w") do out
        open(errfile, "w") do err
            redirect_stdout(out) do
                redirect_stderr(err) do
                    return f()
                end
            end
        end
    end
end
redirect_to_tempfiles(f) = redirect_to_files(f, tempname() * ".log", tempname() * ".err")

function redirect_to_devnull(f)
    with_logger(ConsoleLogger(devnull)) do
        redirect_to_tempfiles() do
            return f()
        end
    end
end

####
#### Optimizers
####

struct ADAM{N, T}
    η::T
    β::SVector{2, T}
    mt::SVector{N, T}
    vt::SVector{N, T}
    βp::SVector{2, T}
end
function ADAM{N, T}(η = 0.001, β = (0.9, 0.999)) where {N, T}
    @assert N >= 1
    S2 = SVector{2, T}
    SN = SVector{N, T}
    return ADAM{N, T}(T(η), S2(β), zero(SN), zero(SN), ones(S2))
end

function update(∇::SVector{N, T}, o::ADAM{N, T}) where {N, T}
    (; η, β, mt, vt, βp) = o

    ε  = T(1e-8)
    βp = @. βp * β
    ηt = η * √(1 - βp[2]) / (1 - βp[1])
    mt = @. β[1] * mt + (1 - β[1]) * ∇
    vt = @. β[2] * vt + (1 - β[2]) * ∇^2
    Δ  = @. ηt * mt / (√vt + ε)

    return Δ, ADAM{N, T}(η, β, mt, vt, βp)
end

#=
function optimize(∇f, x0::SVector{N, T}, lb::SVector{N, T}, ub::SVector{N, T}, o::ADAM{N, T}; maxiters::Int = 1, xtol_rel = T(1e-3)) where {N, T}
    x = x0
    t = inv_xform_periodic(x, lb, ub)
    for i in 1:maxiters
        # Change of variables x->t
        x = xform_periodic(t, lb, ub)
        dxdt = ∇xform_periodic(t, lb, ub)
        dfdx = ∇f(x)
        dfdt = dfdx .* dxdt

        # Update in t-space
        Δt, o = update(dfdt, o)
        t -= Δt

        # Check for convegence in x-space
        xold, x = x, xform_periodic(t, lb, ub)
        maximum(abs.(x - xold)) < max(maximum(abs.(x)), maximum(abs.(xold))) * xtol_rel && break
    end
    return x, o
end

@inline xform_periodic(t::S, lb::S, ub::S) where {N, T, S <: SVector{N, T}} = S(ntuple(i -> clamp(((lb[i] + ub[i]) / 2) + ((ub[i] - lb[i]) / 2) * sinpi(t[i]), lb[i], ub[i]), N))
@inline ∇xform_periodic(t::S, lb::S, ub::S) where {N, T, S <: SVector{N, T}} = S(ntuple(i -> ((ub[i] - lb[i]) / 2) * T(π) * cospi(t[i]), N))
@inline inv_xform_periodic(x::S, lb::S, ub::S) where {N, T, S <: SVector{N, T}} = S(ntuple(i -> asin(clamp((x[i] - ((lb[i] + ub[i]) / 2)) / ((ub[i] - lb[i]) / 2), -one(T), one(T))) / T(π), N))
=#

####
#### Generate (moderately) realistic mock images
####

function mock_t2map_opts(::Type{T} = Float64; kwargs...) where {T}
    t2map_fields = fieldsof(T2mapOptions, Set)
    t2map_kwargs = filter((k, v)::Pair -> k ∈ t2map_fields, kwargs)
    return T2mapOptions{T}(;
        MatrixSize = (2, 2, 2),
        TE = 10e-3,
        nTE = 48,
        T2Range = (10e-3, 2.0),
        nT2 = 40,
        Reg = "lcurve",
        Chi2Factor = 1.02,
        NoiseLevel = 1e-3, # SNR = 60
        t2map_kwargs...,
    )
end

function mock_t2parts_opts(::Type{T} = Float64; kwargs...) where {T}
    t2part_fields = fieldsof(T2partOptions, Set)
    t2part_kwargs = filter((k, v)::Pair -> k ∈ t2part_fields, kwargs)
    return T2partOptions{T}(;
        MatrixSize = (2, 2, 2),
        nT2 = 40,
        T2Range = (10e-3, 2.0),
        SPWin = (10e-3, 40e-3),
        MPWin = (40e-3, 2.0),
        t2part_kwargs...,
    )
end

Base.@kwdef struct MockImageOpts{T}
    MatrixSize::NTuple{3, Int} = (2, 2, 2)
    TE::T = T(10e-3)
    nTE::Int = 48
    SNR::T = T(60.0)
    SFRRange::Tuple{T, T} = (0.05, 0.25)
    T21Range::Tuple{T, T} = (10e-3, 20e-3)
    T22Range::Tuple{T, T} = (50e-3, 100e-3)
    T1::T = T(1.0)
    FlipAngle::T = T(165.0)
end
function mock_image_opts(::Type{T} = Float64; kwargs...) where {T}
    image_fields = fieldsof(MockImageOpts, Set)
    image_kwargs = filter((k, v)::Pair -> k ∈ image_fields, kwargs)
    return MockImageOpts{T}(; image_kwargs...)
end

function mock_image(o::MockImageOpts{T}) where {T}
    (; MatrixSize, TE, nTE, SNR, SFRRange, T21Range, T22Range, T1, FlipAngle) = o
    σ = exp10(-T(SNR) / 20)
    M = zeros(T, (MatrixSize..., nTE))

    function allocate(loop_body!)
        m = zeros(T, nTE)
        work1 = EPGWork_ReIm_DualVector_Split_Dynamic(T, nTE)
        work2 = EPGWork_ReIm_DualVector_Split_Dynamic(T, nTE)
        return loop_body!((; m, work1, work2))
    end

    indices = CartesianIndices(MatrixSize)
    blocksize = max(div(length(indices), Threads.nthreads(), RoundUp), 1)
    tforeach(allocate, indices; blocksize) do I, buffers
        (; m, work1, work2) = buffers
        sfr = SFRRange[1] + (SFRRange[2] - SFRRange[1]) * rand(T) # short T2 fraction
        mfr = 1 - sfr # long T2 fraction
        T21 = T21Range[1] + (T21Range[2] - T21Range[1]) * rand(T) # short T2
        T22 = T22Range[1] + (T22Range[2] - T22Range[1]) * rand(T) # long T2
        dc1 = EPGdecaycurve!(work1, EPGOptions((; ETL = nTE, α = FlipAngle, TE, T2 = T21, T1, β = T(180.0))))
        dc2 = EPGdecaycurve!(work2, EPGOptions((; ETL = nTE, α = FlipAngle, TE, T2 = T22, T1, β = T(180.0))))
        @inbounds begin
            # Note: no need to normalize `m`, since convex combinations of EPG decay curves have maximum value 1,
            #       and therefore the relative noise level `σ` equals the absolute noise level.
            m .= sfr .* dc1 .+ mfr .* dc2 # bi-exponential signal with EPG correction
            @simd for k in 1:nTE
                zR, zI = σ * randn(T), σ * randn(T)
                m[k] = √((m[k] + zR)^2 + zI^2)
            end
            M[I, :] .= m
        end
    end

    return M
end

# Mock CPMG image
function mock_image(::Type{T} = Float64; kwargs...) where {T}
    o = mock_image_opts(T; kwargs...)
    return mock_image(o)
end
function mock_image(o::T2mapOptions{T}; kwargs...) where {T}
    t2map_kwargs = Dict{Symbol, Any}(o)
    return mock_image(; t2map_kwargs..., kwargs...)
end

# Mock T2 distribution, computed with default parameters
function mock_t2dist(o::T2mapOptions = mock_t2map_opts(Float64); kwargs...)
    return T2mapSEcorr(mock_image(o; kwargs...), T2mapOptions(o; kwargs..., Silent = true))[2]
end

# Full mock T2 pipeline
function mock_T2_pipeline(; kwargs...)
    t2map_opts = mock_t2map_opts(; kwargs...)
    t2part_opts = T2partOptions(t2map_opts; SPWin = (t2map_opts.T2Range[1], 40e-3), MPWin = (40e-3, t2map_opts.T2Range[2]))

    image = mock_image(; kwargs...)
    t2maps, t2dist = T2mapSEcorr(image, t2map_opts)
    t2part = T2partSEcorr(t2dist, t2part_opts)

    return image, t2maps, t2dist, t2part
end

# Mock I/O for precompilation
function mock_load_image()
    mktempdir() do dir
        for ndims in 3:4
            img = rand((1:ndims)...)

            file = tempname(dir) * ".mat"
            MAT.matwrite(file, Dict("img" => img))
            @assert img == load_image(file; ndims)

            for ext in [".nii", ".nii.gz"]
                file = tempname(dir) * ext
                NIfTI.niwrite(file, NIfTI.NIVolume(img))
                @assert img == load_image(file; ndims)
            end

            #TODO: read/write dummy PAR/REC and/or XML/REC
        end
    end
end
