####
#### Miscellaneous utils
####

ndigits(x::Int) = x == 0 ? 1 : floor(Int, log10(abs(x))) + 1
logrange(a::Real, b::Real, len::Int) = (r = exp.(range(log(a), log(b); length = len)); r[1] = a; r[end] = b; return r)
normcdf(x::T) where {T} = erfc(-x/sqrt(T(2)))/2 # Cumulative distribution for normal distribution
normccdf(x::T) where {T} = erfc(x/sqrt(T(2)))/2 # Compliment of normcdf, i.e. 1 - normcdf(x)

@inline mul_im(z::Complex) = Complex(-imag(z), real(z)) # optimized i*(a+b*i) = -b+a*i

@inline basisvector(::Type{SVector{D,T}}, i::Int) where {D,T} = SVector{D,T}(ntuple(d -> T(d == i), D))

function meshgrid(::Type{T}, iters...) where {T}
    A = [T(xs) for xs in Iterators.product(iters...)]
    reshape(A, length.(iters)...)
end
meshgrid(iters...) = meshgrid(Tuple, iters...)

@inline function SplitCartesianIndices(sz::NTuple{N,Int}, ::Val{M}) where {N,M}
    @assert 0 <= M <= N
    sz1 = sz[1:M]
    sz2 = sz[M+1:N]
    return CartesianIndices(sz1), CartesianIndices(sz2)
end
@inline SplitCartesianIndices(x::AbstractArray{<:Any,N}, ::Val{M}) where {N,M} = SplitCartesianIndices(size(x), Val(M))

@generated function fieldsof(::Type{T}, ::Type{C} = Tuple) where {T,C}
    fields = fieldnames(T) # fieldnames(T) allocates; hoist to generated function
    return C <: Tuple ?
        :($fields) : # default to returning tuple of field symbols
        :($(C(Symbol[fields...]))) # call container constructor on vector of symbols
end

@inline floattype(xs::Tuple) = float(promote_type(map(typeof, xs)...))
@inline floattype(xs::NamedTuple) = floattype(Tuple(xs))

function set_diag!(A::AbstractMatrix, val)
    @inbounds @simd ivdep for i in 1:min(size(A)...)
        A[i,i] = val
    end
    return A
end

function set_top!(A::AbstractArray, B::AbstractArray)
    @inbounds @simd ivdep for I in CartesianIndices(B)
        A[I] = B[I]
    end
    return A
end

function find_nearest(r::AbstractRange, x::Number)
    idx = x <= r[1] ? 1 :
          x >= r[end] ? length(r) :
          clamp(round(Int, 1 + (x - r[1]) / step(r)), 1, length(r))
    r[idx], idx # nearest value in r to x and corresponding index
end

function local_gridsearch(f, xs, i0)
    get(j) = @inbounds xs[clamp(j, firstindex(xs), lastindex(xs))]
    i = i0
    x⁻, x, x⁺ = get(i-1), get(i), get(i+1)
    y⁻, y, y⁺ = f(x⁻), f(x), f(x⁺)
    while !(y⁻ ≥ y ≤ y⁺) # search for local min
        if y⁻ < y
            i -= 1 # shift left
            x⁻, x, x⁺ = get(i-1), x⁻, x
            y⁻, y, y⁺ = f(x⁻), y⁻, y
        elseif y⁺ < y
            i += 1 # shift right
            x⁻, x, x⁺ = x, x⁺, get(i+1)
            y⁻, y, y⁺ = y, y⁺, f(x⁺)
        else
            break
        end
    end
    return (; x, y, i)
end

function mapfind(f, finder, xs)
    ys = map(f, xs)
    y, i = finder(ys)
    xs[i], y, i
end
mapfindmax(f, xs) = mapfind(f, findmax, xs)
mapfindmin(f, xs) = mapfind(f, findmin, xs)

####
#### Timing utilities
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
    progmeter::Progress
    n::Int
    io::IO
    iobuf::IOBuffer
    iolock::ReentrantLock = Threads.ReentrantLock()
    last_msg::Ref{String} = Ref("")
    ch::Channel{String} = Channel{String}(; spawn = true) do ch
        for msg in ch
            println(io, msg)
            flush(io)
        end
    end
end

function DECAESProgress(io::IO, n::Int, desc::AbstractString; kwargs...)
    iobuf = IOBuffer()
    DECAESProgress(
        progmeter = Progress(n; dt = 0.0, desc = desc, color = :cyan, output = iobuf, barglyphs = BarGlyphs("[=> ]"), kwargs...),
        n = n,
        io = io,
        iobuf = iobuf,
    )
end
DECAESProgress(n::Int, desc::AbstractString; kwargs...) = DECAESProgress(stderr, n, desc; kwargs...)

Base.length(p::DECAESProgress) = p.n
ProgressMeter.next!(p::DECAESProgress) = (ProgressMeter.next!(p.progmeter); maybe_print!(p))
ProgressMeter.finish!(p::DECAESProgress) = (ProgressMeter.finish!(p.progmeter); maybe_print!(p))
ProgressMeter.update!(p::DECAESProgress, counter) = (ProgressMeter.update!(p.progmeter, counter); maybe_print!(p))

function maybe_print!(p::DECAESProgress)
    # Inner progress meter prints to the internal IOBuffer `p.iobuf`.
    # Here, we only print to `p.io` if there is a new message.
    msg = String(take!(p.iobuf)) # Note: take!(::IOBuffer) is threadsafe
    if !isempty(msg)
        msg = replace(msg, "\r" => "")
        msg = replace(msg, "\u1b[K" => "")
        msg = replace(msg, "\u1b[A" => "")
        if msg != p.last_msg[]
            lock(p.iolock) do
                p.last_msg[] = msg
            end
            put!(p.ch, msg)
        end
    end
end

printheader(io, s) = (println(io, ""); printstyled(io, "* " * s * "\n"; color = :cyan))
printbody(io, s) = println(io, s)

# Macro for timing arbitrary code snippet and printing time
macro showtime(io, msg, ex)
    quote
        local io = $(esc(io))
        printheader(io, $(esc(msg)) * " ...")
        local val
        local t = @elapsed val = $(esc(ex))
        printheader(io, "Done ($(round(t; digits = 2)) seconds)")
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
function tforeach(f, x::AbstractArray; blocksize::Integer = default_blocksize())
    nt = Threads.nthreads()
    len = length(x)
    if nt > 1 && len > blocksize
        @sync for p in split_indices(len, blocksize)
            Threads.@spawn begin
                @simd ivdep for i in p
                    f(@inbounds x[i])
                end
            end
        end
    else
        @simd ivdep for i in eachindex(x)
            f(@inbounds x[i])
        end
    end
    return nothing
end

function split_indices(len::Integer, basesize::Integer)
    len′ = Int64(len) # Avoid overflow on 32-bit machines
    np = max(1, div(len′, basesize))
    return (Int(1 + ((i - 1) * len′) ÷ np) : Int((i * len′) ÷ np) for i in 1:np)
end
split_indices(r, basesize::Integer) = isempty(r) ? (r for _ in 1:1) : (r[i[begin] + begin - 1 : i[end] + begin - 1] for i in split_indices(length(r), basesize))

default_blocksize() = 64

# Worker pool for allocating thread-local resources. This is a more robust alternative to
# the (now anti-)pattern of allocating vectors of thread-local storage buffers and indexing them
# by `threadid()`; this is no longer guaranteed to work in v1.7+, as tasks are now allowed to
# migrate across threads. Instead, here we allocate a local `resource` via the `allocate` function
# argument at `@spawn`-time, obviating the need to tie the resource to the `threadid()`.
# 
#   See: https://juliafolds.github.io/data-parallelism/tutorials/concurrency-patterns/#worker_pool
function workerpool(work!, allocate, inputs::Channel, progmeter::Union{DECAESProgress, Progress, Nothing} = nothing; ntasks = Threads.nthreads())
    @sync for _ in 1:ntasks
        Threads.@spawn allocate() do resource
            for input in inputs
                if progmeter === nothing
                    work!(input, resource)
                else
                    @sync begin
                        Threads.@spawn work!(input, resource)
                        ProgressMeter.next!(progmeter)
                    end
                end
            end
        end
    end
end

function workerpool(work!, allocate, inputs, args...; kwargs...)
    ch = Channel{eltype(inputs)}(length(inputs))
    for inds in inputs
        put!(ch, inds)
    end
    close(ch)
    workerpool(work!, allocate, ch, args...; kwargs...)
end

####
#### Logging
####

# https://discourse.julialang.org/t/write-to-file-and-stdout/35042/3
struct Tee{T <: Tuple} <: IO
    streams::T
end
Tee(streams::IO...) = Tee(streams)
Base.flush(t::Tee) = tee(t)
Base.write(t::Tee, x::Array) = tee(Base.write, t, x) # needed to resolve method ambiguity
Base.write(t::Tee, args...) = tee(Base.write, t, args...)
Base.print(t::Tee, args...) = tee(Base.print, t, args...)
Base.println(t::Tee, args...) = tee(Base.println, t, args...)
Base.printstyled(t::Tee, args...; kwargs...) = tee(Base.printstyled, t, args...; kwargs...)

function tee(f, t::Tee, args...; kwargs...)
    for io in t.streams
        f(io, args...; kwargs...)
        flush(io)
    end
end
tee(t::Tee, args...; kwargs...) = tee(io -> nothing, t, args...; kwargs...)

function tee_capture(f; logfile = tempname(), suppress_terminal = false, suppress_logfile = false)
    open(suppress_logfile ? tempname() : logfile, "w+") do io
        io = Tee(suppress_terminal ? devnull : stderr, io)
        logger = ConsoleLogger(io)
        with_logger(logger) do
            f(io)
        end
    end
end

# https://discourse.julialang.org/t/redirect-stdout-and-stderr/13424/3
function redirect_to_files(f, outfile, errfile)
    open(outfile, "w") do out
        open(errfile, "w") do err
            redirect_stdout(out) do
                redirect_stderr(err) do
                    f()
                end
            end
        end
    end
end
redirect_to_tempfiles(f) = redirect_to_files(f, tempname() * ".log", tempname() * ".err")

function redirect_to_devnull(f)
    with_logger(ConsoleLogger(devnull)) do
        redirect_to_tempfiles() do
            f()
        end
    end
end

####
#### Optimizers
####

struct ADAM{N,T}
    η::T
    β::SVector{2,T}
    mt::SVector{N,T}
    vt::SVector{N,T}
    βp::SVector{2,T}
end
function ADAM{N,T}(η = 0.001, β = (0.9, 0.999)) where {N,T}
    @assert N >= 1
    S2 = SVector{2,T}
    SN = SVector{N,T}
    ADAM{N,T}(T(η), S2(β), zero(SN), zero(SN), ones(S2))
end

function update(∇::SVector{N,T}, o::ADAM{N,T}) where {N,T}
    @unpack η, β, mt, vt, βp = o

    ϵ  = T(1e-8)
    βp = @. βp * β
    ηt = η * √(1 - βp[2]) / (1 - βp[1])
    mt = @. β[1] * mt + (1 - β[1]) * ∇
    vt = @. β[2] * vt + (1 - β[2]) * ∇^2
    Δ  = @. ηt * mt / (√vt + ϵ)

    return Δ, ADAM{N,T}(η, β, mt, vt, βp)
end

@inline xform_periodic(t::S, lb::S, ub::S) where {N, T, S <: SVector{N,T}} = S(ntuple(i -> clamp(((lb[i] + ub[i])/2) + ((ub[i] - lb[i])/2) * sinpi(t[i]), lb[i], ub[i]), N))
@inline ∇xform_periodic(t::S, lb::S, ub::S) where {N, T, S <: SVector{N,T}} = S(ntuple(i -> ((ub[i] - lb[i])/2) * T(π) * cospi(t[i]), N))
@inline inv_xform_periodic(x::S, lb::S, ub::S) where {N, T, S <: SVector{N,T}} = S(ntuple(i -> asin(clamp((x[i] - ((lb[i] + ub[i])/2)) / ((ub[i] - lb[i])/2), -one(T), one(T))) / T(π), N))

function optimize(∇f, x0::SVector{N,T}, lb::SVector{N,T}, ub::SVector{N,T}, o::ADAM{N,T}; maxiter::Int = 1, xtol_rel = T(1e-3)) where {N,T}
    x = x0
    t = inv_xform_periodic(x, lb, ub)
    for i in 1:maxiter
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

####
#### Generate (moderately) realistic mock images
####

function mock_t2map_opts(::Type{T} = Float64; kwargs...) where {T}
    T2mapOptions{T}(;
        MatrixSize = (2,2,2),
        TE = 10e-3,
        nTE = 32,
        T2Range = (10e-3, 2.0),
        nT2 = 40,
        Reg = "lcurve",
        SetFlipAngle = 165.0,
        RefConAngle = 150.0,
        kwargs...
    )
end

function mock_t2parts_opts(::Type{T} = Float64; kwargs...) where {T}
    T2partOptions{T}(;
        MatrixSize = (2,2,2),
        nT2 = 40,
        T2Range = (10e-3, 2.0),
        SPWin = (10e-3, 40e-3),
        MPWin = (40e-3, 2.0),
        kwargs...
    )
end

# Mock CPMG image
function mock_image(o::T2mapOptions{T} = mock_t2map_opts(Float64); kwargs...) where {T}
    oldseed = Random.seed!(0)

    @unpack MatrixSize, TE, nTE = T2mapOptions(o; kwargs...)
    SNR = T(50)
    eps = T(10^(-SNR/20))

    α = o.SetFlipAngle === nothing ? T(165.0) : o.SetFlipAngle
    β = o.RefConAngle === nothing ? T(150.0) : o.RefConAngle
    mag() = T(0.85) .* EPGdecaycurve(nTE, α, TE, T(65e-3), T(1), β) .+
            T(0.15) .* EPGdecaycurve(nTE, α, TE, T(15e-3), T(1), β) # bi-exponential signal with EPG correction
    noise(m) = abs(m[1]) .* eps .* randn(T, size(m)) # gaussian noise of size SNR relative to signal amplitude
    noiseysignal() = (m = mag(); sqrt.((m .+ noise(m)).^2 .+ noise(m).^2)) # bi-exponential signal with rician noise

    M = zeros(T, (MatrixSize..., nTE))
    @inbounds for I in CartesianIndices(MatrixSize)
        M[I,:] .= T(1e5 + 1e5*rand()) .* noiseysignal()
    end

    Random.seed!(oldseed)
    return M
end

# Mock T2 distribution, computed with default parameters
function mock_T2_dist(o::T2mapOptions = mock_t2map_opts(Float64); kwargs...)
    T2mapSEcorr(mock_image(o; kwargs...), T2mapOptions(o; kwargs..., Silent = true))[2]
end

# Simple benchmark
function mock_t2map_benchmark(; plot = false, kwargs...)
    t2map_opts = mock_t2map_opts(; MatrixSize = (64,64,64), SetFlipAngle = 165.0, RefConAngle = 180.0, SaveResidualNorm = true, kwargs...)
    t2part_opts = T2partOptions(t2map_opts; SPWin = (t2map_opts.T2Range[1], 40e-3), MPWin = (40e-3, t2map_opts.T2Range[2]))

    image = mock_image(t2map_opts)
    # t2maps, t2dist = Main.@btime T2mapSEcorr($image, $(T2mapOptions(t2map_opts; SetFlipAngle = nothing)))
    t2maps, t2dist = @time T2mapSEcorr(image, T2mapOptions(t2map_opts; SetFlipAngle = nothing))
    t2part = @time T2partSEcorr(t2dist, t2part_opts)

    if plot
        dα = vec(t2maps["alpha"] .- t2map_opts.SetFlipAngle)
        relerr = vec(t2maps["resnorm"] ./ sqrt.(sum(abs2, image; dims = 4)))
        Main.UnicodePlots.histogram(dα; title = "MAE flip angle = $(mean(abs.(dα)))") |> display
        Main.UnicodePlots.histogram(relerr; title = "relative resnorm = $(mean(relerr))") |> display
    end

    return t2maps, t2dist, t2part
end
