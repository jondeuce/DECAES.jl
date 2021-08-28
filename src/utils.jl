####
#### Miscellaneous utils
####

ndigits(x::Int) = x == 0 ? 1 : floor(Int, log10(abs(x))) + 1
logrange(a::Real, b::Real, len::Int) = (r = exp.(range(log(a), log(b); length = len)); r[1] = a; r[end] = b; return r)
normcdf(x::T) where {T} = erfc(-x/sqrt(T(2)))/2 # Cumulative distribution for normal distribution
normccdf(x::T) where {T} = erfc(x/sqrt(T(2)))/2 # Compliment of normcdf, i.e. 1 - normcdf(x)

@inline mul_im(z::Complex) = Complex(-imag(z), real(z)) # optimized i*(a+b*i) = -b+a*i

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

default_blocksize() = 64

####
#### Timing utilities
####

const GLOBAL_TIMER = TimerOutput() # Global timer object
const THREAD_LOCAL_TIMERS = [TimerOutput() for _ in 1:Threads.nthreads()] # Thread-local timer objects
RESET_TIMERS!() = foreach(reset_timer!, THREAD_LOCAL_TIMERS)
TIMER() = @inbounds THREAD_LOCAL_TIMERS[Threads.threadid()]

tic() = time()
toc(t) = tic() - t

function hour_min_sec(t)
    hour = floor(Int, t/3600)
    min = floor(Int, (t - 3600*hour)/60)
    sec = floor(Int, t - 3600*hour - 60*min)
    return (; hour, min, sec)
end

function pretty_time(t)
    if isnan(t) || isinf(t)
        "--h:--m:--s"
    else
        hh, mm, ss = hour_min_sec(t)
        lpad(hh,2,"0") * "h:" * lpad(mm,2,"0") * "m:" * lpad(ss,2,"0") * "s"
    end
end

@with_kw mutable struct DECAESProgress
    progmeter::Progress
    io::IO
    iobuf::IOBuffer
    iolock::ReentrantLock = Threads.ReentrantLock()
    last_msg::AbstractString = ""
end

function DECAESProgress(io::IO, n::Int, desc::AbstractString; kwargs...)
    iobuf = IOBuffer()
    DECAESProgress(
        io = io,
        iobuf = iobuf,
        progmeter = Progress(n;
            dt = 0.0, desc = desc, color = :cyan, output = iobuf, barglyphs = BarGlyphs("[=> ]"),
            kwargs...
        )
    )
end
DECAESProgress(n::Int, desc::AbstractString; kwargs...) = DECAESProgress(stderr, n, desc; kwargs...)

function ProgressMeter.next!(p::DECAESProgress)
    next!(p.progmeter)
    msg = String(take!(p.iobuf)) # take!(::IOBuffer) is threadsafe
    if !isempty(msg)
        msg = replace(msg, "\r" => "")
        msg = replace(msg, "\u1b[K" => "")
        msg = replace(msg, "\u1b[A" => "")
        if msg != p.last_msg
            lock(p.iolock) do
                println(p.io, msg)
                p.last_msg = msg
            end
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
#### Logging
####

# https://discourse.julialang.org/t/write-to-file-and-stdout/35042/3
struct Tee{T <: Tuple} <: IO
    streams::T
end
Tee(streams::IO...) = Tee(streams)
Base.flush(t::Tee) = tee(t)
Base.write(t::Tee, args...; kwargs...) = tee(Base.write, t, args...; kwargs...)
Base.print(t::Tee, args...; kwargs...) = tee(Base.print, t, args...; kwargs...)
Base.println(t::Tee, args...; kwargs...) = tee(Base.println, t, args...; kwargs...)
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
#### Generate (moderately) realistic mock images
####

# Mock CPMG image
function mock_t2map_opts(::Type{T} = Float64; kwargs...) where {T}
    T2mapOptions{T}(;
        MatrixSize = (2,2,2),
        TE = 10e-3,
        nTE = 32,
        T2Range = (10e-3, 2.0),
        nT2 = 40,
        Reg = "lcurve",
        SetFlipAngle = 165.0,
        SetRefConAngle = 150.0,
        kwargs...
    )
end

# Mock CPMG image
function mock_image(o::T2mapOptions{T} = mock_t2map_opts(Float64); kwargs...) where {T}
    oldseed = Random.seed!(0)

    @unpack MatrixSize, TE, nTE = T2mapOptions(o; kwargs...)
    SNR = T(50)
    eps = T(10^(-SNR/20))

    flipangle = o.SetFlipAngle === nothing ? T(165.0) : o.SetFlipAngle
    refcon = o.SetRefConAngle === nothing ? T(150.0) : o.SetRefConAngle
    mag() = T(0.85) .* EPGdecaycurve(nTE, flipangle, TE, T(65e-3), T(1), refcon) .+
            T(0.15) .* EPGdecaycurve(nTE, flipangle, TE, T(15e-3), T(1), refcon) # bi-exponential signal with EPG correction
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
