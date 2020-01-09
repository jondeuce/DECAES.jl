####
#### Miscellaneous utils
####
ndigits(x::Int) = x == 0 ? 1 : floor(Int, log10(abs(x))) + 1
logrange(a::T, b::T, len::Int) where {T} = T(10) .^ range(log10(a), log10(b); length = len)
normcdf(x::T) where {T} = erfc(-x/sqrt(T(2)))/2 # Cumulative distribution for normal distribution
normccdf(x::T) where {T} = erfc(x/sqrt(T(2)))/2 # Compliment of normcdf, i.e. 1 - normcdf(x)

function set_diag!(A::AbstractMatrix, val)
    @inbounds for i in 1:min(size(A)...)
        A[i,i] = val
    end
    return A
end

function find_nearest(r::AbstractRange, x::Number)
    idx = x <= r[1] ? 1 :
          x >= r[end] ? length(r) :
          clamp(round(Int, 1 + (x - r[1]) / step(r)), 1, length(r))
    r[idx], idx # nearest value in r to x and corresponding index
end

# Macro copied from DrWatson.jl in order to not depend on the package
#   See: https://github.com/JuliaDynamics/DrWatson.jl
macro ntuple(vars...)
   args = Any[]
   for i in 1:length(vars)
       push!(args, Expr(:(=), esc(vars[i]), :($(esc(vars[i])))))
   end
   expr = Expr(:tuple, args...)
   return expr
end

# Threaded `foreach` construct, borrowing implementation from ThreadTools.jl
#   See: https://github.com/baggepinnen/ThreadTools.jl/blob/55aaf2bbe735e52cefaad143e7614d4f00e312b0/src/ThreadTools.jl#L57
function tforeach(f, args...; blocksize = 1, callback = () -> nothing)
    # One available thread: call regular foreach
    if Threads.nthreads() == 1
        return foreach(args...) do (args...)
            f(args...)
            callback()
        end
    end

    tasks = if blocksize == 1
        # Spawn one task for each function call
        map(args...) do (args...)
            Threads.@spawn begin
                f(args...)
                callback()
            end
        end
    else
        # Spawn one task for each `blocksize` function calls
        blockargs = Iterators.partition(Iterators.zip(args...), blocksize)
        map(blockargs) do blockargs
            Threads.@spawn begin
                foreach(args -> f(args...), blockargs)
                callback()
            end
        end
    end

    # Fetch task results, running them in parallel
    foreach(wait, tasks)
end

# Running linear regression
@with_kw mutable struct RunningLinReg{T}
    k::Int = 0
    Mx::Tuple{T,T} = (zero(T), zero(T))
    My::Tuple{T,T} = (zero(T), zero(T))
    Sx::Tuple{T,T} = (zero(T), zero(T))
    Sy::Tuple{T,T} = (zero(T), zero(T))
    Sxy::Tuple{T,T} = (zero(T), zero(T))
end

function (r::RunningLinReg)(x)
    x̄, ȳ = mean(r)
    β = cov(r) / var(r)[1]
    α = ȳ - β * x̄
    return α + β * x
end
Statistics.mean(r::RunningLinReg) = (r.Mx[1], r.My[1])
Statistics.var(r::RunningLinReg) = (r.Sx[1]/(r.k-1), r.Sy[1]/(r.k-1))
Statistics.std(r::RunningLinReg) = (sqrt(r.Sx[1]/(r.k-1)), sqrt(r.Sy[1]/(r.k-1)))
Statistics.cov(r::RunningLinReg) = (r.k/(r.k-1)) * r.Sxy[1]
Base.length(r::RunningLinReg) = r.k

function Base.push!(r::RunningLinReg{T}, x, y) where {T}
    r.k += 1
    if r.k == 1
        r.Mx = (T(x), zero(T))
        r.My = (T(y), zero(T))
    else
        r.Mx, r.Sx = _var_update(r.Mx, r.Sx, x, r.k)
        r.My, r.Sy = _var_update(r.My, r.Sy, y, r.k)
        r.Sxy = _cov_update(r.Mx, r.My, r.Sxy, x, y, r.k)
    end
    return r
end
function _var_update(M::Tuple{T,T}, S::Tuple{T,T}, x, k) where {T}
    Mk, Mk′, Sk, Sk′ = M..., S...
    Mk, Mk′ = T(Mk + (x - Mk)/k), Mk
    Sk, Sk′ = T(Sk + (x - Mk′)*(x - Mk)), Sk
    return (Mk, Mk′), (Sk, Sk′)
end
function _cov_update(Mx::Tuple{T,T}, My::Tuple{T,T}, Sxy::Tuple{T,T}, x, y, k) where {T}
    mx, mx′, my, my′, sxy, sxy′ = Mx..., My..., Sxy...
    sxy, sxy′ = T(sxy + (k - 1) * (x - mx′) * (y - my′) / k^2 - sxy / k), sxy
    return (sxy, sxy′)
end

####
#### Timing utilities
####
const TIMER = TimerOutput() # Global timer object

tic() = time()
toc(t) = tic() - t

function hour_min_sec(t)
    hour = floor(Int, t/3600)
    min = floor(Int, (t - 3600*hour)/60)
    sec = floor(Int, t - 3600*hour - 60*min)
    return @ntuple(hour, min, sec)
end

function pretty_time(t)
    if isnan(t) || isinf(t)
        "--h:--m:--s"
    else
        hh, mm, ss = hour_min_sec(t)
        lpad(hh,2,"0") * "h:" * lpad(mm,2,"0") * "m:" * lpad(ss,2,"0") * "s"
    end
end

# Macro for timing arbitrary code snippet and printing time
macro showtime(ex, msg, verb)
    quote
        $(esc(verb)) && println(stdout, "")
        $(esc(verb)) && @info $(esc(msg)) * " ..."
        local val
        local t = @elapsed val = $(esc(ex))
        $(esc(verb)) && @info "Done ($(round(t; digits = 2)) seconds)"
        val
    end
end

####
#### Spline tools
####

# These algorithms were changed compared to the original MATLAB version:
# instead of brute force searching through the splines, numerical methods
# are performed which are much more efficient and much more accurate.
# For direct comparison of results with the MATLAB version, the brute force
# version is implimented and can be used by setting the LEGACY flag.
const LEGACY = Ref(false)
spline_opt(args...)  = LEGACY[] ? _spline_opt_legacy(args...)  : _spline_opt(args...)
spline_root(args...) = LEGACY[] ? _spline_root_legacy(args...) : _spline_root(args...)

function _make_spline(X, Y)
    @assert length(X) == length(Y) && length(X) > 1
    deg_spline = min(3, length(X)-1)
    spl = Dierckx.Spline1D(X, Y; k = deg_spline)
end

function _build_polynomials(spl)
    k = spl.k
    t = get_knots(spl)[1:end-1]
    coeffs = zeros(k+1, length(t))
    coeffs[1, :] .= spl.(t)
    for m in 1:k
        coeffs[m+1, :] .= derivative(spl, t, m) ./ factorial(m)
    end
    return [Poly(coeffs[:, j]) for j in 1:size(coeffs, 2)]
end

# Global minimization through fitting a spline to data (X, Y)
function _spline_opt(spl::Dierckx.Spline1D)
    knots = Dierckx.get_knots(spl)
    polys = _build_polynomials(spl)
    x, y = knots[1], polys[1](0) # initial lefthand point
    @inbounds for (i,p) in enumerate(polys)
        x0, x1 = knots[i], knots[i+1] # spline section endpoints
        _x, _y = x1, p(x1 - x0) # check right endpoint
        (_y < y) && (x = _x; y = _y)
        r = PolynomialRoots.roots(Polynomials.coeffs(polyder(p)))
        for ri in r
            if imag(ri) == 0 # real roots only
                xi = x0 + real(ri)
                if x0 <= xi <= x1 # filter roots within range
                    _x, _y = xi, p(real(ri))
                    (_y < y) && (x = _x; y = _y)
                end
            end
        end
    end

    return @ntuple(x, y)
end
_spline_opt(X::AbstractVector, Y::AbstractVector) = _spline_opt(_make_spline(X, Y))

# MATLAB spline optimization performs global optimization by sampling the spline
# fit to data (X, Y) at points X[1]:0.001:X[end], and uses the minimum value.
# This isn't very efficient; instead we use exact spline optimization and return
# the (x,y) pair such that x ∈ X[1]:0.001:X[end] is nearest to the exact optimum
function _spline_opt_legacy(spl::Dierckx.Spline1D)
    xopt, yopt = _spline_opt(spl)
    knots = get_knots(spl)
    Xs = knots[1]:eltype(knots)(0.001):knots[end] # from MATLAB version
    x, _ = find_nearest(Xs, xopt) # find nearest x in Xs to xopt
    y = spl(x)
    return @ntuple(x, y)
end
_spline_opt_legacy(X::AbstractVector, Y::AbstractVector) = _spline_opt_legacy(_make_spline(X, Y))

# Root finding through fitting a spline to data (X, Y)
function _spline_root(spl::Dierckx.Spline1D, value::Number = 0)
    knots = Dierckx.get_knots(spl)
    polys = _build_polynomials(spl)
    (value != 0) && (polys .-= value)
    x = nothing
    @inbounds for (i,p) in enumerate(polys)
        x0, x1 = knots[i], knots[i+1] # spline section endpoints
        r = Polynomials.roots(p) # zeros of derivative
        for ri in r
            if imag(ri) == 0 # real roots only
                xi = x0 + real(ri)
                if x0 <= xi <= x1 # filter roots within range
                    x = (x === nothing) ? xi : min(x, xi)
                end
            end
        end
        if x !== nothing
            return x
        end
    end
    if x === nothing
        warn("No root was found on the spline domain; returning nothing")
    end

    return x
end
_spline_root(X::AbstractVector, Y::AbstractVector, value::Number = 0) = _spline_root(_make_spline(X, Y), value)

# Brute force root finding through fitting a spline to data (X, Y):
# MATLAB implementation of spline root finding performs root finding by sampling the
# spline fit to data (X, Y) at points X[1]:0.001:X[end], and uses the nearest value.
# This isn't very efficient; instead we use exact spline root finding and return
# the nearest x ∈ X[1]:0.001:X[end] such that the y value is nearest zero
function _spline_root_legacy(spl::Dierckx.Spline1D, value = 0)

    knots = Dierckx.get_knots(spl)
    Xs = knots[1]:eltype(knots)(0.001):knots[end] # from MATLAB version
    xroot = _spline_root(spl, value)
    _, i = find_nearest(Xs, xroot) # find nearest x in Xs to xroot

    # Note that the above finds the nearest x value, but we need the x corresponding to the
    # nearest y value. Since we are near the minimum, just search locally for any minima
    dy(i) = abs(spl(@inbounds(Xs[clamp(i,1,length(Xs))])) - value)
    yleft, y, yright = dy(i-1), dy(i), dy(i+1)
    while !(yleft ≥ y ≤ yright) # search for local min
        if yleft < y
            i -= 1 # shift left
            yleft, y, yright = dy(i-1), yleft, y
        elseif yright < y
            i += 1 # shift right
            yleft, y, yright = y, yright, dy(i+1)
        else
            break
        end
    end
    x = Xs[i]

    return x
end
_spline_root_legacy(X::AbstractVector, Y::AbstractVector, value = 0) = _spline_root_legacy(_make_spline(X, Y), value)

# Perform global optimization over a function `f` which may be evaluated only
# on elements of a discrete vector `X` using surrogate spline functions.
#   `f` is a function taking two inputs `X` and `i` where `i` is an index into `X`
#   `X` discrete vector in which to evaluate `f`
#   `min_n_eval` minimum number of elements of `X` which are evaluated
function surrogate_spline_opt(f, X, min_n_eval::Int = length(X))
    @assert length(X) >= 2

    # Check if X has less than min_n_eval points
    min_n_eval = max(2, min_n_eval)
    if 2 <= length(X) <= min_n_eval
        Y = [f(X,i) for i in 1:length(X)]
        return spline_opt(X, Y)
    end

    # Update observed evaluations, returning true if converged
    function update_state!(Is, Xi, Yi, x, y)
        converged = false
        for i in 1:length(Is)-1
            # Find interval containing x
            if Xi[i] <= x <= Xi[i+1]
                iL, iR = Is[i], Is[i+1]
                iM = (iL + iR) ÷ 2
                if iM == iL || iM == iR
                    # interval cannot be reduced further
                    converged = true
                else
                    # insert and evaluate midpoint
                    insert!(Is, i+1, iM)
                    insert!(Xi, i+1, X[iM])
                    insert!(Yi, i+1, f(X,iM))
                end
                break
            end
        end
        return converged
    end

    # Initialize state
    Is = round.(Int, range(1, length(X), length = min_n_eval))
    Xi = X[Is]
    Yi = [f(X,i) for i in Is]

    # Surrogate minimization
    while true
        x, y = spline_opt(Xi, Yi) # current global minimum estimate
        if update_state!(Is, Xi, Yi, x, y) # update and check for convergence
            return @ntuple(x, y)
        end
    end
end

####
#### Generate (moderately) realistic mock images
####

# Return an (MatrixSize..., nTE) sized array of type T containing random
# bi-exponential signals with a small amount of noise.
function mock_image(::Type{T} = Float64; MatrixSize::NTuple{3,Int} = (2,2,2), nTE::Int = 32, eps = 0.002) where {T}
    oldseed = Random.seed!(0)
    mag() = T(0.8) .* exp.(.-inv(5.5+rand(T)).*(1:nTE)) .+ T(0.2) .* exp.(.-inv(1.5+0.5*rand(T)).*(1:nTE))
    noisey(ε) = (m = mag(); return abs.(m .+ m[1] .* ε .* randn(eltype(m), size(m))))
    M = zeros(T, (MatrixSize..., nTE))
    @inbounds for I in CartesianIndices(MatrixSize)
        M[I,:] .= T(1e4) .* noisey(eps)
    end
    Random.seed!(oldseed)
    return M
end

# Mock T2 distribution, computed with default parameters
mock_T2_dist(args...; nT2 = 40, kwargs...) =
    T2mapSEcorr(mock_image(args...; kwargs...); nT2 = nT2, Silent = true)[2]
