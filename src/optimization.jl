####
#### Rootfinding methods
####

#=
`bisect_root` and `brent_root` are modified codes from Roots.jl:
    https://github.com/JuliaMath/Roots.jl/blob/8a5ff76e8e8305d4ad5719fe1dd665d8a7bd7ec3/src/simple.jl
    https://github.com/JuliaMath/Roots.jl/blob/c1684335e891e518ce304cf99015af0f1a2cb2f4/src/Bracketing/brent.jl#L1

The MIT License (MIT) Copyright (c) 2013 John C. Travers
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#

#### Bisection

function bisect_root(f, a::Number, b::Number; xatol = nothing, xrtol = nothing, ftol = nothing, maxiters::Int = 100)
    T = promote_type(typeof(float(a)), typeof(float(b)))
    x₁, x₂ = T(a), T(b)
    y₁, y₂ = f(x₁), f(x₂)
    return bisect_root(
        f, x₁, x₂, y₁, y₂;
        xatol = xatol === nothing ? zero(T) : T(xatol),
        xrtol = xrtol === nothing ? zero(T) : T(xrtol),
        ftol = ftol === nothing ? zero(T) : T(ftol),
        maxiters,
    )[3:4]
end

function bisect_root(f, x₁::T, x₂::T, y₁::T, y₂::T; xatol::T = zero(T), xrtol::T = zero(T), ftol::T = zero(T), maxiters::Int = 100) where {T}
    if x₁ == x₂ || y₁ * y₂ >= 0
        # Degenerate interval; return endpoint closest to zero
        xₘ, yₘ = abs(y₁) <= abs(y₂) ? (x₁, y₁) : (x₂, y₂)
        return (x₁, y₁, xₘ, yₘ, x₂, y₂)
    end

    if y₂ < 0
        x₁, x₂, y₁, y₂ = x₂, x₁, y₂, y₁
    end

    xₘ = (x₁ + x₂) / 2
    yₘ = f(xₘ)

    for iter in 1:maxiters
        if abs(yₘ) <= ftol || 2 * min(abs(xₘ - x₁), abs(x₂ - xₘ)) <= xatol + xrtol * abs(xₘ)
            break
        end

        if yₘ < 0
            x₁, y₁ = xₘ, yₘ
        else
            x₂, y₂ = xₘ, yₘ
        end

        xₘ = (x₁ + x₂) / 2
        yₘ = f(xₘ)

        if !isfinite(yₘ)
            # Evaluation failed; return endpoint with smallest function value
            xₘ, yₘ = abs(y₁) <= abs(y₂) ? (x₁, y₁) : (x₂, y₂)
            break
        end
    end

    return (x₁, y₁, xₘ, yₘ, x₂, y₂)
end

#### Brent's method (root-finding)

function brent_root(f, x₀::T, x₁::T, fx₀::T = f(x₀), fx₁::T = f(x₁); xatol::T = zero(T), xrtol::T = zero(T), ftol::T = zero(T), maxiters::Int = 100) where {T}
    # Assert bracketing
    fx₀ == 0 && return (x₀, fx₀)
    fx₁ == 0 && return (x₁, fx₁)
    @assert fx₀ * fx₁ < 0 "Root must be bracketed, but f(x = $x₀) = $fx₀ and f(x = $x₁) = $fx₁"

    # Initialize Brent state
    a, b, fa, fb = x₀, x₁, fx₀, fx₁
    if abs(fa) < abs(fb)
        a, b, fa, fb = b, a, fb, fa
    end
    c, d, fc, mflag = x₀, x₀, fx₀, true

    for iter in 1:maxiters
        # Check for sufficiently small bracketing interval
        abs(b - a) <= 2 * (xatol + xrtol * abs(b)) && return (b, fb)

        # Next step depends on points; inverse quadratic
        s::T = inverse_quadratic_step(a, b, c, fa, fb, fc)
        (isnan(s) || isinf(s)) && (s = secant_step(a, b, fa, fb))

        # Guard step
        u, v = (3a + b) / 4, b
        if u > v
            u, v = v, u
        end

        tol = max(xatol, xrtol * max(abs(b), abs(c), abs(d)))
        if !(u < s < v) ||
           (mflag && abs(s - b) >= abs(b - c) / 2) ||
           (!mflag && abs(s - b) >= abs(b - c) / 2) ||
           (mflag && abs(b - c) <= tol) ||
           (!mflag && abs(c - d) <= tol)
            s = (a + b) / 2
            mflag = true
        else
            mflag = false
        end

        fs::T = f(s)
        iszero(fs) && return (s, fs) # exact root found
        (isnan(fs) || isinf(fs)) && return (b, fb) # function failed; return current best guess
        abs(fs) <= ftol && return (s, fs) # function value is below tolerance

        c, fc, d = b, fb, c
        if sign(fa) * sign(fs) < 0
            b, fb = s, fs
        else
            a, fa = s, fs
        end

        if abs(fa) < abs(fb)
            a, b, fa, fb = b, a, fb, fa
        end
    end

    return (b, fb)
end

#### Newton's method with bisection

function newton_bisect_root(f_∂f, x0::T, x1::T, x2::T, y1::T = f_∂f(x1)[1], y2::T = f_∂f(x2)[1]; xrtol::T = √eps(T), xatol::T = eps(T), ftol::T = zero(T), maxiters::Int = 100) where {T <: AbstractFloat}
    # Check initial points
    y1 == 0 && return (x1, y1)
    y2 == 0 && return (x2, y2)
    @assert y1 * y2 < 0 "Root must be bracketed, but f(x = $x1) = $y1 and f(x = $x2) = $y2"
    (y1 > 0) && ((x1, x2, y1, y2) = (x2, x1, y2, y1))

    # Initialize the estimate for the root
    y = T(NaN)
    x = x_old = x0
    Δx = Δx_old = x2 - x1

    for iter in 1:maxiters
        # Evaluate function and derivative at the current estimate
        y, dy = f_∂f(x)
        abs(y) <= ftol && return (x, y)

        # Update bracketing interval
        if y < 0
            x1 = x
        else
            x2 = x
        end

        # Take bisection step if Newton step is outside interval, or not converging fast enough
        if (((x - x2) * dy - y) * ((x - x1) * dy - y) > 0) || (2 * abs(y) > abs(Δx_old * dy))
            x_old, Δx_old = x, Δx
            Δx = (x2 - x1) / 2 # bisection step
            x = (x1 + x2) / 2
        else
            x_old, Δx_old = x, Δx
            Δx = -y / dy # Newton step
            x += Δx
        end

        # Check for convergence
        Δx_tol = xatol + xrtol * abs(x)
        min(abs(Δx), abs(x - x_old)) <= Δx_tol && break
    end

    return (x, y)
end

#### Root-finding helper functions

@inline function secant_step(a, b, fa, fb)
    # See: https://github.com/JuliaMath/Roots.jl/blob/c1684335e891e518ce304cf99015af0f1a2cb2f4/src/utils.jl#L70
    return a - fa * (b - a) / (fb - fa)
end

@inline function inverse_quadratic_step(a::T, b, c, fa, fb, fc) where {T}
    # See: https://github.com/JuliaMath/Roots.jl/blob/c1684335e891e518ce304cf99015af0f1a2cb2f4/src/utils.jl#L114
    s = zero(T)
    s += a * fb * fc / (fa - fb) / (fa - fc) # quad step
    s += b * fa * fc / (fb - fa) / (fb - fc)
    s += c * fa * fb / (fc - fa) / (fc - fb)
    return s
end

function bracket_root_monotonic(f, a::T, δ::T; dilate = 1, mono = +1, maxiters::Int = 100) where {T <: AbstractFloat}
    # Find bracketing interval for the root of a monotonic function.
    # The function is assumed to be increasing (decreasing) when `mono` is positive (negative).
    # Begin at point `a` and step by `δ` until the function changes sign.
    # The step size is multiplied by `dilate` after each step.
    @assert δ > 0 "Initial step size must be positive"
    @assert dilate >= 1 "Dilation factor must be at least 1"
    @assert mono != 0 "Monotonicity must be non-zero"
    fa = f(a)
    !isfinite(fa) && return (a, a, T(NaN), T(NaN))
    fa == 0 && return (a, a, fa, fa)
    sgn_δ = sign(T(mono)) * sign(fa)
    b = a - sgn_δ * δ
    fb = f(b)
    !isfinite(fb) && return (a, a, fa, fa)
    fb == 0 && return (b, b, fb, fb)
    δ *= T(dilate)
    cnt = 0
    while fa * fb > 0 && cnt < maxiters
        a, fa = b, fb
        b = a - sgn_δ * δ
        fb = f(b)
        !isfinite(fb) && return (a, a, fa, fa)
        fb == 0 && return (b, b, fb, fb)
        δ *= T(dilate)
        cnt += 1
    end
    return a < b ? (a, b, fa, fb) : (b, a, fb, fa)
end

function bracket_local_minimum(f, ∂f_∂²f, x1::T, x2::T; xrtol::T = √eps(T), xatol::T = √eps(T), maxdepth::Int = 5) where {T <: AbstractFloat}
    a, b = minmax(x1, x2)
    tol = xatol + xrtol * max(abs(a), abs(b))

    # Search space is interior to [a, b] and slightly assymetric to avoid hitting exact zeros, midpoints, etc., which are susceptible to numerical issues
    a⁺ = a + T(1 - 0.004849834917525) * tol / 2
    b⁻ = b - T(1 + 0.004849834917525) * tol / 2
    ∂ua⁺, ∂²ua⁺ = ∂f_∂²f(a⁺)
    ∂ub⁻, ∂²ub⁻ = ∂f_∂²f(b⁻)

    return bracket_local_minimum(f, ∂f_∂²f, a⁺, b⁻, ∂ua⁺, ∂ub⁻, ∂²ua⁺, ∂²ub⁻, tol, 0, maxdepth)
end

function bracket_local_minimum(f, ∂f_∂²f, a::T, b::T, ∂ua::T, ∂ub::T, ∂²ua::T, ∂²ub::T, tol::T, depth::Int, maxdepth::Int) where {T <: AbstractFloat}
    spl = CubicHermiteInterpolator(a, b, ∂ua, ∂ub, ∂²ua, ∂²ub)
    (x1, x2, x3), (s1, s2, s3) = signedroots(spl, tol / 2) # roots must be at least `tol / 2` from the interval endpoints

    bracket, succ = if isnan(x1)
        # No local minimum found; return failed bracket
        (a, b, ∂ua, ∂ub, ∂²ua, ∂²ub), false
    elseif isnan(x2)
        # One local extremum found; return success if it is a minimum
        (a, b, ∂ua, ∂ub, ∂²ua, ∂²ub), s1 > 0 && ∂ua * ∂ub < 0
    elseif isnan(x3)
        # Two local extrema found; evaluate midpoint
        x = (x1 + x2) / 2
        ∂ux, ∂²ux = ∂f_∂²f(x)
        if s1 > 0
            (a, x, ∂ua, ∂ux, ∂²ua, ∂²ux), ∂ua * ∂ux < 0
        elseif s2 > 0
            (x, b, ∂ux, ∂ub, ∂²ux, ∂²ub), ∂ux * ∂ub < 0
        else
            (a, b, ∂ua, ∂ub, ∂²ua, ∂²ub), false
        end
    else
        if s2 > 0
            # One local minimum, two local maxima
            x⁻, x⁺ = (x1 + x2) / 2, (x2 + x3) / 2
            ∂ux⁻, ∂²ux⁻ = ∂f_∂²f(x⁻)
            ∂ux⁺, ∂²ux⁺ = ∂f_∂²f(x⁺)
            (x⁻, x⁺, ∂ux⁻, ∂ux⁺, ∂²ux⁻, ∂²ux⁺), ∂ux⁻ * ∂ux⁺ < 0
        elseif s1 > 0 && s3 > 0
            # Two local minima, one local maximum
            if spl(x1) < spl(x3)
                x = (x1 + x2) / 2
                ∂ux, ∂²ux = ∂f_∂²f(x)
                (a, x, ∂ua, ∂ux, ∂²ua, ∂²ux), ∂ua * ∂ux < 0
            else
                x = (x2 + x3) / 2
                ∂ux, ∂²ux = ∂f_∂²f(x)
                (x, b, ∂ux, ∂ub, ∂²ux, ∂²ub), ∂ux * ∂ub < 0
            end
        else
            # No local minima; return failed bracket
            (a, b, ∂ua, ∂ub, ∂²ua, ∂²ub), false
        end
    end

    if succ || depth >= maxdepth
        # Successful, or reached maximum depth
        return bracket, succ
    else
        # Bisect the bracket and continue searching
        x = (a + b) / 2
        ∂ux, ∂²ux = ∂f_∂²f(x)

        bracket1, succ1 = bracket_local_minimum(f, ∂f_∂²f, a, x, ∂ua, ∂ux, ∂²ua, ∂²ux, tol, depth + 1, maxdepth)
        succ1 && return bracket1, succ1

        bracket2, succ2 = bracket_local_minimum(f, ∂f_∂²f, x, b, ∂ux, ∂ub, ∂²ux, ∂²ub, tol, depth + 1, maxdepth)
        succ2 && return bracket2, succ2

        return bracket, succ
    end
end

####
#### Optimization methods
####

#=
Brent-Dekker minimization method. The code for `brent_minimize` is modified from Optim.jl:
    https://github.com/JuliaNLSolvers/Optim.jl/blob/1189ba0347ba567e43d1d4de94588aaf8a9e3ac0/src/univariate/solvers/brent.jl#L23

See:
    R. P. Brent (2002) Algorithms for Minimization Without Derivatives. Dover edition. Chapter 6, Section 8.

Optim.jl is licensed under the MIT License:
> Copyright (c) 2012: John Myles White, Tim Holy, and other contributors.
> Copyright (c) 2016: Patrick Kofod Mogensen, John Myles White, Tim Holy, and other contributors.
> Copyright (c) 2017: Patrick Kofod Mogensen, Asbjørn Nilsen Riseth, John Myles White, Tim Holy, and other contributors.
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#

#### Brent's method (minimization)

function brent_minimize(f, x₁::T, x₂::T; xrtol::T = √eps(T), xatol::T = √eps(T), maxiters::Int = 100) where {T <: AbstractFloat}
    @assert x₁ <= x₂ "x₁ must be less than x₂"

    α = 2 - T(φ) # α ≈ 0.381966
    x = x₁ + α * (x₂ - x₁)
    y = f(x)

    Δx_old = Δx = zero(T)
    x_older = x_old = x
    y_older = y_old = y

    iter = 0

    while iter < maxiters
        p = zero(T)
        q = zero(T)
        xₘ = (x₂ + x₁) / 2

        Δx_tol = xatol + xrtol * abs(x)
        if abs(x - xₘ) + (x₂ - x₁) / 2 <= 2 * Δx_tol
            break # converged
        end

        iter += 1

        if abs(Δx_old) > Δx_tol
            # Compute parabola interpolation
            # x + p/q is the optimum of the parabola
            # Also, q is guaranteed to be positive

            r = (x - x_old) * (y - y_older)
            q = (x - x_older) * (y - y_old)
            p = (x - x_older) * q - (x - x_old) * r
            q = 2 * (q - r)

            if q > 0
                p = -p
            else
                q = -q
            end
        end

        if abs(p) < abs(q * Δx_old / 2) && p < q * (x₂ - x) && p < q * (x - x₁)
            # Parabolic interpolation step
            Δx_old = Δx
            Δx = p / q

            # The function must not be evaluated too close to x₁ or x₂
            x_tmp = x + Δx
            if (x_tmp - x₁) < 2 * Δx_tol || (x₂ - x_tmp) < 2 * Δx_tol
                Δx = ifelse(x < xₘ, Δx_tol, -Δx_tol)
            end
        else
            # Golden section step
            Δx_old = ifelse(x < xₘ, x₂ - x, x₁ - x)
            Δx = α * Δx_old
        end

        # The function must not be evaluated too close to x
        if abs(Δx) >= Δx_tol
            x_new = x + Δx
        else
            x_new = x + ifelse(Δx > 0, Δx_tol, -Δx_tol)
        end

        y_new = f(x_new)

        # Update x's and y's
        if y_new < y
            if x_new < x
                x₂ = x
            else
                x₁ = x
            end
            x_older, x_old, x = x_old, x, x_new
            y_older, y_old, y = y_old, y, y_new
        else
            if x_new < x
                x₁ = x_new
            else
                x₂ = x_new
            end
            if y_new <= y_old || x_old == x
                x_older, x_old = x_old, x_new
                y_older, y_old = y_old, y_new
            elseif y_new <= y_older || x_older == x || x_older == x_old
                x_older = x_new
                y_older = y_new
            end
        end
    end

    return (x, y)
end

function newton_bisect_minimize(f, ∂f_∂²f, x1::T, x2::T; xrtol::T = √eps(T), xatol::T = √eps(T), maxdepth::Int = 5, kwargs...) where {T <: AbstractFloat}
    (x⁻, x⁺, ∂ux⁻, ∂ux⁺, ∂²ux⁻, ∂²ux⁺), succ = bracket_local_minimum(f, ∂f_∂²f, x1, x2; xrtol, xatol, maxdepth)
    if !succ
        return (T(NaN), T(NaN))
    else
        x, dx = newton_bisect_root(∂f_∂²f, (x⁻ + x⁺) / 2, x⁻, x⁺, ∂ux⁻, ∂ux⁺; xrtol, xatol, kwargs...)
        return (x, f(x))
    end
end
