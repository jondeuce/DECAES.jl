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

        # if succ1 && succ2
        #     ua, ux, ub = f(a), f(x), f(b)
        #     spl1 = CubicHermiteInterpolator(a, x, ua, ux, ∂ua, ∂ux)
        #     spl2 = CubicHermiteInterpolator(x, b, ux, ub, ∂ux, ∂ub)
        #     if minimize(spl1) < minimize(spl2)
        #         return bracket1, succ1
        #     else
        #         return bracket2, succ2
        #     end
        # elseif succ1
        #     return bracket1, succ1
        # elseif succ2
        #     return bracket2, succ2
        # else
        #     return bracket, succ
        # end
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
    @assert x₁ < x₂ "x₁ must be less than x₂"

    φ = T(Base.MathConstants.φ) # φ ≈ 1.618034
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

# Given a function `f_∂f` which returns both the function value and its derivative at a given point,
# isolate the minimum of the function using a hybrid Brent-Newton method that incorporates derivatives.
# The input points `a`, `b`, and `x0` must be a bracketing triplet of abscissas, i.e. `x0` must lie
# between `a` and `b` and the function value at `x0` must be less than the value at both `a` and `b`.
# The abscissa of the minimum and the minimum function value are returned.
function brent_newton_minimize(f_∂f, a::T, b::T, x0::T, f0::T = f_∂f(x0)[1], df0::T = f_∂f(x0)[2]; xrtol::T = √eps(T), xatol::T = √eps(T), maxiters::Int = 100) where {T <: AbstractFloat}
    @assert a < x0 < b "x0 must lie between a and b"
    tol = xatol + xrtol * max(abs(a), abs(b))

    x_older = x_old = x_new = x0
    f_older = f_old = f_new = f0
    df_older = df_old = df_new = df0
    delta = delta_old = zero(T)

    for iter in 1:maxiters
        x_mid = (a + b) / 2
        tol = xatol + xrtol * abs(x_new)

        if abs(x_new - x_mid) <= (2 * tol - (b - a) / 2)
            break
        end

        if abs(delta_old) > tol
            delta1 = delta2 = 2 * (b - a)
            if df_old != df_new
                delta1 = (x_old - x_new) * df_new / (df_new - df_old)
            end
            if df_older != df_new
                delta2 = (x_older - x_new) * df_new / (df_new - df_older)
            end

            u1 = x_new + delta1
            u2 = x_new + delta2
            accept_delta1 = (a - u1) * (u1 - b) > 0 && df_new * delta1 <= 0
            accept_delta2 = (a - u2) * (u2 - b) > 0 && df_new * delta2 <= 0
            delta_older, delta_old = delta_old, delta

            if accept_delta1 || accept_delta2
                if accept_delta1 && accept_delta2
                    delta = abs(delta1) < abs(delta2) ? delta1 : delta2
                else
                    delta = accept_delta1 ? delta1 : delta2
                end

                if abs(delta) <= abs(delta_older / 2)
                    u = x_new + delta
                    if u - a < 2 * tol || b - u < 2 * tol
                        delta = copysign(tol, x_mid - x_new)
                    end
                else
                    delta_old = df_new >= 0 ? a - x_new : b - x_new
                    delta = delta_old / 2
                end
            else
                delta_old = df_new >= 0 ? a - x_new : b - x_new
                delta = delta_old / 2
            end
        else
            delta_old = df_new >= 0 ? a - x_new : b - x_new
            delta = delta_old / 2
        end

        if abs(delta) >= tol
            u = x_new + delta
            f_u, df_u = f_∂f(u)
        else
            u = x_new + copysign(tol, delta)
            f_u, df_u = f_∂f(u)
            if f_u > f_new
                break
            end
        end

        if f_u <= f_new
            if u >= x_new
                a = x_new
            else
                b = x_new
            end
            x_older, f_older, df_older = x_old, f_old, df_old
            x_old, f_old, df_old = x_new, f_new, df_new
            x_new, f_new, df_new = u, f_u, df_u
        else
            if u < x_new
                a = u
            else
                b = u
            end

            if f_u <= f_old || x_old == x_new
                x_older, f_older, df_older = x_old, f_old, df_old
                x_old, f_old, df_old = u, f_u, df_u
            elseif f_u < f_older || x_older == x_new || x_older == x_old
                x_older, f_older, df_older = u, f_u, df_u
            end
        end
    end

    return (x_new, f_new)
end

#=
function test_brent_newton_minimize(f, a, b; kwargs...)
    x0 = let
        t = range(a, b, 256)
        y = f.(t)
        I = findall(y .< min(y[begin], y[end]))
        rand(t[I])
    end
    a, b = promote(float(a), float(b))
    count = Ref(0)
    function f_∂f(x)
        count[] += 1
        return f(x), ForwardDiff.derivative(f, x)
    end
    for i in 1:2
        count[] = 0
        x, ux = i == 1 ?
                brent_newton_minimize(f_∂f, a, b, x0, f_∂f(x0)...; kwargs...) :
                brent_minimize(first ∘ f_∂f, a, b; kwargs...)
        let
            t = range(a, b; length = 256)
            y = f.(t)
            p = Main.lineplot(t, y; xlim = (a, b), ylim = extrema(y), width = 80, ylabel = "f(x)", title = "ncalls = $(count[]), x = $x, f(u) = $ux")
            Main.scatterplot!(p, [x], [ux]; marker = :diamond, color = :red)
            p |> display
        end
    end
    @info "initial state"
    (; a, x0, b) |> display
    (; fa = f(a), f0 = f(x0), fb = f(b)) |> display
end
=#

function newton_bisect_minimize(f, ∂f_∂²f, x1::T, x2::T; xrtol::T = √eps(T), xatol::T = √eps(T), maxdepth::Int = 5, kwargs...) where {T <: AbstractFloat}
    (x⁻, x⁺, ∂ux⁻, ∂ux⁺, ∂²ux⁻, ∂²ux⁺), succ = bracket_local_minimum(f, ∂f_∂²f, x1, x2; xrtol, xatol, maxdepth)
    if !succ
        return (T(NaN), T(NaN))
    else
        x, dx = newton_bisect_root(∂f_∂²f, (x⁻ + x⁺) / 2, x⁻, x⁺, ∂ux⁻, ∂ux⁺; xrtol, xatol, kwargs...)
        return (x, f(x))
    end
end

#=
function test_newton_bisect_minimize(f, a, b; kwargs...)
    ∂f_∂²f(x) = ForwardDiff.derivative(f, x), only(ForwardDiff.hessian(f ∘ only, SA[x]))
    a, b = promote(float(a), float(b))
    return newton_bisect_minimize(f, ∂f_∂²f, a, b; kwargs...)
end
=#

#### WIP

#=
function bracketing_parabolic_triple(f_∂f, a::T, b::T, ua::T = f_∂f(a)[1], ub::T = f_∂f(b)[1], ∂ua::T = f_∂f(a)[2], ∂ub::T = f_∂f(b)[2]; maxiters::Int = 100) where {T <: AbstractFloat}
    # Use bisection to find triplet of points (a, c, b) such that:
    #   1. a < c < b
    #   2. f(c) < min(f(a), f(b))
    @assert (∂ua < 0 || ∂ub > 0) "Minimum must be bracketed, but f'(x = $a) = $∂ua and f'(x = $b) = $∂ub"
    c = (a + b) / 2
    uc, ∂uc = f_∂f(c)
    for iter in 1:maxiters
        if uc < min(ua, ub)
            # Found midpoint
            # @info "Found midpoint"
            break
        else
            if ∂ua < 0 && ∂ub > 0
                # Both gradients point downhill; contract midpoint toward endpoint with smaller function value
                if ua < ub
                    # @info "Both gradients point downhill: contracting midpoint towards the left"
                    c = (a + c) / 2
                else
                    # @info "Both gradients point downhill: contracting midpoint towards the right"
                    c = (c + b) / 2
                end
            elseif ∂ua < 0
                # Left gradient points downhill; contract midpoint towards the left
                # @info "Left gradient points downhill: contracting midpoint towards the left"
                c = (a + c) / 2
            else
                # Right gradient points downhill; contract midpoint towards the right
                # @info "Right gradient points downhill: contracting midpoint towards the right"
                c = (c + b) / 2
            end
            uc, ∂uc = f_∂f(c)
        end
    end
    return (c, uc, ∂uc)
end

function bracket_merge(((a, ua, ∂ua), (x, ux, ∂ux), (b, ub, ∂ub)), ((c, uc, ∂uc), (y, uy, ∂uy), (d, ud, ∂ud)))
    @assert isnan(x) || a < x < b "x must be between a and b; got a=$a, x=$x, b=$b"
    @assert c < y < d "y must be between c and d; got c=$c, y=$y, d=$d"
    @assert !isnan(y) "y must not be NaN; got y=$y"
    # @info "merging brackets" bracket1=((a, ua, ∂ua), (x, ux, ∂ux), (b, ub, ∂ub)) bracket2=((c, uc, ∂uc), (y, uy, ∂uy), (d, ud, ∂ud))
    if isnan(x)
        if c < a < y && uy < ua < uc
            c, uc, ∂uc = a, ua, ∂ua # LHS: `a` is tighter bracket than `c`
        end
        if y < b < d && uy < ub < ud
            d, ud, ∂ud = b, ub, ∂ub # RHS: `b` is tighter bracket than `d`
        end
        # @info "merged bracket" bracket=((c, uc, ∂uc), (y, uy, ∂uy), (d, ud, ∂ud))
        return ((c, uc, ∂uc), (y, uy, ∂uy), (d, ud, ∂ud))
    elseif uy < ux
        if x < y && ux < uc
            c, uc, ∂uc = x, ux, ∂ux # LHS: `x` is tighter bracket than `c`
        end
        if y < x && ux < ud
            d, ud, ∂ud = x, ux, ∂ux # RHS: `x` is tighter bracket than `d`
        end
        if c < a < y && uy < ua < uc
            c, uc, ∂uc = a, ua, ∂ua # LHS: `a` is tighter bracket than `c`
        end
        if y < b < d && uy < ub < ud
            d, ud, ∂ud = b, ub, ∂ub # RHS: `b` is tighter bracket than `d`
        end
        # @info "merged bracket" bracket=((c, uc, ∂uc), (y, uy, ∂uy), (d, ud, ∂ud))
        return ((c, uc, ∂uc), (y, uy, ∂uy), (d, ud, ∂ud))
    else # ux <= uy
        if y < x && uy < ua
            a, ua, ∂ua = y, uy, ∂uy # LHS: `y` is tighter bracket than `a`
        end
        if x < y && uy < ub
            b, ub, ∂ub = y, uy, ∂uy # RHS: `y` is tighter bracket than `b`
        end
        if a < c < x && ux < uc < ua
            a, ua, ∂ua = c, uc, ∂uc # LHS: `c` is tighter bracket than `a`
        end
        if x < d < b && ux < ud < ub
            b, ub, ∂ub = d, ud, ∂ud # RHS: `d` is tighter bracket than `b`
        end
        # @info "merged bracket" bracket=((a, ua, ∂ua), (x, ux, ∂ux), (b, ub, ∂ub))
        return ((a, ua, ∂ua), (x, ux, ∂ux), (b, ub, ∂ub))
    end
end

function bracket_minimum_bisect_cubic(f_∂f, a::T, b::T, ua::T, ub::T, ∂ua::T, ∂ub::T, depth::Int = 0; xrtol::T = √eps(T), xatol::T = √eps(T), gtol::T = 100 * eps(T), maxdepth::Int = 5) where {T <: AbstractFloat}
    bracket = ((a, ua, ∂ua), (T(NaN), T(NaN), T(NaN)), (b, ub, ∂ub))
    succ = false

    # Evaluate midpoint
    x_mid = (a + b) / 2
    ux_mid, ∂ux_mid = f_∂f(x_mid)
    if ux_mid < min(ua, ub)
        bracket = ((a, ua, ∂ua), (x_mid, ux_mid, ∂ux_mid), (b, ub, ∂ub))
        succ = true
    end

    # Check if interval is sufficiently small
    tol = xatol + xrtol * abs(x_mid)
    if abs((b - a) / 2) <= tol
        return bracket, succ
    end

    if depth < maxdepth
        # If we are less than maximum depth, recursively subdivide interval
        bracket⁻, succ⁻ = bracket_minimum_bisect_cubic(f_∂f, a, x_mid, ua, ux_mid, ∂ua, ∂ux_mid, depth + 1; maxdepth)
        if succ⁻
            bracket = bracket_merge(bracket, bracket⁻)
        end

        bracket⁺, succ⁺ = bracket_minimum_bisect_cubic(f_∂f, x_mid, b, ux_mid, ub, ∂ux_mid, ∂ub, depth + 1; maxdepth)
        if succ⁺
            bracket = bracket_merge(bracket, bracket⁺)
        end
    else
        # At bottom level, try improving guess with cubic spline interpolant
        spl = CubicHermiteInterpolator(a, b, ua, ub, ∂ua, ∂ub)
        x_spl, _ = minimize(spl)
        if a + tol < x_spl < b - tol
            @info "Evaluating spline at bottom: a = $a < x_spl = $x_spl < b = $b"
            ux_spl, ∂ux_spl = f_∂f(x_spl)
            bracket_spl = ((a, ua, ∂ua), (x_spl, ux_spl, ∂ux_spl), (b, ub, ∂ub))
            if succ
                bracket = bracket_merge(bracket, bracket_spl)
            else
                bracket = bracket_spl
            end
        end
    end

    return bracket, !isnan(bracket[2][1]) # succ = !isnan(x)
end

function bracket_minimum_bisect_cubic(f_∂f, a::T, b::T; kwargs...) where {T <: AbstractFloat}
    ua, ∂ua = f_∂f(a)
    ub, ∂ub = f_∂f(b)
    return bracket_minimum_bisect_cubic(f_∂f, a, b, ua, ub, ∂ua, ∂ub; kwargs...)
end

function test_bracket_minimum_bisect_cubic(f, x1, x2; kwargs...)
    x1, x2 = promote(float(x1), float(x2))
    f_∂f = x -> (f(x), ForwardDiff.derivative(f, x))
    x, ux, ∂ux, succ = bracket_minimum_bisect_cubic(f_∂f, x1, x2; kwargs...)
    let
        t = range(x1, x2; length = 256)
        y = f.(t)
        p = Main.lineplot(t, y; xlim = (x1, x2), ylim = extrema(y), width = 80, ylabel = "f(x)", title = "x = $x, f(u) = $ux, f'(u) = $∂ux, success = $succ")
        Main.scatterplot!(p, [x], [ux]; marker = :diamond, color = :red)
        p
    end
end
=#
