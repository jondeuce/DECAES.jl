####
#### Rootfinding methods
####

#=
`secant_method` and `bisection_method` are modified codes from Roots.jl:
    https://github.com/JuliaMath/Roots.jl/blob/8a5ff76e8e8305d4ad5719fe1dd665d8a7bd7ec3/src/simple.jl

The MIT License (MIT) Copyright (c) 2013 John C. Travers
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#
function secant_method(f, xs; xatol = zero(float(real(first(xs)))), xrtol = 8 * eps(one(float(real(first(xs))))), maxiters = 100)
    if length(xs) == 1 # secant needs a, b; only a given
        a  = float(xs[1])
        h  = cbrt(eps(one(real(a))))
        da = h * oneunit(a) + abs(a) * h^2 # adjust for if eps(a) > h
        b  = a + da
    else
        a, b = promote(float(xs[1]), float(xs[2]))
    end
    return secant(f, a, b, f(a), f(b); xatol, xrtol, maxiters)
end

function secant(f, a::T, b::T, fa::T, fb::T; xatol = zero(T), xrtol = 8 * eps(T), maxiters = 100) where {T}
    # No function change; return arbitrary endpoint
    if fb == fa
        return (a, fa)
    end

    cnt = 0
    mbest = abs(fa) < abs(fb) ? a : b
    fbest = min(abs(fa), abs(fb))
    uatol = xatol / oneunit(xatol) * oneunit(real(a))
    adjustunit = oneunit(real(fb)) / oneunit(real(b))

    while cnt < maxiters
        m = b - (b - a) * fb / (fb - fa)
        fm = f(m)

        abs(fm) < abs(fbest) && ((mbest, fbest) = (m, fm))
        iszero(fm) && return (m, fm)
        !isfinite(fm) && return (mbest, fbest) # function failed; bail out
        abs(fm) <= adjustunit * max(uatol, abs(m) * xrtol) && return (m, fm)
        fm == fb && return (m, fm)

        a, b, fa, fb = b, m, fb, fm
        cnt += 1
    end

    return (mbest, fbest) # maxiters reached
end

function bisection_method(f, a::Number, b::Number; xatol = nothing, xrtol = nothing, ftol = nothing, maxiters = 100)
    T = promote_type(typeof(float(a)), typeof(float(b)))
    x₁, x₂ = T(a), T(b)
    y₁, y₂ = f(x₁), f(x₂)

    xatol = xatol === nothing ? zero(T) : T(xatol)
    xrtol = xrtol === nothing ? zero(T) : T(xrtol)
    ftol = ftol === nothing ? zero(T) : T(ftol)

    return bisect(f, x₁, x₂, y₁, y₂; xatol, xrtol, ftol, maxiters)[3:4]
end

function bisect(f, x₁::T, x₂::T, y₁::T, y₂::T; xatol = zero(T), xrtol = zero(T), ftol = zero(T), maxiters = 100) where {T}
    if x₁ == x₂
        xₘ, yₘ = x₁, y₁
        return (x₁, y₁, xₘ, yₘ, x₂, y₂)
    elseif y₁ * y₂ >= 0
        # No sign change; return endpoint closest to zero
        xₘ, yₘ = abs(y1) <= abs(y2) ? (x₁, y₁) : (x₂, y₂)
        return (x₁, y₁, xₘ, yₘ, x₂, y₂)
    end

    if y₂ < 0
        x₁, x₂, y₁, y₂ = x₂, x₁, y₂, y₁
    end

    xₘ = (x₁ + x₂) / 2
    yₘ = f(xₘ)

    cnt = 1
    while cnt < maxiters
        if !isfinite(yₘ) || abs(yₘ) <= ftol || 2 * min(abs(xₘ - x₁), abs(x₂ - xₘ)) <= xatol + max(abs(x₁), abs(x₂)) * xrtol
            return (x₁, y₁, xₘ, yₘ, x₂, y₂)
        end

        if yₘ < 0
            x₁, y₁ = xₘ, yₘ
        else
            x₂, y₂ = xₘ, yₘ
        end

        xₘ = (x₁ + x₂) / 2
        yₘ = f(xₘ)

        cnt += 1
    end

    return (x₁, y₁, xₘ, yₘ, x₂, y₂)
end

function bracketing_interval_monotonic(f, a, δ; dilate = 1, mono = +1, maxiters = 100)
    fa = f(a)
    fa == 0 && return (a, a, fa, fa)
    sgn_δ = mono * sign(fa)
    b = a - sgn_δ * δ
    fb = f(b)
    fb == 0 && return (b, b, fb, fb)
    δ *= dilate
    cnt = 0
    while fa * fb > 0 && cnt < maxiters
        a, fa = b, fb
        b = a - sgn_δ * δ
        fb = f(b)
        fb == 0 && return (b, b, fb, fb)
        δ *= dilate
        cnt += 1
    end
    return a < b ? (a, b, fa, fb) : (b, a, fb, fa)
end

####
#### Optimization methods
####

#=
Brent-Dekker minimization method. The code for `brents_method` is modified from Optim.jl:
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
function brents_method(f, x₁::T, x₂::T; xrtol = sqrt(eps(T)), xatol = eps(T), maxiters::Int = 1_000) where {T <: AbstractFloat}
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

        Δx_tol = xrtol * abs(x) + xatol
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
