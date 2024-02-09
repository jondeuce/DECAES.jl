module NormalHermiteSplinesTests

#=
AUTO-GENERATED FILE - DO NOT EDIT

This file is derived from the following fork of the NormalHermiteSplines.jl package:

    https://github.com/jondeuce/NormalHermiteSplines.jl#eda1e4cf214e291d219509163a54312edfce271a

As it is not possible to depend on a package fork, the above module is included here verbatim.

The `LICENSE.md` file contents from the original repository follows:

################################################################################

MIT License

Copyright (c) 2021 Igor Kohanovsky

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#

using Test
using ..DECAES.NormalHermiteSplines

using DoubleFloats
using LinearAlgebra
using Random
using StaticArrays

@testset "NormalHermiteSplines.jl" begin
    ####
    #### 1D.jl
    ####

    @testset "Test 1D" begin
        x = [0.0, 1.0, 2.0]                        # function nodes
        u = x .^ 2                                   # function values in nodes
        s = [2.0]                                  # function first derivative nodes
        v = [4.0]                                  # function first derivative values

        @testset "Test 1D-RK_H0 kernel" begin
            spl = interpolate(x, u, RK_H0(0.1))  # create spline
            cond = estimate_cond(spl)                 # get estimation of the Gram matrix condition number
            @test cond ≈ 100.0
            σ = evaluate(spl, x)                # evaluate spline in nodes
            @test σ ≈ u                         # compare with exact function values in nodes

            spl = prepare(x, RK_H0(0.1))
            spl = construct(spl, u)
            σ = evaluate(spl, x)
            @test σ ≈ u

            # Check that we get close when evaluating near the nodes
            p = x .+ 1e-4 * randn(size(x))   # evaluation points near the nodes
            f = p .^ 2                       # exact function values in evaluation points
            σ = evaluate(spl, p)           # evaluate spline in evaluation points
            # compare spline values with exact function values in evaluation point
            @test all(isapprox.(σ, f, atol = 0.05))

            σ = evaluate_one(spl, p[3])
            @test σ ≈ f[3] atol = 0.05
        end

        @testset "Test 1D-RK_H1 kernel" begin
            spl = interpolate(x, u, RK_H1(0.1))  # create spline
            cond = estimate_cond(spl)              # get estimation of the gram matrix condition number
            @test cond ≈ 1.0e5
            σ = evaluate(spl, x)           # evaluate spline in nodes
            @test σ ≈ u                    # compare with exact function values in nodes

            d1_σ = evaluate_derivative(spl, 1.0)     # evaluate spline first derivative in the node
            @test d1_σ ≈ 2.0 atol = 0.05              # compare with exact function first derivative value in the node

            # Check that we get close when evaluating near the nodes
            p = x .+ 1e-4 * randn(size(x))   # evaluation points near the nodes
            f = p .^ 2                       # exact function values in evaluation points
            σ = evaluate(spl, p)                # evaluate spline in evaluation points
            # compare spline values with exact function values in evaluation point
            @test all(isapprox.(σ, f, atol = 1e-2))

            ###
            spl = interpolate(x, u, s, v, RK_H1(0.1)) # create spline by function and
            # first derivative values in nodes
            cond = estimate_cond(spl)              # get estimation of the gram matrix condition number
            @test cond ≈ 1.0e5
            σ = evaluate(spl, x)                # evaluate spline in nodes
            @test σ ≈ u                    # compare with exact function values in nodes

            # Check that we get close when evaluating near the nodes
            p = x .+ 1e-3 * randn(size(x))   # evaluation points near the nodes
            f = p .^ 2                       # exact function values in evaluation points
            σ = evaluate(spl, p)                # evaluate spline in evaluation points
            # compare spline values with exact function values in evaluation point
            @test all(isapprox.(σ, f, atol = 1e-2))

            spl = prepare(x, s, RK_H1(0.1))
            spl = construct(spl, u, v)
            σ = evaluate_one(spl, p[3])
            @test σ ≈ f[3] atol = 0.05
        end

        @testset "Test 1D-RK_H2 kernel" begin
            spl = interpolate(x, u, RK_H2(0.1))  # create spline
            cond = estimate_cond(spl)              # get estimation of the gram matrix condition number
            @test cond ≈ 1.0e7
            σ = evaluate(spl, x)           # evaluate spline in nodes
            @test σ ≈ u                    # compare with exact function values in nodes

            d1_σ = evaluate_derivative(spl, 1.0)     # evaluate spline first derivative in the node
            @test d1_σ ≈ 2.0 atol = 0.005              # compare with exact function first derivative value in the node

            # Check that we get close when evaluating near the nodes
            p = x .+ 1e-4 * randn(size(x))   # evaluation points near the nodes
            f = p .^ 2                       # exact function values in evaluation points
            σ = evaluate(spl, p)                # evaluate spline in evaluation points
            # compare spline values with exact function values in evaluation point
            @test all(isapprox.(σ, f, atol = 1e-2))

            ###
            spl = interpolate(x, u, s, v, RK_H2(0.1)) # create spline by function and
            # first derivative values in nodes
            σ = evaluate(spl, x)                # evaluate spline in nodes
            @test σ ≈ u                    # compare with exact function values in nodes

            # Check that we get close when evaluating near the nodes
            p = x .+ 1e-3 * randn(size(x))   # evaluation points near the nodes
            f = p .^ 2                       # exact function values in evaluation points
            σ = evaluate(spl, p)                # evaluate spline in evaluation points
            # compare spline values with exact function values in evaluation point
            @test all(isapprox.(σ, f, atol = 1e-2))
        end
    end

    ####
    #### 2D.jl
    ####

    @testset "Test 2D" begin
        p = [0.0 0.0 1.0 1.0 0.5; 0.0 1.0 0.0 1.0 0.5] # nodes
        u = [0.0; 0.0; 0.0; 0.0; 1.0] # function values in nodes
        u2 = [0.0; 0.0; 0.0; 0.0; 2.0] # second function values in nodes
        t = [0.5 0.5 0.499999; 0.5 0.499999 0.5] # evaluation points

        dp = [0.5 0.5; 0.5 0.5]
        es = [1.0 0.0; 0.0 1.0]
        du = [0.0; 100000.0]

        @testset "Test 2D-RK_H0 kernel" begin
            rk = RK_H0(0.001)
            s = interpolate(p, u, rk)
            σ = evaluate(s, t)
            @test σ[1] ≈ u[5]
            @test isapprox(σ[2], u[5], atol = 1e-5)
            @test isapprox(σ[3], u[5], atol = 1e-5)

            σ1 = evaluate_one(s, t[:, 1])
            @test σ1[1] ≈ u[5]

            rk = RK_H0()
            s = prepare(p, rk) # prepare spline
            s = construct(s, u) # construct spline
            σ1 = evaluate(s, t) # evaluate spline in points
            @test σ[1] ≈ u[5]
            @test isapprox(σ[2], u[5], atol = 1e-5)
            @test isapprox(σ[3], u[5], atol = 1e-5)

            s = construct(s, u2)
            σ2 = evaluate(s, t)
            @test σ2[1] ≈ u2[5]
            @test isapprox(σ2[2], u2[5], atol = 1e-5)
            @test isapprox(σ2[3], u2[5], atol = 1e-5)

            cond = estimate_cond(s)
            @test cond ≈ 10.0

            eps = get_epsilon(s)
            @test isapprox(eps, 0.93, atol = 1e-2)

            est_eps = estimate_epsilon(p)
            @test isapprox(est_eps, 0.93, atol = 1e-2)
        end

        @testset "Test 2D-RK_H1 kernel" begin
            rk = RK_H1(0.001)
            s = interpolate(p, u, rk)
            cond = estimate_cond(s)
            @test cond ≈ 1.e11

            σ = evaluate(s, t)
            @test isapprox(σ[1], u[5], atol = 1e-5)
            @test isapprox(σ[2], u[5], atol = 1e-5)
            @test isapprox(σ[3], u[5], atol = 1e-5)

            grad = evaluate_gradient(s, p[:, 5])
            @test abs(grad[1]) < 1.0e-5 && abs(grad[2]) < 1.0e-5

            rk = RK_H1()
            s = prepare(p, rk) # prepare spline
            cond = estimate_cond(s)
            @test cond ≈ 100.0

            s = construct(s, u) # construct spline

            σ1 = evaluate(s, t) # evaluate spline in points
            @test isapprox(σ[1], u[5], atol = 1e-5)
            @test isapprox(σ[2], u[5], atol = 1e-5)
            @test isapprox(σ[3], u[5], atol = 1e-5)

            s = construct(s, u2)
            σ2 = evaluate(s, t)
            @test isapprox(σ[1], u[5], atol = 1e-5)
            @test isapprox(σ2[2], u2[5], atol = 1e-5)
            @test isapprox(σ2[3], u2[5], atol = 1e-5)

            est_eps = estimate_epsilon(p, dp)
            @test est_eps ≈ 1.94 atol = 1e-2
            est_eps = estimate_epsilon(p, dp, RK_H1())
            @test est_eps ≈ 1.94 atol = 1e-2
            eps = get_epsilon(s)
            @test est_eps ≈ 1.94 atol = 1e-2

            eps = 0.0001
            rk = RK_H1(eps)
            s = interpolate(p, u, dp, es, du, rk)
            cond = estimate_cond(s)
            @test cond == 1.0e14

            σ1 = evaluate_one(s, p[:, 5])
            @test !isapprox(σ1[1], u[5]; atol = 0.1)

            # Same test with extended precision
            rk = RK_H1(Double64(eps))
            p = Double64.(p)
            dp = Double64.(dp)
            es = Double64.(es)
            u = Double64.(u)
            du = Double64.(du)
            t = Double64.(t)
            s = prepare(p, dp, es, rk)
            s = construct(s, u, du)
            σ = evaluate(s, p)
            @test all(isapprox.(σ, u, atol = 1e-5))

            q = estimate_accuracy(s)
            @test q ≈ 14
        end

        @testset "Test 2D-RK_H2 kernel" begin
            p = [0.0 0.0 1.0 1.0 0.5; 0.0 1.0 0.0 1.0 0.5] # nodes
            u = [0.0; 0.0; 0.0; 0.0; 1.0] # function values in nodes
            u2 = [0.0; 0.0; 0.0; 0.0; 2.0] # second function values in nodes
            t = [0.5 0.5 0.499999; 0.5 0.499999 0.5] # evaluation points

            dp = [0.5 0.5; 0.5 0.5]
            es = [1.0 0.0; 0.0 1.0]
            du = [0.0; 100000.0]

            rk = RK_H2(0.01)
            s = interpolate(p, u, rk)
            cond = estimate_cond(s)
            @test cond ≈ 1e11

            σ = evaluate(s, t)
            @test isapprox(σ[1], u[5], atol = 1e-5)
            @test isapprox(σ[2], u[5], atol = 1e-5)
            @test isapprox(σ[3], u[5], atol = 1e-5)

            grad = evaluate_gradient(s, p[:, 5])
            @test abs(grad[1]) < 1.0e-5 && abs(grad[2]) < 1.0e-5

            rk = RK_H2()
            s = prepare(p, rk) # prepare spline
            cond = estimate_cond(s)
            @test cond ≈ 100.0

            s = construct(s, u) # construct spline

            σ1 = evaluate(s, t) # evaluate spline in points
            @test isapprox(σ[1], u[5], atol = 1e-5)
            @test isapprox(σ[2], u[5], atol = 1e-5)
            @test isapprox(σ[3], u[5], atol = 1e-5)

            s = construct(s, u2)
            σ2 = evaluate(s, t)
            @test isapprox(σ[1], u[5], atol = 1e-5)
            @test isapprox(σ2[2], u2[5], atol = 1e-5)
            @test isapprox(σ2[3], u2[5], atol = 1e-5)

            eps = 0.01
            rk = RK_H2(eps)
            s = interpolate(p, u, dp, es, du, rk)
            cond = estimate_cond(s)
            @test cond == 1.0e13

            σ1 = evaluate_one(s, p[:, 5])
            @test !isapprox(σ1[1], u[5]; atol = 0.1)

            q = estimate_accuracy(s)
            @test q == 0

            # Same tests with extended precision
            rk = RK_H2(Double64(eps))
            p = Double64.(p)
            dp = Double64.(dp)
            es = Double64.(es)
            u = Double64.(u)
            du = Double64.(du)
            t = Double64.(t)
            s = prepare(p, dp, es, rk)
            s = construct(s, u, du)

            σ = evaluate(s, p)
            @test all(isapprox.(σ, u, atol = 1e-5))

            q = estimate_accuracy(s)
            @test q ≈ 15
        end
    end

    @testset "Test 2D-Bis" begin
        p = collect([-1.0 2.0; -1.0 4.0; 3.0 2.0; 3.0 4.0; 1.0 3.0]') # function nodes
        u = [0.0; 0.0; 0.0; 0.0; 1.0] # function values in nodes
        t = collect([-1.0 3.0; 0.0 3.0; 1.0 3.0; 2.0 3.0; 3.0 3.0]') # evaluation points

        @testset "Test 2D-Bis-RK_H0 kernel" begin
            spl = prepare(p, RK_H0(0.001)) # prepare spline
            c = estimate_cond(spl) # get estimation of the problem's Gram matrix condition number
            @test c ≈ 100000.0

            spl = construct(spl, u) # construct spline
            vt = [1.0, 3.0]
            σ = evaluate_one(spl, vt) # evaluate spline in the node
            @test σ ≈ 1.0

            wt = [0.0, 3.0]
            σ1 = evaluate_one(spl, wt)

            u2 = [0.0; 0.0; 0.0; 0.0; 2.0]
            spl = construct(spl, u2)
            σ2 = evaluate_one(spl, wt)
            @test σ2 ≈ 2.0 * σ1

            spl = interpolate(p, u, RK_H0(0.001)) # prepare and construct spline
            σ = evaluate_one(spl, vt)
            @test σ ≈ 1.0
        end

        @testset "Test 2D-Bis-RK_H1 kernel" begin
            spl = prepare(p, RK_H1(0.001))
            c = estimate_cond(spl)
            @test c ≈ 1.0e11

            spl = construct(spl, u)
            vt = [1.0, 3.0]
            σ = evaluate_one(spl, vt)
            @test σ ≈ 1.0

            wt = [0.0, 3.0]
            σ1 = evaluate_one(spl, wt)

            u2 = [0.0; 0.0; 0.0; 0.0; 2.0]
            spl = construct(spl, u2)
            σ2 = evaluate_one(spl, wt)
            @test σ2 ≈ 2.0 * σ1

            spl = interpolate(p, u, RK_H1(0.001))
            σ = evaluate_one(spl, vt)
            @test σ ≈ 1.0
        end

        @testset "Test 2D-Bis-RK_H2 kernel" begin
            spl = prepare(p, RK_H2(0.001))
            c = estimate_cond(spl)
            @test c ≈ 1.0e15

            spl = construct(spl, u)

            vt = [1.0, 3.0]
            σ = evaluate_one(spl, vt)
            @test σ ≈ 1.0

            wt = [0.0, 3.0]
            σ1 = evaluate_one(spl, wt)
            u2 = [0.0; 0.0; 0.0; 0.0; 2.0]
            spl = construct(spl, u2)
            σ2 = evaluate_one(spl, wt)
            @test σ2 ≈ 2.0 * σ1

            spl = interpolate(p, u, RK_H2(0.001))
            σ = evaluate_one(spl, vt)
            @test σ ≈ 1.0
        end
    end

    @testset "Test 2D-Grad" begin
        p = collect([0.0 0.0; 1.0 0.0; 0.0 1.0]') # function nodes
        u = [0.0; 0.0; 1.0] # function values in nodes
        t = [0.5; 0.5] # evaluation points

        p2 = collect([0.0 0.0; 2.0 0.0; 0.0 2.0]') # function nodes
        u2 = [0.0; 0.0; 2.0] # function values in nodes
        t2 = [1.0; 1.0] # evaluation points

        @testset "Test 2D-Grad-RK_H1 kernel" begin
            spl = interpolate(p, u, RK_H1(0.001))
            grad = evaluate_gradient(spl, t)
            @test abs(grad[1] + 1.0) ≈ 1.0 atol = 1e-2
            @test abs(grad[2]) ≈ 1.0 atol = 1e-2

            spl = interpolate(p2, u2, RK_H1(0.001))
            grad = evaluate_gradient(spl, t2)
            @test abs(grad[1] + 1.0) ≈ 1.0 atol = 1e-2
            @test abs(grad[2]) ≈ 1.0 atol = 1e-2
        end

        @testset "Test 2D-Grad-RK_H2 kernel" begin
            spl = interpolate(p, u, RK_H2(0.001))
            grad = evaluate_gradient(spl, t)
            @test abs(grad[1] + 1.0) ≈ 1.0 atol = 1e-4
            @test abs(grad[2]) ≈ 1.0 atol = 1e-4

            spl = interpolate(p2, u2, RK_H2(0.001))
            grad = evaluate_gradient(spl, t2)
            @test abs(grad[1] + 1.0) ≈ 1.0 atol = 1e-4
            @test abs(grad[2]) ≈ 1.0 atol = 1e-4
        end
    end

    ####
    #### 3D.jl
    ####

    @testset "Test 3D" begin

        # function get_3D_model(p::Vector{Float64})
        #     r = p[1] + p[2] + p[3]
        #     return r
        # end
        #
        # function get_3D_model_grad()
        #     return [1.0; 1.0; 1.0]
        # end

        p = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0] # function nodes
        u = [1.0; 1.0; 1.0; 0.0] # function values in nodes

        n_1 = size(p, 2)
        dp = Matrix{Float64}(undef, 3, 3 * n_1)
        es = Matrix{Float64}(undef, 3, 3 * n_1)
        du = Vector{Float64}(undef, 3 * n_1)
        grad = [1.0; 1.0; 1.0]

        k = 0
        for i in 1:n_1
            k += 1
            dp[1, k] = p[1, i]
            dp[2, k] = p[2, i]
            dp[3, k] = p[3, i]
            du[k] = grad[1]
            es[1, k] = 1.0
            es[2, k] = 0.0
            es[3, k] = 0.0
            k += 1
            dp[1, k] = p[1, i]
            dp[2, k] = p[2, i]
            dp[3, k] = p[3, i]
            du[k] = grad[2]
            es[1, k] = 0.0
            es[2, k] = 1.0
            es[3, k] = 0.0
            k += 1
            dp[1, k] = p[1, i]
            dp[2, k] = p[2, i]
            dp[3, k] = p[3, i]
            du[k] = grad[3]
            es[1, k] = 0.0
            es[2, k] = 0.0
            es[3, k] = 1.0
        end

        @testset "Test 3D-RK_H0 kernel" begin
            s = interpolate(p, u, RK_H0(0.00001))
            σ1 = evaluate_one(s, [1.0; 0.0; 0.01])
            @test isapprox(σ1, u[1], atol = 1e-3)
            σ = evaluate(s, [1.0 0.5; 0.0 0.5; 0.01 0.5])
            @test isapprox(σ[1], u[1], atol = 1e-3)

            s = prepare(p, RK_H0(0.00001)) # prepare spline
            s = construct(s, u) # construct spline
            σ1 = evaluate_one(s, [1.0; 0.0; 0.01])
            @test isapprox(σ1, u[1], atol = 1e-3)
            σ = evaluate(s, [1.0 0.5; 0.0 0.5; 0.01 0.5])
            @test isapprox(σ[1], u[1], atol = 1e-3)

            cond = estimate_cond(s)
            @test cond ≈ 1.0e6

            iq = estimate_accuracy(s)
            @test iq ≈ 15.0

            eps = get_epsilon(s)
            @test eps ≈ 1.0e-5

            est_eps = estimate_epsilon(p)
            @test isapprox(est_eps, 1.036, atol = 1e-3)
        end

        @testset "Test 3D-RK_H1 kernel" begin
            s = interpolate(p, u, dp, es, du, RK_H1(0.1))
            σ1 = evaluate_one(s, [1.0; 0.0; 0.01])
            @test isapprox(σ1, u[1], atol = 1e-2)
            σ = evaluate(s, [1.0 0.5; 0.0 0.5; 0.01 0.5])
            @test isapprox(σ[1], u[1], atol = 1e-2)

            s = prepare(p, dp, es, RK_H1(0.1)) # prepare spline
            s = construct(s, u, du) # construct spline
            σ1 = evaluate_one(s, [1.0; 0.0; 0.01])
            @test isapprox(σ1, u[1], atol = 1e-2)
            σ = evaluate(s, [1.0 0.5; 0.0 0.5; 0.01 0.5])
            @test isapprox(σ[1], u[1], atol = 1e-2)

            g = evaluate_gradient(s, [0.01; 0.2; 0.3])
            @test all(isapprox.(g, grad, atol = 0.01))

            cond = estimate_cond(s)
            @test cond ≈ 1.0e5

            iq = estimate_accuracy(s) + 1.0
            @test iq ≈ 14.0 atol = 1.0

            eps = get_epsilon(s)
            @test eps ≈ 0.1

            est_eps = estimate_epsilon(p)
            @test isapprox(est_eps, 1.036, atol = 1e-3)

            est_eps = estimate_epsilon(p, dp)
            @test isapprox(est_eps, 1.415, atol = 1e-3)
        end

        @testset "Test 3D-RK_H2 kernel" begin
            s = interpolate(p, u, dp, es, du, RK_H2(0.1))
            σ1 = evaluate_one(s, [1.0; 0.0; 0.01])
            @test isapprox(σ1, u[1], atol = 2e-2)
            σ = evaluate(s, [1.0 0.5; 0.0 0.5; 0.01 0.5])
            @test isapprox(σ[1], u[1], atol = 2e-2)

            s = prepare(p, dp, es, RK_H2(0.1)) # prepare spline
            s = construct(s, u, du) # construct spline
            σ1 = evaluate_one(s, [1.0; 0.0; 0.01])
            @test isapprox(σ1, u[1], atol = 2e-2)
            σ = evaluate(s, [1.0 0.5; 0.0 0.5; 0.01 0.5])
            @test isapprox(σ[1], u[1], atol = 2e-2)

            g = evaluate_gradient(s, [0.01; 0.2; 0.3])
            @test all(isapprox.(g, grad, atol = 0.01))

            cond = estimate_cond(s)
            @test cond ≈ 1.0e8

            iq = estimate_accuracy(s) + 1.0
            @test iq ≈ 13.0 atol = 1.0

            eps = get_epsilon(s)
            @test eps ≈ 0.1

            est_eps = estimate_epsilon(p)
            @test isapprox(est_eps, 1.036, atol = 1e-3)

            est_eps = estimate_epsilon(p, dp)
            @test isapprox(est_eps, 1.415, atol = 1e-3)
        end
    end

    ####
    #### elastic.jl
    ####

    @testset "Elastic" begin
        function random_nodes(n, T, max_size)
            min_bound   = -rand(SVector{n, T})
            max_bound   = rand(SVector{n, T})
            rand_node() = min_bound .+ rand(SVector{n, T}) .* (max_bound .- min_bound)
            rand_dir()  = (x = rand_node(); x / norm(x))

            nodes    = [min_bound, max_bound, (rand_node() for i in 3:max_size)...]
            values   = rand(T, max_size)
            d_nodes  = [rand_node() for _ in 1:n*max_size]
            d_dirs   = [rand_dir() for _ in 1:n*max_size]
            d_values = rand(T, n * max_size)
            return (; min_bound, max_bound, nodes, values, d_nodes, d_dirs, d_values)
        end

        @testset "Elastic Cholesky" begin
            T = Float64
            max_size = 4
            A = rand(MersenneTwister(0), max_size, max_size)
            A = A'A

            # Test incrementally adding columns of `A` to `ElasticCholesky`
            for colperms in [[1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4], [4, 1, 3, 2]]
                C_copy_into_A = ElasticCholesky{T}(max_size)
                C_wrap_A = ElasticCholesky(A)
                for j in 1:max_size
                    # Add (permuted) column `colperms[j]` of `A` to the `colperms[j]`th column of `C.A`
                    cholesky!(C_copy_into_A, colperms[j], A[colperms[1:j], colperms[j]], Val(true))
                    cholesky!(C_wrap_A, colperms[j])

                    for C in [C_copy_into_A, C_wrap_A]
                        # Check the fields of `C`
                        J = colperms[1:C.ncols[]]
                        @test Hermitian(C.A[J, J], :U) ≈ A[J, J]
                        @test UpperTriangular(C.U[J, J]) ≈ cholesky(A[J, J]).U
                        @test C.colperms[1:C.ncols[]] == J
                        @test C.ncols[] == j

                        # Check `ldiv!` gives same result as `LinearAlgebra.cholesky`
                        b = rand(j)
                        @test ldiv!(similar(b), C, b) ≈ cholesky(A[J, J]) \ b
                    end
                end
            end

            # Test incremental factorization of wrapped array
            C = ElasticCholesky(copy(A))
            cholesky!(C)

            @test Hermitian(C.A, :U) ≈ A
            @test UpperTriangular(C.U) ≈ cholesky(A).U
            @test C.colperms == 1:max_size
            @test C.ncols[] == max_size

            b = rand(max_size)
            @test ldiv!(similar(b), C, b) ≈ cholesky(A) \ b
        end

        @testset "Elastic Gram Matrix" begin
            nhs = NormalHermiteSplines
            max_size = 4
            T = Float64

            @testset "RK_H0" begin
                kernel = RK_H0(0.5 + rand())
                for n in 1:3
                    # Test incremental building of Gram matrix, inserting nodes in random order
                    (; nodes) = random_nodes(n, T, max_size)
                    n₁ = length(nodes)
                    A = nhs._gram(nodes, kernel)
                    A′ = zeros(T, n₁, n₁)
                    J = Int[]
                    for j in randperm(n₁)
                        push!(J, j)
                        nhs._gram!(view(A′, J, J), nodes[J[end]], nodes[J[1:end-1]], kernel)
                        @test Hermitian(A′[J, J], :U) ≈ A[J, J] # `A′[J,J]` is upper triangular, `A` is Hermitian
                    end
                end
            end

            @testset "RK_H1" begin
                kernel = RK_H1(0.5 + rand())
                for n in 1:3
                    # Test incremental building of Gram matrix, inserting (derivative-)nodes in random order
                    (; nodes, d_nodes, d_dirs) = random_nodes(n, T, max_size)
                    n₁, n₂ = length(nodes), length(d_nodes)
                    A = nhs._gram(nodes, d_nodes, d_dirs, kernel)
                    A′ = zeros(T, n₁ + n₂, n₁ + n₂)
                    J, J₁, J₂ = Int[], Int[], Int[]
                    for j in randperm(n₁ + n₂)
                        if j <= n₁
                            push!(J₁, j)
                            insert!(J, length(J₁), j)
                            nhs._gram!(view(A′, J, J), nodes[J₁[end]], nodes[J₁[1:end-1]], d_nodes[J₂], d_dirs[J₂], kernel)
                        else
                            push!(J₂, j - n₁)
                            push!(J, j)
                            nhs._gram!(view(A′, J, J), d_nodes[J₂[end]], d_dirs[J₂[end]], nodes[J₁], d_nodes[J₂[1:end-1]], d_dirs[J₂[1:end-1]], kernel)
                        end
                        @test Hermitian(A′[J, J], :U) ≈ A[J, J] # `A′[J,J]` is upper triangular, `A` is Hermitian
                    end
                end
            end
        end

        @testset "Elastic Normal Spline" begin
            nhs = NormalHermiteSplines
            max_size = 3
            T = Float64

            for n in 1:3
                (; min_bound, max_bound, nodes, values, d_nodes, d_dirs, d_values) = random_nodes(n, T, max_size)
                rk_H0 = RK_H0(0.5 + rand())
                rk_H1 = RK_H1(0.5 + rand())
                espl_H0 = ElasticNormalSpline(min_bound, max_bound, max_size, rk_H0)
                espl_H1_0 = ElasticNormalSpline(min_bound, max_bound, max_size, rk_H1)
                espl_H1_1 = ElasticNormalSpline(min_bound, max_bound, max_size, rk_H1)

                for i in 1:max_size
                    # Update `ElasticNormalSpline`
                    nodes′, values′ = nodes[1:i], values[1:i]
                    d_nodes′, d_dirs′, d_values′ = d_nodes[1:n*i], d_dirs[1:n*i], d_values[1:n*i]

                    # Insert regular node
                    insert!(espl_H0, nodes′[i], values′[i])
                    insert!(espl_H1_0, nodes′[i], values′[i])

                    # Insert `n` derivative nodes
                    insert!(espl_H1_1, nodes′[i], values′[i])
                    for k in n * (i - 1) .+ (1:n)
                        insert!(espl_H1_1, d_nodes′[k], d_dirs′[k], d_values′[k])
                    end

                    # Compute `NormalSpline`
                    i == 1 && continue # `NormalSpline` requires at least two nodes′
                    spl_H0 = interpolate(nodes′, values′, rk_H0)
                    spl_H1_0 = interpolate(nodes′, values′, rk_H1)
                    spl_H1_1 = interpolate(nodes′, values′, d_nodes′, d_dirs′, d_values′, rk_H1)

                    for (espl, spl) in [
                        (espl_H0, spl_H0)
                        (espl_H1_0, spl_H1_0)
                        (espl_H1_1, spl_H1_1)
                    ]
                        C = espl._chol
                        n₁ = espl._num_nodes[]
                        J = C.colperms[1:C.ncols[]]

                        # Cholesky factorization of `ElasticNormalSpline` is built incrementally in arbitrary column order of the underlying Gram matrix;
                        # compare with the Cholesky factorization of the gram matrix from `NormalSpline`, permuted appropriately
                        J′ = (j -> ifelse(j > max_size, j - max_size + n₁, j)).(J)
                        C′ = cholesky(nhs._get_gram(spl)[J′, J′])
                        b = randn(C.ncols[])

                        @test nhs._get_kernel(espl) == nhs._get_kernel(spl)
                        @test nhs._get_nodes(espl) ≈ nhs._get_nodes(spl)
                        @test nhs._get_values(espl) ≈ nhs._get_values(spl)
                        @test nhs._get_d_nodes(espl) ≈ nhs._get_d_nodes(spl)
                        @test nhs._get_d_dirs(espl) ≈ nhs._get_d_dirs(spl)
                        @test nhs._get_d_values(espl) ≈ nhs._get_d_values(spl)
                        @test nhs._get_mu(espl) ≈ nhs._get_mu(spl)
                        @test nhs._get_gram(espl) ≈ nhs._get_gram(spl)
                        @test UpperTriangular(C.U[J, J]) ≈ C′.U
                        @test ldiv!(similar(b), C, b) ≈ C′ \ b
                        # @test nhs._get_cond(espl)      ≈ nhs._get_cond(spl)
                        @test nhs._get_min_bound(espl) ≈ nhs._get_min_bound(spl)
                        @test nhs._get_max_bound(espl) ≈ nhs._get_max_bound(spl)
                        @test nhs._get_scale(espl) ≈ nhs._get_scale(spl)
                    end
                end
            end
        end
    end
end

end # module NormalHermiteSplinesTests
