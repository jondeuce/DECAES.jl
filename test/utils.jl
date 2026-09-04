Random.seed!(0) # reproducible randomized tests

@testset "GrowableCache" begin
    c = GrowableCache{Float64, Float64}()
    push!(c, (1.0, 2.0))
    @test c.keys == [1.0] && c.values == [2.0]
    push!(c, (3.0, 4.0))
    @test c.keys == [1.0, 3.0] && c.values == [2.0, 4.0]

    c = GrowableCache{Float64, Float64}()
    pushfirst!(c, (1.0, 2.0))
    @test c.keys == [1.0] && c.values == [2.0]
    pushfirst!(c, (3.0, 4.0))
    @test c.keys == [3.0, 1.0] && c.values == [4.0, 2.0]
    pushfirst!(c, (5.0, 6.0))
    @test c.keys == [5.0, 3.0, 1.0] && c.values == [6.0, 4.0, 2.0]

    for (i, (k, v)) in enumerate(c)
        @test k == [5.0, 3.0, 1.0][i]
        @test v == [6.0, 4.0, 2.0][i]
    end

    c = GrowableCache{Float64, Float64}()
    push!(c, (1.0, 2.0))
    push!(c, (3.0, 4.0))
    push!(c, (5.0, 6.0))
    @test popfirst!(c) == (1.0, 2.0)
    @test length(c) == 2 && collect(keys(c)) == [3.0, 5.0]
    @test popfirst!(c) == (3.0, 4.0)
    push!(c, (7.0, 8.0))
    @test collect(keys(c)) == [5.0, 7.0] # entries pushed after a removal queue behind the survivors

    c = GrowableCache{Float64, Float64}()
    c[1.5] = 2.0
    @test c[1.5] == 2.0
    c[1.5] = 3.0
    @test c[1.5] == 3.0
end

@testset "GrowableCachePairs" begin
    c = GrowableCache{Float64, Float64}()
    p = pairs(c)
    @test p isa GrowableCachePairs

    x, v = rand(5), rand(5)
    for i in 1:length(x)
        push!(p, (x[i], v[i]))
        @test p[i] == (x[i], v[i])
    end
    @test c.keys == x && c.values == v

    sort!(p; by = ((x, v),) -> x)
    @test c.keys == sort(x)
end

@testset "CachedFunction" begin
    count = Ref(0)
    f_inner(x) = (count[] += 1; x^2)
    c = GrowableCache{Float64, Float64}()
    f = CachedFunction(f_inner, c)

    v = f(1.5)
    @test count[] == length(c.keys) == length(c.values) == 1
    @test c.keys[1] == 1.5 && v == c.values[1] == 2.25

    v = f(1.5)
    @test count[] == length(c.keys) == length(c.values) == 1
    @test v == c.values[1] == 2.25

    v = f(2.0)
    @test count[] == length(c.keys) == length(c.values) == 2
    @test c.keys[1] == 1.5 && c.values[1] == 2.25
    @test c.keys[2] == 2.0 && v == c.values[2] == 4.0

    empty!(c)
    v = f(2.0)
    @test count[] == 3
    @test length(c.keys) == length(c.values) == 2
    @test c.keys[1] == 2.0 && v == c.values[2] == 4.0
end

@testset "MappedArray" begin
    x = [-1, 0, 1, 2]
    y = [1.0, 0.0, 1.0, 4.0]

    count = Ref(0)
    f = x -> (count[] += 1; abs2(Float64(x)))
    m = MappedArray{Float64}(f, x)

    @test count[] == 0 && m == y && count[] == 4 # && is guaranteed to evaluate left-to-right
    @test m[1] == m[3] == 1.0 && count[] == 6
    @test DECAES.mapfindmin(Float64, f, x) == (0, 0.0, 2) && count[] == 10
    @test DECAES.mapfindmax(Float64, f, x) == (2, 4.0, 4) && count[] == 14
end

@testset "SVDValsWorkspace" begin
    for (m, n) in Iterators.product(1:5, 1:5)
        A = randn(m, n)
        work = DECAES.SVDValsWorkspace(A)
        @test work.A !== A # should make a copy

        γ0 = svdvals(A)
        γ1 = @inferred(svdvals!(work))
        @test γ0 == γ1 # should match exactly, calling same LAPACK routine

        γ2 = @inferred(svdvals!(work, A))
        @test γ1 === γ2 # returns same internal buffer
        @test γ0 == γ2 # should match exactly, calling same LAPACK routine

        @test @allocations(svdvals!(work)) == 0
        @test @allocations(svdvals!(work, A)) == 0
    end

    # Factorize leading column blocks.
    for (m, n) in ((8, 6), (6, 8), (7, 7))
        A = randn(m, n)
        work = DECAES.SVDValsWorkspace(A)
        copyto!(work.A, A)
        for p in 1:n
            γ = svdvals!(work, p)
            @test @views γ[1:min(m, p)] == svdvals(A[:, 1:p])
            copyto!(work.A, A)
        end
        @test svdvals!(work, A) == svdvals(A)
    end

    bounds_work = DECAES.SVDValsWorkspace(zeros(2, 2))
    @test_throws AssertionError svdvals!(bounds_work, 0)
    @test_throws AssertionError svdvals!(bounds_work, 3)
end

# Allow roundoff from forming A*Q and factorizing the retained columns in addition to the truncation error bounded by the discarded Frobenius norm.
@testset "deflated_eigvals!" begin
    dof(γ², μ, m, n) = max(m - n, 0) + sum(g -> μ^2 / (g + μ^2), γ²)
    orth(k) = Matrix(qr(randn(k, k)).Q)
    for (m, n) in ((12, 8), (8, 12), (8, 8))
        k, ℓ = min(m, n), max(m, n)
        for (name, A) in (
            "full rank" => randn(m, n),
            "exact rank 5" => randn(m, 5) * randn(5, n),
            "graded to 1e-20" => Matrix(qr(randn(ℓ, k)).Q)[m >= n ? (1:m) : (1:n), 1:k] |> X -> (m >= n ? X * Diagonal(exp10.(range(0, -20; length = k))) * orth(k) : (X * Diagonal(exp10.(range(0, -20; length = k))) * orth(k))'),
            "clustered" => (X = Matrix(qr(randn(ℓ, k)).Q) * Diagonal([fill(1.0, k ÷ 2); fill(1e-18, k - k ÷ 2)]) * orth(k); m >= n ? X : X'),
            "zero" => zeros(m, n),
        )
            F = svd(A)
            Q = m >= n ? Matrix(F.V) : Matrix(F.U)
            γ²ref = svdvals(A) .^ 2
            γ² = zeros(k)
            spectrum_work = DECAES.SVDValsWorkspace(A)
            deflation_work = DECAES.SVDValsWorkspace(zeros(ℓ, k))
            DECAES.deflated_eigvals!(γ², spectrum_work, deflation_work, A, Q)
            τ = √DECAES.deflation_tolerance²(k, sum(abs2, A))

            @test issorted(γ²; rev = true)
            @test maximum(abs, .√γ² .- .√γ²ref) <= 5 * (τ + eps() * √maximum(γ²ref; init = 0.0))
            for μ in exp10.(range(-6, 3; length = 12)) .* max(norm(A), eps())
                @test dof(γ², μ, m, n) ≈ dof(γ²ref, μ, m, n) rtol = 1e-12
            end
            name == "full rank" && @test γ² == γ²ref
        end
    end
end

@testset "GriddedSpectrumInterpolator" begin
    # Smooth matrix family with an exact derivative.
    for (m, n) in ((6, 4), (4, 6), (5, 5))
        A0, A1, A2 = randn(m, n), randn(m, n), randn(m, n)
        Aα(α) = A0 .+ α .* A1 .+ α^2 .* A2
        ∂Aα(α) = A1 .+ 2α .* A2
        αs = collect(range(0.5, 2.0; length = 9))
        As = cat((Aα(α) for α in αs)...; dims = 3)
        ∇As = reshape(cat((∂Aα(α) for α in αs)...; dims = 3), m, n, 1, length(αs))
        interp = DECAES.GriddedSpectrumInterpolator(As, ∇As, αs)

        for μ in (1e-2, 1e-1, 1.0)
            for (i, α) in enumerate(αs)
                @test DECAES.gcv_dof_interp(interp, α, m, n, μ) ≈ DECAES.gcv_dof(Aα(α), μ) rtol = 1e-12
            end

            herr, lerr = 0.0, 0.0
            for α in range(αs[1], αs[end]; length = 40)
                d_exact = DECAES.gcv_dof(Aα(α), μ)
                d_herm = DECAES.gcv_dof_interp(interp, α, m, n, μ)
                i = clamp(searchsortedlast(αs, α), 1, length(αs) - 1)
                θ = (α - αs[i]) / (αs[i+1] - αs[i])
                d_lin = (1 - θ) * DECAES.gcv_dof(Aα(αs[i]), μ) + θ * DECAES.gcv_dof(Aα(αs[i+1]), μ)
                herr = max(herr, abs(d_herm - d_exact))
                lerr = max(lerr, abs(d_lin - d_exact))
            end

            @test herr < lerr
        end

        # Compare spectral derivatives with finite differences.
        i = 5
        h = 1e-6
        γ₊, γ₋ = svdvals(Aα(αs[i] + h)), svdvals(Aα(αs[i] - h))
        dγ²_fd = @. (γ₊ - γ₋) * (γ₊ + γ₋) / 2h
        @test interp.dγ²[:, i] ≈ dγ²_fd rtol = 1e-6 atol = 1e-12

        k = min(m, n)
        for j in eachindex(αs)
            Q = interp.Q[:, :, j]
            @test Q'Q ≈ I(k) rtol = 1e-12
            @test svdvals(m >= n ? Aα(αs[j]) * Q : Aα(αs[j])' * Q) ≈ .√interp.γ²[:, j] rtol = 1e-12
        end
    end
end

@testset "split_indices" begin
    function test_valid_partition(p, len)
        @test first(first(p)) == 1
        @test last(last(p)) == len
        @test all(i -> last(p[i-1]) + 1 == first(p[i]), 2:length(p))
    end

    for len in 1:20, minchunksize in 1:20
        p = DECAES.split_indices(; length = len, minchunksize)
        test_valid_partition(p, len)

        if len <= minchunksize
            @test length(p) == 1
            @test length(only(p)) == len
        else
            @test 1 <= length(p) <= len
            @test length(p) == len ÷ minchunksize
            @test all(c -> length(c) >= minchunksize, p)
        end
    end

    for len in 1:10, minchunksize in 1:10, maxpartitions in 1:10
        _basesize = min(len, max(minchunksize, len ÷ maxpartitions))
        p = DECAES.split_indices(; length = len, minchunksize, maxpartitions)
        test_valid_partition(p, len)

        @test length(p) >= 1
        @test all(c -> length(c) >= _basesize, p)
        if len >= minchunksize * maxpartitions
            @test length(p) == maxpartitions
        else
            @test 1 <= length(p) <= min(len, maxpartitions)
            @test length(p) == len ÷ _basesize
        end
    end
end

@testset "sorttuple" begin
    @test DECAES.sorttuple(()) === ()
    @test DECAES.sorttuple((1.0,)) === (1.0,)
    @test DECAES.sorttuple((3, 1, 2)) === (1, 2, 3)

    # Agrees with `sort` on random tuples, and is stable, since equal keys keep their input order
    @testset "n = $n" for n in 0:5
        for _ in 1:50
            t = Tuple(rand(1:4, n))
            @test DECAES.sorttuple(t) === Tuple(sort(collect(t)))
            p = Tuple((v, i) for (i, v) in enumerate(t))
            @test DECAES.sorttuple(p; by = first) === Tuple(sort(collect(p); by = first))
        end
    end

    # NaNs sort last under `lt_nan`, whichever position they enter in
    @testset "NaN ordering" for perm in [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
        t = ((2.0, :b), (NaN, :c), (1.0, :a))[collect(perm)] |> Tuple
        s = DECAES.sorttuple(t; by = first, lt = DECAES.lt_nan)
        @test s[1] === (1.0, :a)
        @test s[2] === (2.0, :b)
        @test isnan(s[3][1]) && s[3][2] === :c
    end

    # Type stable and non-allocating for the tuple sizes used above
    @test (@inferred DECAES.sorttuple((3.0, 1.0, 2.0))) === (1.0, 2.0, 3.0)
    @test @allocated(DECAES.sorttuple((3.0, 1.0, 2.0))) == 0
end
