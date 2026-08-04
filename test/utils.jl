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
end

@testset "GramEigvalsWorkspace" begin
    for (m, n) in Iterators.product(2:6, 2:6)
        A = randn(m, n)
        work = DECAES.GramEigvalsWorkspace(A)

        γ² = @inferred LinearAlgebra.eigvals!(work, A)
        @test length(γ²) == min(m, n)
        @test issorted(γ²) # `syevr` returns ascending eigenvalues
        @test γ² ≈ sort(svdvals(A) .^ 2) rtol = 1e-10 atol = 1e-12 * max(1, sum(abs2, A))

        γ²′ = @inferred LinearAlgebra.eigvals!(work, A)
        @test γ²′ === γ² # returns the same internal buffer
        @test @allocations(LinearAlgebra.eigvals!(work, A)) == 0 # LAPACK workspace is preallocated and reused
    end

    # The accuracy is absolute, not relative: forming the Gram matrix squares the condition number, so only |γᵢ² − σᵢ²| = O(eps·‖A‖²_F) holds, and the small γᵢ² can come out negative.
    for A in (
        (B = randn(6, 4); B[:, 4] = B[:, 3] .+ 1e-8 .* randn(6); B),           # nearly duplicated columns
        (B = randn(6, 4); B[:, 4] = B[:, 3]; B[:, 1] .*= 1e8; B),              # duplicated columns, badly scaled
        Matrix(Diagonal([1.0, 1e-4, 1e-8, 1e-12])),                            # graded spectrum
    )
        γ² = LinearAlgebra.eigvals!(DECAES.GramEigvalsWorkspace(A), A)
        @test issorted(γ²)
        @test maximum(abs, γ² .- sort(svdvals(A) .^ 2)) <= 8 * eps() * sum(abs2, A)
    end
end

@testset "GriddedSpectrumInterpolator" begin
    # Smooth matrix family A(α) = A0 + α·A1 + α²·A2 with exact ∂A/∂α = A1 + 2α·A2, over a grid in α; covers both m ≥ n and m < n
    for (m, n) in ((6, 4), (4, 6), (5, 5))
        A0, A1, A2 = randn(m, n), randn(m, n), randn(m, n)
        Aα(α) = A0 .+ α .* A1 .+ α^2 .* A2
        ∂Aα(α) = A1 .+ 2α .* A2
        αs = collect(range(0.5, 2.0; length = 9))
        As = cat((Aα(α) for α in αs)...; dims = 3)
        ∇As = reshape(cat((∂Aα(α) for α in αs)...; dims = 3), m, n, 1, length(αs))
        interp = DECAES.GriddedSpectrumInterpolator(As, ∇As, αs)

        for μ in (1e-2, 1e-1, 1.0)
            # Exact at grid nodes: Hermite passes through the endpoint values
            for (i, α) in enumerate(αs)
                @test DECAES.gcv_dof_interp(interp, α, m, n, μ) ≈ DECAES.gcv_dof(Aα(α), μ) rtol = 1e-12
            end

            # Between nodes: cubic Hermite is a loose-bounded approximation and beats linear interp of dof on the same grid
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

            # Cubic Hermite is meaningfully more accurate than linear interp of dof
            @test herr < lerr
        end

        # Spectral derivative dγ²/dα = 2σ·uᵀ(∂A/∂α)v matches finite differences of γ² at an interior node
        i = 5
        h = 1e-6
        γ₊, γ₋ = svdvals(Aα(αs[i] + h)), svdvals(Aα(αs[i] - h))
        dγ²_fd = @. (γ₊ - γ₋) * (γ₊ + γ₋) / 2h
        DECAES.gridded_spectrum_slice!(interp, i)
        dγ²_int = interp.dγ²[:, i]
        @test dγ²_int ≈ dγ²_fd rtol = 1e-6 atol = 1e-6
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
