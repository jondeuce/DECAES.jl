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
        γ1 = svdvals!(work)
        @test γ0 == γ1 # should match exactly, calling same LAPACK routine

        γ2 = svdvals!(work, A)
        @test γ1 === γ2 # returns same internal buffer
        @test γ0 == γ2 # should match exactly, calling same LAPACK routine

        @test @allocated(svdvals!(work)) == 0
        @test @allocated(svdvals!(work, A)) == 0
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
