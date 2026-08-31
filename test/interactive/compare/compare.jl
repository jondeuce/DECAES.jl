using Pkg
Pkg.activate(@__DIR__)

using MAT
using Statistics
using UnicodePlots

iqr(x) = quantile(x, 0.75) - quantile(x, 0.25)

function matfiles(dir)
    files = filter(endswith(".mat"), readdir(dir))
    isempty(files) && error("no MAT files found in $dir")
    return files
end

function summarize(file)
    t2maps = matread(file)
    @info "Image statistics" file

    if haskey(t2maps, "resnorm")
        resnorm = filter(isfinite, vec(t2maps["resnorm"]))
        positive = filter(>(0), resnorm)
        @info "Residual norm" median = median(resnorm) half_iqr = iqr(resnorm) / 2 num_zeros = count(iszero, resnorm)
        isempty(positive) || display(histogram(log10.(positive); xlabel = "log10(resnorm)", ylabel = "count", nbins = 64, vertical = true, height = 10, width = 80))
    end

    if haskey(t2maps, "alpha")
        alpha = filter(isfinite, vec(t2maps["alpha"]))
        @info "Refocusing angle" median = median(alpha) half_iqr = iqr(alpha) / 2
        display(histogram(cosd.(alpha); xlabel = "cosd(alpha)", ylabel = "count", nbins = 64, vertical = true, height = 10, width = 80))
    end
    return nothing
end

function relative_error(x, y)
    numerator = mean(abs, x .- y)
    denominator = mean(abs, x)
    return iszero(denominator) ? ifelse(iszero(numerator), zero(numerator), oftype(numerator, Inf)) : numerator / denominator
end

function compare_arrays(key, x, y)
    size(x) == size(y) || return @error "Arrays have different sizes" key size1 = size(x) size2 = size(y)
    finite1 = isfinite.(x)
    finite2 = isfinite.(y)
    finite1 == finite2 || @error "Arrays have different NaN/Inf locations" key
    mask = finite1 .& finite2
    any(mask) || return @warn "Array has no jointly finite values" key

    xfinite = x[mask]
    yfinite = y[mask]
    err = relative_error(xfinite, yfinite)
    if err <= √eps(Float64)
        println(key => err)
    elseif err <= 1e-6
        @warn "Relative error is small but non-negligible" key err
    else
        @error "Relative error is large" key err
        scale = mean(abs, xfinite)
        dx = iszero(scale) ? abs.(xfinite .- yfinite) : abs.(xfinite .- yfinite) ./ scale
        dx_high = quantile(dx, 0.99)
        dxhist = filter(x -> 0 < x <= dx_high, dx)
        @info "Error distribution" key num_zeros = count(iszero, dx) above_q99 = count(>(dx_high), dx) q99 = dx_high
        isempty(dxhist) || display(histogram(dxhist; nbins = 64, vertical = true, height = 10, width = 80))
    end
    return nothing
end

function compare_files(file1, file2)
    data1 = matread(file1)
    data2 = matread(file2)
    @info "Comparing file" file = basename(file1)

    for key in sort!(collect(union(keys(data1), keys(data2))))
        haskey(data1, key) || (@error "Key is missing from first file" key; continue)
        haskey(data2, key) || (@error "Key is missing from second file" key; continue)
        x, y = data1[key], data2[key]
        x isa AbstractArray{<:Number} && y isa AbstractArray{<:Number} || continue
        compare_arrays(key, x, y)
    end
    return nothing
end

function compare_outputs(dir1, dir2)
    @info "Comparing outputs" first = abspath(dir1) second = abspath(dir2)
    files1 = matfiles(dir1)
    files2 = matfiles(dir2)
    for file in sort!(collect(union(files1, files2)))
        file in files1 || (@error "File is missing from first output" file; continue)
        file in files2 || (@error "File is missing from second output" file; continue)
        compare_files(joinpath(dir1, file), joinpath(dir2, file))
    end
    return nothing
end

function main(args = ARGS)
    if length(args) < 2
        println(stderr, "usage: julia --project=. compare.jl OUTPUT_DIR1 OUTPUT_DIR2 [OUTPUT_DIR3 ...]")
        return
    end
    for dir in args, file in filter(endswith(".t2maps.mat"), matfiles(dir))
        summarize(joinpath(dir, file))
    end
    foreach(dirs -> compare_outputs(dirs...), zip(args, Iterators.drop(args, 1)))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
