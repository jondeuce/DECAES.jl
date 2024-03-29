using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)

using MAT
using Statistics
using UnicodePlots

iqr(x) = quantile(x, 0.75) - quantile(x, 0.25)

versions = [
    "1.9" => ["v0.5.1"],
    "1.10" => ["v0.5.2-DEV"],
]
settings_files = [
    "settings.txt",
]
cli_extra_args = `--Reg lcurve  --SaveResidualNorm --SaveRegParam --quiet`

for (julia_version, decaes_versions) in versions
    for decaes_version in decaes_versions, settings in settings_files
        @info "----------------"
        @info "Running DECAES with Julia $(julia_version) and DECAES $(decaes_version)"
        @info "----------------"

        project_dir = joinpath(@__DIR__, "envs", "julia-v" * julia_version, decaes_version)
        output_dir = joinpath(@__DIR__, "output", splitext(settings)[1], "julia-v" * julia_version * "_" * decaes_version)
        @time run(`julia +$(julia_version) --startup-file=no --project=$(project_dir) --threads=$(Threads.nthreads()) -e "using DECAES; main()" @$(joinpath(@__DIR__, "settings", settings)) --output $(output_dir) $(cli_extra_args)`)
        println()
    end
end

for settings in settings_files
    r = r"julia\-v1\.(?<julia_minor>(?:\d)+)_v0\.(?<decaes_major>(?:\d)+)\.(?<decaes_minor>(?:\d)+)(?:\-DEV)?"
    outputs = readdir(joinpath(@__DIR__, "output", splitext(settings)[1]); join = true, sort = true)
    sort!(outputs; by = s -> match(r, s) |> m -> parse.(Int, (m[:julia_minor], m[:decaes_major], m[:decaes_minor])))
    for i in eachindex(outputs)
        t2mapfile = only(filter(endswith(".t2maps.mat"), readdir(outputs[i]; join = true)))
        t2maps = matread(t2mapfile)
        mask = isfinite.(t2maps["resnorm"])
        resnorm = t2maps["resnorm"][mask]
        alpha = t2maps["alpha"][mask]
        @info "Image stats" t2mapfile num_zeros = sum(iszero, resnorm)
        histogram(
            log10.(resnorm);
            xlabel = "log10(resnorm)", ylabel = "count",
            title = "resnorm = $(median(resnorm)) ± $(iqr(resnorm) / 2), log10(resnorm) = $(median(log10.(resnorm))) ± $(iqr(log10.(resnorm)) / 2)",
            nbins = 64, vertical = true, height = 10, width = 80,
        ) |> display
        histogram(
            cosd.(alpha);
            xlabel = "cosd(alpha)", ylabel = "count",
            title = "alpha = $(median(alpha)) ± $(iqr(alpha) / 2), cosd(alpha) = $(median(cosd.(alpha))) ± $(iqr(cosd.(alpha)) / 2)",
            nbins = 64, vertical = true, height = 10, width = 80,
        ) |> display
    end
end

for settings in settings_files
    r = r"julia\-v1\.(?<julia_minor>(?:\d)+)_v0\.(?<decaes_major>(?:\d)+)\.(?<decaes_minor>(?:\d)+)(?:\-DEV)?"
    outputs = readdir(joinpath(@__DIR__, "output", splitext(settings)[1]); join = true, sort = true)
    sort!(outputs; by = s -> match(r, s) |> m -> parse.(Int, (m[:julia_minor], m[:decaes_major], m[:decaes_minor])))
    for i in 1:length(outputs)-1
        @info "----------------"
        @info "Comparing outputs: $(basename(outputs[i])) vs. $(basename(outputs[i+1]))"
        @info "----------------"
        println()

        for file in readdir(outputs[i]; join = false, sort = true)
            endswith(file, ".mat") || continue
            @assert isfile(joinpath(outputs[i+1], file)) "File $(file) not found in $(outputs[i+1])"
            global file1 = joinpath(outputs[i], file)
            global file2 = joinpath(outputs[i+1], file)

            @info "Comparing file: $(file)"
            global data1 = matread(file1)
            global data2 = matread(file2)
            for key in keys(data1)
                @assert key in keys(data2) "Key $(key) not found in $(file2)"
                @assert ndims(data1[key]) == ndims(data2[key]) "Keys $(key) have different number of dimensions"
                ndims(data1[key]) >= 3 || continue

                I = findall(isfinite.(data1[key]))
                if I != (I2 = findall(isfinite.(data2[key])))
                    @error "Keys $(key) have different NaN/Inf locations"
                    I = intersect(I, I2)
                end

                global x1 = data1[key][I]
                global x2 = data2[key][I]
                if key == "mu"
                    x1 .= log.(x1)
                    x2 .= log.(x2)
                end

                # err = maximum(abs, @. (x1 - x2) / ifelse(iszero(x1) || iszero(x2), one(x1), max(abs(x1), abs(x2)))) # maximum relative error
                err = mean(abs, x1 .- x2) / mean(abs, x1 .- mean(x1)) # RAE: relative absolute error
                # err = √(mean(abs2, x1 .- x2) / mean(abs2, x1 .- mean(x1))) # RRSE: root relative squared error

                if err <= √eps()
                    println("$(key => err)") # error is negligible
                    continue
                elseif err <= 1e-6
                    @warn "$(key => err): relative error is small but non-negligible"
                    continue
                else
                    @error "$(key => err): relative error is large"
                    global dx = abs.(x1 .- x2) ./ mean(abs, x1 .- mean(x1))
                    dx_low = 0.0
                    dx_high = quantile(dx, 0.99)
                    nz = count(iszero, dx)
                    n99 = count(>(dx_high), dx)
                    dxhist = filter(x -> dx_low < x < dx_high, dx)
                    @info "$key: nz = $nz, n99 = $n99, $dx_low < dx < $dx_high"
                    !isempty(dxhist) && display(histogram(dxhist; nbins = 64, vertical = true, height = 10, width = 80))
                end
            end
            println()
        end
    end
end
