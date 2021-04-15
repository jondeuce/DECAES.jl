using Dates
using DataFrames
using JSON
using StatsPlots
pyplot(size = (1600,1200))

dateformat() = "yyyy-mm-dd-T-HH-MM-SS"
getnow() = Dates.format(Dates.now(), dateformat())
parsetime(s) = DateTime(s, dateformat())
parsedataset(s) = occursin("48echo", s) ? "240x240x48x48" : occursin("56echo", s) ? "240x240x113x56" : error("Unknown data source: $s")

function load_results()
    df = DataFrame()
    for dir in readdir(joinpath(@__DIR__, "results"); join = true)
        for res in JSON.parsefile(joinpath(dir, "results.json"))["results"]
            push!(df, (
                date = parsetime(split(basename(dir), '_')[1]),
                name = split(basename(dir), '_')[2] |> String,
                time = res["min"] |> Float64,
                dataset = parsedataset(res["parameters"]["input"]),
                julia = res["parameters"]["julia"],
                optlevel = parse(Int, res["parameters"]["optimize"]),
                threads = parse(Int, res["parameters"]["threads"]),
                version = res["parameters"]["version"],
            ))
        end
    end
    return df
end

function plot_results(df; compare_version = "v0.3")
    df = sort(df, [:dataset, :threads, :julia, :version, :optlevel])
    gd = groupby(df, [:dataset, :threads])
    ptimes = []
    pspeedups = []
    for (k,g) in zip(keys(gd), gd)
        groupkeystr = join(["$k = $v" for (k,v) in pairs(k)], ", ")

        ptime = @df g scatter(
            :version, :time, group = :julia,
            title = groupkeystr, xlabel = "version", ylabel = "time [s]",
            marker = (10, [:circle :square :utriangle :dtriangle]),
            legend = :topright,
        )
        push!(ptimes, ptime)

        if false
            # Times relative to `compare_version` runs with exactly the same parameters
            grel = DataFrame()
            for gsub in groupby(deepcopy(g), [:julia, :threads, :optlevel])
                if compare_version ∈ gsub.version
                    gsub.time .= only(gsub.time[gsub.version .== compare_version]) ./ gsub.time
                    append!(grel, gsub)
                end
            end
        else
            # Times relative to minimum `compare_version` time
            grel = deepcopy(g)
            grel.time .= minimum(grel[grel.version .== compare_version, :time]) ./ grel.time
        end
        pspeedup = @df grel scatter(
            :version, :time, group = :julia,
            title = groupkeystr, xlabel = "version", ylabel = "speedup w.r.t. fastest $compare_version",
            marker = (10, [:circle :square :utriangle :dtriangle]),
            legend = :topleft,
        )
        push!(pspeedups, pspeedup)
    end
    plot(ptimes...) |> display
    plot(pspeedups...) |> display
end

df = load_results()
plot_results(df)
