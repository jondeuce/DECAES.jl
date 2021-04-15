using Dates
using ArgParse

function parse_commandline()
    s = ArgParseSettings(
        fromfile_prefix_chars = "@",
    )

    @add_arg_table! s begin
        "--input", "-i"
            help = "one or more settings files"
            nargs = '+' # At least one input is required
            arg_type = String
            required = true
        "--output", "-o"
            help = "output folder suffix; timestamp will be prepended"
            arg_type = String
            required = true
        "--julia"
            help = "path to julia binary"
            nargs = '*' # Zero or more inputs
            arg_type = String
            default = String["julia"]
        "--decaes-version", "-v"
            help = "tag (e.g. 0.3 or v0.3), branch (e.g. master), or git commit (e.g. 338795e)"
            nargs = '*' # Zero or more inputs
            arg_type = String
            default = String["master"]
        "--threads", "-t"
            help = "number of threads"
            nargs = '*' # Zero or more inputs
            arg_type = Any
            default = Any["auto"]
        "--optimize", "-O"
            help = "optimization level"
            arg_type = Int
            nargs = '*' # Zero or more inputs
            default = Int[2]
        "--warmup"
            help = "number of warmup runs"
            arg_type = Int
            default = 1
        "--min-runs"
            help = "minimum number of runs"
            arg_type = Int
            default = 3
        # "--show-output"
        #     help = "print stdout and stderr of the benchmark instead of suppressing"
        #     action = :store_true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    dateformat = "yyyy-mm-dd-T-HH-MM-SS"
    timestamp = Dates.format(Dates.now(), dateformat)
    outfolder = timestamp * "_" * args["output"]
    outpath(xs...) = joinpath(mkpath(outfolder), xs...)

    # Benchmarking command
    cmd = `
    hyperfine
        "JULIA_NUM_THREADS={threads} {julia} --project=.bench.tmp/{julia}/{version} --startup-file=no --quiet --optimize={optimize} -e 'using DECAES; main()' -- @{input} --quiet"
        --prepare "{julia} --startup-file=no -e 'include(\"utils.jl\"); prepare_project(\"{julia}\", \"{version}\")'"
        --warmup $(args["warmup"])
        --min-runs $(args["min-runs"])
        --parameter-list version $(join(args["decaes-version"], ","))
        --parameter-list julia $(join(args["julia"], ","))
        --parameter-list threads $(join(args["threads"], ","))
        --parameter-list optimize $(join(args["optimize"], ","))
        --parameter-list input $(join(args["input"], ","))
        --export-markdown $(outpath("results.md"))
        --export-json $(outpath("results.json"))
    `
    # --show-output

    # Save benchmarking settings/files for future reference
    open(outpath("settings.txt"); write = true) do io
        for arg in ARGS
            startswith(arg, "@") ?
                println(io, readchomp(arg[2:end])) :
                println(io, arg)
        end
    end
    open(outpath("run_benchmarks.jl"); write = true) do io
        println(io, "mkpath(\"$(outpath())\")")
        println(io, "run($cmd)")
    end
    for jl in filter(endswith(".jl"), readdir(@__DIR__; join = true))
        cp(jl, outpath(basename(jl)); force = true)
    end

    # Run benchmarks
    @info "Benchmarking DECAES with settings:\n" * readchomp(outpath("settings.txt"))
    run(cmd)
    rm(".bench.tmp"; force = true, recursive = true)

    return nothing
end

main()
