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
            help = "output folder"
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

    pretty_args = Dict([Symbol(k) => v isa AbstractVector ? join(string.(v), ",") : string(v) for (k,v) in args])
    @info "Benchmarking DECAES with settings:\n" pretty_args...

    run(`
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
        --export-markdown $(joinpath(mkpath(args["output"]), "results.md"))
        --export-json $(joinpath(mkpath(args["output"]), "results.json"))
    `)
    # --show-output
    return nothing
end

main()
