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
        "--threads", "-t"
            help = "number of threads"
            nargs = '*' # Zero or more inputs
            arg_type = Union{Int,String}
            default = Union{Int,String}["auto"]
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
        # "--flag1"
        #     help = "an option without argument, i.e. a flag"
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
        --parameter-list julia $(join(args["julia"], ","))
        --parameter-list input $(join(args["input"], ","))
        --parameter-list threads $(join(args["threads"], ","))
        --parameter-list optimize $(join(args["optimize"], ","))
        --warmup $(args["warmup"])
        --min-runs $(args["min-runs"])
        --export-markdown $(joinpath(mkpath(args["output"]), "results.md"))
        --export-json $(joinpath(mkpath(args["output"]), "results.json"))
        "{julia} --project --startup-file=no --quiet --threads={threads} --optimize={optimize} -e 'using DECAES; main()' @{input}"
    `)
end

main()
