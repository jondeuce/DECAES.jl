"""
    main(command_line_args = ARGS)

Entry point function for command line interface, parsing the command line arguments `ARGS` and subsequently calling one or both of `T2mapSEcorr` and `T2partSEcorr` with the parsed settings.
See the [Arguments](@ref) section for available options.

See also:
* [`T2mapSEcorr`](@ref)
* [`T2partSEcorr`](@ref)
"""
function main(command_line_args::Vector{String} = ARGS)
    # Parse command line arguments. Note that if --legacy flag is passed,
    # inputs will have to be re-parsed since default values are --legacy dependent
    opts = parse_args(command_line_args, ARGPARSE_SETTINGS; as_symbols = true)
    if opts[:legacy] == true
        opts = parse_args(command_line_args, ARGPARSE_SETTINGS_LEGACY; as_symbols = true)
    end

    # Unpack parsed flags, overriding appropriate options fields
    @unpack T2map, T2part, quiet, dry = opts
    @unpack bet, betpath, betargs = opts
    t2map_kwargs = get_parsed_args_subset(opts, T2MAP_FIELDTYPES)
    t2part_kwargs = get_parsed_args_subset(opts, T2PART_FIELDTYPES)

    # Override --Silent flag with --quiet flag, if set
    quiet && (t2map_kwargs[:Silent] = quiet)
    quiet && (t2part_kwargs[:Silent] = quiet)

    # Infer nT2 from input T2 distribution
    !T2map && (delete!(t2part_kwargs, :nT2))

    # Get input file list and output folder list
    inputfiles, outputfolders = get_file_info(opts)

    for (filename, folder) in zip(inputfiles, outputfolders)
        # Make output path
        choppedfilename, filesuffix = chop_allowed_suffix(basename(filename))
        mkpath(folder)

        # Set logger to output to both console and log file
        logfile = joinpath(folder, choppedfilename * ".log")
        open(logfile, "w+") do io
            with_logger(TeeLogger(ConsoleLogger(stdout), ConsoleLogger(io))) do
                try
                    # Starting message/starting time
                    t_start = tic()
                    !quiet && @info "Starting with $(Threads.nthreads()) threads"

                    # Save settings files
                    !dry && map(filter(s -> startswith(s, "@"), command_line_args)) do settingsfile
                        src = settingsfile[2:end] # drop "@" character
                        dst = joinpath(folder, choppedfilename * "." * basename(src))
                        cp(src, dst; force = true)
                    end

                    # Load image(s)
                    image = @showtime(load_image(filename), "Loading input file: $filename", !quiet)

                    # Apply BET mask
                    bet && @showtime(make_apply_bet_mask!(image, betpath, betargs), "Making and applying BET mask with args: '$betargs'", !quiet)

                    # Results variables
                    local maps, dist, parts

                    if T2map
                        # Compute T2 distribution from input 4D multi-echo image
                        maps, dist = @showtime(T2mapSEcorr(image; t2map_kwargs...), "Running T2mapSEcorr on file: $filename", !quiet)

                        # Save results to .mat files
                        savefile = joinpath(folder, choppedfilename * ".t2dist.mat")
                        !dry && @showtime(MAT.matwrite(savefile, Dict("dist" => dist)), "Saving T2 distribution to file: $savefile", !quiet)

                        savefile = joinpath(folder, choppedfilename * ".t2maps.mat")
                        !dry && @showtime(MAT.matwrite(savefile, maps), "Saving T2 parameter maps to file: $savefile", !quiet)
                    else
                        # Input image is the T2 distribution
                        dist = image
                    end

                    if T2part
                        # Analyze T2 distribution to produce parameter maps
                        parts = @showtime(T2partSEcorr(dist; t2part_kwargs...), "Running T2partSEcorr", !quiet)

                        # Save results to .mat file
                        savefile = joinpath(folder, choppedfilename * ".t2parts.mat")
                        !dry && @showtime(MAT.matwrite(savefile, parts), "Saving T2 parts maps to file: $savefile", !quiet)
                    end

                    # Done message
                    if !quiet
                        println(stdout, ""); @info "Finished ($(round(toc(t_start); digits = 2)) seconds)"; println(stdout, "")
                    end
                catch e
                    println(stdout, ""); @warn "Error during processing of file: $filename"
                    println(stdout, ""); @warn sprint(showerror, e, catch_backtrace())
                end
            end
        end
    end

    return nothing
end

function create_argparse_settings(;legacy = false)
    settings = ArgParseSettings(
        prog = "",
        fromfile_prefix_chars = "@",
        error_on_conflict = false,
        exit_after_help = true,
        # exc_handler = ArgParse.debug_handler,
    )

    @add_arg_table settings begin
        "input"
            help = "one or more input filenames or directories. If directories are passed, they are scanned non-recursively for valid input files. Valid file extensions are limited to: $ALLOWED_FILE_SUFFIXES_STRING"
            required = true
            nargs = '+' # At least one input is required
        "--output", "-o"
            help = "output directory. If not specified, output file(s) will be stored in the same location as the corresponding input file(s). Outputs are stored with the same basename as inputs plus added suffixes; see --T2map and --T2part"
        "--T2map"
            help = "call T2mapSEcorr to compute T2 distributions from 4D multi spin-echo input images. T2 distributions and T2 maps produced by T2mapSEcorr are saved as MAT files with extensions .t2dist.mat and .t2maps.mat"
            action = :store_true
        "--T2part"
            help = "call T2partSEcorr to analyze 4D T2 distributions to produce parameter maps. If --T2map is also passed, input 4D arrays are interpreted as multi spin-echo images and T2 distributions are first computed by T2mapSEcorr. If only --T2part is passed, input 4D arrays are interpreted as T2 distributions and only T2partSEcorr is called. Output T2 parts are saved as a MAT file with extension .t2parts.mat"
            action = :store_true
        "--quiet"
            help = "suppress printing to the terminal. Note: 1) errors are not silenced, and 2) this flag overrides the --Silent flag in T2mapSEcorr"
            action = :store_true
        "--dry"
            help = "execute dry run of processing without saving any results"
            action = :store_true
        "--legacy"
            help = "use legacy settings and algorithms from the original MATLAB version. This ensures that the exact same T2-distributions and T2-parts will be produced as those from MATLAB (to machine precision). Note that execution time will be much slower."
            action = :store_true
    end

    add_arg_group(settings,
        "T2mapSEcorr/T2partSEcorr arguments",
        "internal arguments",
    )
    t2map_opts = T2mapOptions{Float64}(nTE = 32, MatrixSize = (1,1,1), legacy = legacy)
    t2part_opts = T2partOptions{Float64}(nT2 = t2map_opts.nT2, MatrixSize = (1,1,1), legacy = legacy)
    opts_args = sorted_arg_table_entries(t2map_opts, t2part_opts)
    add_arg_table(settings, opts_args...)

    add_arg_group(settings,
        "BET arguments",
        "arguments for mask generation using BET",
    )
    @add_arg_table settings begin
        "--bet"
            help = "use the BET brain extraction tool from the FSL library of analyis tools to automatically create a binary brain mask. Only voxels within the binary mask will be analyzed, greatly decreasing processing time."
            action = :store_true
        "--betargs"
            help = "BET optional arguments. Must be passed as a single string with arguments separated by spaces, e.g. '-m -n'. The flag '-m' creates the binary mask and will be added to the list of arguments if not provided."
            arg_type = String
            default = "-m -n -f 0.25 -R"
        "--betpath"
            help = "path to BET executable."
            arg_type = String
            default = "bet"
    end

    return settings
end

function load_image(filename)
    image = if endswith(filename, ".mat")
        data = MAT.matread(filename)
        key = findfirst(x -> x isa AbstractArray{T,4} where {T}, data)
        if key === nothing
            error("No 4D array was found in the input file: $filename")
        end
        data[key]
    elseif endswith(filename, ".nii") || endswith(filename, ".nii.gz")
        data = NIfTI.niread(filename)
        # Check slope field; if scl_slope == 0, data is not scaled and raw data should be returned
        #   See e.g. https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/scl_slopeinter.html
        if data.header.scl_slope == 0
            data.raw[:,:,:,:] # Return raw data
        else
            data[:,:,:,:] # Return scaled data (getindex from the NIfTI package handles scaling)
        end
    else
        error("Currently, only $ALLOWED_FILE_SUFFIXES_STRING files are supported")
    end
    if !(image isa Array{Float64,4})
        # Currently, the pipeline is ~twice as fast on Float64 arrays than Float32 arrays.
        # I'm not clear on exactly why this is at the moment. However, the MATLAB toolbox
        # converts images to double as well, so I'm going to do the same here while loading
        image = convert(Array{Float64,4}, image)
    end
    return image
end

function make_bet_mask(image, betpath, betargs)
    # Split betargs, and ensure that "-m" (make binary mask) is among args
    args = convert(Vector{String}, filter!(!isempty, split(betargs, " ")))
    if "-m" ∉ args
        push!(args, "-m")
    end

    # Create mask using BET and return mask
    mktempdir() do temppath
        tempbase = basename(tempname())
        nifti_imagefile = joinpath(temppath, tempbase * ".nii")
        nifti_maskfile = joinpath(temppath, tempbase * ".bet")
        niwrite(nifti_imagefile, NIVolume(image)) # create nifti file for bet
        run(Cmd([
            betpath;
            nifti_imagefile;
            nifti_maskfile;
            args
        ]))
        # BET appends "_mask" and ".nii.gz" to output file name.
        # Find this file, ensure it is unique, then load and return it
        bet_maskfiles = filter!(file -> startswith(file, tempbase * ".bet_mask"), readdir(temppath))
        @assert length(bet_maskfiles) == 1 # ensure unique
        load_image(joinpath(temppath, bet_maskfiles[1]))
    end
end

function make_apply_bet_mask!(image, betpath, betargs)
    try
        image .*= make_bet_mask(image, betpath, betargs)
    catch e
        @warn "Error while making mask using BET"
        @warn sprint(showerror, e, catch_backtrace())
    end
    return image
end

####
#### Helper functions
####

_parse_or_convert(::Type{T}, s::AbstractString) where {T} = parse(T, s)
_parse_or_convert(::Type{T}, x) where {T} = convert(T, x)
_strip_union_nothing(::Type{Union{T, Nothing}}) where {T} = T
_strip_union_nothing(T::Type) = T
_get_tuple_type(::Type{Union{Tup, Nothing}}) where {Tup <: Tuple} = Tup
_get_tuple_type(::Type{Tup}) where {Tup <: Tuple} = Tup

is_allowed_suffix(filename) = any(endswith.(filename, ALLOWED_FILE_SUFFIXES))
filter_allowed_suffix(filenames) = filter(is_allowed_suffix, filenames)

function chop_allowed_suffix(filename::AbstractString)
    for suffix in ALLOWED_FILE_SUFFIXES
        if endswith(filename, suffix)
            return filename[1:end-length(suffix)], suffix
        end
    end
    error("Currently only $ALLOWED_FILE_SUFFIXES_STRING file types are supported")
end

function get_file_info(opts)
    @unpack input, output = opts    

    inputfiles = String[]
    for path in input
        if isdir(path)
            filenames = joinpath.(path, filter_allowed_suffix(readdir(path)))
            append!(inputfiles, filenames)
        elseif is_allowed_suffix(path)
            push!(inputfiles, path)
        end
    end

    if isempty(inputfiles)
        msg = if !isempty(input) && isfile(input[1])
            "No valid file types were found for processing, but a file name was passed.\n" *
            "Perhaps you meant to prepend an '@' character to a settings file, e.g. @$(input[1])?\n" *
            "If not, note that only $ALLOWED_FILE_SUFFIXES_STRING file types are supported"
        else
            "No valid files were found for processing. Note that currently only $ALLOWED_FILE_SUFFIXES_STRING file types are supported"
        end
        error(msg)
    end

    outputfolders = if output === nothing
        dirname.(inputfiles)
    else
        [output for _ in 1:length(inputfiles)]
    end

    return @ntuple(inputfiles, outputfolders)
end

function sorted_arg_table_entries(options...)
    fields, types, values = Symbol[], Type[], Any[]
    for o in options, (f,T) in zip(fieldnames(typeof(o)), fieldtypes(typeof(o)))
        push!(fields, f); push!(types, T); push!(values, getfield(o, f))
    end
    args = []
    defaults = collect(zip(fields, values, types))
    sort!(defaults; by = tup -> uppercase(string(first(tup)))) # sort alphabetically
    for (k, v, T) in defaults
        (k === :nTE || k === :MatrixSize) && continue # Skip automatically determined parameters
        (k === :vTEparam || k == :legacy) && continue # Skip printing
        push!(args, "--" * string(k))
        if T <: Bool
            push!(args, Dict(:action => ifelse(v, :store_false, :store_true)))
        elseif T <: Union{<:Tuple, Nothing}
            nargs = length(fieldtypes(_get_tuple_type(T)))
            if v === nothing
                push!(args, Dict(:nargs => nargs, :default => v))
            else
                push!(args, Dict(:nargs => nargs, :default => [v...]))
            end
        else
            push!(args, Dict(:default => v))
        end
    end
    return args
end

function get_parsed_args_subset(opts::Dict{Symbol,Any}, subset_fieldtypes::Dict{Symbol,Type})
    kwargs = deepcopy(opts)
    for (k,v) in kwargs
        if k ∉ keys(subset_fieldtypes)
            delete!(kwargs, k)
            continue
        end
        if v isa AbstractString # parse v to appropriate type, which may not be String
            T = subset_fieldtypes[k]
            if !(T <: AbstractString)
                kwargs[k] = _parse_or_convert(_strip_union_nothing(T), v)
            end
        elseif v isa AbstractVector # convert AbstractVector v to appropriate Tuple type
            if isempty(v)
                delete!(kwargs, k) # default v === nothing for a Tuple type results in an empty vector
            else
                T = _get_tuple_type(subset_fieldtypes[k])
                kwargs[k] = tuple(_parse_or_convert.(fieldtypes(T), v)...) # each element should be individually parsed
            end
        end
    end
    return kwargs
end

# Global constants and settings computed during precompilation
const ALLOWED_FILE_SUFFIXES = (".mat", ".nii", ".nii.gz")
const ALLOWED_FILE_SUFFIXES_STRING = join(ALLOWED_FILE_SUFFIXES, ", ", ", and ")

const ARGPARSE_SETTINGS = create_argparse_settings(legacy = false)
const ARGPARSE_SETTINGS_LEGACY = create_argparse_settings(legacy = true)

const T2MAP_FIELDTYPES = Dict{Symbol,Type}(fieldnames(T2mapOptions{Float64}) .=> fieldtypes(T2mapOptions{Float64}))
const T2PART_FIELDTYPES = Dict{Symbol,Type}(fieldnames(T2partOptions{Float64}) .=> fieldtypes(T2partOptions{Float64}))
