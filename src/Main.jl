module Main

if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@optlevel"))
    @eval Base.Experimental.@optlevel 0
end

import ..DECAES: T2mapOptions, T2partOptions, T2mapSEcorr, T2partSEcorr, @unpack, @showtime, printheader, printbody, tic, toc
using ArgParse, Logging, LoggingExtras

include("ParXRec.jl")
import .ParXRec, MAT, NIfTI
export ParXRec, MAT, NIfTI
export main, load_image

"""
    main(command_line_args = ARGS)

Entry point function for command line interface, parsing the command line arguments `ARGS` and subsequently calling one or both of `T2mapSEcorr` and `T2partSEcorr` with the parsed settings.
See the [Arguments](@ref) section for available options.

See also:
* [`T2mapSEcorr`](@ref)
* [`T2partSEcorr`](@ref)
"""
function main(command_line_args::Vector{String} = ARGS)
    # Parse command line arguments
    opts = parse_args(command_line_args, ARGPARSE_SETTINGS; as_symbols = true)

    # Unpack parsed flags, overriding appropriate options fields
    t2map_kwargs = get_parsed_args_subset(opts, T2MAP_FIELDTYPES)
    t2part_kwargs = get_parsed_args_subset(opts, T2PART_FIELDTYPES)

    # If not performing T2-mapping, infer nT2 from input T2 distribution
    if !opts[:T2map]
        delete!(t2part_kwargs, :nT2)
    end

    # Get input file list and output folder list
    for info in get_file_info(opts)
        # Make output path
        if !opts[:dry]
            mkpath(info[:outputfolder])
        end

        # Save settings files
        if !opts[:dry]
            map(filter(s -> startswith(s, "@"), command_line_args)) do settingsfile
                src = settingsfile[2:end] # drop "@" character
                dst = joinpath(info[:outputfolder], info[:choppedinputfile] * "." * basename(src))
                cp(src, dst; force = true)
            end
        end

        # Main processing
        tee_capture(
            logfile = joinpath(info[:outputfolder], info[:choppedinputfile] * ".log"),
            suppress_terminal = opts[:quiet],
            suppress_logfile = opts[:dry],
        ) do io
            try
                _main(io, info, opts, t2map_kwargs, t2part_kwargs)
            catch e
                @warn "Error during processing of file: $(info[:inputfile])"
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end

    return nothing
end

function _main(
        io::IO,
        info::Dict,
        opts::Dict,
        t2map_kwargs::Dict,
        t2part_kwargs::Dict
    )

    # Starting message/starting time
    t_start = tic()
    printheader(io, "Starting with $(Threads.nthreads()) threads")

    # Load image(s)
    image = @showtime(io,
        "Loading input file: $(info[:inputfile])",
        load_image(info[:inputfile]),
    )

    # Apply mask
    if !isnothing(info[:maskfile])
        @showtime(io,
            "Applying mask from file: '$(info[:maskfile])'",
            try_apply_maskfile!(image, info[:maskfile]),
        )
    elseif opts[:bet]
        @showtime(io,
            "Making and applying BET mask with args: '$(opts[:betargs])'",
            try_apply_bet!(image, opts[:betpath], opts[:betargs]),
        )
    end

    # Compute T2 distribution from input 4D multi-echo image
    if opts[:T2map]
        maps, dist = @showtime(io,
            "Running T2mapSEcorr on file: $(info[:inputfile])",
            T2mapSEcorr(image; io = io, t2map_kwargs...),
        )

        # Save T2-distribution to .mat file
        savefile = joinpath(info[:outputfolder], info[:choppedinputfile] * ".t2dist.mat")
        if !opts[:dry]
            @showtime(io,
                "Saving T2 distribution to file: $savefile",
                MAT.matwrite(savefile, Dict("dist" => dist)),
            )
        end

        # Save T2-maps to .mat file
        savefile = joinpath(info[:outputfolder], info[:choppedinputfile] * ".t2maps.mat")
        if !opts[:dry]
            @showtime(io,
                "Saving T2 parameter maps to file: $savefile",
                MAT.matwrite(savefile, maps),
            )
        end
    else
        # Input image is the T2 distribution
        dist = image
    end

    # Analyze T2 distribution to produce parameter maps
    if opts[:T2part]
        parts = @showtime(io,
            "Running T2partSEcorr",
            T2partSEcorr(dist; io = io, t2part_kwargs...),
        )

        # Save T2-parts to .mat file
        savefile = joinpath(info[:outputfolder], info[:choppedinputfile] * ".t2parts.mat")
        if !opts[:dry]
            @showtime(io,
                "Saving T2 parts maps to file: $savefile",
                MAT.matwrite(savefile, parts),
            )
        end
    end

    # Done message
    printheader(io, "Finished ($(round(toc(t_start); digits = 2)) seconds)")

    return nothing
end

function create_argparse_settings(;legacy = false, add_defaults = false)
    settings = ArgParseSettings(
        prog = "",
        fromfile_prefix_chars = "@",
        error_on_conflict = false,
        exit_after_help = true,
        # exc_handler = ArgParse.debug_handler,
    )

    @add_arg_table! settings begin
        "input"
            help = "one or more input filenames. Valid file types are limited to: $ALLOWED_FILE_SUFFIXES_STRING"
            required = true
            nargs = '+' # At least one input is required
        "--mask", "-m"
            help = "one or more mask filenames. Masks are loaded and subsequently applied to the corresponding input files via elementwise multiplication. The number of mask files must equal the number of input files. Valid file types are the same as for input files, and are limited to: $ALLOWED_FILE_SUFFIXES_STRING"
            nargs = '+' # At least one input is required
        "--output", "-o"
            help = "output directory. If not specified, output file(s) will be stored in the same location as the corresponding input file(s). Outputs are stored with the same basename as inputs and additional suffixes; see --T2map and --T2part"
        "--T2map"
            help = "call T2mapSEcorr to compute T2 distributions from 4D multi spin-echo input images. T2 distributions and T2 maps produced by T2mapSEcorr are saved as MAT files with extensions .t2dist.mat and .t2maps.mat"
            action = :store_true
        "--T2part"
            help = "call T2partSEcorr to analyze 4D T2 distributions to produce parameter maps. If --T2map is also passed, input 4D arrays are interpreted as multi spin-echo images and T2 distributions are first computed by T2mapSEcorr. If only --T2part is passed, input 4D arrays are interpreted as T2 distributions and only T2partSEcorr is called. Output T2 parts are saved as a MAT file with extension .t2parts.mat"
            action = :store_true
        "--quiet", "-q"
            help = "suppress printing to the terminal. Note: all terminal output, including errors and warnings, is still printed to the log file"
            action = :store_true
        "--dry"
            help = "execute dry run of processing without saving any results"
            action = :store_true
        "--legacy"
            help = "use legacy settings and algorithms from the original MATLAB version. This ensures that the exact same T2-distributions and T2-parts will be produced as those from MATLAB (to machine precision). Note that execution time will be much slower."
            action = :store_true
    end

    add_arg_group!(settings,
        "T2mapSEcorr/T2partSEcorr arguments",
        "internal arguments",
    )
    t2map_opts = T2mapOptions{Float64}(MatrixSize = (1,1,1), TE = 10e-3, nTE = 32, T2Range = (10e-3, 2.0), nT2 = 40, legacy = legacy)
    t2part_opts = T2partOptions{Float64}(MatrixSize = (1,1,1), nT2 = 40, T2Range = (10e-3, 2.0), SPWin = (10e-3, 40e-3), MPWin = (40e-3, 200.0e-3), legacy = legacy)
    opts_args = sorted_arg_table_entries(t2map_opts, t2part_opts; add_defaults = add_defaults)
    add_arg_table!(settings, opts_args...)

    add_arg_group!(settings,
        "BET arguments",
        "arguments for mask generation using BET",
    )
    @add_arg_table! settings begin
        "--bet"
            help = "use the BET brain extraction tool from the FSL library of analyis tools to automatically create a binary brain mask. Only voxels within the binary mask will be analyzed. Note that if a mask is passed explicitly with the --mask flag, this mask will be used and --bet will be ignored."
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

function sorted_arg_table_entries(
        @nospecialize(t2map_opts),
        @nospecialize(t2part_opts);
        add_defaults = false
    )
    required_fields = (:MatrixSize, :nTE, :nT2, :TE, :T2Range, :SPWin, :MPWin)
    fields, types, values = Symbol[], Type[], Any[]
    for o in [t2map_opts, t2part_opts], (f,T) in zip(fieldnames(typeof(o)), fieldtypes(typeof(o)))
        push!(fields, f)
        push!(types, T)
        if f ∈ required_fields
            push!(values, nothing) # automatically determined or required parameters
        else
            push!(values, getfield(o, f)) # default parameters
        end
    end

    defaults = collect(zip(fields, values, types)) |> triplets -> vcat(
        [triplets[findfirst(first.(triplets) .== f)] for f in required_fields], # required fields first
        sort(filter(t -> t[1] ∉ required_fields, triplets); by = tup -> uppercase(string(first(tup)))), # followed by remaining fields alphabetically
    )

    args = []
    for (f, v, T) in defaults
        (f === :vTEparam || f == :legacy) && continue # vTEparam not implemented; legacy grouped with main ArgParse settings
        push!(args, "--" * string(f))
        if T <: Bool
            if add_defaults
                push!(args, Dict{Symbol,Any}(:action => ifelse(v, :store_false, :store_true)))
            else
                push!(args, Dict{Symbol,Any}(:action => :store_const, :constant => !v, :default => nothing))
            end
        elseif T <: Union{<:Tuple, Nothing}
            nargs = length(fieldtypes(_get_tuple_type(T)))
            props = Dict{Symbol,Any}(
                :nargs   => nargs,
                :default => (add_defaults && !isnothing(v)) ? [v...] : nothing,
            )
            push!(args, props)
        else
            push!(args, Dict{Symbol,Any}(:default => add_defaults ? v : nothing))
        end
        if f === :MatrixSize
            args[end][:help] = "Required parameter; inferred from first three dimensions of input image"
        elseif f === :nTE
            args[end][:help] = "Required parameter; inferred from fourth dimension of input image if --T2map is passed"
        elseif f === :nT2
            args[end][:help] = "Required parameter; inferred from fourth dimension of input image if --T2part (and not --T2map) is passed"
        elseif f === :TE
            args[end][:help] = "Required parameter when --T2map is passed"
        elseif f === :T2Range
            args[end][:help] = "Required parameter when --T2map or --T2part is passed"
        elseif f === :SPWin || f === :MPWin
            args[end][:help] = "Required parameter when --T2part is passed"
        end
    end

    return args
end

function get_parsed_args_subset(
        @nospecialize(opts), # ::Dict{Symbol,Any}
        @nospecialize(subset_fieldtypes), # ::Dict{Symbol,Type}
    )
    kwargs = deepcopy(opts)
    for (k,v) in kwargs
        if k ∉ keys(subset_fieldtypes)
            delete!(kwargs, k)
            continue
        end
        if isnothing(v)
            # Nothing values are always default; skip them
            delete!(kwargs, k)
        elseif v isa AbstractString
            # Parse v to appropriate type, as determined by subset_fieldtypes
            T = subset_fieldtypes[k]
            if !(T <: AbstractString)
                kwargs[k] = _parse_or_convert(_strip_union_nothing(T), v)
            end
        elseif v isa AbstractVector
            if isempty(v)
                # Empty vectors correspond to settings not set by user
                delete!(kwargs, k)
            else
                # Convert AbstractVector v to appropriate Tuple type
                T = _get_tuple_type(subset_fieldtypes[k])
                kwargs[k] = tuple(_parse_or_convert.(fieldtypes(T), v)...) # each element should be individually parsed
            end
        end
    end
    return kwargs
end

function get_file_info(@nospecialize(opts))
    @unpack input, output, mask = opts

    # Read in input files
    inputfiles = String[path for path in input if is_allowed_suffix(path)]

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

    # Get output folders
    outputfolders = if isnothing(output)
        dirname.(inputfiles)
    else
        [output for _ in 1:length(inputfiles)]
    end

    # Get mask files
    maskfiles = if isempty(mask)
        fill(nothing, length(inputfiles))
    elseif length(mask) == length(inputfiles)
        String.(mask)
    else
        error("Number of mask files passed does not equal the number of input image files passed")
    end

    # Create info dictionaries
    info = Dict{Symbol, Union{String, Nothing}}[]
    for (inputfile, outputfolder, maskfile) in zip(inputfiles, outputfolders, maskfiles)
        d = eltype(info)(
            :inputfile => inputfile,
            :outputfolder => outputfolder,
            :maskfile => maskfile,
            :choppedinputfile => chop_allowed_suffix(basename(inputfile)),
        )
        push!(info, d)
    end

    return info
end

function load_image(filename, ::Val{dim} = Val(4)) where {dim}
    image = if maybe_get_suffix(filename) == ".mat"
        # Load first `dim`-dimensional array which is found, or throw an error if none are found
        data = MAT.matread(filename)
        key = findfirst(x -> x isa AbstractArray{T,dim} where {T}, data)
        if isnothing(key)
            error("No 4D array was found in the input file: $filename")
        end
        data[key]

    elseif maybe_get_suffix(filename) ∈ (".nii", ".nii.gz")
        # Check slope field; if scl_slope == 0, data is not scaled and raw data should be returned
        #   See e.g. https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/scl_slopeinter.html
        # Loaded data is coerced to a `dim`-dimensional array
        data = NIfTI.niread(filename)
        if data.header.scl_slope == 0
            data.raw[ntuple(_ -> Colon(), dim)...] # Return raw data
        else
            data[ntuple(_ -> Colon(), dim)...] # Return scaled data (getindex from the NIfTI package handles scaling)
        end

    elseif maybe_get_suffix(filename) ∈ (".par", ".xml", ".rec")
        # Load PAR/REC or XML/REC file, coercing resulting data into a `dim`-dimensional array
        _, data = ParXRec.parxrec(filename)
        data[ntuple(_ -> Colon(), dim)...]

    else
        error("Currently, only $ALLOWED_FILE_SUFFIXES_STRING files are supported")
    end

    # Currently, the pipeline is ~twice as fast on Float64 arrays than Float32 arrays (unclear why).
    # However, the MATLAB toolbox converts images to double as well, so here we simply do the same
    image = convert(Array{Float64,dim}, image)

    return image
end

function try_apply_maskfile!(image, maskfile)
    try
        image .*= load_image(maskfile, Val(3))
    catch e
        @warn "Error while loading mask file: $maskfile"
        @warn sprint(showerror, e, catch_backtrace())
    end
    return image
end

function make_bet_mask(image::Array{T,3}, betpath, betargs) where {T}
    # Split betargs, and ensure that "-m" (make binary mask) is among args
    args = convert(Vector{String}, filter!(!isempty, split(betargs, " ")))
    if "-m" ∉ args
        push!(args, "-m")
    end

    # Create mask using BET and return mask
    mask = mktempdir() do temppath
        tempbase = basename(tempname())
        nifti_imagefile = joinpath(temppath, tempbase * ".nii")
        nifti_maskfile = joinpath(temppath, tempbase * ".bet")
        NIfTI.niwrite(nifti_imagefile, NIfTI.NIVolume(image)) # create nifti file for bet
        run(Cmd([
            betpath;
            nifti_imagefile;
            nifti_maskfile;
            args
        ]))
        # BET appends "_mask" and ".nii.gz" to output file name.
        # Find this file, ensure it is unique, then load and return it
        bet_maskfiles = filter!(file -> startswith(file, tempbase * ".bet_mask"), readdir(temppath))
        @assert length(bet_maskfiles) == 1 # ensure unique; this should never be false using a temp filename
        load_image(joinpath(temppath, bet_maskfiles[1]), Val(3))
    end

    return mask
end
make_bet_mask(image::Array{T,4}, args...; kwargs...) where {T} =
    make_bet_mask(image[:,:,:,1], args...; kwargs...) # use first echo

function try_apply_bet!(image, betpath, betargs)
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
_maybe_get_first(f, xs) = findfirst(f, xs) |> I -> isnothing(I) ? nothing : xs[I]

maybe_get_suffix(filename) = _maybe_get_first(ext -> endswith(lowercase(filename), ext), ALLOWED_FILE_SUFFIXES) # case-insensitive
is_allowed_suffix(filename) = !isnothing(maybe_get_suffix(filename))

function chop_allowed_suffix(filename::AbstractString)
    ext = maybe_get_suffix(filename)
    if !isnothing(ext)
        return filename[1:end-length(ext)]
    else
        error("Currently only $ALLOWED_FILE_SUFFIXES_STRING file types are supported")
    end
end

####
#### Logging
####

# https://discourse.julialang.org/t/write-to-file-and-stdout/35042/3
struct Tee{TIO <: Tuple} <: IO
    streams::TIO
end
Tee(streams::IO...) = Tee(streams)
Base.flush(t::Tee) = do_tee(t, io -> nothing)
function do_tee(t::Tee, f, args...; kwargs...)
    for io in t.streams
        f(io, args...; kwargs...)
        flush(io)
    end
end
for f in [:write, :print, :println, :printstyled], T in [Any, Array, Char, Union{SubString{String},String}]
    @eval Base.$f(t::Tee, x::$T; kwargs...) = do_tee(t, $f, x; kwargs...)
end

function tee_capture(f; logfile = tempname(), suppress_terminal = false, suppress_logfile = false)
    open(suppress_logfile ? tempname() : logfile, "w+") do io
        io = Tee(suppress_terminal ? devnull : stderr, io)
        logger = ConsoleLogger(io)
        with_logger(logger) do
            f(io)
        end
    end
end

# https://discourse.julialang.org/t/redirect-stdout-and-stderr/13424/3
function redirect_to_files(f, outfile, errfile)
    open(outfile, "w") do out
        open(errfile, "w") do err
            redirect_stdout(out) do
                redirect_stderr(err) do
                    f()
                end
            end
        end
    end
end
redirect_to_tempfiles(f) = redirect_to_files(f, tempname() * ".log", tempname() * ".err")

####
#### Global constants
####

const ALLOWED_FILE_SUFFIXES = (".mat", ".nii", ".nii.gz", ".par", ".xml", ".rec")
const ALLOWED_FILE_SUFFIXES_STRING = join(ALLOWED_FILE_SUFFIXES, ", ", ", and ")

const T2MAP_FIELDTYPES = Dict{Symbol,Type}(fieldnames(T2mapOptions{Float64}) .=> fieldtypes(T2mapOptions{Float64}))
const T2PART_FIELDTYPES = Dict{Symbol,Type}(fieldnames(T2partOptions{Float64}) .=> fieldtypes(T2partOptions{Float64}))

const ARGPARSE_SETTINGS = create_argparse_settings(legacy = false, add_defaults = false)
const ARGPARSE_SETTINGS_DECAES = create_argparse_settings(legacy = false, add_defaults = true)
const ARGPARSE_SETTINGS_LEGACY = create_argparse_settings(legacy = true, add_defaults = true)

end # module Main
