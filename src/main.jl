####
#### CLI settings
####

const ALLOWED_FILE_SUFFIXES = (".mat", ".nii", ".nii.gz", ".par", ".xml", ".rec")
const ALLOWED_FILE_SUFFIXES_STRING = join(ALLOWED_FILE_SUFFIXES, ", ", ", and ")

const CLI_SETTINGS = ArgParseSettings(;
    prog = "decaes",
    # description = "DECAES v" * string(VERSION),
    fromfile_prefix_chars = "@",
    error_on_conflict = false,
    exit_after_help = false,
    exc_handler = ArgParse.debug_handler,
    add_version = true,
    version = string(VERSION),
)

add_arg_table!(CLI_SETTINGS,
    "input",
    Dict(
        :nargs => '*', # At least one input is required, but this is checked internally
        :arg_type => String,
        :help => "one or more input filenames. Valid file types are limited to: $ALLOWED_FILE_SUFFIXES_STRING",
    ),
    ["--mask", "-m"],
    Dict(
        :nargs => '+', # If --mask is passed, at least one input is required
        :arg_type => String,
        :help => "one or more mask filenames. Masks are loaded and subsequently applied to the corresponding input files via elementwise multiplication. The number of mask files must equal the number of input files. Valid file types are the same as for input files, and are limited to: $ALLOWED_FILE_SUFFIXES_STRING",
    ),
    ["--output", "-o"],
    Dict(
        :nargs => '+', # If --output is passed, at least one input is required
        :arg_type => String,
        :help => "one or more output directories. If not specified, output file(s) will be stored in the same location as the corresponding input file(s). If one folder is passed, all output files from all processed images will be stored in the same folder. Otherwise, the number of output folders must equal the number of input files. Outputs are stored with the same basename as the input files with additional suffixes; see --T2map and --T2part",
    ),
    "--T2map",
    Dict(
        :action => :store_true,
        :help => "call T2mapSEcorr to compute T2 distributions from 4D multi-spin echo input images. Outputs use the suffixes .t2dist and .t2maps; see --OutputFormat",
    ),
    "--T2part",
    Dict(
        :action => :store_true,
        :help => "call T2partSEcorr to analyze 4D T2 distributions to produce parameter maps. If --T2map is also passed, input 4D arrays are interpreted as multi-spin echo images and T2 distributions are first computed by T2mapSEcorr. If only --T2part is passed, input 4D arrays are interpreted as T2 distributions and only T2partSEcorr is called. Outputs use the suffix .t2parts; see --OutputFormat",
    ),
    ["--quiet", "-q"],
    Dict(
        :action => :store_true,
        :help => "suppress printing to the terminal. Note: all terminal outputs, including errors and warnings, are still printed to the log file",
    ),
    "--dry",
    Dict(
        :action => :store_true,
        :help => "execute dry run of processing without saving any results",
    ),
)

add_arg_group!(CLI_SETTINGS,
    "T2map/T2part required parameters",
    :t2_map_part_required,
)

add_arg_table!(CLI_SETTINGS,
    "--MatrixSize",
    Dict(
        :nargs => 3,
        :arg_type => Int,
        :help => "matrix size of the magnitude image. Inferred automatically as the first three dimensions of the input 4D image",
        :group => :t2_map_part_required,
    ),
    "--nTE",
    Dict(
        :arg_type => Int,
        :help => "number of echoes of the magnitude image. Inferred automatically as the last dimension of the input 4D image",
        :group => :t2_map_part_required,
    ),
    "--TE",
    Dict(
        :arg_type => Float64,
        :help => "inter-echo spacing. Required when --T2map is passed. (units: time, must match --T1 and --T2Range)",
        :group => :t2_map_part_required,
    ),
    "--nT2",
    Dict(
        :arg_type => Int,
        :help => "number of T2 components used in the multi-exponential analysis. Required when --T2map is passed. Inferred from fourth dimension of input image if only --T2part and not --T2map is passed",
        :group => :t2_map_part_required,
    ),
    "--T2Range",
    Dict(
        :nargs => 2,
        :arg_type => Float64,
        :help => "minimum and maximum T2 values. T2 components are logarithmically spaced between these bounds. Required parameter. (units: time, must match --TE)",
        :group => :t2_map_part_required,
    ),
    "--SPWin",
    Dict(
        :nargs => 2,
        :arg_type => Float64,
        :help => "minimum and maximum T2 values of the short peak window. Required parameter when --T2part is passed. (units: time, must match --T2Range)",
        :group => :t2_map_part_required,
    ),
    "--MPWin",
    Dict(
        :nargs => 2,
        :arg_type => Float64,
        :help => "minimum and maximum T2 values of the middle peak window. Required parameter when --T2part is passed. (units: time, must match --T2Range)",
        :group => :t2_map_part_required,
    ),
    "--Reg",
    Dict(
        :arg_type => String,
        :help => "method used for choosing the regularization parameter. One of \"gcv\", \"lcurve\", \"reginska\", \"chi2\", \"mdp\", or \"none\". These flags correspond to generalized cross-validation, the L-curve method, Reginska's minimum-product criterion, the chi-squared method, Morozov's discrepancy principle, and zero regularization. Required parameter",
        :group => :t2_map_part_required,
    ),
    "--RegNorm",
    Dict(
        :arg_type => String,
        :help => "norm of the regularization penalty term. One of \"l2\" or \"l1\", penalizing mu^2 * ||x||_2^2 or mu * ||x||_1, respectively. --Reg=\"gcv\" supports only \"l2\". (default: \"l2\")",
        :group => :t2_map_part_required,
    ),
    "--RegParams",
    Dict(
        :nargs => '+', # If --RegParams is passed, at least one input is required
        :arg_type => Float64,
        :help => "parameters for the regularization method chosen via --Reg. Required parameter if --Reg=\"chi2\" or --Reg=\"mdp\"",
        :group => :t2_map_part_required,
    ),
    "--Chi2Factor",
    Dict(
        :arg_type => Float64,
        :help => "if --Reg=\"chi2\", the T2 distribution is regularized such that the chi^2 goodness of fit is increased by a multiplicative factor --Chi2Factor relative to the unregularized solution. Required parameter when --Reg=\"chi2\". Note: this flag is now deprecated and will be removed in future releases; use --RegParams instead",
        :group => :t2_map_part_required,
    ),
)

add_arg_group!(CLI_SETTINGS,
    "T2map/T2part optional parameters",
    :t2_map_part_optional,
)

add_arg_table!(CLI_SETTINGS,
    "--T1",
    Dict(
        :arg_type => Float64,
        :help => "assumed value of longitudinal T1 relaxation. (default: 1.0) (units: time, must match --TE)",
        :group => :t2_map_part_optional,
    ),
    "--Sigmoid",
    Dict(
        :arg_type => Float64,
        :help => "replace the hard upper limit cutoff time of the short peak window, SPWin[2], with a smoothed sigmoidal cutoff function 'σ' scaled and shifted such that σ(SPWin[2] +/- Sigmoid) = 0.5 -/+ 0.4. Sigmoid is the time scale of the smoothing. (units: time, must match --T2Range)",
        :group => :t2_map_part_optional,
    ),
    "--Threshold",
    Dict(
        :arg_type => Float64,
        :help => "first echo intensity cutoff for empty voxels. Processing is skipped for voxels with intensity <= --Threshold. (default: 0.0) (units: signal magnitude)",
        :group => :t2_map_part_optional,
    ),
)

add_arg_group!(CLI_SETTINGS,
    "B1 correction and stimulated echo correction",
    :B1_SE_corr,
)

add_arg_table!(CLI_SETTINGS,
    "--B1map",
    Dict(
        :nargs => '+', # If --B1map is passed, at least one input is required
        :arg_type => String,
        :help => "one or more B1 map filenames. The B1 maps must have the same matrix sizes as the corresponding images, and are assumed to represent flip angles in units of degrees. The number of B1 map files must equal the number of input files. Valid file types are the same as for input files, and are limited to: $ALLOWED_FILE_SUFFIXES_STRING. (units: degrees)",
        :group => :B1_SE_corr,
    ),
    "--nRefAngles",
    Dict(
        :arg_type => Int,
        :help => "maximum number of flip angles spanning [--MinRefAngle, 180] checked during local refocusing flip angle estimation. (default: 500)",
        :group => :B1_SE_corr,
    ),
    "--nRefAnglesMin",
    Dict(
        :arg_type => Int,
        :help => "initial number of angles evaluated before refinement near likely optima during local refocusing flip angle estimation. (default: 15)",
        :group => :B1_SE_corr,
    ),
    "--MinRefAngle",
    Dict(
        :arg_type => Float64,
        :help => "minimum refocusing angle estimate allowed. (default: 90.0) (units: degrees)",
        :group => :B1_SE_corr,
    ),
    "--SetFlipAngle",
    Dict(
        :arg_type => Float64,
        :help => "to skip B1 inhomogeneity correction, use --SetFlipAngle to assume a fixed refocusing flip angle for all voxels. (units: degrees)",
        :group => :B1_SE_corr,
    ),
    "--RefConAngle",
    Dict(
        :arg_type => Float64,
        :help => "refocusing pulse control angle. The sequence of flip angles used within the extended phase graph algorithm to perform stimulated echo correction is (90, 180, β, β, ..., β), where β is the refocusing pulse control angle. For typical multi-spin echo sequences this parameter should not be changed. (default: 180.0) (units: degrees)",
        :group => :B1_SE_corr,
    ),
)

add_arg_group!(CLI_SETTINGS,
    "Additional save options",
    :save_opts,
)

add_arg_table!(CLI_SETTINGS,
    "--OutputFormat",
    Dict(
        :arg_type => String,
        :default => "mat",
        :help => "format of image outputs. One of \"mat\" or \"nii\". NIfTI outputs are gzipped and inherit the header of a NIfTI input image; non-image data remain in a MAT file. (default: \"mat\")",
        :group => :save_opts,
    ),
    "--NoSaveT2Dist",
    Dict(
        :action => :store_true,
        :help => "skip saving the 4D T2 distribution computed by --T2map, which is large for typical image sizes",
        :group => :save_opts,
    ),
    "--SaveDecayCurve",
    Dict(
        :action => :store_true,
        :help => "include a 4D array of the time domain decay curves resulting from the NNLS fits in the output maps dictionary",
        :group => :save_opts,
    ),
    "--SaveNNLSBasis",
    Dict(
        :action => :store_true,
        :help => "include a 5D (or 2D if --SetFlipAngle is used) array of NNLS basis matrices in the output maps dictionary. Note: this 5D array is extremely large for typical image sizes; in most cases, this flag should only be set when debugging small images",
        :group => :save_opts,
    ),
    "--SaveRegParam",
    Dict(
        :action => :store_true,
        :help => "include 3D arrays of resulting regularization parameters and χ² factors in the output maps dictionary",
        :group => :save_opts,
    ),
    "--SaveResidualNorm",
    Dict(
        :action => :store_true,
        :help => "include a 3D array of the l2-norms of the residuals from the NNLS fits in the output maps dictionary",
        :group => :save_opts,
    ),
)

add_arg_group!(CLI_SETTINGS,
    "BET arguments",
    :bet_args,
)

add_arg_table!(CLI_SETTINGS,
    "--bet",
    Dict(
        :action => :store_true,
        :help => "use the BET brain extraction tool from the FSL library of analysis tools to automatically create a binary brain mask. Only voxels within the binary mask will be analyzed. Note that if a mask is passed explicitly with the --mask flag, this mask will be used and the --bet flag will be ignored",
        :group => :bet_args,
    ),
    "--betargs",
    Dict(
        :arg_type => String,
        :default => "-m -n -f 0.25 -R",
        :help => "BET command line interface arguments. Must be passed as a single string with arguments separated by commas or spaces, e.g. \"-m,-n\". The flag \"-m\" indicates that a binary mask should be computed, and therefore will be added to the list of arguments if not provided",
        :group => :bet_args,
    ),
    "--betpath",
    Dict(
        :arg_type => String,
        :default => "bet",
        :help => "path to BET executable",
        :group => :bet_args,
    ),
)

"""
    main(command_line_args::Vector{String} = ARGS)

Entry point function for command line interface, parsing the command line arguments `ARGS` and subsequently calling one or both of `T2mapSEcorr` and `T2partSEcorr` with the parsed settings.
See the [Arguments](@ref) section for available options.

See also:

  - [`T2mapSEcorr`](@ref)
  - [`T2partSEcorr`](@ref)
"""
main(args::Vector{String} = ARGS) = run_main(args)

function run_main(command_line_args::Vector{String})

    # Parse command line arguments
    opts = parse_cli(command_line_args)
    opts === nothing && return nothing # exit was triggered; return nothing instead of exit(0)

    # Get input file list and output folder list
    for file_info in get_file_infos(opts)
        # Make output path
        if !opts[:dry]
            mkpath(file_info[:outputfolder])
        end

        # Save settings files
        if !opts[:dry]
            map(filter(s -> startswith(s, "@"), command_line_args)) do settingsfile
                src = settingsfile[2:end] # drop "@" character
                dst = joinpath(file_info[:outputfolder], file_info[:choppedinputfile] * "." * basename(src))
                cp(src, dst; force = true)
                return nothing
            end
        end

        # Main processing
        tee_capture(;
            logfile = joinpath(file_info[:outputfolder], file_info[:choppedinputfile] * ".log"),
            suppress_terminal = opts[:quiet],
            suppress_logfile = opts[:dry],
        ) do
            try
                run_main(file_info, opts)
            catch e
                @warn "Error during processing of file: $(file_info[:inputfile])"
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end

    return nothing
end

function run_main(file_info::Dict{Symbol, Any}, opts::Dict{Symbol, Any})

    # Starting message/starting time
    t_start = tic()
    @info "Starting DECAES v$(VERSION) using Julia v$(Base.VERSION) with $(Threads.nthreads()) threads"

    # Load image(s)
    image = @showtime("Loading input file: $(file_info[:inputfile])", load_image(file_info[:inputfile]))
    output_header = !opts[:dry] && opts[:OutputFormat] == "nii" ? nifti_output_header(file_info[:inputfile]) : nothing

    # Apply mask
    if file_info[:maskfile] !== nothing
        @showtime("Applying mask from file: $(file_info[:maskfile])", try_apply_maskfile!(image, file_info[:maskfile]))
    elseif opts[:bet]
        @showtime("Making and applying BET mask with args: $(join(opts[:betargs], " "))", try_apply_bet!(image, opts[:betpath], opts[:betargs]))
    end

    # Compute T2 distribution from input 4D multi-echo image
    if opts[:T2map]
        t2map_opts = t2map_options(image, opts)
        t2map_maps = T2Maps(t2map_opts)
        t2map_dist = T2Distributions(t2map_opts)

        # Load B1 map
        if file_info[:B1mapfile] !== nothing
            @showtime("Loading B1 map from file: $(file_info[:B1mapfile])", try_load_B1mapfile!(t2map_maps, file_info[:B1mapfile]))
        end

        @showtime("Running T2mapSEcorr on file: $(file_info[:inputfile])", T2mapSEcorr!(t2map_maps, t2map_dist, image, t2map_opts))
        maps, dist = convert(Dict{String, Any}, t2map_maps), parent(t2map_dist)

        # Save T2-distribution
        if !opts[:dry] && !opts[:NoSaveT2Dist]
            save_outputs(file_info, opts, output_header, ".t2dist", "T2 distribution", "dist", dist)
        end

        # Save T2-maps
        if !opts[:dry]
            save_outputs(file_info, opts, output_header, ".t2maps", "T2 parameter maps", maps, imagelike_fieldnames(t2map_maps))
        end
    else
        # Input image is the T2 distribution
        dist = image
    end

    # Analyze T2 distribution to produce parameter maps
    if opts[:T2part]
        t2part_opts = t2part_options(dist, opts)
        t2part_parts = T2Parts(t2part_opts)

        @showtime("Running T2partSEcorr", T2partSEcorr!(t2part_parts, dist, t2part_opts))
        parts = convert(Dict{String, Any}, t2part_parts)

        # Save T2-parts
        if !opts[:dry]
            save_outputs(file_info, opts, output_header, ".t2parts", "T2 parts maps", parts, imagelike_fieldnames(t2part_parts))
        end
    end

    # Done message
    @info "Finished ($(round(toc(t_start); digits = 2)) seconds)"

    return nothing
end

####
#### Helper functions
####

function parse_cli(args)
    opts = parse_args(args, CLI_SETTINGS; as_symbols = true)
    if opts !== nothing
        opts = handle_cli_deprecations!(opts)
        opts = verify_cli_args!(opts)
        opts = clean_cli_args!(opts)
    end
    return opts
end

function handle_cli_deprecations!(opts)
    if get(opts, :Chi2Factor, nothing) !== nothing
        if get(opts, :RegParams, Float64[]) != Float64[]
            error_conflicted_flags(:Chi2Factor, :RegParams)
        else
            warn_deprecated_renamed(:Chi2Factor, :RegParams)
            opts[:RegParams] = Float64[opts[:Chi2Factor]]
        end
    end
    delete!(opts, :Chi2Factor)

    return opts
end
warn_deprecated_future_removed(oldflag) = @warn "The flag --$oldflag is deprecated and will be removed in future releases."
warn_deprecated_renamed(oldflag, newflag) = @warn "The flag --$oldflag is deprecated and will be removed in future releases; use --$newflag instead."
error_conflicted_flags(oldflag, newflag) = error("The flag --$newflag and the deprecated flag --$oldflag were both passed; use --$newflag only.")

function verify_cli_args!(opts)
    # Verify argument interdependencies which can't be enforced by ArgParse
    if !(opts[:T2map] || opts[:T2part])
        error("At least one of --T2map or --T2part must be passed")
    end
    if opts[:OutputFormat] ∉ ("mat", "nii")
        error("--OutputFormat must be one of \"mat\" or \"nii\", but --OutputFormat=$(repr(opts[:OutputFormat])) was passed")
    end
    return opts
end

function clean_cli_args!(opts)
    # Preprocess arguments for use
    if opts[:betargs] isa String
        opts[:betargs] = clean_bet_args(opts[:betargs])::Vector{String}
    end

    if opts[:Reg] == "chi2"
        @assert length(opts[:RegParams]) == 1 "Must set chi2 factor via --RegParams when --Reg=\"chi2\""
        opts[:Chi2Factor] = only(opts[:RegParams])
    elseif opts[:Reg] == "mdp"
        @assert length(opts[:RegParams]) == 1 "Must set noise level via --RegParams when --Reg=\"mdp\""
        opts[:NoiseLevel] = only(opts[:RegParams])
    end

    return opts
end

function t2map_options(image::Array, opts::Dict{Symbol, Any})
    fields = fieldsof(T2mapOptions, Set)
    kwargs = Dict{Symbol, Any}()
    for (k, v) in opts
        (v === nothing) && continue # filter unset cli args
        (v isa AbstractVector && isempty(v)) && continue # filter unset cli args (empty vectors are unset cli varargs)
        (k ∉ fields) && continue # filter T2mapOptions fields
        kwargs[k] = v isa AbstractVector ? tuple(v...) : v
    end
    return T2mapOptions(image; kwargs...)
end

function t2part_options(dist::Array, opts::Dict{Symbol, Any})
    fields = fieldsof(T2partOptions, Set)
    kwargs = Dict{Symbol, Any}()
    for (k, v) in opts
        (v === nothing) && continue # filter unset cli args
        (v isa AbstractVector && isempty(v)) && continue # filter unset cli args (empty vectors are unset cli varargs)
        (k === :nT2 && !opts[:T2map]) && continue # nT2 must be explicitly passed, unless not performing T2-mapping, in which case it is inferred from `dist`
        (k ∉ fields) && continue # filter T2mapOptions fields
        kwargs[k] = v isa AbstractVector ? tuple(v...) : v
    end
    return T2partOptions(dist; kwargs...)
end

function get_file_infos(opts::Dict{Symbol, Any})
    # Read in input files
    input = opts[:input]
    @assert !isempty(input) "At least one input file is required"
    inputfiles = String[path for path in input if is_allowed_suffix(path)]

    if isempty(inputfiles)
        msg = if !isempty(input) && isfile(input[1])
            "No valid file types were found for processing, but a file name was passed.\n" *
            "Perhaps you meant to prepend an '@' character to a settings file, e.g. '@$(input[1])'?\n" *
            "If not, note that only $ALLOWED_FILE_SUFFIXES_STRING file types are supported"
        else
            "No valid files were found for processing. Note that currently only $ALLOWED_FILE_SUFFIXES_STRING file types are supported"
        end
        error(msg)
    end

    # Get output folders
    output = opts[:output]
    outputfolders = if isempty(output)
        dirname.(inputfiles) # store results in folder containing corresponding input file
    elseif length(output) == length(inputfiles)
        String.(output) # store results from each input file in the respective output folder
    elseif length(output) == 1
        fill(String(only(output)), length(inputfiles)) # store all results in single folder
    else
        error("Incorrect number of output files passed ($(length(output))); must pass either 1 output folder (all results are stored in this folder), or the same number of output folders as input image files ($(length(inputfiles)))")
    end

    # Get mask files
    mask = opts[:mask]
    maskfiles = if isempty(mask)
        fill(nothing, length(inputfiles)) # no mask passed
    elseif length(mask) == length(inputfiles)
        String.(mask) # one mask passed for each input file
    else
        error("Number of mask files passed ($(length(mask))) does not equal the number of input image files passed ($(length(inputfiles))")
    end

    # Get B1 map files
    B1map = opts[:B1map]
    B1mapfiles = if isempty(B1map)
        fill(nothing, length(inputfiles)) # no B1map passed
    elseif length(B1map) == length(inputfiles)
        @assert opts[:SetFlipAngle] === nothing "Cannot set a fixed flip angle using --SetFlipAngle when passing B1 maps using --B1map"
        String.(B1map) # one B1map passed for each input file
    else
        error("Number of B1 map files passed ($(length(B1map))) does not equal the number of input image files passed ($(length(inputfiles))")
    end

    # Create file_info dictionaries
    file_info = Dict{Symbol, Any}[]
    for (inputfile, outputfolder, maskfile, B1mapfile) in zip(inputfiles, outputfolders, maskfiles, B1mapfiles)
        d = Dict{Symbol, Any}(
            :inputfile => inputfile,
            :outputfolder => outputfolder,
            :maskfile => maskfile,
            :B1mapfile => B1mapfile,
            :choppedinputfile => chop_allowed_suffix(basename(inputfile)),
        )
        push!(file_info, d)
    end

    return file_info
end

# Save either all outputs as a MAT file, or selected image outputs as NIfTI files and the rest as a MAT file.
function save_outputs(file_info, opts, header, suffix, description, maps::Dict{String, Any}, nifti_fields)
    basefile = joinpath(file_info[:outputfolder], file_info[:choppedinputfile] * suffix)
    if opts[:OutputFormat] == "mat"
        @showtime("Saving $description to file: $basefile.mat", MAT.matwrite("$basefile.mat", maps))
    else
        nifti_names = string.(nifti_fields)
        nifti_maps = Dict(name => maps[name] for name in nifti_names)
        metadata = Dict(name => data for (name, data) in maps if name ∉ nifti_names)
        @showtime("Saving $description to files: $basefile.<name>.nii.gz", save_nifti_outputs(basefile, nifti_maps, header))
        isempty(metadata) || @showtime("Saving metadata to file: $basefile.meta.mat", MAT.matwrite("$basefile.meta.mat", metadata))
    end
    return nothing
end

# Save a single output array, named `name` when saved as a MAT file.
function save_outputs(file_info, opts, header, suffix, description, name::String, data::AbstractArray)
    basefile = joinpath(file_info[:outputfolder], file_info[:choppedinputfile] * suffix)
    if opts[:OutputFormat] == "mat"
        @showtime("Saving $description to file: $basefile.mat", MAT.matwrite("$basefile.mat", Dict{String, Any}(name => data)))
    else
        @showtime("Saving $description to file: $basefile.nii.gz", save_nifti("$basefile.nii.gz", data, header))
    end
    return nothing
end

function save_nifti_outputs(basefile, maps, header)
    for name in sort!(collect(keys(maps)))
        save_nifti("$basefile.$name.nii.gz", maps[name], header)
    end
    return nothing
end

# Outputs inherit the input header, and therefore its voxel size and orientation, when their spatial dimensions match the input image.
function save_nifti(savefile, data, header)
    inherit = header !== nothing && ndims(data) >= 3 && ntuple(i -> size(data, i), 3) == ntuple(i -> Int(header.dim[i+1]), 3)
    vol = inherit ? NIfTI.NIVolume(deepcopy(header), data) : NIfTI.NIVolume(data)
    reset_nifti_output_header!(vol.header)
    return NIfTI.niwrite(savefile, vol)
end

function nifti_output_header(inputfile)
    maybe_get_suffix(inputfile) ∈ (".nii", ".nii.gz") || return nothing
    io = NIfTI.niopen(inputfile, "r")
    try
        return first(NIfTI.read_header(io))
    finally
        close(io)
    end
end

function reset_nifti_output_header!(header)
    # Preserve spatial geometry and reset metadata specific to the input values or nonspatial dimensions.
    qfac, Δx, Δy, Δz = header.pixdim[1], header.pixdim[2], header.pixdim[3], header.pixdim[4]
    header.pixdim = (ifelse(iszero(qfac), oneunit(qfac), qfac), Δx, Δy, Δz, oneunit(qfac), zero(qfac), zero(qfac), zero(qfac))
    header.xyzt_units &= 0x07
    header.scl_slope, header.scl_inter = 1, 0 # outputs are saved unscaled, having been scaled at load time
    header.cal_min, header.cal_max = 0, 0
    header.intent_p1, header.intent_p2, header.intent_p3, header.intent_code = 0, 0, 0, 0
    header.intent_name = ntuple(_ -> 0x00, length(header.intent_name))
    header.slice_start, header.slice_end, header.slice_code, header.slice_duration, header.toffset = 0, 0, 0, 0, 0
    return header
end

function load_image(filename, ::Val{N}) where {N}
    if maybe_get_suffix(filename) == ".mat"
        # Load first `N`-dimensional array which is found, or throw an error if none are found
        data = MAT.matread(filename)
        array_keys = findall(x -> x isa AbstractArray{<:Any, N}, data)
        if isempty(array_keys)
            error("No $(N)-D array was found in the input file: $filename")
        end
        if length(array_keys) > 1
            array_keys = sort(array_keys)
            @warn "Multiple possible images found in file: $(filename)\nChoosing variable $(repr(array_keys[1])) out of the following options: $(join(repr.(array_keys), ", "))"
        end
        data = data[array_keys[1]]

    elseif maybe_get_suffix(filename) ∈ (".nii", ".nii.gz")
        # Check slope field; if scl_slope == 0, data is not scaled and raw data should be returned:
        #   See e.g. https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/scl_slopeinter.html
        nii = NIfTI.niread(filename)
        scl_slope, scl_inter = nii.header.scl_slope, nii.header.scl_inter
        if scl_slope == 0
            scl_slope, scl_inter = one(scl_slope), zero(scl_inter)
        end
        data = nii.raw .* scl_slope .+ scl_inter

    elseif maybe_get_suffix(filename) ∈ (".par", ".xml", ".rec")
        rec = ParXRec.load(filename)
        data = parent(rec.data) # get underlying data wrapped by `AxisArray`

    else
        error("Currently, only $ALLOWED_FILE_SUFFIXES_STRING files are supported")
    end

    # Ensure `data` has exactly N dimensions, dropping or selecting trailing dimensions as needed
    data = ensure_ndims(filename, data, Val(N))

    # Currently, the pipeline is ~twice as fast on Float64 arrays than Float32 arrays (unclear why).
    # However, the MATLAB toolbox converts images to double as well, so here we simply do the same
    sz = ntuple(i -> size(data, i), N)
    image = copyto!(Array{Float64, N}(undef, sz), data)

    return image
end
load_image(filename; ndims::Int = 4) = load_image(filename, Val(ndims))

function ensure_ndims(filename, data::AbstractArray{T, D}, ::Val{N}) where {T, D, N}
    if D < N
        return reshape(data, ntuple(i -> (i <= D ? size(data, i) : 1), N)) # add trailing singleton dims
    elseif D == N
        return data # has the expected number of dimensions
    else # D > N
        if any(i -> size(data, i) > 1, (N+1):D)
            @warn "Input file $filename has $D dimensions, expected $N; selecting the first $N-D volume along $(D - N == 1 ? "dimension" : "dimensions") $(join((N+1):D, ","))."
        end
        return view(data, ntuple(i -> (i <= N ? Colon() : 1), D)...) # select first volume along trailing dims
    end
end

function try_load_B1mapfile!(maps::T2Maps, B1mapfile::String)
    try
        load_B1map!(maps, load_image(B1mapfile, Val(3)))
    catch e
        @warn "Error while loading B1 map file: $B1mapfile"
        @warn sprint(showerror, e, catch_backtrace())
    end
    return nothing
end

function try_apply_maskfile!(image::Array{<:Any, 4}, maskfile::String)
    try
        image .*= load_image(maskfile, Val(3))
    catch e
        @warn "Error while loading mask file: $maskfile"
        @warn sprint(showerror, e, catch_backtrace())
    end
    return image
end

function try_apply_bet!(image::Array{<:Any, 4}, betpath::String, betargs::Vector{String})
    try
        image .*= make_bet_mask(image, betpath, betargs)
    catch e
        @warn "Error while making mask using BET"
        @warn sprint(showerror, e, catch_backtrace())
    end
    return image
end

function make_bet_mask(image::Array{<:Any, 3}, betpath::String, betargs::Vector{String})
    # Create mask using BET and return mask
    mask = mktempdir() do temppath
        tempbase = basename(tempname())
        nifti_imagefile = joinpath(temppath, tempbase * ".nii")
        nifti_maskfile = joinpath(temppath, tempbase * ".bet")
        NIfTI.niwrite(nifti_imagefile, NIfTI.NIVolume(image)) # create nifti file for bet
        run(Cmd(String[betpath; nifti_imagefile; nifti_maskfile; betargs]))

        # BET appends "_mask" and ".nii.gz" to output file name.
        # Find this file, ensure it is unique, then load and return it
        bet_maskfiles = filter!(file -> startswith(file, tempbase * ".bet_mask"), readdir(temppath))
        @assert length(bet_maskfiles) == 1 # ensure unique; this should never be false using a temp filename
        return load_image(joinpath(temppath, bet_maskfiles[1]), Val(3))
    end
    return mask
end
make_bet_mask(image::Array{<:Any, 4}, args...; kwargs...) = make_bet_mask(image[:, :, :, 1], args...; kwargs...) # use first echo

function clean_bet_args(betargs::String)
    # Split betargs, and ensure that "-m" (make binary mask) is among args
    dlm = c -> isspace(c) || c == ','
    args = split(betargs, dlm; keepempty = false)
    if "-m" ∉ args
        pushfirst!(args, "-m")
    end
    return convert(Vector{String}, args)
end

maybe_get_first(f, xs) = findfirst(f, xs) |> I -> I === nothing ? nothing : xs[I]
maybe_get_suffix(filename) = maybe_get_first(ext -> endswith(lowercase(filename), ext), ALLOWED_FILE_SUFFIXES) # case-insensitive
is_allowed_suffix(filename) = maybe_get_suffix(filename) !== nothing

function chop_allowed_suffix(filename::AbstractString)
    ext = maybe_get_suffix(filename)
    if ext !== nothing
        return filename[1:end-length(ext)]
    else
        error("Currently only $ALLOWED_FILE_SUFFIXES_STRING file types are supported")
    end
end
