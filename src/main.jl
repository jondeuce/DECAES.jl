####
#### CLI settings
####

const ALLOWED_FILE_SUFFIXES = (".mat", ".nii", ".nii.gz", ".par", ".xml", ".rec")
const ALLOWED_FILE_SUFFIXES_STRING = join(ALLOWED_FILE_SUFFIXES, ", ", ", and ")

const CLI_SETTINGS = ArgParseSettings(;
    prog = "",
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
        :help => "call T2mapSEcorr to compute T2 distributions from 4D multi spin-echo input images. T2 distributions and T2 maps produced by T2mapSEcorr are saved as MAT files with extensions .t2dist.mat and .t2maps.mat",
    ),
    "--T2part",
    Dict(
        :action => :store_true,
        :help => "call T2partSEcorr to analyze 4D T2 distributions to produce parameter maps. If --T2map is also passed, input 4D arrays are interpreted as multi spin-echo images and T2 distributions are first computed by T2mapSEcorr. If only --T2part is passed, input 4D arrays are interpreted as T2 distributions and only T2partSEcorr is called. Output T2 parts are saved as a MAT file with extension .t2parts.mat",
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
    "--legacy",
    Dict(
        :action => :store_true,
        :help => "use legacy settings and algorithms from the original MATLAB pipeline. This ensures that the same T2-distributions and T2-parts will be produced as those from MATLAB. Note that execution time will be much slower, and less robust algorithms will be used",
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
        :help => "inter-echo spacing. Required when --T2map is passed. (units: seconds)",
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
        :help => "minimum and maximum T2 values. T2 components are logarithmically spaced between these bounds. Required parameter. (units: seconds)",
        :group => :t2_map_part_required,
    ),
    "--SPWin",
    Dict(
        :nargs => 2,
        :arg_type => Float64,
        :help => "minimum and maximum T2 values of the short peak window. Required parameter when --T2part is passed. (units: seconds)",
        :group => :t2_map_part_required,
    ),
    "--MPWin",
    Dict(
        :nargs => 2,
        :arg_type => Float64,
        :help => "minimum and maximum T2 values of the middle peak window. Required parameter when --T2part is passed. (units: seconds)",
        :group => :t2_map_part_required,
    ),
    "--Reg",
    Dict(
        :arg_type => String,
        :help => "routine used for choosing regularization parameter. One of \"lcurve\", \"gcv\", \"chi2\", or \"none\", representing L-curve based regularization, generalized cross-validation based regularization, --Chi2Factor based Tikhonov regularization, and no regularization, respectively. Required parameter",
        :group => :t2_map_part_required,
    ),
    "--Chi2Factor",
    Dict(
        :arg_type => Float64,
        :help => "if --Reg=\"chi2\", the T2 distribution is regularized such that the chi^2 goodness of fit is increased by a multiplicative factor --Chi2Factor relative to the unregularized solution. Required parameter when --Reg=\"chi2\"",
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
        :help => "assumed value of longitudinal T1 relaxation. (default: 1.0) (units: seconds)",
        :group => :t2_map_part_optional,
    ),
    "--Sigmoid",
    Dict(
        :arg_type => Float64,
        :help => "replace the hard upper limit cutoff time of the short peak window, SPWin[2], with a smoothed sigmoidal cutoff function 'σ' scaled and shifted such that σ(SPWin[2] +/- Sigmoid) = 0.5 -/+ 0.4. Sigmoid is the time scale of the smoothing. (units: seconds)",
        :group => :t2_map_part_optional,
    ),
    "--Threshold",
    Dict(
        :arg_type => Float64,
        :help => "first echo intensity cutoff for empty voxels. Processing is skipped for voxels with intensity <= --Threshold. (default: 0.0) (units: signal magnitude)",
        :group => :t2_map_part_optional,
    ),
    "--Progress",
    Dict(
        :action => :store_const, # deprecated flags use :store_const with default value nothing (instead of :store_true with default value false)
        :constant => true,
        :default => nothing,
        :help => "Print progress updates during T2 distribution computation. Note: this flag is now deprecated; progress updates are always printed unless the --quiet flag is passed",
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
        :help => "in estimating the local refocusing flip angle to correct for B1 inhomogeneities, up to --nRefAngles angles in the range [--MinRefAngle, 180] are explicitly checked. The optimal angle is then estimated by interpolating between these observations. (default: 32)",
        :group => :B1_SE_corr,
    ),
    "--nRefAnglesMin",
    Dict(
        :arg_type => Int,
        :help => "initial number of angles in the range [--MinRefAngle, 180] to check during flip angle estimation before refinement near likely optima. Setting --nRefAnglesMin equal to --nRefAngles forces all candidate angles to be checked before interpolation. (default: 5)",
        :group => :B1_SE_corr,
    ),
    "--MinRefAngle",
    Dict(
        :arg_type => Float64,
        :help => "minimum refocusing angle for flip angle estimation. (default: 50.0) (units: degrees)",
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
        :help => "refocusing pulse control angle. The sequence of flip angles used within the extended phase graph algorithm to perform stimulated echo correction is (90, 180, β, β, ..., β), where β is the refocusing pulse control angle. For typical multi spin-echo sequences this parameter should not be changed. (default: 180.0) (units: degrees)",
        :group => :B1_SE_corr,
    ),
)

add_arg_group!(CLI_SETTINGS,
    "Additional save options",
    :save_opts,
)

add_arg_table!(CLI_SETTINGS,
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
    main(command_line_args = ARGS)

Entry point function for command line interface, parsing the command line arguments `ARGS` and subsequently calling one or both of `T2mapSEcorr` and `T2partSEcorr` with the parsed settings.
See the [Arguments](@ref) section for available options.

See also:

  - [`T2mapSEcorr`](@ref)
  - [`T2partSEcorr`](@ref)
"""
function main(command_line_args::Vector{String} = ARGS)

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
                return cp(src, dst; force = true)
            end
        end

        # Main processing
        tee_capture(;
            logfile = joinpath(file_info[:outputfolder], file_info[:choppedinputfile] * ".log"),
            suppress_terminal = opts[:quiet],
            suppress_logfile = opts[:dry],
        ) do
            try
                main(file_info, opts)
            catch e
                @warn "Error during processing of file: $(file_info[:inputfile])"
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end

    return nothing
end

function main(file_info::Dict{Symbol, Any}, opts::Dict{Symbol, Any})

    # Starting message/starting time
    t_start = tic()
    @info "Starting DECAES v$(VERSION) using Julia v$(Base.VERSION) with $(Threads.nthreads()) threads"

    # Load image(s)
    image = @showtime(
        "Loading input file: $(file_info[:inputfile])",
        load_image(file_info[:inputfile]),
    )

    # Apply mask
    if file_info[:maskfile] !== nothing
        @showtime(
            "Applying mask from file: $(file_info[:maskfile])",
            try_apply_maskfile!(image, file_info[:maskfile]),
        )
    elseif opts[:bet]
        @showtime(
            "Making and applying BET mask with args: $(join(opts[:betargs], " "))",
            try_apply_bet!(image, opts[:betpath], opts[:betargs]),
        )
    end

    # Compute T2 distribution from input 4D multi-echo image
    if opts[:T2map]
        t2map_opts = t2map_options(image, opts)
        t2map_maps = T2Maps(t2map_opts)
        t2map_dist = T2Distributions(t2map_opts)

        # Load B1 map
        if file_info[:B1mapfile] !== nothing
            @showtime(
                "Loading B1 map from file: $(file_info[:B1mapfile])",
                try_load_B1mapfile!(t2map_maps, file_info[:B1mapfile]),
            )
        end

        maps, dist = @showtime(
            "Running T2mapSEcorr on file: $(file_info[:inputfile])",
            T2mapSEcorr!(t2map_maps, t2map_dist, image, t2map_opts),
        )

        # Save T2-distribution to .mat file
        savefile = joinpath(file_info[:outputfolder], file_info[:choppedinputfile] * ".t2dist.mat")
        if !opts[:dry]
            @showtime(
                "Saving T2 distribution to file: $savefile",
                MAT.matwrite(savefile, Dict{String, Any}("dist" => dist)),
            )
        end

        # Save T2-maps to .mat file
        savefile = joinpath(file_info[:outputfolder], file_info[:choppedinputfile] * ".t2maps.mat")
        if !opts[:dry]
            @showtime(
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
        parts = @showtime(
            "Running T2partSEcorr",
            T2partSEcorr(dist, t2part_options(dist, opts)),
        )

        # Save T2-parts to .mat file
        savefile = joinpath(file_info[:outputfolder], file_info[:choppedinputfile] * ".t2parts.mat")
        if !opts[:dry]
            @showtime(
                "Saving T2 parts maps to file: $savefile",
                MAT.matwrite(savefile, parts),
            )
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
        opts = clean_cli_args!(opts)
    end
    return opts
end

function handle_cli_deprecations!(opts)
    handle_renamed_cli_flag!(opts, :Progress => nothing)
    return opts
end

function handle_renamed_cli_flag!(opts, oldnew::Pair)
    oldflag, newflag = oldnew
    if opts[oldflag] !== nothing
        if newflag === nothing
            @warn "The flag --$oldflag is deprecated and will be removed in future releases."
        elseif opts[newflag] !== nothing
            error("The flag --$newflag and the deprecated flag --$oldflag were both passed; use --$newflag only.")
        else
            @warn "The flag --$oldflag is deprecated and will be removed in future releases; use --$newflag instead."
        end
        if newflag !== nothing
            opts[newflag] = opts[oldflag]
        end
        delete!(opts, oldflag)
    end
    return opts
end

function clean_cli_args!(opts)
    # Preprocess arguments for use
    if opts[:betargs] isa String
        opts[:betargs] = clean_bet_args(opts[:betargs])::Vector{String}
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

    # Currently, the pipeline is ~twice as fast on Float64 arrays than Float32 arrays (unclear why).
    # However, the MATLAB toolbox converts images to double as well, so here we simply do the same
    sz = ntuple(i -> size(data, i), N)
    image = copyto!(Array{Float64, N}(undef, sz), data)

    return image
end
load_image(filename; ndims::Int = 4) = load_image(filename, Val(ndims))

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
