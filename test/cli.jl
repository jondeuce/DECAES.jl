# Arbitrary default required parameters used during testing
default_paramdict = Dict{Symbol,Any}(
    :TE => 8e-3,
    :nT2 => 42,
    :T2Range => (12e-3, 1.8),
    :SPWin => (12e-3, 37e-3),
    :MPWin => (37e-3, 650e-3),
    :Reg => "lcurve",
)

# Legacy settings which previously had defaults
legacy_default_paramdict = Dict{Symbol,Any}(
    :TE => 10e-3,
    :nT2 => 40,
    :T2Range => (15e-3, 2.0),
    :SPWin => (14e-3, 40e-3),
    :MPWin => (40e-3, 200e-3),
    :Reg => "chi2",
    :Chi2Factor => 1.02,
)

# Write 4D image to disk
function write_image(filename, image)
    if endswith(filename, ".mat")
        DECAES.MAT.matwrite(filename, Dict("img" => image))
    else
        DECAES.NIfTI.niwrite(filename, DECAES.NIfTI.NIVolume(image))
    end
end

function matlab_api_main(matlab_args, args)
    mxcall(:decaes, 0, matlab_args..., args...)
end

# Call main function on image file `image`
function run_main(image, args; make_settings_file::Bool)
    global RUN_MATLAB_TESTS

    # Write input image to file for reading
    inputfilename = args[1]
    outputpath = args[3]
    inputfilebasename = joinpath(outputpath, "input")
    write_image(inputfilename, image)

    # Run main, possibly writing CLI args to settings file first
    if make_settings_file
        settings_file = joinpath(outputpath, "settings.txt")
        open(settings_file, "w") do file
            println(file, join(args, "\n"))
        end
        main(["@" * settings_file])
    else
        main(args)
    end

    # Check that only requested files were created
    t2maps_file, t2dist_file, t2parts_file, settings_file = inputfilebasename .* (".t2maps.mat", ".t2dist.mat", ".t2parts.mat", ".settings.txt")
    T2map, T2part = ("--T2map" ∈ args), ("--T2part" ∈ args)

    @test !xor(T2map,  isfile(t2maps_file))
    @test !xor(T2map,  isfile(t2dist_file))
    @test !xor(T2part, isfile(t2parts_file))
    @test !xor(make_settings_file, isfile(settings_file))

    t2maps  = T2map  ? DECAES.MAT.matread(t2maps_file) : nothing
    t2dist  = T2map  ? DECAES.MAT.matread(t2dist_file)["dist"] : nothing
    t2parts = T2part ? DECAES.MAT.matread(t2parts_file) : nothing

    return (; t2maps, t2dist, t2parts)
end

function construct_args(paramdict;
        argstype,
        inputfilename = nothing,
        outputpath = nothing,
        quiet::Bool = true,
        legacy::Bool = false,
        T2map::Bool = true,
        T2part::Bool = true,
    )

    if legacy
        # Add legacy default options to paramdict
        for (k, v) in legacy_default_paramdict
            if k ∉ keys(paramdict)
                paramdict[k] = v
            end
        end
    end

    if argstype === :cli
        #### CLI

        args = [inputfilename, rand(["--output", "-o"]), outputpath]
        T2map  && push!(args, "--T2map")
        T2part && push!(args, "--T2part")
        quiet  && push!(args, rand(["--quiet", "-q"]))
        legacy && push!(args, "--legacy")

        for (param, paramval) in paramdict
            # Default flag/value pairs for `nothing` values; `nothing` is always default if allowable, therefore no flag/val is passed
            if paramval !== nothing
                push!(args, "--" * string(param)) # CLI flags are prepended with "--"
                append!(args,
                    paramval isa Tuple ? [string(x) for x in paramval] : # Pass each arg separately
                    paramval isa Bool  ? [] : # No arg necessary, flag only
                    [string(paramval)] # Pass string
                )
            end
        end

        return args

    elseif argstype === :mat
        #### MATLAB

        @assert legacy
        t2map_args  = T2map  ? Dict{Symbol,Any}() : nothing
        t2part_args = T2part ? Dict{Symbol,Any}() : nothing

        # Only these params are used within MATLAB (possibly spelled differently)
        mat_t2map_params  = [:Chi2Factor, :MinRefAngle, :nRefAngles, :nT2, :RefConAngle, :Reg, :SaveRegParam, :SetFlipAngle, :T1, :T2Range, :TE, :Threshold, :vTEparam]
        mat_t2part_params = [:T2Range, :SPWin, :MPWin, :Sigmoid]

        for (param, paramval) in paramdict
            T2map  && (param ∈ mat_t2map_params)  && jl_to_mat_param!(t2map_args,  param, paramval)
            T2part && (param ∈ mat_t2part_params) && jl_to_mat_param!(t2part_args, param, paramval)
        end

        return t2map_args, t2part_args

    elseif argstype === :jl
        #### Julia

        t2map_args  = T2map  ? Dict{Symbol,Any}(:legacy => legacy) : nothing
        t2part_args = T2part ? Dict{Symbol,Any}(:legacy => legacy) : nothing

        t2map_fields = DECAES.fieldsof(T2mapOptions, Set)
        t2part_fields = DECAES.fieldsof(T2partOptions, Set)
        for (param, paramval) in paramdict
            T2map  && (param ∈ t2map_fields)  && setindex!(t2map_args, paramval, param)
            T2part && (param ∈ t2part_fields) && setindex!(t2part_args, paramval, param)
        end

        return t2map_args, t2part_args
    end
end

function jl_to_mat_param!(opts, param, paramval)

    # T2mapSEcorr parameters which aren't in the MATLAB API
    new_t2map_params = Set{Symbol}([
        :SaveResidualNorm,
        :SaveDecayCurve,
        :SaveNNLSBasis,
        :Silent,
    ])

    if param == :SaveRegParam # renamed parameter
        opts[:Save_regparam] = ifelse(paramval, "yes", "no")
    elseif param == :nRefAngles # renamed parameter
        opts[:nAngles] = paramval
    elseif param == :RefConAngle # renamed parameter
        opts[:RefCon] = paramval
    elseif param == :Reg # renamed value: "no" => "none", "lcurve" => "gcv"
        opts[:Reg] = paramval == "none" ? "no" : paramval == "gcv" ? "lcurve" : paramval
    elseif param == :SPWin # renamed parameter
        opts[:spwin] = paramval
    elseif param == :MPWin # renamed parameter
        opts[:mpwin] = paramval
    elseif param ∉ new_t2map_params # skip Julia-only parameters
        opts[param] = paramval
    end

    return opts
end

field_error_string(x, y) = "max val = $(maximum(abs, y)), max diff = $(maximum(abs, x.-y)), rel diff = $(maximum(abs, (x.-y)./y))"

function test_field!(allpassed, x, y, prefix = "failed:"; kwargs...)
    passed = isapprox(x, y; kwargs..., nans = true)
    allpassed[] &= passed
    !passed && println(prefix * " (" * field_error_string(x, y) * ")")
    @test passed
end

# Compare t2map results for approximately equality
function test_compare_t2map(out1, out2; kwargs...)
    maps1, dist1, maps2, dist2 = out1..., out2...
    allpassed = Ref(true)
    for s in keys(maps1)
        haskey(maps2, s) && test_field!(allpassed, maps1[s], maps2[s], "maps failed: $s"; kwargs...)
    end
    test_field!(allpassed, dist1, dist2, "dist failed"; kwargs...)
    return allpassed[]
end

# Compare t2part results for approximately equality
function test_compare_t2part(part1, part2; kwargs...)
    allpassed = Ref(true)
    for s in keys(part1)
        haskey(part2, s) && test_field!(allpassed, part1[s], part2[s], "parts failed: $s"; kwargs...)
    end
    return allpassed[]
end

# CLI parameter settings to loop over
#   -Each param value will be tested individually, with all other params set to default values
#   -Each list should contain some non-default/edge case values
function run_cli_tests()
    cli_params_perms = Any[
        (:MPWin            .=> [(38e-3, 180e-3)],),
        (:MinRefAngle      .=> [55.0],),
        (:RefConAngle      .=> [172.0],),
        (
            :Reg           .=> ["none", "chi2", "gcv", "lcurve"],
            :Chi2Factor    .=> [nothing, 1.025, nothing, nothing],
        ),
        (:SPWin            .=> [(13e-3, 37e-3)],),
        (:SaveResidualNorm .=> [false, true],),
        (:SaveDecayCurve   .=> [false, true],),
        (:SaveNNLSBasis    .=> [false, true],),
        (:SaveRegParam     .=> [false, true],),
        (:SetFlipAngle     .=> [nothing, 170.0],),
        (:Sigmoid          .=> [nothing, 1.0],),
        (:T1               .=> [0.95],),
        (:T2Range          .=> [(16e-3, 1.8)],),
        (:TE               .=> [8e-3, 11e-3],),
        (:Threshold        .=> [125_000.0, Inf],), # Include non-zero and infinite (i.e. either some or all voxels skipped)
        (:nRefAngles       .=> [9, 10],), # Include odd number
        (:nRefAnglesMin    .=> [4, 5],), # Include odd number
        (:nT2              .=> [2, 47],), # Include odd number
    ]

    make_settings_perms = [false, true]
    file_suffix_perms = [".mat", ".nii", ".nii.gz"] # Note: no PAR/REC or XML/REC, since we can't write to them
    B1map_perms = [false, true]
    legacy_perms = [false, true]
    param_perms = (cli_params_perms, make_settings_perms, file_suffix_perms, B1map_perms, legacy_perms)
    repeat_until(x) = Iterators.take(Iterators.cycle(x), max(length.(param_perms)...))

    for (param_val_lists, make_settings_file, file_suffix, B1map, legacy) in zip(map(repeat_until, param_perms)...), param_val_pairs in zip(param_val_lists...)
        paramdict = deepcopy(default_paramdict)
        for (param, paramval) in param_val_pairs
            paramdict[param] = paramval
        end

        image = DECAES.mock_image(nTE = rand(4:64))
        settings_kwargs_jl = Dict{Symbol,Any}(:argstype => :jl, :quiet => rand([true,false]), :legacy => legacy, :T2map => true, :T2part => true)
        settings_kwargs_cli = Dict{Symbol,Any}(:argstype => :cli, :quiet => rand([true,false]), :legacy => legacy, :T2map => true, :T2part => true)
        jl_t2map_kwargs, jl_t2part_kwargs = construct_args(paramdict; settings_kwargs_jl...)

        # Run T2map and T2part through Julia API for comparison
        t2map, t2dist = DECAES.redirect_to_devnull() do
            T2mapSEcorr(image; jl_t2map_kwargs...)
        end
        t2part = DECAES.redirect_to_devnull() do
            T2partSEcorr(t2dist; jl_t2part_kwargs...)
        end

        # Run CLI with both --T2map and --T2part flags
        mktempdir() do path
            settings_kwargs_cli[:outputpath] = path
            settings_kwargs_cli[:inputfilename] = joinpath(path, "input" * file_suffix)
            cli_t2map_args = construct_args(paramdict; settings_kwargs_cli...)

            if B1map && !("--SetFlipAngle" ∈ cli_t2map_args)
                # Write reference B1map computed above to file and pass filename to DECAES CLI
                B1mapfilename = joinpath(path, "B1" * file_suffix)
                write_image(B1mapfilename, t2map["alpha"])
                append!(cli_t2map_args, ["--B1map", B1mapfilename])
            end

            t2maps_cli, t2dist_cli, t2parts_cli = DECAES.redirect_to_devnull() do
                run_main(image, cli_t2map_args; make_settings_file = make_settings_file)
            end
            t2map_passed = test_compare_t2map((t2map, t2dist), (t2maps_cli, t2dist_cli); rtol = 1e-14)
            t2part_passed = test_compare_t2part(t2part, t2parts_cli; rtol = 1e-14)
            if !(t2map_passed && t2part_passed)
                println("\n ------------------------------- \n")
                println("CLI with --T2map and --T2part failed: $param_val_pairs")
                @show jl_t2map_kwargs
                @show jl_t2part_kwargs
                @show cli_t2map_args
                @show paramdict
                println("\n ------------------------------- \n")
            end
        end

        # Run CLI with --T2part flag only
        mktempdir() do path
            settings_kwargs_cli[:outputpath] = path
            settings_kwargs_cli[:inputfilename] = joinpath(path, "input" * file_suffix)
            settings_kwargs_cli[:T2map] = false
            cli_t2part_args = construct_args(paramdict; settings_kwargs_cli...)

            t2maps_cli, t2dist_cli, t2parts_cli = DECAES.redirect_to_devnull() do
                run_main(t2dist, cli_t2part_args; make_settings_file = make_settings_file)
            end
            t2part_passed = test_compare_t2part(t2part, t2parts_cli; rtol = 1e-14)
            if !t2part_passed
                println("\n ------------------------------- \n")
                println("CLI with --T2part only failed: $param_val_pairs")
                @show jl_t2map_kwargs
                @show jl_t2part_kwargs
                @show cli_t2part_args
                @show paramdict
                println("\n ------------------------------- \n")
            end
        end
    end
end

@testset "Command line interface" begin
    run_cli_tests()
end

# ================================================================================
# UBC MWI Toolbox MATLAB compatibility tests
#   NOTE: For these tests to run, MATLAB must be installed on your default path.
#   Additionally, the MWI NNLS toolbox (https://github.com/ubcmri/ubcmwf)
#   folder "MWI_NNLS_toolbox_0319" (and subfolders) must be added to your
#   default MATLAB path.
# ================================================================================

# Helper functions
matlabify(x::AbstractString) = String(x)
matlabify(x::AbstractArray) = Float64.(x)
matlabify(x::Tuple) = [Float64.(x)...]
matlabify(x::Bool) = x
matlabify(x) = map(Float64, x)
matlabify(kwargs::Base.Iterators.Pairs) = Iterators.flatten([(string(k), matlabify(v)) for (k, v) in kwargs if v !== nothing])

mxT2mapSEcorr(image, maxCores = 1; kwargs...) = MATLAB.mxcall(:T2map_SEcorr_nechoes_2019, 2, image, maxCores, matlabify(kwargs)...)
mxT2partSEcorr(image; kwargs...) = MATLAB.mxcall(:T2part_SEcorr_2019, 1, image, matlabify(kwargs)...)

function matlab_tests()
    # Arbitrary non-default T2mapSEcorr options for testing
    mat_t2map_params_perms = Any[
        (:TE               .=> [9e-3],),
        (:T1               .=> [1.1],),
        (:Threshold        .=> [250.0],),
        (:nT2              .=> [10, 59],), # Include odd number
        (:T2Range          .=> [(8e-3, 1.0)],),
        (:RefConAngle      .=> [175.0],),
        (:MinRefAngle      .=> [60.0],),
        (:nRefAngles       .=> [7, 12],),
        (
            :Reg           .=> ["none", "chi2", "lcurve", "gcv"],
            :Chi2Factor    .=> [nothing, 1.03, nothing, nothing],
        ),
        (:SetFlipAngle     .=> [178.0],),
        (:SaveResidualNorm .=> [false, true],),
        (:SaveDecayCurve   .=> [false, true],),
        (:SaveRegParam     .=> [false, true],),
        (:SaveNNLSBasis    .=> [false, true],),
    ]

    # Arbitrary non-default T2partSEcorr options for testing
    mat_t2part_params_perms = Any[
        (:T2Range    .=> [(11e-3, 1.5)],),
        (:SPWin      .=> [(12e-3, 28e-3)],),
        (:MPWin      .=> [(35e-3, 150e-3)],),
        (:Sigmoid    .=> [1.5],),
    ]

    # Relative tolerance threshold for legacy algorithms to match MATLAB version
    default_rtol = 1e-10

    @testset "T2mapSEcorr" begin
        settings_kwargs_jl = Dict{Symbol,Any}(:argstype => :jl, :quiet => rand([true,false]), :legacy => true, :T2map => true, :T2part => true)
        settings_kwargs_mat = Dict{Symbol,Any}(:argstype => :mat, :quiet => rand([true,false]), :legacy => true, :T2map => true, :T2part => true)

        for param_val_lists in mat_t2map_params_perms, param_val_pairs in zip(param_val_lists...)
            rtol = default_rtol
            paramdict = deepcopy(legacy_default_paramdict)
            for (param, paramval) in param_val_pairs
                paramdict[param] = paramval
            end

            # The MATLAB flag "lcurve" for choosing the regularization parameter uses the Generalized Cross-Validation (GCV) method.
            # There are several issues with comparing the DECAES implementation with MATLAB:
            #   1.  GCV involves minimizing a functional GCV(mu), which is implemented using an internal call to `fminbnd` with a tolerance of 1e-3,
            #       and therefore the Julia outputs would only match to at best a tolerance of 1e-3
            #   2.  There is an error in the `G(mu,C_g,d_g)` subfunction in the MATLAB MWI Toolbox:
            #       -   Numerator should be ||A*x_mu - b||^2, not ||A*x_mu - b||
            #       -   See e.g. equation (1.4) in Fenu, C. et al., 2017, GCV for Tikhonov regularization by partial SVD (https://doi.org/10.1007/s10543-017-0662-0)
            #   3.  There is an error in methodology:
            #       -   Solving regularized ||Ax-b||^2 via [A; mu*I] \ [b; 0] is equivalent to minimizing
            #               ||Ax-b||^2 + mu^2 * ||x||^2
            #           as opposed to
            #               ||Ax-b||^2 + mu * ||x||^2
            #           as is used in the MWI toolbox
            #   4.  The MATLAB equivalent to --Reg="lcurve" has been replaced with the more accurate --Reg="gcv".
            #       In DECAES, --Reg="lcurve" implements a different method which utilizes the L-curve method; see:
            #           A. Cultrera and L. Callegaro, “A simple algorithm to find the L-curve corner in the regularization of ill-posed inverse problems”
            #           IOPSciNotes, vol. 1, no. 2, p. 025004, Aug. 2020, doi: 10.1088/2633-1357/abad0d
            # This comparison is therefore skipped by default. If you have a version in which these errors are fixed,
            # the below line can be modified (with rtol set appropriately in accordance with your solver tolerance)
            if any(param === :Reg && paramval ∈ ("lcurve", "gcv") for (param, paramval) in param_val_pairs)
                continue
                # rtol = 1e-3
            end

            jl_t2map_kwargs,  _ = construct_args(paramdict; settings_kwargs_jl...)
            mat_t2map_kwargs, _ = construct_args(paramdict; settings_kwargs_mat...)

            # Run T2mapSEcorr
            image = DECAES.mock_image(nTE = rand(4:64))
            t2map_out_jl = DECAES.redirect_to_devnull() do
                T2mapSEcorr(image; jl_t2map_kwargs...)
            end
            t2map_out_mat = DECAES.redirect_to_devnull() do
                mxT2mapSEcorr(image; mat_t2map_kwargs...)
            end
            allpassed = test_compare_t2map(t2map_out_jl, t2map_out_mat; rtol = rtol)
            if !allpassed
                println("\n ------------------------------- \n")
                println("MATLAB T2mapSEcorr comparison failed: $param_val_pairs")
                @show jl_t2map_kwargs
                @show mat_t2map_kwargs
                @show paramdict
                println("\n ------------------------------- \n")
            end
        end
    end

    @testset "T2partSEcorr" begin
        settings_kwargs_jl = Dict{Symbol,Any}(:argstype => :jl, :quiet => rand([true,false]), :legacy => true, :T2map => false, :T2part => true)
        settings_kwargs_mat = Dict{Symbol,Any}(:argstype => :mat, :quiet => rand([true,false]), :legacy => true, :T2map => false, :T2part => true)

        for param_val_lists in mat_t2part_params_perms, param_val_pairs in zip(param_val_lists...)
            paramdict = deepcopy(legacy_default_paramdict)
            delete!(paramdict, :nT2) # inferred
            for (param, paramval) in param_val_pairs
                paramdict[param] = paramval
            end
            _, jl_t2part_kwargs  = construct_args(paramdict; settings_kwargs_jl...)
            _, mat_t2part_kwargs = construct_args(paramdict; settings_kwargs_mat...)

            # Run T2partSEcorr
            T2dist = DECAES.mock_T2_dist(nT2 = rand(4:64))
            t2part_jl = DECAES.redirect_to_devnull() do
                T2partSEcorr(T2dist; jl_t2part_kwargs...)
            end
            t2part_mat = DECAES.redirect_to_devnull() do
                mxT2partSEcorr(T2dist; mat_t2part_kwargs...)
            end
            allpassed = test_compare_t2part(t2part_jl, t2part_mat; rtol = default_rtol)
            if !allpassed
                println("\n ------------------------------- \n")
                println("MATLAB T2partSEcorr comparison failed: $param_val_pairs")
                @show jl_t2part_kwargs
                @show mat_t2part_kwargs
                @show paramdict
                println("\n ------------------------------- \n")
            end
        end
    end
end

if RUN_MATLAB_TESTS
    @testset "UBC MWI toolbox" begin
        matlab_tests()
    end
end

nothing
