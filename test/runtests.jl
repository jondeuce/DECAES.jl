using DECAES
using Test

# Write 4D image to disk
function write_image(filename, image)
    if endswith(filename, ".mat")
        DECAES.MAT.matwrite(filename, Dict("img" => image))
    else
        DECAES.NIfTI.niwrite(filename, DECAES.NIfTI.NIVolume(image))
    end
end

# Call main function with arguments "args", optionally writing args to file first
function run_main(args, make_settings_file = false)
    if make_settings_file
        # Write input args to a temporary file and read in the args
        mktempdir() do temppath
            settings_file = joinpath(temppath, "settings.txt")
            open(settings_file, "w") do file
                println(file, join(args, "\n"))
            end
            main(["@" * settings_file])
        end
    else
        # Run main with args directly
        main(args)
    end
    return nothing
end

field_error_string(x, y) = "max val = $(maximum(abs, y)), max diff = $(maximum(abs, x.-y)), rel diff = $(maximum(abs, (x.-y)./y))"

function test_field!(allpassed, x, y, prefix = "failed:"; kwargs...)
    passed = isapprox(x, y; kwargs..., nans = true)
    allpassed[] &= passed
    !passed && println(prefix * " (" * field_error_string(x,y) * ")")
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
const cli_params_perms = Dict{Symbol, Vector{<:Any}}(
    :Chi2Factor       => [1.025],
    :MPWin            => [(38e-3, 180e-3)],
    :MinRefAngle      => [55.0],
    :RefConAngle      => [172.0],
    :Reg              => ["no", "chi2", "lcurve"],
    :SPWin            => [(13e-3, 37e-3)],
    :SaveResidualNorm => [false, true],
    :SaveDecayCurve   => [false, true],
    :SaveNNLSBasis    => [false, true],
    :SaveRegParam     => [false, true],
    :SetFlipAngle     => [nothing, 170.0],
    :Sigmoid          => [nothing, 1.0],
    :T1               => [0.95],
    :T2Range          => [(16e-3, 1.8)],
    :TE               => [8e-3, 11e-3],
    :Threshold        => [0.0, Inf], # Include zero and infinite (i.e. either all voxels included or skipped)
    :nRefAngles       => [9, 10], # Include odd number
    :nRefAnglesMin    => [4, 5], # Include odd number
    :nT2              => [40, 45], # Include odd number
)

@testset "CLI" begin
    image = DECAES.mock_image(nTE = 32 + rand(0:1))

    make_settings_perms = [false, true]
    file_suffix_perms = [".mat", ".nii", ".nii.gz"]
    iters = (cli_params_perms, make_settings_perms, file_suffix_perms)
    nloop = max(length.(iters)...)
    repeat_until(x) = Iterators.take(Iterators.cycle(x), nloop)

    for ((param, valuelist), make_settings_file, file_suffix) in zip(map(repeat_until, iters)...), paramval in valuelist
        # Default flag/value pairs for `nothing` values; `nothing` is always default if allowable, therefore no flag/val is passed
        cliparamflag = []
        cliparamval = []
        if !isnothing(paramval)
            cliparamflag = ["--" * string(param)] # CLI flags are prepended with "--"
            cliparamval =
                paramval isa Tuple ? [string(x) for x in paramval] : # Pass each arg separately
                paramval isa Bool  ? [] : # No arg necessary, flag only
                [string(paramval)] # Pass string
        end

        # Run T2map and T2part through Julia API for comparison
        t2map_args  = param ∈ fieldnames(T2mapOptions)  ? Dict(param => paramval) : Dict{Symbol,Any}()
        t2part_args = param ∈ fieldnames(T2partOptions) ? Dict(param => paramval) : Dict{Symbol,Any}()

        t2map, t2dist = T2mapSEcorr(image; Silent = true, t2map_args...)
        t2part = T2partSEcorr(t2dist; Silent = true, t2part_args...)

        # Run CLI with both --T2map and --T2part flags
        mktempdir() do path
            # Write input image to file for reading
            input_basename = joinpath(path, "input")
            input_fname = input_basename * file_suffix
            write_image(input_fname, image)

            # Run main function
            args = [input_fname, cliparamflag..., cliparamval..., "--output", path, "--quiet", "--T2map", "--T2part"]
            # println("*  T2mapSEcorr CLI test with args: " * join(args, " "))
            run_main(args, make_settings_file)

            # Read in outputs and compare
            t2maps_file, t2dist_file, t2parts_file, settings_file =
                input_basename .* (".t2maps.mat", ".t2dist.mat", ".t2parts.mat", ".settings.txt")
            @test isfile(t2maps_file);  t2maps_cli  = DECAES.MAT.matread(t2maps_file)
            @test isfile(t2dist_file);  t2dist_cli  = DECAES.MAT.matread(t2dist_file)["dist"]
            @test isfile(t2parts_file); t2parts_cli = DECAES.MAT.matread(t2parts_file)
            @test !xor(make_settings_file, isfile(settings_file))

            test_compare_t2map((t2map, t2dist), (t2maps_cli, t2dist_cli); rtol = 1e-14)
            test_compare_t2part(t2part, t2parts_cli; rtol = 1e-14)
        end

        # Run CLI with --T2part flag only
        mktempdir() do path
            # Write input t2dist to .mat file for reading
            input_basename = joinpath(path, "input")
            input_fname = input_basename * file_suffix
            write_image(input_fname, t2dist)

            # Run main function
            args = [input_fname, cliparamflag..., cliparamval..., "--output", path, "--quiet", "--T2part"]
            # println("* T2partSEcorr CLI test with args: " * join(args, " "))
            run_main(args, make_settings_file)

            # Read in outputs and compare
            t2maps_file, t2dist_file, t2parts_file, settings_file =
                input_basename .* (".t2maps.mat", ".t2dist.mat", ".t2parts.mat", ".settings.txt")
            @test isfile(t2parts_file); t2parts_cli = DECAES.MAT.matread(t2parts_file)
            @test !isfile(t2maps_file) && !isfile(t2dist_file)
            @test !xor(make_settings_file, isfile(settings_file))

            test_compare_t2part(t2part, t2parts_cli; rtol = 1e-14)
        end
    end
end

# ================================================================================
# UBC MWI Toolbox MATLAB compatibility tests
#   NOTE: For these tests to run, MATLAB must be installed on your default path.
#   Additionally, the MWI NNLS toolbox (https://github.com/ubcmri/ubcmwf)
#   folder "MWI_NNLS_toolbox_0319" (and subfolders) must be added to your
#   default MATLAB path.
# ================================================================================

try
    @eval using MATLAB

    # Helper functions
    matlabify(x::AbstractString) = String(x)
    matlabify(x::AbstractArray) = Float64.(x)
    matlabify(x::Tuple) = [Float64.(x)...]
    matlabify(x::Bool) = x
    matlabify(x) = map(Float64, x)
    matlabify(kwargs::Base.Iterators.Pairs) = Iterators.flatten([(string(k), matlabify(v)) for (k,v) in kwargs])

    mxT2mapSEcorr(image, maxCores = 6; kwargs...) =
        mxcall(:T2map_SEcorr_nechoes_2019, 2, image, maxCores, matlabify(kwargs)...)

    mxT2partSEcorr(image; kwargs...) =
        mxcall(:T2part_SEcorr_2019, 1, image, matlabify(kwargs)...)

    # T2mapSEcorr parameters which aren't in the MATLAB API
    new_t2map_params = Set{Symbol}([
        :SaveResidualNorm,
        :SaveDecayCurve,
        :SaveNNLSBasis,
        :Silent,
    ])

    # Arbitrary non-default T2mapSEcorr options for testing
    t2map_params_perms = Dict{Symbol, Vector{Any}}(
        :TE               => [9e-3],
        :T1               => [1.1],
        :Threshold        => [250.0],
        :Chi2Factor       => [1.03],
        :nT2              => [30, 59], # Include odd number
        :T2Range          => [(8e-3, 1.0)],
        :RefConAngle      => [175.0],
        :MinRefAngle      => [60.0],
        :nRefAngles       => [4, 12],
        :Reg              => ["no", "chi2", "lcurve"],
        :SetFlipAngle     => [178.0],
        :SaveResidualNorm => [false, true],
        :SaveDecayCurve   => [false, true],
        :SaveRegParam     => [false, true],
        :SaveNNLSBasis    => [false, true],
    )

    # Arbitrary non-default T2partSEcorr options for testing
    t2part_params_perms = Dict{Symbol, Vector{Any}}(
        :T2Range    => [(11e-3, 1.5)],
        :SPWin      => [(12e-3, 28e-3)],
        :MPWin      => [(35e-3, 150e-3)],
        :Sigmoid    => [1.5],
    )

    # Check path is correctly set, and run tests if it is
    mfile_exists(fname) = mxcall(:exist, 1, fname) == 2

    if !mfile_exists("T2map_SEcorr_nechoes_2019") || !mfile_exists("T2part_SEcorr_2019")
        @warn "Files T2map_SEcorr_nechoes_2019.m and T2part_SEcorr_2019.m were not found on the default MATLAB path. " *
              "Modify your default MATLAB path to include these files, restart Julia, and try testing again."
    else
        # Relative tolerance threshold for legacy algorithms to match MATLAB version
        default_rtol = 1e-10

        @testset "T2mapSEcorr" begin
            image = DECAES.mock_image(nTE = 32 + rand(0:1))

            for (param,valuelist) in t2map_params_perms, paramval in valuelist
                # The MATLAB implementation of the L-Curve method uses an internal call to `fminbnd`
                # with a tolerance of 1e-3, and therefore the Julia outputs would only match to a
                # tolerance of 1e-3. Additionally, there is a typo in the `G(mu,C_g,d_g)` subfunction:
                #   - Numerator should be ||A*x_mu - b||^2, not ||A*x_mu - b||
                #   - See e.g. equation (1.4) in Fenu, C. et al., 2017, GCV for Tikhonov regularization by partial SVD (https://doi.org/10.1007/s10543-017-0662-0)
                # There is also a small error in methodology:
                #   - Solving regularized ||Ax-b||^2 via [A; mu*I] \ [b; 0] is equivalent to minimizing (note mu^2, not mu):
                #       ||Ax-b||^2 + mu^2||x||^2
                # Below, this test is therefore skipped by default. If you have a version in which these errors are fixed,
                # the below line can be modified (with rtol set appropriately larger than your solver tolerance)
                rtol = default_rtol
                if param === :Reg && paramval == "lcurve"
                    continue
                    # rtol = 1e-3
                end

                jl_kwargs = Dict{Symbol,Any}(:SaveRegParam => true, :legacy => true, :Silent => true) # default settings
                jl_kwargs[param] = paramval

                mat_kwargs = Dict{Symbol,Any}(:Save_regparam => "yes") # default settings
                if param == :SaveRegParam # renamed parameter
                    mat_kwargs[:Save_regparam] = ifelse(paramval, "yes", "no") # renamed parameter
                elseif param == :nRefAngles # renamed parameter
                    mat_kwargs[:nAngles] = paramval
                elseif param == :RefConAngle # renamed parameter
                    mat_kwargs[:RefCon] = paramval
                elseif param ∉ new_t2map_params # skip Julia-only parameters
                    mat_kwargs[param] = paramval
                end

                # Run T2mapSEcorr
                t2map_out_jl  = T2mapSEcorr(image; jl_kwargs...)
                t2map_out_mat = mxT2mapSEcorr(image; mat_kwargs...)
                allpassed = test_compare_t2map(t2map_out_jl, t2map_out_mat; rtol = rtol)
                !allpassed && println("t2map failed: $param = $paramval")
            end
        end

        @testset "T2partSEcorr" begin
            T2dist = DECAES.mock_T2_dist()

            for (param,valuelist) in t2part_params_perms, paramval in valuelist
                # Run T2partSEcorr
                jl_kwargs  = Dict{Symbol,Any}(param => paramval, :legacy => true, :Silent => true)
                mat_kwargs = Dict{Symbol,Any}(param => paramval)
                t2part_jl  = T2partSEcorr(T2dist; jl_kwargs...)
                t2part_mat = mxT2partSEcorr(T2dist; mat_kwargs...)
                allpassed = test_compare_t2part(t2part_jl, t2part_mat; rtol = default_rtol)
                !allpassed && println("t2part failed: $param = $paramval")
            end
        end
    end
catch e
    @warn "using MATLAB failed; skipping MATLAB tests"
    @warn sprint(showerror, e, catch_backtrace())
end

nothing
