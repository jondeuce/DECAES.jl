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

# Compare t2map results for approximately equality
function test_compare_t2map(out1, out2; kwargs...) 
    for s in keys(out1[1])
        if haskey(out2[1], s)
            @test isapprox(out1[1][s], out2[1][s]; kwargs..., nans = true)
        end
    end
    @test isapprox(out1[2], out2[2]; kwargs..., nans = true)
    return nothing
end

# Compare t2part results for approximately equality
function test_compare_t2part(part1, part2; kwargs...) 
    for s in keys(part1)
        if haskey(part2, s)
            @test isapprox(part1[s], part2[s]; kwargs..., nans = true)
        end
    end
    return nothing
end

# CLI parameter settings to loop over
const cli_params_perms = Dict{Symbol, Any}(
    :Chi2Factor     => 1.025,
    :MPWin          => (0.038, 0.180),
    :MinRefAngle    => 55.0,
    :RefConAngle    => 172.0,
    :Reg            => "chi2",
    :SPWin          => (0.013, 0.037),
    :SaveNNLSBasis  => true,
    :SaveRegParam   => true,
    :SetFlipAngle   => 170.0,
    :Sigmoid        => 1.0,
    :Silent         => true,
    :T1             => 0.95,
    :T2Range        => (0.016, 1.8),
    :TE             => 0.011,
    :Threshold      => 190.0,
    :nRefAngles     => 9,
    :nRefAnglesMin  => 4,
    :nT2            => 45, # Include odd number
)

@testset "CLI" begin
    image = DECAES.mock_image(;nTE = 32 + rand(0:1))

    make_settings_perms = [false, true]
    file_suffix_perms = [".mat", ".nii", ".nii.gz"]
    iters = (cli_params_perms, make_settings_perms, file_suffix_perms)
    nloop = max(length.(iters)...)
    repeat_until(x) = Iterators.take(Iterators.cycle(x), nloop)

    for params in zip(map(repeat_until, iters)...)
        (k, v), make_settings_file, file_suffix = params
        flag = "--" * string(k) # CLI flag
        vals = v isa Tuple ? [string(x) for x in v] : # Pass each arg separately
               v isa Bool  ? [] : # No arg necessary, flag only
               [string(v)] # Pass string

        # Run T2map and T2part through Julia API for comparison
        t2map_args  = k ∈ fieldnames(T2mapOptions)  ? Dict(k => v) : Dict{Symbol,Any}()
        t2part_args = k ∈ fieldnames(T2partOptions) ? Dict(k => v) : Dict{Symbol,Any}()

        t2map, t2dist = T2mapSEcorr(image; Silent = true, t2map_args...)
        t2part = T2partSEcorr(t2dist; Silent = true, t2part_args...)

        # Run CLI with both --T2map and --T2part flags
        mktempdir() do path
            # Write input image to file for reading
            input_basename = joinpath(path, "input")
            input_fname = input_basename * file_suffix
            write_image(input_fname, image)

            # Run main function
            args = [input_fname, flag, vals..., "--output", path, "--quiet", "--T2map", "--T2part"]
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
            args = [input_fname, flag, vals..., "--output", path, "--quiet", "--T2part"]
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
    matlabify(x::AbstractString) = x
    matlabify(x::Bool) = x
    matlabify(x::Tuple) = [Float64.(x)...]
    matlabify(x) = Float64(x)
    matlabify(kwargs::Base.Iterators.Pairs) = Iterators.flatten([(string(k), matlabify(v)) for (k,v) in kwargs])

    mxT2mapSEcorr(image, maxCores = 6; kwargs...) =
        mxcall(:T2map_SEcorr_nechoes_2019, 2, image, maxCores, matlabify(kwargs)...)

    mxT2partSEcorr(image; kwargs...) =
        mxcall(:T2part_SEcorr_2019, 1, image, matlabify(kwargs)...)

    # T2mapSEcorr parameters which aren't in the MATLAB API
    new_t2map_params = Set{Symbol}([
        :SaveNNLSBasis,
        :Silent,
    ])

    # Arbitrary non-default T2mapSEcorr options for testing
    t2map_params_perms = Dict{Symbol, Vector{Any}}(
        :TE            => [0.009],
        :T1            => [1.1],
        :Threshold     => [250.0],
        :Chi2Factor    => [1.03],
        :nT2           => [30, 59], # Include odd number
        :T2Range       => [(0.010, 1.0)],
        :RefConAngle   => [175.0],
        :MinRefAngle   => [60.0],
        :nRefAngles    => [4, 12],
        :Reg           => ["no", "chi2", "lcurve"],
        :SetFlipAngle  => [178.0],
        :SaveRegParam  => [false, true],
        :SaveNNLSBasis => [false, true],
    )

    # Arbitrary non-default T2partSEcorr options for testing
    t2part_params_perms = Dict{Symbol, Vector{Any}}(
        :T2Range    => [(0.010, 1.5)],
        :SPWin      => [(0.010, 0.028)],
        :MPWin      => [(0.035, 0.150)],
        :Sigmoid    => [1.5],
    )

    # Check path is correctly set, and run tests if it is
    mfile_exists(fname) = mxcall(:exist, 1, fname) == 2

    if !mfile_exists("T2map_SEcorr_nechoes_2019") || !mfile_exists("T2part_SEcorr_2019")
        @warn "Files T2map_SEcorr_nechoes_2019.m and T2part_SEcorr_2019.m " *
            "were not found on the default MATLAB path. Modify your MATLAB path " *
            "to include these files, restart Julia, and try testing again."
    else
        @testset "T2mapSEcorr" begin
            image = DECAES.mock_image(;nTE = 32 + rand(0:1))

            for (k,vlist) in t2map_params_perms, v in vlist
                if k === :Reg && v == "lcurve"
                    # The MATLAB implementation of the L-Curve method uses an internal call to "fminbnd"
                    # with a tolerance of 1e-3, and therefore the Julia outputs would only match to a
                    # tolerance of 1e-3. Additionally, there is a typo in the return value of the
                    # "G(mu,C_g,d_g)" subfunction: the numerator of the result "g" is square rooted, when
                    # it should be left squared. See e.g. equation (1.4) in:
                    #   Fenu, C. et al., 2017, GCV for Tikhonov regularization by partial SVD
                    #   https://doi.org/10.1007/s10543-017-0662-0
                    continue
                end

                # println("\n\n# -------- T2mapSEcorr with key = $k, value = $v -------- #\n")
                jl_kwargs = Dict{Symbol,Any}(k => v, :legacy => true, :Silent => true)

                mat_kwargs = Dict{Symbol,Any}()
                if k == :SaveRegParam # renamed parameter
                    mat_kwargs[:Save_regparam] = ifelse(v, "yes", "no") # renamed parameter
                elseif k == :nRefAngles # renamed parameter
                    mat_kwargs[:nAngles] = v
                elseif k == :RefConAngle # renamed parameter
                    mat_kwargs[:RefCon] = v
                elseif k ∉ new_t2map_params # skip Julia-only parameters
                    mat_kwargs[k] = v
                end

                # Run T2mapSEcorr
                t2map_out_jl  = T2mapSEcorr(image; jl_kwargs...)
                t2map_out_mat = mxT2mapSEcorr(image; mat_kwargs...)
                test_compare_t2map(t2map_out_jl, t2map_out_mat; rtol = 1e-10)
            end
        end

        @testset "T2partSEcorr" begin
            T2dist = DECAES.mock_T2_dist()

            for (k,vlist) in t2part_params_perms, v in vlist
                # println("\n\n# -------- T2partSEcorr with key = $k, value = $v -------- #\n")

                # Run T2partSEcorr
                jl_kwargs  = Dict{Symbol,Any}(k => v, :legacy => true, :Silent => true)
                mat_kwargs = Dict{Symbol,Any}(k => v)
                t2part_jl  = T2partSEcorr(T2dist; jl_kwargs...)
                t2part_mat = mxT2partSEcorr(T2dist; mat_kwargs...)
                test_compare_t2part(t2part_jl, t2part_mat; rtol = 1e-10)
            end
        end
    end
catch e
    @warn "using MATLAB failed; skipping MATLAB tests"
    @warn sprint(showerror, e, catch_backtrace())
end

nothing
