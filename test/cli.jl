using DECAES
using Test
using Logging

# Arbitrary default required parameters used during testing
const default_paramdict = Dict{Symbol,Any}(
    :TE => 8e-3,
    :nT2 => 42,
    :T2Range => (12e-3, 1.8),
    :SPWin => (12e-3, 37e-3),
    :MPWin => (37e-3, 650e-3),
)

# Legacy settings which previously had defaults
const legacy_default_paramdict = Dict{Symbol,Any}(
    :TE => 10e-3,
    :nT2 => 40,
    :T2Range => (15e-3, 2.0),
    :SPWin => (14e-3, 40e-3),
    :MPWin => (40e-3, 200e-3),
)

# Write 4D image to disk
function write_image(filename, image)
    if endswith(filename, ".mat")
        DECAES.MAT.matwrite(filename, Dict("img" => image))
    else
        DECAES.NIfTI.niwrite(filename, DECAES.NIfTI.NIVolume(image))
    end
end

# https://discourse.julialang.org/t/redirect-stdout-and-stderr/13424/3
function redirect_to_tempfiles(f)
    open(tempname() * ".log", "w") do out
        open(tempname() * ".err", "w") do err
            redirect_stdout(out) do
                redirect_stderr(err) do
                    f()
                end
            end
        end
    end
end

# Call main function on image file `image`
function run_main(image, args; make_settings_file::Bool)

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

    return DECAES.@ntuple(t2maps, t2dist, t2parts)
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
        for (k,v) in legacy_default_paramdict
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
            if !isnothing(paramval)
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

        for (param, paramval) in paramdict
            T2map  && (param ∈ fieldnames(T2mapOptions))  && setindex!(t2map_args, paramval, param)
            T2part && (param ∈ fieldnames(T2partOptions)) && setindex!(t2part_args, paramval, param)
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
    :Threshold        => [125_000.0, Inf], # Include non-zero and infinite (i.e. either some or all voxels skipped)
    :nRefAngles       => [9, 10], # Include odd number
    :nRefAnglesMin    => [4, 5], # Include odd number
    :nT2              => [2, 47], # Include odd number
)

@testset "CLI" begin
    make_settings_perms = [false, true]
    file_suffix_perms = [".mat", ".nii", ".nii.gz"] # Note: no PAR/REC or XML/REC, since we can't write to them
    iters = (cli_params_perms, make_settings_perms, file_suffix_perms)
    nloop = max(length.(iters)...)
    repeat_until(x) = Iterators.take(Iterators.cycle(x), nloop)

    for ((param, valuelist), make_settings_file, file_suffix) in zip(map(repeat_until, iters)...), paramval in valuelist, legacy in [false, true]
        image = DECAES.mock_image(nTE = rand([4,5,20,67]))
        paramdict = deepcopy(default_paramdict)
        paramdict[param] = paramval

        settings_kwargs_jl = Dict{Symbol, Any}(:argstype => :jl, :quiet => rand([true,false]), :legacy => legacy, :T2map => true, :T2part => true)
        settings_kwargs_cli = Dict{Symbol, Any}(:argstype => :cli, :quiet => rand([true,false]), :legacy => legacy, :T2map => true, :T2part => true)

        # Run T2map and T2part through Julia API for comparison
        jl_t2map_kwargs, jl_t2part_kwargs = construct_args(paramdict; settings_kwargs_jl...)
        t2map, t2dist = with_logger(NullLogger()) do
            T2mapSEcorr(image; jl_t2map_kwargs...)
        end
        t2part = with_logger(NullLogger()) do
            T2partSEcorr(t2dist; jl_t2part_kwargs...)
        end

        # Run CLI with both --T2map and --T2part flags
        mktempdir() do path
            settings_kwargs_cli[:outputpath] = path
            settings_kwargs_cli[:inputfilename] = joinpath(path, "input" * file_suffix)
            cli_t2map_args = construct_args(paramdict; settings_kwargs_cli...)

            t2maps_cli, t2dist_cli, t2parts_cli = redirect_to_tempfiles() do
                run_main(image, cli_t2map_args; make_settings_file = make_settings_file)
            end
            t2map_passed = test_compare_t2map((t2map, t2dist), (t2maps_cli, t2dist_cli); rtol = 1e-14)
            t2part_passed = test_compare_t2part(t2part, t2parts_cli; rtol = 1e-14)
            if !(t2map_passed && t2part_passed)
                println("\n ------------------------------- \n")
                @show paramdict
                @show jl_t2map_kwargs
                @show jl_t2part_kwargs
                @show cli_t2map_args
                println("\n ------------------------------- \n")
            end
        end

        # Run CLI with --T2part flag only
        mktempdir() do path
            settings_kwargs_cli[:outputpath] = path
            settings_kwargs_cli[:inputfilename] = joinpath(path, "input" * file_suffix)
            settings_kwargs_cli[:T2map] = false
            cli_t2part_args = construct_args(paramdict; settings_kwargs_cli...)

            t2maps_cli, t2dist_cli, t2parts_cli = redirect_to_tempfiles() do
                run_main(t2dist, cli_t2part_args; make_settings_file = make_settings_file)
            end
            t2part_passed = test_compare_t2part(t2part, t2parts_cli; rtol = 1e-14)
            if !t2part_passed
                println("\n ------------------------------- \n")
                @show paramdict
                @show jl_t2map_kwargs
                @show jl_t2part_kwargs
                @show cli_t2part_args
                println("\n ------------------------------- \n")
            end
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

# Helper functions
matlabify(x::AbstractString) = String(x)
matlabify(x::AbstractArray) = Float64.(x)
matlabify(x::Tuple) = [Float64.(x)...]
matlabify(x::Bool) = x
matlabify(x) = map(Float64, x)
matlabify(kwargs::Base.Iterators.Pairs) = Iterators.flatten([(string(k), matlabify(v)) for (k,v) in kwargs])

mfile_exists(fname) = MATLAB.mxcall(:exist, 1, fname) == 2

mxT2mapSEcorr(image, maxCores = 6; kwargs...) =
    MATLAB.mxcall(:T2map_SEcorr_nechoes_2019, 2, image, maxCores, matlabify(kwargs)...)

mxT2partSEcorr(image; kwargs...) =
    MATLAB.mxcall(:T2part_SEcorr_2019, 1, image, matlabify(kwargs)...)

# Arbitrary non-default T2mapSEcorr options for testing
const mat_t2map_params_perms = Dict{Symbol, Vector{Any}}(
    :TE               => [9e-3],
    :T1               => [1.1],
    :Threshold        => [250.0],
    :Chi2Factor       => [1.03],
    :nT2              => [10, 59], # Include odd number
    :T2Range          => [(8e-3, 1.0)],
    :RefConAngle      => [175.0],
    :MinRefAngle      => [60.0],
    :nRefAngles       => [7, 12],
    :Reg              => ["no", "chi2", "lcurve"],
    :SetFlipAngle     => [178.0],
    :SaveResidualNorm => [false, true],
    :SaveDecayCurve   => [false, true],
    :SaveRegParam     => [false, true],
    :SaveNNLSBasis    => [false, true],
)

# Arbitrary non-default T2partSEcorr options for testing
const mat_t2part_params_perms = Dict{Symbol, Vector{Any}}(
    :T2Range    => [(11e-3, 1.5)],
    :SPWin      => [(12e-3, 28e-3)],
    :MPWin      => [(35e-3, 150e-3)],
    :Sigmoid    => [1.5],
)

function matlab_tests()
    # Relative tolerance threshold for legacy algorithms to match MATLAB version
    default_rtol = 1e-10

    @testset "T2mapSEcorr" begin
        settings_kwargs_jl = Dict{Symbol, Any}(:argstype => :jl, :quiet => rand([true,false]), :legacy => true, :T2map => true, :T2part => true)
        settings_kwargs_mat = Dict{Symbol, Any}(:argstype => :mat, :quiet => rand([true,false]), :legacy => true, :T2map => true, :T2part => true)

        for (param,valuelist) in mat_t2map_params_perms, paramval in valuelist
            # The MATLAB implementation of the L-Curve method uses an internal call to `fminbnd`
            # with a tolerance of 1e-3, and therefore the Julia outputs would only match to at best
            # a tolerance of 1e-3. Additionally, there is a typo in the `G(mu,C_g,d_g)` subfunction:
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

            paramdict = deepcopy(legacy_default_paramdict)
            paramdict[param] = paramval
            jl_t2map_kwargs,  _ = construct_args(paramdict; settings_kwargs_jl...)
            mat_t2map_kwargs, _ = construct_args(paramdict; settings_kwargs_mat...)

            # Run T2mapSEcorr
            image = DECAES.mock_image(nTE = rand([4,5,20,67]))
            t2map_out_jl = with_logger(NullLogger()) do
                T2mapSEcorr(image; jl_t2map_kwargs...)
            end
            t2map_out_mat = redirect_to_tempfiles() do
                mxT2mapSEcorr(image; mat_t2map_kwargs...)
            end
            allpassed = test_compare_t2map(t2map_out_jl, t2map_out_mat; rtol = rtol)
            !allpassed && println("t2map failed: $param = $paramval")
        end
    end

    @testset "T2partSEcorr" begin
        settings_kwargs_jl = Dict{Symbol, Any}(:argstype => :jl, :quiet => rand([true,false]), :legacy => true, :T2map => false, :T2part => true)
        settings_kwargs_mat = Dict{Symbol, Any}(:argstype => :mat, :quiet => rand([true,false]), :legacy => true, :T2map => false, :T2part => true)

        for (param,valuelist) in mat_t2part_params_perms, paramval in valuelist
            paramdict = deepcopy(legacy_default_paramdict)
            paramdict[param] = paramval
            delete!(paramdict, :nT2) # inferred
            _, jl_t2part_kwargs  = construct_args(paramdict; settings_kwargs_jl...)
            _, mat_t2part_kwargs = construct_args(paramdict; settings_kwargs_mat...)

            # Run T2partSEcorr
            T2dist = DECAES.mock_T2_dist(nT2 = rand([5,19,20,40,60])) #rand([4,5,40,41]))
            t2part_jl = with_logger(NullLogger()) do
                T2partSEcorr(T2dist; jl_t2part_kwargs...)
            end
            t2part_mat = redirect_to_tempfiles() do
                mxT2partSEcorr(T2dist; mat_t2part_kwargs...)
            end
            allpassed = test_compare_t2part(t2part_jl, t2part_mat; rtol = default_rtol)
            !allpassed && println("t2part failed: $param = $paramval")
        end
    end
end

# Try loading MATLAB.jl and running tests
try
    @eval using MATLAB

    if mfile_exists("T2map_SEcorr_nechoes_2019") && mfile_exists("T2part_SEcorr_2019")
        matlab_tests()
    else
        @warn "Files T2map_SEcorr_nechoes_2019.m and T2part_SEcorr_2019.m were not found on the default MATLAB path. " *
            "Modify your default MATLAB path to include these files. For example, add a command such as" *
            "\n\n    addpath /path/to/MWI_NNLS_toolbox_0319\n\n" *
            "to your startup.m file using the appropriate path for your file system. Then, try testing again."
    end
catch e
    @warn "Failed to load Julia package MATLAB.jl; skipping UBCMWF MATLAB tests"
    @warn sprint(showerror, e, catch_backtrace())
end

nothing