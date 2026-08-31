Random.seed!(0) # reproducible randomized tests (image params, arg permutations, tolerances)

# Arbitrary default required parameters used during testing (nTE and nT2 handled separately)
default_paramdict = Dict{Symbol, Any}(
    :TE => 8e-3,
    :T2Range => (12e-3, 1.8),
    :SPWin => (12e-3, 37e-3),
    :MPWin => (37e-3, 650e-3),
    :Reg => "lcurve",
)

# Write 4D image to disk
function write_image(filename, image)
    if endswith(filename, ".mat")
        DECAES.MAT.matwrite(filename, Dict("img" => image))
    else
        DECAES.NIfTI.niwrite(filename, DECAES.NIfTI.NIVolume(image))
    end
end

# Build t2map and t2part arguments for calling `DECAES.main` via the CLI or Julia API
function construct_args(
    paramdict;
    argstype,
    inputfilename = nothing,
    outputpath = nothing,
    quiet::Bool = true,
    T2map::Bool = true,
    T2part::Bool = true,
)

    paramdict = copy(paramdict)
    if argstype === :cli
        #### CLI

        args = [inputfilename, rand(["--output", "-o"]), outputpath]
        T2map && push!(args, "--T2map")
        T2part && push!(args, "--T2part")
        quiet && push!(args, rand(["--quiet", "-q"]))

        for (param, paramval) in paramdict
            param ∈ (:Silent, :Threaded) && continue # params not handled by CLI
            paramval === nothing && continue # `nothing` is always default if allowable, therefore no flag/val is passed
            paramval isa Bool && !paramval && continue # only pass boolean flags if true
            push!(args, "--" * string(param)) # push flag name
            if paramval isa Float64 || paramval isa Int || paramval isa Tuple
                append!(args, [string(x) for x in paramval]) # pass each arg separately
            elseif paramval isa String
                push!(args, paramval) # pass string
            elseif paramval isa Bool
                # pass nothing
            else
                error("Unsupported type for paramval: $(typeof(paramval))")
            end
        end

        return args

    elseif argstype === :jl
        #### Julia

        t2map_args  = T2map ? Dict{Symbol, Any}() : nothing
        t2part_args = T2part ? Dict{Symbol, Any}() : nothing

        t2map_fields = DECAES.fieldsof(T2mapOptions, Set)
        t2part_fields = DECAES.fieldsof(T2partOptions, Set)

        if paramdict[:Reg] == "chi2"
            paramdict[:Chi2Factor] = paramdict[:RegParams]
        elseif paramdict[:Reg] == "mdp"
            paramdict[:NoiseLevel] = paramdict[:RegParams]
        end

        for (param, paramval) in paramdict
            T2map && (param ∈ t2map_fields) && (t2map_args[param] = paramval)
            T2part && (param ∈ t2part_fields) && (t2part_args[param] = paramval)
        end

        return t2map_args, t2part_args
    end
end

# Populate `paramdict` with random image parameters
function image_params!(paramdict)
    # Image parameters
    get!(paramdict, :MatrixSize, (2, 2, 2))
    get!(paramdict, :nTE, rand(4:64))
    get!(paramdict, :nT2, rand(4:64))
    return paramdict
end

# Generate a mock 4D image for testing
function construct_test_image(paramdict; kwargs...)
    image = DECAES.mock_image(;
        MatrixSize = paramdict[:MatrixSize],
        nTE = paramdict[:nTE],
        nT2 = paramdict[:nT2],
        kwargs...,
    )
    image ./= mean(@views image[:, :, :, 1]) # normalize first-echo signal intensity to unit mean
    return image
end

# Call main function on image file `image`
function run_main(image, args; make_settings_file::Bool)
    # Write input image to file for reading
    inputfilename = args[1]
    outputpath = args[3]
    inputfilebasename = joinpath(outputpath, "input")
    write_image(inputfilename, image)

    # Run main, possibly writing CLI args to settings file first
    try
        if make_settings_file
            settings_file = joinpath(outputpath, "settings.txt")
            open(settings_file, "w") do file
                return println(file, join(args, "\n"))
            end
            DECAES.redirect_to_devnull() do
                return main(["@" * settings_file])
            end
        else
            DECAES.redirect_to_devnull() do
                return main(args)
            end
        end
    catch e
        @info "CLI failed with settings:"
        display(args)
        rethrow(e)
    end

    # Check that only requested files were created
    t2maps_file, t2dist_file, t2parts_file, settings_file = inputfilebasename .* (".t2maps.mat", ".t2dist.mat", ".t2parts.mat", ".settings.txt")
    T2map, T2part = ("--T2map" ∈ args), ("--T2part" ∈ args)

    @test !xor(T2map, isfile(t2maps_file))
    @test !xor(T2map, isfile(t2dist_file))
    @test !xor(T2part, isfile(t2parts_file))
    @test !xor(make_settings_file, isfile(settings_file))

    t2maps  = T2map ? DECAES.MAT.matread(t2maps_file) : nothing
    t2dist  = T2map ? DECAES.MAT.matread(t2dist_file)["dist"] : nothing
    t2parts = T2part ? DECAES.MAT.matread(t2parts_file) : nothing

    return (; t2maps, t2dist, t2parts)
end

function showall(; kwargs...)
    for (k, v) in kwargs
        @info string(k) * " => " * sprint(show, MIME"text/plain"(), v)
    end
end

function test_field!(allpassed, x, y, prefix = "failed:"; atol = 0.0, rtol = atol > 0 ? 0.0 : √eps())
    passed = size(x) == size(y) && isapprox(x, y; atol, rtol, nans = true)
    allpassed[] &= passed
    !passed && @warn prefix * " (" * field_error_string(x, y) * ")"
    @test x ≈ y atol = atol rtol = rtol nans = true
end
field_error_string(x, y) = size(x) != size(y) ? "size(x) = $(size(x)), size(y) = $(size(y))" : "size = $(size(y)), max val = $(maximum(abs, y)), max diff = $(maximum(abs, x.-y)), rel diff = $(maximum(abs, (x.-y)./y))"

# Compare t2map results for approximately equality. Fields named in `skip` are not compared.
function test_compare_t2map(maps1, dist1, maps2, dist2; skip = [], kwargs...)
    allpassed = Ref(true)
    for s in keys(maps1)
        haskey(maps2, s) && s ∉ skip && test_field!(allpassed, maps1[s], maps2[s], "maps failed: $s"; kwargs...)
    end
    "dist" ∉ skip && test_field!(allpassed, dist1, dist2, "dist failed"; kwargs...)
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
#   -Each list should contain *only* non-default and/or edge-case values
function run_cli_tests()
    cli_params_perms = Any[
        (:MPWin .=> [(38e-3, 180e-3)],),
        (:MinRefAngle .=> [55.0],),
        (:RefConAngle .=> [172.0],),
        (
            :Reg       .=> ["none", "gcv", "lcurve", "reginska", "chi2", "mdp"],
            :RegParams .=> [nothing, nothing, nothing, nothing, 1.025, 3e-4],
        ),
        (
            :Reg       .=> ["lcurve", "reginska", "chi2", "mdp"],
            :RegNorm   .=> ["l1", "l1", "l1", "l1"],
            :RegParams .=> [nothing, nothing, 1.025, 3e-4],
        ),
        (:SPWin .=> [(13e-3, 37e-3)],),
        (:SaveResidualNorm .=> [true],),
        (:SaveDecayCurve .=> [true],),
        (:SaveNNLSBasis .=> [true],),
        (:SaveRegParam .=> [true],),
        (:SetFlipAngle .=> [170.0],),
        (:Sigmoid .=> [1.0],),
        (:Silent .=> [true],),
        (:T1 .=> [0.95],),
        (:T2Range .=> [(16e-3, 1.8)],),
        (:TE .=> [11e-3],),
        (:Threaded .=> [true],),
        (:Threshold .=> [1.0, Inf, -Inf],), # Include non-zero and infinite (i.e. either some, all, or no voxels skipped)
        (:nRefAngles .=> [9, 10],), # Include even/odd
        (:nRefAnglesMin .=> [4, 7],), # Include even/odd
        (:nTE .=> [4, 5, 8, 47],), # Include even/odd, and minimum number (4)
        (:nT2 .=> [2, 3, 8, 47],), # Include even/odd, and minimum number (2)
    ]

    make_settings_perms = [false, true]
    file_suffix_perms = [".mat", ".nii", ".nii.gz"] # Note: no PAR/REC or XML/REC, since we can't write to them
    B1map_perms = [false, true]
    param_perms = (cli_params_perms, make_settings_perms, file_suffix_perms, B1map_perms)
    repeat_until(x) = Iterators.take(Iterators.cycle(x), maximum(length, param_perms))

    for (param_val_lists, make_settings_file, file_suffix, B1map) in zip(map(repeat_until, param_perms)...), param_val_pairs in zip(param_val_lists...)
        paramdict = deepcopy(default_paramdict)
        for (param, paramval) in param_val_pairs
            paramdict[param] = paramval
        end

        image_params!(paramdict)
        image = construct_test_image(paramdict)

        settings_kwargs_jl = Dict{Symbol, Any}(:argstype => :jl, :quiet => rand([true, false]), :T2map => true, :T2part => true)
        settings_kwargs_cli = Dict{Symbol, Any}(:argstype => :cli, :quiet => rand([true, false]), :T2map => true, :T2part => true)
        jl_t2map_kwargs, jl_t2part_kwargs = construct_args(paramdict; settings_kwargs_jl...)

        # Run T2map and T2part through Julia API for comparison
        t2map, t2dist = DECAES.redirect_to_devnull() do
            return T2mapSEcorr(image; jl_t2map_kwargs...)
        end
        t2part = DECAES.redirect_to_devnull() do
            return T2partSEcorr(t2dist; jl_t2part_kwargs...)
        end

        # Run CLI with both --T2map and --T2part flags
        mktempdir() do path
            settings_kwargs_cli[:outputpath] = path
            settings_kwargs_cli[:inputfilename] = joinpath(path, "input" * file_suffix)
            cli_t2map_args = construct_args(paramdict; settings_kwargs_cli...)

            # The CLI is compared against a Julia API run given the same inputs, so that what is tested here is the CLI.
            # Passing a B1 map changes the input, so the reference run is repeated with the same map and both runs then see the same basis.
            # That a rerun given the fitted α reproduces the fit is a separate claim, asserted by `flip angle round trip` in `t2map.jl`.
            B1_passed = B1map && !("--SetFlipAngle" ∈ cli_t2map_args)
            t2map_ref, t2dist_ref, t2part_ref = t2map, t2dist, t2part
            if B1_passed
                B1mapfilename = joinpath(path, "B1" * file_suffix)
                write_image(B1mapfilename, t2map["alpha"])
                append!(cli_t2map_args, ["--B1map", B1mapfilename])

                opts_B1 = DECAES.T2mapOptions(image; jl_t2map_kwargs...)
                maps_B1 = DECAES.T2Maps(opts_B1)
                DECAES.load_B1map!(maps_B1, DECAES.load_image(B1mapfilename, Val(3)))
                t2map_ref, t2dist_ref = DECAES.redirect_to_devnull() do
                    return DECAES.T2mapSEcorr!(maps_B1, DECAES.T2Distributions(opts_B1), image, opts_B1)
                end
                t2part_ref = DECAES.redirect_to_devnull() do
                    return T2partSEcorr(t2dist_ref; jl_t2part_kwargs...)
                end
            end

            t2maps_cli, t2dist_cli, t2parts_cli = run_main(image, cli_t2map_args; make_settings_file)

            t2map_passed = test_compare_t2map(t2map_ref, t2dist_ref, t2maps_cli, t2dist_cli; rtol = 1e-14)
            t2part_passed = test_compare_t2part(t2part_ref, t2parts_cli; rtol = 1e-14)
            if !(t2map_passed && t2part_passed)
                println("\n ------------------------------- \n")
                @error "CLI with --T2map and --T2part failed"
                showall(; param_val_pairs, paramdict, jl_t2map_kwargs, jl_t2part_kwargs, cli_t2map_args)
                println("\n ------------------------------- \n")
            end
        end

        # Run CLI with --T2part flag only
        mktempdir() do path
            settings_kwargs_cli[:outputpath] = path
            settings_kwargs_cli[:inputfilename] = joinpath(path, "input" * file_suffix)
            settings_kwargs_cli[:T2map] = false
            cli_t2part_args = construct_args(paramdict; settings_kwargs_cli...)

            t2maps_cli, t2dist_cli, t2parts_cli = run_main(t2dist, cli_t2part_args; make_settings_file)
            t2part_passed = test_compare_t2part(t2part, t2parts_cli; rtol = 1e-14)
            if !t2part_passed
                println("\n ------------------------------- \n")
                @error "CLI with --T2part only failed"
                showall(; param_val_pairs, paramdict, jl_t2map_kwargs, jl_t2part_kwargs, cli_t2part_args)
                println("\n ------------------------------- \n")
            end
        end
    end
end

@testset "Command line interface" begin
    run_cli_tests()
end

nothing
