using DECAES, JET

function test_mock_T2_dist()
    # Mock T2 distribution, computed with default parameters
    opts = DECAES.mock_t2map_opts(Float64; Silent = true)
    image = DECAES.mock_image(opts)
    @report_opt T2mapSEcorr(image, opts)
end

function test_voxelwise_T2_distribution()
    # Inner loop of T2mapSEcorr, with default parameters
    opts = DECAES.mock_t2map_opts(Float64; Silent = true)
    maps = DECAES.T2Maps(opts)
    dist = DECAES.T2Distributions(opts)
    thread_buffer = DECAES.thread_buffer_maker(opts)
    signal = rand(opts.nTE)
    I = CartesianIndex(1,1,1)
    DECAES.voxelwise_T2_distribution!(thread_buffer, maps, dist, signal, opts, I) # warmup run
    @report_opt DECAES.voxelwise_T2_distribution!(thread_buffer, maps, dist, signal, opts, I)
end

nlogfile = readdir(@__DIR__) |> fs -> filter(!isnothing, match.(r".*\.(\d+)\.log", fs)) |> ms -> isempty(ms) ? 0 : maximum(parse.(Int, first.(ms)))
logfile = joinpath(@__DIR__, "jet_results.$(lpad(nlogfile + 1, 3, '0')).log")
report = test_voxelwise_T2_distribution()
write(logfile, sprint(show, report))
