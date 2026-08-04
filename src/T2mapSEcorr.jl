# Internal convenience container for holding T2 maps outputs
@with_kw_noshow struct T2Maps{T}
    echotimes::Vector{T}
    t2times::Vector{T}
    refangleset::Union{T, Vector{T}}
    decaybasisset::Union{Matrix{T}, Array{T, 3}}
    gdn::Array{T, 3}
    ggm::Array{T, 3}
    gva::Array{T, 3}
    fnr::Array{T, 3}
    snr::Array{T, 3}
    alpha::Array{T, 3}
    is_alpha_provided::Base.RefValue{Bool}
    resnorm::Union{Nothing, Array{T, 3}}
    decaycurve::Union{Nothing, Array{T, 4}}
    mu::Union{Nothing, Array{T, 3}}
    chi2factor::Union{Nothing, Array{T, 3}}
    decaybasis::Union{Nothing, Matrix{T}, Array{T, 5}}
end

Base.convert(::Type{Dict{Symbol, Any}}, maps::T2Maps) = Dict{Symbol, Any}(Any[f => getfield(maps, f) for f in fieldsof(T2Maps, Vector) if getfield(maps, f) isa Array])
Base.convert(::Type{Dict{String, Any}}, maps::T2Maps) = Dict{String, Any}(Any[string(k) => v for (k, v) in convert(Dict{Symbol, Any}, maps)])

function T2Maps(opts::T2mapOptions{T}) where {T}
    thread_buffer = thread_buffer_maker(opts)
    return T2Maps(;
        # Misc. processing parameters
        echotimes     = convert(Array{T}, copy(opts.TE .* (1:opts.nTE))),
        t2times       = convert(Array{T}, copy(thread_buffer.T2_times)),
        refangleset   = opts.SetFlipAngle === nothing ? convert(Array{T}, copy(thread_buffer.flip_angles)) : T(opts.SetFlipAngle),
        decaybasisset = opts.SetFlipAngle === nothing ?
        convert(Array{T}, copy(thread_buffer.flip_angle_work.decay_basis_set_ensemble.decay_basis_set)) :
        convert(Array{T}, copy(thread_buffer.flip_angle_work.decay_basis)),

        # Default output maps
        gdn = tfill(T(NaN), opts.MatrixSize...),
        ggm = tfill(T(NaN), opts.MatrixSize...),
        gva = tfill(T(NaN), opts.MatrixSize...),
        fnr = tfill(T(NaN), opts.MatrixSize...),
        snr = tfill(T(NaN), opts.MatrixSize...),
        alpha = tfill(T(NaN), opts.MatrixSize...),
        is_alpha_provided = Ref(false),

        # Optional output maps
        resnorm    = !opts.SaveResidualNorm ? nothing : tfill(T(NaN), opts.MatrixSize...),
        decaycurve = !opts.SaveDecayCurve ? nothing : tfill(T(NaN), opts.MatrixSize..., opts.nTE),
        mu         = !opts.SaveRegParam ? nothing : tfill(T(NaN), opts.MatrixSize...),
        chi2factor = !opts.SaveRegParam ? nothing : tfill(T(NaN), opts.MatrixSize...),
        decaybasis = !opts.SaveNNLSBasis ? nothing :
        opts.SetFlipAngle === nothing ?
        tfill(T(NaN), opts.MatrixSize..., opts.nTE, opts.nT2) : # unique decay basis set for each voxel
        convert(Array{T}, copy(thread_buffer.decay_basis)), # single decay basis set used for all voxels
    )
end

function load_B1map!(maps::T2Maps, alpha)
    maps.alpha .= alpha
    maps.is_alpha_provided[] = true
    return maps
end
@inline is_B1map_provided(maps::T2Maps) = maps.is_alpha_provided[]

# Internal convenience container for holding T2 distributions
@with_kw_noshow struct T2Distributions{T}
    distributions::Array{T, 4}
end

function T2Distributions(opts::T2mapOptions{T}) where {T}
    return T2Distributions(;
        distributions = tfill(T(NaN), opts.MatrixSize..., opts.nT2),
    )
end

@inline Base.parent(dist::T2Distributions) = dist.distributions
@inline Base.convert(::Type{Array{T, 4}}, dist::T2Distributions) where {T} = convert(Array{T, 4}, parent(dist))

"""
    T2mapSEcorr(image::Array{T,4}; <keyword arguments>)
    T2mapSEcorr(image::Array{T,4}, opts::T2mapOptions{T})

Uses nonnegative least squares (NNLS) to compute T2 distributions in the presence of stimulated echos by optimizing the refocusing pulse flip angle.
Records parameter maps and T2 distributions for further partitioning.

# Arguments

  - `image`: 4D array with intensity data as `(row, column, slice, echo)`
  - A series of optional keyword argument settings which will be used to construct a [`T2mapOptions`](@ref) struct internally, or a [`T2mapOptions`](@ref) struct directly

# Outputs

  - `maps`: dictionary containing parameter maps with the following fields:

      + **Default Fields**

          * `"echotimes"`     Echo times of time signal (length `nTE` 1D array)
          * `"t2times"`       T2 times corresponding to T2-distributions (length `nT2` 1D array)
          * `"refangleset"`   Refocusing angles used during flip angle optimization (length `nRefAngles` 1D array by default; scalar if `SetFlipAngle` is used)
          * `"decaybasisset"` Decay basis sets corresponding to `"refangleset"` (`nTE x nT2 x nRefAngles` 3D array by default; `nTE x nT2` 2D array if `SetFlipAngle` is used)
          * `"gdn"`:          Map of general density = sum(T2distribution) (`MatrixSize` 3D array)
          * `"ggm"`:          Map of general geometric mean of T2-distribution (`MatrixSize` 3D array)
          * `"gva"`:          Map of general variance (`MatrixSize` 3D array)
          * `"fnr"`:          Map of fit to noise ratio = gdn / √(sum(residuals.^2) / (nTE-1)) (`MatrixSize` 3D array)
          * `"snr"`:          Map of signal to noise ratio = maximum(signal) / std(residuals) (`MatrixSize` 3D array)
          * `"alpha"`:        Map of optimized refocusing pulse flip angle (`MatrixSize` 3D array)

      + **Optional Fields**

          * `"resnorm"`:      ``\\ell^2``-norm of NNLS fit residuals; see `SaveResidualNorm` option (`MatrixSize` 3D array)
          * `"decaycurve"`:   Signal decay curve resulting from NNLS fit; see `SaveDecayCurve` option (`MatrixSize x nTE` 4D array)
          * `"mu"`:           Regularization parameter used during from NNLS fit; see `SaveRegParam` option (`MatrixSize` 3D array)
          * `"chi2factor"`:   ``\\chi^2`` increase factor relative to unregularized NNLS fit; see `SaveRegParam` option (`MatrixSize` 3D array)
          * `"decaybasis"`:   Decay bases resulting from flip angle optimization; see `SaveNNLSBasis` option (`MatrixSize x nTE x nT2` 5D array, or `nTE x nT2` 2D array if `SetFlipAngle` is used)

  - `distributions`: T2-distribution array with data as `(row, column, slice, T2 amplitude)` (`MatrixSize x nT2` 4D array)

# Examples

```julia-repl
julia> image = DECAES.mock_image(; MatrixSize = (100, 100, 1), nTE = 48); # mock image with size 100x100x1x48

julia> maps, dist = T2mapSEcorr(image; TE = 10e-3, nT2 = 40, T2Range = (10e-3, 2.0), Reg = "lcurve", Silent = true); # compute the T2-maps and T2-distribution

julia> maps
Dict{String, Any} with 10 entries:
  "echotimes"     => [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,…
  "t2times"       => [0.01, 0.0114551, 0.013122, 0.0150315, 0.0172188…
  "refangleset"   => [50.0, 54.1935, 58.3871, 62.5806, 66.7742, 70.96…
  "gdn"           => [1.26381 1.27882 … 1.2463 1.25091; 1.29848 1.243…
  "fnr"           => [379.9 437.541 … 446.88 386.396; 485.27 360.591 …
  "alpha"         => [165.461 166.286 … 164.614 164.389; 163.735 164.…
  "gva"           => [0.691794 0.440231 … 0.0490302 0.1253; 0.849798 …
  "ggm"           => [0.0663333 0.0705959 … 0.056455 0.0576729; 0.053…
  "snr"           => [312.773 364.031 … 363.463 313.372; 372.631 313.…
  "decaybasisset" => [0.0277684 0.0315296 … 0.0750511 0.0751058; 0.04…
```

See also:

  - [`T2partSEcorr`](@ref)
  - [`lsqnonneg`](@ref)
  - [`lsqnonneg_tikh`](@ref)
  - [`lsqnonneg_gcv`](@ref)
  - [`lsqnonneg_lcurve`](@ref)
  - [`lsqnonneg_reginska`](@ref)
  - [`lsqnonneg_chi2`](@ref)
  - [`lsqnonneg_mdp`](@ref)
  - [`EPGdecaycurve`](@ref)
"""
T2mapSEcorr(image::Array{T, 4}; kwargs...) where {T} = T2mapSEcorr(image, T2mapOptions(image; kwargs...))
T2mapSEcorr(image::Array{T, 4}, opts::T2mapOptions{T}) where {T} = T2mapSEcorr!(T2Maps(opts), T2Distributions(opts), image, opts)

function T2mapSEcorr!(
    maps::T2Maps{T},
    dist::T2Distributions{T},
    image::Array{T, 4},
    opts::T2mapOptions{T},
) where {T}

    # =========================================================================
    # Initialize output data structures and thread-local buffers
    # =========================================================================
    @assert size(image) == (opts.MatrixSize..., opts.nTE)

    # Print settings to terminal
    !opts.Silent && @info show_string(opts)

    # =========================================================================
    # Process all pixels
    # =========================================================================

    # For each worker in the worker pool, allocate a separate thread-local buffer, then run the work function `work!`.
    global_buffer = global_buffer_maker(opts)
    function with_thread_buffer(work!)
        thread_buffer = thread_buffer_maker(opts, global_buffer)
        return work!(thread_buffer)
    end

    # Run analysis in parallel
    indices = filter(I -> image[I, 1] > opts.Threshold, CartesianIndices(opts.MatrixSize))
    if isempty(indices)
        !opts.Silent && @warn "No voxels found with first-echo signal intensity above threshold $(opts.Threshold).\nConsider lowering the threshold or checking the input data."
        return convert(Dict{String, Any}, maps), convert(Array{T, 4}, dist)
    end
    ntasks = opts.Threaded ? Threads.nthreads() : 1
    indices_blocks = split_indices(; length = length(indices), minchunksize = default_blocksize())

    # Run analysis in parallel
    with_singlethreaded_blas() do
        workerpool(with_thread_buffer, indices_blocks; ntasks, verbose = !opts.Silent) do inds, thread_buffer
            reset_voxel_chains!(thread_buffer)
            GC.@preserve thread_buffer maps dist image @inbounds for j in inds
                I = indices[j]
                voxelwise_T2_distribution!(thread_buffer, maps, dist, uview(image, I, :), opts, I)
            end
        end
    end

    return convert(Dict{String, Any}, maps), convert(Array{T, 4}, dist)
end

# Reset every cross-voxel warm-start chain, namely the flip-search per-gridpoint active sets, at the start of each voxel block.
# Without this, a chain persists across whichever blocks a worker happens to pull, so the dynamic block-to-worker assignment perturbs near-tie search decisions and the output depends on both the run and the thread count.
# Resetting makes each block's result a function of its own voxels alone, at the cost of one cold-started voxel per block.
function reset_voxel_chains!(thread_buffer)
    (; flip_angle_work) = thread_buffer
    flip_angle_work.decay_basis_set_ensemble !== nothing && reset_warmstart!(flip_angle_work.decay_basis_set_ensemble.nnls_search_prob)
    return nothing
end

# =========================================================
# Main loop function
# =========================================================
function voxelwise_T2_distribution!(thread_buffer, maps::T2Maps{T}, dist::T2Distributions{T}, signal::AbstractVector{T}, opts::T2mapOptions{T}, I::CartesianIndex) where {T}
    (; decay_data, decay_scale, flip_angle_work, T2_dist_work) = thread_buffer

    # Copy decay curve into the thread buffer and normalize
    @inbounds begin
        max_signal = zero(T)
        for i in 1:opts.nTE
            bᵢ = signal[i]
            max_signal = bᵢ > max_signal ? bᵢ : max_signal
            decay_data[i] = bᵢ
        end
        if max_signal > 0
            @simd for i in 1:opts.nTE
                decay_data[i] /= max_signal
            end
        end
        decay_scale[] = max_signal
    end

    if is_B1map_provided(maps)
        # Load flip angle from provided B1 map
        @inbounds flip_angle_work.α[] = maps.alpha[I]

        # Compute basis using the provided flip angle, routed through the same `final_decay_basis!` as the flip-search path, so a rerun that supplies the fitted α as a B1 map uses an identical basis.
        final_decay_basis!(flip_angle_work)
    else
        # Find optimum flip angle and compute EPG decay basis
        optimize_flip_angle!(flip_angle_work, opts)
    end

    # Calculate T2 distribution and map parameters
    T2_distribution!(T2_dist_work)

    # Save loop results to outputs
    save_results!(thread_buffer, maps, dist, opts, I)

    return nothing
end

# =========================================================
# EPG decay basis set construction
# =========================================================
struct EPGBasisSetFunctor{
    T,
    ETL,
    opt_vars,
    Tθ <: EPGParameterization{T},
    W <: AbstractEPGWorkspace{T},
    F <: EPGFunctor{T, ETL, opt_vars},
    J <: EPGJacobianFunctor{T, ETL, opt_vars},
}
    θ::Tθ
    opt_vars::Val{opt_vars} #TODO: SymbolVector{opt_vars}?
    T2_times::Vector{T}
    epg_work::W
    epg_functor!::F
    epg_jac_functor!::J
end

function EPGBasisSetFunctor(o::T2mapOptions{T}, θ::EPGParameterization{T}, opt_vars::Val) where {T}
    epg_work = EPGdecaybasis_work(θ)
    epg_functor! = EPGFunctor(θ, opt_vars)
    epg_jac_functor! = EPGJacobianFunctor(θ, opt_vars)
    return EPGBasisSetFunctor(θ, opt_vars, T2_component_times(o), epg_work, epg_functor!, epg_jac_functor!)
end

#### EPG basis set

function epg_decay_basis!(f::EPGBasisSetFunctor{T}, decay_basis::AbstractMatrix{T}, θ::EPGParameterization{T}) where {T}
    return epg_decay_basis!(decay_basis, f.epg_work, θ, f.T2_times)
end
epg_decay_basis!(f::EPGBasisSetFunctor{T}, decay_basis::AbstractMatrix{T}, x::SVector{D, T}, opt_vars::Val) where {D, T} = epg_decay_basis!(f, decay_basis, restructure(f.θ, x, opt_vars))
epg_decay_basis!(f::EPGBasisSetFunctor{T}, decay_basis::AbstractMatrix{T}, x::SVector{D, T}) where {D, T} = epg_decay_basis!(f, decay_basis, restructure(f.θ, x, f.opt_vars))

function epg_decay_basis!(decay_basis::AbstractMatrix{T}, decay_curve_work::AbstractEPGWorkspace{T}, θ::EPGParameterization{T}, T2_times::AbstractVector) where {T}
    # Compute the NNLS basis over T2 space
    @inbounds for j in 1:length(T2_times)
        decay_curve = uview(decay_basis, :, j)
        θj = restructure(θ, (; T2 = T2_times[j])) # remake options with T2 of basis `j`
        EPGdecaycurve!(decay_curve, decay_curve_work, θj)
    end
    return decay_basis
end

function epg_decay_basis(θ::EPGParameterization{T}, T2_times::AbstractVector) where {T}
    decay_basis = zeros(T, echotrainlength(θ), length(T2_times))
    decay_curve_work = EPGdecaycurve_work(θ)
    return epg_decay_basis!(decay_basis, decay_curve_work, θ, T2_times)
end

#### Jacobian of EPG basis set

function ∇epg_decay_basis!(f::EPGBasisSetFunctor{T}, ∇decay_basis::AbstractArray{T, 3}, decay_basis::AbstractMatrix{T}, θ::EPGParameterization{T}) where {T}
    return ∇epg_decay_basis!(∇decay_basis, decay_basis, f.epg_jac_functor!, θ, f.T2_times)
end
∇epg_decay_basis!(f::EPGBasisSetFunctor{T}, ∇decay_basis::AbstractArray{T, 3}, decay_basis::AbstractMatrix{T}, x::SVector{D, T}, opt_vars::Val) where {D, T} = ∇epg_decay_basis!(f, ∇decay_basis, decay_basis, restructure(f.θ, x, opt_vars))
∇epg_decay_basis!(f::EPGBasisSetFunctor{T}, ∇decay_basis::AbstractArray{T, 3}, decay_basis::AbstractMatrix{T}, x::SVector{D, T}) where {D, T} = ∇epg_decay_basis!(f, ∇decay_basis, decay_basis, restructure(f.θ, x, f.opt_vars))

function ∇epg_decay_basis!(∇decay_basis::AbstractArray{T, 3}, decay_basis::AbstractMatrix{T}, decay_curve_jac!::EPGJacobianFunctor{T}, θ::EPGParameterization{T}, T2_times::AbstractVector) where {T}
    # Compute the NNLS basis over T2 space
    @inbounds for j in 1:length(T2_times)
        decay_curve = uview(decay_basis, :, j)
        ∇decay_curve = uview(∇decay_basis, :, j, :)
        θj = restructure(θ, (; T2 = T2_times[j])) # remake options with T2 of basis `j`
        decay_curve_jac!(∇decay_curve, decay_curve, θj)
    end
    return ∇decay_basis
end

function ∇epg_decay_basis(θ::EPGParameterization{T}, T2_times::AbstractVector, Fs::NTuple{N, Symbol}) where {T, N}
    nTE, nT2 = echotrainlength(θ), length(T2_times)
    decay_basis = zeros(T, nTE, nT2)
    ∇decay_basis = zeros(T, nTE, nT2, N)
    decay_curve_jac! = EPGJacobianFunctor(θ, Fs)
    ∇epg_decay_basis!(∇decay_basis, decay_basis, decay_curve_jac!, θ, T2_times)
    return decay_basis, ∇decay_basis
end

# =========================================================
# Ensemble of EPG decay basis sets for discrete parameter search
# =========================================================
struct EPGBasisSetEnsemble{
    D,
    T,
    ETL,
    opt_vars,
    D1, # D1 = D + 2
    D2, # D2 = D + 3
    F <: EPGBasisSetFunctor{T, ETL, opt_vars},
    P <: NNLSDiscreteSurrogateSearch{D, T},
}
    opt_vars::Val{opt_vars} #TODO: SymbolVector{opt_vars}?
    decay_basis_set::Array{T, D1}
    ∇decay_basis_set::Array{T, D2}
    epg_basis_functor!::F
    nnls_search_prob::P
end

function EPGBasisSetEnsemble(o::T2mapOptions{T}, θ::EPGParameterization{T}, opt_vars::Val, decay_data::AbstractVector{T}) where {T}
    @assert opt_vars === Val((:α,)) "Optimization variable must be the flip angle :α" #TODO: generalize?
    D = 1 # D = length(opt_vars) #TODO: generalize?
    opt_ranges = (flip_angles(o),)
    decay_basis_set = zeros(T, o.nTE, o.nT2, length.(opt_ranges)...)
    ∇decay_basis_set = zeros(T, o.nTE, o.nT2, D, length.(opt_ranges)...)
    epg_basis_functor! = EPGBasisSetFunctor(o, θ, opt_vars)
    nnls_search_prob = NNLSDiscreteSurrogateSearch(decay_basis_set, ∇decay_basis_set, opt_ranges, decay_data; legacy = o.legacy)
    return EPGBasisSetEnsemble(opt_vars, decay_basis_set, ∇decay_basis_set, epg_basis_functor!, nnls_search_prob)
end

function epg_decay_basis!(work::EPGBasisSetEnsemble{D, T}, θ::EPGParameterization{T}) where {D, T}
    @inbounds for I in CartesianIndices(work.nnls_search_prob.αs)
        x = work.nnls_search_prob.αs[I]
        θ = restructure(θ, x, work.opt_vars)
        decay_basis = uview(work.decay_basis_set, :, :, I)
        epg_decay_basis!(work.epg_basis_functor!, decay_basis, θ)
    end
    return work
end
epg_decay_basis!(work::EPGBasisSetEnsemble{D, T}, x::SVector{D, T}, opt_vars::Val) where {D, T} = epg_decay_basis!(work, restructure(work.epg_basis_functor!.θ, x, opt_vars))
epg_decay_basis!(work::EPGBasisSetEnsemble{D, T}, x::SVector{D, T}) where {D, T} = epg_decay_basis!(work, restructure(work.epg_basis_functor!.θ, x, work.opt_vars))

function ∇epg_decay_basis!(work::EPGBasisSetEnsemble{D, T}, θ::EPGParameterization{T}) where {D, T}
    @inbounds for I in CartesianIndices(work.nnls_search_prob.αs)
        x = work.nnls_search_prob.αs[I]
        θ = restructure(θ, x, work.opt_vars)
        decay_basis = uview(work.decay_basis_set, :, :, I)
        ∇decay_basis = uview(work.∇decay_basis_set, :, :, :, I)
        ∇epg_decay_basis!(work.epg_basis_functor!, ∇decay_basis, decay_basis, θ)
    end
    return work
end
∇epg_decay_basis!(work::EPGBasisSetEnsemble{D, T}, x::SVector{D, T}, opt_vars::Val) where {D, T} = ∇epg_decay_basis!(work, restructure(work.epg_basis_functor!.θ, x, opt_vars))
∇epg_decay_basis!(work::EPGBasisSetEnsemble{D, T}, x::SVector{D, T}) where {D, T} = ∇epg_decay_basis!(work, restructure(work.epg_basis_functor!.θ, x, work.opt_vars))

# Internal runtime toggle: after the discrete surrogate search, refine the continuous minimizer α* with one true (f, f′) evaluation via the cosine-series basis and a sub-bracket cubic-Hermite re-minimization.
# Costs one extra basis build and NNLS solve per voxel in exchange for the accuracy gain.
const FLIP_ALPHA_POLISH = Ref(true)

# Scratch for the α-polish: the NNLS problem at α₀ plus the two columns written by the single-column derivative kernel.
struct AlphaPolishWorkspace{T, P}
    prob::P # NNLSProblem workspace at α₀
    Acol::Vector{T} # ETL, value column written by `epg_decay_basis_∂α_col!` and discarded
    ∂col::Vector{T} # ETL, α-derivative column
end
function AlphaPolishWorkspace(decay_basis::Matrix{T}, decay_data::Vector{T}) where {T}
    ETL = size(decay_basis, 1)
    return AlphaPolishWorkspace(NNLSProblem(decay_basis, decay_data), zeros(T, ETL), zeros(T, ETL))
end

# =========================================================
# Flip angle optimization
# =========================================================
struct FlipAngleOptimizationWorkspace{T, B, E, S, R, C, P}
    decay_basis::Matrix{T} # decay basis at the current flip angle `α`
    decay_data::Vector{T} # decay curve data
    decay_basis_set::B # B <: EPGBasisSetFunctor{T, ETL}
    decay_basis_set_ensemble::E # E <: Union{Nothing, EPGBasisSetEnsemble{1, T, ETL}}
    α::Base.RefValue{T} # current flip angle, in degrees
    α_surrogate::S # S <: Union{Nothing, AbstractSurrogate{1, T}}
    α_searcher::R # R <: Union{Nothing, DiscreteSurrogateSearcher{1, T}}; reused across voxels
    decay_basis_work::C # C <: Union{Nothing, EPGCosineSeriesBasis{T}}; evaluates the decay basis at arbitrary α, or nothing when the cosine representation does not apply
    α_polish_work::P # P <: Union{Nothing, AlphaPolishWorkspace{T}}; loss/gradient evaluation at the continuous surrogate minimizer (see FLIP_ALPHA_POLISH)
end

function FlipAngleOptimizationWorkspace(o::T2mapOptions{T}, decay_basis::Matrix{T}, decay_data::Vector{T}, shared_decay_basis_work::Union{Nothing, EPGCosineSeriesBasis{T}} = nothing) where {T}
    α = Ref(o.SetFlipAngle === nothing ? T(NaN) : o.SetFlipAngle)
    θ = default_epg_parameters(o)
    decay_basis_set = EPGBasisSetFunctor(o, θ, Val((:α,)))

    if o.SetFlipAngle !== nothing
        # Compute basis for fixed `SetFlipAngle`
        epg_decay_basis!(decay_basis_set, decay_basis, SA{T}[α[]])
        decay_basis_set_ensemble = nothing
        α_surrogate = nothing
        α_searcher = nothing
    else
        # Compute basis for each angle
        decay_basis_set_ensemble = EPGBasisSetEnsemble(o, θ, Val((:α,)), decay_data)
        ∇epg_decay_basis!(decay_basis_set_ensemble, θ)
        α_surrogate = o.legacy ?
                      CubicSplineSurrogate(decay_basis_set_ensemble.nnls_search_prob; legacy = true) :
                      CubicHermiteSplineSurrogate(decay_basis_set_ensemble.nnls_search_prob)
        α_searcher = DiscreteSurrogateSearcher(α_surrogate.grid) # reused per voxel; see `optimize_flip_angle!`
    end

    decay_basis_work = shared_decay_basis_work !== nothing ? EPGCosineSeriesBasis(shared_decay_basis_work) :
                       !o.legacy && o.SetFlipAngle === nothing && θ isa EPGConstantFlipAngleOptions ?
                       EPGCosineSeriesBasis(θ, decay_basis_set.T2_times) : nothing

    # α-polish needs the exact continuous-α basis (cosine series) and the gradient-carrying surrogate
    α_polish_work = decay_basis_work !== nothing && α_surrogate isa CubicHermiteSplineSurrogate ? AlphaPolishWorkspace(decay_basis, decay_data) : nothing

    return FlipAngleOptimizationWorkspace(decay_basis, decay_data, decay_basis_set, decay_basis_set_ensemble, α, α_surrogate, α_searcher, decay_basis_work, α_polish_work)
end

# One refinement of the surrogate minimizer: the discrete search only ever evaluates the true loss f(α) = min_{x≥0} ‖A(α)x − b‖² at grid nodes, so the returned continuous α* carries the cubic-Hermite interpolation error of the final grid cell, O(h⁴) in the cell width.
# The cosine-series basis makes a true off-grid evaluation cheap: build A(α*), solve NNLS, form the envelope gradient g* = 2·(∂A/∂α·x)ᵀ(Ax − b), then re-minimize the cubic Hermite on the sub-bracket [αₗ, α*] or [α*, αᵣ] selected by sign(g*), whose endpoint data are true (f, f′) values.
# Returns `true` iff `work.decay_basis` already holds the exact basis at the final `work.α[]`, letting the caller skip the final rebuild.
function polish_flip_angle!(work::FlipAngleOptimizationWorkspace{T}) where {T}
    (; grid, seen, u, ∇u) = work.α_surrogate
    α₀ = work.α[]

    # Refine only when α₀ lies strictly between grid nodes; landing on a node needs no refinement, since the loss there is already a true evaluation.
    i = searchsortedlast(grid, SA{T}[α₀]; by = first)
    (1 <= i < length(grid) && first(grid[i]) < α₀) || return false

    # Bracket α₀ by the nearest evaluated node on either side, since `u` and `∇u` hold true (f, f′) data only where the search evaluated.
    il, ir = i, i + 1
    while il >= 1 && !seen[il]
        il -= 1
    end
    while ir <= length(grid) && !seen[ir]
        ir += 1
    end
    (il >= 1 && ir <= length(grid)) || return false

    # True loss and envelope gradient at α₀
    f₀, g₀ = polish_loss_grad!(work, α₀)

    # Re-minimize the cubic Hermite on the sub-bracket containing the descent direction
    spl = g₀ > 0 ? CubicHermiteInterpolator(first(grid[il]), α₀, u[il], f₀, first(∇u[il]), g₀) :
          CubicHermiteInterpolator(α₀, first(grid[ir]), f₀, u[ir], g₀, first(∇u[ir]))
    αnew, _ = minimize(spl)

    # If the refinement returns α₀, let the caller skip the final rebuild
    αnew == α₀ && return true
    work.α[] = αnew
    return false
end

# Polish evaluation: build the basis A(α₀), solve NNLS warm-started from the search's final active set, which is the exact support at α₀, then accumulate the residual and envelope gradient over the support columns.
function polish_loss_grad!(work::FlipAngleOptimizationWorkspace{T}, α₀::T) where {T}
    (; decay_basis, decay_basis_set_ensemble, decay_basis_work, α_polish_work) = work
    epg_decay_basis!(decay_basis, decay_basis_work, α₀) # basis at α₀; leaves `c` current
    solve_unreg!(α_polish_work.prob, decay_basis_set_ensemble.nnls_search_prob.nnls_work.nnls_work)
    x = NNLS.solution(α_polish_work.prob.nnls_work)

    # The solve left r = b − A_P x_P current, so f₀ = ‖r‖² and the envelope gradient 2·(∂A_P x_P)ᵀ(A_P x_P − b) is −2·Σ_j x_j⟨∂A[:, j], r⟩, one dot product per support column.
    wk = α_polish_work.prob.nnls_work
    r = NNLS.residual(wk)
    (; Acol, ∂col) = α_polish_work # `Acol` is discarded; the value column is already in `decay_basis`
    cosine_∂α_features!(decay_basis_work, α₀)
    g₀ = zero(T)
    @inbounds for j in NNLS.components(wk) # the support is already known; scanning all n columns for positivity would rediscover it
        epg_decay_basis_∂α_col!(Acol, ∂col, decay_basis_work, α₀, j)
        g₀ = muladd(x[j], dot(∂col, r), g₀)
    end
    return dot(r, r), -2 * g₀
end

function optimize_flip_angle!(work::FlipAngleOptimizationWorkspace, o::T2mapOptions)

    if o.SetFlipAngle === nothing
        # Find optimal flip angle
        empty!(work.α_surrogate)
        advance_warmstart!(work.decay_basis_set_ensemble.nnls_search_prob) # cross-voxel NNLS warm starting
        reset!(work.α_searcher) # reuse the searcher's buffers instead of allocating one per voxel
        initialize!(work.α_surrogate, work.α_searcher; mineval = o.nRefAnglesMin, maxeval = o.nRefAngles)
        α_opt, _ = bisection_search(work.α_surrogate, work.α_searcher; maxeval = o.nRefAngles)
        work.α[] = α_opt[1]

        # Refine the surrogate minimizer against a true off-grid loss evaluation; `true` means the basis at the final α is already built
        basis_current = FLIP_ALPHA_POLISH[] && work.α_polish_work !== nothing && polish_flip_angle!(work)

        # Compute basis using optimized flip angles
        basis_current || final_decay_basis!(work)
    end

    return nothing
end

# Build the T2-stage decay basis at the current flip angle `work.α[]`.
# Uses the exact cosine-series evaluation when it applies, and the exact EPG rebuild otherwise.
function final_decay_basis!(work::FlipAngleOptimizationWorkspace)
    α = work.α[]
    if work.decay_basis_work !== nothing
        epg_decay_basis!(work.decay_basis, work.decay_basis_work, α)
    else
        epg_decay_basis!(work.decay_basis_set, work.decay_basis, SA[α])
    end
    return nothing
end

# =========================================================
# T2-distribution fitting
# =========================================================
abstract type RegularizationMethod end
struct NoRegularization <: RegularizationMethod end
struct LCurve <: RegularizationMethod end
struct GCV <: RegularizationMethod end
struct Reginska <: RegularizationMethod end
struct ChiSquared{T} <: RegularizationMethod
    Chi2Factor::T
    legacy::Bool
end
struct MDP{T} <: RegularizationMethod
    NoiseLevel::T
end

function regularization_method(o::T2mapOptions)
    reg =
        o.Reg == "none"     ? NoRegularization() : # Fit T2 distribution using unregularized NNLS
        o.Reg == "lcurve"   ? LCurve() : # Fit T2 distribution using L-curve-based regularized NNLS
        o.Reg == "gcv"      ? GCV() : # Fit T2 distribution using GCV-based regularized NNLS
        o.Reg == "reginska" ? Reginska() : # Fit T2 distribution using Reginska's minimum-product criterion
        o.Reg == "chi2"     ? ChiSquared(o.Chi2Factor, o.legacy) : # Fit T2 distribution using chi2-based regularized NNLS
        o.Reg == "mdp"      ? MDP(o.NoiseLevel) : # Fit T2 distribution using Morizov discrepancy principle-based regularized NNLS
        error("Unrecognized regularization method: $(o.Reg)")
    return reg
end

nnls_workspace(::NoRegularization, decay_basis::AbstractMatrix{T}, decay_data::AbstractVector{T}, args...) where {T} = lsqnonneg_work(decay_basis, decay_data)
nnls_workspace(::LCurve, decay_basis::AbstractMatrix{T}, decay_data::AbstractVector{T}, nnls_prob_seed, args...) where {T} = lsqnonneg_lcurve_work(decay_basis, decay_data, nnls_prob_seed)
nnls_workspace(::GCV, decay_basis::AbstractMatrix{T}, decay_data::AbstractVector{T}, nnls_prob_seed, dof_interpolator) where {T} = lsqnonneg_gcv_work(decay_basis, decay_data, nnls_prob_seed, dof_interpolator)
nnls_workspace(::Reginska, decay_basis::AbstractMatrix{T}, decay_data::AbstractVector{T}, nnls_prob_seed, args...) where {T} = lsqnonneg_reginska_work(decay_basis, decay_data, nnls_prob_seed)
nnls_workspace(::ChiSquared, decay_basis::AbstractMatrix{T}, decay_data::AbstractVector{T}, nnls_prob_seed, args...) where {T} = lsqnonneg_chi2_work(decay_basis, decay_data, nnls_prob_seed)
nnls_workspace(::MDP, decay_basis::AbstractMatrix{T}, decay_data::AbstractVector{T}, nnls_prob_seed, args...) where {T} = lsqnonneg_mdp_work(decay_basis, decay_data, nnls_prob_seed)

struct T2DistWorkspace{Reg, T, W}
    reg::Reg
    nnls_work::W
    decay_basis::Matrix{T}
    decay_data::Vector{T}
    decay_scale::Base.RefValue{T}
    μ::Base.RefValue{T}
    χ²fact::Base.RefValue{T}
end

function T2DistWorkspace(reg::RegularizationMethod, decay_basis::Matrix{T}, decay_data::Vector{T}, decay_scale::Base.RefValue{T}, nnls_prob_seed = nothing, dof_interpolator = nothing) where {T}
    # Note: `T2_distribution!(::T2DistWorkspace)` methods defined below assume that references to the EPG decay bases `A::AbstractMatrix` and the MSE signal `b::AbstractVector`
    # are stored in the `nnls_work` workspace field and that `A` and `b` have been populated with the appropriate data.
    # The flip-search seed and dof-interpolation source (or nothing) are stored inside `nnls_work` rather than threaded per solve.
    μ, χ²fact = Ref(T(NaN)), Ref(T(NaN))
    nnls_work = nnls_workspace(reg, decay_basis, decay_data, nnls_prob_seed, dof_interpolator)
    return T2DistWorkspace(reg, nnls_work, decay_basis, decay_data, decay_scale, μ, χ²fact)
end

function T2_distribution!(t2work::T2DistWorkspace{NoRegularization, T}) where {T}
    (; nnls_work, μ, χ²fact) = t2work
    μ[], χ²fact[] = zero(T), one(T)
    return lsqnonneg!(nnls_work)
end

function T2_distribution!(t2work::T2DistWorkspace{LCurve, T}) where {T}
    (; nnls_work, μ, χ²fact) = t2work
    x, μ[], χ²fact[] = lsqnonneg_lcurve!(nnls_work)
    return x
end

function T2_distribution!(t2work::T2DistWorkspace{GCV, T}) where {T}
    (; nnls_work, μ, χ²fact) = t2work
    x, μ[], χ²fact[] = lsqnonneg_gcv!(nnls_work)
    return x
end

function T2_distribution!(t2work::T2DistWorkspace{Reginska, T}) where {T}
    (; nnls_work, μ, χ²fact) = t2work
    x, μ[], χ²fact[] = lsqnonneg_reginska!(nnls_work)
    return x
end

function T2_distribution!(t2work::T2DistWorkspace{ChiSquared{T}, T}) where {T}
    (; reg, nnls_work, μ, χ²fact) = t2work
    x, μ[], χ²fact[] = lsqnonneg_chi2!(nnls_work, reg.Chi2Factor, reg.legacy)
    return x
end

function T2_distribution!(t2work::T2DistWorkspace{MDP{T}, T}) where {T}
    (; reg, nnls_work, decay_basis, decay_data, decay_scale, μ, χ²fact) = t2work
    σ = reg.NoiseLevel / decay_scale[] # homoscedastic standard deviation: σ² = 𝔼[||ηᵢ||²] = 𝔼[(bᵢ - b̂ᵢ)²]
    δ = √(T(length(decay_data))) * σ # noise vector norm estimate: δ² = 𝔼[||η||²] = n * σ²
    x, μ[], χ²fact[] = lsqnonneg_mdp!(nnls_work, δ)
    return x
end

T2_distribution(t2work::T2DistWorkspace) = solution(t2work.nnls_work)

# =========================================================
# Save thread local results to output maps
# =========================================================
function save_results!(thread_buffer, maps::T2Maps{T}, dist::T2Distributions{T}, o::T2mapOptions{T}, I::CartesianIndex) where {T}
    (; logT2_times, decay_basis, decay_data, decay_scale, decay_curvefit, residuals, flip_angle_work, T2_dist_work) = thread_buffer
    T2_dist = T2_distribution(T2_dist_work)

    @inbounds begin
        # Rescale results to original signal scale
        max_signal = decay_scale[]
        @simd for i in 1:o.nTE
            decay_data[i] *= max_signal
        end
        @simd for j in 1:o.nT2
            T2_dist[j] *= max_signal
        end

        # Compute signal decay curve fit and residuals
        mul!(decay_curvefit, decay_basis, T2_dist)
        @simd for i in 1:o.nTE
            residuals[i] = decay_curvefit[i] - decay_data[i]
        end

        # Compute distribution parameters
        Σ_dist = sum(T2_dist)
        Σ_res² = sum(abs2, residuals)
        σ_res = std(residuals)
        log_ggm = dot(T2_dist, logT2_times) / Σ_dist
        log1p_gva = zero(T)
        @simd for j in 1:o.nT2
            log1p_gva += abs2(logT2_times[j] - log_ggm) * T2_dist[j]
        end
        log1p_gva /= Σ_dist
    end

    # Compute and save parameters of distribution
    (; gdn, ggm, gva, fnr, snr, alpha) = maps
    @inbounds begin
        gdn[I] = Σ_dist # general density
        ggm[I] = exp(log_ggm) # general geometric mean
        gva[I] = expm1(log1p_gva) # general variance
        fnr[I] = Σ_dist / √(Σ_res² / (o.nTE - 1)) # fit to noise ratio
        snr[I] = max_signal / σ_res # signal to noise ratio
        alpha[I] = flip_angle_work.α[] # optimized refocusing pulse flip angle
    end

    # Save distribution
    @inbounds for j in 1:o.nT2
        dist.distributions[I, j] = T2_dist[j]
    end

    # Optionally save regularization parameters
    if maps.mu !== nothing && maps.chi2factor !== nothing # o.SaveRegParam == true
        local mu::Array{T, 3}, chi2factor::Array{T, 3} = maps.mu, maps.chi2factor
        @inbounds mu[I], chi2factor[I] = T2_dist_work.μ[], T2_dist_work.χ²fact[]
    end

    # Optionally save ℓ²-norm of residuals
    if maps.resnorm !== nothing # o.SaveResidualNorm == true
        local resnorm::Array{T, 3} = maps.resnorm
        @inbounds resnorm[I] = √Σ_res²
    end

    # Optionally save signal decay curve from fit
    if maps.decaycurve !== nothing # o.SaveDecayCurve == true
        local decaycurve::Array{T, 4} = maps.decaycurve
        @inbounds for i in 1:o.nTE
            decaycurve[I, i] = decay_curvefit[i]
        end
    end

    # Optionally save NNLS basis
    if maps.decaybasis !== nothing # o.SaveNNLSBasis == true
        if o.SetFlipAngle === nothing
            local decaybasis::Array{T, 5} = maps.decaybasis
            @inbounds for J in CartesianIndices((o.nTE, o.nT2))
                decaybasis[I, J] = decay_basis[J]
            end
        end
    end

    return nothing
end

# =========================================================
# Utility functions
# =========================================================
function thread_buffer_maker(o::T2mapOptions{T}, global_buffer = nothing) where {T}
    decay_basis = zeros(T, o.nTE, o.nT2)
    decay_data = zeros(T, o.nTE)
    decay_scale = Ref(one(T))
    flip_angle_work = FlipAngleOptimizationWorkspace(o, decay_basis, decay_data, global_buffer === nothing ? nothing : global_buffer.decay_basis_work)
    ensemble = flip_angle_work.decay_basis_set_ensemble
    nnls_prob_seed = ensemble === nothing ? nothing : ensemble.nnls_search_prob.nnls_work.nnls_work
    dof_interpolator = ensemble === nothing || !(regularization_method(o) isa GCV) ? nothing : (GriddedSpectrumInterpolator(ensemble.decay_basis_set, ensemble.∇decay_basis_set, flip_angles(o)), flip_angle_work.α)
    return (;
        T2_times         = logrange(o.T2Range..., o.nT2),
        logT2_times      = log.(logrange(o.T2Range..., o.nT2)),
        flip_angles      = flip_angles(o),
        refcon_angles    = refcon_angles(o),
        decay_basis      = decay_basis,
        decay_data       = decay_data,
        decay_scale      = decay_scale,
        decay_curvefit   = zeros(T, o.nTE),
        residuals        = zeros(T, o.nTE),
        decay_curve_work = EPGdecaycurve_work(default_epg_parameters(o)),
        flip_angle_work  = flip_angle_work,
        T2_dist_work     = T2DistWorkspace(regularization_method(o), decay_basis, decay_data, decay_scale, nnls_prob_seed, dof_interpolator),
    )
end

# Master decay basis workspace for the current options, or `nothing` when the cosine representation does not apply, namely in legacy mode, for a fixed flip angle, or for a non-constant flip angle.
# The coefficient tensor is voxel- and thread-independent, so one instance is built here and its read-only coefficients are shared by every worker's buffer instead of each holding a private copy.
function global_buffer_maker(o::T2mapOptions{T}) where {T}
    θ = default_epg_parameters(o)
    decay_basis_work = !o.legacy && o.SetFlipAngle === nothing && θ isa EPGConstantFlipAngleOptions ? EPGCosineSeriesBasis(θ, T2_component_times(o)) : nothing
    return (; decay_basis_work)
end

function default_epg_parameters(o::T2mapOptions{T}) where {T}
    return o.RefConAngle == 180 ?
        EPGConstantFlipAngleOptions((; ETL = o.nTE, α = T(NaN), TE = o.TE, T2 = T(NaN), T1 = o.T1)) :
        EPGOptions((; ETL = o.nTE, α = T(NaN), TE = o.TE, T2 = T(NaN), T1 = o.T1, β = o.RefConAngle))
end
