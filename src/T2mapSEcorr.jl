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

Base.convert(::Type{Dict{Symbol, Any}}, maps::T2Maps) = Dict{Symbol, Any}(Any[f => getfield(maps, f) for f in fieldnames(T2Maps) if getfield(maps, f) isa Array])
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
        gdn = fill(T(NaN), opts.MatrixSize...),
        ggm = fill(T(NaN), opts.MatrixSize...),
        gva = fill(T(NaN), opts.MatrixSize...),
        fnr = fill(T(NaN), opts.MatrixSize...),
        snr = fill(T(NaN), opts.MatrixSize...),
        alpha = fill(T(NaN), opts.MatrixSize...),
        is_alpha_provided = Ref(false),

        # Optional output maps
        resnorm    = !opts.SaveResidualNorm ? nothing : fill(T(NaN), opts.MatrixSize...),
        decaycurve = !opts.SaveDecayCurve ? nothing : fill(T(NaN), opts.MatrixSize..., opts.nTE),
        mu         = !opts.SaveRegParam ? nothing : fill(T(NaN), opts.MatrixSize...),
        chi2factor = !opts.SaveRegParam ? nothing : fill(T(NaN), opts.MatrixSize...),
        decaybasis = !opts.SaveNNLSBasis ? nothing :
        opts.SetFlipAngle === nothing ?
        fill(T(NaN), opts.MatrixSize..., opts.nTE, opts.nT2) : # unique decay basis set for each voxel
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
        distributions = fill(T(NaN), opts.MatrixSize..., opts.nT2),
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
          * `"fnr"`:          Map of fit to noise ratio = gdn / sqrt(sum(residuals.^2) / (nTE-1)) (`MatrixSize` 3D array)
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
julia> image = DECAES.mock_image(; MatrixSize = (100, 100, 1), nTE = 32); # mock image with size 100x100x1x32

julia> maps, dist = T2mapSEcorr(image; TE = 10e-3, nT2 = 40, T2Range = (10e-3, 2.0), Reg = "lcurve", Silent = true); # compute the T2-maps and T2-distribution

julia> maps
Dict{String, Any} with 10 entries:
  "echotimes"     => [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,…
  "t2times"       => [0.01, 0.0114551, 0.013122, 0.0150315, 0.0172188, 0.01…
  "refangleset"   => [50.0, 54.1935, 58.3871, 62.5806, 66.7742, 70.9677, 75…
  "gdn"           => [1.3787e5 177386.0 … 1.95195e5 1.3515e5; 1.8281e5 1.54…
  "fnr"           => [188.642 204.698 … 216.84 242.163; 183.262 199.053 … 2…
  "alpha"         => [180.0 180.0 … 180.0 180.0; 180.0 180.0 … 180.0 180.0;…
  "gva"           => [0.48029 0.554944 … 0.65303 0.595152; 0.510848 0.54559…
  "ggm"           => [0.0521713 0.0510269 … 0.0494394 0.0503616; 0.052195 0…
  "snr"           => [152.536 164.471 … 171.872 192.296; 147.898 161.285 … …
  "decaybasisset" => [0.0277684 0.0315296 … 0.0750511 0.0751058; 0.0469882 …
```

See also:

  - [`T2partSEcorr`](@ref)
  - [`lsqnonneg`](@ref)
  - [`lsqnonneg_chi2`](@ref)
  - [`lsqnonneg_gcv`](@ref)
  - [`lsqnonneg_lcurve`](@ref)
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

    # For each worker in the worker pool, allocate a separete thread-local buffer, then run the work function `work!`
    function with_thread_buffer(work!)
        thread_buffer = thread_buffer_maker(opts)
        return work!(thread_buffer)
    end

    # Run analysis in parallel
    indices = filter(I -> image[I, 1] > opts.Threshold, CartesianIndices(opts.MatrixSize))
    if isempty(indices)
        !opts.Silent && @warn "No voxels found with first-echo signal intensity above threshold $(opts.Threshold).\nConsider lowering the threshold or checking the input data."
        return convert(Dict{String, Any}, maps), convert(Array{T, 4}, dist)
    end
    ntasks = opts.Threaded ? Threads.nthreads() : 1
    indices_blocks = split_indices(length(indices), default_blocksize())

    # Run analysis in parallel
    signals = permutedims(image[indices, :]) # permute image for cache locality

    with_singlethreaded_blas() do
        workerpool(with_thread_buffer, indices_blocks; ntasks = ntasks, verbose = !opts.Silent) do inds, thread_buffer
            @inbounds for j in inds
                voxelwise_T2_distribution!(thread_buffer, maps, dist, uview(signals, :, j), opts, indices[j])
            end
        end
    end

    return convert(Dict{String, Any}, maps), convert(Array{T, 4}, dist)
end

# =========================================================
# Main loop function
# =========================================================
function voxelwise_T2_distribution!(thread_buffer, maps::T2Maps, dist::T2Distributions, signal::AbstractVector, opts::T2mapOptions, I::CartesianIndex)
    (; decay_data, flip_angle_work, T2_dist_work) = thread_buffer

    # Copy decay curve into the thread buffer
    @inbounds for j in 1:opts.nTE
        decay_data[j] = signal[j]
    end

    if is_B1map_provided(maps)
        # Load flip angle from provided B1 map
        @inbounds flip_angle_work.α[] = maps.alpha[I]

        # Compute basis using provided flip angle
        epg_decay_basis!(flip_angle_work.decay_basis_set, flip_angle_work.decay_basis, SA[flip_angle_work.α[]])
    else
        # Find optimum flip angle and compute EPG decay basis
        optimize_flip_angle!(flip_angle_work, opts)
    end

    # Calculate T2 distribution and map parameters
    t2_distribution!(T2_dist_work)

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
    opt_vars::Val{opt_vars} #TODO SymbolVector{opt_vars}
    T2_times::Vector{T}
    epg_work::W
    epg_functor!::F
    epg_jac_functor!::J
end

function EPGBasisSetFunctor(o::T2mapOptions{T}, θ::EPGParameterization{T}, opt_vars::Val) where {T}
    epg_work = EPGdecaycurve_work(θ)
    epg_functor! = EPGFunctor(θ, opt_vars)
    epg_jac_functor! = EPGJacobianFunctor(θ, opt_vars)
    return EPGBasisSetFunctor(θ, opt_vars, t2_times(o), epg_work, epg_functor!, epg_jac_functor!)
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
        decay_curve = uview(decay_basis, :, j) # `UnsafeArrays.uview` is a bit faster than `Base.view`
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
        decay_curve = uview(decay_basis, :, j) # `UnsafeArrays.uview` is a bit faster than `Base.view`
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
    A1 <: AbstractArray{T},
    A2 <: AbstractArray{T},
    F <: EPGBasisSetFunctor{T, ETL, opt_vars},
    P <: NNLSDiscreteSurrogateSearch{D, T},
}
    opt_vars::Val{opt_vars} #TODO SymbolVector{opt_vars}
    decay_basis_set::A1
    ∇decay_basis_set::A2
    epg_basis_functor!::F
    nnls_search_prob::P
end

function EPGBasisSetEnsemble(o::T2mapOptions{T}, θ::EPGOptions{T}, opt_vars::Val, decay_data::AbstractVector{T}) where {T}
    @assert opt_vars === Val((:α,)) "Optimization variable must be the flip angle :α" #TODO: generalize?
    D = 1 # length(opt_vars)
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

# =========================================================
# Flip angle optimization
# =========================================================
struct FlipAngleOptimizationWorkspace{T, M <: AbstractMatrix{T}, V <: AbstractVector{T}, B, E, S}
    decay_basis::M
    decay_data::V
    decay_basis_set::B # B <: EPGBasisSetFunctor{T, ETL}
    decay_basis_set_ensemble::E # E <: Union{Nothing, EPGBasisSetEnsemble{1, T, ETL}}
    α::Base.RefValue{T}
    α_surrogate::S # S <: Union{Nothing, AbstractSurrogate{1, T}}
end

function FlipAngleOptimizationWorkspace(o::T2mapOptions{T}, decay_basis::AbstractMatrix{T}, decay_data::AbstractVector{T}) where {T}
    α = Ref(o.SetFlipAngle === nothing ? T(NaN) : o.SetFlipAngle)
    θ = EPGOptions((; ETL = o.nTE, α = α[], TE = o.TE, T2 = T(NaN), T1 = o.T1, β = o.RefConAngle))
    decay_basis_set = EPGBasisSetFunctor(o, θ, Val((:α,)))

    if o.SetFlipAngle !== nothing
        # Compute basis for fixed `SetFlipAngle`
        epg_decay_basis!(decay_basis_set, decay_basis, SA{T}[α[]])
        decay_basis_set_ensemble = nothing
        α_surrogate = nothing
    else
        # Compute basis for each angle
        decay_basis_set_ensemble = EPGBasisSetEnsemble(o, θ, Val((:α,)), decay_data)
        ∇epg_decay_basis!(decay_basis_set_ensemble, θ)
        α_surrogate = o.legacy ?
                      CubicSplineSurrogate(decay_basis_set_ensemble.nnls_search_prob; legacy = true) :
                      NormalHermiteSplineSurrogate(decay_basis_set_ensemble.nnls_search_prob)
    end

    return FlipAngleOptimizationWorkspace(decay_basis, decay_data, decay_basis_set, decay_basis_set_ensemble, α, α_surrogate)
end

function optimize_flip_angle!(work::FlipAngleOptimizationWorkspace, o::T2mapOptions)

    if o.SetFlipAngle === nothing
        # Find optimal flip angle
        empty!(work.α_surrogate)
        nnls_searcher = DiscreteSurrogateSearcher(work.α_surrogate; mineval = o.nRefAnglesMin, maxeval = o.nRefAngles)
        α_opt, _ = bisection_search(work.α_surrogate, nnls_searcher; maxeval = o.nRefAngles)
        work.α[] = α_opt[1]

        # Compute basis using optimized flip angles
        epg_decay_basis!(work.decay_basis_set, work.decay_basis, SA[work.α[]])
    end

    return nothing
end

# =========================================================
# T2-distribution fitting
# =========================================================
abstract type RegularizationMethod end
struct NoRegularization <: RegularizationMethod end
struct ChiSquared{T} <: RegularizationMethod
    Chi2Factor::T
    legacy::Bool
end
struct GCV <: RegularizationMethod end
struct LCurve <: RegularizationMethod end

function regularization_method(o::T2mapOptions)
    return o.Reg == "none"   ? NoRegularization() : # Fit T2 distribution using unregularized NNLS
           o.Reg == "chi2"   ? ChiSquared(o.Chi2Factor, o.legacy) : # Fit T2 distribution using chi2-based regularized NNLS
           o.Reg == "gcv"    ? GCV() : # Fit T2 distribution using GCV-based regularized NNLS
           o.Reg == "lcurve" ? LCurve() : # Fit T2 distribution using L-curve-based regularized NNLS
           error("Unrecognized regularization method: $(o.Reg)")
end

nnls_workspace(::NoRegularization, decay_basis::AbstractMatrix{T}, decay_data::AbstractVector{T}) where {T} = lsqnonneg_work(decay_basis, decay_data)
nnls_workspace(::ChiSquared, decay_basis::AbstractMatrix{T}, decay_data::AbstractVector{T}) where {T} = lsqnonneg_chi2_work(decay_basis, decay_data)
nnls_workspace(::GCV, decay_basis::AbstractMatrix{T}, decay_data::AbstractVector{T}) where {T} = lsqnonneg_gcv_work(decay_basis, decay_data)
nnls_workspace(::LCurve, decay_basis::AbstractMatrix{T}, decay_data::AbstractVector{T}) where {T} = lsqnonneg_lcurve_work(decay_basis, decay_data)

struct T2DistWorkspace{Reg, T, W}
    reg::Reg
    nnls_work::W
    μ::Base.RefValue{T}
    χ²fact::Base.RefValue{T}
end
function T2DistWorkspace(reg::RegularizationMethod, decay_basis::AbstractMatrix{T}, decay_data::AbstractVector{T}, μ::Base.RefValue{T} = Ref(T(NaN)), χ²fact::Base.RefValue{T} = Ref(T(NaN))) where {T}
    return T2DistWorkspace(reg, nnls_workspace(reg, decay_basis, decay_data), μ, χ²fact)
end

function nnls!(t2work::T2DistWorkspace{NoRegularization, T}) where {T}
    return x, t2work.μ[], t2work.χ²fact[] = lsqnonneg!(t2work.nnls_work), zero(T), one(T)
end

function nnls!(t2work::T2DistWorkspace{ChiSquared{T}, T}) where {T}
    return x, t2work.μ[], t2work.χ²fact[] = lsqnonneg_chi2!(t2work.nnls_work, t2work.reg.Chi2Factor; legacy = t2work.reg.legacy)
end

function nnls!(t2work::T2DistWorkspace{GCV, T}) where {T}
    return x, t2work.μ[], t2work.χ²fact[] = lsqnonneg_gcv!(t2work.nnls_work)
end

function nnls!(t2work::T2DistWorkspace{LCurve, T}) where {T}
    return x, t2work.μ[], t2work.χ²fact[] = lsqnonneg_lcurve!(t2work.nnls_work)
end

t2_distribution!(t2work::T2DistWorkspace) = nnls!(t2work)
solution(t2work::T2DistWorkspace) = solution(t2work.nnls_work)

# =========================================================
# Save thread local results to output maps
# =========================================================
function save_results!(thread_buffer, maps::T2Maps{T}, dist::T2Distributions{T}, o::T2mapOptions{T}, I::CartesianIndex) where {T}
    (; T2_dist_work, flip_angle_work, logT2_times, decay_data, decay_basis, decay_calc, residuals, gva_buf) = thread_buffer
    T2_dist = solution(T2_dist_work)

    # Compute and save parameters of distribution
    (; gdn, ggm, gva, fnr, snr, alpha) = maps
    @inbounds begin
        mul!(decay_calc, decay_basis, T2_dist)
        Σ_dist = sum(T2_dist)
        residuals .= decay_calc .- decay_data
        log_ggm = dot(T2_dist, logT2_times) / Σ_dist
        log1p_gva = zero(T)
        @simd for j in 1:o.nT2
            log1p_gva += abs2(logT2_times[j] - log_ggm) * T2_dist[j]
        end
        log1p_gva /= Σ_dist
        gdn[I] = Σ_dist
        ggm[I] = exp(log_ggm)
        gva[I] = expm1(log1p_gva)
        fnr[I] = Σ_dist / sqrt(sum(abs2, residuals) / (o.nTE - 1))
        snr[I] = maximum(decay_data) / std(residuals)
        alpha[I] = flip_angle_work.α[]
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
        @inbounds resnorm[I] = sqrt(sum(abs2, residuals))
    end

    # Optionally save signal decay curve from fit
    if maps.decaycurve !== nothing # o.SaveDecayCurve == true
        local decaycurve::Array{T, 4} = maps.decaycurve
        @inbounds for j in 1:o.nTE
            decaycurve[I, j] = decay_calc[j]
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
function thread_buffer_maker(o::T2mapOptions{T}) where {T}
    decay_basis = zeros(T, o.nTE, o.nT2)
    decay_data  = zeros(T, o.nTE)
    return (;
        T2_times         = logrange(o.T2Range..., o.nT2),
        logT2_times      = log.(logrange(o.T2Range..., o.nT2)),
        flip_angles      = flip_angles(o),
        refcon_angles    = refcon_angles(o),
        decay_basis      = decay_basis,
        decay_data       = decay_data,
        decay_calc       = zeros(T, o.nTE),
        residuals        = zeros(T, o.nTE),
        gva_buf          = zeros(T, o.nT2),
        decay_curve_work = EPGdecaycurve_work(T, o.nTE),
        flip_angle_work  = FlipAngleOptimizationWorkspace(o, decay_basis, decay_data),
        T2_dist_work     = T2DistWorkspace(regularization_method(o), decay_basis, decay_data),
    )
end
