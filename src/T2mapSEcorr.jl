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
    θ = default_epg_parameters(opts)
    T2_times = T2_component_times(opts)
    decay_basis_set = opts.SetFlipAngle === nothing ? epg_grid_model(opts, θ).As : epg_decay_basis(restructure(θ, (; α = deg2rad(opts.SetFlipAngle))), T2_times)
    return T2Maps(;
        # Misc. processing parameters
        echotimes     = convert(Array{T}, opts.TE .* (1:opts.nTE)),
        t2times       = convert(Array{T}, T2_times),
        refangleset   = opts.SetFlipAngle === nothing ? convert(Array{T}, rad2deg.(flip_angles(opts))) : T(opts.SetFlipAngle), # outputs are in degrees
        decaybasisset = decay_basis_set,

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
        opts.SetFlipAngle === nothing ? tfill(T(NaN), opts.MatrixSize..., opts.nTE, opts.nT2) : copy(decay_basis_set), # per voxel or global decay basis set
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
          * `"gdn"`:          Map of general density = sum(T2distribution) (Units: same as input signal) (`MatrixSize` 3D array)
          * `"ggm"`:          Map of general geometric mean of T2-distribution (Units: time, must match `T2Range`) (`MatrixSize` 3D array)
          * `"gva"`:          Map of general variance of the T2-distribution (Units: none) (`MatrixSize` 3D array)
          * `"fnr"`:          Map of fit to noise ratio = gdn / √(sum(residuals.^2) / (nTE-1)) (Units: none) (`MatrixSize` 3D array)
          * `"snr"`:          Map of signal to noise ratio = maximum(signal) / std(residuals) (Units: none) (`MatrixSize` 3D array)
          * `"alpha"`:        Map of optimized refocusing pulse flip angle (Units: degrees) (`MatrixSize` 3D array)

      + **Optional Fields**

          * `"resnorm"`:      ``\\ell^2``-norm of NNLS fit residuals; see `SaveResidualNorm` option (`MatrixSize` 3D array)
          * `"decaycurve"`:   Signal decay curve resulting from NNLS fit; see `SaveDecayCurve` option (`MatrixSize x nTE` 4D array)
          * `"mu"`:           Regularization parameter used during the NNLS fit; see `SaveRegParam` option (`MatrixSize` 3D array)
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
# A chain that persisted across blocks would let the dynamic block-to-worker assignment perturb near-tie search decisions, making the output depend on the run and the thread count.
# Resetting makes each block's result a function of its own voxels alone, at the cost of one cold-started voxel per block.
function reset_voxel_chains!(thread_buffer)
    (; flip_angle_work) = thread_buffer
    flip_angle_work.nnls_search_prob !== nothing && reset_warmstart!(flip_angle_work.nnls_search_prob)
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
        @inbounds flip_angle_work.α[] = deg2rad(maps.alpha[I]) # output is stored in degrees

        # Compute basis using the provided flip angle, routed through the same `final_decay_basis!` as the flip-search path, so a rerun that supplies the fitted α as a B1 map uses an identical basis.
        final_decay_basis!(flip_angle_work)
    else
        # Find optimum flip angle and compute EPG decay basis
        optimize_flip_angle!(flip_angle_work, opts)
    end

    # Calculate T2 distribution and map parameters
    T2_dist = T2_distribution!(T2_dist_work)

    # Save loop results to outputs
    save_results!(thread_buffer, maps, dist, T2_dist, opts, I)

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
# Shared grid model for the discrete flip-angle search
# =========================================================

# Decay bases, their α-derivatives, and Gram matrices at every grid angle. None depends on the signal or the worker, so one read-only copy is built per run and shared by every thread-local buffer.
function epg_grid_model(o::T2mapOptions{T}, θ::EPGParameterization{T}) where {T}
    αs = flip_angles(o)
    basis_set = EPGBasisSetFunctor(o, θ, Val((:α,)))
    As, ∇As, Gs = zeros(T, o.nTE, o.nT2, length(αs)), zeros(T, o.nTE, o.nT2, 1, length(αs)), zeros(T, o.nT2, o.nT2, length(αs))
    @views for (i, α) in enumerate(αs)
        A = As[:, :, i] # bound once so that `mul!` sees `A' === A` and takes the symmetric rank-k path, making the Gram exactly symmetric
        ∇epg_decay_basis!(basis_set, ∇As[:, :, :, i], A, restructure(θ, (; α)))
        mul!(Gs[:, :, i], A', A)
    end
    return (; As, ∇As, Gs)
end

# Scratch for the α-polish: the NNLS problem at the polish point, plus the columns and accumulators the envelope derivatives are assembled from one support column at a time.
struct AlphaPolishWorkspace{T, P}
    prob::P
    Acol::Vector{T} # ETL, value column, discarded
    ∂Acol::Vector{T} # ETL, α-derivative column
    ∂²Acol::Vector{T} # ETL, α-second-derivative column
    ∂Ax::Vector{T} # ETL, Σ_j x_j ∂A[:, j]
    ∂²Ax::Vector{T} # ETL, Σ_j x_j ∂²A[:, j]
    q::Vector{T} # nT2, one entry per passive column; see `polish_grad_hess`
end
function AlphaPolishWorkspace(decay_basis::Matrix{T}, decay_data::Vector{T}) where {T}
    ETL, nT2 = size(decay_basis)
    return AlphaPolishWorkspace(NNLSProblem(decay_basis, decay_data), (zeros(T, ETL) for _ in 1:5)..., zeros(T, nT2))
end

# =========================================================
# Flip angle optimization
# =========================================================
struct FlipAngleOptimizationWorkspace{T, B, E, S, R, C, P}
    decay_basis::Matrix{T} # decay basis at the current flip angle `α`
    decay_data::Vector{T} # decay curve data
    decay_basis_set::B # B <: EPGBasisSetFunctor{T, ETL}
    nnls_search_prob::E # E <: Union{Nothing, NNLSDiscreteSurrogateSearch{1, T}}; reads the shared grid model, owns the per-voxel solve state
    α::Base.RefValue{T} # current flip angle, in degrees
    α_surrogate::S # S <: Union{Nothing, AbstractSurrogate{1, T}}
    α_searcher::R # R <: Union{Nothing, DiscreteSurrogateSearcher{1, T}}; reused across voxels
    decay_basis_work::C # C <: Union{Nothing, EPGCosineSeriesBasis{T}}; evaluates the decay basis at arbitrary α, or nothing when the cosine representation does not apply
    α_polish_work::P # P <: Union{Nothing, AlphaPolishWorkspace{T}}; loss/gradient evaluation at the continuous surrogate minimizer, see `polish_flip_angle!`
end

function FlipAngleOptimizationWorkspace(o::T2mapOptions{T}, decay_basis::Matrix{T}, decay_data::Vector{T}, global_buffer = global_buffer_maker(o)) where {T}
    α = Ref(o.SetFlipAngle === nothing ? T(NaN) : deg2rad(o.SetFlipAngle))
    θ = default_epg_parameters(o)
    decay_basis_set = EPGBasisSetFunctor(o, θ, Val((:α,)))

    if o.SetFlipAngle !== nothing
        # Compute basis for fixed `SetFlipAngle`
        epg_decay_basis!(decay_basis_set, decay_basis, SA{T}[α[]])
        nnls_search_prob = nothing
        α_surrogate = nothing
        α_searcher = nothing
    else
        (; As, ∇As, Gs) = global_buffer.grid_model
        nnls_search_prob = NNLSDiscreteSurrogateSearch(As, ∇As, Gs, (flip_angles(o),), decay_data; legacy = o.legacy)
        α_surrogate = o.legacy ? CubicSplineSurrogate(nnls_search_prob; legacy = true) : CubicHermiteSplineSurrogate(nnls_search_prob)
        α_searcher = DiscreteSurrogateSearcher(α_surrogate.grid) # reused per voxel; see `optimize_flip_angle!`
    end

    # Each worker wraps the shared read-only cosine coefficients in its own evaluation scratch
    decay_basis_work = global_buffer.decay_basis_work === nothing ? nothing : EPGCosineSeriesBasis(global_buffer.decay_basis_work)

    # α-polish needs the gradient-carrying surrogate. The continuous-α basis is the cosine series when `RefConAngle == 180`, and the EPG recurrence with AD derivative columns otherwise.
    α_polish_work = α_surrogate isa CubicHermiteSplineSurrogate ? AlphaPolishWorkspace(decay_basis, decay_data) : nothing

    return FlipAngleOptimizationWorkspace(decay_basis, decay_data, decay_basis_set, nnls_search_prob, α, α_surrogate, α_searcher, decay_basis_work, α_polish_work)
end

# One refinement of the surrogate minimizer, followed by a certificate. The discrete search evaluates the true loss f(α) = min_{x≥0} ‖A(α)x − b‖² only at grid nodes, so its continuous minimizer α₀ carries the cubic-Hermite interpolation error of the enclosing cell.
# A true off-grid evaluation builds A(α₀), solves NNLS, and forms the envelope gradient g₀ = -2·rᵀA′(α₀)x. Minimizing both adjacent cubics then proposes one candidate α₁.
# The returned angle is the lex-minimum of true losses over the best evaluated node, α₀, and α₁, so f(α̂) ≤ min_{j evaluated} f(αⱼ) however badly the interpolant misbehaves.
# Returns `true` iff `work.decay_basis` already holds the exact basis at the final `work.α[]`, letting the caller skip the final rebuild.
function polish_flip_angle!(work::FlipAngleOptimizationWorkspace{T}) where {T}
    (; grid, seen, u, ∇u) = work.α_surrogate
    α₀ = work.α[]
    jstar = best_seen_index(work.α_surrogate)

    # Landing on a node leaves nothing to refine, since every candidate would then be a value the search already has.
    i = searchsortedlast(grid, SA{T}[α₀]; by = first)
    if !(1 <= i < length(grid) && first(grid[i]) < α₀)
        work.α[] = first(grid[jstar])
        seed_downstream_from_node!(work, jstar)
        return false
    end
    # `bisection_search` returns either a resolved proposal or an exhausted budget, and the budget equals the grid length, so exhausting it evaluates every node and resolves every cell. Either way the Hermite data below are true evaluations.
    @assert seen[i] && seen[i+1] "bisection_search returned an unresolved proposal"

    # True loss and envelope derivatives at α₀
    f₀, g₀, f″₀ = polish_loss_grad!(work, α₀)

    # Minimize both adjacent cubics and take the lower prediction. Ties break toward the candidate nearer α₀ and then the larger angle, hence the negated sort key.
    αl, ũl = minimize(CubicHermiteInterpolator(first(grid[i]), α₀, u[i], f₀, first(∇u[i]), g₀))
    αr, ũr = minimize(CubicHermiteInterpolator(α₀, first(grid[i+1]), f₀, u[i+1], g₀, first(∇u[i+1])))
    α₁ = -min((ũl, abs(αl - α₀), -αl), (ũr, abs(αr - α₀), -αr))[3]

    # A Newton step on the true loss uses only local data at α₀, so the cell width the cubics interpolate over does not limit it. It is taken when the local model is a genuine minimum and stays in the cell; otherwise the cubic proposal stands.
    if f″₀ > 0
        αₙ = α₀ - g₀ / f″₀
        first(grid[i]) < αₙ < first(grid[i+1]) && (α₁ = αₙ)
    end

    # One true evaluation at the proposal. A candidate that returns α₀ or lands on a bracket endpoint already has an exact loss, so it costs no basis build and no solve.
    f₁, solved_at_α₁ = α₁ == α₀ ? (f₀, false) :
                       α₁ == first(grid[i]) ? (u[i], false) :
                       α₁ == first(grid[i+1]) ? (u[i+1], false) :
                       (polish_loss!(work, α₁), true)

    # Certify against true losses only, with the same tie-breaking as above
    αstar = first(grid[jstar])
    α̂ = -min((u[jstar], abs(αstar - α₀), -αstar), (f₀, zero(T), -α₀), (f₁, abs(α₁ - α₀), -α₁))[3]
    work.α[] = α̂

    # The winner is the angle of the last solve exactly when the basis needs no rebuild, in which case the polish workspace is left solved at α̂ and the T2 stage adopts it.
    # A grid winner instead seeds that solve from the support the search stored for its node.
    basis_current = α̂ == (solved_at_α₁ ? α₁ : α₀)
    !basis_current && α̂ == αstar && seed_downstream_from_node!(work, jstar)
    return basis_current
end

# Build A(α) into `work.decay_basis`. At `RefConAngle == 180` the finite cosine series evaluates the basis exactly and cheaply; otherwise the EPG recurrence evaluates the same model directly, which is slower but not an approximation.
polish_basis!(work::FlipAngleOptimizationWorkspace{T}, cosine::EPGCosineSeriesBasis{T}, α::T) where {T} = epg_decay_basis!(work.decay_basis, cosine, α) # leaves the trigonometric features current at α
polish_basis!(work::FlipAngleOptimizationWorkspace{T}, ::Nothing, α::T) where {T} = epg_decay_basis!(work.decay_basis_set, work.decay_basis, SA{T}[α])

# Envelope derivatives of f(α) = min_{x≥0} ‖A(α)x − b‖², over the support only: a full derivative tensor is never needed off grid.
# With r = b − Ax, ∂A and ∂²A the α-derivatives of the passive columns A_P, and G = A_PᵀA_P, constrained variable projection gives
#   f′ = −2·rᵀ∂Ax,   f″ = 2‖∂Ax‖² − 2·rᵀ∂²Ax − 2·qᵀG⁻¹q,   q = A_Pᵀ(∂Ax) − ∂Aᵀr.
# The final term is what distinguishes the profiled curvature from that of a frozen x, and it is available for the cost of one triangular solve: the NNLS solve already left the factor G = UᵀU, so qᵀG⁻¹q = ‖U⁻ᵀq‖².
# Valid where the passive set and the response signs are constant, which is why the returned angle is certified against true losses rather than trusted.
# Returns `f″ = NaN` when second derivatives are unavailable, which the caller reads as "no Newton candidate".
function polish_grad_hess(work::FlipAngleOptimizationWorkspace{T}, cosine::EPGCosineSeriesBasis{T}, α::T, x, r, idx) where {T}
    (; decay_basis, α_polish_work) = work
    (; Acol, ∂Acol, ∂²Acol, ∂Ax, ∂²Ax, q) = α_polish_work # `Acol` is discarded; the value column is already in `decay_basis`
    (; nnls_work) = α_polish_work.prob
    fill!(∂Ax, zero(T))
    fill!(∂²Ax, zero(T))
    g = zero(T)
    @inbounds for (t, j) in enumerate(idx)
        epg_decay_basis_∂α_col!(Acol, ∂Acol, ∂²Acol, cosine, α, j)
        xⱼ, dⱼ = x[j], zero(T)
        @simd for i in eachindex(∂Ax)
            ∂ᵢ = ∂Acol[i]
            ∂Ax[i] = muladd(xⱼ, ∂ᵢ, ∂Ax[i])
            ∂²Ax[i] = muladd(xⱼ, ∂²Acol[i], ∂²Ax[i])
            dⱼ = muladd(∂ᵢ, r[i], dⱼ)
        end
        q[t] = -dⱼ # the A_Pᵀ(∂Ax) half needs the whole ∂Ax, so it is added in the second pass below
        g = muladd(xⱼ, dⱼ, g)
    end
    p = length(idx) # A_Pᵀ(∂Ax) needs ∂Ax whole, so it cannot fuse into the pass above
    @inbounds @views for (t, j) in enumerate(idx)
        q[t] += dot(decay_basis[:, j], ∂Ax)
    end
    qₚ = @views q[1:p] # the solve is in place, and `q` is not read again
    NNLS.solve_triangular_system!(qₚ, NNLS.choleskyfactor(nnls_work, Val(:U)), p, Val(true))
    return -2 * g, 2 * (dot(∂Ax, ∂Ax) - dot(r, ∂²Ax) - dot(qₚ, qₚ))
end

function polish_grad_hess(work::FlipAngleOptimizationWorkspace{T}, ::Nothing, α::T, x, r, idx) where {T}
    (; decay_basis_set, α_polish_work) = work
    (; Acol, ∂Acol) = α_polish_work
    θα, ∇col = restructure(decay_basis_set.θ, (; α)), reshape(∂Acol, :, 1) # the Jacobian writes a matrix, which `∂Acol` supplies without a second buffer
    g = zero(T)
    @inbounds for j in idx
        decay_basis_set.epg_jac_functor!(∇col, Acol, restructure(θα, (; T2 = decay_basis_set.T2_times[j])))
        g = muladd(x[j], dot(∂Acol, r), g)
    end
    #TODO Supply ∂²A here so general β also gets a Newton candidate. Nested AD over the EPG recurrence would work but a hand-written α-derivative would be far cheaper.
    return -2 * g, T(NaN)
end

# Polish evaluation: build the basis A(α), solve NNLS warm-started from the search's final active set, and return the true loss ‖A(α)x − b‖². That active set is the support at the last grid node the search solved, not at its continuous proposal.
# Leaves `work.decay_basis` holding A(α) and the solve's residual current, so `polish_loss_grad!` can extend it with the envelope gradient.
function polish_loss!(work::FlipAngleOptimizationWorkspace{T}, α::T) where {T}
    (; nnls_search_prob, decay_basis_work, α_polish_work) = work
    polish_basis!(work, decay_basis_work, α)
    solve_unreg!(α_polish_work.prob, nnls_search_prob.nnls_work.nnls_work)
    r = NNLS.residual(α_polish_work.prob.nnls_work)
    return dot(r, r)
end

# Polish evaluation with the envelope-theorem derivatives, for the bracket endpoint the Hermite step interpolates from.
function polish_loss_grad!(work::FlipAngleOptimizationWorkspace{T}, α₀::T) where {T}
    f₀ = polish_loss!(work, α₀)

    # The solve left r = b − A_P x_P current, so the support is already known; scanning all n columns for positivity would rediscover it.
    (; nnls_work) = work.α_polish_work.prob
    x, r = NNLS.solution(nnls_work), NNLS.residual(nnls_work)
    g₀, f″₀ = polish_grad_hess(work, work.decay_basis_work, α₀, x, r, NNLS.components(nnls_work))
    return f₀, g₀, f″₀
end

# Source the T2 stage draws its unregularized solve from; see `NNLSUnregSource`.
# The polish problem shares `decay_basis` and `decay_data` with that stage and is the only one ever solved at an off-grid α, so it is preferred; the search workspace can only seed.
unreg_source(work::FlipAngleOptimizationWorkspace) = work.α_polish_work === nothing ? work.nnls_search_prob.nnls_work.nnls_work : work.α_polish_work.prob

# Seed the T2-stage solve from the support the search stored for grid node `j`, which is exact at αⱼ.
# Only `idx` and `nsetp` are read from an unsolved source, so the rest of the workspace may be stale.
# Every caller reaches here through the Hermite surrogate, which is constructed together with the polish workspace.
function seed_downstream_from_node!(work::FlipAngleOptimizationWorkspace, j::Int)
    (; seen_idx, seen_nsetp) = work.nnls_search_prob
    (; nnls_work) = work.α_polish_work.prob
    @inbounds for t in 1:seen_nsetp[j]
        nnls_work.idx[t] = seen_idx[t, j]
    end
    nnls_work.nsetp[] = seen_nsetp[j]
    return nothing
end

function optimize_flip_angle!(work::FlipAngleOptimizationWorkspace, o::T2mapOptions)

    if o.SetFlipAngle === nothing
        # Find optimal flip angle
        empty!(work.α_surrogate)
        advance_warmstart!(work.nnls_search_prob) # cross-voxel NNLS warm starting
        reset!(work.α_searcher) # reuse the searcher's buffers instead of allocating one per voxel
        initialize!(work.α_surrogate, work.α_searcher; mineval = o.nRefAnglesMin, maxeval = o.nRefAngles)
        α_opt, _ = bisection_search(work.α_surrogate, work.α_searcher; maxeval = o.nRefAngles)
        work.α[] = α_opt[1]

        # Refine the surrogate minimizer against true off-grid loss evaluations; `true` means the basis at the final α is already built.
        # The legacy cubic-spline surrogate has no polish workspace: it carries no envelope derivatives, and its returned angle is the MATLAB-compatible reference.
        basis_current = work.α_polish_work !== nothing && polish_flip_angle!(work)

        # Compute basis using optimized flip angles
        basis_current || final_decay_basis!(work)
    end

    return nothing
end

# Build the T2-stage decay basis at the current flip angle `work.α[]`.
# Uses the exact cosine-series evaluation when it applies, and the exact EPG rebuild otherwise.
# Overwriting the basis invalidates the polish problem's solution against it; see `issolved`.
function final_decay_basis!(work::FlipAngleOptimizationWorkspace)
    α = work.α[]
    if work.decay_basis_work !== nothing
        epg_decay_basis!(work.decay_basis, work.decay_basis_work, α)
    else
        epg_decay_basis!(work.decay_basis_set, work.decay_basis, SA[α])
    end
    work.α_polish_work !== nothing && (work.α_polish_work.prob.nnls_work.solved[] = false)
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

nnls_workspace(::NoRegularization, decay_basis::AbstractMatrix{T}, decay_data::AbstractVector{T}, nnls_prob_seed, args...) where {T} = NNLSUnregProblem(decay_basis, decay_data, nnls_prob_seed)
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

# =========================================================
# Save thread local results to output maps
# =========================================================
function save_results!(thread_buffer, maps::T2Maps{T}, dist::T2Distributions{T}, T2_dist::AbstractVector{T}, o::T2mapOptions{T}, I::CartesianIndex) where {T}
    (; logT2_times, decay_basis, decay_data, decay_scale, decay_curvefit, residuals, flip_angle_work, T2_dist_work) = thread_buffer

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
        alpha[I] = rad2deg(flip_angle_work.α[]) # optimized refocusing pulse flip angle, in degrees
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
function thread_buffer_maker(o::T2mapOptions{T}, global_buffer = global_buffer_maker(o)) where {T}
    decay_basis = zeros(T, o.nTE, o.nT2)
    decay_data = zeros(T, o.nTE)
    decay_scale = Ref(one(T))
    flip_angle_work = FlipAngleOptimizationWorkspace(o, decay_basis, decay_data, global_buffer)
    search_prob = flip_angle_work.nnls_search_prob
    nnls_prob_seed = search_prob === nothing ? nothing : unreg_source(flip_angle_work)
    dof_interpolator = search_prob === nothing || !(regularization_method(o) isa GCV) ? nothing : (GriddedSpectrumInterpolator(search_prob.As, search_prob.∇As, flip_angles(o)), flip_angle_work.α)
    return (;
        T2_times         = logrange(o.T2Range..., o.nT2),
        logT2_times      = log.(logrange(o.T2Range..., o.nT2)),
        flip_angles      = flip_angles(o),
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

# Everything the workers share read-only: the cosine coefficient tensor, and the decay bases, α-derivatives, and Gram matrices over the flip-angle grid.
# All are voxel- and thread-independent, so one instance is built per run and every worker reads it.
# `decay_basis_work` is `nothing` when the cosine representation does not apply, namely in legacy mode, for a fixed flip angle, or for a non-constant flip angle; `grid_model` is `nothing` when there is no flip-angle grid to search.
function global_buffer_maker(o::T2mapOptions{T}) where {T}
    θ = default_epg_parameters(o)
    decay_basis_work = !o.legacy && o.SetFlipAngle === nothing && θ isa EPGConstantFlipAngleOptions ? EPGCosineSeriesBasis(θ, T2_component_times(o)) : nothing
    grid_model = o.SetFlipAngle === nothing ? epg_grid_model(o, θ) : nothing
    return (; decay_basis_work, grid_model)
end

function default_epg_parameters(o::T2mapOptions{T}) where {T}
    return o.RefConAngle == 180 ?
           EPGConstantFlipAngleOptions((; ETL = o.nTE, α = T(NaN), TE = o.TE, T2 = T(NaN), T1 = o.T1)) :
           EPGOptions((; ETL = o.nTE, α = T(NaN), TE = o.TE, T2 = T(NaN), T1 = o.T1, β = deg2rad(o.RefConAngle)))
end
