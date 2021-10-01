####
#### Unregularized NNLS problem
####

struct NNLSProblem{T, MC <: AbstractMatrix{T}, Vd <: AbstractVector{T}, W}
    C::MC
    d::Vd
    m::Int
    n::Int
    nnls_work::W
end
function NNLSProblem(C::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    m, n = size(C)
    nnls_work = NNLS.NNLSWorkspace(C, d)
    NNLSProblem(C, d, m, n, nnls_work)
end

function solve!(work::NNLSProblem, C, d)
    # Solve NNLS problem
    NNLS.load!(work.nnls_work, C, d)
    NNLS.nnls!(work.nnls_work)
    return solution(work)
end
solve!(work::NNLSProblem) = solve!(work, work.C, work.d)

solution(work::NNLSProblem) = NNLS.solution(work.nnls_work)

chi2(work::NNLSProblem) = NNLS.residualnorm(work.nnls_work)^2

"""
    lsqnonneg(C::AbstractMatrix, d::AbstractVector)

Returns the nonnegative least-squares (NNLS) solution, X, of the equation:

```math
X = \\mathrm{argmin}_{x \\ge 0} ||Cx - d||_2^2
```

# Arguments
- `C::AbstractMatrix`: Left hand side matrix acting on `x`
- `d::AbstractVector`: Right hand side vector

# Outputs
- `X::AbstractVector`: NNLS solution
"""
lsqnonneg(C, d) = lsqnonneg!(lsqnonneg_work(C, d))
lsqnonneg_work(C, d) = NNLSProblem(C, d)
lsqnonneg!(work::NNLSProblem) = solve!(work)
lsqnonneg!(work::NNLSProblem{T}, C::AbstractMatrix{T}, d::AbstractVector{T}) where {T} = solve!(work, C, d)
