####
#### NNLS utilities
####
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
lsqnonneg(C, d) = lsqnonneg!(lsqnonneg_work(C, d), C, d)
lsqnonneg_work(C, d) = NNLS.NNLSWorkspace(C, d)
function lsqnonneg!(work, C, d)
    NNLS.load!(work, C, d)
    NNLS.nnls!(work)
    return work.x
end
