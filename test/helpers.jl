is_ci() = lowercase(get(ENV, "CI", "false")) == "true"

function allocated_bytes(f::F, args::Vararg{Any, N}) where {F, N}
    f(args...)
    return @allocated f(args...)
end
