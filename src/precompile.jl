function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing

    # for T in (Float32, Float64)
    #     precompile(Tuple{typeof(lsqnonneg), Matrix{T}, Vector{T}})
    #     precompile(Tuple{typeof(lsqnonneg_reg), Matrix{T}, Vector{T}, T})
    #     precompile(Tuple{typeof(lsqnonneg_lcurve), Matrix{T}, Vector{T}})
    #     precompile(Tuple{typeof(EPGdecaycurve), Int, T, T, T, T, T})
    #     precompile(Tuple{typeof(T2mapSEcorr), Array{T,4}})
    #     precompile(Tuple{typeof(T2partSEcorr), Array{T,4}})
    # end

    for T in (Float32, Float64)
        precompile(Tuple{typeof(sorted_arg_table_entries), T2mapOptions{T}, T2partOptions{T}})
    end
    precompile(Tuple{typeof(get_parsed_args_subset), Dict{Symbol,Any}, Dict{Symbol,Type}})
    precompile(Tuple{typeof(create_argparse_settings)})
    precompile(Tuple{typeof(ArgParse.parse_args), Vector{String}, ArgParse.ArgParseSettings})
    # precompile(Tuple{typeof(main), Vector{String}})
end
