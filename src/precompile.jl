function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing

    for T in (Float32, Float64)
        precompile(Tuple{typeof(lsqnonneg), Matrix{T}, Vector{T}})
        precompile(Tuple{typeof(lsqnonneg_reg), Matrix{T}, Vector{T}, T})
        precompile(Tuple{typeof(lsqnonneg_lcurve), Matrix{T}, Vector{T}})
        precompile(Tuple{typeof(EPGdecaycurve), Int, T, T, T, T, T})
        precompile(Tuple{typeof(T2mapSEcorr), Array{T,4}})
        precompile(Tuple{typeof(T2partSEcorr), Array{T,4}})
    end

    precompile(Tuple{typeof(ArgParse.parse_args), Array{T,1} where T<:AbstractString, ArgParse.ArgParseSettings})
    precompile(Tuple{typeof(ArgParse.parse_args), Array{T,1} where T, ArgParse.ArgParseSettings})
    precompile(Tuple{typeof(ArgParse.parse_item_wrapper), Type, AbstractString})
    precompile(Tuple{typeof(ArgParse.parse1_optarg), ArgParse.ParserState, ArgParse.ArgParseSettings, ArgParse.ArgParseField, Any, AbstractString})
    precompile(Tuple{typeof(ArgParse.preparse), Channel, ArgParse.ParserState, ArgParse.ArgParseSettings})
    precompile(Tuple{typeof(ArgParse.parse1_flag), ArgParse.ParserState, ArgParse.ArgParseSettings, ArgParse.ArgParseField, Bool, AbstractString})
    precompile(Tuple{typeof(ArgParse.read_args_from_files), Any, Any})
    precompile(Tuple{typeof(ArgParse.check_long_opt_name), AbstractString, ArgParse.ArgParseSettings})
    precompile(Tuple{typeof(ArgParse.check_short_opt_name), AbstractString, ArgParse.ArgParseSettings})

    for T in (Float32, Float64)
        precompile(Tuple{typeof(sorted_arg_table_entries), T2mapOptions{T}, T2partOptions{T}})
    end
    precompile(Tuple{typeof(get_parsed_args_subset), Dict{Symbol,Any}, Dict{Symbol,Type}})
    precompile(Tuple{typeof(create_argparse_settings)})
    precompile(Tuple{typeof(main), Vector{String}})
end
