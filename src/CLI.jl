"""
    DECAES.CLI

Experimental [Julia app](https://pkgdocs.julialang.org/v1/apps/) entry point for Julia v1.12 and later.
Install the `decaes` app using the package manager:

```bash
\$ julia --project=@decaes -e 'using Pkg; Pkg.Apps.add("DECAES")'
```

The installed `decaes` app is then usable via:

```bash
\$ decaes <JULIA ARGS> -- <COMMAND LINE ARGS>
```

This forwards command line arguments to [`DECAES.main`](@ref).
Equivalently, the app CLI can be called via:

```bash
\$ julia --project=@decaes --threads=auto -m DECAES.CLI <COMMAND LINE ARGS>
```
"""
module CLI

using ..DECAES: run_main

(@main)(args::Vector{String}) = run_main(args)

end # module CLI
