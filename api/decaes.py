"""
Load the DECAES.jl Julia package from python.
DECAES.jl will be installed automatically, if necessary.

This module requires the julia Python package to be installed.
See the following documentation for instructions:

    https://github.com/JuliaPy/pyjulia#quick-usage

Quick usage:

    import decaes
    decaes.initialize()
    from decaes import DECAES

    # Call methods from the DECAES module as usual, e.g.
    DECAES.T2mapSEcorr(image; kwargs...)

This version of decaes.py was written for DECAES v0.5.1.
"""

import julia
import os
import subprocess
import tempfile
from julia.api import Julia

# Uninitialized DECAES module
global DECAES

# Uninitialized julia runtime object
global julia_runtime

# Various default settings
default_runtime = "julia"
default_num_threads = os.cpu_count()
default_project = ""
default_compiled_modules = True

# Set environment variable defaults
if "JULIA_NUM_THREADS" not in os.environ:
    os.environ["JULIA_NUM_THREADS"] = str(default_num_threads)

if "JULIA_PROJECT" not in os.environ:
    os.environ["JULIA_PROJECT"] = default_project

def install(
        runtime = default_runtime,
        project = default_project,
    ):
    """
    Build julia module and install DECAES
    """
    if project is not None:
        os.environ["JULIA_PROJECT"] = project

    # Build julia module using given runtime
    julia.install(julia = runtime)

    # Install DECAES.jl
    decaes_install_script = """try
            @info "Trying to import DECAES..."
            @eval import DECAES
        catch e
            @error "`import DECAES` failed" exception=(e, catch_backtrace())
            @info "Installing DECAES..."
            import Pkg
            Pkg.add("DECAES")
            @eval import DECAES
        end
        """.replace("        ", "")

    with tempfile.TemporaryDirectory() as jl_dir:
        jl_script_name = os.path.join(jl_dir, 'decaes_install.jl')
        with open(jl_script_name, "w") as jl_script:
            jl_script.write(decaes_install_script)
        subprocess.check_call([runtime, "--startup-file=no", jl_script_name])

def initialize(
        runtime = default_runtime,
        project = default_project,
        compiled_modules = default_compiled_modules,
        **kwargs,
    ):
    """
    Initialize julia runtime and import DECAES
    """
    if project is not None:
        os.environ["JULIA_PROJECT"] = project

    # Initialize Julia runtime. Precompilation of Julia modules is not supported on Debian-based
    # Linux distributions such as Ubuntu, or python on installations via Conda.
    #   See: https://pyjulia.readthedocs.io/en/stable/troubleshooting.html#your-python-interpreter-is-statically-linked-to-libpython
    global julia_runtime
    julia_runtime = Julia(
        runtime = runtime,
        compiled_modules = compiled_modules,
        **kwargs,
    )

    # Import DECAES
    global DECAES
    from julia import DECAES
