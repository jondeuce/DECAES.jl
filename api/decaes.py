import argparse
import os
import julia

parser = argparse.ArgumentParser(
    description = """DECAES (DE)composition and (C)omponent (A)nalysis of (E)xponential (S)ignals

    Call out to the DECAES command line interface (CLI).

    The documented optional arguments below are used to initialize Julia and/or DECAES.
    Any additional arguments are forwarded to the DECAES CLI; see the documentation
    for more information:

        https://jondeuce.github.io/DECAES.jl/dev/cli/#Arguments

    Example:

        Install DECAES if necessary, start Julia with 8 threads, and run DECAES
        using settings specified by 'decaes_settings_file.txt':

            python decaes.py --install --threads=8 @/path/to/decaes_settings_file.txt

    NOTE:

        This script requires pyjulia to be installed. See the documentation
        for installation instructions:

            https://github.com/JuliaPy/pyjulia#quick-usage
    """,
    formatter_class = argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--runtime", default = "julia", type = str, help = "path to julia runtime binary; defaults to 'julia'")
parser.add_argument("--threads", default = os.cpu_count(), type = int, help = "number of threads to start julia with; defaults to os.cpu_count()")
parser.add_argument("--project", type = str, help = "julia project environment; if unspecified, the default global environment is used")
parser.add_argument("--install", action = "store_true", help = "install DECAES into the julia project environment specified by --project")

args, decaes_args = parser.parse_known_args()

# Configure Julia environment variables
os.environ["JULIA_NUM_THREADS"] = str(args.threads)

if args.project is not None:
    os.environ["JULIA_PROJECT"] = args.project

# Initialize Julia runtime. Precompilation of Julia modules is not supported on Debian-based
# Linux distributions such as Ubuntu, or python on installations via Conda.
#   See: https://pyjulia.readthedocs.io/en/stable/troubleshooting.html#your-python-interpreter-is-statically-linked-to-libpython
from julia.api import Julia
jl = Julia(compiled_modules = False, runtime = args.runtime)
from julia import Base
from julia import Pkg

# Install DECAES, if necessary
if args.install:
    Pkg.add("DECAES")

# Try importing DECAES
try:
    from julia import DECAES

except ImportError:
    print("""
    Error importing DECAES. Most likely, DECAES is not installed in the current julia project environment:

        {project}

    Pass the --install flag to have DECAES automatically installed.
    Alternatively, manually install DECAES into the above project environment using the command:

        {runtime} --project={project} -e 'import Pkg; Pkg.add("DECAES")'
    """
    .format(runtime = args.runtime, project = os.path.dirname(Base.active_project())))
    raise

# Run DECAES
DECAES.main(decaes_args)
