import argparse
import os
import subprocess

parser = argparse.ArgumentParser(
    description = """DECAES (DE)composition and (C)omponent (A)nalysis of (E)xponential (S)ignals

    Call out to the DECAES command line interface (CLI).

    The documented optional arguments below are used to initialize Julia and/or DECAES.
    Any additional arguments are forwarded to the DECAES CLI; see the documentation
    for more information:

        https://jondeuce.github.io/DECAES.jl/dev/cli/#Arguments

    Example:

        Start Julia with 8 threads and run DECAES using parameters specified by the
        settings file 'decaes_settings_file.txt':

            python decaes.py --threads=8 @/path/to/decaes_settings_file.txt
    """,
    formatter_class = argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--runtime", default = "julia", type = str, help = "path to julia runtime binary; defaults to 'julia'")
parser.add_argument("--threads", default = os.cpu_count(), type = int, help = "number of threads to start julia with; defaults to os.cpu_count()")
parser.add_argument("--project", type = str, help = "julia project environment; if unspecified, the default global environment is used")
parser.add_argument("--quiet", action = "store_true", help = "suppress terminal output")

args, decaes_args = parser.parse_known_args()

# Configure Julia environment variables
os.environ["JULIA_NUM_THREADS"] = str(args.threads)

if args.project is not None:
    os.environ["JULIA_PROJECT"] = args.project

def jlcall(jlargs):
    cmd = [
        args.runtime,
        "--startup-file=no",
        "--optimize=3",
    ]

    if args.quiet:
        cmd.append("--quiet")

    for jlarg in jlargs:
        cmd.append(str(jlarg))

    subprocess.run(cmd)

# Import DECAES, installing if not found in project environment
decaes_cmd = "try import DECAES; catch e; import Pkg; Pkg.add(\"DECAES\"); import DECAES; end; DECAES.main()"

# Silence DECAES, if requested
if args.quiet:
    decaes_args.append("--quiet")

# Build Julia command args
jlargs = ["-e", decaes_cmd, "--", *decaes_args]

jlcall(jlargs)
