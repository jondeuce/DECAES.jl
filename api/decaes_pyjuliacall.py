"""
Load the DECAES.jl Julia package from python.
DECAES.jl will be installed automatically, if necessary.

This module requires the juliacall Python package to be installed.
See the following documentation for instructions:

    https://juliapy.github.io/PythonCall.jl/stable/juliacall/

Quick usage:

    from decaes import DECAES

    # Call methods from the DECAES module as usual, e.g.
    DECAES.T2mapSEcorr(image, kwargs...)

This version of decaes.py was written for DECAES v0.6.0.
"""
from juliacall import Main as jl
import juliapkg
global DECAES

try:
    jl.seval("using DECAES")
    DECAES = jl.DECAES
except:
    juliapkg.require_julia("1.9")
    juliapkg.add("DECAES", "d84fb938-a666-558e-89d9-d531edc6724f")
    jl.seval("using DECAES")
    DECAES = jl.DECAES
