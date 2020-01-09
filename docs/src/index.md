# DECAES.jl

## Introduction

Myelin Water Imaging (DECAES) is a magnetic resonance imaging (MRI) technique used to visualize the myelin water contained in the brain's white matter.
This package implements a method of computing the myelin water fraction (MWF) - the fraction of water trapped in myelin lipid bilayers relative to the total water in the region - which was pioneered at the University of British Columbia (UBC) by Alex MacKay and Ken Whittal.

DECAES works by analyzing multi spin-echo type MRI scans.
The multi-echo signal is decomposed into exponential components using a regularized inverse Laplace transform-based technique.
This method involves solving the regularized nonnegative least squares (NNLS) problem

```math
X = \mathrm{argmin}_{x \ge 0} ||Cx - d||_2^2 + \mu||x||_2^2
```

where $d$ is the signal decay data, $C$ is a matrix of exponential decay bases, and $\mu$ is a regularization parameter.
$C$ is constructed using the extended phase graph algorithm with stimulated echo correction and consists of bases with varying $T_2$ times.
The output $X$ is the spectrum of (nonnegative) exponential decay times, interpreted physically as the distribution of $T_2$-times arising from the multi-echo signal.
Through analysing the resulting $T_2$-distribution, one can separate the contribution due to the myelin water from the intra- and extra-cellular water and compute the MWF.

Basics of myelin water imaging:
1. <https://doi.org/10.1002/mrm.1910310614>
2. <https://doi.org/10.1016/0022-2364(89)90011-5>
3. <https://doi.org/10.1016/j.neuroimage.2012.06.064>
4. <https://doi.org/10.1002/mrm.23157>

Validation of myelin water imaging:
1. <https://doi.org/10.1016/j.neuroimage.2007.12.008>
2. <https://doi.org/10.1016/j.neuroimage.2017.03.065>
3. <https://doi.org/10.1016/j.neuroimage.2019.05.042>

Some applications of myelin water imaging:
1. <https://doi.org/10.1177/1352458517723717>
2. <https://doi.org/10.1038/s41598-018-33112-8>
3. <https://doi.org/10.1371/journal.pone.0150215>

## [Installation](@id installation)

Start `julia` from the command line, type `]` to enter the package manager REPL mode (the `julia>` prompt will be replaced by a `pkg>` prompt), and enter the following command:

```julia
pkg> add https://github.com/jondeuce/DECAES.jl.git
```

Once the package is finished installing, type the backspace key to exit the package manager REPL mode (the `julia>` prompt should reappear).
Exit Julia using the keyboard shortcut `Ctrl+D`, or by typing `exit()`.

## Table of contents

```@contents
Pages = [
    "t2map.md",
    "t2part.md",
    "cli.md",
    "ref.md",
]
Depth = 1
```

## Acknowledgements

* Porting to Julia was done by Jonathan Doucette (email: jdoucette@phas.ubc.ca) in November 2019. This work was funded by NSERC (016-05371) and CIHR (RN382474-418628) (PI Alexander Rauscher, University of British Columbia)
* Christian Kames contributed to optimizing the Julia port for both speed and memory efficiency
* Original MATLAB code was written by Thomas Prasloski (email: tprasloski@gmail.com).
Modifications to the MATLAB code were made by Vanessa Wiggermann to enable processing on various MATLAB versions in February 2019.
The Julia port is based on this modified version.
