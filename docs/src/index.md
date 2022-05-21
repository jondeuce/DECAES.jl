# DEcomposition and Component Analysis of Exponential Signals (DECAES)

## Table of contents

```@contents
Pages = [
    "index.md",
    "t2map.md",
    "t2part.md",
    "cli.md",
    "ref.md",
]
Depth = 2
```

## Introduction

DECAES provides tools for decomposing multi-exponential signals which arise from multi spin-echo magnetic resonance imaging (MRI) scans into exponential components.
The main decomposition method used is an inverse Laplace transform-based technique which involves solving the regularized nonnegative least squares (NNLS) inverse problem

```math
X = \mathrm{argmin}_{x \ge 0} ||Ax - b||_2^2 + \mu^2 ||x||_2^2
```

where $b$ is the signal magnitude data, $A$ is a matrix of exponential decay bases, and $\mu$ is a regularization parameter.
$A$ is constructed using the extended phase graph algorithm with stimulated echo correction.
The columns of $A$ are exponential bases with differing characteristic $T_2$ decay times $T_{2,j}$.

The output $X$ is the spectrum of (nonnegative) exponential decay amplitudes.
Amplitude $X_j$ of the spectrum $X$ is therefore interpreted physically as the amount of the signal $b$ which decays with time constant $T_{2,j}$.
For this reason, the spectrum $X$ is commonly referred to as the $T_2$ *distribution*.
DECAES provides methods for [computing $T_2$-distributions](@ref t2map).

## [Installation](@id installation)

Start `julia` from the command line, type `]` to enter the package manager REPL mode (the `julia>` prompt will be replaced by a `pkg>` prompt), and enter the following command:

```julia
pkg> add DECAES
```

Once the package is finished installing, type the backspace key to exit the package manager REPL mode (the `julia>` prompt should reappear).
Exit Julia using the keyboard shortcut `Ctrl+D`, or by typing `exit()`.

## [Updating DECAES](@id updating)

To update DECAES to the latest version, start `julia` from the command line, type `]` to enter the package manager REPL mode, and enter the following:

```julia
pkg> update DECAES
```

## Myelin water imaging

Myelin water imaging (MWI) is an MRI technique used to visualize the myelin water contained within the sheaths of myelinated axons within the body, such as within the brain's white matter.

Through analysing $T_2$-distributions computed from multi spin-echo MRI scans, one can separate the contribution due to the myelin water from the intra- and extra-cellular water and compute the myelin water fraction (MWF).
The MWF describes the fraction of water trapped between myelin lipid bilayers relative to the total water in the region.
DECAES provides methods for [computing the MWF](@ref t2part).

MWI was pioneered at the University of British Columbia by Alex MacKay and Ken Whittal.

Basics of myelin water imaging:
* <https://doi.org/10.1002/mrm.1910310614>
* <https://doi.org/10.1016/0022-2364(89)90011-5>
* <https://doi.org/10.1016/j.neuroimage.2012.06.064>
* <https://doi.org/10.1002/mrm.23157>

Validation of myelin water imaging:
* <https://doi.org/10.1016/j.neuroimage.2007.12.008>
* <https://doi.org/10.1016/j.neuroimage.2017.03.065>
* <https://doi.org/10.1016/j.neuroimage.2019.05.042>

Some applications of myelin water imaging:
* <https://doi.org/10.1177/1352458517723717>
* <https://doi.org/10.1038/s41598-018-33112-8>
* <https://doi.org/10.1371/journal.pone.0150215>

## Acknowledgements

* Porting to Julia was done by Jonathan Doucette (email: jdoucette@physics.ubc.ca) in November 2019. This work was funded by NSERC (016-05371) and CIHR (RN382474-418628) under PI Alexander Rauscher at the University of British Columbia
* Christian Kames (email: ckames@physics.ubc.ca) contributed to optimizing the Julia port for both speed and memory efficiency, as well as writing the PAR/XML/REC file reader used internally
* Original MATLAB code was written by Thomas Prasloski (email: tprasloski@gmail.com). Modifications to the MATLAB code were made by Vanessa Wiggermann to enable processing on various MATLAB versions in February 2019. The Julia port is based on this modified version

## Citing this work

[![Z Med Phys](https://cdn.ncbi.nlm.nih.gov/corehtml/query/egifs/https:--linkinghub.elsevier.com-ihub-images-PubMedLink.gif)](https://doi.org/10.1016/j.zemedi.2020.04.001)

If you use DECAES in your research, please cite the following:

```tex
@article{DECAES.jl-2020,
  title = {{{DECAES}} - {{DEcomposition}} and {{Component Analysis}} of {{Exponential Signals}}},
  author = {Doucette, Jonathan and Kames, Christian and Rauscher, Alexander},
  year = {2020},
  month = may,
  issn = {1876-4436},
  doi = {10.1016/j.zemedi.2020.04.001},
  journal = {Zeitschrift Fur Medizinische Physik},
  keywords = {Brain,Luminal Water Imaging,MRI,Myelin Water Imaging,Prostate},
  language = {eng},
  pmid = {32451148}
}
```
