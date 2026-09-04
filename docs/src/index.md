# DEcomposition and Component Analysis of Exponential Signals (DECAES)

## Table of contents

```@contents
Pages = [
    "index.md",
    "cli.md",
    "ref.md",
    "internals.md",
]
Depth = 2
```

## Introduction

DECAES decomposes multiexponential signals from multi-spin echo magnetic resonance imaging (MRI) into exponential components by solving the regularized nonnegative least-squares (NNLS) problem

```math
x_{\mu} = \underset{x \ge 0}{\operatorname{argmin}}\; \lVert Ax - b\rVert_2^2 + \mu^2 \lVert x\rVert_2^2,
```

where $b$ is the signal magnitude, $A$ contains exponential decay bases constructed using the extended phase graph (EPG) algorithm with stimulated echo correction, and $\mu$ is the regularization parameter.
Each component $x_{\mu,j}$ is the nonnegative amplitude associated with decay time $T_{2,j}$, so $x_\mu$ is called the [$T_2$ distribution](@ref t2map).

## [Installation](@id installation)

Install DECAES using Julia v1.9 or later:

```bash
julia --project=@decaes -e 'import Pkg; Pkg.add("DECAES"); Pkg.build("DECAES")'
```

This installs DECAES in the named project `@decaes` and builds the command-line launcher `~/.julia/bin/decaes`.

## [Updating DECAES](@id updating)

Update DECAES with:

```bash
julia --project=@decaes -e 'import Pkg; Pkg.update("DECAES"); Pkg.build("DECAES")'
```

## Myelin water imaging

Myelin water imaging (MWI) uses $T_2$ distributions from multi-spin echo MRI to distinguish signal arising from myelin water and from intra- and extracellular water.
The myelin water fraction (MWF) is the fraction of water associated with the short-$T_2$ myelin-water component.
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
