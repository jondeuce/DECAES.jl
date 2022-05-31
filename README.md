# DEcomposition and Component Analysis of Exponential Signals (DECAES)

<p align="left">
<img width="500px" src="https://github.com/jondeuce/DECAES.jl/blob/c2956262063841c8c2dc27f4e0ee20593ef32697/docs/src/assets/logo.gif">
</p>

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jondeuce.github.io/DECAES.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jondeuce.github.io/DECAES.jl/dev)
[![Build Status](https://github.com/jondeuce/DECAES.jl/workflows/CI/badge.svg)](https://github.com/jondeuce/DECAES.jl/actions?query=workflow%3ACI)
[![codecov.io](https://codecov.io/github/jondeuce/DECAES.jl/branch/master/graph/badge.svg)](http://codecov.io/github/jondeuce/DECAES.jl/branch/master)

DECAES is a *fast* Julia implementation of the [MATLAB toolbox](https://mriresearch.med.ubc.ca/news-projects/myelin-water-fraction/) from the [UBC MRI Research Centre](https://mriresearch.med.ubc.ca/) for computing voxelwise [T2-distributions](https://doi.org/10.1016/0022-2364(89)90011-5) from multi spin-echo MRI images using the extended phase graph algorithm with stimulated echo corrections.
Post-processing of these T2-distributions allows for the computation of measures such as the [myelin water fraction (MWF)](https://doi.org/10.1002/mrm.1910310614) or the [luminal water fraction (LWF)](https://doi.org/10.1148/radiol.2017161687).

DECAES is written in the open-source [Julia programming language](https://julialang.org/).
Julia and command line interfaces are available through this package.
The [examples repository](https://github.com/jondeuce/mwiexamples) additionally provides a MATLAB interface via the MATLAB function [`decaes.m`](https://github.com/jondeuce/DECAES.jl/blob/master/api/decaes.m), as well as a Python interface via the [`decaes.py`](https://github.com/jondeuce/DECAES.jl/blob/master/api/decaes.py) module.
If you use DECAES in your research, please [cite](CITATION.bib) our work.

## Installation

In Julia v1.6 or later you can install DECAES from the Pkg REPL:
```
pkg> add DECAES
```

## Documentation

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jondeuce.github.io/DECAES.jl/dev)

Find package documentation at the above link, which includes:
* The command line interface [API](https://jondeuce.github.io/DECAES.jl/dev/cli), available [command line arguments](https://jondeuce.github.io/DECAES.jl/dev/cli/#Arguments-1), and [examples](https://jondeuce.github.io/DECAES.jl/dev/cli/#Examples-1)
* API reference detailing how to use DECAES.jl from within Julia
* Other internals and algorithmic details

## DECAES tutorial

If you are new to DECAES, the best place to start is the [examples repository](https://github.com/jondeuce/mwiexamples).
There, we provide a walk-through tutorial for using the MATLAB and command-line interfaces for DECAES, including example multi spin-echo (MSE) data for performing MWI.

## Command Line Interface

This toolbox provides a command line interface (CLI) for processing from the terminal.
The CLI takes image files as inputs and performs one or both of T2-distribution computation and T2-parts analysis, the latter of which performs post-processing of the T2-distribution to calculate parameters such as the MWF or LWF.
The input image must be one of the following file types:

1. [NIfTI file](https://nifti.nimh.nih.gov/) with extension `.nii`, or [gzip](https://www.gzip.org/) compressed NIfTI file with extension `.nii.gz`
2. [MATLAB file](https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html) with extension `.mat`
3. Philips [PAR/REC](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage#Philips_PAR.2FREC_Images) file pair with extensions `.par` and `.rec` (or `.PAR` and `.REC`)
4. Philips XML/REC file pair with extensions `.xml` and `.rec` (or `.XML` and `.REC`)

All output files are saved as `.mat` files in format `v7.3`.

* **Note:** if your data is in DICOM format, the [freely available `dcm2niix` tool](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage) is able to convert [DICOM](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage#General_Usage) files into NIfTI format.

### Basic usage

There are two equivalent ways to use the CLI, assuming DECAES is already installed:

**1. Helper script:** Create a script called e.g. `decaes.jl` with the following contents (or, download the script located [here](https://github.com/jondeuce/DECAES.jl/blob/master/api/decaes.jl)):

```julia
using DECAES # load the package
main() # call CLI entrypoint function
```

This script can then be invoked from the command line as follows:

```bash
$ julia --threads=auto decaes.jl -- <COMMAND LINE ARGS> # --threads=auto enables parallel processing
```

**2. Julia `-e` flag:** The contents of the above script can be written directly at the command line using the `-e` (for "evaluate") flag:

```bash
$ julia --threads=auto -e 'using DECAES; main()' -- <COMMAND LINE ARGS> # --threads=auto enables parallel processing
```

## Benchmarks

<center>

| Dataset     | Matrix Size     | CPU               | Cores | MATLAB     | **DECAES** |
| :---:       | :---:           | :---:             | :---: | :---:      | :---:      |
| 48-echo MSE | 240 x 240 x 48  | Intel i5 4200U    | 2     | 4h:35m:18s | **7m:49s** |
| 56-echo MSE | 240 x 240 x 113 | Xeon E5-2640 (x2) | 12    | 1h:25m:01s | **2m:39s** |
| 48-echo MSE | 240 x 240 x 48  | Xeon E5-2640 (x2) | 12    | 59m:40s    | **1m:40s** |
| 56-echo MSE | 240 x 240 x 113 | Ryzen 9 3950X     | 16    | 22m:33s    | **43s**    |
| 48-echo MSE | 240 x 240 x 48  | Ryzen 9 3950X     | 16    | 17m:56s    | **27s**    |

</center>

Benchmarking notes:

* MATLAB scripts used were from the `MWI_NNLS_toolbox_0319` subfolder of the [ubcmwf github repository](https://github.com/ubcmri/ubcmwf)
* DECAES.jl was compiled into an [app](https://julialang.github.io/PackageCompiler.jl/stable/apps.html) using the `--compile` flag to reduce compile time overhead
* Both implementations made use of precomputed brain masks to skip voxels outside of the brain

## JuliaCon 2021

[![JuliaCon 2021 - Matlab to Julia: Hours to Minutes for MRI Image Analysis](https://imgur.com/u364KDv)](https://www.youtube.com/watch?v=6OxsK2R5VkA)

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
