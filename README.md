# DEcomposition and Component Analysis of Exponential Signals (DECAES)

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jondeuce.github.io/DECAES.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jondeuce.github.io/DECAES.jl/dev)
[![Build Status](https://travis-ci.com/jondeuce/DECAES.jl.svg?branch=master)](https://travis-ci.com/jondeuce/DECAES.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jondeuce/DECAES.jl?svg=true)](https://ci.appveyor.com/project/jondeuce/DECAES-jl)
[![Codecov](https://codecov.io/gh/jondeuce/DECAES.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jondeuce/DECAES.jl)
[![Coveralls](https://coveralls.io/repos/github/jondeuce/DECAES.jl/badge.svg?branch=master)](https://coveralls.io/github/jondeuce/DECAES.jl?branch=master)
[![Build Status](https://api.cirrus-ci.com/github/jondeuce/DECAES.jl.svg)](https://cirrus-ci.com/github/jondeuce/DECAES.jl)

DECAES.jl is a *fast* Julia implementation of the [MATLAB toolbox](https://mriresearch.med.ubc.ca/news-projects/myelin-water-fraction/) from the [UBC MRI Research Centre](https://mriresearch.med.ubc.ca/) for computing voxelwise [T2-distributions](https://doi.org/10.1016/0022-2364(89)90011-5) from multi spin-echo MRI images using the extended phase graph algorithm with stimulated echo corrections.
Post-processing of these T2-distributions allows for the computation of measures such as the [myelin water fraction (MWF)](https://doi.org/10.1002/mrm.1910310614) or the [luminal water fraction (LWF)](https://doi.org/10.1148/radiol.2017161687).

DECAES.jl is written in the open-source [Julia programming language](https://julialang.org/).
Julia and command line interfaces are available through this package.
The [examples repository](https://github.com/jondeuce/mwiexamples) additionally provides a MATLAB interface via the MATLAB function `decaes.m`.

## Installation

In Julia v1.3 or later you can install DECAES.jl from the Pkg REPL:
```
pkg> add DECAES
```

## Command Line Interface

This toolbox provides a command line interface (CLI) for processing from the terminal.
The CLI takes as input `.nii`, `.nii.gz`, or `.mat` files and performs one or both of T2-distribution computation and T2-parts analysis, the latter of which performs post-processing of the T2-distribution to calculate parameters such as the MWF or LWF.

* **Note:** if your data is in DICOM or PAR/REC format, the [freely available `dcm2niix` tool](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage) is able to convert both [DICOM](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage#General_Usage) and [PAR/REC](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage#Philips_PAR.2FREC_Images) files into NIfTI format

There are two equivalent ways to use the CLI, assuming DECAES.jl is already installed:

**1. Helper script:** Create a script called e.g. `decaes.jl` with the following contents:

```julia
using DECAES # load the package
main() # call CLI entrypoint function
```

This script can then be invoked from the command line as follows:

```bash
$ export JULIA_NUM_THREADS=4 # set JULIA_NUM_THREADS > 1 to enable parallel processing
$ julia decaes.jl <COMMAND LINE ARGS>
```

**2. Julia `-e` flag:** The contents of the above script can be written directly at the command line using the `-e` (for "evaluate") flag:

```bash
$ export JULIA_NUM_THREADS=4
$ julia -e 'using DECAES; main()' -- <COMMAND LINE ARGS>
```

## Documentation

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jondeuce.github.io/DECAES.jl/dev)

Find package documentation at the above link, which includes:
* The command line interface [API](https://jondeuce.github.io/DECAES.jl/dev/cli), available [command line arguments](https://jondeuce.github.io/DECAES.jl/dev/cli/#Arguments-1), and [examples](https://jondeuce.github.io/DECAES.jl/dev/cli/#Examples-1)
* API reference detailing how to use DECAES.jl from within Julia
* Other internals and algorithmic details

## Examples repository

See the [examples repository](https://github.com/jondeuce/mwiexamples) for a walk-through guide for using the CLI, including example CPMG data for performing MWI, as well as a script for calling the CLI from MATLAB.

# Benchmarks

Comparison of processing time on various data sets.
The MATLAB implementation uses the scripts contained within the `MWI_NNLS_toolbox_0319` folder in the [ubcmwf github repository](https://github.com/ubcmri/ubcmwf).
The Julia implementation uses DECAES.jl.

**Processor:** 3.60GHz Intel Core i7-7700 with 4 CPU cores/8 threads:

<center>

| Toolbox                | Parallelism        | Image Size             | T2-Distribution | Speedup   | Total Time   | Speedup   |
| :---:                  | :---:              | :---:                  | :---:           | :---:     | :---:        | :---:     |
| MWI_NNLS_toolbox_0319  | 4 workers + parfor | 240x240x48x48 + mask   | 01h:29m:35s     | -         | 01h:30m:37s  | -         |
| DECAES.jl              | 4 threads          | 240x240x48x48 + mask   | 00h:01m:35s     | **57X**   | 00h:02m:34s  | **35X**   |
| DECAES.jl              | 8 threads          | 240x240x48x48 + mask   | 00h:01m:24s     | **64X**   | 00h:02m:14s  | **41X**   |
|                        |                    |                        |                 |           |              |           |
| MWI_NNLS_toolbox_0319  | 4 workers + parfor | 240x240x113x56 + mask  | 02h:25m:19s     | -         | 02h:27m:52s  | -         |
| DECAES.jl              | 4 threads          | 240x240x113x56 + mask  | 00h:02m:34s     | **57X**   | 00h:04m:00s  | **37X**   |
| DECAES.jl              | 8 threads          | 240x240x113x56 + mask  | 00h:02m:20s     | **62X**   | 00h:03m:38s  | **41X**   |
|                        |                    |                        |                 |           |              |           |
| MWI_NNLS_toolbox_0319  | 4 workers + parfor | 240x240x48x48          | 02h:53m:13s     | -         | 02h:54m:24s  | -         |
| DECAES.jl              | 8 threads          | 240x240x48x48          | 00h:03m:35s     | **48X**   | 00h:04m:11s  | **42X**   |
|                        |                    |                        |                 |           |              |           |
| MWI_NNLS_toolbox_0319  | 4 workers + parfor | 240x240x113x56         | 09h:35m:17s     | -         | 09h:39m:33s  | -         |
| DECAES.jl              | 8 threads          | 240x240x113x56         | 00h:11m:49s     | **49X**   | 00h:12m:36s  | **46X**   |

</center>

<!--
Benchmarks for small datasets from the [examples repository](https://github.com/jondeuce/mwiexamples)
| Toolbox                | Parallelism        | Image Size             | T2-Distribution | Speedup   | Total Time   | Speedup   |
| :---:                  | :---:              | :---:                  | :---:           | :---:     | :---:        | :---:     |
| MWI_NNLS_toolbox_0319  | 4 workers + parfor | 175x140x1x56           | 00h:03m:03s     |    -      | 00h:03m:04s  |    -      |
| DECAES.jl              | 1 thread           | 175x140x1x56           | 00h:00m:13s     | **14X**   | 00h:00m:30s  | **6.1X**  |
| DECAES.jl              | 4 threads          | 175x140x1x56           | 00h:00m:07s     | **26X**   | 00h:00m:26s  | **7.1X**  |
| DECAES.jl              | 8 threads          | 175x140x1x56           | 00h:00m:05s     | **37X**   | 00h:00m:23s  | **8.0X**  |
|                        |                    |                        |                 |           |              |           |
| MWI_NNLS_toolbox_0319  | 4 workers + parfor | 175x140x8x56           | 00h:19m:56s     | -         | 00h:20m:02s  | -         |
| DECAES.jl              | 1 thread           | 175x140x8x56           | 00h:01m:28s     | **14X**   | 00h:01m:46s  | **11X**   |
| DECAES.jl              | 4 threads          | 175x140x8x56           | 00h:00m:26s     | **46X**   | 00h:00m:46s  | **26X**   |
| DECAES.jl              | 8 threads          | 175x140x8x56           | 00h:00m:23s     | **52X**   | 00h:00m:43s  | **28X**   |
-->

**Processor:** 2.10GHz Intel Xeon Gold 6130 with 16 CPU cores/32 threads:

<center>

| Toolbox                | Parallelism        | Image Size             | T2-Distribution | Total Time  |
| :---:                  | :---:              | :---:                  | :---:           | :---:       |
| DECAES.jl              | 4 threads          | 240x240x48x48 + mask   | 04m:01s         | 04m:14s     |
| DECAES.jl              | 8 threads          | 240x240x48x48 + mask   | 02m:13s         | 02m:28s     |
| DECAES.jl              | 16 threads         | 240x240x48x48 + mask   | 01m:11s         | 01m:24s     |
| DECAES.jl              | 32 threads         | 240x240x48x48 + mask   | 00m:57s         | 01m:09s     |
|                        |                    |                        |                 |             |
| DECAES.jl              | 4 threads          | 240x240x113x56 + mask  | 05m:56s         | 06m:25s     |
| DECAES.jl              | 8 threads          | 240x240x113x56 + mask  | 03m:45s         | 04m:21s     |
| DECAES.jl              | 16 threads         | 240x240x113x56 + mask  | 02m:09s         | 02m:37s     |
| DECAES.jl              | 32 threads         | 240x240x113x56 + mask  | 01m:40s         | 02m:09s     |

</center>

**Note:** images sizes which include "+ mask" used brain masks generated with the [BET tool](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide) (done automatically using the `--bet` flag for DECAES.jl, and manually for MATLAB) and only processed voxels within the brain mask.

The T2-Distribution column shows the time taken to complete the most costly step of the analysis pipeline, calling the function [`T2mapSEcorr`](https://jondeuce.github.io/DECAES.jl/dev/ref.html#DECAES.T2mapSEcorr).
This function performs the voxelwise nonnegative least-squares (NNLS) analysis to compute T2-distributions.
The Total Time column includes image loading and saving time, Julia startup and compilation time, BET brain mask generation, and the post-processing step of calling the [`T2partSEcorr`](https://jondeuce.github.io/DECAES.jl/dev/ref.html#DECAES.T2partSEcorr).
Note that MATLAB startup time is not included in the Total Time.

Notes regarding parallelism:
* MATLAB parallelism is implemented via `parfor` loops, executing the independent voxelwise T2-distribution computations in parallel.
MATLAB `parfor` loops are rather restrictive, as each loop iteration is executed on separate MATLAB processes.
Each loop iteration must be completed independent from each other, which among other restrictions, means memory cannot be shared between loop iterations.
* Julia parallelism is implemented via the more flexible [task-based multi-threading](https://julialang.org/blog/2019/07/multithreading) model of parallelism.
Communication between threads is possible, and memory can be easily shared and reused among threads.
This allows one to perform memory allocation up front: thread-local memory buffers, containing e.g. pre-allocated matrices for intermediate calculations, can be created outside of the parallel loop and efficiently re-used.
* Julia multithreading makes use of hyperthreads by default.
It is possible to configure MATLAB to use hyperthreads in `parfor` loops, though it is [not generally beneficial](https://www.mathworks.com/matlabcentral/answers/80129-definitive-answer-for-hyperthreading-and-the-parallel-computing-toolbox-pct) and indeed we found a ~20% slowdown.
