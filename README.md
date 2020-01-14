# DEcomposition and Component Analysis of Exponential Signals (DECAES)

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jondeuce.github.io/DECAES.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jondeuce.github.io/DECAES.jl/dev)
[![Build Status](https://travis-ci.com/jondeuce/DECAES.jl.svg?branch=master)](https://travis-ci.com/jondeuce/DECAES.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jondeuce/DECAES.jl?svg=true)](https://ci.appveyor.com/project/jondeuce/DECAES-jl)
[![Codecov](https://codecov.io/gh/jondeuce/DECAES.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jondeuce/DECAES.jl)
[![Coveralls](https://coveralls.io/repos/github/jondeuce/DECAES.jl/badge.svg?branch=master)](https://coveralls.io/github/jondeuce/DECAES.jl?branch=master)
[![Build Status](https://api.cirrus-ci.com/github/jondeuce/DECAES.jl.svg)](https://cirrus-ci.com/github/jondeuce/DECAES.jl)

DECAES.jl is a *fast* Julia implementation of the [MATLAB toolbox](https://mriresearch.med.ubc.ca/news-projects/myelin-water-fraction/) from the [UBC MRI Research Centre](https://mriresearch.med.ubc.ca/) for computing voxelwise [T2-distributions](https://doi.org/10.1016/0022-2364(89)90011-5) of multi spin-echo MRI images using the extended phase graph algorithm with stimulated echo corrections.
Post-processing of these T2-distributions allows for the computation of measures such as the [myelin water fraction (MWF)](https://doi.org/10.1002/mrm.1910310614) or the [luminal water fraction (LWF)](https://doi.org/10.1148/radiol.2017161687).

DECAES.jl is written in the open-source [Julia programming language](https://julialang.org/).
Julia and command line interfaces are available through this package.
The [examples repository](https://github.com/jondeuce/mwiexamples) additionally provides a MATLAB interface via the MATLAB function `decaes.m`.

## Installation

In Julia v1.3 you can install DECAES.jl from the Pkg REPL:
```
pkg> add https://github.com/jondeuce/DECAES.jl.git
```
which will track the `master` branch of the package.

## Command Line Interface

This toolbox provides a command line interface (CLI) for processing from the terminal.
The CLI takes as input `.nii`, `.nii.gz`, or `.mat` files and performs one or both of T2-distribution computation and T2-parts analysis, the latter of which performs post-processing of the T2-distribution to calculate parameters such as the MWF or LWF.

There are two equivalent ways use the CLI, assuming DECAES.jl is already installed:

**1. Helper script:** Create a script called e.g. `decaes.jl` with the following contents:

```julia
using DECAES # load the package
main() # call CLI entrypoint function
```

This script can then be invoked from the command line as follows:

```bash
$ export JULIA_NUM_THREADS=4
$ julia decaes.jl <COMMAND LINE ARGS>
```

**2. Julia `-e` flag:** The contents of the above script can be written directly at the command line using the `-e` flag (`-e` for "evaluate"):

```bash
$ export JULIA_NUM_THREADS=4
$ julia -e 'using DECAES; main()' <COMMAND LINE ARGS>
```

## Documentation

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jondeuce.github.io/DECAES.jl/dev)

Find package documentation at the above link, which includes:
* The command line interface [API](https://jondeuce.github.io/DECAES.jl/dev/cli), available [command line arguments](https://jondeuce.github.io/DECAES.jl/dev/cli/#Arguments-1), and [examples](https://jondeuce.github.io/DECAES.jl/dev/cli/#Examples-1)
* API reference detailing how to use DECAES.jl from within Julia
* Other internals and algorithmic details

## Examples repository

See the [examples repository](https://github.com/jondeuce/mwiexamples) for a walk-through guide for using the CLI, including example MWI data, as well as a script for calling the CLI from MATLAB.

# Benchmarks

Comparison of processing time on various data sets.
The MATLAB implementation uses the scripts contained within the `MWI_NNLS_toolbox_0319` folder in the [ubcmwf github repository](https://github.com/ubcmri/ubcmwf).
The Julia implementation uses DECAES.jl.

**Processor:** 3.60GHz Intel Core i7-7700 with 4 CPU cores/8 threads:

<center>

| Toolbox                | Parallelism        | Image Size             | T2-Distribution | Speedup   | Total Time   | Speedup   |
| :---:                  | :---:              | :---:                  | :---:           | :---:     | :---:        | :---:     |
| MWI_NNLS_toolbox_0319  | 4 workers + parfor | 175x140x1x56           | 00h:03m:03s     |    -      | 00h:03m:04s  |    -      |
| DECAES.jl              | 1 thread           | 175x140x1x56           | 00h:00m:14s     | **13X**   | 00h:00m:27s  | **6.8X**  |
| DECAES.jl              | 4 threads          | 175x140x1x56           | 00h:00m:04s     | **46X**   | 00h:00m:19s  | **9.7X**  |
| DECAES.jl              | 8 threads          | 175x140x1x56           | 00h:00m:03s     | **61X**   | 00h:00m:16s  | **12X**   |
|                        |                    |                        |                 |           |              |           |
| MWI_NNLS_toolbox_0319  | 4 workers + parfor | 175x140x8x56           | 00h:19m:56s     | -         | 00h:20m:02s  | -         |
| DECAES.jl              | 1 thread           | 175x140x8x56           | 00h:01m:54s     | **10X**   | 00h:02m:11s  | **9.2X**  |
| DECAES.jl              | 4 threads          | 175x140x8x56           | 00h:00m:42s     | **28X**   | 00h:00m:59s  | **20X**   |
| DECAES.jl              | 8 threads          | 175x140x8x56           | 00h:00m:26s     | **46X**   | 00h:00m:41s  | **29X**   |
|                        |                    |                        |                 |           |              |           |
| MWI_NNLS_toolbox_0319  | 4 workers + parfor | 240x240x48x48          | 02h:53m:13s     | -         | 02h:54m:24s  | -         |
| DECAES.jl              | 8 threads          | 240x240x48x48          | 00h:04m:25s     | **39X**   | 00h:05m:03s  | **35X**   |
|                        |                    |                        |                 |           |              |           |
| MWI_NNLS_toolbox_0319  | 4 workers + parfor | 240x240x48x48 + mask   | 01h:29m:35s     | -         | 01h:30m:37s  | -         |
| DECAES.jl              | 4 threads          | 240x240x48x48 + mask   | 00h:02m:11s     | **41X**   | 00h:02m:46s  | **33X**   |
| DECAES.jl              | 8 threads          | 240x240x48x48 + mask   | 00h:01m:47s     | **50X**   | 00h:02m:15s  | **40X**   |
|                        |                    |                        |                 |           |              |           |
| MWI_NNLS_toolbox_0319  | 4 workers + parfor | 240x240x113x56         | 09h:35m:17s     | -         | 09h:39m:33s  | -         |
| DECAES.jl              | 8 threads          | 240x240x113x56         | 00h:14m:36s     | **39X**   | 00h:16m:40s  | **35X**   |
|                        |                    |                        |                 |           |              |           |
| MWI_NNLS_toolbox_0319  | 4 workers + parfor | 240x240x113x56 + mask  | 02h:25m:19s     | -         | 02h:27m:52s  | -         |
| DECAES.jl              | 4 threads          | 240x240x113x56 + mask  | 00h:04m:15s     | **30X**   | 00h:05m:07s  | **29X**   |
| DECAES.jl              | 8 threads          | 240x240x113x56 + mask  | 00h:02m:59s     | **49X**   | 00h:03m:49s  | **39X**   |

</center>

**Processor:** 2.10GHz Intel Xeon Gold 6130 with 16 CPU cores/32 threads:

<center>

| Toolbox                | Parallelism        | Image Size             | T2-Distribution | Speedup   | Total Time      | Speedup   |
| :---:                  | :---:              | :---:                  | :---:           | :---:     | :---:           | :---:     |
| DECAES.jl              | 1 threads          | 240x240x48x48 + mask   | 00h:13m:13s     | -         | 00h:13m:41s     | -         |
| DECAES.jl              | 16 threads         | 240x240x48x48 + mask   | 00h:01m:30s     | 8.8X      | 00h:01m:58s     | 7.0X      |
| DECAES.jl              | 28 threads         | 240x240x48x48 + mask   | **00h:01m:14s** | **11X**   | **00h:01m:41s** | **8.1X**  |
| DECAES.jl              | 32 threads         | 240x240x48x48 + mask   | 00h:01m:21s     | 9.8X      | 00h:01m:49s     | 7.5X      |
|                        |                    |                        |                 |           |                 |           |
| DECAES.jl              | 1 threads          | 240x240x113x56 + mask  | 00h:22m:57s     | -         | 00h:23m:46s     | -         |
| DECAES.jl              | 16 threads         | 240x240x113x56 + mask  | 00h:02m:47s     | 8.2X      | 00h:03m:37s     | 6.6X      |
| DECAES.jl              | 28 threads         | 240x240x113x56 + mask  | **00h:02m:14s** | **10X**   | **00h:02m:58s** | **8.0X**  |
| DECAES.jl              | 32 threads         | 240x240x113x56 + mask  | 00h:02m:20s     | 9.8X      | 00h:03m:06s     | 7.7X      |

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
