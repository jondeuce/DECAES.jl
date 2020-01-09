# Command Line Interface

DECAES.jl provides a command line interface (CLI) for calling the main analysis functions [`T2mapSEcorr`](@ref) and [`T2partSEcorr`](@ref).

## Using the CLI

Assuming you have [DECAES.jl installed](@ref installation), there are two equivalent ways to use the CLI:

**1. Helper script:** Create a simple Julia script which calls the entrypoint function [`main`](@ref) provided by this package. For example, save the following code in a Julia script called `decaes.jl`:

```julia
using DECAES # load the package
main() # call command line interface
```

Run this script from the command line using `julia` as follows:

```bash
$ julia decaes.jl image.nii <COMMAND LINE ARGS>
```

**2. Julia `-e` flag:** The contents of the above script can equivalently be passed directly to Julia using the `-e` flag (`-e` for "evaluate"):

```bash
$ julia -e 'using DECAES; main()' image.nii <COMMAND LINE ARGS>
```

For the remainder of this section, we will make use of the `decaes.jl` script from option 1.

## Multithreading

Multithreaded parallel processing can be enabled by setting the `JULIA_NUM_THREADS` environment variable as follows:

```@example
println("\$ export JULIA_NUM_THREADS=$(Threads.nthreads())") # hide
println("\$ julia decaes.jl image.nii <COMMAND LINE ARGS>") # hide
```

This is highly recommended to speed up computation time, but is not strictly required.

!!! note
    From the [Julia documentation](https://docs.julialang.org/en/v1/manual/parallel-computing/#Setup-1):
    > [The keyword `export`] works on bourne shells on Linux and OSX.
    > Note that if you're using a C shell on these platforms, you should use the keyword `set` instead of `export`.
    > If you're on Windows, start up the command line in the location of `julia.exe` and use `set` instead of `export`.

## File types

The input image (`image.nii` above) must be one of two file types:
1. [NIfTI file](https://nifti.nimh.nih.gov/) with extension `.nii`, or [gzip](https://www.gzip.org/) compressed NIfTI file with extension `.nii.gz`. See [NIfTI.jl](https://github.com/JuliaIO/NIfTI.jl) for more information
2. [MATLAB file](https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html) with extension `.mat`. **Note:** `.mat` files saved in the oldest format `v4` are not supported, but all newer formats (`v6`, `v7`, and `v7.3`) are supported. See [MAT.jl](https://github.com/JuliaIO/MAT.jl) for more information

Output files are saved as `.mat` files, written in format `v7.3`. 

The arguments `<COMMAND LINE ARGS>` are parsed by the entrypoint function [`main`](@ref); available options are detailed in the [Arguments](@ref) section.

## Arguments

Available command line arguments are broken into three categories:

1. **Positional arguments:** these are the input files and/or folders. Input files/folders are typically placed at the beginning of `<COMMAND LINE ARGS>`.
2. **Optional arguments:** settings governing the analysis pipeline. See below for details.
3. **[`T2mapSEcorr`](@ref)/[`T2partSEcorr`](@ref) arguments:** settings for computing the $T_2$-distribution and subsequent $T_2$-parts analysis. See below for the full parameter list; see [`T2mapSEcorr`](@ref) and [`T2partSEcorr`](@ref) for parameter descriptions. Note: if no default is shown, the parameter is unused by default.

```@example
using DECAES # hide
DECAES.ArgParse.show_help(DECAES.ARGPARSE_SETTINGS; exit_when_done = false) # hide
```

!!! note
    If desired, the $T_2$-distribution computation and the $T_2$-parts analysis may be performed separately:
    * When the `--T2map` flag is passed, or both `--T2map` and `--T2part` flags are passed, input arrays should be 4D with data as (row, column, slice, echo)
    * When only the `--T2part` flag is passed, input arrays should be 4D with data as (row, column, slice, $T_2$ bin)

## Examples

### Default options

```@setup callmain
using DECAES
const imfile = "image.nii.gz"
function callmain(args...)
    image = DECAES.mock_image(MatrixSize = (100,100,1))
    cd(tempdir()) do
        try
            DECAES.NIfTI.niwrite(imfile, DECAES.NIfTI.NIVolume(image))
            main(String[args...])
        finally
            isfile(imfile) && rm(imfile)
        end
    end
    nothing
end
```

Suppose you have a multi spin-echo image file `image.nii` which you would like to perform $T_2$ analysis on.
We can call [`T2mapSEcorr`](@ref) and [`T2partSEcorr`](@ref) on the file `image.nii`, with default options, using `decaes.jl` as follows:

```@example
println("\$ export JULIA_NUM_THREADS=$(Threads.nthreads())") # hide
println("\$ julia decaes.jl image.nii --T2map --T2part") # hide
```

After a few seconds, the script should begin running with the following messages appearing as the script progresses (note that real images will take longer to process than this toy example):

```@example callmain
callmain(imfile, "--T2map", "--T2part") # hide
```

The script will produce four files, each with the input filename (without suffix) used as a prefix:

1. `image.t2dist.mat`: MATLAB file containing the $T_2$-distributions
2. `image.t2maps.mat`: MATLAB file containing $T_2$-distribution property maps and NNLS fit parameters; see [`T2mapSEcorr`](@ref)
3. `image.t2parts.mat`: MATLAB file containing $T_2$-parts analysis results such as the MWF; see [`T2partSEcorr`](@ref)
4. `image.log`: Log file containing the console output

### [Multiple input files](@id multiinput)

Multiple input files (possibly of different file types) can be passed in the obvious way:

```@example
println("\$ export JULIA_NUM_THREADS=$(Threads.nthreads())") # hide
println("\$ julia decaes.jl image1.nii image2.mat image3.nii.gz --T2map --T2part") # hide
```

Input folders can be automatically read, too:

```@example
println("\$ export JULIA_NUM_THREADS=$(Threads.nthreads())") # hide
println("\$ julia decaes.jl /path/to/input/folder/ --T2map --T2part") # hide
```

Input files and folders can me mixed and matched:

```@example
println("\$ export JULIA_NUM_THREADS=$(Threads.nthreads())") # hide
println("\$ julia decaes.jl image1.nii /path/to/input/folder/ image2.mat --T2map --T2part") # hide
```

!!! note
    The rules for finding valid files within folders are deliberately simplistic:

    1. If a `.nii` or `.nii.gz` file is found, it is loaded and is assumed to be 4D. An error will occur of the image is not 4D.
    2. If a `.mat` file is found, it is loaded and only the first 4D array which is found within the loaded file is used; multiple 4D arrays should not be stored in the same `.mat` file. An error will occur if no 4D image is found.

### [Specify output folder](@id outfolder)

By default, output files are saved in the same location as the corresponding input file.
If you'd like to save them in a different folder, you can use the `-o` or `--output` flag:

```@example
println("\$ export JULIA_NUM_THREADS=$(Threads.nthreads())") # hide
println("\$ julia decaes.jl image.nii --T2map --T2part --ouput /path/to/output/folder/") # hide
```

The requested output folder will be created if it does not already exist.

### [Non-default parameters](@id nondefault)

Parameter values can be set to non-default values at the command line.
For example, we can set the echo time `TE` to 8ms, the number of $T_2$ bins `nT2` to 60, and the $T_2$ distribution range to [10ms, 1.5s] as follows:

```@example
println("\$ export JULIA_NUM_THREADS=$(Threads.nthreads())") # hide
println("\$ julia decaes.jl image.nii --T2map --T2part --TE 0.008 --nT2 60 --T2Range 0.010 1.5") # hide
```

### Automatic brain masking with BET

The [BET brain extraction tool](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide) from the [FSL library of analysis tools](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL) can be used to automatically generate a brain mask prior to analysis.
Only voxels within the generated brain mask will be processed, greatly reducing analysis time.
To use BET, pass the `--bet` flag:

```@example
println("\$ export JULIA_NUM_THREADS=$(Threads.nthreads())") # hide
println("\$ julia decaes.jl image.nii --T2map --T2part --bet") # hide
```

If `bet` is not on your system path, you can pass the path to the `bet` binary with the `--betpath` flag.
Additionally, you can pass arguments to `bet` with the `--betargs` flag:

```@example
println("\$ export JULIA_NUM_THREADS=$(Threads.nthreads())") # hide
println("\$ julia decaes.jl image.nii --T2map --T2part --bet --betpath /path/to/bet --betargs '-m -n'") # hide
```

Note that `bet` arguments must be passed as a single string to `--betargs`, separated by spaces, as shown above.

### Settings files

Parameter values can be read in from a settings file.
For example, we can combine the examples from the [Specify output folder](@ref outfolder) and [Non-default parameters](@ref nondefault) sections by creating a `settings.txt` file with the following contents:

```
/path/to/image.nii
--output
/path/to/output/folder/
--T2map
--T2part
--TE
0.008
--nT2
60
--T2Range
0.010
1.5
```

If this file is located at `/path/to/settings.txt`, simply prefix the filepath with the `@` character to have the file contents read into the [`main`](@ref) function:

```@example
println("\$ export JULIA_NUM_THREADS=$(Threads.nthreads())") # hide
println("\$ julia decaes.jl @/path/to/settings.txt") # hide
```

!!! note
    * The use of settings files is highly recommended for both reproducibility and for self-documentation. The input settings file will be automatically copied into the output folder for each processed image, with the image filename prepended. In this case, for example, the copied settings file would be called `image.settings.txt`
    * Only one flag or value is allowed per line within a settings file. Flags which require multiple inputs (e.g. `--T2Range` above) must use one line for each input
    * The extension of the settings file is ignored; `.txt` is arbitrary in this example
    * Though not strictly necessary, using full input- and output paths is recommended. This way, one doesn't rely on relative paths and can e.g. call `julia /path/to/decaes.jl @/path/to/settings.txt` from any directory

## Legacy options

During the MATLAB port to Julia, some algorithms were replaced with mathematically identical but computationally more efficient algorithms which may cause small differences in output parameter maps, and some default options were changed.

For example, the flip angle optimization procedure requires finding the root of a cubic spline.
In MATLAB this was performed by evaluating the spline on a very fine mesh and choosing the value nearest zero.
During profiling it was found that this was a time consuming operation, and therefore in Julia this was replaced by an efficient rootfinding method.

The differences due to algorithmic changes like the one above are quite small.
For example, most tests will pass when using a relative tolerance of ``10^{-3}``, and almost all tests pass with a relative tolerance of ``10^{-2}``.
That is to say that nearly all outputs are identical to 3 or more significant digits, which includes T2-distributions, MWF maps, etc.

The `--legacy` flag is available if *exact* reproducibility is required compared to the MATLAB version.
This will ensure that all outputs match to nearly machine precision (a relative tolerance of ``10^{-10}`` is used during testing).
Note however that the `--legacy` flag may cause a significant slowdown in processing time due to less efficient internal algorithms being used, and is therefore not recommended unless absolutely necessary.
Differences due to changes in default parameters can always be overridden by passing in the desired value explicitly (e.g. `--SPWin 0.014 0.040`) without the need for the `--legacy` flag.

**Default options with the `--legacy` flag**

```@example
using DECAES # hide
DECAES.ArgParse.show_help(DECAES.ARGPARSE_SETTINGS_LEGACY; exit_when_done = false) # hide
```
