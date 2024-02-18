# Command Line Interface

DECAES provides a command line interface (CLI) for calling the main analysis functions: [`T2mapSEcorr`](@ref) for computing $T_2$-distributions, and [`T2partSEcorr`](@ref) for running $T_2$-parts analysis on the resulting $T_2$-distributions for computing measures such as the myelin water fraction.

## Using the CLI

Assuming you have [DECAES installed](@ref installation), there are two equivalent ways to use the CLI:

**1. Helper script:** Create a simple Julia script which calls the entrypoint function [`main`](@ref) provided by this package. For example, save the following code in a Julia script called `decaes.jl` (or, download the script located [here](https://github.com/jondeuce/DECAES.jl/blob/master/api/decaes.jl)):

```julia
using DECAES # load the package
main() # call command line interface
```

An image file `image.nii` can be passed to DECAES by running this script with `julia` from the command line:

```bash
$ julia decaes.jl -- image.nii <COMMAND LINE ARGS>
```

**2. Julia `-e` flag:** The contents of the above script can equivalently be passed directly to `julia` using the `-e` (for "evaluate") flag:

```bash
$ julia -e 'using DECAES; main()' -- image.nii <COMMAND LINE ARGS>
```

Either way of calling the CLI forwards the arguments `<COMMAND LINE ARGS>` to the entrypoint function [`main`](@ref).
Available command line arguments are detailed in the [Arguments](@ref) section.

For the remainder of this section, we will make use of the `decaes.jl` script from option 1.

## File types

Input files must be one of the following file types:

1. [NIfTI file](https://nifti.nimh.nih.gov/) with extension `.nii`, or [gzip](https://www.gzip.org/) compressed NIfTI file with extension `.nii.gz`. See [NIfTI.jl](https://github.com/JuliaIO/NIfTI.jl) for more information.
2. [MATLAB file](https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html) with extension `.mat`. **Note:** `.mat` files saved in the oldest format `v4` are not supported, but all newer formats (`v6`, `v7`, and `v7.3`) are supported. See [MAT.jl](https://github.com/JuliaIO/MAT.jl) for more information.
3. Philips [PAR/REC](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage#Philips_PAR.2FREC_Images) file pair with extensions `.par` and `.rec` (or `.PAR` and `.REC`).
4. Philips XML/REC file pair with extensions `.xml` and `.rec` (or `.XML` and `.REC`).

All output files are saved as `.mat` files in format `v7.3`.

!!! note
    If your data is in DICOM format, the [freely available `dcm2niix` tool](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage) is able to convert [DICOM](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage#General_Usage) files into NIfTI format

!!! note
    * Input files are interpreted as 4D arrays (or 3D arrays for mask files) when loaded; ensure that the underlying data is stored with the first three dimensions as (row, column, slice), and the last dimension as echo (or $T_2$ bin, or omitted for mask files)
    * Images read from NIfTI and PAR/XML/REC files are coerced into the appropriate dimension; errors or unexpected behaviour may occur if the data is not stored with the correct dimensions
    * MATLAB files are searched for arrays with the appropriate dimension; the first such array that is found is used, otherwise an error will occur. **Multiple 3D/4D arrays should not be stored in the same `.mat` file)**

## Arguments

Available command line arguments are broken into four categories:

1. **Positional arguments:** these are the input files. Input files are typically placed at the beginning of `<COMMAND LINE ARGS>`.
2. **Optional arguments:** settings governing the analysis pipeline. See below for details.
3. **[`T2mapSEcorr`](@ref)/[`T2partSEcorr`](@ref) arguments:** settings for computing the $T_2$-distribution and subsequent $T_2$-parts analysis. Required arguments are listed first; see below for the full parameter list, and see [`T2mapSEcorr`](@ref) and [`T2partSEcorr`](@ref) for parameter descriptions. Note: if no default is shown, the parameter is unused by default.
4. **B1 correction and stimulated echo correction:** settings for controlling B1 correction and stimulated echo correction. These settings typically need not be modified. However, the following options are noteworthy: `--SetFlipAngle` allows for prescribing a fixed flip angle for all voxels instead of estimating the flip angle per-voxel; and `--B1map` allows for passing B1 maps (units of degrees) directly, replacing the flip angle estimation step.
5. **Additional save options:** optional additional output maps
6. **[BET](@ref bet) arguments:** settings for governing automatic brain mask generation using the BET brain extraction tool; [see below](@ref bet) for details.

```@example
using DECAES # hide
DECAES.ArgParse.show_help(DECAES.CLI_SETTINGS; exit_when_done = false) # hide
```

!!! note
    If desired, the $T_2$-distribution computation and the $T_2$-parts analysis may be performed separately:
    * When the `--T2map` flag is passed, or both `--T2map` and `--T2part` flags are passed, input image arrays should be 4D with data as (row, column, slice, echo)
    * When only the `--T2part` flag is passed, input image arrays should be 4D with data as (row, column, slice, $T_2$ bin)

## Outputs

DECAES will produce up to five output files, each with the input filename (without suffix) used as a prefix.
For example, if the input file is called `image.nii`, the possible output files are:

1. `image.t2dist.mat`: MATLAB file containing the 4D array of $T_2$-distributions (produced when `--T2map` flag is passed)
2. `image.t2maps.mat`: MATLAB file containing $T_2$-distribution property maps and NNLS fit parameters; see [`T2mapSEcorr`](@ref) (produced when `--T2map` flag is passed)
3. `image.t2parts.mat`: MATLAB file containing $T_2$-parts analysis results such as the MWF; see [`T2partSEcorr`](@ref) (produced when `--T2part` flag is passed)
4. `image.log`: Log file containing the console output
5. `image.settings.txt`: Copy of settings file which was named `settings.txt` (produced when [passing settings as a file](@ref settingsfiles))

If the `--dry` flag is passed, none of the above files will be produced.

## Multithreading

Multithreaded parallel processing can be enabled by passing the `julia` command line flag `--threads=auto`:

```@example
println("\$ julia --threads=auto decaes.jl -- image.nii <COMMAND LINE ARGS>") # hide
```

This is highly recommended to speed up computation time, but is not strictly required.

## Examples

### [Default options](@id defaultoptions)

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
callmain(imfile, "--T2map", "--T2part", "--dry", "--quiet", "--TE", "10e-3", "--nT2", "40", "--T2Range", "10e-3", "2.0", "--SPWin", "10e-3", "40e-3", "--MPWin", "40e-3", "200.0e-3", "--Reg", "lcurve") # precompile
```

Suppose you have a multi spin-echo image file `image.nii` which you would like to perform $T_2$ analysis on.
We can call [`T2mapSEcorr`](@ref) and [`T2partSEcorr`](@ref) on the file `image.nii` using `decaes.jl`.
We pass the required arguments with the appropriate flags and leave the remaining parameters at default values:

```@example
println("\$ julia --threads=auto decaes.jl -- image.nii --T2map --T2part --TE 10e-3 --nT2 40 --T2Range 10e-3 2.0 --SPWin 10e-3 40e-3 --MPWin 40e-3 200.0e-3 --Reg lcurve") # hide
```

After a few seconds, the script should begin running with the following messages appearing as the script progresses (note that real images will take longer to process than this toy example):

```@example callmain
callmain(imfile, "--T2map", "--T2part", "--TE", "10e-3", "--nT2", "40", "--T2Range", "10e-3", "2.0", "--SPWin", "10e-3", "40e-3", "--MPWin", "40e-3", "200.0e-3", "--Reg", "lcurve") # hide
```

### [Settings files](@id settingsfiles)

As there are several required parameters, it is convenient to store settings for DECAES in a settings file.
Using the same options from the [previous section](@ref defaultoptions), we create a file `settings.txt` with the following contents:

```
/path/to/image.nii
--T2map
--T2part
--TE
10e-3
--nT2
40
--T2Range
10e-3
2.0
--SPWin
10e-3
40e-3
--MPWin
40e-3
200.0e-3
--Reg
lcurve
```

If this file is located at `/path/to/settings.txt`, simply prefix the filepath with the `@` character to have the file contents read into the [`main`](@ref) function:

```@example
println("\$ julia --threads=auto decaes.jl -- @/path/to/settings.txt") # hide
```

!!! note
    * The use of settings files is highly recommended for both reproducibility and for self-documentation. The input settings file will be automatically copied into the output folder for each processed image, with the image filename prepended. In this case, for example, the copied settings file would be called `image.settings.txt`
    * Only one flag or value is allowed per line within a settings file. Flags which require multiple inputs (e.g. `--T2Range` above) must use one line for each input
    * The extension of the settings file is ignored; `.txt` is arbitrary in this example
    * Though not strictly necessary, using full input- and output paths is recommended. This way, one doesn't rely on relative paths and can e.g. call `julia /path/to/decaes.jl @/path/to/settings.txt` from any directory

### [Default settings files](@id nondefault)

Suppose you have a default settings file `default.txt`, similar to the `settings.txt` file in the [above section](@ref settingsfiles).
Settings in `default.txt` can be individually overridden.
For example, if we are interested in changing the number of $T_2$ bins `nT2` to 60, but leaving all other parameters the same, run the following:

```@example
println("\$ julia --threads=auto decaes.jl -- @/path/to/default.txt --nT2 60") # hide
```

### [Multiple input files](@id multiinput)

Multiple input files (possibly of different file types) can be passed in the obvious way:

```@example
println("\$ julia --threads=auto decaes.jl -- image1.nii image2.mat image3.nii.gz image4.par <COMMAND LINE ARGS>") # hide
```

Equivalently, place multiple image paths at the top of your settings file, with each path on a new line.

### [Specify output folder](@id outfolder)

By default, output files are saved in the same location as the corresponding input file.
If you'd like to save them in a different folder, you can use the `-o` or `--output` flag:

```@example
println("\$ julia --threads=auto decaes.jl -- image.nii --output /path/to/output/folder/ <COMMAND LINE ARGS>") # hide
```

The requested output folder will be created if it does not already exist.

Equivalently, add `--output` and `/path/to/output/folder/` as consecutive lines in your settings file.

### [Passing image masks](@id passmasks)

Image masks can be passed into DECAES using the `-m` or `--mask` flag:

```@example
println("\$ julia --threads=auto decaes.jl -- image.nii --mask /path/to/mask.nii <COMMAND LINE ARGS>") # hide
```

The mask file is loaded and applied to the input image via elementwise multiplication over the spatial dimensions, e.g. the mask is applied to each echo of a 4D multi-echo input image.

If multiple image files are passed, multiple corresponding mask files can be passed, too:

```@example
println("\$ julia --threads=auto decaes.jl -- image1.nii image2.mat --mask /path/to/mask1.mat /path/to/mask2.nii.gz <COMMAND LINE ARGS>") # hide
```

Equivalently, add `--mask`, `/path/to/mask1.mat`, `/path/to/mask2.mat`, ...  as consecutive lines in your settings file.

!!! note
    If input images have been manually masked such that they are zero outside regions of interest, a mask need not be passed.
    The `--Threshold` parameter of [`T2mapSEcorr`](@ref) controls a first echo intensity cutoff threshold (default value 0.0), below which voxels are automatically skipped during processing

### [Automatic brain masking with BET](@id bet)

The [BET brain extraction tool](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide) from the [FSL library of analysis tools](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL) can be used to automatically generate a brain mask prior to analysis.
Only voxels within the generated brain mask will be processed, greatly reducing analysis time.
To use BET, pass the `--bet` flag:

```@example
println("\$ julia --threads=auto decaes.jl -- image.nii --bet <COMMAND LINE ARGS>") # hide
```

If `bet` is not on your system path, you can pass the path to the `bet` binary with the `--betpath` flag.
Additionally, you can pass arguments to `bet` with the `--betargs` flag:

```@example
println("\$ julia --threads=auto decaes.jl -- image.nii --bet --betpath /path/to/bet --betargs -m,-n <COMMAND LINE ARGS>") # hide
```

Note that `bet` arguments must be passed as a single string to `--betargs`, separated by commas or spaces, as shown above.

Equivalently, add `--bet`, `--betpath`, `/path/to/bet`, `--betargs`, `-m,-n` as consecutive lines in your settings file.

!!! note
    If a mask file is passed using the `--mask` flag, the `--bet` flag will be ignored and the mask file will be used

## Legacy options

During the port from MATLAB to Julia, some algorithms were improved using computationally more efficient algorithms, and some default parameters were modified or removed as well.
This may cause small differences in output parameter maps.
As an example, the flip angle optimization procedure requires finding the root of a cubic spline.
In MATLAB this was performed by evaluating the spline on a very fine mesh and choosing the value nearest zero.
During profiling it was found that this was a time consuming operation, and therefore in Julia this was replaced by an efficient rootfinding method tailored for cubic splines.

The differences due to algorithmic changes like the one above are quite small.
For example, most tests in the DECAES test suite will pass when using a relative tolerance of ``10^{-3}``, and almost all tests pass with a relative tolerance of ``10^{-2}``.
That is to say that nearly all outputs are identical to 3 or more significant digits, which includes $T_2$-distributions, MWF maps, etc.
It should be emphasized, though, that as these differences arise from improved algorithms, any discrepencies are merely small improvements.

The `--legacy` flag is available if *exact* reproducibility is required compared to the MATLAB version.
This will ensure that all outputs match to nearly machine precision (a relative tolerance of ``10^{-10}`` is used during testing).
Note however that the `--legacy` flag may cause a significant slowdown in processing time due to less efficient algorithms being used internally, and is therefore not recommended unless absolutely necessary.
Differences due to changes in default parameters can always be overridden by passing in the desired value explicitly without the need for the `--legacy` flag.
