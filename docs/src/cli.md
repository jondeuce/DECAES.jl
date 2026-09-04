# Command Line Interface

The DECAES command-line interface (CLI) computes [$T_2$ distributions](@ref t2map) and derived quantities such as the [myelin water fraction](@ref t2part).

## Using the CLI

After [installing DECAES](@ref installation), use one of these equivalent commands:

**1. Recommended: `decaes` launcher**

```bash
decaes <COMMAND LINE ARGS>
```

!!! note
    Add `~/.julia/bin` to your `PATH` to avoid writing the full path `~/.julia/bin/decaes`.

**2. Julia `-e` flag**

```bash
julia --project=@decaes --threads=auto -e 'using DECAES; main()' -- <COMMAND LINE ARGS>
```

!!! note
    The flag `--threads=auto` enables parallel processing, which is critical for maximizing DECAES performance.

Both commands pass `<COMMAND LINE ARGS>` to [`main`](@ref).
The examples below use the `decaes` launcher.

## File types

Supported input formats are:

1. [NIfTI](https://nifti.nimh.nih.gov/): `.nii` or `.nii.gz`; see [NIfTI.jl](https://github.com/JuliaIO/NIfTI.jl)
2. [MATLAB](https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html): `.mat` in v6, v7, or v7.3 format; see [MAT.jl](https://github.com/JuliaIO/MAT.jl)
3. Philips [PAR/REC](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage#Philips_PAR.2FREC_Images): `.par` and `.rec` (or `.PAR` and `.REC`); see [ParXRec.jl](https://github.com/kamesy/ParXRec.jl)
4. Philips XML/REC: `.xml` and `.rec` (or `.XML` and `.REC`); see [ParXRec.jl](https://github.com/kamesy/ParXRec.jl)

Outputs are saved as MATLAB files in format `v7.3` by default.
Pass `--OutputFormat nii` to save image outputs as NIfTI files instead.

!!! note
    Convert DICOM data to NIfTI using [`dcm2niix`](https://github.com/rordenlab/dcm2niix).

!!! note
    * Image arrays must have dimensions `(row, column, slice, echo)` for multi-echo data and `(row, column, slice, T2 bin)` for $T_2$ distributions. Masks must have dimensions `(row, column, slice)`.
    * For MATLAB files, DECAES uses the first array with the required shape. Store only one candidate image array per file.

## Arguments

The CLI accepts the arguments below, grouped by purpose:

1. Input image paths and general options such as the output directory
2. Required and optional settings for $T_2$ mapping and $T_2$-parts analysis
3. B1 and stimulated-echo correction settings
4. Optional output maps
5. Automatic brain masking with [BET](@ref bet)

See also [`T2mapOptions`](@ref) and [`T2partOptions`](@ref).

```@example
using DECAES # hide
DECAES.ArgParse.show_help(DECAES.CLI_SETTINGS; exit_when_done = false) # hide
```

!!! note
    `--T2map` requires multi-echo data. `--T2part` is typically run after `--T2map`, but can be run independently if given a precomputed $T_2$ distribution.

## Outputs

Output filenames use the input filename as a prefix.
For an input named `image.nii`, DECAES may produce:

1. `image.t2dist.mat`: $T_2$ distributions from `--T2map`
2. `image.t2maps.mat`: $T_2$ distribution properties and fit parameters from `--T2map`; see [`T2mapSEcorr`](@ref)
3. `image.t2parts.mat`: derived quantities such as MWF from `--T2part`; see [`T2partSEcorr`](@ref)
4. `image.log`: console output
5. `image.settings.txt`: copy of an input [settings file](@ref settingsfiles)

If `--NoSaveT2Dist` is passed, the large $T_2$ distribution file is not saved;
it is not needed if one is only interested in derived image maps.

If the `--dry` flag is passed, none of the above files will be produced.

### [NIfTI outputs](@id nifti)

Passing `--OutputFormat nii` saves the $T_2$ distribution and derived maps as gzipped NIfTI files:

1. `image.t2dist.nii.gz`: $T_2$ distributino
2. `image.t2maps.meta.mat`: non-image metadata such as echo times, $T_2$ times, etc.
1. `image.t2maps.<name>.nii.gz`: image shaped outputs from [`T2mapSEcorr`](@ref)
2. `image.t2parts.<name>.nii.gz`: image shaed outputs from [`T2partSEcorr`](@ref)

NIfTI output files inherit the header of a NIfTI input image, preserving its voxel size and orientation.

## Examples

### [Default options](@id defaultoptions)

```@setup callmain
using DECAES
const imfile = "image.nii.gz"
function callmain(args...)
    image = DECAES.mock_image(; MatrixSize = (100, 100, 1))
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

Compute the $T_2$ distribution and $T_2$ parts for a multi-spin echo image using:

```@example
println("\$ decaes image.nii --T2map --T2part --TE 10e-3 --nT2 40 --T2Range 10e-3 2.0 --SPWin 10e-3 40e-3 --MPWin 40e-3 200.0e-3 --Reg lcurve") # hide
```

This command:

1. Reads the multi-echo data from `image.nii`
2. Uses `--T2map` to compute the $T_2$ distribution, then `--T2part` to compute quantities derived from it
3. Sets the echo spacing to `10e-3`, the number of $T_2$ bins to 40, and the $T_2$ range to `[10e-3, 2.0]`
4. Defines the short- and middle-$T_2$ peak windows with `--SPWin` and `--MPWin`
5. Selects L-curve regularization with `--Reg lcurve`

The time-valued arguments use seconds here, but any consistent unit may be used.
Acquisition-dependent values such as `TE`, `T2Range`, `SPWin`, and `MPWin` should be chosen for the data rather than copied blindly from this example.

Below is an example pipeline output on a small synthetic image:

```@example callmain
callmain(imfile, "--T2map", "--T2part", "--TE", "10e-3", "--nT2", "40", "--T2Range", "10e-3", "2.0", "--SPWin", "10e-3", "40e-3", "--MPWin", "40e-3", "200.0e-3", "--Reg", "lcurve") # hide
```

### [Settings files](@id settingsfiles)

Settings files make long commands easier to reuse and record.
They place one flag or value on each line;
an option with multiple values, such as `--T2Range`, therefore uses multiple lines.
For example, `/path/to/settings.txt` could contain:

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

Prefix the path with `@` to use it:

```@example
println("\$ decaes @/path/to/settings.txt") # hide
```

!!! note
    DECAES copies each settings file to the output directory for reproducibility. The extension is arbitrary. Using absolute input and output paths allows the settings file to be used from any working directory.

### [Default settings files](@id nondefault)

Arguments written after a settings file override values from that file.
This is useful when most settings remain fixed across analyses.
For example, the following command uses `default.txt` but changes `nT2` to 60:

```@example
println("\$ decaes @/path/to/default.txt --nT2 60") # hide
```

### [Multiple input files](@id multiinput)

One command can process multiple input files using the same analysis settings.
The files may use different supported formats:

```@example
println("\$ decaes image1.nii image2.mat image3.nii.gz image4.par <COMMAND LINE ARGS>") # hide
```

In a settings file, place each image path on a separate line.

### [Specify output folder](@id outfolder)

Outputs are saved in the same folder as the inputs by default.
Use `-o` or `--output` to select another directory:

```@example
println("\$ decaes image.nii --output /path/to/output/folder/ <COMMAND LINE ARGS>") # hide
```

DECAES creates the output directory if needed.

### [Passing image masks](@id passmasks)

Pass an image mask using `-m` or `--mask`.
Voxels outside the mask are skipped and are represented by `NaN` in output maps:

```@example
println("\$ decaes image.nii --mask /path/to/mask.nii <COMMAND LINE ARGS>") # hide
```

For multiple images, pass the corresponding masks in the same order:

```@example
println("\$ decaes image1.nii image2.mat --mask /path/to/mask1.mat /path/to/mask2.nii.gz <COMMAND LINE ARGS>") # hide
```

!!! note
    A separate mask is unnecessary for images that are already masked to zero outside the region of interest. See [`T2mapOptions`](@ref) for the `Threshold` behavior.

### [Automatic brain masking with BET](@id bet)

If no mask is available, DECAES can call FSL's [BET brain extraction tool](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide) to generate one.
Only voxels inside the generated mask are processed:

```@example
println("\$ decaes image.nii --bet <COMMAND LINE ARGS>") # hide
```

If `bet` is not on `PATH`, specify its location with `--betpath`.
Pass BET arguments as one comma- or space-separated string using `--betargs`:

```@example
println("\$ decaes image.nii --bet --betpath /path/to/bet --betargs -m,-n <COMMAND LINE ARGS>") # hide
```

!!! note
    `--mask` takes precedence over `--bet`.
