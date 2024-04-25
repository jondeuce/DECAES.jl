# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - Apr 17, 2024

### Changed
- This version, released concurrently with v0.5.2, brings no new features but includes significant internal algorithmic improvements. If you're new to DECAES, this is the version to use. For existing users, please note that after upgrading to v0.6.0, it is important to rerun DECAES on all subjects used in a study. While any changes represent improved estimates, one should nevertheless exercise caution and avoid mixing pre-v0.6.0 DECAES results with v0.6.0 results, as doing so could potentially conflate changes due to algorithmic improvements with changes in tissue properties. If for some reason you are unable to rerun all results, you can still use v0.5.2, which provides the new `decaes` launcher, faster computation, and negligible changes in results.
- Flip angle estimation has been made faster and more robust: stricter tolerances are used in the minimization problem, and cubic Hermite splines are used as surrogate functions (faster to evaluate and can be minimized analytically)
- The [chi-squared regularization method](https://jondeuce.github.io/DECAES.jl/dev/ref/#DECAES.lsqnonneg_chi2) now uses Brent's method for rootfinding instead of binary search. This results in both more accurate and faster rootfinding, i.e. setting `--Reg chi2 --Chi2Factor 1.02` will result in `Chi2Factor` maps much closer to the requested `1.02`.
- The [generalized cross-validation (GCV) regularization method](https://jondeuce.github.io/DECAES.jl/dev/ref/#DECAES.lsqnonneg_gcv) is solved with stricter tolerances, trading off a small speed regression for more accurate minimization of the GCV objective
- The [L-curve regularization method](https://jondeuce.github.io/DECAES.jl/dev/ref/#DECAES.lsqnonneg_lcurve) is solved with stricter tolerances, trading off a small speed regression for more accurate location of the point of maximum curvature on the L-curve

## [0.5.2] - Apr 17, 2024

### Added
- More convenient command line interface: upon installation, DECAES will create a launcher script `decaes.sh` (`decaes.bat` on Windows) located in `~/.julia/bin`. Ensure this folder is added to your `PATH`, then run DECAES via `decaes <command line args>`.
- New experimental regularization method: use [Morozov's Discrepancy Principle (MDP)](https://jondeuce.github.io/DECAES.jl/dev/ref/#DECAES.lsqnonneg_mdp) by passing the CLI flags `--Reg mdp --RegParams <noise level>`, where `<noise level>` is an estimate of the voxelwise noise level. For Gaussian noise, this corresponds to the standard deviation.

### Changed
- Deprecated the `--Chi2Factor` flag in favour of `--RegParams` to unify the interface for passing parameters to regularization methods
- The bottleneck of DECAES - solving nonnegative least-squares (NNLS) problems - has been largely rewritten and received major performance and robustness improvements (credit @kamesy)
- The [generalized cross-validation (GCV) regularization method](https://jondeuce.github.io/DECAES.jl/dev/ref/#DECAES.lsqnonneg_gcv) now requires the number of T2 components, `--nT2`, to be less than or equal to the number of echoes, in accordance with [the literature](https://epubs.siam.org/doi/10.1137/1034115)
- The [generalized cross-validation (GCV) regularization method](https://jondeuce.github.io/DECAES.jl/dev/ref/#DECAES.lsqnonneg_gcv) is now *much* faster to compute (~5-10X), with speed now comparable to the L-curve and chi-squared methods
- Deprecated the experimental `--compile` CLI flag in favour of the `decaes` launcher script

## [0.5.0] - May 16, 2023

### Changed
- Support for Julia versions below v1.9 is dropped

## [0.4.0] - Jun 23, 2021

### Changed
- Performance improvements: DECAES now runs up to 20% faster; approximately 1-2 mins for whole brain scans when providing brain masks (or using built-in BET support) and using multithreading
- Improved documentation: starting DECAES with the `--help` flag now shows detailed descriptions for all CLI settings
- Improved MATLAB interface: the MATLAB interface script `decaes.m` has been updated with an API which more closely resembles the command line interface; please download the new version and run `help decaes` for more information (download here: https://github.com/jondeuce/mwiexamples/blob/master/decaes.m)
- Choice of T2 regularization method now required: the previous default regularization method must now be set explicitly, if desired, by passing `--Reg chi2 --Chi2Factor 1.02`. See below for additional new regularization methods.

### Added
- Improved chi-squared based T2 regularization: the desired relative increase in chi-squared (for example, 1.02X when using the previous default `--Reg chi2 --Chi2Factor 1.02`) is now solved more accurately
- New T2 distribution regularization method #1: parameter-free regularization using the L-curve method can be used by passing `--Reg lcurve` to DECAES (see the algorithm here: https://iopscience.iop.org/article/10.1088/2633-1357/abad0d)
- New T2 distribution regularization method #2: parameter-free regularization using the generalized cross-validation (GCV) method can be used by passing `--Reg gcv` to DECAES (see section 6.3 here: https://doi.org/10.1137/1034115)

## [0.3.0] - Jul 2, 2020

### Changed
- Removed default values for `TE`, `nT2`, `T2Range`, `SPWin`, and `MPWin`
- As a consequence of the above change, the corresponding CLI flags `--TE`, `--nT2`, `--T2Range`, `--SPWin`, and `--MPWin` are now required when the flags `--T2map` and `--T2part` are passed

## [0.2.0] - Jun 20, 2020

### Added

- Added PAR/XML/REC read support

## [0.1.0] - Apr 14, 2020

A changelog is not maintained for this version.

This version marks the initial release of DECAES, as well as the registering of the package.
