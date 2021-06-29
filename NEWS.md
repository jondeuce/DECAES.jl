# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
