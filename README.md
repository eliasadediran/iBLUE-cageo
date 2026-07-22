# iBLUE

This repository contains the implementation of iBLUE: An Algorithmic Framework for Quantifying Interpolation Uncertainty in Bathymetric Surfaces Using Spectral and Spatial Statistical Estimators.

The framework consists of three principal components:

1. Construction of a statistical representation of unsampled seafloor structure from observed bathymetric data.
2. Classification of the reconstructed seabed into low- or high-complexity morphological regimes based on its spectral slope.
3. Estimation of interpolation uncertainty using regime-specific spectral and spatial statistical estimators.

## Project layout

- `data/` contains the example bathymetric datasets used by the workflow.
- `notebooks/` contains demonstration notebooks that reproduce the analysis pipeline.
- `src/` contains the Python package implementing the core algorithms.
- `results/` is used for generated outputs such as plots and summary files.

## How it works

The workflow proceeds through three stages:

1. The bathymetric surface is processed to construct a statistical representation of unsampled seafloor structure.
2. The reconstructed seabed is classified into low- or high-complexity morphological regimes based on its spectral slope.
3. Interpolation uncertainty is estimated using regime-specific spectral and spatial statistical methods.

## Setup

Create or activate the conda environment from `environment.yml`.
This environment includes GDAL and the other dependencies needed by the notebook workflow.

```bash
conda env create -f environment.yml
conda activate iBLUE
```

## Running the notebook

Open the notebook in `notebooks/` and run the cells from top to bottom.
Make sure the selected notebook kernel points to the `iBLUE` environment so that `osgeo` and GDAL are available.

## Repository modules

- `src/helpers/` contains file loading, matrix operations, plotting, and statistics helpers.
- `src/algorithms/` contains the core estimators, classifier, and reconstruction routines.
- `notebooks/` provides end-to-end examples of the analysis workflow.
