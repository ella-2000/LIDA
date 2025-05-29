# NOTE: these questions were taken from a data analysis assignment 

# Spectral Analysis with Gaussian Fitting 

## Overview

This project involves the analysis of spectral data from a spectrometer. The goal is to fit the spectrum of Sodium in the region around 590 nm using Gaussian functions, evaluate the fit, and analyze the results. The analysis is done using Python with libraries such as NumPy, Matplotlib, and SciPy.

## Questions Addressed

The following questions are addressed in the analysis:

1. **Fit the spectrum with a single spectral line**:
   - Fit the spectrum data using a Gaussian function of the form:
     I = I_0 e^{-\left(\frac{\lambda - \lambda_0}{\sigma_s}\right)^2}

2. **Determine the wavelength of the spectral line and its uncertainty**:
   - From the best-fit values, determine λ_0 ± σ_{λ_0} and the instrument resolution σ_s and its uncertainty.

3. **Plot the residuals**:
   - Plot the residuals between the data and the best-fit Gaussian function.

4. **Calculate the χ² and P-value for the single-Gaussian fit**.

5. **Fit with two Gaussian components**:
   - Add a second Gaussian component and investigate if the two Gaussians produce better residuals and χ² P-value.

6. **Plot individual Gaussian profiles**:
   - Plot the measured spectrum with the individual profile of each line overlaid, clearly distinguishable through different colors.

7. **State measured values**:
   - State the measured value for the wavelength and uncertainty of each line, the ratio of their intensities, and the spectrometer resolution σ_s and its uncertainty.

## Requirements

To run the script, you need to have the following libraries installed:

- NumPy
- Matplotlib
- SciPy

## Usage

1. **Data File**: Ensure you have the `spectrum.txt` file containing the wavelength and voltage data. The first column should be wavelengths in meters, and the second column should be the corresponding voltage readings.

2. **Run the Script**: Execute the Python script in an environment where the above libraries are installed. Ensure to update the path to the `spectrum.txt` file in the script if necessary.


