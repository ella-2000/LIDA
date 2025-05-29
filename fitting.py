# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

# Load the spectrum data
# Ensure to replace the path with the correct path to your 'spectrum.txt' file
λ, V = np.loadtxt("C:/Users/ellaj/OneDrive/Documents/LIDA/spectrum.txt", unpack=True)

# Convert wavelengths from meters to nanometers
λ = λ * 1e9  # Convert to nm

# Define the Gaussian function for a single spectral line
def gaussian_single(λ, λ_0, I_0, σ_s):
    return I_0 * np.exp(-(((λ - λ_0) / σ_s) ** 2))

# (a) Fit the spectrum with a single spectral line
initial_guess = [590, 2, 1]  # Initial guess for λ_0, I_0, σ_s
popt, pcov = curve_fit(gaussian_single, λ, V, p0=initial_guess, sigma=np.full_like(V, 0.1), absolute_sigma=True)

# Print the fitted parameters
print("Single Gaussian Fit Parameters:")
print(f"λ_0 = {popt[0]:.3g} +/- {np.sqrt(pcov[0, 0]):.3g} nm")
print(f"I_0 = {popt[1]:.3g} +/- {np.sqrt(pcov[1, 1]):.3g} V")
print(f"σ_s = {popt[2]:.3g} +/- {np.sqrt(pcov[2, 2]):.3g} nm")

# (b) Plot the fitted curve
fit_λ = np.arange(586, 592.5, 0.001)
plt.figure(figsize=(10, 6))
plt.plot(λ, V, label="Data", color='blue', marker='o', markersize=3, linestyle='None')
plt.plot(fit_λ, gaussian_single(fit_λ, *popt), "r", label="Fitted Curve (Single Gaussian)")
plt.xlabel("λ [nm]")
plt.ylabel("V [V]")
plt.title("Spectrum Fitted with Single Spectral Line")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# (c) Plot the residuals between the data and the best-fit Gaussian function
plt.figure(figsize=(10, 4))
residuals = V - gaussian_single(λ, *popt)
plt.errorbar(λ, residuals, fmt='.r')
plt.title('Residuals (Single Gaussian Fit)')
plt.ylabel('V(measured) - V(bestfit)')
plt.xlabel('λ [nm]')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.grid()
plt.tight_layout()
plt.show()

# (d) Calculate the χ² and P-value for the single-Gaussian fit
def csq(V, V_err, expected):
    return np.sum(((V - expected) / V_err) ** 2)

csqsum_single = csq(V, np.full_like(V, 0.1), gaussian_single(λ, *popt))
print(f'χ² (Single Gaussian) = {csqsum_single:.2f}')
dof_single = len(λ) - len(popt)  # Degrees of freedom for single Gaussian
print(f'dof (Single Gaussian) = {dof_single}')
p_value_single = chi2.sf(csqsum_single, dof_single)
print(f'p-value (Single Gaussian) = {p_value_single:.4e}')

# (e) Fit with two Gaussian components
def two_gaussians(λ, λ_01, I_01, σ_s1, λ_02, I_02, σ_s2):
    return (I_01 * np.exp(-((λ - λ_01) / σ_s1) ** 2) +
            I_02 * np.exp(-((λ - λ_02) / σ_s2) ** 2))

# Initial guess for two Gaussians
initial_guess_2 = [589.6, 2, 1, 590.4, 2, 1]
popt_2, pcov_2 = curve_fit(two_gaussians, λ, V, p0=initial_guess_2, sigma=np.full_like(V, 0.1), absolute_sigma=True)

# Print the fitted parameters for the two Gaussian fit
print("\nTwo Gaussian Fit Parameters:")
print(f"λ_01 = {popt_2[0]:.3g} +/- {np.sqrt(pcov_2[0, 0]):.3g} nm")
print(f"I_01 = {popt_2[1]:.3g} +/- {np.sqrt(pcov_2[1, 1]):.3g} V")
print(f"σ_s1 = {popt_2[2]:.3g} +/- {np.sqrt(pcov_2[2, 2]):.3g} nm")
print(f"λ_02 = {popt_2[3]:.3g} +/- {np.sqrt(pcov_2[3, 3]):.3g} nm")
print(f"I_02 = {popt_2[4]:.3g} +/- {np.sqrt(pcov_2[4, 4]):.3g} V")
print(f"σ_s2 = {popt_2[5]:.3g} +/- {np.sqrt(pcov_2[5, 5]):.3g} nm")

# Plot the combined two Gaussian fit
plt.figure(figsize=(10, 6))
plt.plot(λ, V, label="Data", color='blue', marker='o', markersize=3, linestyle='None')
plt.plot(fit_λ, two_gaussians(fit_λ, *popt_2), "r", label="Fitted Curve (Two Gaussians)")
plt.xlabel("λ [nm]")
plt.ylabel("V [V]")
plt.title("Spectrum Fitted with Two Spectral Lines")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot the residuals for the two Gaussian fit
plt.figure(figsize=(10, 4))
residuals_2 = V - two_gaussians(λ, *popt_2)
plt.errorbar(λ, residuals_2, fmt='.r')
plt.title('Residuals (Two Gaussian Fit)')
plt.ylabel('V(measured) - V(bestfit)')
plt.xlabel('λ [nm]')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.grid()
plt.tight_layout()
plt.show()

# (f) Plot the measured spectrum with the individual profile of each line overlaid
plt.figure(figsize=(10, 6))
plt.plot(λ, V, label="Data", color='blue', marker='o', markersize=3, linestyle='None')
fit_curve_1 = gaussian_single(λ, *popt_2[:3])
fit_curve_2 = gaussian_single(λ, *popt_2[3:])
plt.plot(λ, fit_curve_1, label="Gaussian 1", color="g")
plt.plot(λ, fit_curve_2, label="Gaussian 2", color="m")
plt.xlabel("λ [nm]")
plt.ylabel("V [V]")
plt.title("Spectrum Fitted with Two Gaussians")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# (g) State the measured value for the wavelength and uncertainty of each line
λ_01_measured = popt_2[0]
λ_02_measured = popt_2[3]
λ_01_uncertainty = np.sqrt(np.diag(pcov_2))[0]
λ_02_uncertainty = np.sqrt(np.diag(pcov_2))[3]
print(f"Measured wavelength of Gaussian 1: {λ_01_measured:.3f} +/- {λ_01_uncertainty:.3f} nm")
print(f"Measured wavelength of Gaussian 2: {λ_02_measured:.3f} +/- {λ_02_uncertainty:.3f} nm")

I_01_measured = popt_2[1]
I_02_measured = popt_2[4]
intensity_ratio = I_01_measured / I_02_measured
print(f"Intensity ratio (I_01/I_02): {intensity_ratio:.3f}")

σ_s1_measured = popt_2[2]
σ_s2_measured = popt_2[5]
σ_s_measured = np.mean([σ_s1_measured, σ_s2_measured])
σ_s_uncertainty = np.mean([np.sqrt(pcov_2[2, 2]), np.sqrt(pcov_2[5, 5])])
print(f"Measured spectrometer resolution: {σ_s_measured:.3f} +/- {σ_s_uncertainty:.3f} nm")