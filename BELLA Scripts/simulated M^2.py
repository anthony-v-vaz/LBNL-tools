import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Parameters for simulation
w0 = 1e-3  # Beam waist in mm
z0 = 0  # Position of the beam waist in mm
beam_wavelength_nm = 520
beam_wavelength_mm = beam_wavelength_nm * 1e-6  # Convert nm to mm

# Define the model function for M^2 fit
def msquared_model(z, M_squared):
    lambda_mm = beam_wavelength_mm
    return w0**2 * (1 + ((z - z0)**2 * (M_squared**2 * lambda_mm / (np.pi * w0**2))**2))

# Simulate data for different M^2 values
M_squared_values = [1, 1.5, 3]
z_data = np.linspace(-10, 10, 100)  # Simulate measurements from z = -10mm to 10mm

simulated_data = {}
for M_squared in M_squared_values:
    u_data_squared = msquared_model(z_data, M_squared)
    # Add some noise to simulate measurement uncertainty
    noise = np.random.normal(0, 0.05 * max(u_data_squared), len(u_data_squared))
    simulated_data[M_squared] = u_data_squared + noise

# Perform the fit for each M^2 value and plot
plt.figure(figsize=(10, 5))
for M_squared, u_data_squared in simulated_data.items():
    # Fit function should only adjust M_squared, keep w0 and z0 constant
    fit_func = lambda z, M: msquared_model(z, M)
    popt, pcov = curve_fit(fit_func, z_data, u_data_squared, p0=[M_squared])

    # Plot the simulated data and the fit
    plt.scatter(z_data, np.sqrt(u_data_squared), label=f'Simulated Data M^2={M_squared}', alpha=0.6)
    plt.plot(z_data, np.sqrt(msquared_model(z_data, *popt)), label=f'Fit M^2={popt[0]:.2f}')

plt.xlabel('Stage Z (mm)')
plt.ylabel('Beam Width (mm)')
plt.title('Simulated Beam Width Data and M^2 Fits')
plt.legend()
plt.show()
