import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tkinter import filedialog, Tk
from pathlib import Path

# Wavelength in nm converted to mm
beam_wavelength = 520 * 1e-6

# Pixel size in microns converted to mm (to be compatible with stage z)
pixel_size_mm = 2.2 * 1e-3

# Define the model function for M^2 fit, according to the provided equation
def msquared_model(z, w0, z0, M_squared):
    lambda_mm = beam_wavelength
    return w0**2 * (1 + ((z - z0)**2 * ((M_squared**2 * lambda_mm / (np.pi * w0**2))**2)))

# Function to perform the curve fit
def perform_fit(z_data, w_data_squared):
    initial_guesses = [np.min(w_data_squared), np.mean(z_data), 1]  # w0, z0, M^2
    bounds = ([0, -np.inf, 0], [np.inf, np.inf, np.inf])  # Bounds for w0, z0, M^2
    popt, pcov = curve_fit(msquared_model, z_data, w_data_squared, p0=initial_guesses, bounds=bounds)
    return popt, pcov

# Open a dialog to select the file
root = Tk()
root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
file_path = filedialog.askopenfilename()  # Show an "Open" dialog box and return the path to the selected file

if file_path:
    # Load the data
    data = pd.read_csv(Path(file_path), sep='\t')  # Adjust the separator if needed

    # Extract relevant data
    # Ensure that only rows with non-zero 'Stage Z' values are used
    filtered_data = data[data['P2-MANPAR value1 Alias:Stage Z'] != 0]

    # Group by 'Stage Z' and calculate mean for each group
    grouped_data = filtered_data.groupby('P2-MANPAR value1 Alias:Stage Z').mean()

    # Now you have the averaged second moments for each stage z value
    avg_second_mom_x45 = grouped_data['P2-FS_RefOut 2ndmomx45 T -999.00_tony'] * pixel_size_mm
    avg_second_mom_y45 = grouped_data['P2-FS_RefOut 2ndmomy45 T -999.00_tony'] * pixel_size_mm

    # Perform the fit for avg_second_mom_x45 squared
    popt_x45, pcov_x45 = perform_fit(grouped_data.index, avg_second_mom_x45**2)

    # Perform the fit for avg_second_mom_y45 squared
    popt_y45, pcov_y45 = perform_fit(grouped_data.index, avg_second_mom_y45**2)

    # Extract the M^2 values from the fit parameters
    M_squared_x45 = popt_x45[2]
    M_squared_y45 = popt_y45[2]

    # Print the M^2 values
    print(f"The M^2 value for avg_second_mom_x45 is: {M_squared_x45}")
    print(f"The M^2 value for avg_second_mom_y45 is: {M_squared_y45}")

    # Plot the data and the fit
    plt.figure(figsize=(10, 5))
    plt.scatter(grouped_data.index, avg_second_mom_x45, label='Avg 2ndmomx45')
    plt.scatter(grouped_data.index, avg_second_mom_y45, label='Avg 2ndmomy45', color='r')

    # Generate fit lines for plotting
    z_fit = np.linspace(np.min(grouped_data.index), np.max(grouped_data.index), 100)
    fit_line_x45 = msquared_model(z_fit, *popt_x45)
    fit_line_y45 = msquared_model(z_fit, *popt_y45)

    plt.plot(z_fit, np.sqrt(fit_line_x45), label='Fit 2ndmomx45', color='blue')
    plt.plot(z_fit, np.sqrt(fit_line_y45), label='Fit 2ndmomy45', color='orange')

    plt.xlabel('Stage Z (mm)')
    plt.ylabel('Average Second Moment (mm)')
    plt.title('Average Second Moments vs. Stage Z with M^2 Fit')
    plt.legend()
    plt.show()

else:
    print("File not selected.")
