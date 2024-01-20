import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tkinter import filedialog, Tk
from pathlib import Path
import math

# Wavelength in nm
beam_wavelength = 520

# Pixel size in microns converted to mm (to be compatible with stage z)
pixel_size_mm = 2.2 * 0.001
initial_guesses = [0.1, 1.0, 0.0]  # You may need to adjust these based on your data

# Define the model function for M^2 fit, according to the equation W(z)^2 = W0^2 + M^4 * (lambda/(pi*W0))^2 * (z - z0)^2
def M2_model(z, W0, M2, z0):
    lambda_mm = beam_wavelength * 1e-6  # Convert wavelength to mm
    return W0**2 + (M2 * (lambda_mm / (np.pi * W0))**2) * (z - z0)**2

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

    stage_z = filtered_data['P2-MANPAR value1 Alias:Stage Z']
    second_mom_x45 = filtered_data['P2-FS_RefOut 2ndmomx45 T -999.00_tony'] * pixel_size_mm
    second_mom_y45 = filtered_data['P2-FS_RefOut 2ndmomy45 T -999.00_tony'] * pixel_size_mm
    # print(second_mom_x45.iloc[0])
    # Check if all columns have the same length
    if len(stage_z) == len(second_mom_x45) == len(second_mom_y45):
        # Plotting the data
        plt.figure(figsize=(10, 5))
        plt.scatter(stage_z, second_mom_x45, label='2ndmomx45 ')
        plt.scatter(stage_z, second_mom_y45, label='2ndmomy45 ', color='r')
        plt.xlabel('Stage Z (mm)')
        plt.ylabel('Second Moment (mm)')
        plt.title('Second Moments vs. Stage Z')
        plt.legend()

        try:
            popt_x, pcov_x = curve_fit(M2_model, stage_z, second_mom_x45 ** 2, p0=initial_guesses, maxfev=10000)
            popt_y, pcov_y = curve_fit(M2_model, stage_z, second_mom_y45 ** 2, p0=initial_guesses, maxfev=10000)
            # ... [the rest of your script remains unchanged]
        except RuntimeError as e:
            print(f"An error occurred during curve fitting: {e}")

        # Plot the fit
        x_range = np.linspace(min(stage_z), max(stage_z), 100)
        plt.plot(x_range, M2_model(x_range, *popt_x), label='M² Fit for 2ndmomx45', color='blue')
        plt.plot(x_range, M2_model(x_range, *popt_y), label='M² Fit for 2ndmomy45', color='orange')
        plt.legend()

        plt.show()
    else:
        print("The lengths of the data columns are not the same.")
else:
    print("File not selected.")
