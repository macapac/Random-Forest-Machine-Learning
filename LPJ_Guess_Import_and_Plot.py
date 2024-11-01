import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

def import_LPJ_output():
    # Hardcoded file path for LPJ-GUESS output
    file_path = r'C:\Users\asus\Desktop\Lund\Computational Modelling\Machine Learning Forest\Project_datasets\LPJ-GUESS_output_BERN1.csv'
    data = pd.read_csv(file_path)
    return data

def plot_all_data_ccrs(data):
    projection = ccrs.PlateCarree()

    # Ensure 'figs' directory exists
    if not os.path.exists('figs'):
        os.makedirs('figs')

    # List of variables to plot
    variables = [col for col in data.columns if col not in ['Lon', 'Lat']]

    # Create longitude and latitude grid
    lons = np.unique(data['Lon'])
    lats = np.unique(data['Lat'])

    # Loop through variables and plot them on a global map using contour plots
    for var in variables:
        # Reshape the variable data to match the lat/lon grid
        var_data = data.pivot(index='Lat', columns='Lon', values=var).values

        plt.figure(figsize=(20, 8))
        ax = plt.axes(projection=projection)

        # Add coastlines
        ax.coastlines()

        # Create contour plot for the variable
        contour = plt.contourf(lons, lats, var_data, 60, cmap='viridis', transform=projection)
        plt.colorbar(contour, label=var)

        plt.tight_layout()
        plt.title(f'Spatial distribution of {var}')

        # Save the plot as an image file
        plt.savefig(f'figs/{var}.png')
        plt.close()

if __name__ == '__main__':
    # Import data using hardcoded path
    LPJ_output = import_LPJ_output()
    # Plot all variables
    plot_all_data_ccrs(LPJ_output)
