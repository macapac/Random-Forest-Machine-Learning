import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load the CSV dataset
data = pd.read_csv(r'C:\Users\asus\Desktop\Lund\Computational Modelling\Machine Learning Forest\Project_datasets\data_index_2.csv')

# Ensure 'Lon' and 'Lat' columns exist before converting to GeoDataFrame
if 'Lon' in data.columns and 'Lat' in data.columns:
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Lon, data.Lat))
else:
    raise ValueError("The CSV file must contain 'Lon' and 'Lat' columns.")

# Load the world map
world = gpd.read_file(r'C:\Users\asus\Desktop\Lund\Computational Modelling\Machine Learning Forest\110m_cultural')

# Define the figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle('Global Distribution of Biomes and Carbon Flux Variables', fontsize=20)

# Define each variable's plot settings
variables = ['Biome_obs', 'Biome_Cmax', 'NPP', 'VegC']
titles = ['Global Distribution of Biome_obs', 
          'Global Distribution of Biome_Cmax', 
          'Net Primary Production (NPP)', 
          'Vegetation Carbon (VegC)']
color_maps = ['viridis', 'plasma', 'YlGnBu', 'OrRd']  # Choose appropriate colormaps

# Plot each variable on a separate subplot
for ax, var, title, cmap in zip(axs.flatten(), variables, titles, color_maps):
    world.plot(ax=ax, color='lightgray')
    gdf.plot(ax=ax, column=var, cmap=cmap, legend=True, markersize=5)
    ax.set_title(title)

# Show the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
plt.show()


# Plotting only the chosen biomes and countries:
filtered_gdf = gdf[(gdf['Biome_obs'].isin([8, 9])) & (gdf['UN'].isin([68, 76]))]

# Create a figure and an axis to plot on
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot the filtered GeoDataFrame on the same axis
filtered_gdf.plot(ax=ax, column='Biome_obs', cmap='viridis', markersize=5)

# Plot the country boundaries
world.boundary.plot(ax=ax, edgecolor='black', linewidth=0.8)

# Set the title for the plot
ax.set_title('Chosen Biomes, Yellow: Tropical rain forest, Purple: Tropical seasonal forest')

# Set the x and y axis limits to focus on the Americas
ax.set_xlim([-85, -30])  # Longitude
ax.set_ylim([-60, 15])    # Latitude

# Show the plot
plt.tight_layout()
plt.show()
