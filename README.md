# Random-Forest-Machine-Learning
Using random forest machine learning to predict plant geography and carbon fluxes

In the project datasets, you will find the following files:

(1) Climate datasets:

a: Predaymean1961_1990.csv, Tswrfdaymean1961_1990.csv, Tmaxdaymean1961_1990.csv, Tmindaymean1961_1990, Tmpdaymean1961_1990.csv (these variables are calculated as the multi-year average for a yearly basis (365days). 

b: Predaymean1961-1990_statics.csv,  Tswrfdaymean1961_1990_statics.csv, Tmaxdaymean1961_1990_statics.csv, Tmindaymean1961_1990_statics, Tmpdaymean1961_1990_statics.csv (these variables are calculated as mean, median and standard deviation of four seasons from "a" group datasets.  

(2) other datasets:

c. soilmap_center_interpolated.dat (soil texture datasets)

d. gridlist_pan_gfed_ISO3_UN.txt  (the country and continent codes)

e. LPJ-GUESS_output_BERN1.csv (LPJ GUESS outputs and the observed biomes)

f. example datasets (data_index-example.csv and data_index_2.csv)

g. scripts: Biome.py.ipynb for plotting the biome map using "the legend of biomes.txt" 

For your self-learning to be familiar with machine learning, you can first look at the script BERN01Demo.ipynb. 


# BERN01_random_forest
data information:
1. There are five climate variables which are daily average values during 1961-1990. They are Tswrf (Total shortwave radiation flux, W m-2), Pre (Precipitation, mm day-1), Tmp (Daily mean temperature, K), 
Tmax (Daily maximum temperature, K), and Tmin (Daily minimum temperature, K). 

2. In the file LPJ-GUESS_output BERN1.csv, the following variables are collected:
NPP: net primary productivity (kg C m-2 year-1)
SoilR: soil respiration (kg C m-2 year-1)
MaxBiomeCmass: The maximum biomass from a single biome (kg C m-2)
MxbiomeLAI: The maximum leaf area index from a single biome (unitless)
VegC: Vegetation carbon poo (kg C m-2)l
LitterC: Litter carbon pool (kg C m-2)
SoilC: Soil carbon pool (kg C m-2)
Biome_Cmass: The biome type based on the maximum biomass (category)
Biome_LAI: The biome type based on the maximum LAI (category)
Biome_obs: The observed biome type (category)

3. legend of biome

4. soilmap_center_interpolated.dat: the texture inforamtion for each grid cell (Percentage).

Path for loading the CSV dataset:
data = pd.read_csv(r'C:\Users\asus\Desktop\Project_datasets\data_index_2.csv')

Path for loading the world map:
world = gpd.read_file(r'C:\Users\asus\Desktop\Lund\Computational Modelling\Machine Learning Forest\110m_cultural')
