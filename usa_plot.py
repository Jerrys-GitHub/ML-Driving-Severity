import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt

# ---------------------------
# Load world countries (via geodatasets)
# ---------------------------
countries_path = geodatasets.get_path("naturalearth.countries")
countries = gpd.read_file(countries_path)

# Filter for USA
usa = countries[countries["name"] == "United States of America"]

# ---------------------------
# Create GeoDataFrame for your points
# ---------------------------
gdf = gpd.GeoDataFrame(
    regional_df,
    geometry=gpd.points_from_xy(regional_df.Start_Lng, regional_df.Start_Lat),
    crs="EPSG:4326"
)

# ---------------------------
# Plot
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 8))

# Plot USA base map
usa.to_crs("EPSG:4326").plot(ax=ax, color="white", edgecolor="black")

# Plot sample of points
gdf.sample(10000).plot(ax=ax, color="red", markersize=1)

plt.title("USA Accident Points")
plt.show()
