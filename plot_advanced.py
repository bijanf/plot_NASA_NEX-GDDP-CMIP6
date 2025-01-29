import os
from cdo import Cdo
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

# Configuration
SHOW_ERA5 = True  # Flag to control whether ERA5 is processed and plotted
INPUT_DIR = "/media/bijan/BIJI/nc-files"  # Directory for global NetCDF files
OUTPUT_DIR = "./out"  # Directory for preprocessed files
LAT_RANGE = (35, 57)  # Latitude range for the region of interest
LON_RANGE = (46, 88)  # Longitude range for the region of interest
#LAT_RANGE = (-90, 90)  # Latitude range for the region of interest
#LON_RANGE = (-180, 180)  # Longitude range for the region of interest
#LAT_RANGE = (47.0, 55.0)  # Latitude range for Germany
#LON_RANGE = (5.5, 15.5)   # Longitude range for Germany
#LAT_RANGE = (35.0, 72.0)  # Latitude range for Europe
#LON_RANGE = (-25.0, 45.0)  # Longitude range for Europe
#LAT_RANGE = (24.0, 49.0)  # Latitude range for the contiguous United States
#LON_RANGE = (-125.0, -66.5)  # Longitude range for the contiguous United States
VARIABLE = "tas"  # Variable to process ("pr" for precipitation)
SSP_EXPERIMENTS = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
CLIMATOLOGY_START = 1981
CLIMATOLOGY_END = 2010

if VARIABLE == "tas":
    ERA5_FILE = "era_1950_2014_ym.nc"  # Global NetCDF ERA5 file
else: 
    ERA5_FILE = "pr_yearmean_ERA5_1940-2024.nc"  # Global NetCDF ERA5 file

OUTPUT_PATH = VARIABLE + "_anomalies_ensemble_time_series_CA.png"
FUTURE_PERIOD_START = 2070
FUTURE_PERIOD_END = 2099
MAP_OUTPUT_PATH = VARIABLE + "ensemble_mean_maps_2070-2099_CA.png"
n_years = 1  # Define the number of years for the running mean

# List of ISIMIP models to include
isimip_models = [
    "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL",
    "CNRM-CM6-1", "CNRM-ESM2-1", "CanESM5", "EC-Earth3", "MIROC6"
]

# Initialize CDO
cdo = Cdo()

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to extract the year from filenames
def extract_year(filename):
    match = re.search(r"_(\d{4})_", filename)  # Look for _YYYY_ pattern
    if match:
        return int(match.group(1))  # Extract the year as an integer
    else:
        raise ValueError(f"Year not found in filename: {filename}")

def process_era5_with_cdo(input_file, lat_range, lon_range, clim_start, clim_end, output_dir):
    """Process ERA5 data using CDO to calculate regional anomalies."""
    #if VARIABLE == "pr":
    #    return None  # Skip ERA5 processing for precipitation

    # Step 1: Subset the region
    regional_file = os.path.join(output_dir, "era5_regional.nc")
    if not os.path.exists(regional_file):
        cdo.sellonlatbox(
            lon_range[0], lon_range[1], lat_range[0], lat_range[1],
            input=input_file,
            output=regional_file
        )

    # Step 2: Calculate climatology (mean for 1981-2010)
    climatology_file = os.path.join(output_dir, "era5_climatology.nc")
    if not os.path.exists(climatology_file):
        cdo.timmean(
            input=f"-selyear,{clim_start}/{clim_end} {regional_file}",
            output=climatology_file
        )

    # Step 3: Calculate anomalies (subtract climatology)
    anomalies_file = os.path.join(output_dir, "era5_anomalies.nc")
    if not os.path.exists(anomalies_file):
        cdo.sub(
            input=f"{regional_file} {climatology_file}",
            output=anomalies_file
        )

    # Step 4: Calculate spatial mean (fldmean)
    anomalies_mean_file = os.path.join(output_dir, "era5_anomalies_mean.nc")
    if not os.path.exists(anomalies_mean_file):
        cdo.fldmean(
            input=anomalies_file,
            output=anomalies_mean_file
        )

    return anomalies_mean_file

def preprocess_ssp_models():
    """Preprocess SSP experiments, merging files by model and scenario."""
    processed_files = {}
    historical_climatology = {}

    # Step 1: Process historical files first
    print("Processing historical experiment...")
    model_files = {}

    # Group historical files by model
    for f in os.listdir(INPUT_DIR):
        if VARIABLE in f and "historical" in f:
            model = f.split("_")[2]  # Assuming model name is the 3rd part of the filename
            if model in isimip_models:  # Only process models in the ISIMIP list
                if model not in model_files:
                    model_files[model] = []
                model_files[model].append(os.path.join(INPUT_DIR, f))

    # Merge historical files by model and calculate climatology
    for model, files in model_files.items():
        # Sort files by year
        files = sorted(files, key=extract_year)  # Updated sorting method

        print(f"Processing historical model: {model}, files: {files}")

        merged_file = os.path.join(OUTPUT_DIR, f"{model}_historical_merged.nc")
        if not os.path.exists(merged_file):
            cdo.mergetime(input=" ".join(files), output=merged_file)

        regional_file = os.path.join(OUTPUT_DIR, f"{model}_historical_regional.nc")
        if not os.path.exists(regional_file):
            cdo.sellonlatbox(
                LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1],
                input=merged_file,
                output=regional_file
            )

        climatology_file = os.path.join(OUTPUT_DIR, f"{model}_historical_climatology.nc")
        if not os.path.exists(climatology_file):
            cdo.timmean(
                input=f"-selyear,{CLIMATOLOGY_START}/{CLIMATOLOGY_END} {regional_file}",
                output=climatology_file
            )
        historical_climatology[model] = climatology_file

    # Step 2: Process SSP experiments and associate with historical climatology
    for experiment in SSP_EXPERIMENTS:
        print(f"Processing SSP experiment: {experiment}...")
        model_files = {}

        # Group SSP files by model
        for f in os.listdir(INPUT_DIR):
            if VARIABLE in f and experiment in f:
                model = f.split("_")[2]  # Assuming model name is the 3rd part of the filename
                if model not in model_files:
                    model_files[model] = []
                model_files[model].append(os.path.join(INPUT_DIR, f))

        for model, files in model_files.items():
            # Sort files by year
            files = sorted(files, key=extract_year)  # Updated sorting method

            print(f"Processing model: {model} for experiment: {experiment}, files: {files}")

            if model not in historical_climatology:
                print(f"Skipping {model} for {experiment}: Historical climatology not found.")
                continue

            merged_file = os.path.join(OUTPUT_DIR, f"{model}_{experiment}_merged.nc")
            if not os.path.exists(merged_file):
                cdo.mergetime(input=" ".join(files), output=merged_file)

            regional_file = os.path.join(OUTPUT_DIR, f"{model}_{experiment}_regional.nc")
            if not os.path.exists(regional_file):
                cdo.sellonlatbox(
                    LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1],
                    input=merged_file,
                    output=regional_file
                )

            anomalies_file = os.path.join(OUTPUT_DIR, f"{model}_{experiment}_anomalies.nc")
            if not os.path.exists(anomalies_file):
                # Use the precomputed historical climatology
                climatology_file = historical_climatology[model]
                cdo.sub(
                    input=f"{regional_file} {climatology_file}",
                    output=anomalies_file
                )

            anomalies_mean_file = os.path.join(OUTPUT_DIR, f"{model}_{experiment}_anomalies_mean.nc")
            if not os.path.exists(anomalies_mean_file):
                cdo.fldmean(
                    input=anomalies_file,
                    output=anomalies_mean_file
                )

            processed_files[f"{model}_{experiment}"] = anomalies_mean_file

    return processed_files

def preprocess_all_models_and_era5():
    """Preprocess SSP experiments and include ERA5 in the combined plots."""
    era5_anomalies_file = None
    if SHOW_ERA5:
        era5_anomalies_file = process_era5_with_cdo(
            ERA5_FILE, LAT_RANGE, LON_RANGE, CLIMATOLOGY_START, CLIMATOLOGY_END, OUTPUT_DIR
        )
    

    print("Preprocessing SSP experiments...")
    processed_files = preprocess_ssp_models()

    # Add ERA5 as a pseudo-experiment if not processing precipitation
    if era5_anomalies_file:
        processed_files["ERA5"] = era5_anomalies_file

    return processed_files

def plot_ensemble_anomalies(processed_files, output_path):
    """Plot the ensemble mean and spread of anomalies in a single plot with independent logic for time axes."""
    ensemble_data = {}
    sns.set(style="whitegrid")

    # Create the main figure
    fig, ax1 = plt.subplots(figsize=(12, 6))
    #ax2 = ax1.twinx()

    # Determine experiments to plot, including ERA5 if present
    experiments = list(SSP_EXPERIMENTS)
    if "ERA5" in processed_files:
        experiments.append("ERA5")

    for experiment in experiments:
        experiment_files = [file for key, file in processed_files.items() if experiment in key]
        if not experiment_files:
            continue

        datasets = []
        variable_name = (
            "pr" if experiment == "ERA5" and VARIABLE == "pr" else
            "var167" if experiment == "ERA5" and VARIABLE == "tas" else
            VARIABLE
        )

        for file in experiment_files:
            ds = xr.open_dataset(file)
            # Convert units for ERA5 and models if variable is pr
            if VARIABLE == "pr":
                if experiment == "ERA5":
                    # ERA5: Confirm variable is in kg/m²/s (adjust if necessary)
                    ds["pr"] = (ds["pr"] * 86400)  #  → mm/day
                    ds["pr"].attrs["units"] = "mm/day"
                else:
                    # SSP models: Already annual means in kg/m²/s
                    ds["pr"] = ds["pr"] * 86400 # Convert to mm/day
                    ds["pr"].attrs["units"] = "mm/day"


            # Drop time-bound variables to avoid conflicts
            for possible_time_var in ["time_bnds", "time_bounds", "time_bound"]:
                if possible_time_var in ds.variables:
                    ds = ds.drop_vars(possible_time_var)

            # Normalize time to integer years
            if "time" in ds.coords:
                if hasattr(ds["time"], "dt"):
                    ds = ds.assign_coords(time=ds["time"].dt.year)
                else:
                    ds = ds.assign_coords(time=ds["time"].to_index().year)

            # Ensure time is an integer and drop duplicates
            ds["time"] = ds["time"].astype(int)
            unique_time_mask = ~pd.Index(ds["time"].values).duplicated(keep="first")
            ds = ds.isel(time=unique_time_mask)

            # Convert units for precipitation
            if VARIABLE == "pr":
                ds[variable_name] = ds[variable_name]  # Convert kg/m²/s to mm/day
                ds[variable_name].attrs['units'] = 'mm/day'

            datasets.append(ds)

        # Concatenate datasets for the current experiment
        try:
            data = xr.concat([ds[variable_name] for ds in datasets], dim="model")
        except ValueError as e:
            print(f"Error concatenating data for {experiment}: {e}")
            continue

        # Calculate ensemble statistics
        ensemble_mean = data.quantile(0.5, dim="model", skipna=True)
        percentile_75 = data.quantile(0.75, dim="model", skipna=True)
        percentile_25 = data.quantile(0.25, dim="model", skipna=True)

        # Convert to DataFrame for plotting
        mean_df = ensemble_mean.to_dataframe(name="anomaly").reset_index()
        mean_df["anomaly_running_mean"] = mean_df["anomaly"].rolling(window=n_years, center=True).mean()

        percentile_75_df = percentile_75.to_dataframe(name="anomaly_75").reset_index()
        percentile_25_df = percentile_25.to_dataframe(name="anomaly_25").reset_index()
        percentile_75_df["anomaly_running_mean_75"] = percentile_75_df["anomaly_75"].rolling(window=n_years, center=True).mean()
        percentile_25_df["anomaly_running_mean_25"] = percentile_25_df["anomaly_25"].rolling(window=n_years, center=True).mean()

        # Plot mean
        if experiment == "ERA5":
            sns.lineplot(
                data=mean_df,
                x="time",
                y="anomaly_running_mean",
                label=f"{experiment}",
                linewidth=2,
                color="black",  # Solid black line for ERA5
                linestyle="-",  # Solid line style
                ax=ax1
            )
        else:
            sns.lineplot(
                data=mean_df,
                x="time",
                y="anomaly_running_mean",
                label=f"{experiment}",
                linewidth=2,
                ax=ax1
            )

        # Calculate the spread (75th - 25th percentile)
        spread_df = percentile_75 - percentile_25
        spread_df = spread_df.to_dataframe(name="anomaly_spread").reset_index()
        
        # Add spread to the plot
        ax1.fill_between(
            mean_df["time"],
            percentile_25_df["anomaly_running_mean_25"],
            percentile_75_df["anomaly_running_mean_75"],
            alpha=0.2,
            label=None
        )
    # ========== Add These Lines ========== #
    # Customize the right y-axis to make ticks lighter
    #ax2.spines["right"].set_color("lightgray")  # Lighten the spine (axis line)
    #ax2.tick_params(
    #    axis="y",          # Target the y-axis
    #    which="both",      # Affect both major and minor ticks
    #    colors="lightgray",# Color of the ticks
    #    labelcolor="gray"  # Color of the tick labels
    #)
    # ===================================== #
    # Add titles and labels
    ax1.set_xlabel("Year", fontsize=18)
    ax1.set_ylabel("Precipitation Anomaly (mm/day)" if VARIABLE == "pr" else "Temperature Anomaly (°C)", fontsize=18)
    ll = ax1.legend(loc="upper left", title="", frameon=False, fontsize=18)
    ll.set_zorder(5)
    ax1.grid(False)  # This removes grid lines from the plot

    # Mirror the y-axis on the right side
    #ax2.set_ylabel("Precipitation Anomaly (mm/day)" if VARIABLE == "pr" else "Temperature Anomaly (°C)", fontsize=12)
    #ax2.set_ylim(ax1.get_ylim())  # Ensure the y-axis limits are the same as ax1
    #ax2.grid(False)  # This removes grid lines from the plot
    # Remove upper and right spines
    sns.despine()
    ax1.set_ylim(-0.3, 0.5)
    # Add xticks and yticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(output_path, dpi=300)
    plt.show()

    print("All experiments plotted together.")

def compute_ensemble_maps():
    """Compute ensemble mean maps for SSPs (2070-2099) relative to climatology."""
    ensemble_maps = {}

    # Load historical climatology for all models
    historical_climatology = {}
    for model in isimip_models:
        climatology_file = os.path.join(OUTPUT_DIR, f"{model}_historical_climatology.nc")
        if os.path.exists(climatology_file):
            historical_climatology[model] = xr.open_dataset(climatology_file)[VARIABLE]

    # Process each SSP scenario
    for experiment in ["ssp126", "ssp245", "ssp370", "ssp585"]:
        model_anomalies = []
        
        for model in isimip_models:
            # Load future data (2070-2099)
            regional_file = os.path.join(OUTPUT_DIR, f"{model}_{experiment}_regional.nc")
            if not os.path.exists(regional_file):
                print(f"Skipping {model}_{experiment}: Regional file not found.")
                continue
            
            ds_future = xr.open_dataset(regional_file)
            if "time" not in ds_future.coords:
                print(f"Skipping {model}_{experiment}: Time coordinate missing.")
                continue
            
            # Select future period and compute mean
            ds_future = ds_future.sel(time=slice(str(FUTURE_PERIOD_START), str(FUTURE_PERIOD_END)))
            future_mean = ds_future.mean(dim="time", skipna=True)[VARIABLE]
            
            # Subtract historical climatology (same model)
            if model in historical_climatology:
                anomaly = future_mean - historical_climatology[model]
                model_anomalies.append(anomaly)
        
        if model_anomalies:
            # Compute ensemble mean across models
            ensemble_mean = xr.concat(model_anomalies, dim="model").mean(dim="model")
            ensemble_maps[experiment] = ensemble_mean
    
    return ensemble_maps

import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.io.shapereader import Reader, natural_earth
from shapely.geometry import box
from matplotlib.colors import BoundaryNorm

def add_country_labels(ax, lat_range, lon_range):
    """Adds country labels for countries within the specified latitude and longitude range."""
    map_bbox = box(lon_range[0], lat_range[0], lon_range[1], lat_range[1])
    shapefile_path = natural_earth(
        category="cultural", name="admin_0_countries", resolution="110m"
    )

    for country in Reader(shapefile_path).records():
        country_name = country.attributes["NAME"]
        country_geometry = country.geometry
        if map_bbox.intersects(country_geometry):  # Check if the country is within the map bounds
            centroid = country_geometry.centroid
            ax.text(
                centroid.x,
                centroid.y,
                country_name,
                transform=ccrs.PlateCarree(),
                fontsize=8,  # Adjust font size for readability
                color="black",
                ha="center",
                zorder=5,
                alpha=0.7
            )

def add_selected_country_labels(ax):
    """Adds labels only for Kazakhstan, Uzbekistan, Turkmenistan, Tajikistan, and Kyrgyzstan."""
    shapefile_path = natural_earth(
        category="cultural", name="admin_0_countries", resolution="110m"
    )
    
    selected_countries = {"Kazakhstan", "Uzbekistan", "Turkmenistan", "Tajikistan", "Kyrgyzstan"}

    for country in Reader(shapefile_path).records():
        country_name = country.attributes["NAME"]
        if country_name in selected_countries:
            centroid = country.geometry.centroid
            ax.text(
                centroid.x,
                centroid.y,
                country_name,
                transform=ccrs.PlateCarree(),
                fontsize=8,  # Slightly larger for better readability
                color="black",
                fontweight=500,
                ha="center",
                zorder=5,
                alpha=0.8
            )

def plot_ensemble_maps(ensemble_maps, output_path):
    """Plot ensemble mean anomaly maps for SSP scenarios with optimized layout and colorbar placement, including country labels for selected countries."""
    fig = plt.figure(figsize=(10, 6))
    projection = ccrs.PlateCarree()

    # Define colormap and levels based on variable
    if VARIABLE == "pr":
        levels = np.arange(-0.6, 0.61, 0.05)
        cmap = plt.get_cmap("BrBG", len(levels) - 1)
        norm = BoundaryNorm(levels, ncolors=len(levels) - 1)
    else:
        levels = np.arange(-1, 8, 0.5)
        cmap = plt.get_cmap("Reds", len(levels) - 1)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)

    # Create subplots for each SSP scenario
    axes = []
    for i, experiment in enumerate(["ssp126", "ssp245", "ssp370", "ssp585"], 1):
        ax = fig.add_subplot(2, 2, i, projection=projection)
        if experiment not in ensemble_maps:
            continue

        data = ensemble_maps[experiment]
        if "time" in data.dims and VARIABLE == "pr":
            data = data.mean(dim="time") * 86400
        if "time" in data.dims:
            data = data.mean(dim="time")

        mesh = data.plot.pcolormesh(
            ax=ax,
            transform=projection,
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
        )

        ax.coastlines(linewidth=1, color="k")
        ax.add_feature(cfeature.BORDERS, linestyle="-")

        # Add country labels for selected countries only
        add_selected_country_labels(ax)

        ax.set_title(f"{experiment.upper()} (2070-2099)", fontsize=12)
        axes.append(ax)

    # Adjust layout and add colorbar
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.08, wspace=0.05, hspace=0.05)
    cbar_ax = fig.add_axes([0.05, 0.04, 0.9, 0.03])
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal", extend="both" if VARIABLE == "pr" else "both")
    cbar_label = "Precipitation Anomaly (mm/day)" if VARIABLE == "pr" else "Temperature Anomaly (°C)"
    cbar.set_label(cbar_label, fontsize=12)
    cbar.set_ticks(levels[::2] if VARIABLE == "pr" else levels)  # Reduce tick density for pr

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()





def main():
    print("Preprocessing datasets...")
    processed_files = preprocess_all_models_and_era5()

    print("Plotting ensemble anomalies...")
    plot_ensemble_anomalies(processed_files, OUTPUT_PATH)

    # Add these lines to compute and plot maps
    print("Computing ensemble maps...")
    ensemble_maps = compute_ensemble_maps()
    
    print("Plotting ensemble maps...")
    plot_ensemble_maps(ensemble_maps, MAP_OUTPUT_PATH)

if __name__ == "__main__":
    main()