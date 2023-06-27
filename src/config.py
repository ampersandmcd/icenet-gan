"""
Adapted from Tom Andersson's https://github.com/tom-andersson/icenet-paper/blob/main/icenet/config.py
"""
import os
import pandas as pd

"""
Defines globals used throughout the codebase.
"""

###############################################################################
### Folder structure naming system
###############################################################################

data_folder = "/data/hpcdata/users/anddon76/icenet-gan-data"
obs_data_folder = os.path.join(data_folder, "obs")
cmip6_data_folder = os.path.join(data_folder, "cmip6")
mask_data_folder = os.path.join(data_folder, "masks")
forecast_data_folder = os.path.join(data_folder, "forecasts")
network_dataset_folder = os.path.join(data_folder, "network_datasets")

repo_folder = "/users/anddon76/icenet/icenet-gan"
dataloader_config_folder = os.path.join(repo_folder, "dataloader_configs")
networks_folder = os.path.join(repo_folder, "trained_networks")
figure_folder = os.path.join(repo_folder, "figures")
video_folder = os.path.join(repo_folder, "videos")

results_folder = os.path.join(repo_folder, "results")
forecast_results_folder = os.path.join(results_folder, "forecast_results")
permute_and_predict_results_folder = os.path.join(results_folder, "permute_and_predict_results")
uncertainty_results_folder = os.path.join(results_folder, "uncertainty_results")

active_grid_cell_file_format = "active_grid_cell_mask_{}.npy"
land_mask_filename = "land_mask.npy"
region_mask_filename = "region_mask.npy"

###############################################################################
### Polar hole/missing months
###############################################################################

# Pre-defined polar hole radii (in number of 25km x 25km grid cells)
#   The polar hole radii were determined from Sections 2.1, 2.2, and 2.3 of
#   http://osisaf.met.no/docs/osisaf_cdop3_ss2_pum_sea-ice-conc-climate-data-record_v2p0.pdf
polarhole1_radius = 28
polarhole2_radius = 11
polarhole3_radius = 3

# Whether or not to mask out the 3rd polar hole mask from
# Nov 2005 to Dec 2015 with a radius of only 3 grid cells. Including it creates
# some complications when analysing performance on a validation set that
# overlaps with the 3rd polar hole period.
use_polarhole3 = False

polarhole1_fname = "polarhole1_mask.npy"
polarhole2_fname = "polarhole2_mask.npy"
polarhole3_fname = "polarhole3_mask.npy"

# Final month that each of the polar holes apply
# NOTE: 1st of the month chosen arbitrarily throughout as always working wit
#   monthly averages
polarhole1_final_date = pd.Timestamp("1987-06-01")  # 1987 June
polarhole2_final_date = pd.Timestamp("2005-10-01")  # 2005 Oct
polarhole3_final_date = pd.Timestamp("2015-12-01")  # 2015 Dec

missing_dates = [pd.Timestamp("1986-4-1"), pd.Timestamp("1986-5-1"),
                 pd.Timestamp("1986-6-1"), pd.Timestamp("1987-12-1")]

###############################################################################
### Weights and biases config (https://docs.wandb.ai/guides/track/advanced/environment-variables)
###############################################################################

# Get API key from https://wandb.ai/authorize
WANDB_API_KEY = "c9359bdb7a98988cb4d1b0a92098e2a8f6bda29a"
WANDB_USERNAME = "andrewmcdonald"
# Absolute path to store wandb generated files (folder must exist)
#   Note: user must have write access
WANDB_DIR = "/users/anddon76/icenet/icenet-gan/wandb"
# Absolute path to wandb config dir (
WANDB_CONFIG_DIR = "/users/anddon76/icenet/icenet-gan/wandb-config"
WANDB_CACHE_DIR = "/users/anddon76/icenet/icenet-gan/wandb-cache"

###############################################################################
### ECMWF details
###############################################################################

ECMWF_API_KEY = "f771fcaf1d5f5f2997d1bdff3e1ef011"
ECMWF_API_EMAIL = "arm99@cam.ac.uk"

###############################################################################
### Other environment variables
###############################################################################
IMAGEIO_FFMPEG_EXE = "/users/anddon76/bin/ffmpeg"
