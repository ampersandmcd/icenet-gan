"""
Adapted from Tom Andersson's https://github.com/tom-andersson/icenet-paper/blob/main/icenet/preproc_icenet_data.py
Modified to be modular and importable.
Modified to forego CMIP6 by default as this project does not use CMIP6.
"""
import os
from src.utils import IceNetDataPreProcessor
from src import config


def preproc_icenet_data():
    '''
    Use the IceNetDataPreProcessor class to normalise and save climate variables in
    NumPy format for training IceNet.

    Raw data is loaded from the data/obs/ and data/cmip6/ folders and the
    processed data is saved in data/network_datasets/<dataset_name>/.

    Normalisation parameters computed over the observational training data are
    stored in a JSON file at data/network_datasets/<dataset_name>/norm_params.json
    so that they are only computed once. Similarly, monthly climatology fields
    used for computing anomaly fields are saved next to the raw NetCDF files so that
    climatologies are only computed once for each variable.

    Note that producing the SIC linear trend forecast fields over all the climate
    simulations can take on the order of an hour to compute.
    '''

    # Path to the dataloader configuration JSON file
    dataloader_config_fpath = os.path.join(config.dataloader_config_folder, "2023_06_24_1235_icenet_gan.json")
    
    # Whether to preprocess observational data
    preproc_obs_data = True

    # False: Normalise data to have mean=0 and standard deviation=1 or min=-1 and max=+1.
    # True: Normalise data to have min=-1 and max=+1.
    minmax = False

    # If 'anom' is True, compute and process anomaly from the climatology over
    #   the training years. If 'abs' is True, process the absolute data.
    preproc_vars = {
        'siconca': {'anom': False, 'abs': True, 'linear_trend': True},
        'tas': {'anom': True, 'abs': False},
        'ta500': {'anom': True, 'abs': False},
        'tos': {'anom': True, 'abs': False},
        'rsds': {'anom': True, 'abs': False},
        'rsus': {'anom': True, 'abs': False},
        'psl': {'anom': True, 'abs': False},
        'zg500': {'anom': True, 'abs': False},
        'zg250': {'anom': True, 'abs': False},
        'ua10': {'anom': False, 'abs': True},
        'uas': {'anom': False, 'abs': True},
        'vas': {'anom': False, 'abs': True},
        'land': {'metadata': True, 'include': True},
        'circmonth': {'metadata': True, 'include': True}
    }

    n_linear_years = 35  # Number of past years to used in the linear trend projections
    verbose_level = 2  # for debugging

    # note that preprocessing is executed upon object construction
    dpp = IceNetDataPreProcessor(
        dataloader_config_fpath=dataloader_config_fpath,
        preproc_vars=preproc_vars,
        n_linear_years=n_linear_years,
        minmax=minmax,
        verbose_level=verbose_level,
        preproc_obs_data=preproc_obs_data,
    )


if __name__ == "__main__":
    preproc_icenet_data()
