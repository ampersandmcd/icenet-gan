"""
Functions to evaluate IceNet forecasts.
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.metrics import f1_score
from dateutil.relativedelta import relativedelta


def binary_accuracy(preds: xr.DataArray, target: xr.DataArray, mask: xr.DataArray, leadtimes_to_evaluate: list):
    """
    Compute binary accuracy between preds and target, weighted by mask.
    """
    if len(preds.ice_class) == 1:
        # sic value
        preds_bin = xr.where(preds > 0.15, 1, 0).sel(leadtime=leadtimes_to_evaluate).values
    else:
        # ternary classification
        preds_bin = preds.reduce(np.argmax, dim="ice_class", keepdims=True)
        preds_bin = xr.where(preds_bin > 0, 1, 0).sel(leadtime=leadtimes_to_evaluate).values

    target_bin = xr.where(target > 0.15, 1, 0).sel(leadtime=leadtimes_to_evaluate).values
    mask = mask.sel(leadtime=leadtimes_to_evaluate).values

    score = ((preds_bin == target_bin) * mask).sum()
    total = mask.sum()
    return score / total


def binary_f1(preds: xr.DataArray, target: xr.DataArray, mask: xr.DataArray, leadtimes_to_evaluate: list):
    """
    Compute binary f1 score between preds and target, weighted by mask.
    """
    if len(preds.ice_class) == 1:
        # sic value
        preds_bin = xr.where(preds > 0.15, 1, 0).sel(leadtime=leadtimes_to_evaluate).values
    else:
        # ternary classification
        preds_bin = preds.reduce(np.argmax, dim="ice_class", keepdims=True)
        preds_bin = xr.where(preds_bin > 0, 1, 0).sel(leadtime=leadtimes_to_evaluate).values

    target_bin = xr.where(target > 0.15, 1, 0).sel(leadtime=leadtimes_to_evaluate).values
    mask = mask.sel(leadtime=leadtimes_to_evaluate).values

    return f1_score(y_true=target_bin.ravel(), y_pred=preds_bin.ravel(), sample_weight=mask.ravel())


def ternary_accuracy(preds: xr.DataArray, target: xr.DataArray, mask: xr.DataArray, leadtimes_to_evaluate: list):
    """
    Compute ternary accuracy between preds and target, weighted by mask.
    """
    if len(preds.ice_class) == 1:
        # sic value
        preds_ter = xr.where(preds > 0.15, 1, 0).sel(leadtime=leadtimes_to_evaluate).values
        preds_ter += xr.where(preds > 0.85, 1, 0).sel(leadtime=leadtimes_to_evaluate).values
    else:
        # ternary classification
        preds_ter = preds.reduce(np.argmax, dim="ice_class", keepdims=True).sel(leadtime=leadtimes_to_evaluate).values

    target_ter = xr.where(target > 0.15, 1, 0).sel(leadtime=leadtimes_to_evaluate).values
    target_ter += xr.where(target > 0.85, 1, 0).sel(leadtime=leadtimes_to_evaluate).values
    mask = mask.sel(leadtime=leadtimes_to_evaluate).values

    score = ((preds_ter == target_ter) * mask).sum()
    total = mask.sum()
    return score / total


def ternary_f1(preds: xr.DataArray, target: xr.DataArray, mask: xr.DataArray, leadtimes_to_evaluate: list):
    """
    Compute ternary macro-f1 score between preds and target, weighted by mask.
    """
    if len(preds.ice_class) == 1:
        # sic value
        preds_ter = xr.where(preds > 0.15, 1, 0).sel(leadtime=leadtimes_to_evaluate).values
        preds_ter += xr.where(preds > 0.85, 1, 0).sel(leadtime=leadtimes_to_evaluate).values
    else:
        # ternary classification
        preds_ter = preds.reduce(np.argmax, dim="ice_class", keepdims=True).sel(leadtime=leadtimes_to_evaluate).values

    target_ter = xr.where(target > 0.15, 1, 0).sel(leadtime=leadtimes_to_evaluate).values
    target_ter += xr.where(target > 0.85, 1, 0).sel(leadtime=leadtimes_to_evaluate).values
    mask = mask.sel(leadtime=leadtimes_to_evaluate).values

    return f1_score(y_true=target_ter.ravel(), y_pred=preds_ter.ravel(), average="macro", sample_weight=mask.ravel())


def sie_error(preds: xr.DataArray, target: xr.DataArray, mask: xr.DataArray, leadtimes_to_evaluate: list):
    """
    Compute sea ice extent error between preds and target, weighted by mask.
    """
    if len(preds.ice_class) == 1:
        # sic value
        preds_bin = xr.where(preds > 0.15, 1, 0).sel(leadtime=leadtimes_to_evaluate)
    else:
        # ternary classification
        preds_bin = preds.reduce(np.argmax, dim="ice_class", keepdims=True)
        preds_bin = xr.where(preds_bin > 0, 1, 0).sel(leadtime=leadtimes_to_evaluate)

    target_bin = xr.where(target > 0.15, 1, 0).sel(leadtime=leadtimes_to_evaluate)
    mask = mask.sel(leadtime=leadtimes_to_evaluate)

    preds_sie = (preds_bin * mask.values).sum(("xc", "yc"))
    target_sie = (target_bin * mask.values).sum(("xc", "yc"))
    # each pixel is 25km x 25km
    mean_sie_error = (preds_sie - target_sie).mean(("time", "leadtime")) * (25**2)  

    return mean_sie_error.item()


def visualise_forecast(forecast_dict, forecast_mask, date, diff=False, true_forecast=None, ternerise=True):

    leadtimes = list(forecast_dict.values())[0].leadtime.values
    fig, ax = plt.subplots(len(forecast_dict), len(leadtimes), 
                           figsize=(6*len(leadtimes), 6*len(forecast_dict)))
    
    for i, (forecast_name, forecast) in enumerate(forecast_dict.items()):
        for leadtime in leadtimes:

            pred = forecast.sel(time=date, leadtime=leadtime)
            mask = forecast_mask.sel(time=date, leadtime=leadtime)
            if len(pred.ice_class) == 1 and ternerise:
                # sic value
                pred_ter = xr.where(pred > 0.15, 1, 0)
                pred_ter += xr.where(pred > 0.85, 1, 0)
                pred = pred_ter
            elif ternerise:
                # ternary classification
                pred = pred.reduce(np.argmax, dim="ice_class", keepdims=True)
            pred = pred * mask.values

            if diff:
                # plot the difference between the forecast and the truth
                true = true_forecast.sel(time=date, leadtime=leadtime)
                true_ter = xr.where(true > 0.15, 1, 0)
                true_ter += xr.where(true > 0.85, 1, 0)
                true_ter = true_ter.squeeze()
                to_plot = pred - true_ter
                vmin, center = -2, 0
            else:
                # plot the forecast
                to_plot = pred
                vmin, center = 0, False

            to_plot = to_plot.squeeze()
            if ternerise:
                xr.plot.contourf(to_plot, ax=ax[i, leadtime-1], add_colorbar=False, vmin=vmin, center=center)
            else:
                xr.plot.imshow(to_plot, ax=ax[i, leadtime-1], add_colorbar=False)

            ax[i, leadtime-1].set_title(f"{forecast_name} Prediction {date + relativedelta(months=leadtime)}"
                                        f"\nInitialised {date}")
            ax[i, leadtime-1].set_xlabel("Eastings [km]")
            ax[i, leadtime-1].set_ylabel("Northings [km]")
            
    plt.tight_layout()
    plt.show()
