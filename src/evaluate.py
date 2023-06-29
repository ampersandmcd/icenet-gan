"""
Functions to evaluate IceNet forecasts.
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.metrics import f1_score
from dateutil.relativedelta import relativedelta
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


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


def ssim(preds: xr.DataArray, target: xr.DataArray, leadtimes_to_evaluate: list):
    """
    Compute structural similarity score between preds and target to evaluate image structure.
    Because this metric analyses holistic image structure, we do not mask it.
    Expects preds to be a 3-channel probability map and target to be a 1-channel SIC map.
    """
    scores = []
    for time in preds.time:
        for leadtime in leadtimes_to_evaluate:
            # can only evaluate one rgb image at a time
            p = preds.sel(time=time, leadtime=leadtime).squeeze()
            t = target.sel(time=time, leadtime=leadtime).squeeze()
            t, p = xr.broadcast(t, p)  # make target 3-channeled by repeating single channel
            score = structural_similarity(t, p, data_range=1.0, channel_axis=-1)
            scores.append(score)

    return np.mean(scores)


def psnr(preds: xr.DataArray, target: xr.DataArray, leadtimes_to_evaluate: list):
    """
    Compute peak-signal-to-noise ratio between preds and target to evaluate image structure.
    Because this metric analyses holistic image structure, we do not mask it.
    Expects preds to be a 3-channel probability map and target to be a 1-channel SIC map.
    """
    scores = []
    for time in preds.time:
        for leadtime in leadtimes_to_evaluate:
            # can only evaluate one rgb image at a time
            p = preds.sel(time=time, leadtime=leadtime).squeeze()
            t = target.sel(time=time, leadtime=leadtime).squeeze()
            t, p = xr.broadcast(t, p)  # make target 3-channeled by repeating single channel
            score = peak_signal_noise_ratio(t, p, data_range=1.0)
            scores.append(score)

    return np.mean(scores)


def rapsd_field(field, fft_method=np.fft, return_freq=True, d=1.0, normalize=False, **fft_kwargs):
    """
    From https://github.com/pySTEPS/pysteps/blob/master/pysteps/utils/spectral.py#L100
    Because this is the only function we use from pySTEPS, we copy it to avoid a 200mb install.
    """
    def compute_centred_coord_array(M, N):
        """
        From https://github.com/pySTEPS/pysteps/blob/master/pysteps/utils/arrays.py
        """
        if M % 2 == 1:
            s1 = np.s_[-int(M / 2) : int(M / 2) + 1]
        else:
            s1 = np.s_[-int(M / 2) : int(M / 2)]

        if N % 2 == 1:
            s2 = np.s_[-int(N / 2) : int(N / 2) + 1]
        else:
            s2 = np.s_[-int(N / 2) : int(N / 2)]

        YC, XC = np.ogrid[s1, s2]
        return YC, XC

    if len(field.shape) != 2:
        raise ValueError(
            f"{len(field.shape)} dimensions are found, but the number "
            "of dimensions should be 2"
        )

    if np.sum(np.isnan(field)) > 0:
        raise ValueError("input field should not contain nans")

    m, n = field.shape

    yc, xc = compute_centred_coord_array(m, n)
    r_grid = np.sqrt(xc * xc + yc * yc).round()
    l = max(field.shape[0], field.shape[1])

    if l % 2 == 1:
        r_range = np.arange(0, int(l / 2) + 1)
    else:
        r_range = np.arange(0, int(l / 2))

    if fft_method is not None:
        psd = fft_method.fftshift(fft_method.fft2(field, **fft_kwargs))
        psd = np.abs(psd) ** 2 / psd.size
    else:
        psd = field

    result = []
    for r in r_range:
        mask = r_grid == r
        psd_vals = psd[mask]
        result.append(np.mean(psd_vals))

    result = np.array(result)

    if normalize:
        result /= np.sum(result)

    if return_freq:
        freq = np.fft.fftfreq(l, d=d)
        freq = freq[r_range]
        return result, freq
    else:
        return result


def rapsd(preds: xr.DataArray, target: xr.DataArray, leadtimes_to_evaluate: list):
    """
    Compute radially-averaged power spectral density to evaluate image structure.
    Because this metric analyses holistic image structure, we do not mask it.
    Expects preds to be a 3-channel probability map and target to be a 1-channel SIC map.
    """
    preds_psd = {"no_ice": [], "marginal_ice": [], "full_ice": []}
    target_psd = []
    for time in preds.time:
        for leadtime in leadtimes_to_evaluate:
            for ice_class in ["no_ice", "marginal_ice", "full_ice"]:
                # can only evaluate one 2D image at a time
                p = preds.sel(time=time, leadtime=leadtime, ice_class=ice_class).squeeze()
                p_psd, f = rapsd_field(p)
                preds_psd[ice_class].append(p_psd)
            t = target.sel(time=time, leadtime=leadtime).squeeze()
            t_psd, f = rapsd_field(t)
            target_psd.append(t_psd)

    # make results into np arrays and take mean across forecasts
    for ice_class in ["no_ice", "marginal_ice", "full_ice"]:
        preds_psd[ice_class] = np.array(preds_psd[ice_class]).mean(axis=0)
    target_psd = np.array(target_psd).mean(axis=0)
    f =  np.array(f)

    # f will be the same for all images since all images are the same size
    return preds_psd, target_psd, f
    

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
