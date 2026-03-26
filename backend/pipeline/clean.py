import numpy as np
from scipy.signal import medfilt


def normalize_flux(flux: list) -> list:
    arr = np.array(flux)
    return (arr / np.nanmedian(arr)).tolist()


def detrend_flux(time: list, flux: list, window_hours: float = 12.0) -> list:
    """
    Remove long-term stellar variability with a sliding median filter.
    12h window preserves transit signals (1-6h) while removing slow variability.
    """
    time_arr = np.array(time)
    flux_arr = np.array(flux)

    # Fill NaN before medfilt — scipy medfilt propagates NaN into neighbouring windows
    nan_mask = np.isnan(flux_arr)
    if nan_mask.any():
        flux_arr = flux_arr.copy()
        flux_arr[nan_mask] = np.nanmedian(flux_arr)

    cadence_minutes = np.median(np.diff(time_arr)) * 24 * 60
    window_points = int(window_hours * 60 / cadence_minutes)
    window_points = min(window_points, len(flux_arr) - 1)
    if window_points % 2 == 0:
        window_points += 1
    window_points = max(window_points, 3)

    trend = medfilt(flux_arr, kernel_size=window_points)
    trend[np.isnan(trend) | (trend == 0)] = 1.0
    detrended = flux_arr / trend

    return detrended.tolist()


def remove_outliers(flux: list, sigma: float = 7.0) -> tuple[list, list]:
    """
    Remove extreme outliers (cosmic rays, readout glitches) at 7-sigma.
    High sigma avoids clipping real astrophysical events.
    """
    arr = np.array(flux)
    median = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - median))
    std_estimate = mad * 1.4826

    mask = np.abs(arr - median) < sigma * std_estimate
    cleaned = arr.copy()
    cleaned[~mask] = np.nan

    return cleaned.tolist(), mask.tolist()
