import numpy as np
import pywt

TESS_SYSTEMATIC_PERIODS = {
    "orbital": 13.7,
    "momentum_dump": 3.125,
    "scattered_light": 1.0,
    "half_orbital": 6.85,
}

SYSTEMATIC_TOLERANCE_FRAC = 0.15
SYSTEMATIC_TOLERANCE_MAX = 0.5


MAX_CWT_POINTS = 3000

def run_wavelet(time: list, flux: list) -> dict:
    """
    Continuous wavelet transform producing a 2D time-period power map.
    Identifies TESS instrumental systematics for automatic rejection.
    """
    time_arr = np.array(time)
    flux_arr = np.array(flux)

    nan_mask = np.isnan(flux_arr)
    if nan_mask.any():
        flux_arr[nan_mask] = np.nanmedian(flux_arr)

    # Downsample before CWT to keep runtime reasonable (<10s)
    step = max(1, len(flux_arr) // MAX_CWT_POINTS)
    time_ds = time_arr[::step]
    flux_ds = flux_arr[::step]

    dt = float(np.median(np.diff(time_ds)))
    n_periods = 64
    periods = np.logspace(np.log10(0.1), np.log10(30.0), n_periods)
    central_freq = pywt.central_frequency("morl")
    scales = (central_freq * periods) / dt

    coefficients, _ = pywt.cwt(flux_ds, scales, "morl", sampling_period=dt)
    power = np.abs(coefficients) ** 2

    for i in range(n_periods):
        row_median = np.median(power[i])
        if row_median > 0:
            power[i] = power[i] / row_median

    subsample = max(1, len(time_ds) // 2000)
    time_sub = time_ds[::subsample].tolist()
    power_sub = power[:, ::subsample].tolist()

    integrated_power = power.mean(axis=1)
    systematic_hits = []
    for name, sys_period in TESS_SYSTEMATIC_PERIODS.items():
        tol = min(sys_period * SYSTEMATIC_TOLERANCE_FRAC, SYSTEMATIC_TOLERANCE_MAX)
        window = np.abs(periods - sys_period) < tol
        if window.any():
            local_power = integrated_power[window].max()
            global_max = integrated_power.max()
            if global_max > 0 and local_power > 0.6 * global_max:
                systematic_hits.append({
                    "name": name,
                    "period_days": round(sys_period, 3),
                    "relative_power": round(float(local_power / global_max), 3),
                })

    sys_period_values = [s["period_days"] for s in systematic_hits]
    dominant_idxs = np.argsort(integrated_power)[::-1]
    dominant_periods = []
    for idx in dominant_idxs:
        p = float(periods[idx])
        is_sys = any(
            abs(p - sp) / sp < SYSTEMATIC_TOLERANCE_FRAC
            for sp in sys_period_values
        )
        if not is_sys:
            dominant_periods.append(round(p, 3))
        if len(dominant_periods) == 3:
            break

    return {
        "time": time_sub,
        "periods": periods.tolist(),
        "power": power_sub,
        "tess_systematic_periods": systematic_hits,
        "dominant_periods": dominant_periods,
    }


_SYSTEMATIC_POWER_RATIO = 3.5
_MIN_EVENT_PERIOD_FRAC = 0.05


def event_matches_systematic(
    event_time_center: float, wavelet_result: dict,
    event_duration_hours: float = 0.0,
) -> str | None:
    if not wavelet_result or not wavelet_result.get("tess_systematic_periods"):
        return None

    time_arr = np.array(wavelet_result["time"])
    periods_arr = np.array(wavelet_result["periods"])
    power_arr = np.array(wavelet_result["power"])

    time_idx = int(np.argmin(np.abs(time_arr - event_time_center)))

    for systematic in wavelet_result["tess_systematic_periods"]:
        sys_period = systematic["period_days"]

        if event_duration_hours > 0:
            sys_hours = sys_period * 24.0
            if event_duration_hours < _MIN_EVENT_PERIOD_FRAC * sys_hours:
                continue

        period_idx = int(np.argmin(np.abs(periods_arr - sys_period)))

        if period_idx < len(power_arr) and time_idx < len(power_arr[period_idx]):
            local_power = float(power_arr[period_idx][time_idx])
            median_power = float(np.median(power_arr[period_idx]))
            if local_power > _SYSTEMATIC_POWER_RATIO * median_power:
                return systematic["name"]

    return None
