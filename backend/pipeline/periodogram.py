import numpy as np
from astropy.timeseries import BoxLeastSquares
from astropy import units as u
from scipy.signal import savgol_filter

_SHORT_PERIOD_BOUNDARY = 5.0
_N_SHORT = 10000
_N_LONG_MIN = 1000
_N_LONG_CAP = 100000
_MAX_CHART = 5000

_GAP_THRESHOLD_DAYS = 2.0
_SAVGOL_WINDOW_HOURS = 24.0
_TOP_BLS_CANDIDATES = 12
_PERIOD_DEDUP_FRAC = 0.02
_PERIOD_DEDUP_ABS = 0.08

# Event-based period search
_DIP_TYPES_FOR_PERIOD = frozenset({
    "transit",
    "exocomet",
    "asymmetric",
    "depth_anomaly",
    "non_periodic",
    "unknown",
})
# Pairwise d/k aliases: keep k small so noise does not prefer ~1 d super-harmonics.
_EVENT_PAIR_HARMONICS = 8
_MAX_PERIOD_CANDIDATES_TO_SCORE = 800


def _segment_indices(time_arr: np.ndarray) -> list[tuple[int, int]]:
    """Split at gaps > _GAP_THRESHOLD_DAYS (multi-sector safe)."""
    if len(time_arr) == 0:
        return []
    splits = [0]
    dt = np.diff(time_arr)
    for i, d in enumerate(dt):
        if d > _GAP_THRESHOLD_DAYS:
            splits.append(i + 1)
    splits.append(len(time_arr))
    return list(zip(splits[:-1], splits[1:]))


def _fill_nan_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    if n == 0:
        return y
    med = float(np.nanmedian(y))
    x = np.arange(n)
    m = np.isfinite(y)
    if m.sum() < 2:
        y[~m] = med
        return y
    y[~m] = np.interp(x[~m], x[m], y[m])
    return y


def _flatten_for_bls(time_arr: np.ndarray, flux_arr: np.ndarray) -> np.ndarray:
    """
    Gap-aware Savitzky–Golay detrending (~1 day window) to suppress stellar
    variability before BLS while preserving typical transit widths.
    """
    time_arr = np.asarray(time_arr, dtype=float)
    flux_arr = np.asarray(flux_arr, dtype=float)
    out = np.full_like(flux_arr, np.nan, dtype=float)

    for a, b in _segment_indices(time_arr):
        segment = flux_arr[a:b].copy()
        tseg = time_arr[a:b]
        nseg = len(segment)
        if nseg < 15:
            out[a:b] = segment
            continue

        segment = _fill_nan_1d(segment)
        if np.median(segment) != 0:
            segment = segment / (np.nanmedian(segment) + 1e-12)

        cadence_days = float(np.median(np.diff(tseg)))
        if not np.isfinite(cadence_days) or cadence_days <= 0:
            cadence_days = 1.0 / 24.0

        window_pts = int(_SAVGOL_WINDOW_HOURS / (cadence_days * 24.0))
        window_pts = max(5, min(window_pts, nseg - 1))
        if window_pts % 2 == 0:
            window_pts -= 1
        window_pts = max(5, min(window_pts, nseg - (1 - nseg % 2)))
        if window_pts < 5 or window_pts >= nseg:
            out[a:b] = segment
            continue

        try:
            trend = savgol_filter(segment, window_length=window_pts, polyorder=2, mode="interp")
            trend = np.where(np.abs(trend) < 1e-12, 1.0, trend)
            out[a:b] = segment / trend
        except Exception:
            out[a:b] = segment

    return out


def _build_bls_candidates(
    pg,
    power: np.ndarray,
    sde: np.ndarray,
    max_n: int = _TOP_BLS_CANDIDATES,
) -> list[dict]:
    """Top *max_n* distinct periods by SDE."""
    order = np.argsort(sde)[::-1]
    candidates: list[dict] = []
    seen_periods: list[float] = []

    for idx in order:
        idx = int(idx)
        p = float(pg.period[idx].value)
        if any(abs(p - sp) < max(_PERIOD_DEDUP_FRAC * p, _PERIOD_DEDUP_ABS) for sp in seen_periods):
            continue
        seen_periods.append(p)
        candidates.append({
            "period": p,
            "transit_time": float(pg.transit_time[idx].value),
            "power": float(power[idx]),
            "sde": float(sde[idx]),
            "transit_duration_hours": float(pg.duration[idx].to(u.hour).value),
            "source": "bls",
        })
        if len(candidates) >= max_n:
            break

    return candidates


def _inject_long_period_candidate(
    pg,
    power: np.ndarray,
    sde: np.ndarray,
    candidates: list[dict],
    min_p_days: float = 12.0,
) -> list[dict]:
    """
    Append the highest-SDE trial with P >= *min_p_days* if not already represented.
    Short-period aliases often dominate SDE rankings; this keeps a long-period
    peak available for partition / P_ref anchoring (e.g. K2-18 ~33 d).
    """
    if len(candidates) == 0:
        return candidates
    p_days = np.asarray(pg.period.to(u.day).value, dtype=float)
    mask = p_days >= float(min_p_days)
    if not np.any(mask):
        return candidates
    long_idx = np.flatnonzero(mask)
    idx = int(long_idx[int(np.argmax(sde[mask]))])
    p = float(p_days[idx])
    for c in candidates:
        cp = float(c["period"])
        if abs(p - cp) < max(0.02 * p, 0.08):
            return candidates
    return candidates + [{
        "period": p,
        "transit_time": float(pg.transit_time[idx].value),
        "power": float(power[idx]),
        "sde": float(sde[idx]),
        "transit_duration_hours": float(pg.duration[idx].to(u.hour).value),
        "source": "bls",
    }]


def run_bls(time: list, flux: list) -> dict:
    """
    Adaptive-grid BLS on flattened flux: dense at long periods, SDE scoring,
    and multiple candidate peaks for ensemble validation.
    """
    time_arr = np.array(time, dtype=float)
    flux_arr = np.array(flux, dtype=float)

    mask = np.isfinite(flux_arr) & np.isfinite(time_arr)
    time_arr = time_arr[mask]
    flux_arr = flux_arr[mask]

    if len(time_arr) < 100:
        return _empty_result()

    flux_for_bls = _flatten_for_bls(time_arr, flux_arr)

    baseline_days = float(time_arr[-1] - time_arr[0])
    trial_durations = np.array([0.05, 0.1, 0.15, 0.2, 0.25])
    max_dur = float(trial_durations.max())
    min_period = max(0.3, max_dur * 1.05)
    max_period = max(baseline_days / 2.0, 1.0, min_period * 1.1)
    if max_period <= min_period:
        return _empty_result()

    f_min = 1.0 / max_period
    f_max = 1.0 / min_period
    dur_qty = trial_durations * u.day
    bls = BoxLeastSquares(time_arr * u.day, flux_for_bls)

    f_boundary = 1.0 / _SHORT_PERIOD_BOUNDARY

    if max_period > _SHORT_PERIOD_BOUNDARY and f_boundary > f_min:
        df_long = max_dur / (max_period * baseline_days)
        n_long = int(np.ceil((min(f_boundary, f_max) - f_min) / df_long))
        n_long = max(_N_LONG_MIN, min(n_long, _N_LONG_CAP))
        long_freqs = np.linspace(f_min, min(f_boundary, f_max), n_long, endpoint=False)
    else:
        long_freqs = np.array([])

    short_start = max(f_boundary, f_min) if len(long_freqs) else f_min
    short_freqs = np.linspace(short_start, f_max, _N_SHORT)

    coarse_freqs = np.concatenate([long_freqs, short_freqs]) if len(long_freqs) else short_freqs
    coarse_periods = 1.0 / coarse_freqs

    pg = bls.power(coarse_periods * u.day, duration=dur_qty)
    power = np.asarray(pg.power, dtype=float)

    p_mean = float(np.nanmean(power))
    p_std = float(np.nanstd(power)) + 1e-10
    sde = (power - p_mean) / p_std

    best_idx = int(np.argmax(power))
    best_period = float(pg.period[best_idx].value)
    best_power = float(power[best_idx])
    best_sde = float(sde[best_idx])

    stats = bls.compute_stats(
        pg.period[best_idx],
        pg.duration[best_idx],
        pg.transit_time[best_idx],
    )

    candidates = _build_bls_candidates(pg, power, sde, _TOP_BLS_CANDIDATES)
    candidates = _inject_long_period_candidate(pg, power, sde, candidates)

    chart_p = pg.period.value
    chart_pow = power
    sort_idx = np.argsort(chart_p)
    chart_p = chart_p[sort_idx]
    chart_pow = chart_pow[sort_idx]
    if len(chart_p) > _MAX_CHART:
        step = len(chart_p) // _MAX_CHART
        chart_p = chart_p[::step]
        chart_pow = chart_pow[::step]

    return {
        "best_period_days": best_period,
        "best_power": best_power,
        "best_sde": best_sde,
        "transit_duration_hours": float(pg.duration[best_idx].to(u.hour).value),
        "transit_time": float(pg.transit_time[best_idx].value),
        "depth_ppm": float(stats["depth"][0] * 1e6),
        "periods": chart_p.tolist(),
        "powers": chart_pow.tolist(),
        "candidates": candidates,
    }


def _phase_fold_score(times: np.ndarray, period: float, t0: float, tol: float = 0.10) -> int:
    """Count events within ±tol phase of transit (two-sided wrap)."""
    if period <= 0:
        return 0
    n = 0
    for t in times:
        ph = ((float(t) - t0) % period) / period
        if ph < tol or ph > (1.0 - tol):
            n += 1
    return n


def find_period_from_events(
    events: list[dict],
    time_min: float,
    time_max: float,
    min_period: float | None = None,
    max_period: float | None = None,
) -> dict | None:
    """
    Infer a trial period from pairwise time separations of dip-like events.
    Returns a synthetic high-SDE candidate for ensemble partition when BLS is weak.
    """
    times = sorted({
        float(e["time_center"])
        for e in events
        if e.get("event_type") in _DIP_TYPES_FOR_PERIOD
    })
    if len(times) < 2:
        return None

    baseline = float(time_max - time_min)
    if baseline <= 0:
        return None

    # Do not tie min period to a fraction of baseline — that excludes genuine
    # ~month-long planets on multi-sector light curves.
    min_p = min_period if min_period is not None else 0.25
    max_p = max_period if max_period is not None else max(baseline / 2.0, min_p * 1.1)
    if max_p <= min_p:
        return None

    # Only use pairwise separations shorter than a gap threshold so multi-sector
    # baselines do not seed spurious long-period harmonics.
    _max_pair_for_harmonics = min(120.0, max(30.0, 0.25 * baseline))

    cand_periods: set[float] = set()
    for i, ti in enumerate(times):
        for tj in times[i + 1 :]:
            d = abs(tj - ti)
            if d < 0.05 or d > _max_pair_for_harmonics:
                continue
            for k in range(1, _EVENT_PAIR_HARMONICS + 1):
                p = d / k
                if min_p <= p <= max_p:
                    cand_periods.add(p)

    if not cand_periods:
        return None

    period_list = sorted(cand_periods)
    if len(period_list) > _MAX_PERIOD_CANDIDATES_TO_SCORE:
        step = max(1, len(period_list) // _MAX_PERIOD_CANDIDATES_TO_SCORE)
        period_list = period_list[::step][: _MAX_PERIOD_CANDIDATES_TO_SCORE]

    times_arr = np.array(times, dtype=float)

    pair_diffs: list[float] = []
    for i, ti in enumerate(times):
        for tj in times[i + 1 :]:
            d = abs(tj - ti)
            if d > 0.05:
                pair_diffs.append(d)

    if not pair_diffs:
        return None

    # Ignore cross-sector gaps when anchoring — prefer shortest plausible spacing.
    short_diffs = [
        d for d in pair_diffs
        if d < min(120.0, max(30.0, 0.25 * baseline))
    ]
    p_anchor = min(short_diffs) if short_diffs else min(pair_diffs)
    p_anchor = max(p_anchor, min_p * 0.5)

    # Reward longer periods (closer to fundamental) when hit counts tie — breaks
    # spurious super-harmonics (~few days) that align scattered dips by chance.
    _LOG_P_COEF = 0.9

    best_p = None
    best_t0 = None
    best_eff = -1e18
    best_n = 0

    for p in period_list:
        for t0 in times:
            n_hit = _phase_fold_score(times_arr, p, t0, tol=0.10)
            eff = float(n_hit) + _LOG_P_COEF * np.log(max(p / p_anchor, 1e-6))
            if best_p is None:
                best_eff, best_n, best_p, best_t0 = eff, n_hit, p, t0
            elif eff > best_eff + 1e-12:
                best_eff, best_n, best_p, best_t0 = eff, n_hit, p, t0
            elif abs(eff - best_eff) <= 1e-12 and p < best_p:
                best_eff, best_n, best_p, best_t0 = eff, n_hit, p, t0

    if best_p is None or best_t0 is None or best_n < 2:
        return None

    return {
        "period": float(best_p),
        "transit_time": float(best_t0),
        "power": 0.0,
        "sde": 8.0,
        "transit_duration_hours": 0.0,
        "source": "events",
        "n_phase_matched": best_n,
    }


def phase_fold(time: list, flux: list, period: float, t0: float) -> dict:
    time_arr = np.array(time)
    flux_arr = np.array(flux)

    phase = ((time_arr - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    sort_idx = np.argsort(phase)

    return {
        "phase": phase[sort_idx].tolist(),
        "flux": flux_arr[sort_idx].tolist(),
    }


def _empty_result():
    return {
        "best_period_days": 0,
        "best_power": 0,
        "best_sde": 0.0,
        "transit_duration_hours": 0,
        "transit_time": 0,
        "depth_ppm": 0,
        "periods": [],
        "powers": [],
        "candidates": [],
    }
