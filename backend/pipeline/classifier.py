import numpy as np
from enum import Enum
from .wavelet import event_matches_systematic
from .centroid import centroid_shift_during_event
from .periodogram import find_period_from_events


class EventType(str, Enum):
    TRANSIT = "transit"
    ASYMMETRIC = "asymmetric"
    DEPTH_ANOMALY = "depth_anomaly"
    NON_PERIODIC = "non_periodic"
    EXOCOMET = "exocomet"
    STELLAR_FLARE = "stellar_flare"
    STELLAR_SPOT = "stellar_spot"
    ECLIPSING_BINARY = "eclipsing_binary"
    STELLAR_VARIABILITY = "stellar_variability"
    SYSTEMATIC = "systematic"
    CONTAMINATION = "contamination"
    UNKNOWN = "unknown"


CENTROID_CONTAMINATION_THRESHOLD = 10.0
CENTROID_CAUTION_THRESHOLD = 3.0

MAX_SYSTEMATIC_DEPTH_PPM = 5000

MAX_EVENTS = 20
MIN_REGION_POINTS = 5
MIN_DURATION_HOURS = 0.25

_ASYMMETRY_CAP = 20.0
_MIN_SLOPE_POINTS = 6
_BLS_PHASE_TOLERANCE = 0.10
_BLS_MIN_SDE = 6.0
_BLS_MIN_PARTITION_SDE = 5.0
_BLS_REF_POOL_MIN_SDE = 5.0
_BLS_GROUP_MIN = 2
_BLS_DEPTH_CV_MAX = 0.35
_BLS_DUR_CV_MAX = 0.50
_BLS_REF_DISTANCE_SIGMA = 0.14
_PARTITION_LONG_BASELINE_MIN_D = 55.0
_PARTITION_LONG_PERIOD_MIN_D = 12.0


def _phase_tol_for_period(period: float) -> float:
    """Widen phase gate for long periods (~±3.5 d in time for P~33 d)."""
    if period >= 18.0:
        return float(min(0.14, max(_BLS_PHASE_TOLERANCE, 3.5 / period)))
    return _BLS_PHASE_TOLERANCE


def _refine_transit_epoch(
    matched_events: list[dict],
    period: float,
    t0: float,
) -> float:
    """Shift *t0* so dip times align to a common transit epoch (linear bias fix)."""
    if period <= 0 or len(matched_events) < 1:
        return t0
    t0 = float(t0)
    residuals: list[float] = []
    for e in matched_events:
        t = float(e["time_center"])
        n = float(np.round((t - t0) / period))
        r = t - t0 - n * period
        r = (r + period / 2.0) % period - period / 2.0
        residuals.append(r)
    return float(t0 + float(np.mean(residuals)))


def _partition_cv_ok(
    matched: list[dict],
    dur_cap: float,
    period: float = 0.0,
) -> bool:
    if len(matched) < _BLS_GROUP_MIN:
        return False
    depths = np.array([e["depth_ppm"] for e in matched], dtype=float)
    durations = np.array([e["duration_hours"] for e in matched], dtype=float)
    depth_cv = float(np.std(depths) / (np.mean(np.abs(depths)) + 1e-8))
    dur_cv = float(np.std(durations) / (np.mean(durations) + 1e-8))
    depth_cap = _BLS_DEPTH_CV_MAX
    if period >= 18.0 and len(matched) >= 4:
        depth_cap = max(depth_cap, 0.42)
    if depth_cv > depth_cap:
        return False
    if dur_cv > dur_cap:
        return False
    return True


def _rematch_astro_to_period(
    astro_events: list[dict],
    period: float,
    t0: float,
    tol: float,
) -> tuple[list[dict], list[dict]]:
    matched: list[dict] = []
    unmatched: list[dict] = []
    for e in astro_events:
        if e["event_type"] == _FLARE_TYPE:
            continue
        if _phase_consistent(e["time_center"], period, t0, tol=tol):
            matched.append(e)
        else:
            unmatched.append(e)
    return matched, unmatched


def _refine_partition_epoch(
    astro_events: list[dict],
    partition: dict,
) -> dict:
    """
    Refine BLS *t0* using matched dips and rematch all events with a long-period
    phase tolerance. Reduces false 'stellar variability' when grid *t0* is offset.
    """
    period = float(partition["period"])
    t0_cur = float(partition["t0"])
    tol = _phase_tol_for_period(period)
    dur_cap = _BLS_DUR_CV_MAX
    if period >= _PARTITION_LONG_PERIOD_MIN_D:
        dur_cap = max(dur_cap, 0.55)

    matched = list(partition["matched"])
    if len(matched) < _BLS_GROUP_MIN:
        return partition

    for _ in range(5):
        t0_new = _refine_transit_epoch(matched, period, t0_cur)
        new_m, _ = _rematch_astro_to_period(astro_events, period, t0_new, tol)
        if len(new_m) < _BLS_GROUP_MIN or not _partition_cv_ok(new_m, dur_cap, period):
            break
        matched = new_m
        if abs(t0_new - t0_cur) < 1e-5:
            t0_cur = t0_new
            break
        t0_cur = t0_new

    matched, unmatched = _rematch_astro_to_period(astro_events, period, t0_cur, tol)
    if len(matched) < _BLS_GROUP_MIN or not _partition_cv_ok(matched, dur_cap, period):
        return partition

    depths = np.array([e["depth_ppm"] for e in matched], dtype=float)
    durations = np.array([e["duration_hours"] for e in matched], dtype=float)
    return {
        "period": period,
        "t0": t0_cur,
        "matched": matched,
        "unmatched": unmatched,
        "n_transits": len(matched),
        "mean_depth": float(np.mean(depths)),
        "mean_dur": float(np.mean(durations)),
    }


def _robust_depth_ppm(flux_arr: np.ndarray) -> float:
    """Mean of the bottom 20% of flux values — far more robust than nanmin."""
    clean = flux_arr[~np.isnan(flux_arr)]
    if len(clean) == 0:
        return 0.0
    n_bottom = max(2, len(clean) // 5)
    bottom = np.sort(clean)[:n_bottom]
    return float((1.0 - np.mean(bottom)) * 1e6)


def _phase_consistent(
    time_center: float,
    period: float,
    t0: float,
    tol: float = _BLS_PHASE_TOLERANCE,
) -> bool:
    """True when *time_center* is within ±*tol* in phase of transit at (*t0*, *period*)."""
    if period <= 0 or t0 <= 0:
        return False
    phase = ((time_center - t0) % period) / period
    return phase < tol or phase > (1.0 - tol)


def _matching_trusted_candidate(
    time_center: float,
    bls_result: dict | None,
) -> dict | None:
    """
    Among BLS candidates with SDE >= threshold, return the highest-SDE candidate
    whose phase matches *time_center*. Falls back to the global best peak if trusted.
    """
    if bls_result is None:
        return None

    best: dict | None = None
    best_sde = -1e9

    for c in bls_result.get("candidates", []):
        if (c.get("sde") or 0) < _BLS_MIN_SDE:
            continue
        p = c.get("period") or 0
        t0 = c.get("transit_time") or 0
        if p <= 0 or t0 <= 0:
            continue
        if _phase_consistent(time_center, p, t0):
            sde = float(c.get("sde", 0))
            if sde > best_sde:
                best_sde = sde
                best = c

    if best is not None:
        return best

    if (bls_result.get("best_sde") or 0) >= _BLS_MIN_SDE:
        p = bls_result.get("best_period_days", 0) or 0
        t0 = bls_result.get("transit_time", 0) or 0
        if p > 0 and t0 > 0 and _phase_consistent(time_center, p, t0):
            return {
                "period": p,
                "transit_time": t0,
                "sde": float(bls_result.get("best_sde", 0)),
                "power": float(bls_result.get("best_power", 0)),
                "source": "bls",
            }
    return None


def _bls_phase_consistent(time_center: float, bls_result: dict | None) -> bool:
    """True when a trusted BLS peak predicts *time_center* near transit phase."""
    return _matching_trusted_candidate(time_center, bls_result) is not None


def find_dip_events(
    time: list,
    flux: list,
    scores: list,
    wavelet_result: dict,
    centroid_result: dict,
    bls_result: dict | None = None,
    threshold: float = 0.5,
) -> list[dict]:
    time_arr = np.array(time)
    flux_arr = np.array(flux)
    score_arr = np.array(scores)

    above = score_arr > threshold
    events = []

    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1

            region_time = time_arr[i:j]
            region_flux = flux_arr[i:j]
            region_score = score_arr[i:j]

            duration_h = float((region_time[-1] - region_time[0]) * 24) if len(region_time) > 1 else 0
            if len(region_time) < MIN_REGION_POINTS or duration_h < MIN_DURATION_HOURS:
                i = j + 1
                continue

            time_center = float(region_time[np.argmin(region_flux)])
            duration_hours = float((region_time[-1] - region_time[0]) * 24)
            depth_ppm = _robust_depth_ppm(region_flux)
            event_score = float(region_score.max())

            bls_match = _bls_phase_consistent(time_center, bls_result)

            systematic_name = event_matches_systematic(
                time_center, wavelet_result, event_duration_hours=duration_hours,
            )
            if systematic_name and abs(depth_ppm) < MAX_SYSTEMATIC_DEPTH_PPM and not bls_match:
                events.append({
                    "time_center": time_center,
                    "duration_hours": round(duration_hours, 2),
                    "depth_ppm": round(depth_ppm, 1),
                    "anomaly_score": round(event_score, 3),
                    "event_type": EventType.SYSTEMATIC.value,
                    "confidence": 0.85,
                    "notes": f"Wavelet power at TESS {systematic_name} period. Not astrophysical.",
                    "centroid_shift_arcsec": -1.0,
                    "systematic_match": systematic_name,
                })
                i = j + 1
                continue

            shift_arcsec = centroid_shift_during_event(time_center, duration_hours, centroid_result)
            if shift_arcsec > CENTROID_CONTAMINATION_THRESHOLD:
                events.append({
                    "time_center": time_center,
                    "duration_hours": round(duration_hours, 2),
                    "depth_ppm": round(depth_ppm, 1),
                    "anomaly_score": round(event_score, 3),
                    "event_type": EventType.CONTAMINATION.value,
                    "confidence": 0.82,
                    "notes": f"Centroid shifts {shift_arcsec:.1f} arcsec — background source.",
                    "centroid_shift_arcsec": shift_arcsec,
                    "systematic_match": None,
                })
                i = j + 1
                continue

            event_type, confidence, notes = classify_event(
                region_time, region_flux, region_score, bls_result,
            )

            if event_type == EventType.STELLAR_FLARE:
                time_center = float(region_time[np.argmax(region_flux)])

            if CENTROID_CAUTION_THRESHOLD < shift_arcsec <= CENTROID_CONTAMINATION_THRESHOLD:
                notes += f" Centroid shift {shift_arcsec:.1f} arcsec — borderline."
                confidence *= 0.8

            events.append({
                "time_center": time_center,
                "duration_hours": round(duration_hours, 2),
                "depth_ppm": round(depth_ppm, 1),
                "anomaly_score": round(event_score, 3),
                "event_type": event_type.value,
                "confidence": round(confidence, 3),
                "notes": notes,
                "centroid_shift_arcsec": shift_arcsec,
                "systematic_match": None,
            })

            i = j + 1
        else:
            i += 1

    events.sort(key=lambda e: e["anomaly_score"], reverse=True)
    return events[:MAX_EVENTS]


def classify_event(time, flux, scores, bls_result=None) -> tuple[EventType, float, str]:
    flux_arr = np.array(flux)
    time_arr = np.array(time)

    mid = int(np.argmin(flux_arr))
    mid = max(2, min(mid, len(flux_arr) - 3))
    ingress = flux_arr[:mid]
    egress = flux_arr[mid:]

    if len(ingress) >= _MIN_SLOPE_POINTS and len(egress) >= _MIN_SLOPE_POINTS:
        ingress_slope = np.polyfit(range(len(ingress)), ingress, 1)[0]
        egress_slope = np.polyfit(range(len(egress)), egress, 1)[0]
        raw_ratio = abs(ingress_slope) / (abs(egress_slope) + 1e-10)
        asymmetry_ratio = min(raw_ratio, _ASYMMETRY_CAP)
    else:
        ingress_slope = egress_slope = 0.0
        asymmetry_ratio = 1.0
    depth_ppm = _robust_depth_ppm(flux_arr)
    duration_hours = (time_arr[-1] - time_arr[0]) * 24

    time_center = float(time_arr[np.argmin(flux_arr)])
    match_cand = _matching_trusted_candidate(time_center, bls_result)
    bls_match = match_cand is not None

    # --- Stellar flare (brightening — unambiguous) ---
    if flux_arr.max() > 1.005 and egress_slope < 0:
        return (EventType.STELLAR_FLARE, 0.85,
                f"Flux increase {(flux_arr.max()-1)*1e6:.0f} ppm — magnetic reconnection flare.")

    # --- BLS-consistent periodic transit ---
    if bls_match and depth_ppm < 20000:
        bls_p = (match_cand or {}).get("period", 0) or 0
        return (EventType.TRANSIT, 0.90,
                f"Periodic transit (BLS P={bls_p:.3f}d) — depth {depth_ppm:.0f} ppm over {duration_hours:.1f}h.")

    # --- Standard transit (checked before exotic classes) ---
    if depth_ppm < 15000 and asymmetry_ratio < 2.5:
        return (EventType.TRANSIT, 0.81,
                f"Symmetric dip {depth_ppm:.0f} ppm over {duration_hours:.1f}h — planetary transit.")

    # --- Extreme depth anomaly ---
    if depth_ppm > 15000:
        return (EventType.DEPTH_ANOMALY, 0.79,
                f"Depth {depth_ppm:.0f} ppm ({depth_ppm/10000:.1f}%) — exceeds planetary threshold.")

    # --- Exocomet (requires very strong asymmetry) ---
    if asymmetry_ratio > 5.0 and duration_hours < 8:
        return (EventType.EXOCOMET, 0.72,
                f"Asymmetry {asymmetry_ratio:.1f}x — sharp ingress, extended egress. Exocomet candidate.")

    # --- Moderate asymmetry ---
    if 2.5 < asymmetry_ratio <= 5.0:
        return (EventType.ASYMMETRIC, 0.68,
                f"Ingress/egress asymmetry {asymmetry_ratio:.1f}x — ringed body or grazing geometry.")

    # --- Starspot ---
    if duration_hours > 24:
        return (EventType.STELLAR_SPOT, 0.55,
                f"Gradual {duration_hours:.0f}h dimming — starspot rotation.")

    # --- Non-periodic catch-all ---
    if duration_hours < 24 and depth_ppm > 200:
        return (EventType.NON_PERIODIC, 0.50,
                f"Single non-periodic event — depth {depth_ppm:.0f} ppm, {duration_hours:.1f}h.")

    return (EventType.UNKNOWN, 0.40,
            f"No classifier match — depth {depth_ppm:.0f} ppm, "
            f"duration {duration_hours:.1f}h, asymmetry {asymmetry_ratio:.2f}x. "
            f"Requires spectroscopic follow-up.")


_NON_ASTRO_TYPES = {EventType.SYSTEMATIC.value, EventType.CONTAMINATION.value}
_FLARE_TYPE = EventType.STELLAR_FLARE.value
MIN_ENSEMBLE_EVENTS = 3


def _merge_period_candidates(
    bls_result: dict,
    events: list[dict],
    time_range: tuple[float, float] | None,
) -> list[dict]:
    """BLS global best + SDE-ranked candidates + optional event period (deduped)."""
    merged: list[dict] = []
    seen: list[float] = []

    def _dedup(p: float) -> bool:
        return any(abs(p - sp) < max(0.02 * p, 0.08) for sp in seen)

    # Always include argmax-power peak (may differ slightly from top-SDE grid point).
    best_p = float(bls_result.get("best_period_days") or 0)
    best_t0 = float(bls_result.get("transit_time") or 0)
    if best_p > 0 and best_t0 > 0 and not _dedup(best_p):
        merged.append({
            "period": best_p,
            "transit_time": best_t0,
            "sde": float(bls_result.get("best_sde", 0)),
            "power": float(bls_result.get("best_power", 0)),
            "source": "bls",
        })
        seen.append(best_p)

    for c in bls_result.get("candidates", []):
        p = c.get("period") or 0
        if p <= 0 or _dedup(p):
            continue
        row = dict(c)
        row.setdefault("source", "bls")
        merged.append(row)
        seen.append(float(p))

    if time_range is not None:
        t_min, t_max = time_range
        ev_cand = find_period_from_events(events, t_min, t_max)
        if ev_cand:
            p = float(ev_cand["period"])
            if p > 0 and not _dedup(p):
                merged.append(ev_cand)

    return merged


def _partition_trusted_period_ref(bls_result: dict) -> tuple[float, bool]:
    """
    Reference period for alias down-weighting: the **longest** period among the
    global best and top candidates with modest SDE (>= _BLS_REF_POOL_MIN_SDE).
    Argmax-*power* can sit on a short alias (~10 d) while ~33 d still has SDE ~5–6.
    """
    pool: list[float] = []
    bs = float(bls_result.get("best_sde") or 0)
    bp = float(bls_result.get("best_period_days") or 0)
    if bp > 0 and bs >= _BLS_REF_POOL_MIN_SDE:
        pool.append(bp)
    for c in bls_result.get("candidates", []):
        p = float(c.get("period") or 0)
        s = float(c.get("sde") or 0)
        if p > 0 and s >= _BLS_REF_POOL_MIN_SDE:
            pool.append(p)
    if not pool:
        return 0.0, False
    max_sde = bs
    for c in bls_result.get("candidates", []):
        max_sde = max(max_sde, float(c.get("sde") or 0))
    trust = max_sde >= _BLS_MIN_SDE
    return float(max(pool)), trust


def _partition_by_period_candidates(
    astro_events: list[dict],
    candidates: list[dict],
    bls_result: dict,
    baseline_days: float | None = None,
) -> dict | None:
    """
    Try each period candidate; pick the one that maximizes n_matched / (1 + depth_cv)
    subject to depth/duration consistency gates. BLS candidates need SDE >=
    *_BLS_MIN_PARTITION_SDE*; event-sourced candidates are always eligible.

    When any BLS peak has SDE >= *_BLS_MIN_SDE*, BLS rows are scaled by
    exp(-|P-P_ref|/(σ P_ref)) with *P_ref* = longest period in the moderate-SDE pool.

    For baselines longer than *_PARTITION_LONG_BASELINE_MIN_D*, if any candidate
    with P >= *_PARTITION_LONG_PERIOD_MIN_D* passes the gates, that **long-period**
    winner is chosen over shorter periods (stops ~10 d aliases on multi-sector data).
    """
    best_any: dict | None = None
    best_any_metric = -1.0
    best_long: dict | None = None
    best_long_metric = -1.0
    p_ref, trust_bls_ref = _partition_trusted_period_ref(bls_result)

    for cand in candidates:
        period = cand.get("period") or 0
        t0 = cand.get("transit_time") or 0
        sde = float(cand.get("sde", 0))
        source = cand.get("source", "bls")

        if period <= 0 or t0 <= 0:
            continue
        if source == "bls" and sde < _BLS_MIN_PARTITION_SDE:
            continue

        p_f = float(period)
        tol = _phase_tol_for_period(p_f)
        matched: list[dict] = []
        unmatched: list[dict] = []
        for e in astro_events:
            if e["event_type"] == _FLARE_TYPE:
                continue
            if _phase_consistent(e["time_center"], p_f, float(t0), tol=tol):
                matched.append(e)
            else:
                unmatched.append(e)

        if len(matched) < _BLS_GROUP_MIN:
            continue

        depths = np.array([e["depth_ppm"] for e in matched], dtype=float)
        durations = np.array([e["duration_hours"] for e in matched], dtype=float)
        depth_cv = float(np.std(depths) / (np.mean(np.abs(depths)) + 1e-8))
        dur_cv = float(np.std(durations) / (np.mean(durations) + 1e-8))

        dur_cap = _BLS_DUR_CV_MAX
        if p_f >= _PARTITION_LONG_PERIOD_MIN_D:
            dur_cap = max(dur_cap, 0.55)

        if depth_cv > _BLS_DEPTH_CV_MAX or dur_cv > dur_cap:
            continue

        metric = len(matched) / (1.0 + depth_cv)
        # Penalize short BLS trials vs P_ref only. If P_ref is itself a short alias,
        # do not penalize long-period trials — |P-P_ref|/P_ref would crush P~33 d.
        if (
            trust_bls_ref
            and p_ref > 0
            and source != "events"
            and p_f < _PARTITION_LONG_PERIOD_MIN_D
        ):
            h = abs(p_f - p_ref) / p_ref
            metric *= float(np.exp(-h / _BLS_REF_DISTANCE_SIGMA))
        # Short event-inferred periods: down-rank vs P_ref when P_ref is long enough.
        if (
            trust_bls_ref
            and p_ref > 0
            and source == "events"
            and p_f < _PARTITION_LONG_PERIOD_MIN_D
            and p_ref >= _PARTITION_LONG_PERIOD_MIN_D
        ):
            h = abs(p_f - p_ref) / p_ref
            metric *= float(np.exp(-h / _BLS_REF_DISTANCE_SIGMA))

        part = {
            "period": p_f,
            "t0": float(t0),
            "matched": matched,
            "unmatched": unmatched,
            "n_transits": len(matched),
            "mean_depth": float(np.mean(depths)),
            "mean_dur": float(np.mean(durations)),
        }

        if metric > best_any_metric:
            best_any_metric = metric
            best_any = part
        if p_f >= _PARTITION_LONG_PERIOD_MIN_D and metric > best_long_metric:
            best_long_metric = metric
            best_long = part

    use_long = (
        baseline_days is not None
        and baseline_days >= _PARTITION_LONG_BASELINE_MIN_D
        and best_long is not None
    )
    return best_long if use_long else best_any


def analyze_event_ensemble(
    events: list[dict],
    bls_result: dict,
    scores: list,
    time_range: tuple[float, float] | None = None,
) -> tuple[list[dict], float | None]:
    """
    Second-pass classifier that examines all events as a group.

    1. Period-consistency partition — try BLS SDE-trusted peaks and event-inferred
       periods; best partition promotes matched dips to TRANSIT and demotes
       non-matched astrophysical events to STELLAR_VARIABILITY.
    2. Eclipsing-binary check — regular spacing + bimodal depths.
    3. Stellar-variability check — inconsistent depths/durations + many anomalies.

    Returns (events, selected_period_days) where *selected_period_days* is set when
    a valid period partition was applied.
    """
    astro = [e for e in events if e["event_type"] not in _NON_ASTRO_TYPES]
    if len(astro) < MIN_ENSEMBLE_EVENTS:
        return events, None

    # --- Period-consistency partition (multi-candidate) ---
    candidates = _merge_period_candidates(bls_result, events, time_range)
    baseline_days = None
    if time_range is not None:
        baseline_days = float(time_range[1] - time_range[0])
    partition = _partition_by_period_candidates(
        astro, candidates, bls_result, baseline_days=baseline_days,
    )
    if partition is not None:
        partition = _refine_partition_epoch(astro, partition)
        period = partition["period"]
        n_tr = partition["n_transits"]
        mean_d = partition["mean_depth"]
        mean_dur = partition["mean_dur"]

        matched_times = {e["time_center"] for e in partition["matched"]}
        unmatched_times = {e["time_center"] for e in partition["unmatched"]}

        for e in events:
            tc = e["time_center"]
            if tc in matched_times:
                e["event_type"] = EventType.TRANSIT.value
                e["confidence"] = 0.92
                e["notes"] = (
                    f"Periodic transit (P={period:.3f}d, {n_tr} events) — "
                    f"depth {mean_d:.0f} ppm over {mean_dur:.1f}h."
                )
            elif tc in unmatched_times and e["event_type"] not in _NON_ASTRO_TYPES:
                if e["event_type"] == _FLARE_TYPE:
                    continue
                orig_type = e["event_type"]
                orig_depth = e["depth_ppm"]
                e["event_type"] = EventType.STELLAR_VARIABILITY.value
                e["confidence"] = 0.75
                e["notes"] = (
                    f"Not consistent with P={period:.3f}d — "
                    f"stellar activity ({orig_type}: {orig_depth:.0f} ppm)."
                )
        return events, float(period)

    # --- Eclipsing binary / variability (no valid period partition) ---
    depths = np.array([e["depth_ppm"] for e in astro])
    durations = np.array([e["duration_hours"] for e in astro])
    times = np.array(sorted(e["time_center"] for e in astro))

    depth_mean = float(np.mean(depths))
    depth_cv = float(np.std(depths) / (depth_mean + 1e-8))
    dur_cv = float(np.std(durations) / (np.mean(durations) + 1e-8))

    intervals = np.diff(times)
    mean_interval = float(np.mean(intervals)) if len(intervals) else 0.0
    interval_cv = float(np.std(intervals) / (mean_interval + 1e-8)) if len(intervals) > 1 else 1.0

    anomaly_frac = sum(1 for s in scores if s > 0.5) / max(len(scores), 1)

    bls_period = bls_result.get("best_period_days", 0) or 0

    eb_result = _check_eclipsing_binary(
        astro, depths, times, intervals, mean_interval, interval_cv,
        depth_cv, bls_period,
    )
    if eb_result is not None:
        return _reclassify(events, eb_result), None

    sv_result = _check_stellar_variability(
        astro, depth_cv, dur_cv, anomaly_frac,
    )
    if sv_result is not None:
        return _reclassify(events, sv_result), None

    return events, None


def _check_eclipsing_binary(
    astro, depths, times, intervals, mean_interval, interval_cv,
    depth_cv, bls_period,
) -> dict | None:
    """Detect eclipsing binaries via regular spacing + bimodal depths + BLS harmonics."""
    if len(astro) < MIN_ENSEMBLE_EVENTS or interval_cv > 0.5:
        return None

    bls_is_subharmonic = False
    if bls_period > 0 and mean_interval > 0:
        ratio = mean_interval / bls_period
        for mult in [1.5, 2.0, 2.5, 3.0]:
            if abs(ratio - mult) < 0.2:
                bls_is_subharmonic = True
                break

    depth_sorted = np.sort(depths)
    mid = len(depth_sorted) // 2
    depth_ratio = float(np.mean(depth_sorted[mid:]) / (np.mean(depth_sorted[:mid]) + 1e-8))
    bimodal_depths = depth_ratio > 1.5 and depth_cv > 0.25

    if not (bls_is_subharmonic or bimodal_depths):
        return None

    true_period = mean_interval if bls_is_subharmonic else bls_period
    return {
        "event_type": EventType.ECLIPSING_BINARY.value,
        "confidence": 0.88,
        "note_template": (
            f"Eclipsing binary — {len(astro)} events at ~{mean_interval:.2f}d intervals, "
            f"depth ratio {depth_ratio:.1f}x, true period ~{true_period:.3f}d. "
            f"Not planetary."
        ),
    }


def _check_stellar_variability(astro, depth_cv, dur_cv, anomaly_frac) -> dict | None:
    """Detect stellar variability via inconsistent depths/durations and pervasive anomalies."""
    triggered = False

    if depth_cv > 0.35 and dur_cv > 0.4 and anomaly_frac > 0.10:
        triggered = True
    elif depth_cv > 0.50 and len(astro) >= 5:
        triggered = True
    elif anomaly_frac > 0.15 and len(astro) >= MIN_ENSEMBLE_EVENTS and depth_cv > 0.25:
        triggered = True

    if not triggered:
        return None

    return {
        "event_type": EventType.STELLAR_VARIABILITY.value,
        "confidence": 0.85,
        "note_template": (
            f"Stellar variability — {len(astro)} events with variable depths "
            f"(CV={depth_cv:.2f}) and durations (CV={dur_cv:.2f}). "
            f"{anomaly_frac * 100:.0f}% of light curve anomalous. Not planetary."
        ),
    }


def _reclassify(events: list[dict], result: dict) -> list[dict]:
    """Apply ensemble reclassification to all astrophysical events."""
    for e in events:
        if e["event_type"] in _NON_ASTRO_TYPES:
            continue
        original_type = e["event_type"]
        original_depth = e["depth_ppm"]
        original_dur = e["duration_hours"]
        e["event_type"] = result["event_type"]
        e["confidence"] = result["confidence"]
        e["notes"] = (
            f"{result['note_template']} "
            f"[Originally {original_type}: {original_depth:.0f} ppm, {original_dur:.1f}h]"
        )
    return events
