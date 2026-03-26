import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

STELLAR_CROSSING_HOURS = {
    "O": 3.00, "B": 2.00, "A": 1.20, "F": 0.80,
    "G": 0.60, "K": 0.40, "M": 0.15,
}
DEFAULT_CROSSING_HOURS = 0.60


def analyze(tic_id: str, time: list, flux: list,
            unknown_events: list[dict], period_result: dict,
            stellar_type: str = None) -> dict:
    if not unknown_events:
        return {"ran": False, "reason": "No UNKNOWN events", "composite_score": 0.0, "hypotheses": []}

    morphology = analyze_morphology(time, flux, unknown_events, stellar_type)
    entropy = analyze_timing_entropy(unknown_events)
    ir_excess = query_wise_excess(tic_id)
    catalog = check_catalog_membership(tic_id)

    modules = {
        "morphology": morphology["score"],
        "timing": entropy["score"],
        "ir_excess": ir_excess["score"],
        "catalog": catalog["score"],
    }
    nonzero = [s for s in modules.values() if s > 0]
    composite = float(np.exp(np.mean(np.log(nonzero)))) if nonzero else 0.0
    n_contributing = len(nonzero)
    n_total_modules = len(modules)

    summary = _generate_summary(composite, n_contributing, n_total_modules, modules)

    techno_result = {
        "ran": True,
        "composite_score": round(composite, 4),
        "morphology": morphology,
        "timing_entropy": entropy,
        "ir_excess": ir_excess,
        "catalog_membership": catalog,
        "summary": summary,
    }

    from .hypothesis import generate_hypotheses
    try:
        techno_result["hypotheses"] = generate_hypotheses(
            unknown_events, techno_result, period_result,
        )
    except Exception as e:
        logger.warning("Hypothesis generation failed: %s", e)
        techno_result["hypotheses"] = []

    return techno_result


def analyze_morphology(time: list, flux: list, events: list[dict],
                       stellar_type: str = None) -> dict:
    time_arr = np.array(time)
    flux_arr = np.array(flux)

    crossing_hours = STELLAR_CROSSING_HOURS.get(
        (stellar_type or "G")[0].upper(), DEFAULT_CROSSING_HOURS
    )

    results = []
    for event in events:
        tc = event["time_center"]
        dur_h = event["duration_hours"]
        half_dur_days = (dur_h / 2.0) / 24.0

        mask = np.abs(time_arr - tc) < half_dur_days
        if mask.sum() < 10:
            continue

        t_ev = time_arr[mask]
        f_ev = flux_arr[mask]
        valid = ~np.isnan(f_ev)
        t_ev, f_ev = t_ev[valid], f_ev[valid]
        if len(f_ev) < 10:
            continue

        min_idx = np.argmin(f_ev)
        if min_idx > 2:
            ingress_dur = (t_ev[min_idx] - t_ev[0]) * 24
        else:
            ingress_dur = dur_h / 2
        ingress_too_fast = ingress_dur < crossing_hours

        sorted_flux = np.sort(f_ev)
        bottom_20 = sorted_flux[:max(3, len(sorted_flux) // 5)]
        floor_std = float(np.std(bottom_20))
        overall_std = float(np.std(f_ev))
        flat_floor = floor_std < 0.05 * overall_std if overall_std > 0 else False

        has_substructure = False
        if len(f_ev) > 20:
            poly_c = np.polyfit(np.arange(len(f_ev)), f_ev, 2)
            residual = f_ev - np.polyval(poly_c, np.arange(len(f_ev)))
            ac = np.correlate(residual, residual, mode="full")
            ac = ac[len(ac) // 2:]
            ac = ac / (ac[0] + 1e-12)
            peaks = [i for i in range(2, len(ac) - 1)
                     if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > 0.3]
            has_substructure = len(peaks) >= 2

        mid = len(f_ev) // 2
        left, right = f_ev[:mid], f_ev[mid:mid + mid][::-1]
        if len(left) == len(right) and len(left) > 5:
            sym_corr = float(np.corrcoef(left, right)[0, 1])
        else:
            sym_corr = 0.0
        hyper_symmetric = sym_corr > 0.99

        flags = sum([ingress_too_fast, flat_floor, has_substructure, hyper_symmetric])
        results.append({
            "time_center": tc, "flags": flags, "score": round(flags / 4.0, 3),
            "ingress_too_fast": ingress_too_fast, "flat_floor": flat_floor,
            "has_substructure": has_substructure, "hyper_symmetric": hyper_symmetric,
            "symmetry_correlation": round(sym_corr, 4),
        })

    if not results:
        return {"score": 0.0, "events": [], "note": "Insufficient data points."}

    best = max(results, key=lambda r: r["score"])
    return {"score": best["score"], "events": results, "note": _morphology_note(best)}


def _morphology_note(r: dict) -> str:
    flags = []
    if r["ingress_too_fast"]:
        flags.append("ingress faster than stellar crossing time")
    if r["flat_floor"]:
        flags.append("anomalously flat dip floor")
    if r["has_substructure"]:
        flags.append("periodic sub-structure within dip")
    if r["hyper_symmetric"]:
        flags.append(f"bilateral symmetry {r['symmetry_correlation']:.4f}")
    return ("Morphological anomalies: " + "; ".join(flags) + ".") if flags else "Morphology consistent with natural occultation."


def analyze_timing_entropy(events: list[dict]) -> dict:
    if len(events) < 3:
        return {"score": 0.0, "entropy": None, "n_events": len(events),
                "note": f"Only {len(events)} UNKNOWN event(s) — need 3+ for timing analysis."}

    times = sorted([e["time_center"] for e in events])
    intervals = np.diff(times)
    if len(intervals) < 2:
        return {"score": 0.0, "entropy": None, "n_events": len(events),
                "note": "Insufficient intervals."}

    intervals_norm = intervals / (intervals.max() + 1e-12)
    n_bins = min(10, len(intervals))
    counts, _edges = np.histogram(intervals_norm, bins=n_bins)
    total = counts.sum()
    probs = counts / total if total > 0 else counts
    mask = probs > 0
    entropy = -float(np.sum(probs[mask] * np.log2(probs[mask])))
    n_occupied = int(mask.sum())
    max_ent = np.log2(n_occupied) if n_occupied > 1 else 1.0
    norm_ent = entropy / max_ent if max_ent > 0 else 0

    _unique_ratios = sorted({a / b for a in range(1, 7) for b in range(1, 7)})
    n_unique = len(_unique_ratios)
    ratio_matches = 0
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            ratio = intervals[i] / (intervals[j] + 1e-12)
            for target in _unique_ratios:
                if abs(ratio - target) < 0.02:
                    ratio_matches += 1
                    break

    constants = {"pi": 3.14159265, "e": 2.71828183, "phi": 1.61803399}
    const_matches: set[str] = set()
    for name, val in constants.items():
        for i in range(len(intervals)):
            for j in range(len(intervals)):
                if i != j and abs(intervals[i] / (intervals[j] + 1e-12) - val) < 0.01:
                    const_matches.add(name)

    n_pairs = max(1, len(intervals) * (len(intervals) - 1) / 2)
    struct_score = min(1.0, (ratio_matches / (n_pairs * n_unique)) * 50 + len(const_matches) * 0.3)
    ent_score = max(0, 1.0 - norm_ent)
    combined = ent_score * 0.4 + struct_score * 0.6

    return {
        "score": round(combined, 3), "entropy": round(float(entropy), 4),
        "normalized_entropy": round(float(norm_ent), 4), "n_events": len(events),
        "intervals_days": [round(float(x), 4) for x in intervals],
        "ratio_matches": ratio_matches, "constant_matches": sorted(const_matches),
        "note": f"Entropy {norm_ent:.2f}, {ratio_matches} ratio matches.",
    }


def query_wise_excess(tic_id: str) -> dict:
    try:
        from astroquery.ipac.irsa import Irsa
        from astroquery.mast import Catalogs
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        tic_data = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC", radius=0.001 * u.deg)
        if len(tic_data) == 0:
            return {"score": 0.0, "available": False, "note": "TIC lookup failed."}

        ra, dec = float(tic_data[0]["ra"]), float(tic_data[0]["dec"])
        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        wise = Irsa.query_region(coord, catalog="allwise_p3as_psd", radius=6 * u.arcsec)

        if len(wise) == 0:
            return {"score": 0.0, "available": False, "note": "No AllWISE match."}

        w = wise[0]
        w1, w2 = float(w["w1mpro"]), float(w["w2mpro"])
        w3 = float(w["w3mpro"]) if not np.ma.is_masked(w["w3mpro"]) else None
        w4 = float(w["w4mpro"]) if not np.ma.is_masked(w["w4mpro"]) else None

        w1_w3 = (w1 - w3) if w3 is not None else None
        w1_w4 = (w1 - w4) if w4 is not None else None

        excess_w3 = max(0, (w1_w3 - 0.2) / 2.0) if w1_w3 is not None else 0
        excess_w4 = max(0, (w1_w4 - 0.3) / 3.0) if w1_w4 is not None else 0
        score = min(1.0, max(excess_w3, excess_w4))

        return {
            "score": round(score, 3), "available": True,
            "w1": round(w1, 3), "w2": round(w2, 3),
            "w3": round(w3, 3) if w3 else None, "w4": round(w4, 3) if w4 else None,
            "w1_w3": round(w1_w3, 3) if w1_w3 else None,
            "w1_w4": round(w1_w4, 3) if w1_w4 else None,
            "note": f"W1-W3={w1_w3:.2f}" if w1_w3 else "No W3/W4 detections.",
        }
    except Exception as e:
        logger.warning(f"WISE query failed: {e}")
        return {"score": 0.0, "available": False, "note": f"WISE query failed: {e}"}


def _simbad_col(row, *names, default=""):
    """Read a column by trying multiple names (handles astroquery <0.4.8 and >=0.4.8)."""
    for name in names:
        try:
            val = row[name]
            if val is not None:
                return str(val)
        except (KeyError, IndexError):
            continue
    return default


def check_catalog_membership(tic_id: str) -> dict:
    try:
        from astroquery.simbad import Simbad
        from astroquery.mast import Catalogs
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        tic_data = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC", radius=0.001 * u.deg)
        if len(tic_data) == 0:
            return {"score": 0.0, "available": False, "note": "TIC lookup failed."}

        ra, dec = float(tic_data[0]["ra"]), float(tic_data[0]["dec"])
        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        custom = Simbad()
        custom.add_votable_fields("otype", "otypes")
        result = custom.query_region(coord, radius=6 * u.arcsec)

        known_types = []
        simbad_otype = None
        if result is not None and len(result) > 0:
            simbad_otype = _simbad_col(result[0], "otype", "OTYPE", "main_type")
            all_types = _simbad_col(result[0], "otypes", "OTYPES")
            var_ind = ["V*", "Pu*", "Ce*", "Mi*", "RR*", "dS*", "SX*", "gD*", "BY*"]
            bin_ind = ["**", "EB*", "SB*", "El*", "Sy*"]
            for vi in var_ind:
                if vi in all_types:
                    known_types.append(f"variable ({vi})")
            for bi in bin_ind:
                if bi in all_types:
                    known_types.append(f"binary ({bi})")

        if known_types:
            score = 0.1
        elif simbad_otype and simbad_otype not in ("Star", "*", "?"):
            score = 0.3
        else:
            score = 0.9

        return {
            "score": round(score, 3), "available": True,
            "simbad_type": simbad_otype, "known_types": known_types,
            "note": f"Known: {', '.join(known_types)}" if known_types else "No known classification.",
        }
    except Exception as e:
        logger.warning(f"Catalog query failed: {e}")
        return {"score": 0.5, "available": False, "note": f"Catalog query failed: {e}"}


def _generate_summary(composite: float, n_contributing: int,
                      n_total: int, modules: dict[str, float]) -> str:
    if composite < 0.1:
        return "Technosignature: no anomalous indicators."

    elevated = [name for name, s in modules.items() if s >= 0.3]
    coverage = f"{n_contributing}/{n_total} modules contributed"

    if composite < 0.3:
        return f"Technosignature: weak indicators ({composite:.3f}). {coverage}. Standard follow-up."
    if composite < 0.6:
        return (f"Technosignature: moderate indicators ({composite:.3f}). "
                f"Elevated: {', '.join(elevated) or 'none'}. {coverage}. Priority follow-up recommended.")
    return (f"ELEVATED TECHNOSIGNATURE INDICATORS ({composite:.3f}). "
            f"Elevated: {', '.join(elevated) or 'none'}. {coverage}. "
            "Request independent reanalysis.")
