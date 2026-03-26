"""
Hypothesis generator for UNKNOWN events.

Given features extracted from unclassified anomalous events and
technosignature analysis results, produces a ranked list of candidate
explanations covering both natural astrophysical phenomena and
artificial (technosignature) origins.

Each hypothesis defines expected feature signatures and scores
observed data against them.  The final output is sorted by match
score so the analyst can evaluate the most plausible explanations
first.
"""

import numpy as np
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(
    events: list[dict],
    techno: dict,
    period_result: dict,
) -> dict[str, Any]:
    depths = np.array([e["depth_ppm"] for e in events], dtype=float)
    durations = np.array([e["duration_hours"] for e in events], dtype=float)
    centroids = np.array(
        [e["centroid_shift_arcsec"] for e in events
         if e.get("centroid_shift_arcsec", -1) >= 0],
        dtype=float,
    )

    morphology = techno.get("morphology", {})
    morph_events = morphology.get("events", [])
    timing = techno.get("timing_entropy", {})
    ir = techno.get("ir_excess", {})
    catalog = techno.get("catalog_membership", {})

    return {
        "n_events": len(events),
        "depth_mean": float(np.mean(depths)),
        "depth_std": float(np.std(depths)) if len(depths) > 1 else 0.0,
        "depth_cv": float(np.std(depths) / (np.mean(np.abs(depths)) + 1e-8)),
        "depth_min": float(np.min(depths)),
        "depth_max": float(np.max(depths)),
        "dur_mean": float(np.mean(durations)),
        "dur_std": float(np.std(durations)) if len(durations) > 1 else 0.0,
        "dur_cv": float(np.std(durations) / (np.mean(durations) + 1e-8)),
        "centroid_mean": float(np.mean(centroids)) if len(centroids) else 0.0,
        "centroid_max": float(np.max(centroids)) if len(centroids) else 0.0,
        # morphology flags (any event)
        "any_fast_ingress": any(e.get("ingress_too_fast") for e in morph_events),
        "any_flat_floor": any(e.get("flat_floor") for e in morph_events),
        "any_substructure": any(e.get("has_substructure") for e in morph_events),
        "any_hyper_symmetric": any(e.get("hyper_symmetric") for e in morph_events),
        "max_symmetry": max(
            (e.get("symmetry_correlation", 0) for e in morph_events), default=0,
        ),
        # timing
        "timing_entropy": timing.get("normalized_entropy", 1.0),
        "timing_score": timing.get("score", 0.0),
        "ratio_matches": timing.get("ratio_matches", 0),
        "constant_matches": timing.get("constant_matches", []),
        # IR excess
        "ir_score": ir.get("score", 0.0),
        "ir_available": ir.get("available", False),
        "w1_w3": ir.get("w1_w3"),
        "w1_w4": ir.get("w1_w4"),
        # catalog
        "catalog_score": catalog.get("score", 0.5),
        "simbad_type": catalog.get("simbad_type"),
        "known_types": catalog.get("known_types", []),
        # period
        "best_period": float(period_result.get("best_period_days") or 0),
        "best_sde": float(period_result.get("best_sde") or 0),
    }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float, center: float = 0.5, steepness: float = 10.0) -> float:
    return float(1.0 / (1.0 + np.exp(-steepness * (x - center))))


def _in_range(val: float, lo: float, hi: float) -> float:
    if lo <= val <= hi:
        return 1.0
    dist = min(abs(val - lo), abs(val - hi))
    span = hi - lo if hi > lo else 1.0
    return float(max(0.0, 1.0 - dist / span))


def _weighted_score(indicators: list[tuple[float, float]]) -> float:
    total_w = sum(w for w, _ in indicators)
    if total_w == 0:
        return 0.0
    return float(sum(w * s for w, s in indicators) / total_w)


# ---------------------------------------------------------------------------
# Natural hypotheses
# ---------------------------------------------------------------------------

def _score_debris_disk(f: dict) -> tuple[float, list[str]]:
    """Circumstellar debris disk / dust cloud transit (cf. Boyajian's Star)."""
    reasons = []
    indicators = []

    # Variable depths expected
    if f["depth_cv"] > 0.4:
        indicators.append((3.0, min(1.0, f["depth_cv"] / 1.5)))
        reasons.append(f"Highly variable depths (CV={f['depth_cv']:.2f}) consistent with clumpy dust")
    else:
        indicators.append((3.0, 0.2))
        reasons.append(f"Low depth variability (CV={f['depth_cv']:.2f}) — less consistent with dust")

    # Aperiodic / weak periodicity expected
    if f["best_sde"] < 6:
        indicators.append((2.0, 0.8))
        reasons.append("No strong periodicity — consistent with irregular dust transits")
    else:
        indicators.append((2.0, 0.3))
        reasons.append(f"Periodic signal detected (SDE={f['best_sde']:.1f}) — less typical for dust")

    # IR excess is a strong indicator
    if f["ir_available"] and f["ir_score"] > 0.3:
        indicators.append((3.0, min(1.0, f["ir_score"] * 1.5)))
        reasons.append(f"Infrared excess detected (score={f['ir_score']:.2f}) — warm dust signature")
    elif f["ir_available"]:
        indicators.append((2.0, 0.2))
        reasons.append("No significant infrared excess")
    else:
        indicators.append((1.0, 0.4))

    # Multiple events expected
    if f["n_events"] >= 3:
        indicators.append((1.5, min(1.0, f["n_events"] / 5.0)))
        reasons.append(f"{f['n_events']} events — multiple transiting dust clumps")
    else:
        indicators.append((1.5, 0.3))

    # Long durations more consistent
    if f["dur_mean"] > 6:
        indicators.append((1.0, min(1.0, f["dur_mean"] / 24.0)))
        reasons.append(f"Extended durations (avg {f['dur_mean']:.1f}h) — spatially extended dust")

    return _weighted_score(indicators), reasons


def _score_disintegrating_planet(f: dict) -> tuple[float, list[str]]:
    """Disintegrating / evaporating planet with cometary dust tail."""
    reasons = []
    indicators = []

    # Some periodicity expected (planet still orbits)
    if 4 < f["best_sde"]:
        indicators.append((2.5, min(1.0, f["best_sde"] / 10.0)))
        reasons.append(f"Periodicity hint (SDE={f['best_sde']:.1f}) — remnant orbital signal")
    else:
        indicators.append((2.5, 0.2))
        reasons.append("No periodicity — less consistent with orbiting body")

    # Variable depths (dust tail varies orbit-to-orbit)
    if 0.3 < f["depth_cv"] < 1.5:
        indicators.append((2.0, 0.8))
        reasons.append(f"Moderate depth variability (CV={f['depth_cv']:.2f}) — varying dust production")
    else:
        indicators.append((2.0, 0.3))

    # Shallow-to-moderate depths (small body)
    if f["depth_mean"] < 5000:
        indicators.append((1.5, 0.7))
        reasons.append(f"Moderate depth ({f['depth_mean']:.0f} ppm) — sub-planetary occultation")
    else:
        indicators.append((1.5, 0.3))
        reasons.append(f"Deep events ({f['depth_mean']:.0f} ppm) — larger than expected for disintegrating body")

    # Short-medium durations
    if 1 < f["dur_mean"] < 12:
        indicators.append((1.0, 0.7))
        reasons.append(f"Duration range consistent with close-in orbit ({f['dur_mean']:.1f}h)")

    return _weighted_score(indicators), reasons


def _score_ring_system(f: dict) -> tuple[float, list[str]]:
    """Circumplanetary ring system (cf. J1407b super-ring)."""
    reasons = []
    indicators = []

    # Deep, long transits
    if f["depth_max"] > 5000:
        indicators.append((2.0, min(1.0, f["depth_max"] / 20000.0)))
        reasons.append(f"Deep dimming ({f['depth_max']:.0f} ppm) — large occulting area")
    else:
        indicators.append((2.0, 0.3))

    if f["dur_mean"] > 8:
        indicators.append((2.0, min(1.0, f["dur_mean"] / 30.0)))
        reasons.append(f"Long duration ({f['dur_mean']:.1f}h) — extended ring structure")
    else:
        indicators.append((2.0, 0.3))

    # Substructure within dip (ring gaps)
    if f["any_substructure"]:
        indicators.append((3.0, 0.9))
        reasons.append("Periodic substructure within dip — ring gap signature")
    else:
        indicators.append((3.0, 0.2))

    # Some symmetry expected but not perfect
    if 0.7 < f["max_symmetry"] < 0.99:
        indicators.append((1.5, 0.7))
        reasons.append(f"Moderate symmetry ({f['max_symmetry']:.3f}) — tilted ring system")

    # Few events (long orbital period)
    if f["n_events"] <= 3:
        indicators.append((1.0, 0.7))
        reasons.append(f"Few events ({f['n_events']}) — long-period orbit")

    return _weighted_score(indicators), reasons


def _score_trojan_swarm(f: dict) -> tuple[float, list[str]]:
    """Trojan asteroid / co-orbital body swarm."""
    reasons = []
    indicators = []

    # Multiple shallow events at similar period
    if f["n_events"] >= 3:
        indicators.append((2.0, min(1.0, f["n_events"] / 6.0)))
        reasons.append(f"{f['n_events']} events — multiple co-orbital bodies")
    else:
        indicators.append((2.0, 0.2))

    if f["depth_mean"] < 2000:
        indicators.append((2.0, 0.7))
        reasons.append(f"Shallow depths ({f['depth_mean']:.0f} ppm) — small bodies")
    else:
        indicators.append((2.0, 0.3))

    # Some periodicity expected
    if f["best_sde"] > 4:
        indicators.append((2.0, min(1.0, f["best_sde"] / 8.0)))
        reasons.append(f"Periodic signal (SDE={f['best_sde']:.1f}) — shared orbital period")
    else:
        indicators.append((2.0, 0.3))

    # Low depth variability (similar-sized bodies)
    if f["depth_cv"] < 0.4:
        indicators.append((1.5, 0.7))
        reasons.append(f"Consistent depths (CV={f['depth_cv']:.2f}) — uniform body sizes")
    else:
        indicators.append((1.5, 0.3))

    return _weighted_score(indicators), reasons


def _score_brown_dwarf(f: dict) -> tuple[float, list[str]]:
    """Brown dwarf transit (too small to fuse hydrogen, larger than Jupiter)."""
    reasons = []
    indicators = []

    # Deep transits (brown dwarfs are ~Jupiter-sized but denser)
    if 5000 < f["depth_mean"] < 30000:
        indicators.append((2.5, 0.8))
        reasons.append(f"Deep transits ({f['depth_mean']:.0f} ppm) — brown dwarf occultation range")
    elif f["depth_mean"] >= 30000:
        indicators.append((2.5, 0.5))
        reasons.append(f"Very deep ({f['depth_mean']:.0f} ppm) — possible stellar-mass companion")
    else:
        indicators.append((2.5, 0.2))

    # Strict periodicity
    if f["best_sde"] > 6:
        indicators.append((2.5, min(1.0, f["best_sde"] / 10.0)))
        reasons.append(f"Strong periodicity (SDE={f['best_sde']:.1f}) — Keplerian orbit")
    else:
        indicators.append((2.5, 0.2))

    # Flat-bottomed transits
    if f["any_flat_floor"]:
        indicators.append((2.0, 0.8))
        reasons.append("Flat-bottomed dip — total occultation by opaque body")
    else:
        indicators.append((2.0, 0.3))

    # Low depth variability (rigid body = consistent depth)
    if f["depth_cv"] < 0.2:
        indicators.append((1.5, 0.8))
        reasons.append(f"Very consistent depths (CV={f['depth_cv']:.2f}) — rigid spherical body")

    return _weighted_score(indicators), reasons


def _score_exomoon(f: dict) -> tuple[float, list[str]]:
    """Exomoon transit — additional dips near a periodic signal."""
    reasons = []
    indicators = []

    # Few events
    if 2 <= f["n_events"] <= 5:
        indicators.append((1.5, 0.6))
        reasons.append(f"{f['n_events']} events — could be moon + planet compound")
    else:
        indicators.append((1.5, 0.3))

    # Variable timing (moon position changes each orbit)
    if f["timing_entropy"] > 0.5:
        indicators.append((2.0, min(1.0, f["timing_entropy"])))
        reasons.append(f"Timing variability (entropy={f['timing_entropy']:.2f}) — moon orbital motion")
    else:
        indicators.append((2.0, 0.3))

    # Shallow depths (moon is smaller than planet)
    if f["depth_mean"] < 3000:
        indicators.append((1.5, 0.6))
        reasons.append(f"Shallow events ({f['depth_mean']:.0f} ppm) — sub-planetary body")

    # Moderate depth variability (moon vs planet contribution)
    if 0.2 < f["depth_cv"] < 0.8:
        indicators.append((1.5, 0.6))
        reasons.append(f"Moderate depth scatter (CV={f['depth_cv']:.2f}) — compound transit morphology")

    return _weighted_score(indicators), reasons


def _score_instrument_artifact(f: dict) -> tuple[float, list[str]]:
    """Unidentified instrumental/systematic artifact."""
    reasons = []
    indicators = []

    # High centroid shifts suggest off-target
    if f["centroid_max"] > 5:
        indicators.append((2.0, min(1.0, f["centroid_max"] / 10.0)))
        reasons.append(f"Large centroid shift ({f['centroid_max']:.1f}\") — possible scattered light")
    else:
        indicators.append((2.0, 0.2))

    # Known SIMBAD type lowers artifact likelihood
    if f["known_types"]:
        indicators.append((1.5, 0.2))
        reasons.append(f"Known object type ({', '.join(f['known_types'][:2])}) — astrophysical origin more likely")
    else:
        indicators.append((1.5, 0.5))

    # Very shallow depths could be noise
    if f["depth_mean"] < 500:
        indicators.append((1.5, 0.6))
        reasons.append(f"Very shallow ({f['depth_mean']:.0f} ppm) — within instrumental noise range")
    else:
        indicators.append((1.5, 0.2))

    # Very few events
    if f["n_events"] <= 2:
        indicators.append((1.0, 0.5))
        reasons.append("Few events — could be stochastic detector behavior")

    return _weighted_score(indicators), reasons


# ---------------------------------------------------------------------------
# Artificial / technosignature hypotheses
# ---------------------------------------------------------------------------

def _score_megastructure(f: dict) -> tuple[float, list[str]]:
    """
    Partial Dyson sphere or orbiting megastructure.

    Expected: irregular deep dimming, variable depths, aperiodic or
    quasi-periodic, infrared excess from waste heat radiation,
    uncatalogued star.  (cf. Boyajian's Star / KIC 8462852)
    """
    reasons = []
    indicators = []

    # Aperiodic deep dimming
    if f["depth_cv"] > 0.5:
        indicators.append((2.0, min(1.0, f["depth_cv"] / 2.0)))
        reasons.append(f"Variable occultation depths (CV={f['depth_cv']:.2f}) — non-uniform structure geometry")

    if f["best_sde"] < 5:
        indicators.append((1.5, 0.7))
        reasons.append("No strict periodicity — asymmetric orbital structure or construction in progress")
    else:
        indicators.append((1.5, 0.3))
        reasons.append(f"Periodic signal (SDE={f['best_sde']:.1f}) — could be stable orbital megastructure")

    # IR excess from waste heat (Kardashev Type II civilization)
    if f["ir_available"] and f["ir_score"] > 0.3:
        indicators.append((3.0, min(1.0, f["ir_score"] * 1.5)))
        reasons.append(f"Infrared excess (score={f['ir_score']:.2f}) — potential waste heat from energy collection")
    elif f["ir_available"]:
        indicators.append((2.0, 0.2))
        reasons.append("No infrared excess — no detectable waste heat")
    else:
        indicators.append((1.0, 0.4))

    # Uncatalogued star (no known explanation)
    if f["catalog_score"] > 0.7:
        indicators.append((2.0, f["catalog_score"]))
        reasons.append("Uncatalogued object — no known natural classification")
    else:
        indicators.append((2.0, 0.2))
        reasons.append(f"Known object type — natural explanation more likely")

    # Deep events
    if f["depth_max"] > 1000:
        indicators.append((1.0, min(1.0, f["depth_max"] / 10000.0)))
        reasons.append(f"Significant dimming ({f['depth_max']:.0f} ppm max)")

    # Multiple events
    if f["n_events"] >= 3:
        indicators.append((1.0, min(1.0, f["n_events"] / 5.0)))
        reasons.append(f"{f['n_events']} separate occultation events")

    return _weighted_score(indicators), reasons


def _score_transit_beacon(f: dict) -> tuple[float, list[str]]:
    """
    Artificial transit beacon — engineered object placed in orbit
    to produce a deliberately detectable, non-natural transit signal.

    Expected: hyper-symmetric dip profile, ingress faster than stellar
    crossing time (sharp edges), flat dip floor, possibly non-natural
    depth or timing structure.
    """
    reasons = []
    indicators = []

    # Hyper-symmetry is the strongest indicator
    if f["any_hyper_symmetric"]:
        indicators.append((3.0, 0.9))
        reasons.append(f"Hyper-symmetric dip (r={f['max_symmetry']:.4f}) — exceeds natural occultation symmetry")
    elif f["max_symmetry"] > 0.95:
        indicators.append((3.0, 0.6))
        reasons.append(f"High symmetry (r={f['max_symmetry']:.4f}) — near artificial threshold")
    else:
        indicators.append((3.0, 0.1))
        reasons.append(f"Normal symmetry (r={f['max_symmetry']:.4f})")

    # Ingress faster than stellar crossing time
    if f["any_fast_ingress"]:
        indicators.append((2.5, 0.85))
        reasons.append("Ingress faster than stellar crossing time — non-natural edge sharpness")
    else:
        indicators.append((2.5, 0.15))

    # Flat floor
    if f["any_flat_floor"]:
        indicators.append((2.0, 0.7))
        reasons.append("Anomalously flat dip floor — engineered opaque geometry")
    else:
        indicators.append((2.0, 0.2))

    # Mathematical timing structure
    if f["constant_matches"]:
        indicators.append((2.0, 0.8))
        reasons.append(f"Timing encodes mathematical constants: {', '.join(f['constant_matches'])}")
    elif f["ratio_matches"] > 3:
        indicators.append((2.0, 0.5))
        reasons.append(f"{f['ratio_matches']} integer-ratio timing matches — possible structured signal")
    else:
        indicators.append((2.0, 0.1))

    return _weighted_score(indicators), reasons


def _score_clarke_exobelt(f: dict) -> tuple[float, list[str]]:
    """
    Clarke exobelt — artificial satellite belt in synchronous orbit
    around an exoplanet, causing shallow periodic dips.

    Expected: shallow, consistent-depth periodic dips with duration
    matching planetary transit + ring-like extension.
    """
    reasons = []
    indicators = []

    # Shallow periodic dips
    if f["depth_mean"] < 2000:
        indicators.append((2.0, 0.7))
        reasons.append(f"Shallow transits ({f['depth_mean']:.0f} ppm) — satellite belt occultation scale")
    else:
        indicators.append((2.0, 0.2))

    if f["best_sde"] > 5:
        indicators.append((2.5, min(1.0, f["best_sde"] / 10.0)))
        reasons.append(f"Periodic signal (SDE={f['best_sde']:.1f}) — stable orbital belt")
    else:
        indicators.append((2.5, 0.2))

    # Consistent depth (uniform belt density)
    if f["depth_cv"] < 0.3:
        indicators.append((2.0, 0.7))
        reasons.append(f"Consistent depths (CV={f['depth_cv']:.2f}) — uniform orbital structure density")
    else:
        indicators.append((2.0, 0.2))

    # Substructure (gaps in the belt)
    if f["any_substructure"]:
        indicators.append((1.5, 0.7))
        reasons.append("Substructure detected — belt density variations or gaps")

    # Not catalogued
    if f["catalog_score"] > 0.5:
        indicators.append((1.0, f["catalog_score"]))

    return _weighted_score(indicators), reasons


def _score_laser_beacon(f: dict) -> tuple[float, list[str]]:
    """
    Laser / optical communication beacon.

    Expected: very short duration events, mathematically structured
    timing, sharp morphology.  Low-energy optical SETI target.
    """
    reasons = []
    indicators = []

    # Very short events
    if f["dur_mean"] < 3:
        indicators.append((2.5, min(1.0, (3 - f["dur_mean"]) / 2.5)))
        reasons.append(f"Short duration ({f['dur_mean']:.1f}h) — pulse-like morphology")
    else:
        indicators.append((2.5, 0.1))
        reasons.append(f"Long duration ({f['dur_mean']:.1f}h) — inconsistent with optical pulse")

    # Mathematical timing structure
    if f["constant_matches"]:
        indicators.append((3.0, 0.9))
        reasons.append(f"Timing encodes mathematical constants: {', '.join(f['constant_matches'])} — deliberate information content")
    elif f["ratio_matches"] > 3:
        indicators.append((3.0, min(1.0, f["ratio_matches"] / 8.0)))
        reasons.append(f"{f['ratio_matches']} integer-ratio matches — structured pulse train")
    else:
        indicators.append((3.0, 0.1))
        reasons.append("No mathematical timing structure detected")

    # Low timing entropy (regular)
    if f["timing_entropy"] < 0.4:
        indicators.append((2.0, 1.0 - f["timing_entropy"]))
        reasons.append(f"Low timing entropy ({f['timing_entropy']:.2f}) — highly ordered signal")
    else:
        indicators.append((2.0, 0.2))

    # Sharp morphology
    if f["any_fast_ingress"]:
        indicators.append((1.5, 0.7))
        reasons.append("Sharp ingress — pulse-like edge")

    return _weighted_score(indicators), reasons


def _score_stellar_engine(f: dict) -> tuple[float, list[str]]:
    """
    Stellar engine / Shkadov thruster — large reflector partially
    occluding the star to redirect radiation for propulsion.

    Expected: persistent or very long duration dimming, single event
    or slowly varying, asymmetric.
    """
    reasons = []
    indicators = []

    # Long duration
    if f["dur_mean"] > 20:
        indicators.append((2.5, min(1.0, f["dur_mean"] / 48.0)))
        reasons.append(f"Very long events ({f['dur_mean']:.1f}h avg) — persistent occultation")
    else:
        indicators.append((2.5, 0.15))
        reasons.append(f"Short events ({f['dur_mean']:.1f}h) — inconsistent with stationary reflector")

    # Few events (one big structure, not many)
    if f["n_events"] <= 2:
        indicators.append((2.0, 0.7))
        reasons.append(f"Single/few events ({f['n_events']}) — monolithic structure")
    else:
        indicators.append((2.0, 0.2))

    # Deep dimming
    if f["depth_max"] > 5000:
        indicators.append((1.5, min(1.0, f["depth_max"] / 20000.0)))
        reasons.append(f"Significant occultation ({f['depth_max']:.0f} ppm) — large-scale structure")
    else:
        indicators.append((1.5, 0.3))

    # IR excess from redirected energy
    if f["ir_available"] and f["ir_score"] > 0.2:
        indicators.append((2.0, min(1.0, f["ir_score"] * 1.5)))
        reasons.append(f"Infrared excess — redirected stellar radiation")
    elif f["ir_available"]:
        indicators.append((1.0, 0.2))

    return _weighted_score(indicators), reasons


def _score_solar_collectors(f: dict) -> tuple[float, list[str]]:
    """
    Orbital solar collector swarm — multiple structures in various
    orbits harvesting stellar energy.

    Expected: multiple events with moderate depth variation, some
    substructure, possible quasi-periodicity, IR excess from
    waste heat.
    """
    reasons = []
    indicators = []

    # Multiple events (swarm = many objects)
    if f["n_events"] >= 4:
        indicators.append((2.5, min(1.0, f["n_events"] / 8.0)))
        reasons.append(f"{f['n_events']} events — multi-body transit swarm")
    elif f["n_events"] >= 2:
        indicators.append((2.5, 0.4))
    else:
        indicators.append((2.5, 0.1))

    # Moderate depth variation (different orbital distances / sizes)
    if 0.3 < f["depth_cv"] < 1.0:
        indicators.append((2.0, 0.7))
        reasons.append(f"Moderate depth scatter (CV={f['depth_cv']:.2f}) — varied collector sizes/orbits")
    else:
        indicators.append((2.0, 0.3))

    # IR excess from energy processing
    if f["ir_available"] and f["ir_score"] > 0.2:
        indicators.append((2.0, min(1.0, f["ir_score"] * 1.5)))
        reasons.append(f"Infrared excess — waste heat from energy processing")
    elif f["ir_available"]:
        indicators.append((1.0, 0.2))

    # Substructure within events
    if f["any_substructure"]:
        indicators.append((1.5, 0.7))
        reasons.append("Internal substructure — multiple objects within transit window")

    # Quasi-periodic (collectors in similar orbits)
    if 3 < f["best_sde"] < 8:
        indicators.append((1.0, 0.5))
        reasons.append(f"Quasi-periodic (SDE={f['best_sde']:.1f}) — loosely shared orbital plane")

    return _weighted_score(indicators), reasons


# ---------------------------------------------------------------------------
# Hypothesis definitions
# ---------------------------------------------------------------------------

HYPOTHESIS_DEFINITIONS: list[dict[str, Any]] = [
    # --- Natural ---
    {
        "id": "debris_disk",
        "name": "Circumstellar debris disk",
        "category": "natural",
        "description": (
            "A cloud of dust and rocky debris orbiting the star, likely from "
            "collisions between planetesimals or a disrupted body. Causes "
            "irregular, aperiodic dimming with variable depth. May produce "
            "detectable infrared excess from warm dust grains."
        ),
        "score_fn": _score_debris_disk,
    },
    {
        "id": "disintegrating_planet",
        "name": "Disintegrating planet",
        "category": "natural",
        "description": (
            "A small rocky planet in an extremely close orbit being destroyed "
            "by intense stellar radiation. Produces a comet-like dust tail "
            "that varies in density orbit-to-orbit, causing depth variations "
            "with underlying periodicity."
        ),
        "score_fn": _score_disintegrating_planet,
    },
    {
        "id": "ring_system",
        "name": "Circumplanetary ring system",
        "category": "natural",
        "description": (
            "A giant planet with an extensive ring system (like Saturn but much "
            "larger). Rings cause deep, long-duration transits with possible "
            "substructure from ring gaps. The archetype is J1407b, whose rings "
            "blocked up to 95%% of its host star's light."
        ),
        "score_fn": _score_ring_system,
    },
    {
        "id": "trojan_swarm",
        "name": "Trojan asteroid swarm",
        "category": "natural",
        "description": (
            "A cloud of co-orbital asteroids sharing a planet's orbit "
            "(at the L4/L5 Lagrange points). Would produce multiple shallow "
            "transit events at the same period with consistent depths."
        ),
        "score_fn": _score_trojan_swarm,
    },
    {
        "id": "brown_dwarf",
        "name": "Brown dwarf transit",
        "category": "natural",
        "description": (
            "A substellar object (13–80 Jupiter masses) too small for hydrogen "
            "fusion. Produces deep, flat-bottomed, strictly periodic transits. "
            "Similar in radius to Jupiter but much denser."
        ),
        "score_fn": _score_brown_dwarf,
    },
    {
        "id": "exomoon",
        "name": "Exomoon transit",
        "category": "natural",
        "description": (
            "A large moon orbiting an exoplanet, causing additional shallow "
            "dips near the main planetary transit. The moon's changing orbital "
            "position produces transit-timing and depth variations."
        ),
        "score_fn": _score_exomoon,
    },
    {
        "id": "instrument_artifact",
        "name": "Instrumental artifact",
        "category": "natural",
        "description": (
            "An unidentified systematic effect in the TESS detector — stray "
            "light, thermal settling, CCD charge transfer issues, or data "
            "processing artifacts that mimic astrophysical signals."
        ),
        "score_fn": _score_instrument_artifact,
    },
    # --- Artificial / Technosignature ---
    {
        "id": "megastructure",
        "name": "Partial Dyson sphere / megastructure",
        "category": "artificial",
        "description": (
            "A hypothetical large-scale engineering project by an advanced "
            "civilization partially enclosing a star to harvest its energy "
            "(Kardashev Type II). Causes irregular deep dimming and should "
            "produce detectable infrared waste heat. The leading artificial "
            "hypothesis for Boyajian's Star (KIC 8462852)."
        ),
        "score_fn": _score_megastructure,
    },
    {
        "id": "transit_beacon",
        "name": "Artificial transit beacon",
        "category": "artificial",
        "description": (
            "An engineered object deliberately placed in orbit around a star "
            "to produce an unmistakably artificial transit signature — designed "
            "to be detected by civilizations like ours surveying for transits. "
            "Expected markers: impossibly sharp edges, perfect bilateral symmetry, "
            "and timing that encodes mathematical information."
        ),
        "score_fn": _score_transit_beacon,
    },
    {
        "id": "clarke_exobelt",
        "name": "Clarke exobelt",
        "category": "artificial",
        "description": (
            "A dense belt of artificial satellites in synchronous orbit around "
            "an exoplanet (analogous to Earth's geostationary belt, but vastly "
            "denser). Causes shallow, periodic transit signatures with consistent "
            "depth and possible fine substructure from belt density variations."
        ),
        "score_fn": _score_clarke_exobelt,
    },
    {
        "id": "laser_beacon",
        "name": "Laser / optical communication beacon",
        "category": "artificial",
        "description": (
            "A powerful directed-energy beacon producing brief, sharp brightness "
            "variations on a mathematically structured schedule — an interstellar "
            "lighthouse. Expected: short-duration pulses with low timing entropy "
            "and intervals encoding mathematical constants or prime numbers."
        ),
        "score_fn": _score_laser_beacon,
    },
    {
        "id": "stellar_engine",
        "name": "Stellar engine / Shkadov thruster",
        "category": "artificial",
        "description": (
            "A colossal mirror or reflector held in a stationary position "
            "relative to a star, redirecting stellar radiation to produce "
            "thrust and move the entire star system. Causes persistent, "
            "long-duration asymmetric dimming from the monolithic structure."
        ),
        "score_fn": _score_stellar_engine,
    },
    {
        "id": "solar_collectors",
        "name": "Orbital solar collector swarm",
        "category": "artificial",
        "description": (
            "A fleet of energy-harvesting structures in various orbits around "
            "a star — an early-stage Dyson swarm. Produces multiple transit "
            "events with moderate depth variation, quasi-periodic timing, "
            "and infrared waste heat."
        ),
        "score_fn": _score_solar_collectors,
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_hypotheses(
    unknown_events: list[dict],
    techno_result: dict,
    period_result: dict,
) -> list[dict]:
    """
    Produce a ranked list of hypothesis explanations for UNKNOWN events.

    Returns a list of dicts sorted by descending match score, each with:
        id, name, category, description, score, reasoning
    """
    if not unknown_events:
        return []

    features = _extract_features(unknown_events, techno_result, period_result)

    hypotheses = []
    for h_def in HYPOTHESIS_DEFINITIONS:
        try:
            score, reasoning = h_def["score_fn"](features)
        except Exception as e:
            logger.warning("Hypothesis %s failed: %s", h_def["id"], e)
            score, reasoning = 0.0, [f"Scoring failed: {e}"]

        hypotheses.append({
            "id": h_def["id"],
            "name": h_def["name"],
            "category": h_def["category"],
            "description": h_def["description"],
            "score": round(float(score), 3),
            "reasoning": reasoning,
        })

    hypotheses.sort(key=lambda h: h["score"], reverse=True)
    return hypotheses
