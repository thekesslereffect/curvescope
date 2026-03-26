import logging
import threading

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

WINDOW_SIZE = 128
STRIDE = 32
LATENT_DIM = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_cache: dict[str, tuple[nn.Module, float]] = {}
_cache_lock = threading.Lock()
_stats_cache: dict[str, tuple[dict, float]] = {}
_stats_lock = threading.Lock()

_ENC_FLAT = 128 * (WINDOW_SIZE // 4)


class LightCurveAutoencoder(nn.Module):
    """
    1D convolutional autoencoder with a strict bottleneck (no skip connections).

    Uses strided convolutions instead of MaxPool so the model *learns* what to
    downsample rather than blindly taking the max — this preserves more useful
    information while still forcing everything through the bottleneck.

    For anomaly detection the bottleneck is essential: the model can only
    reconstruct patterns present in training data.  Unseen patterns (transits,
    flares) get high reconstruction error → high anomaly score.
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(_ENC_FLAT, LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, _ENC_FLAT),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (128, WINDOW_SIZE // 4)),
            nn.ConvTranspose1d(128, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.ConvTranspose1d(32, 1, 7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _get_model(weights_path: Path) -> nn.Module:
    """Load model once; reload if the file has been updated (new training run)."""
    key = str(weights_path)
    mtime = weights_path.stat().st_mtime if weights_path.exists() else 0.0
    with _cache_lock:
        if key in _model_cache:
            cached_model, cached_mtime = _model_cache[key]
            if cached_mtime == mtime:
                return cached_model
    model = LightCurveAutoencoder().to(device)
    if weights_path.exists():
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            logger.info("Loaded autoencoder weights from %s", weights_path)
        except RuntimeError as exc:
            logger.warning("Weights at %s are incompatible (architecture changed?) — retrain required: %s",
                           weights_path, exc)
    else:
        logger.warning("No weights at %s — using untrained model (scores will be uncalibrated)", weights_path)
    model.eval()
    with _cache_lock:
        _model_cache[key] = (model, mtime)
    return model


def _get_training_stats() -> dict | None:
    """Load training error statistics (mean, std, p99) from the stats file."""
    stats_path = settings.model_stats_path
    if not stats_path.exists():
        return None
    key = str(stats_path)
    mtime = stats_path.stat().st_mtime
    with _stats_lock:
        if key in _stats_cache:
            cached, cached_mtime = _stats_cache[key]
            if cached_mtime == mtime:
                return cached
    try:
        data = np.load(stats_path)
        stats = {
            "mean_error": float(data["mean_error"]),
            "std_error": float(data["std_error"]),
            "p99_error": float(data["p99_error"]),
        }
        with _stats_lock:
            _stats_cache[key] = (stats, mtime)
        return stats
    except Exception:
        return None


def score_light_curve(flux: list, weights_path: str) -> dict:
    """
    Slide a window across the light curve and compute reconstruction error.

    Two complementary scores are produced:

    1. **Local score** (per-point): within-curve MAD-based z-score that finds
       which parts of *this* curve are unusual relative to the rest.
       Good for spotting isolated transits/flares in otherwise quiet curves.

    2. **Global score**: compares this curve's median reconstruction error
       against the training distribution. Catches curves that are *entirely*
       unusual (e.g. eclipsing binaries, pulsating variables) even if
       internally consistent.

    The final anomaly_score = max(local_95th, global_score).
    """
    weights_p = Path(weights_path)
    model = _get_model(weights_p)

    arr = np.array(flux, dtype=np.float32)
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        arr[nan_mask] = np.nanmedian(arr)

    raw_errors = np.zeros(len(arr))
    counts = np.zeros(len(arr))

    starts = list(range(0, len(arr) - WINDOW_SIZE, STRIDE))
    last_start = max(0, len(arr) - WINDOW_SIZE)
    if not starts or starts[-1] < last_start:
        starts.append(last_start)
    BATCH = 128
    window_errors = []

    with torch.no_grad():
        for b0 in range(0, len(starts), BATCH):
            batch_starts = starts[b0 : b0 + BATCH]
            windows = np.empty((len(batch_starts), WINDOW_SIZE), dtype=np.float32)
            for idx, start in enumerate(batch_starts):
                w = arr[start : start + WINDOW_SIZE]
                w = (w - w.mean()) / (w.std() + 1e-8)
                windows[idx] = w

            x = torch.from_numpy(windows).unsqueeze(1).to(device)
            recon = model(x).squeeze(1).cpu().numpy()
            errors = ((windows - recon) ** 2).mean(axis=1)
            window_errors.extend(errors.tolist())

            for idx, start in enumerate(batch_starts):
                raw_errors[start : start + WINDOW_SIZE] += errors[idx]
                counts[start : start + WINDOW_SIZE] += 1

            del x, recon
        if device.type == "cuda":
            torch.cuda.empty_cache()

    counts = np.maximum(counts, 1)
    raw_errors = raw_errors / counts

    # --- Local score: within-curve relative scoring (MAD) ---
    med_err = np.median(raw_errors)
    mad_err = np.median(np.abs(raw_errors - med_err))
    robust_std = mad_err * 1.4826 + 1e-8
    local_z = (raw_errors - med_err) / robust_std
    local_scores = 1.0 / (1.0 + np.exp(-(local_z - 2.5)))
    local_95 = float(np.percentile(local_scores, 95)) if len(local_scores) > 0 else 0.0

    # --- Global score: flag curves whose median error is multiples of p99 ---
    # Only curves the model fundamentally cannot reconstruct (eclipsing binaries,
    # pulsators) have median errors 3x+ the training p99. Sigmoid centered at 3x
    # means a curve needs to be dramatically worse than any training sample.
    global_score = 0.0
    train_stats = _get_training_stats()
    if train_stats and len(window_errors) > 0:
        curve_median_err = float(np.median(window_errors))
        p99 = train_stats["p99_error"] + 1e-8
        ratio = curve_median_err / p99
        global_score = float(1.0 / (1.0 + np.exp(-(ratio - 3.0))))

    combined = max(local_95, global_score)

    return {
        "score_per_point": local_scores.tolist(),
        "combined_score": combined,
        "local_score": local_95,
        "global_novelty": global_score,
    }
