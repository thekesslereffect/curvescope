"""
Training script for the light curve autoencoder.
CLI: cd backend && .\\venv\\Scripts\\activate && python -m pipeline.train
API: POST /api/train (see routers/settings.py)
"""

from __future__ import annotations

import sys
import os
import threading
import logging
from dataclasses import dataclass, field, replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightkurve as lk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from pipeline.autoencoder import LightCurveAutoencoder, WINDOW_SIZE, device
from pipeline.clean import detrend_flux, normalize_flux, remove_outliers
from config import settings, load_training_targets

logger = logging.getLogger(__name__)

os.environ["LIGHTKURVE_CACHE_DIR"] = str(settings.mast_cache_dir)


def _get_training_tics() -> list[str]:
    """Read TIC IDs from the training_targets.json file."""
    targets = load_training_targets()
    return [f"TIC {t['tic_id']}" for t in targets]

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 3e-4


@dataclass
class TrainingHyperParams:
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_targets: int | None = None  # None = use full target list
    custom_tics: list[str] | None = None  # if set, used instead of training_targets.json

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_targets": self.max_targets,
            "custom_tics_count": len(self.custom_tics) if self.custom_tics else None,
        }


@dataclass
class TrainingMonitor:
    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    phase: str = "idle"
    epoch: int = 0
    total_epochs: int = DEFAULT_EPOCHS
    loss: float | None = None
    targets_fetched: int = 0
    total_targets: int = 0
    windows_count: int = 0
    message: str = ""
    error: str | None = None
    hyperparams: TrainingHyperParams = field(default_factory=TrainingHyperParams)
    loss_history: list[tuple[int, float]] = field(default_factory=list)
    reconstruction_samples: list[dict] = field(default_factory=list)
    error_histogram: dict = field(default_factory=dict)
    network_activations: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        self.phase = "idle"
        self.epoch = 0
        self.total_epochs = DEFAULT_EPOCHS
        self.loss = None
        self.targets_fetched = 0
        self.total_targets = len(_get_training_tics())
        self.windows_count = 0
        self.message = ""
        self.error = None
        self.hyperparams = TrainingHyperParams()
        self.loss_history = []
        self.reconstruction_samples = []
        self.error_histogram = {}
        self.network_activations = []

    def to_dict(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "phase": self.phase,
                "epoch": self.epoch,
                "total_epochs": self.total_epochs,
                "loss": self.loss,
                "targets_fetched": self.targets_fetched,
                "total_targets": self.total_targets,
                "windows_count": self.windows_count,
                "message": self.message,
                "error": self.error,
                "hyperparams": self.hyperparams.to_dict(),
                "loss_history": [
                    {"epoch": e[0], "loss": round(e[1], 6), **({"val_loss": round(e[2], 6)} if len(e) > 2 else {})}
                    for e in self.loss_history
                ],
                "reconstruction_samples": self.reconstruction_samples,
                "error_histogram": self.error_histogram,
                "network_activations": self.network_activations,
            }


training_monitor = TrainingMonitor()
_train_thread: threading.Thread | None = None


def get_default_hyperparams() -> TrainingHyperParams:
    return TrainingHyperParams()


def prepare_windows(flux_list: list) -> np.ndarray:
    arr = np.array(flux_list, dtype=np.float32)
    arr = arr[~np.isnan(arr)]
    windows = []
    for i in range(0, len(arr) - WINDOW_SIZE, 32):
        w = arr[i : i + WINDOW_SIZE]
        w = (w - w.mean()) / (w.std() + 1e-8)
        windows.append(w)
    return np.array(windows) if windows else np.empty((0, WINDOW_SIZE))


def _build_reconstruction_samples(originals: np.ndarray, reconstructions: np.ndarray, errors: np.ndarray) -> list[dict]:
    """Pick representative windows: 3 best, 3 random, 3 worst reconstructions."""
    n = len(errors)
    if n == 0:
        return []

    sorted_idx = np.argsort(errors)
    best_idx = sorted_idx[:3].tolist()
    worst_idx = sorted_idx[-3:].tolist()

    rng = np.random.default_rng(42)
    mid_start = max(3, n // 4)
    mid_end = min(n - 3, 3 * n // 4)
    if mid_end > mid_start:
        random_idx = rng.choice(np.arange(mid_start, mid_end), size=min(3, mid_end - mid_start), replace=False).tolist()
    else:
        random_idx = sorted_idx[n // 2 : n // 2 + 3].tolist()

    samples = []
    for label, indices in [("best", best_idx), ("typical", random_idx), ("worst", worst_idx)]:
        for i in indices:
            samples.append({
                "label": label,
                "original": np.round(originals[i], 4).tolist(),
                "reconstructed": np.round(reconstructions[i], 4).tolist(),
                "error": round(float(errors[i]), 6),
            })
    return samples


def _capture_activations(model: nn.Module, sample_windows: np.ndarray) -> list[dict]:
    """Run sample windows through the model and capture layer-by-layer activations."""
    activations: list[tuple[str, np.ndarray]] = []
    hooks = []

    layer_names = {
        "encoder.2": "Encoder Conv1 (32 filters)",
        "encoder.6": "Encoder Conv2 (64 filters)",
        "encoder.10": "Encoder Conv3 (128 filters)",
        "encoder.12": "Bottleneck (64-dim)",
        "decoder.1": "Decoder Linear",
        "decoder.5": "Decoder DeConv1 (64 filters)",
        "decoder.9": "Decoder DeConv2 (32 filters)",
        "decoder.11": "Output (1 channel)",
    }

    def make_hook(name):
        def hook_fn(_module, _input, output):
            arr = output.detach().cpu().numpy()
            activations.append((name, arr))
        return hook_fn

    for mod_name, module in model.named_modules():
        if mod_name in layer_names:
            hooks.append(module.register_forward_hook(make_hook(layer_names[mod_name])))

    x = torch.from_numpy(sample_windows).unsqueeze(1).to(next(model.parameters()).device)
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    MAX_CHANNELS = 32
    MAX_SPATIAL = 64

    result = []
    for name, arr in activations:
        if arr.ndim == 3:
            act = arr[0]
            channels, spatial = act.shape
            ch_step = max(1, channels // MAX_CHANNELS)
            sp_step = max(1, spatial // MAX_SPATIAL)
            act_ds = act[::ch_step, ::sp_step]
            result.append({
                "name": name,
                "shape": [channels, spatial],
                "data": np.round(act_ds, 3).tolist(),
                "type": "heatmap",
            })
        elif arr.ndim == 2:
            vec = arr[0]
            result.append({
                "name": name,
                "shape": [len(vec)],
                "data": np.round(vec, 3).tolist(),
                "type": "bar",
            })
    return result


def _build_error_histogram(errors: np.ndarray, n_bins: int = 40) -> dict:
    """Build a histogram of reconstruction errors for the frontend."""
    if len(errors) == 0:
        return {}
    counts, edges = np.histogram(errors, bins=n_bins)
    bin_centers = ((edges[:-1] + edges[1:]) / 2)
    return {
        "bin_centers": np.round(bin_centers, 6).tolist(),
        "counts": counts.tolist(),
        "mean": round(float(errors.mean()), 6),
        "std": round(float(errors.std()), 6),
        "p99": round(float(np.percentile(errors, 99)), 6),
        "total_windows": int(len(errors)),
    }


def _run_training_inner() -> None:
    mon = training_monitor
    try:
        with mon.lock:
            hp = replace(mon.hyperparams)
            if hp.custom_tics:
                base_targets = [f"TIC {t}" for t in hp.custom_tics]
            else:
                base_targets = _get_training_tics()
            targets = base_targets[: hp.max_targets] if hp.max_targets is not None else base_targets
            mon.total_targets = len(targets)
            mon.total_epochs = hp.epochs
            mon.phase = "fetching"
            source = "scan data" if hp.custom_tics else "built-in list"
            mon.message = f"Loading {len(targets)} targets ({source})"
            mon.loss_history = []

        from pipeline import training_cache
        from pipeline.fetch import clear_mast_downloads

        all_windows = []
        cache_hits = 0
        logger.info("Loading %s training targets (cache_dir=%s)", len(targets), settings.training_cache_dir)
        logger.info("Using device: %s", device)
        logger.info("Hyperparams: epochs=%s batch=%s lr=%s", hp.epochs, hp.batch_size, hp.learning_rate)

        for idx, tic in enumerate(targets, 1):
            with mon.lock:
                if not mon.running:
                    mon.phase = "idle"
                    mon.message = "Cancelled"
                    return
            try:
                cached = training_cache.load(tic)
                if cached is not None and len(cached) > 0:
                    all_windows.append(cached)
                    cache_hits += 1
                    logger.info("  [%s/%s] %s — cache hit (%d windows)", idx, len(targets), tic, len(cached))
                    with mon.lock:
                        mon.targets_fetched = idx
                        mon.windows_count = sum(len(w) for w in all_windows)
                    continue

                logger.info("  [%s/%s] %s — fetching (S3 then MAST fallback)...", idx, len(targets), tic)
                results = lk.search_lightcurve(tic, mission="TESS", author="SPOC", exptime=120)
                if len(results) == 0:
                    logger.info("    No SPOC data, skipping")
                    with mon.lock:
                        mon.targets_fetched = idx
                    continue

                lc_collection = None
                try:
                    from pipeline.s3_fetch import extract_s3_urls_from_search, download_fits_parallel_sync
                    pairs = extract_s3_urls_from_search(results)
                    pairs = [(u, f) for u, f in pairs if f.endswith("_lc.fits")]
                    if pairs:
                        s3_cache = settings.mast_cache_dir / "s3_cache"
                        paths = download_fits_parallel_sync(pairs, s3_cache)
                        if paths:
                            lcs = [obj for p in paths if (obj := lk.read(str(p))) and hasattr(obj, "normalize")]
                            if lcs:
                                lc_collection = lk.LightCurveCollection(lcs)
                            logger.info("    S3: %d/%d files", len(paths), len(pairs))
                except Exception as e:
                    logger.debug("    S3 failed (%s), using lightkurve", e)
                    lc_collection = None

                if lc_collection is None:
                    lc_collection = results.download_all(cache=True)
                only_lcs = [item for item in lc_collection if isinstance(item, lk.LightCurve)]
                if not only_lcs:
                    logger.info("    No LC objects in collection, skipping")
                    with mon.lock:
                        mon.targets_fetched = idx
                    continue
                lc_collection = lk.LightCurveCollection(only_lcs)
                lc = lc_collection.stitch()
                lc = lc.remove_nans()
                flux = normalize_flux(lc.flux.value.tolist())
                flux, _ = remove_outliers(flux)
                detrended = detrend_flux(lc.time.value.tolist(), flux)
                windows = prepare_windows(detrended)
                if len(windows) > 0:
                    all_windows.append(windows)
                    training_cache.save(tic, windows)
                    logger.info("    %s windows extracted & cached", len(windows))
                with mon.lock:
                    mon.targets_fetched = idx
                    mon.windows_count = sum(len(w) for w in all_windows)
            except Exception as e:
                logger.warning("    Skipped: %s", e)
                with mon.lock:
                    mon.targets_fetched = idx

        clear_mast_downloads()
        logger.info("Target loading complete: %d cache hits, %d fetched from MAST",
                     cache_hits, len(targets) - cache_hits)

        if not all_windows:
            with mon.lock:
                mon.phase = "error"
                mon.error = "No training data collected"
                mon.message = mon.error
                mon.running = False
            logger.error("No training data collected")
            return

        n_tics = len(all_windows)
        rng = np.random.default_rng(42)

        if n_tics >= 5:
            n_val_tics = max(1, int(n_tics * 0.2))
            tic_perm = rng.permutation(n_tics)
            val_tic_idx = sorted(tic_perm[:n_val_tics])
            train_tic_idx = sorted(tic_perm[n_val_tics:])
            val_arr = np.concatenate([all_windows[i] for i in val_tic_idx], axis=0)
            train_arr = np.concatenate([all_windows[i] for i in train_tic_idx], axis=0)
            logger.info("TIC-level split: %d train TICs (%d windows), %d val TICs (%d windows)",
                        len(train_tic_idx), len(train_arr), len(val_tic_idx), len(val_arr))
        else:
            all_flat = np.concatenate(all_windows, axis=0)
            n_val_w = max(1, int(len(all_flat) * 0.2))
            perm = rng.permutation(len(all_flat))
            val_arr = all_flat[perm[:n_val_w]]
            train_arr = all_flat[perm[n_val_w:]]
            logger.info("Window-level split (< 5 TICs): %d train, %d val windows",
                        len(train_arr), len(val_arr))

        all_arr = np.concatenate(all_windows, axis=0)
        n_total = len(all_arr)

        with mon.lock:
            mon.phase = "training"
            mon.message = f"Training on {len(train_arr)} windows ({len(val_arr)} held out for validation)"
            mon.windows_count = n_total

        X_train = torch.tensor(train_arr).unsqueeze(1).to(device)
        X_val = torch.tensor(val_arr).unsqueeze(1).to(device)
        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=False, drop_last=False)

        model = LightCurveAutoencoder().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.epochs, eta_min=hp.learning_rate * 0.01)

        best_val_loss = float("inf")
        best_state = None

        viz_sample = all_arr[len(all_arr) // 2 : len(all_arr) // 2 + 1].astype(np.float32)
        viz_original = np.round(viz_sample[0], 4).tolist()

        logger.info("Training autoencoder (%s epochs, cosine LR %s→%s, %d train / %d val windows)",
                     hp.epochs, hp.learning_rate, hp.learning_rate * 0.01, len(train_arr), len(val_arr))
        for epoch in range(hp.epochs):
            with mon.lock:
                if not mon.running:
                    mon.phase = "idle"
                    mon.message = "Cancelled during training"
                    return
            model.train()
            total_loss = 0.0
            for (batch,) in train_loader:
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            train_avg = total_loss / max(len(train_loader), 1)

            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    val_total += criterion(model(batch), batch).item()
            val_avg = val_total / max(len(val_loader), 1)

            if val_avg < best_val_loss:
                best_val_loss = val_avg
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            try:
                live_acts = _capture_activations(model, viz_sample)
                x_viz = torch.from_numpy(viz_sample).unsqueeze(1).to(device)
                with torch.no_grad():
                    recon_viz = model(x_viz).squeeze(1).cpu().numpy()
                live_recon = {
                    "label": "live",
                    "original": viz_original,
                    "reconstructed": np.round(recon_viz[0], 4).tolist(),
                    "error": round(float(((viz_sample[0] - recon_viz[0]) ** 2).mean()), 6),
                }
                del x_viz, recon_viz
            except Exception:
                live_acts = []
                live_recon = None

            with mon.lock:
                mon.epoch = epoch + 1
                mon.loss = round(train_avg, 6)
                mon.message = f"Epoch {epoch + 1}/{hp.epochs} · train {train_avg:.6f} · val {val_avg:.6f}"
                mon.loss_history.append((epoch + 1, float(train_avg), float(val_avg)))
                mon.network_activations = live_acts
                if live_recon:
                    mon.reconstruction_samples = [live_recon]
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info("  Epoch %s/%s  train=%.6f  val=%.6f  lr=%.2e",
                            epoch + 1, hp.epochs, train_avg, val_avg, scheduler.get_last_lr()[0])

        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info("Restored best checkpoint (val_loss=%.6f)", best_val_loss)

        model.eval()
        X_all = torch.tensor(all_arr).unsqueeze(1).to(device)
        eval_dataset = TensorDataset(X_all)
        eval_loader = DataLoader(eval_dataset, batch_size=hp.batch_size, shuffle=False, drop_last=False)

        all_errors: list[float] = []
        all_originals: list[np.ndarray] = []
        all_recons: list[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in eval_loader:
                output = model(batch)
                per_window = ((batch - output) ** 2).mean(dim=(1, 2))
                all_errors.extend(per_window.cpu().numpy().tolist())
                all_originals.append(batch.squeeze(1).cpu().numpy())
                all_recons.append(output.squeeze(1).cpu().numpy())
        err_arr = np.array(all_errors)
        orig_all = np.concatenate(all_originals, axis=0)
        recon_all = np.concatenate(all_recons, axis=0)

        samples = _build_reconstruction_samples(orig_all, recon_all, err_arr)
        hist = _build_error_histogram(err_arr)

        median_idx = int(np.argsort(err_arr)[len(err_arr) // 2])
        sample_for_activations = orig_all[median_idx : median_idx + 1].astype(np.float32)
        try:
            activations = _capture_activations(model, sample_for_activations)
        except Exception as e:
            logger.warning("Activation capture failed: %s", e)
            activations = []

        with mon.lock:
            mon.reconstruction_samples = samples
            mon.error_histogram = hist
            mon.network_activations = activations
            mon.phase = "saving"
            mon.message = "Saving weights and stats"
        del orig_all, recon_all, all_originals, all_recons

        weights_path = settings.model_weights_path
        stats_path = settings.model_stats_path
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), weights_path)
        np.savez(
            stats_path,
            mean_error=err_arr.mean(),
            std_error=err_arr.std(),
            p99_error=np.percentile(err_arr, 99),
        )

        with mon.lock:
            mon.phase = "complete"
            mon.message = f"Saved to {weights_path}"
            mon.running = False
        logger.info("Model saved to %s", weights_path)
    except Exception as e:
        logger.exception("Training failed")
        with mon.lock:
            mon.phase = "error"
            mon.error = str(e)
            mon.message = str(e)
            mon.running = False


def apply_hyperparam_overrides(base: TrainingHyperParams, overrides: dict | None) -> TrainingHyperParams:
    if not overrides:
        return base
    epochs = int(overrides["epochs"]) if overrides.get("epochs") is not None else base.epochs
    batch_size = int(overrides["batch_size"]) if overrides.get("batch_size") is not None else base.batch_size
    learning_rate = float(overrides["learning_rate"]) if overrides.get("learning_rate") is not None else base.learning_rate
    max_targets = base.max_targets
    if "max_targets" in overrides:
        v = overrides["max_targets"]
        max_targets = None if v is None else int(v)
    custom_tics = overrides.get("custom_tics")
    return TrainingHyperParams(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_targets=max_targets,
        custom_tics=custom_tics,
    )


def run_training_async(overrides: dict | None = None) -> tuple[bool, str]:
    global _train_thread
    mon = training_monitor
    with mon.lock:
        if mon.running:
            return False, "Training already running"
        mon.reset()
        mon.hyperparams = apply_hyperparam_overrides(TrainingHyperParams(), overrides)
        mon.running = True

    _train_thread = threading.Thread(target=_run_training_inner, daemon=True, name="autoencoder-train")
    _train_thread.start()
    return True, "Training started"


def get_training_status() -> dict:
    return training_monitor.to_dict()


def train() -> None:
    """CLI entry: blocking training without async monitor conflicts."""
    with training_monitor.lock:
        if training_monitor.running:
            logger.error("Training already in progress via API")
            return
        training_monitor.reset()
        training_monitor.running = True
    try:
        _run_training_inner()
    finally:
        with training_monitor.lock:
            if training_monitor.phase not in ("complete", "error"):
                training_monitor.running = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    train()
