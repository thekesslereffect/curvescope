# TESS Anomaly Detector — Full Build Specification

**Version:** 1.1  
**Platform:** Windows 11 (local development)  
**Stack:** Next.js 14 · Python 3.11 · FastAPI · PostgreSQL  
**Purpose:** Pull real TESS light curve data, run an autoencoder anomaly detector, visualize results, and flag events that cannot be explained by known astrophysical models.

---

## 1. What This App Actually Does

1. User enters a star name or TIC ID (TESS Input Catalog number)
2. Backend fetches real light curve data from NASA MAST via `lightkurve`
3. Data is cleaned, detrended, and normalized
4. An autoencoder neural network scores each time window for anomalousness
5. A BLS (Box Least Squares) periodogram finds orbital periods
6. A continuous wavelet transform maps which periodicities are active at which times, used to automatically reject TESS instrumental systematics
7. Centroid shift analysis on the Target Pixel File determines whether flagged events originate from the target star or a background contaminant
8. Events above a threshold are classified into known categories (transit, flare, exocomet, etc.) or flagged as UNKNOWN — only after passing both the wavelet and centroid filters
9. Results are displayed as interactive charts with a ranked event list
10. User can export flagged events and submit to TFOP (TESS Follow-up Observing Program)

---

## 2. Prerequisites — Windows Setup

Install these in order before anything else.

### 2.1 Core Tools

| Tool | Version | Download |
|------|---------|----------|
| Node.js | 20 LTS | https://nodejs.org |
| Python | 3.11.x | https://python.org/downloads — check "Add to PATH" during install |
| Git | latest | https://git-scm.com |
| PostgreSQL | 16 | https://postgresql.org/download/windows |
| pgAdmin 4 | bundled with PostgreSQL | (comes with PostgreSQL installer) |

### 2.2 Windows-Specific Notes

```powershell
# After installing Python, verify in PowerShell:
python --version        # should say 3.11.x
pip --version           # should say 23.x or higher

# If python command not found, add to PATH manually:
# Settings → System → Advanced → Environment Variables
# Add: C:\Users\<you>\AppData\Local\Programs\Python\Python311\
# Add: C:\Users\<you>\AppData\Local\Programs\Python\Python311\Scripts\

# Install virtualenv globally
pip install virtualenv
```

### 2.3 PostgreSQL Setup

During PostgreSQL installation:
- Port: `5432` (default)
- Password: set something you'll remember, e.g. `tesslocal`
- Locale: default

After install, open pgAdmin and create a database:
```
Database name: tess_anomaly
Owner: postgres
```

---

## 3. Project Structure

```
tess-anomaly/
│
├── frontend/                   # Next.js app
│   ├── app/
│   │   ├── page.tsx            # Home — search + recent analyses
│   │   ├── analyze/
│   │   │   └── [tic]/
│   │   │       └── page.tsx    # Analysis results for a target
│   │   ├── targets/
│   │   │   └── page.tsx        # Saved targets list
│   │   └── layout.tsx
│   ├── components/
│   │   ├── LightCurveChart.tsx
│   │   ├── AnomalyScoreChart.tsx
│   │   ├── PeriodogramChart.tsx
│   │   ├── PhaseFoldChart.tsx
│   │   ├── WaveletChart.tsx        # NEW — time-period heatmap
│   │   ├── CentroidChart.tsx       # NEW — centroid drift plot
│   │   ├── EventFlagList.tsx
│   │   ├── MetricCards.tsx
│   │   ├── SearchBar.tsx
│   │   └── StatusBadge.tsx
│   ├── lib/
│   │   ├── api.ts              # Fetch wrappers for backend
│   │   └── types.ts            # TypeScript interfaces
│   ├── public/
│   └── package.json
│
├── backend/                    # Python FastAPI app
│   ├── main.py                 # FastAPI entry point
│   ├── routers/
│   │   ├── analyze.py          # POST /analyze — main pipeline trigger
│   │   ├── targets.py          # GET/POST /targets — saved targets CRUD
│   │   └── events.py           # GET /events — flagged event queries
│   ├── pipeline/
│   │   ├── fetch.py            # lightkurve data fetching + TPF fetch
│   │   ├── clean.py            # detrending, normalization, outlier removal
│   │   ├── autoencoder.py      # PyTorch autoencoder model + scoring
│   │   ├── periodogram.py      # BLS period search via astropy
│   │   ├── wavelet.py          # NEW — continuous wavelet transform + systematic rejection
│   │   ├── centroid.py         # NEW — TPF centroid shift analysis
│   │   ├── classifier.py       # Event type classification logic (uses wavelet + centroid)
│   │   ├── technosignature.py  # Post-classification technosignature analysis
│   │   └── export.py           # CSV / TFOP format export
│   ├── models/
│   │   └── weights/            # Saved model .pt files go here
│   ├── db/
│   │   ├── database.py         # SQLAlchemy engine + session
│   │   ├── models.py           # ORM table definitions
│   │   └── migrations/         # Alembic migration scripts
│   ├── requirements.txt
│   └── .env
│
├── .gitignore
└── README.md
```

---

## 4. Backend — Python / FastAPI

### 4.1 Setup

```powershell
# From project root
cd backend
python -m virtualenv venv
.\venv\Scripts\activate        # Windows activation

pip install -r requirements.txt
```

### 4.2 requirements.txt

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
sqlalchemy==2.0.30
alembic==1.13.1
psycopg2-binary==2.9.9
python-dotenv==1.0.1
pydantic==2.7.1

# Astronomy
lightkurve==2.4.2
astropy==6.0.1
astroquery==0.4.7    # SIMBAD, Gaia, WISE catalog queries
numpy==1.26.4
scipy==1.13.0
PyWavelets==1.6.0    # continuous wavelet transform

# ML
torch==2.3.0          # CPU version fine for this use case
scikit-learn==1.4.2

# Utilities
httpx==0.27.0
pandas==2.2.2
```

> **Windows note on PyTorch:** Install CPU-only version. Full GPU build is large and unnecessary for this pipeline. If you want GPU support later: https://pytorch.org/get-started/locally

### 4.3 .env

```env
DATABASE_URL=postgresql://postgres:tesslocal@localhost:5432/tess_anomaly
MODEL_WEIGHTS_PATH=./models/weights/autoencoder_v1.pt
MAST_CACHE_DIR=./data/cache
LOG_LEVEL=info
```

### 4.4 Database Schema (db/models.py)

```python
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship, DeclarativeBase
from datetime import datetime
import enum

class Base(DeclarativeBase):
    pass

class AnalysisStatus(enum.Enum):
    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"

class Target(Base):
    __tablename__ = "targets"
    id            = Column(Integer, primary_key=True)
    tic_id        = Column(String, unique=True, nullable=False, index=True)
    common_name   = Column(String)
    ra            = Column(Float)
    dec           = Column(Float)
    magnitude     = Column(Float)
    stellar_type  = Column(String)
    created_at    = Column(DateTime, default=datetime.utcnow)
    analyses      = relationship("Analysis", back_populates="target")

class Analysis(Base):
    __tablename__ = "analyses"
    id              = Column(Integer, primary_key=True)
    target_id       = Column(Integer, ForeignKey("targets.id"))
    sector          = Column(String)              # e.g. "1-3" or "all"
    status          = Column(Enum(AnalysisStatus), default=AnalysisStatus.pending)
    anomaly_score   = Column(Float)              # global score 0-1
    known_period    = Column(Float)              # days, from BLS
    flag_count      = Column(Integer, default=0)
    raw_flux        = Column(JSON)               # {time: [], flux: []}
    detrended_flux  = Column(JSON)
    score_timeline  = Column(JSON)               # {time: [], score: []}
    periodogram     = Column(JSON)               # {period: [], power: []}
    wavelet         = Column(JSON)               # {time: [], periods: [], power: [[]], tess_systematic_periods: []}
    centroid        = Column(JSON)               # {time: [], col: [], row: [], shift_flagged: bool, max_shift_arcsec: float}
    technosignature = Column(JSON)               # {entropy_score: float, morphology: {}, ir_excess: {}, summary: str}
    created_at      = Column(DateTime, default=datetime.utcnow)
    target          = relationship("Target", back_populates="analyses")
    events          = relationship("FlaggedEvent", back_populates="analysis")

class EventType(enum.Enum):
    transit          = "transit"
    asymmetric       = "asymmetric"
    depth_anomaly    = "depth_anomaly"
    non_periodic     = "non_periodic"
    exocomet         = "exocomet"
    stellar_flare    = "stellar_flare"
    stellar_spot     = "stellar_spot"
    unknown          = "unknown"

class FlaggedEvent(Base):
    __tablename__ = "flagged_events"
    id             = Column(Integer, primary_key=True)
    analysis_id    = Column(Integer, ForeignKey("analyses.id"))
    event_type     = Column(Enum(EventType))
    time_center    = Column(Float)               # BTJD timestamp
    duration_hours = Column(Float)
    depth_ppm      = Column(Float)               # parts per million
    anomaly_score  = Column(Float)               # 0-1, this event specifically
    confidence     = Column(Float)               # classifier confidence
    notes          = Column(String)              # auto-generated description
    centroid_shift_arcsec = Column(Float)        # max centroid shift during event (0 = on-target)
    systematic_match      = Column(String)       # if wavelet matched a known TESS period, name it here
    analysis       = relationship("Analysis", back_populates="events")
```

### 4.5 Data Fetching (pipeline/fetch.py)

```python
import lightkurve as lk
import numpy as np
from pathlib import Path
import os

CACHE_DIR = os.getenv("MAST_CACHE_DIR", "./data/cache")

def resolve_target(identifier: str) -> dict:
    """
    Accepts a TIC ID like 'TIC 234994474' or a name like 'K2-18'.
    Returns basic stellar parameters from the TESS Input Catalog.
    """
    results = lk.search_lightcurve(identifier, mission="TESS")
    if len(results) == 0:
        raise ValueError(f"No TESS data found for: {identifier}")
    
    # Extract TIC ID from first result
    tic_id = str(results[0].target_name)
    
    return {
        "tic_id": tic_id,
        "common_name": identifier,
        "available_sectors": list(set(results.table["sequence_number"].tolist()))
    }

def fetch_light_curve(tic_id: str, sector: str = "all") -> dict:
    """
    Downloads TESS light curve data from MAST.
    Uses 2-minute cadence (SPOC pipeline) where available,
    falls back to 20-second or FFI data.
    
    Returns dict with time array and flux array as Python lists.
    Caches downloads locally to avoid repeated MAST calls.
    """
    search_sector = None if sector == "all" else int(sector)
    
    results = lk.search_lightcurve(
        f"TIC {tic_id}",
        mission="TESS",
        author="SPOC",
        sector=search_sector,
        exptime=120    # 2-minute cadence preferred
    )
    
    if len(results) == 0:
        # Fall back to TESS-SPOC 20s or QLP
        results = lk.search_lightcurve(
            f"TIC {tic_id}",
            mission="TESS",
            sector=search_sector
        )
    
    if len(results) == 0:
        raise ValueError(f"No light curve data available for TIC {tic_id}")
    
    # Download and stitch all sectors
    lc_collection = results.download_all(
        cache=True,
        download_dir=CACHE_DIR
    )
    lc = lc_collection.stitch()
    
    # Remove NaNs
    lc = lc.remove_nans()
    
    return {
        "time": lc.time.value.tolist(),      # BTJD
        "flux": lc.flux.value.tolist(),
        "flux_err": lc.flux_err.value.tolist(),
        "sector_count": len(results)
    }
```

### 4.6 Cleaning & Detrending (pipeline/clean.py)

```python
import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import uniform_filter1d

def normalize_flux(flux: list) -> list:
    """Normalize flux to median = 1.0"""
    arr = np.array(flux)
    return (arr / np.nanmedian(arr)).tolist()

def detrend_flux(time: list, flux: list, window_hours: float = 12.0) -> list:
    """
    Remove long-term stellar variability using a sliding median filter.
    Window size in hours — 12h removes variability longer than half a day
    while preserving transit signals (typically 1-6 hours).
    
    This is the most important tuning parameter in the pipeline.
    Too short: removes real transits. Too long: misses slow variability.
    """
    time_arr = np.array(time)
    flux_arr = np.array(flux)
    
    # Convert window from hours to number of cadences
    # TESS 2-min cadence: 30 points per hour
    cadence_minutes = np.median(np.diff(time_arr)) * 24 * 60
    window_points = int(window_hours * 60 / cadence_minutes)
    if window_points % 2 == 0:
        window_points += 1   # must be odd for medfilt
    
    trend = medfilt(flux_arr, kernel_size=window_points)
    detrended = flux_arr / trend
    
    return detrended.tolist()

def remove_outliers(flux: list, sigma: float = 7.0) -> tuple[list, list]:
    """
    Remove extreme outliers (cosmic rays, detector artifacts).
    Uses 7-sigma to avoid clipping real astrophysical events like deep
    transits or flares. Only the most extreme instrumental artifacts
    (cosmic ray hits, readout glitches) get removed.
    """
    arr = np.array(flux)
    median = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - median))
    std_estimate = mad * 1.4826
    
    mask = np.abs(arr - median) < sigma * std_estimate
    cleaned = arr.copy()
    cleaned[~mask] = np.nan
    
    return cleaned.tolist(), mask.tolist()
```

### 4.7 Autoencoder (pipeline/autoencoder.py)

```python
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

WINDOW_SIZE = 128      # data points per window (~4.3 hours at 2-min cadence)
STRIDE = 32            # overlap between windows
LATENT_DIM = 16        # bottleneck size — lower = more compression = more sensitive

class LightCurveAutoencoder(nn.Module):
    """
    1D convolutional autoencoder.
    Trained on "normal" light curves to learn what boring looks like.
    Reconstruction error on new data = anomaly score.
    
    Architecture rationale:
    - Conv layers capture local temporal patterns (transit shapes, flare profiles)
    - Bottleneck forces compression — weird patterns can't be reconstructed well
    - Symmetric decoder mirrors encoder
    """
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),                           # 128 → 64
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),                           # 64 → 32
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 32, LATENT_DIM)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 64 * 32),
            nn.Unflatten(1, (64, 32)),
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),               # 32 → 64
            nn.ConvTranspose1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),               # 64 → 128
            nn.ConvTranspose1d(16, 1, kernel_size=7, padding=3),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def score_light_curve(flux: list, weights_path: str) -> dict:
    """
    Slides a window across the detrended light curve and computes
    reconstruction error at each position. High error = anomalous.
    
    Returns:
        score_per_point: list of anomaly scores aligned to input flux length
        global_score: single 0-1 score for the whole observation
    """
    model = LightCurveAutoencoder()
    
    if Path(weights_path).exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        # No trained weights yet — use untrained model as placeholder
        # Real scores require training first (see Section 4.8)
        pass
    
    model.eval()
    
    arr = np.array(flux, dtype=np.float32)
    
    # Fill NaNs with median for inference
    nan_mask = np.isnan(arr)
    arr[nan_mask] = np.nanmedian(arr)
    
    # Normalize window by window
    scores = np.zeros(len(arr))
    counts = np.zeros(len(arr))
    
    with torch.no_grad():
        for start in range(0, len(arr) - WINDOW_SIZE, STRIDE):
            window = arr[start:start + WINDOW_SIZE]
            
            # Normalize this window
            w_mean = window.mean()
            w_std = window.std() + 1e-8
            window_norm = (window - w_mean) / w_std
            
            x = torch.tensor(window_norm).unsqueeze(0).unsqueeze(0)  # [1, 1, 128]
            reconstruction = model(x).squeeze().numpy()
            
            # MSE reconstruction error
            error = ((window_norm - reconstruction) ** 2).mean()
            
            scores[start:start + WINDOW_SIZE] += error
            counts[start:start + WINDOW_SIZE] += 1
    
    counts = np.maximum(counts, 1)
    scores = scores / counts
    
    # Normalize scores against training distribution
    # Load training stats saved during training (see Section 4.8)
    stats_path = Path(weights_path).with_suffix(".stats.npz")
    if stats_path.exists():
        stats = np.load(stats_path)
        train_mean = float(stats["mean_error"])
        train_std = float(stats["std_error"])
        # Z-score relative to training, then sigmoid to 0-1
        z_scores = (scores - train_mean) / (train_std + 1e-8)
        scores = 1.0 / (1.0 + np.exp(-z_scores))  # sigmoid
    else:
        # Fallback: percentile-based normalization (less reliable)
        if scores.max() > 0:
            scores = scores / np.percentile(scores[scores > 0], 99)
            scores = np.clip(scores, 0, 1)
    
    return {
        "score_per_point": scores.tolist(),
        "global_score": float(np.percentile(scores, 99))
    }
```

### 4.8 Training the Autoencoder

The model needs to be trained before it produces meaningful scores. Training data = "normal" light curves from TESS. The more boring stars you train on, the better it learns what normal looks like.

```python
# backend/pipeline/train.py
# Run this once before using the app:
# python -m pipeline.train

import lightkurve as lk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from autoencoder import LightCurveAutoencoder, WINDOW_SIZE
from clean import detrend_flux, normalize_flux, remove_outliers
import numpy as np

TRAINING_TARGETS = [
    # A mix of quiet, well-studied stars with no known anomalies
    # These teach the model what "normal" looks like
    "TIC 261136679",    # quiet M dwarf
    "TIC 38846515",     # solar-type
    "TIC 144700903",    # quiet K star
    "TIC 207468071",
    "TIC 149603524",
    # Add more — the more the better. 50+ is ideal.
    # Avoid known variable stars, binary systems, or stars with planets
]

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

def prepare_windows(flux_list: list) -> np.ndarray:
    """Slice a light curve into overlapping windows for training."""
    arr = np.array(flux_list, dtype=np.float32)
    arr = arr[~np.isnan(arr)]
    windows = []
    for i in range(0, len(arr) - WINDOW_SIZE, 32):
        w = arr[i:i + WINDOW_SIZE]
        w = (w - w.mean()) / (w.std() + 1e-8)
        windows.append(w)
    return np.array(windows)

def train():
    all_windows = []
    
    print(f"Fetching {len(TRAINING_TARGETS)} training targets from MAST...")
    for tic in TRAINING_TARGETS:
        try:
            results = lk.search_lightcurve(tic, mission="TESS", author="SPOC", exptime=120)
            if len(results) == 0:
                continue
            lc = results[0].download()
            lc = lc.remove_nans()
            flux = normalize_flux(lc.flux.value.tolist())
            flux, _ = remove_outliers(flux)
            detrended = detrend_flux(lc.time.value.tolist(), flux)
            windows = prepare_windows(detrended)
            all_windows.append(windows)
            print(f"  {tic}: {len(windows)} windows")
        except Exception as e:
            print(f"  {tic}: skipped ({e})")
    
    all_windows = np.concatenate(all_windows, axis=0)
    print(f"\nTotal training windows: {len(all_windows)}")
    
    X = torch.tensor(all_windows).unsqueeze(1)   # [N, 1, 128]
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LightCurveAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print("\nTraining autoencoder...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg = total_loss / len(loader)
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={avg:.6f}")
    
    # Compute training error distribution for score normalization at inference
    model.eval()
    all_errors = []
    with torch.no_grad():
        for (batch,) in loader:
            output = model(batch)
            errors = ((batch - output) ** 2).mean(dim=(1, 2))
            all_errors.extend(errors.numpy().tolist())
    all_errors = np.array(all_errors)

    import os
    os.makedirs("models/weights", exist_ok=True)
    torch.save(model.state_dict(), "models/weights/autoencoder_v1.pt")

    # Save training error statistics alongside weights
    np.savez(
        "models/weights/autoencoder_v1.stats.npz",
        mean_error=all_errors.mean(),
        std_error=all_errors.std(),
        p99_error=np.percentile(all_errors, 99),
    )
    print(f"\nTraining error stats: mean={all_errors.mean():.6f}, std={all_errors.std():.6f}")
    print("Model saved to models/weights/autoencoder_v1.pt")
    print("Stats saved to models/weights/autoencoder_v1.stats.npz")

if __name__ == "__main__":
    train()
```

Training takes ~10-30 minutes on a CPU depending on how many targets you use. Run it once, weights are saved, app uses them from then on.

### 4.9 Periodogram (pipeline/periodogram.py)

```python
import numpy as np
from astropy.timeseries import BoxLeastSquares
from astropy import units as u

def run_bls(time: list, flux: list) -> dict:
    """
    Box Least Squares period search.
    Looks for periodic box-shaped dips consistent with planetary transits.
    
    Returns the best-fit period, transit duration, depth,
    and the full power spectrum for visualization.
    """
    time_arr = np.array(time)
    flux_arr = np.array(flux)
    
    # Remove NaNs
    mask = ~np.isnan(flux_arr)
    time_arr = time_arr[mask]
    flux_arr = flux_arr[mask]
    
    # Period grid: search 0.2 days to half the total baseline
    # 0.2d minimum catches ultra-short-period planets (e.g. Kepler-78b at 0.35d)
    baseline_days = time_arr[-1] - time_arr[0]
    min_period = 0.2
    max_period = baseline_days / 2.0
    
    periods = np.exp(np.linspace(
        np.log(min_period),
        np.log(max_period),
        5000
    ))
    
    bls = BoxLeastSquares(time_arr * u.day, flux_arr)
    periodogram = bls.power(
        periods * u.day,
        duration=[0.05, 0.1, 0.2] * u.day    # search transit durations 1.2h, 2.4h, 4.8h
    )
    
    best_idx = np.argmax(periodogram.power)
    best_period = float(periodogram.period[best_idx].value)
    
    stats = bls.compute_stats(
        periodogram.period[best_idx],
        periodogram.duration[best_idx],
        periodogram.transit_time[best_idx]
    )
    
    return {
        "best_period_days": best_period,
        "best_power": float(periodogram.power[best_idx]),
        "transit_duration_hours": float(periodogram.duration[best_idx].to(u.hour).value),
        "depth_ppm": float(stats["depth"][0] * 1e6),
        "periods": periodogram.period.value.tolist(),
        "powers": periodogram.power.tolist()
    }

def phase_fold(time: list, flux: list, period: float, t0: float) -> dict:
    """
    Fold the light curve on the best period.
    Returns phase (-0.5 to 0.5) and flux arrays for the phase plot.
    """
    time_arr = np.array(time)
    flux_arr = np.array(flux)
    
    phase = ((time_arr - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    
    sort_idx = np.argsort(phase)
    
    return {
        "phase": phase[sort_idx].tolist(),
        "flux": flux_arr[sort_idx].tolist()
    }
```

### 4.10 Wavelet Decomposition (pipeline/wavelet.py)

The continuous wavelet transform (CWT) produces a 2D map of the light curve: time on the x-axis, period (timescale) on the y-axis, and power as the color/intensity at each cell. Unlike the BLS periodogram — which collapses all of time into a single power spectrum — the CWT shows you *when* a given periodicity is active. This serves two purposes in the pipeline:

1. **Automatic TESS systematic rejection.** TESS has well-documented instrumental artifacts at known periods: the spacecraft orbital period (~13.7 days), momentum dumps (~3.125 days), and scattered light systematics (~1 day). Any flagged event whose anomaly score peak coincides with strong wavelet power at these periods gets automatically downgraded from UNKNOWN to SYSTEMATIC.

2. **Persistent vs. transient anomaly discrimination.** A real astrophysical event (transit, exocomet) shows up as a brief localized blob in wavelet space. A recurring instrumental artifact shows up as a horizontal stripe spanning the full time baseline. These look completely different and the distinction is unambiguous.

```python
# backend/pipeline/wavelet.py

import numpy as np
import pywt

# Known TESS instrumental periods in days.
# Any signal peaking at these periods is almost certainly an artifact.
TESS_SYSTEMATIC_PERIODS = {
    "orbital":        13.7,    # spacecraft orbital period
    "momentum_dump":  3.125,   # reaction wheel momentum dump cadence
    "scattered_light": 1.0,    # once-per-day scattered light from Earth/Moon
    "half_orbital":   6.85,    # harmonic of orbital period
}

# Tolerance in days for matching a detected peak to a known systematic
SYSTEMATIC_TOLERANCE = 0.3

def run_wavelet(time: list, flux: list) -> dict:
    """
    Compute the continuous wavelet transform of the detrended light curve.

    Uses the Morlet wavelet (complex sinusoid modulated by a Gaussian),
    which gives good time-frequency resolution for oscillatory signals.

    Returns a dict ready to serialize to JSON and store in the DB:
      - time: subsampled time axis (every Nth point for manageable JSON size)
      - periods: period axis in days (log-spaced, 0.1 to 30 days)
      - power: 2D list [n_periods][n_time] of normalized wavelet power
      - tess_systematic_periods: list of period values where TESS artifacts dominate
      - dominant_periods: top 3 periods by integrated power (for quick inspection)
    """
    time_arr = np.array(time)
    flux_arr = np.array(flux)

    # Fill NaNs
    nan_mask = np.isnan(flux_arr)
    flux_arr[nan_mask] = np.nanmedian(flux_arr)

    # Median cadence in days
    dt = float(np.median(np.diff(time_arr)))

    # Period axis: log-spaced from 0.1 to 30 days
    # Convert to CWT scales: scale = period / (dt * central_frequency)
    # For Morlet wavelet, central_frequency ≈ 1.0
    n_periods = 64
    periods = np.logspace(np.log10(0.1), np.log10(30.0), n_periods)
    scales = periods / dt

    # Run CWT using the Morlet wavelet
    # pywt.cwt returns (coefficients [n_scales x n_time], frequencies)
    coefficients, _ = pywt.cwt(flux_arr, scales, "morl", sampling_period=dt)

    # Power = squared magnitude of complex coefficients
    power = np.abs(coefficients) ** 2

    # Normalize each period row by its median so all periods are comparable
    # Without this, long periods always dominate just due to scale
    for i in range(n_periods):
        row_median = np.median(power[i])
        if row_median > 0:
            power[i] = power[i] / row_median

    # Subsample time axis for JSON — keep every 4th point (still ~thousands of points)
    subsample = 4
    time_sub = time_arr[::subsample].tolist()
    power_sub = power[:, ::subsample].tolist()

    # Find periods where TESS systematics dominate
    # A systematic dominates if it has the highest time-integrated power
    # within its tolerance window
    integrated_power = power.mean(axis=1)  # [n_periods]
    systematic_hits = []
    for name, sys_period in TESS_SYSTEMATIC_PERIODS.items():
        # Find closest period in our axis
        idx = np.argmin(np.abs(periods - sys_period))
        window = np.abs(periods - sys_period) < SYSTEMATIC_TOLERANCE
        if window.any():
            local_power = integrated_power[window].max()
            global_max = integrated_power.max()
            if local_power > 0.6 * global_max:
                systematic_hits.append({
                    "name": name,
                    "period_days": round(sys_period, 3),
                    "relative_power": round(float(local_power / global_max), 3)
                })

    # Top 3 dominant periods by integrated power (excluding systematics)
    sys_period_values = [s["period_days"] for s in systematic_hits]
    dominant_idxs = np.argsort(integrated_power)[::-1]
    dominant_periods = []
    for idx in dominant_idxs:
        p = float(periods[idx])
        # Skip if this period is within tolerance of a known systematic
        is_systematic = any(abs(p - sp) < SYSTEMATIC_TOLERANCE for sp in sys_period_values)
        if not is_systematic:
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

def event_matches_systematic(event_time_center: float, wavelet_result: dict,
                              time: list, tolerance_days: float = 0.5) -> str | None:
    """
    Check whether a flagged event's time center falls within a high-power
    wavelet region at a known TESS systematic period.

    Returns the systematic name if matched (e.g. "orbital"), None if clean.

    How it works:
      - For each known systematic period, check the wavelet power at the
        event's time position and that period
      - If power > 2× the median power at that period, the event is flagged
        as potentially systematic
    """
    if not wavelet_result or not wavelet_result.get("tess_systematic_periods"):
        return None

    time_arr = np.array(wavelet_result["time"])
    periods_arr = np.array(wavelet_result["periods"])
    power_arr = np.array(wavelet_result["power"])   # [n_periods, n_time]

    # Find nearest time index to event center
    time_idx = int(np.argmin(np.abs(time_arr - event_time_center)))

    for systematic in wavelet_result["tess_systematic_periods"]:
        sys_period = systematic["period_days"]
        period_idx = int(np.argmin(np.abs(periods_arr - sys_period)))

        local_power = float(power_arr[period_idx][time_idx])
        median_power = float(np.median(power_arr[period_idx]))

        if local_power > 2.0 * median_power:
            return systematic["name"]

    return None
```

### 4.11 Centroid Shift Analysis (pipeline/centroid.py)

TESS stores not just the summed flux of your target but the individual pixel values of a small postage stamp around it — the Target Pixel File (TPF). Typically 5×5 to 11×11 pixels at ~21 arcseconds per pixel.

During a genuine transit, the dimming comes from the target star itself so the photometric centroid (the flux-weighted center of brightness) stays fixed. If the dip is caused by a background eclipsing binary blended into the aperture, the centroid shifts toward the contaminant during the dip — because the flux from the contaminant is temporarily reduced, pulling the center of brightness away from it.

This is the single most important false-positive test in the exoplanet vetting workflow. TFOP requires it before they'll observe a candidate.

```python
# backend/pipeline/centroid.py

import numpy as np
import lightkurve as lk

TESS_PLATE_SCALE_ARCSEC = 21.0    # arcseconds per TESS pixel

def compute_centroid(tic_id: str, sector: str = "all") -> dict:
    """
    Download the Target Pixel File for the target and compute the
    flux-weighted centroid position at each timestamp.

    Returns centroid column and row positions over time, plus a
    summary of whether any significant shift was detected.

    A shift > 1 pixel (~21 arcsec) during a dip is a strong indicator
    of a background contaminating source.
    """
    search_sector = None if sector == "all" else int(sector)

    tpf_results = lk.search_targetpixelfile(
        f"TIC {tic_id}",
        mission="TESS",
        sector=search_sector,
        author="SPOC"
    )

    if len(tpf_results) == 0:
        return {"available": False}

    # Download all available sector TPFs and stitch centroids
    # Using all sectors catches contamination that may only appear at certain roll angles
    tpfs = tpf_results.download_all(cache=True)
    tpf = tpfs[0] if len(tpfs) == 1 else tpfs[0]

    # Compute flux-weighted centroid at each cadence
    # lightkurve provides this directly
    centroid_col, centroid_row = tpf.estimate_centroids(aperture_mask="pipeline")

    time = tpf.time.value.tolist()
    col = centroid_col.value.tolist()
    row = centroid_row.value.tolist()

    # Remove NaNs for statistics
    col_arr = np.array(col)
    row_arr = np.array(row)
    valid = ~(np.isnan(col_arr) | np.isnan(row_arr))
    col_clean = col_arr[valid]
    row_clean = row_arr[valid]

    if len(col_clean) == 0:
        return {"available": False}

    # Baseline centroid (median position across full observation)
    col_baseline = float(np.median(col_clean))
    row_baseline = float(np.median(row_clean))

    # Centroid displacement from baseline at each point (in pixels)
    col_disp = col_arr - col_baseline
    row_disp = row_arr - row_baseline
    displacement_pixels = np.sqrt(col_disp**2 + row_disp**2)
    displacement_arcsec = displacement_pixels * TESS_PLATE_SCALE_ARCSEC

    max_shift_pixels = float(np.nanmax(displacement_pixels))
    max_shift_arcsec = float(np.nanmax(displacement_arcsec))
    rms_shift_arcsec = float(np.nanstd(displacement_arcsec))

    # Flag as contaminated if peak displacement > 1 pixel during any point
    # 1 pixel = 21 arcsec = significant for a point source target
    shift_flagged = max_shift_pixels > 1.0

    return {
        "available": True,
        "time": time,
        "col": col_arr.tolist(),
        "row": row_arr.tolist(),
        "col_baseline": col_baseline,
        "row_baseline": row_baseline,
        "displacement_arcsec": displacement_arcsec.tolist(),
        "max_shift_arcsec": round(max_shift_arcsec, 2),
        "rms_shift_arcsec": round(rms_shift_arcsec, 2),
        "shift_flagged": shift_flagged,
    }

def centroid_shift_during_event(event_time_center: float, event_duration_hours: float,
                                 centroid_result: dict) -> float:
    """
    Given a flagged event, measure the maximum centroid displacement
    specifically during the event window.

    Returns displacement in arcseconds. Values > 10 arcsec strongly
    suggest background contamination. Values < 3 arcsec are consistent
    with an on-target signal.
    """
    if not centroid_result.get("available"):
        return -1.0    # -1 signals "centroid data unavailable"

    time_arr = np.array(centroid_result["time"])
    disp_arr = np.array(centroid_result["displacement_arcsec"])

    half_dur_days = (event_duration_hours / 2.0) / 24.0
    in_event = np.abs(time_arr - event_time_center) < half_dur_days

    if not in_event.any():
        return 0.0

    return round(float(np.nanmax(disp_arr[in_event])), 2)
```

### 4.12 Updated Classifier (pipeline/classifier.py)

The classifier now receives wavelet and centroid results and applies them as pre-classification filters. An event that matches a TESS systematic period or shows centroid shift is downgraded *before* the heuristic rules run — this prevents garbage from ever reaching the UNKNOWN bucket.

```python
# backend/pipeline/classifier.py  (updated)

import numpy as np
from dataclasses import dataclass
from enum import Enum
from .wavelet import event_matches_systematic
from .centroid import centroid_shift_during_event

class EventType(str, Enum):
    TRANSIT          = "transit"
    ASYMMETRIC       = "asymmetric"
    DEPTH_ANOMALY    = "depth_anomaly"
    NON_PERIODIC     = "non_periodic"
    EXOCOMET         = "exocomet"
    STELLAR_FLARE    = "stellar_flare"
    STELLAR_SPOT     = "stellar_spot"
    SYSTEMATIC       = "systematic"       # NEW — wavelet-identified TESS artifact
    CONTAMINATION    = "contamination"    # NEW — centroid shift indicates background source
    UNKNOWN          = "unknown"

# Centroid shift thresholds in arcseconds
CENTROID_CONTAMINATION_THRESHOLD = 10.0   # > 10 arcsec = almost certainly background binary
CENTROID_CAUTION_THRESHOLD       = 3.0    # 3-10 arcsec = flagged but not conclusive

def find_dip_events(time: list, flux: list, scores: list,
                    wavelet_result: dict, centroid_result: dict,
                    threshold: float = 0.5) -> list[dict]:
    """
    Find contiguous high-score regions, extract event parameters,
    apply wavelet systematic filter and centroid contamination filter,
    then classify the remainder.
    """
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

            if len(region_time) < 3:
                i = j + 1
                continue

            time_center = float(region_time[np.argmin(region_flux)])
            duration_hours = float((region_time[-1] - region_time[0]) * 24)
            depth_ppm = float((1.0 - np.nanmin(region_flux)) * 1e6)
            event_score = float(region_score.max())

            # --- Filter 1: Wavelet systematic check ---
            systematic_name = event_matches_systematic(time_center, wavelet_result, time)
            if systematic_name:
                events.append({
                    "time_center": time_center,
                    "duration_hours": round(duration_hours, 2),
                    "depth_ppm": round(depth_ppm, 1),
                    "anomaly_score": round(event_score, 3),
                    "event_type": EventType.SYSTEMATIC.value,
                    "confidence": 0.85,
                    "notes": f"Wavelet power peaks at TESS {systematic_name} period "
                             f"({systematic_name} = known instrumental artifact). "
                             f"Not astrophysical.",
                    "centroid_shift_arcsec": -1.0,
                    "systematic_match": systematic_name,
                })
                i = j + 1
                continue

            # --- Filter 2: Centroid shift check ---
            shift_arcsec = centroid_shift_during_event(time_center, duration_hours, centroid_result)
            if shift_arcsec > CENTROID_CONTAMINATION_THRESHOLD:
                events.append({
                    "time_center": time_center,
                    "duration_hours": round(duration_hours, 2),
                    "depth_ppm": round(depth_ppm, 1),
                    "anomaly_score": round(event_score, 3),
                    "event_type": EventType.CONTAMINATION.value,
                    "confidence": 0.82,
                    "notes": f"Centroid shifts {shift_arcsec:.1f} arcsec during event — "
                             f"signal originates from a background source, not the target star. "
                             f"Likely background eclipsing binary.",
                    "centroid_shift_arcsec": shift_arcsec,
                    "systematic_match": None,
                })
                i = j + 1
                continue

            # --- Standard heuristic classification ---
            event_type, confidence, notes = classify_event(region_time, region_flux, region_score)

            # Append centroid caution note if shift is borderline
            if CENTROID_CAUTION_THRESHOLD < shift_arcsec <= CENTROID_CONTAMINATION_THRESHOLD:
                notes += (f" Centroid shift {shift_arcsec:.1f} arcsec during event — "
                          f"borderline, recommend follow-up centroid analysis.")
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

    return sorted(events, key=lambda e: e["anomaly_score"], reverse=True)


def classify_event(time, flux, scores) -> tuple[EventType, float, str]:
    """
    Heuristic shape-based classifier. Only runs after wavelet and centroid
    filters have been passed. See original classifier.py for full logic.
    """
    flux_arr = np.array(flux)
    time_arr = np.array(time)

    mid = len(flux_arr) // 2
    ingress = flux_arr[:mid]
    egress = flux_arr[mid:]

    ingress_slope = np.polyfit(range(len(ingress)), ingress, 1)[0] if len(ingress) > 2 else 0
    egress_slope  = np.polyfit(range(len(egress)),  egress,  1)[0] if len(egress)  > 2 else 0

    asymmetry_ratio = abs(ingress_slope) / (abs(egress_slope) + 1e-10)
    depth_ppm       = (1.0 - flux_arr.min()) * 1e6
    duration_hours  = (time_arr[-1] - time_arr[0]) * 24

    if flux_arr.max() > 1.005 and ingress_slope < 0:
        return (EventType.STELLAR_FLARE, 0.85,
                f"Flux increase {(flux_arr.max()-1)*1e6:.0f} ppm — magnetic reconnection flare.")

    if asymmetry_ratio > 3.0 and duration_hours < 8:
        return (EventType.EXOCOMET, 0.72,
                f"Asymmetry {asymmetry_ratio:.1f}× — sharp ingress, extended egress tail. Consistent with sungrazing comet.")

    if 1.8 < asymmetry_ratio <= 3.0:
        return (EventType.ASYMMETRIC, 0.68,
                f"Ingress/egress asymmetry {asymmetry_ratio:.1f}× — possible ringed body, oblate planet, or grazing geometry.")

    if depth_ppm > 10000:
        return (EventType.DEPTH_ANOMALY, 0.79,
                f"Depth {depth_ppm:.0f} ppm ({depth_ppm/10000:.1f}%) — exceeds planetary threshold. Possible dust cloud or unresolved binary.")

    if depth_ppm < 5000 and asymmetry_ratio < 1.5:
        return (EventType.TRANSIT, 0.81,
                f"Symmetric dip {depth_ppm:.0f} ppm over {duration_hours:.1f}h — consistent with planetary transit.")

    if duration_hours > 24:
        return (EventType.STELLAR_SPOT, 0.55,
                f"Gradual {duration_hours:.0f}h dimming — consistent with large starspot rotation.")

    # Non-periodic single event — no BLS match and no classifier match
    if duration_hours < 24 and depth_ppm > 200:
        return (EventType.NON_PERIODIC, 0.50,
                f"Single non-periodic event — depth {depth_ppm:.0f} ppm, duration {duration_hours:.1f}h. "
                f"No BLS period match. Possible one-time occultation or transient event.")

    return (EventType.UNKNOWN, 0.40,
            f"No classifier match after systematic and centroid filters — "
            f"depth {depth_ppm:.0f} ppm, duration {duration_hours:.1f}h, asymmetry {asymmetry_ratio:.2f}×. "
            f"Requires spectroscopic follow-up.")
```

### 4.13 API Routes (routers/analyze.py)

```python
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel
from db.database import get_db
from db import models
from pipeline import fetch, clean, autoencoder, periodogram, wavelet, centroid, classifier, technosignature
import os

router = APIRouter(prefix="/api")

class AnalyzeRequest(BaseModel):
    identifier: str      # TIC ID or common name
    sector: str = "all"

class AnalyzeResponse(BaseModel):
    analysis_id: int
    status: str

@router.post("/analyze", response_model=AnalyzeResponse)
async def start_analysis(req: AnalyzeRequest, background_tasks: BackgroundTasks, db=Depends(get_db)):
    """
    Kick off an analysis pipeline run in the background.
    Returns immediately with an analysis ID.
    Frontend polls /api/analysis/{id} for status and results.
    """
    # Create placeholder record
    analysis = models.Analysis(status=models.AnalysisStatus.pending)
    db.add(analysis)
    db.commit()
    
    background_tasks.add_task(run_pipeline, analysis.id, req.identifier, req.sector)
    
    return {"analysis_id": analysis.id, "status": "pending"}

@router.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: int, db=Depends(get_db)):
    analysis = db.query(models.Analysis).filter_by(id=analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis

async def run_pipeline(analysis_id: int, identifier: str, sector: str):
    """Full pipeline — runs in background thread."""
    db = next(get_db())
    analysis = db.query(models.Analysis).filter_by(id=analysis_id).first()
    
    try:
        analysis.status = models.AnalysisStatus.running
        db.commit()
        
        # 1. Resolve target + fetch light curve
        target_info = fetch.resolve_target(identifier)
        raw_data = fetch.fetch_light_curve(target_info["tic_id"], sector)
        
        # 2. Save or retrieve target record
        target = db.query(models.Target).filter_by(tic_id=target_info["tic_id"]).first()
        if not target:
            target = models.Target(**target_info)
            db.add(target)
            db.commit()
        analysis.target_id = target.id
        
        # 3. Clean + detrend
        flux_norm = clean.normalize_flux(raw_data["flux"])
        flux_clean, _ = clean.remove_outliers(flux_norm)
        flux_detrended = clean.detrend_flux(raw_data["time"], flux_clean)
        
        # 4. Autoencoder scoring
        weights = os.getenv("MODEL_WEIGHTS_PATH")
        score_result = autoencoder.score_light_curve(flux_detrended, weights)
        
        # 5. BLS periodogram
        period_result = periodogram.run_bls(raw_data["time"], flux_detrended)

        # 6. Wavelet transform — identifies TESS systematics before classification
        wavelet_result = wavelet.run_wavelet(raw_data["time"], flux_detrended)

        # 7. Centroid shift analysis — identifies background contamination before classification
        centroid_result = centroid.compute_centroid(target_info["tic_id"], sector)
        
        # 8. Classify events — wavelet and centroid filters applied inside
        events = classifier.find_dip_events(
            raw_data["time"],
            flux_detrended,
            score_result["score_per_point"],
            wavelet_result=wavelet_result,
            centroid_result=centroid_result,
        )

        # 9. Technosignature analysis on UNKNOWN events
        unknown_events = [e for e in events if e["event_type"] == "unknown"]
        techno_result = technosignature.analyze(
            tic_id=target_info["tic_id"],
            time=raw_data["time"],
            flux=flux_detrended,
            unknown_events=unknown_events,
            period_result=period_result,
        )
        
        # 10. Persist results
        analysis.raw_flux       = {"time": raw_data["time"], "flux": flux_norm}
        analysis.detrended_flux = {"time": raw_data["time"], "flux": flux_detrended}
        analysis.score_timeline = {"time": raw_data["time"], "score": score_result["score_per_point"]}
        analysis.periodogram    = {"period": period_result["periods"], "power": period_result["powers"]}
        analysis.wavelet        = wavelet_result
        analysis.centroid       = centroid_result
        analysis.technosignature = techno_result
        analysis.anomaly_score  = score_result["global_score"]
        analysis.known_period   = period_result["best_period_days"]
        analysis.flag_count     = len([e for e in events if e["event_type"] not in ("systematic", "contamination")])
        
        for ev in events:
            db.add(models.FlaggedEvent(analysis_id=analysis.id, **ev))
        
        analysis.status = models.AnalysisStatus.complete
        db.commit()
        
    except Exception as e:
        analysis.status = models.AnalysisStatus.failed
        db.commit()
        raise e
```

### 4.14 Running the Backend

```powershell
cd backend
.\venv\Scripts\activate

# Initialize database
alembic init db/migrations
alembic revision --autogenerate -m "initial"
alembic upgrade head

# Train the model (first time only, takes 10-30 mins)
python -m pipeline.train

# Start the API server
uvicorn main:app --reload --port 8000
```

API will be live at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

---

## 5. Frontend — Next.js 14

### 5.1 Setup

```powershell
cd frontend
npx create-next-app@14 . --typescript --tailwind --app --no-src-dir
npm install recharts @tanstack/react-query axios
```

### 5.2 TypeScript Types (lib/types.ts)

```typescript
export type AnalysisStatus = "pending" | "running" | "complete" | "failed"

export type EventType =
  | "transit"
  | "asymmetric"
  | "depth_anomaly"
  | "non_periodic"
  | "exocomet"
  | "stellar_flare"
  | "stellar_spot"
  | "systematic"
  | "contamination"
  | "unknown"

export interface Target {
  id: number
  tic_id: string
  common_name: string
  ra: number
  dec: number
  magnitude: number
  stellar_type: string
}

export interface WaveletSystematic {
  name: string
  period_days: number
  relative_power: number
}

export interface WaveletResult {
  time: number[]
  periods: number[]
  power: number[][]    // [n_periods][n_time]
  tess_systematic_periods: WaveletSystematic[]
  dominant_periods: number[]
}

export interface CentroidResult {
  available: boolean
  time: number[]
  col: number[]
  row: number[]
  col_baseline: number
  row_baseline: number
  displacement_arcsec: number[]
  max_shift_arcsec: number
  rms_shift_arcsec: number
  shift_flagged: boolean
}

export interface FlaggedEvent {
  id: number
  event_type: EventType
  time_center: number
  duration_hours: number
  depth_ppm: number
  anomaly_score: number
  confidence: number
  notes: string
  centroid_shift_arcsec: number   // -1 = data unavailable
  systematic_match: string | null
}

export interface Analysis {
  id: number
  target: Target
  sector: string
  status: AnalysisStatus
  anomaly_score: number
  known_period: number
  flag_count: number
  raw_flux: { time: number[]; flux: number[] }
  detrended_flux: { time: number[]; flux: number[] }
  score_timeline: { time: number[]; score: number[] }
  periodogram: { period: number[]; power: number[] }
  wavelet: WaveletResult
  centroid: CentroidResult
  events: FlaggedEvent[]
}

export const EVENT_LABELS: Record<EventType, string> = {
  transit:       "Transit",
  asymmetric:    "Asymmetric",
  depth_anomaly: "Depth anomaly",
  non_periodic:  "Non-periodic",
  exocomet:      "Exocomet",
  stellar_flare: "Stellar flare",
  stellar_spot:  "Stellar spot",
  systematic:    "Systematic",     // TESS instrumental artifact
  contamination: "Contamination",  // background eclipsing binary
  unknown:       "Unknown",
}

export const EVENT_COLORS: Record<EventType, string> = {
  transit:       "#185FA5",
  asymmetric:    "#A32D2D",
  depth_anomaly: "#A32D2D",
  non_periodic:  "#854F0B",
  exocomet:      "#3B6D11",
  stellar_flare: "#854F0B",
  stellar_spot:  "#5F5E5A",
  systematic:    "#888780",    // gray — dismissed
  contamination: "#888780",    // gray — dismissed
  unknown:       "#A32D2D",
}
```

### 5.3 API Client (lib/api.ts)

```typescript
import axios from "axios"
import type { Analysis } from "./types"

const client = axios.create({ baseURL: "http://localhost:8000/api" })

export async function startAnalysis(identifier: string, sector = "all") {
  const { data } = await client.post("/analyze", { identifier, sector })
  return data as { analysis_id: number; status: string }
}

export async function getAnalysis(id: number): Promise<Analysis> {
  const { data } = await client.get(`/analysis/${id}`)
  return data
}

export async function pollAnalysis(
  id: number,
  onUpdate: (a: Analysis) => void,
  intervalMs = 2000
): Promise<Analysis> {
  return new Promise((resolve, reject) => {
    const timer = setInterval(async () => {
      try {
        const analysis = await getAnalysis(id)
        onUpdate(analysis)
        if (analysis.status === "complete") {
          clearInterval(timer)
          resolve(analysis)
        }
        if (analysis.status === "failed") {
          clearInterval(timer)
          reject(new Error("Analysis failed"))
        }
      } catch (e) {
        clearInterval(timer)
        reject(e)
      }
    }, intervalMs)
  })
}
```

### 5.4 Key Component — LightCurveChart

```typescript
// components/LightCurveChart.tsx
"use client"
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceArea, ResponsiveContainer } from "recharts"
import type { FlaggedEvent } from "@/lib/types"

interface Props {
  time: number[]
  flux: number[]
  events?: FlaggedEvent[]
  height?: number
}

export function LightCurveChart({ time, flux, events = [], height = 200 }: Props) {
  // Downsample for performance if > 50k points
  const MAX_POINTS = 5000
  const step = Math.max(1, Math.floor(time.length / MAX_POINTS))
  
  const data = time
    .filter((_, i) => i % step === 0)
    .map((t, i) => ({ t: +t.toFixed(2), f: +flux[i * step]?.toFixed(5) }))
    .filter(d => d.f !== undefined && !isNaN(d.f))

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
        <XAxis
          dataKey="t"
          tick={{ fontSize: 10, fontFamily: "monospace" }}
          tickLine={false}
          label={{ value: "BTJD", position: "insideBottomRight", fontSize: 10 }}
        />
        <YAxis
          domain={["auto", "auto"]}
          tick={{ fontSize: 10, fontFamily: "monospace" }}
          tickLine={false}
          tickFormatter={v => v.toFixed(3)}
          width={50}
        />
        <Tooltip
          contentStyle={{ fontSize: 11, fontFamily: "monospace" }}
          formatter={(v: number) => [v.toFixed(5), "flux"]}
          labelFormatter={(t: number) => `t = ${t} BTJD`}
        />
        {events.map(ev => (
          <ReferenceArea
            key={ev.id}
            x1={+(ev.time_center - ev.duration_hours / 48).toFixed(2)}
            x2={+(ev.time_center + ev.duration_hours / 48).toFixed(2)}
            fill={ev.event_type === "unknown" ? "rgba(163,45,45,0.15)" : "rgba(24,95,165,0.1)"}
            stroke={ev.event_type === "unknown" ? "#A32D2D" : "#185FA5"}
            strokeWidth={1}
          />
        ))}
        <Line
          type="monotone"
          dataKey="f"
          stroke="#185FA5"
          strokeWidth={1}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
```

### 5.5 WaveletChart Component (components/WaveletChart.tsx)

The wavelet heatmap renders as a 2D canvas image — period on the y-axis (log scale, low periods at top), time on the x-axis, and wavelet power as color intensity. Horizontal bands at known TESS systematic periods are overlaid as dashed lines so you can immediately see whether any anomaly lines up with an artifact.

```typescript
// components/WaveletChart.tsx
"use client"
import { useEffect, useRef } from "react"
import type { WaveletResult } from "@/lib/types"

interface Props {
  wavelet: WaveletResult
  height?: number
}

// TESS systematic period labels overlaid on the heatmap
const SYSTEMATIC_LABELS: Record<string, string> = {
  orbital:        "13.7d orbital",
  momentum_dump:  "3.1d momentum",
  scattered_light: "1.0d scatter",
  half_orbital:   "6.9d harmonic",
}

export function WaveletChart({ wavelet, height = 200 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !wavelet?.power?.length) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const nPeriods = wavelet.periods.length
    const nTime    = wavelet.time.length
    const W = canvas.width
    const H = canvas.height

    // Find global max power for normalization
    let maxPower = 0
    for (const row of wavelet.power)
      for (const v of row)
        if (v > maxPower) maxPower = v

    // Draw heatmap pixel by pixel using ImageData for performance
    const imageData = ctx.createImageData(W, H)

    for (let py = 0; py < H; py++) {
      // Map pixel y → period index (log scale, low period at top)
      const periodIdx = Math.floor((py / H) * nPeriods)
      
      for (let px = 0; px < W; px++) {
        // Map pixel x → time index
        const timeIdx = Math.floor((px / W) * nTime)
        const power = wavelet.power[periodIdx]?.[timeIdx] ?? 0
        const norm  = Math.min(power / maxPower, 1)

        // Inferno-like colormap: black → purple → orange → yellow
        const r = Math.floor(255 * Math.pow(norm, 0.6))
        const g = Math.floor(255 * Math.pow(norm, 1.8))
        const b = Math.floor(255 * (norm < 0.5 ? norm * 1.5 : 1 - norm))

        const idx = (py * W + px) * 4
        imageData.data[idx]     = r
        imageData.data[idx + 1] = g
        imageData.data[idx + 2] = b
        imageData.data[idx + 3] = 255
      }
    }

    ctx.putImageData(imageData, 0, 0)

    // Overlay dashed lines at TESS systematic periods
    const periods = wavelet.periods
    const logMin  = Math.log10(periods[0])
    const logMax  = Math.log10(periods[periods.length - 1])

    ctx.setLineDash([4, 4])
    ctx.lineWidth = 1

    for (const sys of wavelet.tess_systematic_periods) {
      const logP = Math.log10(sys.period_days)
      const py   = Math.floor(((logP - logMin) / (logMax - logMin)) * H)

      ctx.strokeStyle = "rgba(255, 255, 255, 0.6)"
      ctx.beginPath()
      ctx.moveTo(0, py)
      ctx.lineTo(W, py)
      ctx.stroke()

      // Label
      ctx.setLineDash([])
      ctx.fillStyle = "rgba(255,255,255,0.8)"
      ctx.font = "10px monospace"
      ctx.fillText(SYSTEMATIC_LABELS[sys.name] ?? sys.name, 4, py - 3)
      ctx.setLineDash([4, 4])
    }

    ctx.setLineDash([])
  }, [wavelet])

  return (
    <div style={{ position: "relative" }}>
      <canvas
        ref={canvasRef}
        width={800}
        height={height}
        style={{ width: "100%", height: `${height}px`, display: "block" }}
      />
      <div style={{
        position: "absolute", bottom: 4, right: 8,
        fontSize: 10, fontFamily: "monospace",
        color: "rgba(255,255,255,0.5)"
      }}>
        period (days, log) ↑ · time →
      </div>
    </div>
  )
}
```

### 5.6 CentroidChart Component (components/CentroidChart.tsx)

Displays centroid displacement from baseline in arcseconds over time. A flat line near zero is what you want. Any spike during a flagged event window suggests the signal is off-target.

```typescript
// components/CentroidChart.tsx
"use client"
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ReferenceLine, ReferenceArea, ResponsiveContainer
} from "recharts"
import type { CentroidResult, FlaggedEvent } from "@/lib/types"

interface Props {
  centroid: CentroidResult
  events?: FlaggedEvent[]
  height?: number
}

export function CentroidChart({ centroid, events = [], height = 140 }: Props) {
  if (!centroid?.available) {
    return (
      <div style={{
        height, display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 12, fontFamily: "monospace",
        color: "var(--color-text-tertiary)"
      }}>
        TPF centroid data unavailable for this target
      </div>
    )
  }

  const data = centroid.time.map((t, i) => ({
    t: +t.toFixed(2),
    d: +(centroid.displacement_arcsec[i] ?? 0).toFixed(2),
  })).filter(d => !isNaN(d.d))

  // Contamination threshold line
  const contaminationThreshold = 10.0

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
        <XAxis
          dataKey="t"
          tick={{ fontSize: 10, fontFamily: "monospace" }}
          tickLine={false}
          label={{ value: "BTJD", position: "insideBottomRight", fontSize: 10 }}
        />
        <YAxis
          tick={{ fontSize: 10, fontFamily: "monospace" }}
          tickLine={false}
          tickFormatter={v => `${v}"`}
          width={36}
          label={{ value: 'arcsec', angle: -90, position: 'insideLeft', fontSize: 10 }}
        />
        <Tooltip
          contentStyle={{ fontSize: 11, fontFamily: "monospace" }}
          formatter={(v: number) => [`${v.toFixed(1)}"`, "centroid shift"]}
          labelFormatter={(t: number) => `t = ${t} BTJD`}
        />

        {/* Shade event windows */}
        {events.map(ev => (
          <ReferenceArea
            key={ev.id}
            x1={+(ev.time_center - ev.duration_hours / 48).toFixed(2)}
            x2={+(ev.time_center + ev.duration_hours / 48).toFixed(2)}
            fill={ev.event_type === "contamination"
              ? "rgba(163,45,45,0.15)"
              : "rgba(24,95,165,0.08)"}
          />
        ))}

        {/* Contamination threshold */}
        <ReferenceLine
          y={contaminationThreshold}
          stroke="#A32D2D"
          strokeDasharray="4 4"
          label={{ value: "contamination threshold", fontSize: 9, fill: "#A32D2D", position: "insideTopRight" }}
        />

        <Line
          type="monotone"
          dataKey="d"
          stroke={centroid.shift_flagged ? "#A32D2D" : "#185FA5"}
          strokeWidth={1}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
```

### 5.7 Analysis Page (app/analyze/[tic]/page.tsx)

```typescript
"use client"
import { useEffect, useState } from "react"
import { startAnalysis, pollAnalysis } from "@/lib/api"
import { LightCurveChart } from "@/components/LightCurveChart"
import { AnomalyScoreChart } from "@/components/AnomalyScoreChart"
import { PeriodogramChart } from "@/components/PeriodogramChart"
import { WaveletChart } from "@/components/WaveletChart"
import { CentroidChart } from "@/components/CentroidChart"
import { EventFlagList } from "@/components/EventFlagList"
import { MetricCards } from "@/components/MetricCards"
import type { Analysis } from "@/lib/types"

export default function AnalyzePage({ params }: { params: { tic: string } }) {
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [status, setStatus] = useState<string>("idle")

  const tabs = ["raw", "detrended", "periodogram", "wavelet", "centroid"] as const
  const [activeTab, setActiveTab] = useState<typeof tabs[number]>("raw")

  useEffect(() => {
    const run = async () => {
      setStatus("starting")
      const { analysis_id } = await startAnalysis(decodeURIComponent(params.tic))
      setStatus("running")
      await pollAnalysis(analysis_id, setAnalysis)
      setStatus("complete")
    }
    run()
  }, [params.tic])

  return (
    <main className="max-w-5xl mx-auto px-6 py-8">
      <div className="mb-6">
        <h1 className="text-sm font-mono uppercase tracking-widest text-gray-500 mb-1">
          Analysis
        </h1>
        <p className="text-xl font-mono font-medium">
          {decodeURIComponent(params.tic)}
        </p>
        <p className="text-xs font-mono text-gray-400 mt-1">
          {status === "running" ? "Pipeline running..." : analysis?.target?.tic_id}
        </p>
      </div>

      {analysis && <MetricCards analysis={analysis} />}

      <div className="flex gap-0 mb-4 border-b border-gray-200 dark:border-gray-800">
        {tabs.map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-xs font-mono uppercase tracking-wider border-b-2 -mb-px transition-colors ${
              activeTab === tab
                ? "border-gray-900 dark:border-gray-100 text-gray-900 dark:text-gray-100"
                : "border-transparent text-gray-400 hover:text-gray-600"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {analysis && (
        <div className="border border-gray-200 dark:border-gray-800 rounded-lg p-4 mb-4">
          {activeTab === "raw" && (
            <LightCurveChart
              time={analysis.raw_flux.time}
              flux={analysis.raw_flux.flux}
              events={analysis.events}
              height={220}
            />
          )}
          {activeTab === "detrended" && (
            <LightCurveChart
              time={analysis.detrended_flux.time}
              flux={analysis.detrended_flux.flux}
              events={analysis.events}
              height={220}
            />
          )}
          {activeTab === "periodogram" && (
            <PeriodogramChart
              periods={analysis.periodogram.period}
              powers={analysis.periodogram.power}
              bestPeriod={analysis.known_period}
              height={220}
            />
          )}
          {activeTab === "wavelet" && (
            <>
              <div className="flex justify-between items-center mb-3">
                <p className="text-xs font-mono uppercase tracking-widest text-gray-400">
                  Wavelet power spectrum
                </p>
                {analysis.wavelet.tess_systematic_periods.length > 0 && (
                  <div className="flex gap-2">
                    {analysis.wavelet.tess_systematic_periods.map(s => (
                      <span key={s.name} className="text-xs font-mono px-2 py-0.5 rounded
                        bg-gray-100 dark:bg-gray-800 text-gray-500">
                        {s.name} artifact detected
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <WaveletChart wavelet={analysis.wavelet} height={200} />
              <p className="text-xs font-mono text-gray-400 mt-2">
                Dominant non-systematic periods: {analysis.wavelet.dominant_periods.map(p => `${p}d`).join(", ") || "none"}
              </p>
            </>
          )}
          {activeTab === "centroid" && (
            <>
              <div className="flex justify-between items-center mb-3">
                <p className="text-xs font-mono uppercase tracking-widest text-gray-400">
                  Centroid displacement
                </p>
                {analysis.centroid.available && (
                  <span className={`text-xs font-mono px-2 py-0.5 rounded ${
                    analysis.centroid.shift_flagged
                      ? "bg-red-50 dark:bg-red-950 text-red-700 dark:text-red-300"
                      : "bg-green-50 dark:bg-green-950 text-green-700 dark:text-green-300"
                  }`}>
                    {analysis.centroid.shift_flagged
                      ? `max shift ${analysis.centroid.max_shift_arcsec}" — contamination suspected`
                      : `max shift ${analysis.centroid.max_shift_arcsec}" — on target`}
                  </span>
                )}
              </div>
              <CentroidChart
                centroid={analysis.centroid}
                events={analysis.events}
                height={160}
              />
            </>
          )}
        </div>
      )}

      {analysis && (
        <div className="border border-gray-200 dark:border-gray-800 rounded-lg p-4 mb-4">
          <p className="text-xs font-mono uppercase tracking-widest text-gray-400 mb-3">
            Anomaly score
          </p>
          <AnomalyScoreChart
            time={analysis.score_timeline.time}
            scores={analysis.score_timeline.score}
            height={100}
          />
        </div>
      )}

      {analysis && analysis.events.length > 0 && (
        <EventFlagList events={analysis.events} />
      )}

      {status === "running" && !analysis && (
        <div className="text-center py-20 font-mono text-sm text-gray-400">
          Fetching TESS data from MAST...
        </div>
      )}
    </main>
  )
}
```

### 5.8 Running the Frontend

```powershell
cd frontend
npm run dev
```

App live at `http://localhost:3000`

---

## 6. Running Everything Together

```powershell
# Terminal 1 — backend
cd backend
.\venv\Scripts\activate
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev
```

Open `http://localhost:3000`, type `K2-18` or any TIC ID, hit Analyze.

---

## 7. Good First Targets to Test With

| Target | TIC ID | Why interesting |
|--------|--------|-----------------|
| K2-18 | TIC 234994474 | Has known planet + tentative DMS detection |
| TRAPPIST-1 | TIC 259377017 | 7 planets, complex transit timing |
| KIC 8462852 | (Kepler, use Lightkurve K2) | Boyajian's Star — the most anomalous known |
| TOI-700 | TIC 150428135 | Rocky planet in habitable zone |
| Any quiet M dwarf | — | Good baseline test — should score near 0 |

---

## 8. Export & TFOP Submission

Once you find an interesting UNKNOWN event, export it as a standardized report:

```python
# backend/pipeline/export.py

def export_tfop_report(analysis, event) -> dict:
    """
    Format a flagged event for submission to the TESS Follow-up
    Observing Program (TFOP) working group.
    https://tess.mit.edu/followup
    """
    return {
        "tic_id": analysis.target.tic_id,
        "common_name": analysis.target.common_name,
        "ra": analysis.target.ra,
        "dec": analysis.target.dec,
        "event_time_btjd": event.time_center,
        "duration_hours": event.duration_hours,
        "depth_ppm": event.depth_ppm,
        "anomaly_score": event.anomaly_score,
        "classifier_result": event.event_type,
        "notes": event.notes,
        "pipeline_version": "1.0",
        "detection_method": "1D convolutional autoencoder + BLS",
    }
```

---

## 9. Extending the Pipeline

Once the basics work, these are the highest-leverage improvements in priority order:

**Multi-sector wavelet consistency**
Run the wavelet on each TESS sector separately and compare. A real astrophysical signal should produce consistent dominant periods across sectors. A systematic that changes between sectors (because TESS's pointing shifts slightly) should be detectable as an inconsistency. This upgrades the wavelet from a single-sector filter to a multi-sector one.

**Better training data selection**
The autoencoder is only as good as what "normal" looks like to it. Curate a cleaner training set — stars confirmed to have no planets, no variability, no companions. A contaminated training set teaches the model that some weirdness is normal.

**Cross-match against known catalogs before flagging UNKNOWN**
Before an event reaches UNKNOWN status, query SIMBAD, Gaia DR3 variable star catalog, and the TESS exoplanet catalog automatically. If a match exists, classify as KNOWN_VARIABLE or KNOWN_PLANET rather than UNKNOWN. This keeps the UNKNOWN bucket meaningful — it should only contain things that genuinely aren't in any catalog.

**Information-theoretic tests on UNKNOWN event sequences**
For targets with multiple UNKNOWN events — compute the Shannon entropy of the event timing sequence. Natural processes (comets, variable stars) produce high-entropy timing. A low-entropy sequence — events spaced in a pattern, or at intervals encoding a mathematical constant — would be the most extraordinary possible finding. Implement as a post-classification step that annotates any UNKNOWN-bearing target with an entropy score.

**Sector-to-sector depth consistency for recurring events**
If an event recurs across multiple sectors, its depth should be consistent if it's a real planet. Depth that changes sector-to-sector suggests a variable occulter — dust, a disintegrating planet, or something stranger.

---

## 10. What a Real Finding Would Look Like

With the wavelet and centroid filters in place, the UNKNOWN bucket is now much cleaner. A genuine finding would require all of the following:

1. Global anomaly score > 0.85
2. One or more events classified UNKNOWN — meaning they passed both the wavelet systematic filter and the centroid contamination filter
3. Centroid displacement during the event < 3 arcseconds (firmly on-target)
4. No wavelet power at TESS systematic periods during the event window
5. The event's morphology doesn't match any known classifier category
6. BLS finds no clean period matching the event — or finds multiple incommensurate periods simultaneously

At that point the pipeline has done everything it can with photometry alone. The technosignature module (Section 11) runs automatically on every UNKNOWN event, checking for the specific patterns that would distinguish an artificial signal from a natural one. If the technosignature score is elevated, you:

1. Check all available TESS sectors for the same target — does the event recur?
2. Look for archival Kepler or K2 data on the same star
3. The pipeline has already queried WISE infrared photometry for excess emission (Dyson structure signature) — review the results
4. Post to the TFOP working group with the exported report, requesting ground-based spectroscopic follow-up
5. If spectroscopy shows no obvious stellar variability explanation, post a preprint to arXiv astro-ph

The wavelet and centroid filters mean that by the time an event survives to UNKNOWN, it has already been checked against the two most common explanations for false positives. The technosignature module then checks for the specific hallmarks of artificiality. That's three layers of vetting before a result reaches a human.

---

## 11. Technosignature Detection Module (pipeline/technosignature.py)

This is the section that actually looks for signs of advanced extraterrestrial engineering. It runs as a post-classification step on every target that produces UNKNOWN events — events that survived the wavelet systematic filter, the centroid contamination filter, and the heuristic classifier without matching any known astrophysical category.

The module implements four independent tests. Each test produces a score and a human-readable interpretation. The tests are deliberately conservative — they are designed to produce a high score only when the data is genuinely difficult to explain with known physics. A high score does not mean aliens; it means "this warrants immediate follow-up with better instruments."

### 11.1 What We're Actually Looking For

A Dyson sphere, swarm, or megastructure partially occluding a star would produce specific observable signatures that natural phenomena do not:

1. **Non-physical light curve morphology.** Natural occultations (planets, comets, dust clouds) produce smooth curves governed by orbital mechanics and diffraction. An artificial structure could produce sharp geometric edges, perfectly flat dimming floors, or internal sub-structure within a single dip. The sharpest natural ingress is limited by the finite size of the star — any ingress faster than the stellar crossing time is physically impossible for a spherical body.

2. **Low-entropy timing patterns.** Natural transits occur at intervals determined by Kepler's third law — a single period plus noise. Comets and variable stars produce high-entropy (random-looking) timing. An artificial signal could produce intervals that encode information: evenly spaced but not orbital, at ratios of small integers, or encoding mathematical constants. The Shannon entropy of the inter-event interval distribution quantifies this.

3. **Anomalous infrared excess.** Any structure absorbing optical starlight must re-radiate the energy as thermal infrared (thermodynamics has no exceptions). A partial Dyson swarm around a Sun-like star would show normal optical brightness but anomalous mid-infrared excess at 10-25 microns. WISE W3 (12μm) and W4 (22μm) bands are sensitive to exactly this emission.

4. **Catalog non-membership.** If the star hosting an UNKNOWN event is not in any variable star catalog, not in any eclipsing binary catalog, has no known companions in Gaia DR3, and shows no spectral peculiarities in SIMBAD — then there is no known explanation for the event. This is the final filter: the event is not just unexplained by the pipeline, but unexplained by all existing astronomical knowledge.

### 11.2 The Code

```python
# backend/pipeline/technosignature.py

import numpy as np
from scipy import stats as sp_stats
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Stellar crossing time in hours for main-sequence stars by spectral type
# Any ingress faster than this is physically impossible for a spherical occulter
STELLAR_CROSSING_HOURS = {
    "O": 0.30, "B": 0.20, "A": 0.12, "F": 0.08,
    "G": 0.06, "K": 0.04, "M": 0.02,
}
DEFAULT_CROSSING_HOURS = 0.06  # assume Sun-like if unknown


def analyze(tic_id: str, time: list, flux: list,
            unknown_events: list[dict], period_result: dict,
            stellar_type: str = None) -> dict:
    """
    Run all technosignature tests on a target's UNKNOWN events.

    This function is called automatically by the pipeline for every
    analysis that produces at least one UNKNOWN event. It produces
    a composite technosignature score and individual test results.

    Returns a dict ready to serialize to JSON and store in the DB.
    """
    if not unknown_events:
        return {
            "ran": False,
            "reason": "No UNKNOWN events to analyze",
            "composite_score": 0.0,
        }

    # --- Test 1: Light curve morphology ---
    morphology = analyze_morphology(time, flux, unknown_events, stellar_type)

    # --- Test 2: Timing entropy ---
    entropy = analyze_timing_entropy(unknown_events)

    # --- Test 3: WISE infrared excess ---
    ir_excess = query_wise_excess(tic_id)

    # --- Test 4: Catalog cross-match ---
    catalog = check_catalog_membership(tic_id)

    # Composite score: geometric mean of individual scores
    # Geometric mean penalizes any single low score heavily —
    # a genuine technosignature should score high on ALL tests
    scores = [
        morphology["score"],
        entropy["score"],
        ir_excess["score"],
        catalog["score"],
    ]
    nonzero = [s for s in scores if s > 0]
    composite = float(np.exp(np.mean(np.log(nonzero)))) if nonzero else 0.0

    # Human-readable summary
    summary = generate_summary(composite, morphology, entropy, ir_excess, catalog)

    return {
        "ran": True,
        "composite_score": round(composite, 4),
        "morphology": morphology,
        "timing_entropy": entropy,
        "ir_excess": ir_excess,
        "catalog_membership": catalog,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Test 1: Light curve morphology analysis
# ---------------------------------------------------------------------------

def analyze_morphology(time: list, flux: list, events: list[dict],
                       stellar_type: str = None) -> dict:
    """
    Check each UNKNOWN event for non-physical light curve shapes.

    Looks for:
    - Ingress faster than the stellar crossing time (impossible for spherical body)
    - Perfectly flat dip floor (natural dips have limb-darkening curvature)
    - Internal sub-structure (periodic ripples within a single dip)
    - Geometric symmetry (bilateral symmetry beyond what orbital mechanics produces)
    """
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
        t_ev = t_ev[valid]
        f_ev = f_ev[valid]
        if len(f_ev) < 10:
            continue

        # --- Ingress speed ---
        min_idx = np.argmin(f_ev)
        ingress_t = t_ev[:min_idx] if min_idx > 2 else t_ev[:len(t_ev)//2]
        ingress_f = f_ev[:min_idx] if min_idx > 2 else f_ev[:len(f_ev)//2]
        if len(ingress_t) > 2:
            ingress_duration_hours = (ingress_t[-1] - ingress_t[0]) * 24
            ingress_too_fast = ingress_duration_hours < crossing_hours
        else:
            ingress_duration_hours = dur_h / 2
            ingress_too_fast = False

        # --- Flat floor detection ---
        # A perfectly flat floor has near-zero variance in the bottom 20% of flux
        sorted_flux = np.sort(f_ev)
        bottom_20 = sorted_flux[:max(3, len(sorted_flux) // 5)]
        floor_std = float(np.std(bottom_20))
        overall_std = float(np.std(f_ev))
        flat_floor = floor_std < 0.05 * overall_std if overall_std > 0 else False

        # --- Sub-structure detection ---
        # Detrend the dip region with a 2nd-order polynomial and look for
        # residual periodicity via autocorrelation
        if len(f_ev) > 20:
            poly_coeffs = np.polyfit(np.arange(len(f_ev)), f_ev, 2)
            residual = f_ev - np.polyval(poly_coeffs, np.arange(len(f_ev)))
            autocorr = np.correlate(residual, residual, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-12)
            # Look for secondary peaks above 0.3
            peaks = []
            for i in range(2, len(autocorr) - 1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                    peaks.append(i)
            has_substructure = len(peaks) >= 2
        else:
            has_substructure = False

        # --- Bilateral symmetry ---
        mid = len(f_ev) // 2
        left = f_ev[:mid]
        right = f_ev[mid:mid+len(left)][::-1]
        if len(left) == len(right) and len(left) > 5:
            symmetry_corr = float(np.corrcoef(left, right)[0, 1])
        else:
            symmetry_corr = 0.0
        # Natural transits have ~0.8-0.95 symmetry. > 0.99 is suspicious.
        hyper_symmetric = symmetry_corr > 0.99

        # Score this event
        flags = sum([ingress_too_fast, flat_floor, has_substructure, hyper_symmetric])
        event_score = flags / 4.0

        results.append({
            "time_center": tc,
            "ingress_duration_hours": round(ingress_duration_hours, 4),
            "ingress_too_fast": ingress_too_fast,
            "crossing_time_limit_hours": crossing_hours,
            "flat_floor": flat_floor,
            "floor_std_ratio": round(floor_std / (overall_std + 1e-12), 4),
            "has_substructure": has_substructure,
            "symmetry_correlation": round(symmetry_corr, 4),
            "hyper_symmetric": hyper_symmetric,
            "flags": flags,
            "score": round(event_score, 3),
        })

    if not results:
        return {"score": 0.0, "events": [], "note": "No events had enough data points"}

    best = max(results, key=lambda r: r["score"])
    return {
        "score": best["score"],
        "events": results,
        "note": _morphology_note(best),
    }


def _morphology_note(result: dict) -> str:
    flags = []
    if result["ingress_too_fast"]:
        flags.append(f"ingress ({result['ingress_duration_hours']:.3f}h) faster than "
                     f"stellar crossing time ({result['crossing_time_limit_hours']}h)")
    if result["flat_floor"]:
        flags.append("anomalously flat dip floor (floor/overall std ratio "
                     f"{result['floor_std_ratio']:.4f})")
    if result["has_substructure"]:
        flags.append("periodic sub-structure detected within dip")
    if result["hyper_symmetric"]:
        flags.append(f"bilateral symmetry {result['symmetry_correlation']:.4f} "
                     f"exceeds natural transit range")
    if not flags:
        return "Morphology consistent with natural occultation."
    return "Morphological anomalies: " + "; ".join(flags) + "."


# ---------------------------------------------------------------------------
# Test 2: Timing entropy analysis
# ---------------------------------------------------------------------------

def analyze_timing_entropy(events: list[dict]) -> dict:
    """
    Compute the Shannon entropy of the inter-event interval distribution.

    Natural processes:
    - Periodic transits → near-zero entropy (single spike in histogram)
    - Random events (comets, flares) → high entropy (uniform-ish distribution)

    An artificial signal might produce:
    - Low entropy but NOT matching an orbital period (structured non-orbital timing)
    - Intervals at ratios of small integers (e.g., 1:2:3)
    - Intervals encoding a mathematical constant

    With < 3 events, entropy is unreliable and we return a neutral score.
    """
    if len(events) < 3:
        return {
            "score": 0.0,
            "entropy": None,
            "n_events": len(events),
            "note": f"Only {len(events)} UNKNOWN event(s) — need 3+ for timing analysis.",
        }

    times = sorted([e["time_center"] for e in events])
    intervals = np.diff(times)

    if len(intervals) < 2:
        return {
            "score": 0.0,
            "entropy": None,
            "n_events": len(events),
            "note": "Insufficient intervals for entropy computation.",
        }

    # Normalize intervals to [0, 1] range
    intervals_norm = intervals / (intervals.max() + 1e-12)

    # Shannon entropy of binned interval distribution
    n_bins = min(10, len(intervals))
    hist, _ = np.histogram(intervals_norm, bins=n_bins, density=True)
    hist = hist[hist > 0]
    bin_width = 1.0 / n_bins
    entropy = -np.sum(hist * bin_width * np.log2(hist * bin_width + 1e-12))
    max_entropy = np.log2(n_bins)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Check for small-integer ratios between intervals
    ratio_matches = 0
    small_integers = [1, 2, 3, 4, 5, 6]
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            ratio = intervals[i] / (intervals[j] + 1e-12)
            for a in small_integers:
                for b in small_integers:
                    if abs(ratio - a / b) < 0.02:
                        ratio_matches += 1

    n_pairs = len(intervals) * (len(intervals) - 1) / 2
    ratio_fraction = ratio_matches / (n_pairs * len(small_integers)**2) if n_pairs > 0 else 0

    # Check for mathematical constant encoding
    known_constants = {
        "pi": 3.14159265,
        "e": 2.71828183,
        "phi": 1.61803399,
        "sqrt2": 1.41421356,
    }
    constant_matches = []
    for name, val in known_constants.items():
        for i in range(len(intervals)):
            for j in range(len(intervals)):
                if i != j and abs(intervals[i] / (intervals[j] + 1e-12) - val) < 0.01:
                    constant_matches.append(name)

    # Score: low entropy + structured intervals = interesting
    # High entropy + no structure = natural (low score)
    # Very low entropy matching an orbital period = just a planet (low score)
    structure_score = min(1.0, ratio_fraction * 50 + len(constant_matches) * 0.3)
    entropy_score = max(0, 1.0 - normalized_entropy)

    # Combined: high score requires BOTH low entropy AND non-orbital structure
    combined = (entropy_score * 0.4 + structure_score * 0.6)

    return {
        "score": round(combined, 3),
        "entropy": round(float(entropy), 4),
        "normalized_entropy": round(float(normalized_entropy), 4),
        "n_events": len(events),
        "intervals_days": [round(float(x), 4) for x in intervals],
        "ratio_matches": ratio_matches,
        "constant_matches": list(set(constant_matches)),
        "note": _entropy_note(normalized_entropy, ratio_matches, constant_matches, len(events)),
    }


def _entropy_note(norm_entropy: float, ratio_matches: int,
                  constant_matches: list, n_events: int) -> str:
    if norm_entropy > 0.8:
        return f"High entropy ({norm_entropy:.2f}) — timing appears random, consistent with natural process."
    if constant_matches:
        return (f"Low entropy ({norm_entropy:.2f}) with interval ratios matching "
                f"mathematical constant(s): {', '.join(set(constant_matches))}. "
                f"Extraordinary claim requires extraordinary evidence — request follow-up.")
    if ratio_matches > 3:
        return (f"Low entropy ({norm_entropy:.2f}) with {ratio_matches} small-integer "
                f"ratio matches between intervals. Pattern is structured but could be harmonic "
                f"of an undetected orbital period.")
    return f"Moderate entropy ({norm_entropy:.2f}) — no strong timing structure detected."


# ---------------------------------------------------------------------------
# Test 3: WISE infrared excess
# ---------------------------------------------------------------------------

def query_wise_excess(tic_id: str) -> dict:
    """
    Query the AllWISE catalog for the target and check for anomalous
    mid-infrared excess.

    A partial Dyson swarm absorbs optical light and re-radiates it
    as thermal infrared at ~300K (for a structure at ~1 AU from a Sun-like
    star). This produces excess emission in WISE W3 (12μm) and W4 (22μm)
    relative to the photospheric prediction from W1 (3.4μm) and W2 (4.6μm).

    Normal stars have W1-W4 ≈ 0 (Vega-calibrated system).
    Debris disks have W1-W4 ≈ 0.5-2.
    A Dyson swarm could produce W1-W4 > 3 depending on coverage fraction.
    """
    try:
        from astroquery.ipac.irsa import Irsa
        from astropy.coordinates import SkyCoord
        from astroquery.mast import Catalogs
        import astropy.units as u

        # Get coordinates from TIC
        tic_data = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC", radius=0.001 * u.deg)
        if len(tic_data) == 0:
            return {"score": 0.0, "available": False, "note": "TIC lookup failed."}

        ra = float(tic_data[0]["ra"])
        dec = float(tic_data[0]["dec"])
        coord = SkyCoord(ra=ra, dec=dec, unit="deg")

        # Query AllWISE catalog within 6 arcsec
        wise_results = Irsa.query_region(
            coord, catalog="allwise_p3as_psd", radius=6 * u.arcsec
        )

        if len(wise_results) == 0:
            return {"score": 0.0, "available": False, "note": "No AllWISE match within 6 arcsec."}

        w = wise_results[0]
        w1 = float(w["w1mpro"])
        w2 = float(w["w2mpro"])
        w3 = float(w["w3mpro"]) if w["w3mpro"] is not np.ma.masked else None
        w4 = float(w["w4mpro"]) if w["w4mpro"] is not np.ma.masked else None

        # Color indices (lower magnitude = brighter)
        w1_w2 = w1 - w2
        w1_w3 = (w1 - w3) if w3 is not None else None
        w1_w4 = (w1 - w4) if w4 is not None else None

        # Main-sequence stars should have W1-W3 ≈ 0 ± 0.2 and W1-W4 ≈ 0 ± 0.3
        # Infrared excess is positive W1-W3 or W1-W4
        excess_w3 = max(0, (w1_w3 - 0.2) / 2.0) if w1_w3 is not None else 0
        excess_w4 = max(0, (w1_w4 - 0.3) / 3.0) if w1_w4 is not None else 0
        score = min(1.0, max(excess_w3, excess_w4))

        return {
            "score": round(score, 3),
            "available": True,
            "w1": round(w1, 3),
            "w2": round(w2, 3),
            "w3": round(w3, 3) if w3 else None,
            "w4": round(w4, 3) if w4 else None,
            "w1_w3": round(w1_w3, 3) if w1_w3 else None,
            "w1_w4": round(w1_w4, 3) if w1_w4 else None,
            "note": _ir_note(w1_w3, w1_w4),
        }

    except Exception as e:
        logger.warning(f"WISE query failed for TIC {tic_id}: {e}")
        return {"score": 0.0, "available": False, "note": f"WISE query failed: {e}"}


def _ir_note(w1_w3: Optional[float], w1_w4: Optional[float]) -> str:
    if w1_w3 is None and w1_w4 is None:
        return "No WISE W3/W4 detections — infrared excess test inconclusive."
    parts = []
    if w1_w3 is not None:
        if w1_w3 > 2.0:
            parts.append(f"Strong W3 excess (W1-W3 = {w1_w3:.2f})")
        elif w1_w3 > 0.5:
            parts.append(f"Moderate W3 excess (W1-W3 = {w1_w3:.2f}) — possible debris disk")
        else:
            parts.append(f"W3 normal (W1-W3 = {w1_w3:.2f})")
    if w1_w4 is not None:
        if w1_w4 > 3.0:
            parts.append(f"Strong W4 excess (W1-W4 = {w1_w4:.2f}) — anomalous thermal emission")
        elif w1_w4 > 1.0:
            parts.append(f"Moderate W4 excess (W1-W4 = {w1_w4:.2f})")
        else:
            parts.append(f"W4 normal (W1-W4 = {w1_w4:.2f})")
    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Test 4: Catalog cross-match
# ---------------------------------------------------------------------------

def check_catalog_membership(tic_id: str) -> dict:
    """
    Query SIMBAD and Gaia for known classifications of this star.

    If the star is a known variable, known binary, or has known companions,
    the UNKNOWN events are more likely to have a mundane explanation that
    the heuristic classifier simply didn't cover.

    A star with NO catalog matches — not variable, not binary, no companions,
    no special spectral features — is the most interesting case.
    """
    try:
        from astroquery.simbad import Simbad
        from astroquery.mast import Catalogs
        import astropy.units as u

        # Get coordinates from TIC
        tic_data = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC", radius=0.001 * u.deg)
        if len(tic_data) == 0:
            return {"score": 0.0, "available": False, "note": "TIC lookup failed."}

        ra = float(tic_data[0]["ra"])
        dec = float(tic_data[0]["dec"])

        # Query SIMBAD
        Simbad.add_votable_fields("otype", "otypes", "flux(V)")
        simbad_result = Simbad.query_region(
            f"{ra} {dec}", radius="0d0m6s"
        )

        known_types = []
        simbad_otype = None
        if simbad_result is not None and len(simbad_result) > 0:
            simbad_otype = str(simbad_result[0]["OTYPE"])
            all_types = str(simbad_result[0].get("OTYPES", ""))

            variable_indicators = ["V*", "Pu*", "Ce*", "Mi*", "RR*", "dS*", "SX*", "gD*", "BY*"]
            binary_indicators = ["**", "EB*", "SB*", "El*", "Sy*"]

            for vi in variable_indicators:
                if vi in all_types:
                    known_types.append(f"variable ({vi})")
            for bi in binary_indicators:
                if bi in all_types:
                    known_types.append(f"binary ({bi})")

        # Score: no catalog matches = highest score (most mysterious)
        if known_types:
            score = 0.1  # known variable or binary — events probably have mundane explanation
        elif simbad_otype and simbad_otype not in ("Star", "*", "?"):
            score = 0.3  # has a classification but not variable/binary
        else:
            score = 0.9  # no known classification — genuinely unexplained

        return {
            "score": round(score, 3),
            "available": True,
            "simbad_type": simbad_otype,
            "known_types": known_types,
            "note": _catalog_note(simbad_otype, known_types),
        }

    except Exception as e:
        logger.warning(f"Catalog query failed for TIC {tic_id}: {e}")
        return {"score": 0.5, "available": False, "note": f"Catalog query failed: {e}"}


def _catalog_note(simbad_otype: Optional[str], known_types: list) -> str:
    if known_types:
        return (f"Star has known classification(s): {', '.join(known_types)}. "
                f"UNKNOWN events likely have a mundane explanation the classifier missed.")
    if simbad_otype and simbad_otype not in ("Star", "*", "?"):
        return f"SIMBAD type: {simbad_otype}. Not flagged as variable or binary, but has a classification."
    return "No variable star or binary classification in SIMBAD. Star is genuinely unremarkable in all catalogs."


# ---------------------------------------------------------------------------
# Summary generator
# ---------------------------------------------------------------------------

def generate_summary(composite: float, morphology: dict, entropy: dict,
                     ir_excess: dict, catalog: dict) -> str:
    """Generate a human-readable summary of all technosignature test results."""
    if composite < 0.1:
        return "Technosignature analysis: no anomalous indicators. Events are likely natural but unclassified."

    if composite < 0.3:
        return (f"Technosignature analysis: weak indicators (composite {composite:.3f}). "
                f"Morphology and timing are broadly consistent with natural processes. "
                f"Recommend standard TFOP follow-up.")

    if composite < 0.6:
        parts = [f"Technosignature analysis: moderate indicators (composite {composite:.3f})."]
        if morphology["score"] > 0.3:
            parts.append(f"Morphology: {morphology['note']}")
        if entropy["score"] > 0.3:
            parts.append(f"Timing: {entropy['note']}")
        if ir_excess["score"] > 0.3:
            parts.append(f"IR: {ir_excess['note']}")
        parts.append("Recommend priority spectroscopic follow-up and multi-epoch photometry.")
        return " ".join(parts)

    # composite >= 0.6
    parts = [f"ELEVATED TECHNOSIGNATURE INDICATORS (composite {composite:.3f})."]
    parts.append(f"Morphology ({morphology['score']:.2f}): {morphology['note']}")
    parts.append(f"Timing ({entropy['score']:.2f}): {entropy['note']}")
    parts.append(f"IR excess ({ir_excess['score']:.2f}): {ir_excess['note']}")
    parts.append(f"Catalog ({catalog['score']:.2f}): {catalog['note']}")
    parts.append("THIS DOES NOT MEAN ALIENS. This means all four independent tests returned "
                 "elevated scores simultaneously, which is statistically unusual. "
                 "Request immediate ground-based spectroscopic follow-up, JWST mid-IR photometry "
                 "if available, and independent reanalysis of the TESS pixel data.")
    return " ".join(parts)
```

### 11.3 TypeScript Types for Technosignature Results

Add to `lib/types.ts`:

```typescript
export interface TechnosignatureMorphologyEvent {
  time_center: number
  ingress_duration_hours: number
  ingress_too_fast: boolean
  crossing_time_limit_hours: number
  flat_floor: boolean
  floor_std_ratio: number
  has_substructure: boolean
  symmetry_correlation: number
  hyper_symmetric: boolean
  flags: number
  score: number
}

export interface TechnosignatureResult {
  ran: boolean
  reason?: string
  composite_score: number
  morphology?: {
    score: number
    events: TechnosignatureMorphologyEvent[]
    note: string
  }
  timing_entropy?: {
    score: number
    entropy: number | null
    normalized_entropy: number | null
    n_events: number
    intervals_days: number[]
    ratio_matches: number
    constant_matches: string[]
    note: string
  }
  ir_excess?: {
    score: number
    available: boolean
    w1?: number
    w2?: number
    w3?: number | null
    w4?: number | null
    w1_w3?: number | null
    w1_w4?: number | null
    note: string
  }
  catalog_membership?: {
    score: number
    available: boolean
    simbad_type: string | null
    known_types: string[]
    note: string
  }
  summary: string
}
```

Add `technosignature` field to the `Analysis` interface:

```typescript
export interface Analysis {
  // ... existing fields ...
  technosignature: TechnosignatureResult
}
```

### 11.4 Interpreting the Composite Score

| Composite Score | Interpretation | Action |
|----------------|----------------|--------|
| 0.0 – 0.1 | No anomalous indicators | Standard TFOP if UNKNOWN events exist |
| 0.1 – 0.3 | Weak indicators | One test scored above baseline; likely natural |
| 0.3 – 0.6 | Moderate indicators | Multiple tests elevated; priority follow-up recommended |
| 0.6 – 0.8 | Strong indicators | Statistically unusual combination; request ground-based spectroscopy |
| 0.8 – 1.0 | Extraordinary indicators | All four tests elevated simultaneously; request independent reanalysis before any announcement |

The composite score uses geometric mean specifically because a genuine technosignature must score high on ALL tests simultaneously. A star with anomalous IR but normal morphology is almost certainly a debris disk, not a Dyson swarm. A star with low-entropy timing but normal IR is probably an undetected harmonic oscillation. Only when morphology, timing, IR, and catalog non-membership all point the same direction does the composite score become meaningful.

### 11.5 Why These Tests and Not Others

**Why not radio SETI?** Radio telescopes have their own pipelines (Breakthrough Listen, SETI@home). This pipeline operates entirely on optical/infrared photometry — what TESS actually measures. Radio follow-up is a logical next step for any high-scoring target.

**Why not look for laser pulses?** TESS's 2-minute cadence cannot resolve nanosecond laser pulses. TESS is sensitive to structures that occlude starlight over minutes to hours — megastructures, not communication beams.

**Why geometric mean?** Arithmetic mean would let a single high score (e.g., a debris disk with high IR excess) inflate the composite. Geometric mean requires breadth: every test must contribute. This matches the logic of technosignature detection — you need convergent evidence across independent observables.

**Why include catalog cross-match as a "test"?** Because the prior probability of a genuinely unexplained event is radically different for a star with no known peculiarities versus a known RS CVn variable. The catalog test encodes this prior.