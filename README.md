# CurveScope

**TESS light-curve anomaly lab** — scan NASA TESS photometry with a trainable autoencoder, classify events, and score unidentified signals against natural and artificial hypotheses.

## What it does

CurveScope downloads light curves from NASA's TESS space telescope, scores them with a neural network, classifies what it finds, and — when something defies natural explanation — ranks 13 candidate hypotheses including six types of alien technology.

### Pipeline stages

1. **Download** — FITS light curves fetched directly from AWS S3 (`stpubdata` public bucket) with parallel async downloads, falling back to the MAST API when S3 paths can't be resolved.

2. **Clean** — Normalize flux, remove outliers, detrend slow stellar variability with a sliding median filter.

3. **Autoencoder scoring** — A 1D convolutional autoencoder trained on quiet stars learns to reconstruct "normal" flux patterns. Regions it can't reconstruct get high anomaly scores (0–1).

4. **Period search** — Box Least Squares (BLS) periodogram finds repeating transit signals across multiple candidate periods.

5. **Wavelet transform** — Continuous wavelet decomposition flags TESS spacecraft artifacts (orbital period, momentum dumps) so they aren't confused with astrophysics.

6. **Centroid analysis** — Target Pixel File (TPF) analysis checks whether the star's position shifts during brightness dips — large shifts indicate the signal comes from a nearby contaminating source, not the target.

7. **Classification** — Multi-pass event classifier:
   - Per-event decision tree: transit, flare, exocomet, eclipsing binary, depth anomaly, starspot, etc.
   - Ensemble pass: period-consistency partition, eclipsing binary check, stellar variability check
   - Events that survive all classifiers as **UNKNOWN** trigger the technosignature module

8. **Technosignature analysis** — Four independent modules score UNKNOWN events:
   - **Morphology** — ingress speed, floor flatness, bilateral symmetry, substructure
   - **Timing entropy** — mathematical structure in event spacing (integer ratios, constants)
   - **IR excess** — AllWISE mid-infrared colors for anomalous thermal emission
   - **Catalog membership** — SIMBAD lookup for prior classification

9. **Hypothesis generator** — Ranks 13 candidate explanations scored against observed features:
   - **Natural (7):** debris disk, disintegrating planet, ring system, Trojan swarm, brown dwarf, exomoon, instrumental artifact
   - **Artificial (6):** Dyson sphere/megastructure, artificial transit beacon, Clarke exobelt, laser beacon, stellar engine, solar collector swarm

10. **Sector scanning** — Batch-processes entire TESS sectors (~16,000+ targets). Prefetches all product metadata with a single MAST query to eliminate per-target API round-trips.

## Quick start

### Prerequisites

- Python 3.11+ with venv
- Node.js 18+
- ~2 GB disk for FITS cache per sector (configurable location)

### First-time setup

```bash
# 1. Clone
git clone https://github.com/your-username/tess-lightcurveanomoly.git
cd tess-lightcurveanomoly

# 2. Backend
cd backend
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# 3. Configure data directory
# Edit backend/.env — set DATA_DIR to where you want data stored
# e.g. DATA_DIR=E:\tess-data

# 4. Frontend
cd ../frontend
npm install

# 5. Root (for concurrently)
cd ..
npm install
```

### Run

From the repository root:

```bash
npm run dev
```

This kills any stale processes on ports 8000/3000, then starts both servers:

- **API:** http://127.0.0.1:8000 (OpenAPI docs at `/docs`)
- **App:** http://localhost:3000

`Ctrl+C` stops both.

### Train the autoencoder

Before scanning, the model needs training data. Either:

- Use **Settings → Train model** in the UI, or
- Run `python -m pipeline.train` from `backend/` with the venv activated

Training fetches ~85 known-quiet stars and trains the autoencoder. Takes a few minutes.

### Run servers separately

```bash
# Terminal 1 — backend (Windows)
cd backend
.\venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev
```

On macOS/Linux, replace `.\venv\Scripts\python.exe` with `./venv/bin/python`.

## Configuration

### `backend/.env`

```env
DATA_DIR=E:\tess-data          # Base directory for all data
# MAST_CACHE_DIR=...           # Override FITS cache location
# MODEL_WEIGHTS_DIR=...        # Override model weights location
# DATABASE_URL=sqlite:///...   # Override database location
LOG_LEVEL=info
```

### `frontend/.env`

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000/api
```

Change this if the backend runs on a different host/port.

### Data persistence

- `DATA_DIR` controls where the SQLite database, MAST/S3 FITS cache, and model weights (`autoencoder_v1.pt` + `autoencoder_v1.stats.npz`) are stored.
- Use an absolute path so training and the server always resolve the same location.
- On startup, check the log for `model_weights=present` / `missing` to verify.

## Features

- **Single-target analysis** — Enter a TIC ID or common name, get full pipeline results with interactive charts
- **Sector scanning** — Batch-scan entire TESS sectors, ranking all targets by anomaly score
- **S3 direct downloads** — Parallel async FITS downloads from the TESS public S3 bucket
- **Sector prefetching** — Single bulk MAST query for all products in a sector before scanning begins
- **Interactive charts** — Raw/detrended light curves, BLS periodogram, wavelet power spectrum, centroid motion, TPF pixel viewer
- **Hypothesis analysis** — Ranked natural + artificial explanations for UNKNOWN events with per-hypothesis reasoning
- **Image export** — Export black-and-white cover images of any target's light curve for sharing
- **Iterative training** — Retrain the autoencoder on confirmed-quiet stars from completed scans to improve each cycle

## Stack

| Layer | Tech |
|---|---|
| Backend | Python, FastAPI, PyTorch, Lightkurve, Astroquery, httpx |
| Frontend | Next.js 14, React 18, TypeScript 5, Tailwind CSS, Recharts |
| Data sources | TESS FITS (AWS S3 / MAST), AllWISE (IRSA), SIMBAD |
| Storage | SQLite, in-memory chart cache |
| Model | 1D convolutional autoencoder |

## Project structure

```
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Settings from .env
│   ├── routers/
│   │   └── analyze.py          # API endpoints + pipeline orchestration
│   ├── pipeline/
│   │   ├── fetch.py            # Light curve download (S3 + lightkurve)
│   │   ├── s3_fetch.py         # Direct S3 downloads + sector prefetch
│   │   ├── clean.py            # Normalize, detrend, outlier removal
│   │   ├── autoencoder.py      # Model architecture + scoring
│   │   ├── train.py            # Training loop
│   │   ├── periodogram.py      # BLS period search
│   │   ├── wavelet.py          # CWT + systematic detection
│   │   ├── centroid.py         # TPF centroid analysis
│   │   ├── classifier.py       # Event classification + ensemble
│   │   ├── technosignature.py  # Technosignature scoring modules
│   │   ├── hypothesis.py       # Ranked hypothesis generator
│   │   └── scanner.py          # Sector batch scanner
│   └── db/
│       ├── models.py           # SQLAlchemy models
│       └── database.py         # Session management
├── frontend/
│   ├── app/                    # Next.js pages
│   ├── components/             # React components
│   └── lib/                    # API client, types, branding
├── scripts/
│   └── kill-stale.js           # Pre-dev process cleanup
└── package.json                # Root: runs both servers via concurrently
```

## Branding

- **Web:** [`frontend/lib/brand.ts`](frontend/lib/brand.ts)
- **API:** [`backend/brand.py`](backend/brand.py)
