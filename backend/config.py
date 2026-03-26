import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Directory containing this file (the `backend` package). Used so relative DATA_DIR etc.
# do not depend on the shell's current working directory (uvicorn from repo root vs backend).
_BACKEND_DIR = Path(__file__).resolve().parent
_ENV_FILE = _BACKEND_DIR / ".env"


def _load_env() -> None:
    load_dotenv(_ENV_FILE, override=True)


_load_env()


def _resolve_storage_path(raw: str, *, anchor: Path) -> Path:
    """
    Resolve a path from env. Absolute paths are normalized with .resolve().
    Relative paths are resolved against `anchor` (backend dir), not process cwd.
    """
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (anchor / p).resolve()


class Settings:
    def __init__(self):
        self.data_dir = _resolve_storage_path(os.getenv("DATA_DIR", "./data"), anchor=_BACKEND_DIR)

        raw_mast = os.getenv("MAST_CACHE_DIR")
        if raw_mast:
            self.mast_cache_dir = _resolve_storage_path(raw_mast, anchor=_BACKEND_DIR)
        else:
            self.mast_cache_dir = (self.data_dir / "cache").resolve()

        raw_weights = os.getenv("MODEL_WEIGHTS_DIR")
        if raw_weights:
            self.model_weights_dir = _resolve_storage_path(raw_weights, anchor=_BACKEND_DIR)
        else:
            self.model_weights_dir = (self.data_dir / "weights").resolve()

        db_url = os.getenv("DATABASE_URL")
        if db_url:
            self.database_url = db_url
        else:
            db_path = self.data_dir / "tess_anomaly.db"
            self.database_url = f"sqlite:///{db_path}"

        self.model_weights_path = self.model_weights_dir / "autoencoder_v1.pt"
        self.model_stats_path = self.model_weights_dir / "autoencoder_v1.stats.npz"
        self.training_targets_path = self.data_dir / "training_targets.json"
        self.training_cache_dir = (self.data_dir / "training_cache").resolve()
        self.log_level = os.getenv("LOG_LEVEL", "info")

        self._ensure_dirs()

    def _ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mast_cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_weights_dir.mkdir(parents=True, exist_ok=True)
        self.training_cache_dir.mkdir(parents=True, exist_ok=True)

    def reload_from_env(self) -> None:
        _load_env()
        self.__init__()


def write_data_dir_to_env(new_path: str) -> None:
    """Persist DATA_DIR to backend/.env and reload process env."""
    p = Path(new_path).expanduser()
    path_str = str(p.resolve())
    lines: list[str] = []
    if _ENV_FILE.exists():
        lines = _ENV_FILE.read_text(encoding="utf-8").splitlines()
    found = False
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("DATA_DIR=") or stripped.startswith("export DATA_DIR="):
            new_lines.append(f"DATA_DIR={path_str}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"DATA_DIR={path_str}")
    _ENV_FILE.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
    _load_env()


settings = Settings()


# ---- Training targets JSON ------------------------------------------------

_DEFAULT_TARGETS = [
    {"tic_id": "261136679", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "38846515",  "anomaly_score": None, "source": "built-in"},
    {"tic_id": "144700903", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "207468071", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "149603524", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "261108236", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "55652896",  "anomaly_score": None, "source": "built-in"},
    {"tic_id": "92226327",  "anomaly_score": None, "source": "built-in"},
    {"tic_id": "167600516", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "220459811", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "271748799", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "362249359", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "12421862",  "anomaly_score": None, "source": "built-in"},
    {"tic_id": "382188847", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "441462736", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "300015238", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "259962054", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "410214986", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "25155310",  "anomaly_score": None, "source": "built-in"},
    {"tic_id": "394137592", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "141608198", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "388857263", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "229945862", "anomaly_score": None, "source": "built-in"},
    {"tic_id": "49899799",  "anomaly_score": None, "source": "built-in"},
    {"tic_id": "165370459", "anomaly_score": None, "source": "built-in"},
]


def load_training_targets() -> list[dict]:
    p = settings.training_targets_path
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    save_training_targets(_DEFAULT_TARGETS)
    return list(_DEFAULT_TARGETS)


def save_training_targets(targets: list[dict]) -> None:
    p = settings.training_targets_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(targets, indent=2) + "\n", encoding="utf-8")
