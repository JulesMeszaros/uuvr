import os
import urllib.request
from pathlib import Path

WEIGHTS_URL = "https://huggingface.co/fastrolling/uvr/resolve/main/Main_Models/2_HP-UVR.pth"
WEIGHTS_FILENAME = "2_HP-UVR.pth"


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache")
    return Path(base) / "uuvr"


def ensure_weights() -> Path:
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / WEIGHTS_FILENAME
    if not weights_path.exists():
        print(f"Téléchargement des poids du modèle vers {weights_path} ...")
        tmp_path = weights_path.with_suffix(".pth.part")
        urllib.request.urlretrieve(WEIGHTS_URL, tmp_path)
        tmp_path.rename(weights_path)
    return weights_path
