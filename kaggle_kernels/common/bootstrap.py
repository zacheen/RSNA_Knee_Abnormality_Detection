"""Pasted at the top of every cloud kernel. Finds the assets dataset, puts it on sys.path and fills in missing packages."""
import glob
import importlib.util
import subprocess
import sys
from pathlib import Path


def find_assets() -> Path:
    # Mount depth differs between Kaggle images. The name is unique to the assets dataset.
    for depth in range(1, 5):
        hits = glob.glob("/kaggle/input/" + "*/" * depth + "build_cache.py")
        if hits:
            return Path(hits[0]).parent
    raise FileNotFoundError("rsna-knee-assets dataset is not attached")


# import name -> pip name. Scoring runs offline, so anything the image lacks must come from wheels in the dataset.
REQUIRED = {"pydicom": "pydicom", "pylibjpeg": "pylibjpeg", "libjpeg": "pylibjpeg-libjpeg",
            "openjpeg": "pylibjpeg-openjpeg", "timm": "timm", "cv2": "opencv-python-headless"}


def ensure_packages(assets: Path):
    missing = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print("installing from local wheels:", missing, flush=True)
        # --no-index and --no-deps keep pip from trying to resolve anything over the network, which is unreachable at scoring.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", str(assets),
                        *missing], check=True)


ASSETS = find_assets()
ensure_packages(ASSETS)
sys.path.insert(0, str(ASSETS))
print("assets", ASSETS, "python", sys.version.split()[0], flush=True)
