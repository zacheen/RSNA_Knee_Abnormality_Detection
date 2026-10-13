"""Paths and constants shared by every stage, resolved for either a local checkout or a Kaggle kernel."""
import glob
import os
import subprocess
from pathlib import Path

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
# Image slots in cache order, each (plane, fluid-sensitive). The first three are the original one-series-per-plane
# layout, so a 5-slot volume sliced to [:3] is exactly a 3-slot volume and 3-slot checkpoints run on it unchanged.
SLOT_SPECS = {
    "Sagittal": ("Sagittal", True), "Coronal": ("Coronal", True), "Axial": ("Axial", True),
    "Sagittal_nonFS": ("Sagittal", False), "Coronal_nonFS": ("Coronal", False),
}
ALL_SLOTS = list(SLOT_SPECS)
assert ALL_SLOTS[:3] == ["Sagittal", "Coronal", "Axial"], "3-slot caches and checkpoints depend on this prefix"
# select_series excludes series already taken, so a plane's FS slot must come before its non-FS slot.
assert all(any(SLOT_SPECS[t] == (SLOT_SPECS[s][0], True) for t in ALL_SLOTS[:i])
           for i, s in enumerate(ALL_SLOTS) if not SLOT_SPECS[s][1])
PLANES = ALL_SLOTS[: int(os.environ.get("RSNA_SLOTS", "3"))]


def is_slot_prefix(slots, available) -> bool:
    """True when slots are the leading entries of available, the only way a model may read a volume with more slots."""
    return list(slots) == list(available)[: len(slots)]

N_SLICES = int(os.environ.get("RSNA_N_SLICES", "32"))
IMG_SIZE = int(os.environ.get("RSNA_IMG_SIZE", "192"))
# Crop to foreground and pad to square before resizing. Off by default so caches built without it stay valid.
CROP_FG = os.environ.get("RSNA_CROP_FG", "0") == "1"
# Saved next to the weights by train.py and compared by infer.py. One dict, so a key cannot exist on only one side.
SETTINGS = {"img_size": IMG_SIZE, "n_slices": N_SLICES, "crop_fg": CROP_FG, "labels": LABELS, "planes": PLANES}

REPO = Path(__file__).resolve().parents[1]
ON_KAGGLE = Path("/kaggle/input").exists()


def code_version() -> str:
    """Commit the code came from, -dirty when src/ had uncommitted edits.

    Deployed copies read the VERSION file that tools/code_version.sh writes next to them. A local checkout asks git.
    """
    f = Path(__file__).resolve().parent / "VERSION"
    if f.exists():
        return f.read_text().strip()
    try:
        git = ["git", "-C", str(REPO)]
        head = subprocess.run([*git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run([*git, "status", "--porcelain", "--", "src"], capture_output=True, text=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _find_data_root() -> Path:
    if "RSNA_DATA_ROOT" in os.environ:
        return Path(os.environ["RSNA_DATA_ROOT"])
    if ON_KAGGLE:
        # Depth-limited globs. A recursive glob would walk all 819k DICOMs on the network mount.
        for depth in range(1, 4):
            hits = sorted(glob.glob("/kaggle/input/" + "*/" * depth + "train_series.csv"))
            if hits:
                return Path(hits[0]).parent
        raise FileNotFoundError("competition data is not attached to this kernel")
    return REPO / "data" / "raw"


DATA_ROOT = _find_data_root()
WORK = Path(os.environ.get("RSNA_WORK", "/kaggle/working" if ON_KAGGLE else str(REPO / "data" / "work")))
