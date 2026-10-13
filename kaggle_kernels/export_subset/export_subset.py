"""Copy a small slice of the competition DICOMs into one zip so it can be pulled with `kaggle kernels output`.

The CLI cannot download a folder, and per-file downloads get rate limited after roughly 100 files,
so a kernel that packs the subset server-side is the only practical way to get DICOMs onto a laptop.
"""
import glob
import os
import zipfile

import pandas as pd

# The competition mount point differs between Kaggle images, so locate it by a file it must contain.
# Depth-limited, because a recursive glob would walk all 819k DICOMs on the network mount.
COMP = next(os.path.dirname(p) for d in range(1, 4) for p in glob.glob("/kaggle/input/" + "*/" * d + "train_series.csv"))
print("competition root", COMP)
N_LABELED = int(os.environ.get("N_LABELED", "20"))
OUT = "/kaggle/working/subset.zip"

train = pd.read_csv(f"{COMP}/train.csv")
labels = [c for c in train.columns if c not in ("StudyInstanceUID", "Report")]
labeled = train[train[labels].notna().all(axis=1)]
picked = labeled.sample(n=min(N_LABELED, len(labeled)), random_state=0).StudyInstanceUID.tolist()
test_ids = pd.read_csv(f"{COMP}/test.csv").StudyInstanceUID.tolist()

# Kaggle keeps at most 500 output items, so everything goes into a single archive.
# ZIP_STORED because DICOM pixel data is often already JPEG-compressed and deflate wastes CPU on it.
n_files = 0
with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    for split, ids in (("train_series", picked), ("test_series", test_ids)):
        for uid in ids:
            root = f"{COMP}/{split}/{uid}"
            for dirpath, _, files in os.walk(root):
                for f in files:
                    full = os.path.join(dirpath, f)
                    zf.write(full, os.path.relpath(full, COMP))
                    n_files += 1

print(f"studies train={len(picked)} test={len(test_ids)} files={n_files} zip_bytes={os.path.getsize(OUT)}")
