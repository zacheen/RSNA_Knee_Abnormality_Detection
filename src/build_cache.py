"""Convert studies into (slots, N_SLICES, IMG_SIZE, IMG_SIZE) uint8 arrays, one slot per selected series.

The same load_study() is used by infer.py, so training cache and test-time inputs cannot drift apart.

Usage
  python src/build_cache.py --split train [--only-present] [--limit 20] [--shard 0 --num-shards 4] [--zip]
  python src/build_cache.py --split test
"""
import argparse
import os
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CROP_FG, DATA_ROOT, IMG_SIZE, N_SLICES, PLANES, SLOT_SPECS, WORK  # noqa: E402
from dicom_io import long_path, read_series  # noqa: E402


def select_series(study_uid: str, series: pd.DataFrame, split_dir: Path) -> dict:
    """Pick one series per slot: fluid-sensitive first, then most slices.

    An FS slot falls back to any series of its plane, which keeps 3-slot caches unchanged. A non-FS slot takes only
    non-FS series and never one an earlier slot took, which is why SLOT_SPECS lists FS slots first.
    """
    picked = {}
    for slot in PLANES:
        plane, fluid = SLOT_SPECS[slot]
        cand = series[(series.Anatomical_Plane == plane) & ~series.SeriesInstanceUID.isin(picked.values())].copy()
        if not fluid:
            cand = cand[cand.Fluid_Sensitive == 0]
        dirs = [long_path(split_dir / study_uid / s) for s in cand.SeriesInstanceUID]
        cand["n_files"] = [len(os.listdir(d)) if d.is_dir() else 0 for d in dirs]
        cand = cand[cand.n_files > 0]
        picked[slot] = None if cand.empty else \
            cand.sort_values(["Fluid_Sensitive", "n_files"], ascending=False).iloc[0].SeriesInstanceUID
    return picked


def load_study(study_uid: str, series: pd.DataFrame, split_dir: Path):
    """Return (volume uint8 [P,D,S,S], present bool [P]) over the PLANES slots. A missing or unreadable slot stays zero."""
    vol = np.zeros((len(PLANES), N_SLICES, IMG_SIZE, IMG_SIZE), np.uint8)
    present = np.zeros(len(PLANES), bool)
    picked = select_series(study_uid, series, split_dir)
    # Index by PLANES, which also fixes the model's pool order and the has_* columns.
    for p, plane in enumerate(PLANES):
        s_uid = picked[plane]
        if s_uid is None:
            continue
        try:
            vol[p] = read_series(split_dir / study_uid / s_uid, N_SLICES, IMG_SIZE, CROP_FG)
            present[p] = True
        except Exception as e:  # one corrupt series must not cost the whole study
            print(f"[warn] {study_uid} {plane} {s_uid}: {type(e).__name__}: {e}", flush=True)
    return vol, present


def _worker(args):
    study_uid, series, split_dir, out_dir = args
    t0 = time.time()
    vol, present = load_study(study_uid, series, split_dir)
    np.save(out_dir / f"{study_uid}.npy", vol)
    return {"StudyInstanceUID": study_uid, **{f"has_{p}": bool(x) for p, x in zip(PLANES, present)},
            "seconds": round(time.time() - t0, 2), "crop_fg": CROP_FG}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--only-present", action="store_true", help="skip studies whose folder is absent (local subset)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--zip", action="store_true", help="pack output into one zip (Kaggle keeps at most 500 output items)")
    ap.add_argument("--zip-dir", type=Path, default=WORK,
                    help="on Kaggle point WORK at /kaggle/tmp and this at /kaggle/working, so only the zip counts toward the 20 GB output cap")
    args = ap.parse_args()

    split_dir = DATA_ROOT / f"{args.split}_series"
    studies = pd.read_csv(DATA_ROOT / f"{args.split}.csv").StudyInstanceUID.tolist()
    if args.only_present:
        studies = [s for s in studies if (split_dir / s).is_dir()]
    studies = studies[args.shard::args.num_shards]
    if args.limit:
        studies = studies[: args.limit]
    series = pd.read_csv(DATA_ROOT / f"{args.split}_series.csv")
    by_study = dict(tuple(series.groupby("StudyInstanceUID")))

    tag = f"{args.split}_s{args.shard}of{args.num_shards}"
    out_dir = WORK / "cache" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, by_study[s], split_dir, out_dir) for s in studies if s in by_study]
    print(f"{tag}: {len(jobs)} studies, {args.workers} workers, out={out_dir}", flush=True)

    t0, rows = time.time(), []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_worker, jobs), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {el:.0f}s elapsed  {el / i:.2f}s/study wall", flush=True)
    meta = pd.DataFrame(rows)
    meta.to_csv(out_dir / "meta.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s. Plane coverage:", meta.filter(like="has_").mean().round(3).to_dict())

    if args.zip:
        args.zip_dir.mkdir(parents=True, exist_ok=True)
        zpath = args.zip_dir / f"cache_{tag}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
            for f in out_dir.iterdir():
                zf.write(f, f"{tag}/{f.name}")
        for f in out_dir.iterdir():
            f.unlink()
        out_dir.rmdir()
        print(f"zipped to {zpath} ({zpath.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
