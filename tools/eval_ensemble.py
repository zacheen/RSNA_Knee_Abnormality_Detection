"""Score every equal-weight combination of trained model groups on the 58 radiologist-labeled studies.

A group is a set of checkpoints plus the cache they were trained on, so groups with different cache settings can
be combined. Checkpoints are averaged within a group first, each with mirror TTA.

Usage
  python tools/eval_ensemble.py r2b="data/models/r2b/*.pt"@"data/work/cache/train_s0of1" r3="data/models/r3/*.pt"@"data/work/cache/train_s*of2"
  --partial scores only the ground-truth studies the caches hold (local smoke tests on a subset).
"""
import argparse
import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import ALL_SLOTS, DATA_ROOT, LABELS, is_slot_prefix  # noqa: E402
from model import build_from_checkpoint, load_checkpoint, predict_tta  # noqa: E402
from train import CacheDataset, macro_auc  # noqa: E402


def _index(cache_glob: str, gt: pd.DataFrame):
    paths, present, crop, layouts = {}, {}, set(), set()
    for d in sorted(glob.glob(cache_glob)):
        meta = pd.read_csv(Path(d) / "meta.csv")
        # Older caches carry no crop_fg column and were all built uncropped.
        crop.add(bool(meta.crop_fg.iloc[0]) if "crop_fg" in meta else False)
        # Every slot the cache holds, regardless of RSNA_SLOTS, so groups with different slot counts can be combined.
        slots = [s for s in ALL_SLOTS if f"has_{s}" in meta]
        if not is_slot_prefix(slots, ALL_SLOTS):
            raise SystemExit(f"{d} holds slots {slots}, not a prefix of {ALL_SLOTS}")
        layouts.add(tuple(slots))
        for r in meta[meta.StudyInstanceUID.isin(gt.index)].itertuples(index=False):
            paths[r.StudyInstanceUID] = Path(d) / f"{r.StudyInstanceUID}.npy"
            present[r.StudyInstanceUID] = np.array([getattr(r, f"has_{s}") for s in slots], bool)
    if len(crop) != 1 or len(layouts) != 1:
        raise SystemExit(f"{cache_glob} matches no cache, or caches with mixed crop_fg {crop} or slots {layouts}")
    return paths, present, crop.pop(), list(layouts.pop())


def group_probs(ckpt_glob: str, cache_glob: str, gt: pd.DataFrame, partial: bool, device):
    """Return (study ids, mean probabilities over the group's checkpoints)."""
    ckpts = sorted(glob.glob(ckpt_glob))
    if not ckpts:
        raise SystemExit(f"no checkpoints match {ckpt_glob}")
    paths, present, crop, slots = _index(cache_glob, gt)
    missing = set(gt.index) - set(paths)
    if missing and not partial:
        raise SystemExit(f"{cache_glob} lacks {len(missing)} of the ground-truth studies")
    uids = [u for u in gt.index if u in paths]
    if not uids:
        raise SystemExit(f"{cache_glob} holds none of the ground-truth studies")
    shape = np.load(paths[uids[0]], mmap_mode="r").shape
    dl = DataLoader(CacheDataset(uids, paths, present, gt, False), batch_size=2)
    out = []
    for c in ckpts:
        ck = load_checkpoint(c)
        # A mismatched cache would still run (global pooling accepts any size) and only score worse.
        want = (ck.get("n_slices"), ck.get("img_size"), ck["crop_fg"])
        if want != (shape[1], shape[2], crop) or not is_slot_prefix(ck["planes"], slots):
            raise SystemExit(f"{c} expects (n_slices, img_size, crop_fg)={want}, {cache_glob} holds "
                             f"{(shape[1], shape[2], crop)}, or slots {ck['planes']} that are not a prefix of {slots}")
        m = build_from_checkpoint(ck).to(device).eval()
        ps = []
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
            for vol, pres, _ in dl:
                ps.append(predict_tta(m, vol.to(device), pres.to(device)).float().cpu().numpy())
        out.append(np.concatenate(ps))
    return uids, np.mean(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", help='name="<checkpoint glob>"@"<cache dir glob>"')
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = pd.read_csv(DATA_ROOT / "train.csv").set_index("StudyInstanceUID")
    gt = tr[tr[LABELS].notna().all(axis=1)][LABELS].astype(np.float32)
    groups, ids = {}, set()
    for spec in args.groups:
        name, rest = spec.split("=", 1)
        ckpt_glob, cache_glob = rest.rsplit("@", 1)
        uids, groups[name] = group_probs(ckpt_glob, cache_glob, gt, args.partial, device)
        ids.add(tuple(uids))
    if len(ids) != 1:
        raise SystemExit("groups were scored on different studies, so they cannot be combined")
    uids = list(ids.pop())
    y = gt.loc[uids].values
    for name, p in groups.items():
        print(f"{name}: {macro_auc(y, p)[0]:.4f}", flush=True)
    rows = []
    for k in range(2, len(groups) + 1):
        for combo in itertools.combinations(groups, k):
            rows.append({"combo": " + ".join(combo), "macro_auc": round(macro_auc(y, np.mean([groups[g] for g in combo], 0))[0], 4)})
    if rows:
        print(pd.DataFrame(rows).sort_values("macro_auc", ascending=False).to_string(index=False))
    out = ROOT / "data" / "work"
    out.mkdir(parents=True, exist_ok=True)
    for g, p in groups.items():
        np.savez(out / f"val_probs_{g}.npz", uids=np.array(uids), probs=p, labels=np.array(LABELS))


if __name__ == "__main__":
    main()
