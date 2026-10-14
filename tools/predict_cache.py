"""Predict every study of a cache with several checkpoints, each with mirror TTA, and write the mean as a csv.

Each batch is read once and passed through every model, so the cost of reading the cache is paid once rather than
once per checkpoint. The output has the label-table layout, so tools/blend_labels.py can mix it with report labels.
The models were trained on these same studies, so their predictions are in-sample and work as smoothed targets, not
as held-out estimates.

Usage
  RSNA_SLOTS=5 RSNA_N_SLICES=20 ... python tools/predict_cache.py --cache "data/work/cache_v4/train_*" \
      --ckpt data/models/W8/kneenet_ep12.pt data/models/W9/kneenet_ep12.pt --out data/work/pred_W.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import LABELS  # noqa: E402
from infer import mismatch  # noqa: E402
from model import build_from_checkpoint, load_checkpoint, predict_tta  # noqa: E402
from train import CacheDataset, index_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", required=True)
    ap.add_argument("--ckpt", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bs", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for c in args.ckpt:
        ck = load_checkpoint(c)
        if wrong := mismatch(ck):
            raise SystemExit(f"{c} was trained with different settings (checkpoint, current): {wrong}")
        models.append(build_from_checkpoint(ck).to(device).eval())
    paths, present = index_cache(args.cache)
    uids = sorted(paths)
    dummy = pd.DataFrame(0.0, index=uids, columns=LABELS)  # CacheDataset wants targets, which are unused here
    dl = DataLoader(CacheDataset(uids, paths, present, dummy, False), batch_size=args.bs)
    out, t0 = [], time.time()
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for i, (vol, pres, _) in enumerate(dl, 1):
            vol, pres = vol.to(device), pres.to(device)
            out.append(torch.stack([predict_tta(m, vol, pres) for m in models]).mean(0).float().cpu().numpy())
            if i % 200 == 0:
                done = i * args.bs
                print(f"{done}/{len(uids)} {time.time() - t0:.0f}s, about {(time.time() - t0) / done * (len(uids) - done) / 60:.0f} min left", flush=True)
    df = pd.DataFrame(np.concatenate(out), index=pd.Index(uids, name="StudyInstanceUID"), columns=LABELS)
    df.reset_index().to_csv(args.out, index=False, lineterminator="\n")
    print(f"wrote {args.out} rows={len(df)} with {len(models)} models in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
