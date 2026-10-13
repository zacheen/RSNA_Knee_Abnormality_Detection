"""Write submission.csv for whatever test.csv is present. That is 3 example studies on commit and about 1300 on scoring.

Studies are decoded in a process pool while the GPU predicts, and a wall-clock guard fills any study not reached
before the budget with 0.5, so a slow rerun still produces a valid file instead of a Submission Error.

Usage
  python src/infer.py --ckpt data/work/models/kneenet_ep08.pt data/work/models/kneenet_ep09.pt [--budget-hours 8.5]
Several checkpoints are averaged, each with mirror TTA.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cache import load_study  # noqa: E402
from config import DATA_ROOT, LABELS, SETTINGS, WORK, is_slot_prefix  # noqa: E402
from model import build_from_checkpoint, load_checkpoint, predict_tta, to_input  # noqa: E402

_series = None


def _init(series_csv):
    global _series
    _series = dict(tuple(pd.read_csv(series_csv).groupby("StudyInstanceUID")))


def _load(uid):
    if uid not in _series:
        return uid, None, None
    vol, present = load_study(uid, _series[uid], DATA_ROOT / "test_series")
    return uid, vol, present


def load_model(path: Path, device) -> torch.nn.Module:
    ck = load_checkpoint(path)
    # A mismatch would not crash (global pooling accepts any size), only score worse silently.
    wrong = {k: (ck[k], v) for k, v in SETTINGS.items() if k != "planes" and k in ck and ck[k] != v}
    # Fewer slots is fine when they are a prefix, because both architectures drop the trailing slots of the volume.
    if not is_slot_prefix(ck["planes"], SETTINGS["planes"]):
        wrong["planes"] = (ck["planes"], SETTINGS["planes"])
    if wrong:
        raise SystemExit(f"{path} was trained with different settings (checkpoint, current): {wrong}")
    return build_from_checkpoint(ck).to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--budget-hours", type=float, default=8.5)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, default=WORK / "submission.csv")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(c, device) for c in args.ckpt]
    print(f"loaded {len(models)} checkpoint(s)", flush=True)

    uids = pd.read_csv(DATA_ROOT / "test.csv").StudyInstanceUID.tolist()
    preds, t_first = {}, None
    with Pool(args.workers, initializer=_init, initargs=(DATA_ROOT / "test_series.csv",)) as pool:
        for i, (uid, vol, present) in enumerate(pool.imap(_load, uids), 1):
            if time.time() - t0 > args.budget_hours * 3600:
                print(f"[budget] stopping at {i - 1}/{len(uids)}, rest filled with 0.5", flush=True)
                pool.terminate()
                break
            if vol is None:
                continue
            with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
                v = to_input(vol).unsqueeze(0).to(device)
                p = torch.from_numpy(present).unsqueeze(0).to(device)
                preds[uid] = torch.stack([predict_tta(m, v, p) for m in models]).mean(0).cpu().numpy()[0]
            if t_first is None:  # first study predicted, not study 1, which may be skipped
                t_first = time.time()
            if i % 100 == 0 or i == len(uids):
                # At 3 studies both figures are loose. The upper bound spreads model load and pool start-up over
                # 3 studies. The lower bound counts them once, but studies 2-3 decode while study 1 loads, so
                # their rate looks almost free. chain_submit.sh gates on the upper bound, which errs only toward
                # not submitting.
                el = time.time() - t0
                low = (t_first - t0) + (time.time() - t_first) / max(i - 1, 1) * 1299
                print(f"  {i}/{len(uids)} {el:.0f}s, projected 1300 studies {el / i * 1300 / 3600:.2f}h "
                      f"(lower bound {low / 3600:.2f}h)", flush=True)

    sub = pd.DataFrame({"StudyInstanceUID": uids})
    probs = np.stack([preds.get(u, np.full(len(LABELS), 0.5, np.float32)) for u in uids])
    sub[LABELS] = probs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False, lineterminator="\n")
    print(f"wrote {args.out} rows={len(sub)} predicted={len(preds)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
