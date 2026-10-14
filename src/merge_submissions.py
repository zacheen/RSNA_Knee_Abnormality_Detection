"""Merge infer.py submissions from one run per cache layout, weighting each file by its model count.

Each run averages its own checkpoints equally, so weighting by model count gives every checkpoint equal weight overall.
--weights replaces the model counts with fixed shares, for example to give an older family a small vote.

Usage
  python src/merge_submissions.py --out submission.csv sub_v3.csv sub_v4.csv [--weights 0.2 0.8]
"""
import argparse
from pathlib import Path

import pandas as pd

from config import LABELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", type=Path, help="submission csvs, each with a <csv>.models file beside it")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", type=float, nargs="+", help="one per part, instead of the model counts")
    args = ap.parse_args()
    subs = [pd.read_csv(p).set_index("StudyInstanceUID") for p in args.parts]
    weights = args.weights or [int(Path(f"{p}.models").read_text()) for p in args.parts]
    if len(weights) != len(subs):
        raise SystemExit(f"{len(weights)} weights for {len(subs)} parts")
    if any(not s.index.equals(subs[0].index) for s in subs):
        raise SystemExit("the parts list different studies or a different order")
    merged = sum(s[LABELS] * w for s, w in zip(subs, weights)) / sum(weights)
    merged.reset_index().to_csv(args.out, index=False, lineterminator="\n")
    print(f"merged {len(subs)} parts with weights {weights} into {args.out}")


if __name__ == "__main__":
    main()
