"""Average several report-label tables into one soft-label jsonl that src/train.py reads, and score tables on the 58 gold studies.

Each source is a .jsonl in our format or a .csv with StudyInstanceUID and the 12 label columns, values in [0, 1].
Our "not addressed" (-1) counts as missing, so a cell is the mean of the sources that answered it. A cell no source
answered is written as -1, which train.py masks out.

Usage
  python tools/blend_labels.py --score data/work/report_labels_v2s.jsonl data/external_labels/steven_v4.csv
  python tools/blend_labels.py --out data/work/report_labels_blend.jsonl data/work/report_labels_v2s.jsonl data/external_labels/*.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_ROOT, LABELS  # noqa: E402
from train import macro_auc  # noqa: E402


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        df = pd.DataFrame([json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l])
    else:
        df = pd.read_csv(path)
    df = df.drop_duplicates("StudyInstanceUID", keep="last").set_index("StudyInstanceUID")[LABELS]
    df = df.apply(pd.to_numeric, errors="coerce").astype(float)
    return df.where((df >= 0) & (df <= 1))


def blend(paths: list) -> pd.DataFrame:
    tables = [load(p) for p in paths]
    ids = sorted(set().union(*(t.index for t in tables)))
    stack = np.stack([t.reindex(ids).values for t in tables])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
    return pd.DataFrame(mean, index=pd.Index(ids, name="StudyInstanceUID"), columns=LABELS)


def gold_auc(df: pd.DataFrame) -> float:
    tr = pd.read_csv(DATA_ROOT / "train.csv").set_index("StudyInstanceUID")
    gold = tr[tr[LABELS].notna().all(axis=1)][LABELS].astype(float)
    pred = df.reindex(gold.index).fillna(0.5)
    return macro_auc(gold.values, pred.values)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--score", action="store_true", help="score each source alone, then their blend")
    args = ap.parse_args()
    if args.score:
        for p in args.sources:
            print(f"{gold_auc(load(p)):.4f}  {p.name}")
    df = blend(args.sources)
    print(f"{gold_auc(df):.4f}  blend of {len(args.sources)}, {int(df.isna().sum().sum())} cells unanswered")
    if args.out:
        with args.out.open("w", encoding="utf-8", newline="\n") as f:
            for uid, row in df.iterrows():
                f.write(json.dumps({"StudyInstanceUID": uid, **{l: (-1 if np.isnan(v) else round(float(v), 4))
                                                                for l, v in row.items()}}) + "\n")
        print("wrote", args.out, len(df))


if __name__ == "__main__":
    main()
