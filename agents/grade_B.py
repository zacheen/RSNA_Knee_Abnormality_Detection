"""Score a task-B predictions.csv, or the human pipeline's report_labels.jsonl, on the hidden half of the ground truth.

Usage
  python agents/grade_B.py data/agent_ws/B_claude/predictions.csv
  python agents/grade_B.py data/work/report_labels.jsonl
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import LABELS  # noqa: E402
from report_labels import HIDDEN_GT as HIDDEN, load_labels, to_scores  # noqa: E402


def load_pred(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return to_scores(load_labels(path))
    return pd.read_csv(path).set_index("StudyInstanceUID")[LABELS].astype(float)


def main():
    gt = pd.read_csv(HIDDEN).set_index("StudyInstanceUID")
    pred = load_pred(Path(sys.argv[1])).reindex(gt.index)
    missing = int(pred.isna().any(axis=1).sum())
    pred = pred.fillna(0.5)
    aucs = {lab: roc_auc_score(gt[lab], pred[lab]) for lab in LABELS if 0 < gt[lab].sum() < len(gt)}
    print(f"hidden studies={len(gt)} missing_predictions={missing}")
    print({k: round(v, 3) for k, v in aucs.items()})
    print(f"macro AUC = {np.mean(list(aucs.values())):.4f} over {len(aucs)} labels")


if __name__ == "__main__":
    main()
