"""Build an isolated workspace for task B (report -> labels) and keep half of the radiologist labels hidden for grading.

Usage
  python agents/prepare_B.py --ws data/agent_ws/B_<agent_name>
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_ROOT, LABELS  # noqa: E402
from report_labels import HIDDEN_GT as HIDDEN  # noqa: E402



def split_gt():
    tr = pd.read_csv(DATA_ROOT / "train.csv")
    gt = tr[tr[LABELS].notna().all(axis=1)][["StudyInstanceUID", *LABELS]].sort_values("StudyInstanceUID")
    # Fixed seed, so every agent and the human baseline are graded on the same 29 studies.
    hidden = gt.sample(frac=0.5, random_state=0)
    return tr, gt.drop(hidden.index), hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", type=Path, required=True)
    args = ap.parse_args()
    tr, visible, hidden = split_gt()
    HIDDEN.parent.mkdir(parents=True, exist_ok=True)
    hidden.to_csv(HIDDEN, index=False)

    ws = args.ws
    ws.mkdir(parents=True, exist_ok=True)
    tr[["StudyInstanceUID", "Report"]].to_csv(ws / "reports.csv", index=False)
    visible.astype({k: int for k in LABELS}).to_csv(ws / "labels_visible.csv", index=False)
    shutil.copy(ROOT / "agents" / "tasks" / "B_report_labels.md", ws / "TASK.md")
    print(f"workspace {ws}: reports={len(tr)} visible_gt={len(visible)} hidden_gt={len(hidden)} -> {HIDDEN}")


if __name__ == "__main__":
    main()
