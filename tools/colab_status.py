"""Print the cell outputs Colab autosaved into the runner notebook on Drive, so a run can be followed without the browser.

Colab writes outputs back every few minutes, so the newest lines may lag the live session.

Usage
  python tools/colab_status.py ["H:/我的雲端硬碟/rsna_knee/runner.ipynb"] [--tail 8]
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("notebook", nargs="?", default="H:/我的雲端硬碟/rsna_knee/runner.ipynb")
ap.add_argument("--tail", type=int, default=8)
args = ap.parse_args()
nb = json.load(open(args.notebook, encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    for out in cell.get("outputs", []):
        text = "".join(out.get("text", [])) or "".join(out.get("data", {}).get("text/plain", []))
        if out.get("output_type") == "error":
            text = f"{out.get('ename')}: {out.get('evalue')}"
        lines = [line for line in text.splitlines() if line.strip()]
        print(f"--- cell {i} ({len(lines)} lines)")
        print("\n".join(lines[-args.tail:]))
