"""Extract one value per finding from the free-text reports. -1 always means the report does not address it.

Prompt v1 returns 1 present or 0 stated absent. Prompt v2 returns a probability in [0, 1] that a radiologist
grading the images would mark the finding present, so "mild" and "moderate" stay apart as ranking information.
-1 matters because a report that never mentions a structure is not evidence that it is normal.
train.py masks -1 out of the loss instead of treating it as a negative, and uses v2 values as soft targets.

Backends
  claude_cli  `claude -p` on this machine, billed to the Claude Code subscription, no API key needed
  anthropic   ANTHROPIC_API_KEY
  openai      OPENAI_API_KEY
  keyword     offline English-only regex, only for smoke tests. Non-English reports come out all -1.

Results are appended to a jsonl keyed by StudyInstanceUID, so an interrupted run resumes where it stopped.

Usage
  python src/report_labels.py --backend claude_cli --only-gt --limit 10
  python src/report_labels.py --backend claude_cli --workers 4
  python src/report_labels.py --score
  python src/report_labels.py --prompt v2 --out data/work/report_labels_v2.jsonl --only-gt
  python src/report_labels.py --score --out data/work/report_labels_v2.jsonl --subset visible
"""
import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_ROOT, LABELS, REPO, WORK  # noqa: E402

OUT = WORK / "report_labels.jsonl"
# One file per prompt, because resuming skips every study already in the file, so mixing prompts would go unnoticed.
DEFAULT_OUT = {"v1": OUT, "v2": WORK / "report_labels_v2.jsonl", "v3": WORK / "report_labels_v3.jsonl"}
# Written by agents/prepare_B.py. Prompt tuning must never look at these, or the B-layer comparison is void.
HIDDEN_GT = REPO / "data" / "agent_hidden" / "B_hidden_gt.csv"

CRITERIA_V1 = """Grade each finding for the imaged knee. Borderline or equivocal findings count as absent (0).
- ACL: high-grade partial (>50% fibers) or complete tear. Sprain, mucoid degeneration or thickening alone is 0.
- MCL: high-grade partial or complete acute tear. Low-grade sprain or old/chronic change is 0.
- Medial Meniscus / Lateral Meniscus: tear reaching the articular surface, or a truncated, displaced or absent fragment. Intrasubstance degeneration is 0.
- Medial OA / Lateral OA / PF OA: high-grade (>50% thickness, grade 3-4) cartilage loss over a moderate or large area (about 1 cm or more) in that compartment (medial tibiofemoral, lateral tibiofemoral, patellofemoral).
- Effusion: moderate or large joint effusion.
- Synovitis: synovial thickening or inflammation.
- Baker's: moderate or large popliteal (Baker) cyst.
- Contusion: bone bruise or marrow edema from impact, without a fracture line.
- Fracture: acute fracture line or cortical break.
Use 1 when the report states the finding, 0 when the report states it is absent or normal, -1 when the report does not address it.
Reports may be in any language."""

CRITERIA_V2 = """For each finding, give the probability (0.0 to 1.0) that a musculoskeletal radiologist grading the MR images of this knee would mark it present, using the report as evidence.
Findings
- ACL: anterior cruciate ligament tear.
- MCL: medial collateral ligament tear or significant acute injury.
- Medial Meniscus / Lateral Meniscus: meniscal tear of that side, including truncated, displaced or absent fragments.
- Medial OA / Lateral OA / PF OA: cartilage loss in that compartment (medial tibiofemoral, lateral tibiofemoral, patellofemoral).
- Effusion: joint effusion.
- Synovitis: synovial thickening or inflammation.
- Baker's: popliteal (Baker) cyst.
- Contusion: bone bruise or marrow edema from impact, without a fracture line.
- Fracture: acute fracture line or cortical break.
Calibration. Graded wording carries information, so use the whole range.
- Stated absent or normal is about 0.03.
- Effusion. Physiological or trace about 0.15, small or mild about 0.6, moderate or large about 0.95.
- Ligaments. Intact but thickened or degenerated about 0.1, low-grade sprain about 0.3, partial tear about 0.6, high-grade or complete tear about 0.95.
- Menisci. Intrasubstance degeneration not reaching the surface about 0.1, possible or equivocal tear about 0.5, definite tear about 0.95.
- Cartilage. Grade 1 to 2 chondromalacia about 0.2, grade 3 or partial-thickness focal loss about 0.6, grade 4, full-thickness or diffuse loss about 0.9.
- Cysts. Tiny about 0.3, small about 0.6, moderate or large about 0.95.
Use -1 only when the report gives no information about that structure at all.
Reports may be in any language."""

# v3 = v2 plus extra rules for the four worst-scoring labels, tuned on the visible half of the ground truth only.
# Synovitis is rarely named in reports, so related signs get a probability instead of -1.
_V2_LAST_RULE = "Use -1 only when the report gives no information about that structure at all."
# The extra rules go before v2's closing -1 rule so it stays last. If that sentence is edited, replace() would match
# nothing and v3 would silently be a copy of v2 under a v3 name, hence this check.
assert _V2_LAST_RULE in CRITERIA_V2, "CRITERIA_V3 is built by inserting before this sentence of CRITERIA_V2"
CRITERIA_V3 = CRITERIA_V2.replace(
    _V2_LAST_RULE,
    """Extra rules for four findings.
- Synovitis. Named synovitis, synovial thickening or hypertrophy about 0.9. Hoffa fat pad edema, or a moderate or large effusion with debris or a complex appearance, about 0.5. A moderate or large simple effusion with no synovial comment about 0.35. A small effusion with no synovial comment about 0.2. Use -1 only when neither the synovium nor the effusion is described.
- Lateral Meniscus. Postoperative meniscus without a described new tear about 0.4. Meniscal extrusion of that side without a described tear about 0.4.
- MCL. Periligamentous or adjacent soft-tissue edema without a described tear about 0.4. Chronic thickening without edema about 0.15.
- PF OA. Chondromalacia patellae or trochlear cartilage loss follows the cartilage scale above. Patellofemoral osteophytes or a named patellofemoral arthrosis without a cartilage grade about 0.6.
Use -1 only when the report gives no information about that structure at all.""")

_TAIL = ("\nReturn ONLY a JSON object mapping each report id to an object with exactly these keys: "
         + json.dumps(LABELS) + ". {values} No prose, no code fences.\n\n")
SOFT = {"v1": False, "v2": True, "v3": True}
PROMPTS = {
    "v1": "You label knee MRI radiology reports.\n" + CRITERIA_V1 + _TAIL.format(values="Values are 1, 0 or -1."),
    "v2": "You label knee MRI radiology reports.\n" + CRITERIA_V2
          + _TAIL.format(values="Values are a probability between 0 and 1, or -1."),
    "v3": "You label knee MRI radiology reports.\n" + CRITERIA_V3
          + _TAIL.format(values="Values are a probability between 0 and 1, or -1."),
}


def _build_prompt(batch: pd.DataFrame, head: str) -> str:
    body = "".join(f"### id={i}\n{r}\n\n" for i, r in zip(batch.index, batch.Report))
    return head + body


def _value(v, soft: bool) -> float:
    """A probability in [0, 1] when soft, else only 0.0 or 1.0. Anything unusable becomes -1.0."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return -1.0
    if not soft:
        return v if v in (0.0, 1.0) else -1.0
    return round(v, 3) if 0.0 <= v <= 1.0 else -1.0


def _parse(text: str, ids, soft: bool) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    raw = json.loads(m.group(0))
    # A dropped id would otherwise be written as all -1 and never retried, indistinguishable from a silent report.
    missing = [i for i in ids if str(i) not in raw]
    if missing:
        raise ValueError(f"model output skipped ids {missing}")
    return {i: {k: _value(raw[str(i)].get(k), soft) for k in LABELS} for i in ids}


def call_claude_cli(prompt: str, model: str) -> str:
    # --tools "" makes it a pure text call. --no-session-persistence keeps 400+ batch calls out of the session list.
    cmd = ["claude", "-p", "--model", model, "--tools", "", "--no-session-persistence", "--output-format", "json"]
    res = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=600)
    if res.returncode != 0:
        # Usage-limit refusals arrive on stdout as JSON with an empty stderr, so both are needed to see why.
        raise RuntimeError(f"claude exited {res.returncode}: stderr={res.stderr[:200]!r} stdout={res.stdout[:300]!r}")
    return json.loads(res.stdout)["result"]


def call_anthropic(prompt: str, model: str) -> str:
    import anthropic
    msg = anthropic.Anthropic().messages.create(
        model=model, max_tokens=4096, messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text


def call_openai(prompt: str, model: str) -> str:
    import openai
    r = openai.OpenAI().chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content


_KW = {
    "ACL": r"anterior cruciate|\bACL\b", "MCL": r"medial collateral|\bMCL\b",
    "Medial Meniscus": r"medial menisc", "Lateral Meniscus": r"lateral menisc",
    "Medial OA": r"medial (?:compartment|femorotibial)", "Lateral OA": r"lateral (?:compartment|femorotibial)",
    "PF OA": r"patellofemoral|trochle|patellar cartilage", "Effusion": r"effusion",
    "Synovitis": r"synovitis|synovial thicken", "Baker's": r"baker|popliteal cyst",
    "Contusion": r"contusion|bone bruise|marrow edema|bone marrow oedema", "Fracture": r"fracture",
}
_NEG = re.compile(r"\b(no|without|intact|normal|unremarkable|negative for)\b", re.I)


def keyword_labels(report: str) -> dict:
    """Sentence-level keyword match, deliberately crude."""
    out = {k: -1 for k in LABELS}
    for sent in re.split(r"(?<=[.\n])", report):
        for lab, pat in _KW.items():
            if re.search(pat, sent, re.I):
                state = 0 if _NEG.search(sent) else 1
                out[lab] = max(out[lab], state)
    return out


def _done_ids(out: Path) -> set:
    if not out.exists():
        return set()
    return {json.loads(line)["StudyInstanceUID"] for line in out.read_text(encoding="utf-8").splitlines() if line}


def run(backend: str, model: str, batch_size: int, workers: int, only_gt: bool, limit: int, prompt: str, out: Path):
    df = pd.read_csv(DATA_ROOT / "train.csv")
    if only_gt:
        df = df[df[LABELS].notna().all(axis=1)]
    done = _done_ids(out)
    todo = df[~df.StudyInstanceUID.isin(done)]
    if limit:
        todo = todo.head(limit)
    shown = "n/a" if backend == "keyword" else prompt
    print(f"{len(done)} already labeled, {len(todo)} to go, backend={backend} prompt={shown} out={out}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    if backend == "keyword":
        with out.open("a", encoding="utf-8") as f:
            for uid, rep in zip(todo.StudyInstanceUID, todo.Report):
                f.write(json.dumps({"StudyInstanceUID": uid, **keyword_labels(rep)}) + "\n")
        return

    call = {"claude_cli": call_claude_cli, "anthropic": call_anthropic, "openai": call_openai}[backend]
    batches = [todo.iloc[i:i + batch_size] for i in range(0, len(todo), batch_size)]

    def job(b):
        return b, _parse(call(_build_prompt(b, PROMPTS[prompt]), model), b.index, SOFT[prompt])

    with ThreadPoolExecutor(workers) as ex, out.open("a", encoding="utf-8") as f:
        futs = [ex.submit(job, b) for b in batches]
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                b, res = fut.result()
            except Exception as e:  # a failed batch is retried on the next run because nothing was written
                print(f"[batch failed] {type(e).__name__}: {e}", flush=True)
                continue
            for i, uid in zip(b.index, b.StudyInstanceUID):
                f.write(json.dumps({"StudyInstanceUID": uid, **res[i]}) + "\n")
            f.flush()
            print(f"  batch {n}/{len(batches)} written", flush=True)

    # Failed batches are only logged above, so check coverage here rather than let training see fewer labels unnoticed.
    left = len(set(df.StudyInstanceUID) - _done_ids(out))
    if left:
        raise SystemExit(f"{left} studies still unlabeled. Rerun the same command to retry them.")
    print(f"all {len(df)} studies labeled")


def load_labels(path: Path = OUT) -> pd.DataFrame:
    """One row per study with the 12 label columns. A study labeled twice keeps its latest line."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows).drop_duplicates("StudyInstanceUID", keep="last").set_index("StudyInstanceUID")[LABELS]


def to_scores(labels: pd.DataFrame) -> pd.DataFrame:
    """Rankable scores for AUC. "Not addressed" (-1) sits at 0.5, between absent and present."""
    labels = labels.astype(float)
    return labels.where(labels >= 0, 0.5)


def score(path: Path, subset: str):
    """Compare report-derived labels with the radiologist ground truth on the studies that have both.

    Tune prompts with subset="visible". "hidden" exists only to report the final B-layer number.
    """
    from sklearn.metrics import roc_auc_score

    gt = pd.read_csv(DATA_ROOT / "train.csv").set_index("StudyInstanceUID")
    gt = gt[gt[LABELS].notna().all(axis=1)][LABELS].astype(int)
    if subset != "all":
        if not HIDDEN_GT.exists():
            raise SystemExit(f"{HIDDEN_GT} is missing. Run agents/prepare_B.py once to create the split.")
        hidden = set(pd.read_csv(HIDDEN_GT).StudyInstanceUID)
        gt = gt[gt.index.isin(hidden) == (subset == "hidden")]
    raw = load_labels(path).reindex(gt.index).dropna(how="all")
    pred, ranked_all = raw.astype(float), to_scores(raw)
    gt = gt.loc[pred.index]
    print(f"{len(pred)} ground-truth studies ({subset}) have report labels from {path}")
    rows = []
    for lab in LABELS:
        p, g = pred[lab].astype(float), gt[lab]
        stated = p != -1
        hard = (p >= 0.5).astype(int)  # -1 reads as absent, the naive interpretation
        auc = roc_auc_score(g, ranked_all[lab]) if 0 < g.sum() < len(g) else float("nan")
        tp, fp, fn = int(((hard == 1) & (g == 1)).sum()), int(((hard == 1) & (g == 0)).sum()), int(((hard == 0) & (g == 1)).sum())
        rows.append({"label": lab, "gt_pos": int(g.sum()), "addressed": round(stated.mean(), 2),
                     "acc_when_addressed": round(float((hard[stated] == g[stated]).mean()) if stated.any() else float("nan"), 2),
                     "precision": round(tp / max(tp + fp, 1), 2), "recall": round(tp / max(tp + fn, 1), 2),
                     "auc": round(auc, 3)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print(f"macro AUC = {table.auc.mean():.4f}")


def merge(base: Path, override: Path, labels: list, out: Path):
    """Write base with the given label columns taken from override.

    Studies missing from override, and override values of -1 (not addressed), keep the base value.
    """
    # Cast to float: a v1 file loads as int64, and pandas refuses floats in an int column.
    b, o = load_labels(base).astype(float), load_labels(override).astype(float)
    common = b.index.intersection(o.index)
    patch = o.loc[common, labels]
    b.loc[common, labels] = patch.where(patch >= 0, b.loc[common, labels])
    dropped = len(o.index.difference(b.index))
    if dropped:
        print(f"[warn] {dropped} studies exist only in {override} and were dropped")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for uid, row in b.iterrows():
            f.write(json.dumps({"StudyInstanceUID": uid, **{k: float(row[k]) for k in LABELS}}) + "\n")
    print(f"{out}: {len(b)} studies, {labels} taken from {override} for {len(common)} of them")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["claude_cli", "anthropic", "openai", "keyword"], default="claude_cli")
    ap.add_argument("--model", default=os.environ.get("RSNA_LABEL_MODEL", "sonnet"))
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only-gt", action="store_true", help="only the 58 studies with radiologist labels")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="v1")
    ap.add_argument("--out", type=Path, default=None, help="defaults to the file for --prompt")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--subset", choices=["all", "visible", "hidden"], default="all")
    ap.add_argument("--merge", nargs=2, type=Path, metavar=("BASE", "OVERRIDE"),
                    help="write BASE with --merge-labels taken from OVERRIDE to --out")
    ap.add_argument("--merge-labels", nargs="+", choices=LABELS)
    a = ap.parse_args()
    if a.merge:
        if not (a.out and a.merge_labels):
            ap.error("--merge needs --merge-labels and an explicit --out")
        merge(*a.merge, a.merge_labels, a.out)
        raise SystemExit
    a.out = a.out or DEFAULT_OUT[a.prompt]
    if a.score:
        score(a.out, a.subset)
    else:
        run(a.backend, a.model, a.batch_size, a.workers, a.only_gt, a.limit, a.prompt, a.out)
