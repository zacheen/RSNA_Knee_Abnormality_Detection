# Task B. Label knee MRI reports

Work only inside the current folder.

## Inputs

- `reports.csv` has 4407 radiology reports with columns `StudyInstanceUID` and `Report`. They are written in several languages, including Spanish, Greek, Bulgarian and English.
- `labels_visible.csv` has radiologist ground truth for 29 of those studies, one binary column per finding.

## Findings

Twelve binary findings are graded for the imaged knee. Borderline findings count as absent.

| column | positive when |
|---|---|
| ACL | high-grade partial or complete anterior cruciate ligament tear |
| MCL | high-grade partial or complete acute medial collateral ligament tear |
| Medial Meniscus | medial meniscus tear reaching the surface, or a truncated or displaced fragment |
| Lateral Meniscus | the same for the lateral meniscus |
| Medial OA | high-grade cartilage loss over about 1 cm or more in the medial tibiofemoral compartment |
| Lateral OA | the same in the lateral tibiofemoral compartment |
| PF OA | the same in the patellofemoral compartment |
| Effusion | moderate or large joint effusion |
| Synovitis | synovial thickening or inflammation |
| Baker's | moderate or large popliteal cyst |
| Contusion | bone bruise without a fracture line |
| Fracture | acute fracture line or cortical break |

The ground truth was graded by radiologists looking at the images, not at the report text, so the two can disagree.

## Deliverable

Write `predictions.csv` with one row per study in `reports.csv`. The columns are `StudyInstanceUID` followed by the twelve finding names exactly as above. Each value is a probability between 0 and 1.

## Scoring

Macro-averaged ROC AUC over the twelve findings, measured on a hidden set of other radiologist-labeled studies. Only the ranking within each column matters.

## Rules

- Run Python only as `conda run -n rsna_knee python`. Never install or upgrade packages.
- Never read or write anything outside this folder.
- You may call any LLM you can reach, including the `claude` CLI.
- Keep a short `NOTES.md` saying what you tried and what each attempt scored on `labels_visible.csv`.
