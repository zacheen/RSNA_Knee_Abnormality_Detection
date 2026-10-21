# Session handoff for the RSNA Knee Abnormality Detection project

This file is meant to be given to a fresh Claude session so that it can answer questions about this project without the original conversation. It was written on 2026-10-21, one day before the competition deadline of 2026-10-22. Treat every number here as measured on that date. When a question needs more detail than this file holds, the authoritative record is the experiment log table in section 8 of `RUNBOOK.md`, which has one row per experiment in Traditional Chinese.

## 1. What the project is

- Kaggle competition "RSNA Knee Abnormality Detection" (2026). Each training or test case is one knee MRI study with 3 to 14 DICOM series (mean 5.5) across sagittal, coronal and axial planes.
- The task is to predict 12 binary findings per study. They are ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's cyst, Contusion and Fracture.
- The metric is macro AUC over the 12 findings.
- It is an offline code competition. Inference runs inside a Kaggle notebook with no internet and a 9 hour limit on about 1,300 hidden test studies. The public leaderboard uses roughly 400 of them and reports three decimals.
- Of 4,407 training studies only 58 carry radiologist labels for all 12 findings (called "gold" or "GT 58" below). The other 4,349 have only a free-text radiology report in several languages, about half of the Latin-script ones English, plus 321 Greek and 220 Cyrillic.
- Raw data is about 570 GB, too large for the local machine.
- The owner also wanted to measure how far an AI agent gets. Most of the coding, experiment scheduling and analysis in this project was done by Claude Code, with the human giving direction and starting Colab runtimes. A layered agent-versus-human comparison was designed in `RUNBOOK.md` section 5 but never run.

## 2. Final result

- Best public LB is **0.944**, reached by several different ensembles.
- The two final submissions chosen on 2026-10-21 are both 0.944 public.

| Git tag | Kaggle kernel and version | Submission ref | Contents |
|---|---|---|---|
| `final-submit-16-members` | rsna-knee-submit-probe v9 | 57303091 | 16 WindowNet members averaged equally |
| `final-submit-v45` | rsna-knee-submit-v45 v1 | 57330296 | The 10-member ensemble at 0.85 plus V5A on cache v5 at 0.15 |

- The 10-member ensemble (submission 57288977) is W8, W9, P1, W11, W12 (efficientnet family) plus W22, W23, W25, W28 (tf_efficientnetv2_b0) and W27 (convnext_nano).
- The 16-member ensemble adds W5, W14, W26 (b0 seeds), W30 (effv2 on OOF labels) and W33, W34 (convnext_nano seeds).
- The half-share ensemble with full-frame effv2 (submission 57337022) also scored 0.944 but was not chosen, because W39 alone scored only 0.931 and a 50 percent weight on it looked risky for the private split.
- No external data and no third-party trained weights were used. Backbones start from ImageNet weights through timm.

## 3. Pipeline and where the code lives

The repo is the single source of truth. Kaggle and Colab receive copies of `src/`.

| Path | Role |
|---|---|
| `src/report_labels.py` | Report text to 12 soft labels with an LLM, -1 for findings not mentioned |
| `src/build_cache.py`, `src/dicom_io.py` | DICOM to a compact uint8 cache per study, used for both training and inference |
| `src/config.py` | Paths and cache settings, read from env vars RSNA_SLOTS, RSNA_N_SLICES, RSNA_IMG_SIZE, RSNA_CROP_MM, RSNA_BAND |
| `src/model.py` | KneeNet (old 2.5D model) and WindowNet (current model), plus checkpoint loading |
| `src/train.py` | Training, gold-58 validation, fold support, resume, averaging of the last 3 epochs |
| `src/infer.py`, `src/merge_submissions.py` | Inference with mirror TTA, and weighted merging of two submission files |
| `tools/` | Kaggle push scripts, cache download, probe submission gate, ensemble evaluation, Colab watcher |
| `kaggle_kernels/` | One folder per Kaggle kernel (cache builds, training runs, submission variants) |
| `colab/runner.ipynb` | Colab runner that reads a jobs file from Google Drive |
| `RUNBOOK.md` | Setup, commands, measured facts, and the full experiment log in section 8 |
| `SUMMARY_zh.md`, `COURSE_REPORT_zh.md`, `report/RSNA_Knee_Report.pptx` | Chinese summary, course report notes and an 11-slide deck |

`data/` and `notes/` are local only and not in git. `data/rounds/r8/` holds the round 8 to 10 research notes and the Codex and Gemini replies. `notes/claude_memory/` holds the agent's project memory.

## 4. The current model, WindowNet on cache v4

- Cache v4 keeps 5 series slots per study (sagittal, coronal and axial fat-suppressed, plus sagittal and coronal non-fat-suppressed), 20 slices each over the 2 to 98 percent band of the stack, cropped to 128 mm of physical extent using DICOM pixel spacing and resized to 320 px. It is about 44 GB for the training set.
- Every 3 adjacent slices become one RGB window. A 2D ImageNet backbone encodes every window. Training takes a random 288 px crop and 32 random windows per step, evaluation a centre crop and all windows.
- Each finding has its own attention query over all windows of all slots, with slot and depth embeddings.
- The standard recipe is batch 2, head lr 1e-3, backbone lr 2e-4, weight decay 0.02, dropout 0, 12 epochs, predictions averaged over the last 3 epoch checkpoints, mirror TTA at inference.
- Labels for most members are the blended soft labels (`report_labels_blend.jsonl`).

## 5. Labels

Agreement with radiologists (macro AUC). The 58 gold studies were split into 29 used for prompt tuning and 29 held out.

| Version | What changed | Tuning 29 | Held-out 29 |
|---|---|---|---|
| Keyword rules | No LLM | | 0.6496 |
| v1 | Labels 1, 0, -1 | 0.7918 | 0.7471 |
| v2 | Probabilities graded by described severity | 0.8767 | 0.8065 |
| v3 | Extra rules for 4 findings | 0.8827 | 0.8285 |
| v2s | v2 with Synovitis from v3 | 0.8870 | 0.8262 |

On all 58, v2s scores 0.8655 and the blend of v2s with 7 public CC0 label sets scores 0.8923. Report text and image reading disagree by nature. Prompt v1 had recall of only 0.44 to 0.60 for MCL, Synovitis and Effusion against the radiologists.

## 6. Leaderboard history

| Stage | Public LB |
|---|---|
| Old KneeNet, 6 attention members on cache v1 | 0.882 |
| Plus depth-conv members | 0.888 |
| Ten KneeNet members including resnet34 and a 5-slot member | 0.893 |
| WindowNet, 3 members (convnext_tiny, resnet34, convnext_tiny seed 1) | 0.939 |
| WindowNet, 5 members adding efficientnet_b0 | 0.942 |
| Efficientnet only, 5 members | 0.943 |
| Plus 4 effv2 and 1 convnext_nano, 10 members | 0.944 |

Single-model probes on the public LB were W8 (b0) 0.938, P1 (b0 on in-sample pseudo labels) 0.939, P3 (b0 on 5-fold OOF pseudo labels) 0.937, W2 (resnet34) 0.920, W39 (full-frame effv2) 0.931, W46 (b0 with depth-conv) 0.935, W47 (b0 with attention dropout 0.2) 0.934.

## 7. What did not beat 0.944

| Idea | Result |
|---|---|
| 16 members instead of 10 | 0.944 |
| Mix in V5A (3 FS slots of 32 slices) at 0.15 | 0.944 |
| Mix in full-frame effv2 W39 and W41 at 0.5 | 0.944 |
| Mix in the 160 mm crop group W43 and W44 at 0.2 | 0.943 |
| Mix in v2s-label members W4 and W16 at 0.2 | 0.943 |
| Mix in the old KneeNet family at 0.2 | 0.942 |
| OOF pseudo labels, hard labels, depth-conv, attention dropout | All worse as single models, see section 6 and the gold numbers in `RUNBOOK.md` |
| efficientnet_b1 | Diverged to NaN at epoch 4 with lr 1e-3 |
| efficientnet_b2, regnety_016, resize to 384, 18 epochs, 48 train windows, gradient accumulation to batch 16 | No gain on gold |

## 8. Contradictions worth studying

1. Gold-58 ranking does not transfer to the LB. W39 was the best single model on gold at 0.9301 and second worst on the LB at 0.931. The same recipe over 4 seeds gave 0.9301, 0.9026, 0.9062 and 0.9120, so seed noise on gold is about 0.02 to 0.04.
2. Better labels, unmeasurable model effect. Label quality rose from 0.8655 to 0.8923, but models trained on each differ by only 0.004 to 0.006 on gold, in opposite directions on the two architectures, and no single-model LB comparison was made. The correct reading is that no effect could be measured, not that there is none.
3. In-sample pseudo labels (P1 0.939) did not lose to OOF pseudo labels (P3 0.937), against the usual expectation.
4. Depth-conv helped the old KneeNet (gold about 0.841 to 0.856) but hurt WindowNet on the LB (0.935 against 0.938).
5. Third-party ablations did not reproduce. Dropping non-FS series and widening the crop both made our models worse.

## 9. Operational facts and lessons

- Kaggle GPU quota is 45 hours per week, at most 2 GPU sessions at once and 5 submissions per UTC day. Submitting an existing kernel version costs no quota, but pushing a new version runs a GPU commit run that does. There is no API to cancel a kernel or to select final submissions.
- CPU kernels build caches without GPU quota, about 40 to 100 minutes per 4-shard cache.
- Only one Colab runtime at a time worked reliably on this account.
- On Windows, stopping a background bash task from Claude Code did not kill the script itself. Three such loops survived and one pushed a duplicate Kaggle training run, so kill by process id and confirm.
- The gold 58 set is too small for decisions. Most decisions in the last week were made by single-question LB probes.

## 10. What was not tried and why

| Direction | Why not |
|---|---|
| CoAtNet-rmlp-2-rw-384 backbone (a third party reports 0.949 single) | 3 to 5 times slower than b0 on a T4, and that 0.949 partly used OAI external data |
| External data such as OAI | Legality unclear and access gated |
| 5 slots of 32 slices | Needs a new cache and retraining, no time to validate before the deadline |
| A second round of LLM label refinement | Needs re-extraction of 4,400 reports and retraining |

## 11. How to answer follow-up questions

- For any experiment ID such as W8, P3 or V5A, search for it in `RUNBOOK.md` section 8. Each row gives the setting, the gold score, the LB score when probed, and where the checkpoints went.
- For the reasoning behind round 8 to 10 choices, read `data/rounds/r8/research_web.md`, `codex2.md` and `agy.md` on the owner's machine.
- For code behaviour, read `src/model.py` (class WindowNet) and `src/train.py`.
