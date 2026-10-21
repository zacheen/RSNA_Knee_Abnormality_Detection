# RSNA Knee Abnormality Detection, session summary

Written on 2026-10-21. This is the human-readable account of the whole working session, from the first baseline on 2026-10-09 to the final submissions on 2026-10-21. For a version written to be loaded into a new Claude session, see `SESSION_HANDOFF.md` in the same folder.

## The short version

- We built a knee MRI model that scores **0.944** macro AUC on the public leaderboard, using no external data and no third-party trained weights.
- Only 58 of 4,407 training studies had radiologist labels, so the training labels came from an LLM reading the radiology reports.
- The largest single gain came from redesigning how the MRI slices are presented to the model, which moved the score from 0.893 to 0.939. Every later change together added 0.005.
- The small gold set of 58 studies repeatedly ranked models in the opposite order from the leaderboard, so late decisions were made by submitting one question at a time to the leaderboard.
- The final two submissions are tagged in git as `final-submit-16-members` and `final-submit-v45`.

## Phase 1, building the pipeline (2026-10-09 to 10-11)

- The raw data is about 570 GB, so the work was split across three places. Report labelling and coding ran locally, DICOM-to-cache conversion ran on Kaggle CPU kernels where the data already lives, and training ran on Kaggle GPU, Colab and a local RTX 5060.
- Reports were turned into labels with an LLM. A keyword baseline reached 0.6496 on the held-out 29 gold studies. The first LLM prompt reached 0.7471. Switching from three-state labels to probabilities graded by severity reached 0.8065, the biggest step in the labelling work.
- The first model, KneeNet, encoded 32 slices per plane with resnet18 and pooled them with attention. Its gold score was 0.8167 at first and about 0.84 after fixing the training recipe.
- An important early lesson came from changing only the random seed, which moved the gold score from 0.8409 to 0.8019. Differences of 0.02 to 0.04 on gold are noise.

## Phase 2, first leaderboard scores and the old architecture (10-12 to 10-13)

- After the Kaggle account was verified, six KneeNet members scored 0.882. Adding members that also see slice order (depth-conv pooling) gave 0.888, and ten members including resnet34 gave 0.893.
- Labels were improved by averaging our labels with 7 public CC0 label sets. On all 58 gold studies this raised label quality from 0.8655 to 0.8923.

## Phase 3, WindowNet (10-13 to 10-14)

- A new architecture, WindowNet, changed three things at once. It reads 5 series instead of 3, crops 128 mm of real anatomy using the DICOM pixel spacing, and stacks 3 adjacent slices into one RGB image for a 2D ImageNet backbone, with one attention query per finding.
- Three WindowNet members scored 0.939, a jump of 0.046 over the best KneeNet ensemble. Five members scored 0.942, and five efficientnet members alone scored 0.943.
- Single-model probes showed efficientnet_b0 at 0.938 and resnet34 at 0.920, so weaker backbones were dropped.

## Phase 4, searching for more (10-14 to 10-17)

- Five-fold out-of-fold pseudo labels were built, the approach the forum rated highest. The resulting model scored 0.937, no better than the baseline.
- Many variants were tried on gold and none clearly helped, including longer training, larger batches through gradient accumulation, larger input size, more training windows, efficientnet_b1 and b2, regnety_016, and a cache with 3 series of 32 slices.
- Adding a different model family helped. Four tf_efficientnetv2_b0 members plus one convnext_nano member, combined with the five efficientnet members, reached **0.944**.
- Growing that to 16 members, or mixing in other input variants, kept the score at 0.944 exactly.

## Phase 5, last ideas (10-17 to 10-20)

- A full-frame tf_efficientnetv2_b0 (W39) scored 0.9301 on gold, the best single score of the project, but 0.931 on the leaderboard. Three more seeds of the same recipe scored 0.9026, 0.9062 and 0.9120 on gold, so W39 had been lucky.
- A wider 160 mm crop, tested because a third party reported that tight crops hurt MCL, scored 0.943 in the ensemble.
- Codex was asked which earlier failures deserved a retry on the new architecture. Four were tried. Hard labels scored 0.8994 on gold against 0.9159, depth-conv scored 0.935 alone on the leaderboard, attention dropout 0.934, and mixing in v2s-label members 0.943. None beat 0.944.
- Codex and the agent both concluded that 0.944 is close to the ceiling of this pipeline in the time left.

## Phase 6, wrapping up (10-20 to 10-21)

- Two final submissions were chosen, both 0.944 public. One is the 16-member ensemble. The other is the 10-member ensemble with a small 0.15 share of a model that reads a different slice layout. The half-and-half ensemble with full-frame effv2 also scored 0.944 but was not chosen, because one of its halves scored only 0.931 alone.
- Write-ups were produced in Chinese, a plain summary in `SUMMARY_zh.md`, course report notes in `COURSE_REPORT_zh.md` and an 11-slide deck in `report/RSNA_Knee_Report.pptx`.

## Findings

1. **Input design mattered more than the model.** One redesign of the input gave 0.046. Backbones, training tricks and more members gave 0.005 in total.
2. **Report text works as a weak label.** With only 58 image-labelled studies, LLM soft labels from reports were enough to train models at the 0.94 level.
3. **Diversity comes from model families, not seeds.** Mixing efficientnet, effv2 and convnext families helped, adding seeds of the same recipe did not.
4. **A small validation set misleads.** The 58 gold studies ranked models against the leaderboard several times, and seed noise on them is as large as most method effects.
5. **0.944 is probably the ceiling of this approach with these labels.** Going further would likely need a much larger backbone, external data or better labels.

## Open questions for later work

- How large a validation set, or how many seeds, would be needed to measure effects of about 0.005 reliably.
- Whether better report labels improve the model at all. Label quality rose by 0.027, but the model difference stayed inside the noise.
- Why in-sample pseudo labels did not lose to out-of-fold ones here.
- Why depth context helped the old architecture but not WindowNet, which already stacks adjacent slices in its input.
- Why third-party input ablations, such as dropping non-fat-suppressed series or widening the crop, did not reproduce on our data.

## Things learned about the tooling

- Kaggle allows 45 GPU hours a week, 2 GPU sessions at once and 5 submissions a day. Submitting an existing kernel version costs no GPU quota, while pushing a new version does.
- Using single-question leaderboard probes (one model alone, or one change to the best ensemble) was the most reliable way to decide anything.
- On this Windows machine, stopping a background task from Claude Code did not stop the script behind it. Surviving scripts once pushed a duplicate training run, so background jobs must be killed by process id.
