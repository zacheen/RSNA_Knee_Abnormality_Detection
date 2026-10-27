# AI suggestions that the experiments refuted

This file collects suggestions made by AI assistants during the competition that our own experiments then contradicted. It is material for the final course report. Every number comes from the experiment log in `RUNBOOK.md` section 8, and each item names the experiment ID so it can be traced there.

Three assistants made suggestions.
- Claude Code ran the project and proposed most experiments, often after reading public repositories and papers.
- Codex (gpt-6-astra) answered round 10 on 2026-10-20. Its reply is in `data/rounds/r8/codex2.md`.
- Antigravity (Gemini 3.8 Flash) answered round 9 on 2026-10-17. Its reply is in `data/rounds/r8/agy.md`.

How to read the evidence.
- A public leaderboard (LB) probe scores about 400 hidden test studies and is the stronger evidence.
- A score on the 58 expert-labeled studies (gold) moves by 0.02 to 0.04 between seeds alone, so a single gold difference smaller than that is only a direction.
- Codex hedged most of its predictions, saying "likely flat" beside each expected gain. So the fair claim is that the predicted gain did not appear and the result went the other way, not that Codex promised a gain.

## Suggestions from Codex (round 10)

1. **Hard 0/1 labels instead of soft labels (W45).**
   - Suggestion. Threshold the blended soft labels at 0.5 to separate the effect of softness from label quality, with a plausible gain of 0.001 to 0.002.
   - Result. Gold fell from 0.9159 (W8, same seed) to 0.8994, the largest drop being Effusion from 0.953 to 0.853. Soft labels remain better on WindowNet.
2. **Blend the v2s-label members at 20% (W4, W16).**
   - Suggestion. A zero-cost test of label-source diversity, with 0 to 0.001 gain expected.
   - Result. Public LB 0.943 against 0.944 for the ensemble alone.
3. **Residual depth convolution inside WindowNet (W46, W48).**
   - Suggestion. Codex ranked it the idea with the highest upside, 0.001 to 0.003, reasoning that depth-conv had helped the older KneeNet. It specified contiguous windows per slot and a zero-initialized last layer, and our implementation followed that.
   - Result. Gold 0.9109 and 0.9107 against 0.9159, and public LB 0.935 against 0.938 for W8 alone.
4. **Dropout 0.2 inside the finding attention (W47).**
   - Suggestion. Old head-dropout results would not transfer to attention dropout, with a possible gain of 0.001.
   - Result. Gold 0.9129 and public LB 0.934 against 0.938 for W8.

## Suggestions from Gemini (round 9)

5. **Full-frame training as a source of diversity (W39 to W42).**
   - Suggestion. Train tf_efficientnetv2_b0 and convnext_nano on the full 320 px frame instead of random 288 px crops, on the premise that full-frame b0 had already added diversity.
   - Result. W39 reached gold 0.9301, the best single model of the competition, but scored only 0.931 on the public LB, below W8 at 0.938. Three more seeds of the same recipe gave 0.9026, 0.9062 and 0.9120, so the 0.9301 was mostly seed luck. Mixing W39 and W41 into the 0.944 ensemble at half weight still gave 0.944.
6. **In-sample pseudo labels work and OOF pseudo labels fail.**
   - Suggestion. Gemini treated P1 (0.939) against W8 (0.938) as proof that in-sample pseudo labels improve a model, and proposed iterating them with the 0.944 ensemble.
   - Result. A 0.001 LB difference is below what three decimals can resolve, and P1 did not beat W8 on gold (0.9136 against 0.9159). The evidence supports no difference rather than an improvement. We did not run the iterated version, so this item is a refuted premise rather than a refuted experiment.

## Suggestions from Claude Code

7. **Better labels should give better models (L8, W4, W16).**
   - Suggestion. Blending our v2s labels with seven public label sets raised agreement with radiologists from 0.8655 to 0.8923, and the plan assumed the image models would follow.
   - Result. Models trained on each label set differ by 0.004 to 0.006 on gold, in opposite directions on the two architectures, which is inside seed noise. The honest conclusion is that no effect could be measured, not that the labels do not matter.
8. **Out-of-fold pseudo labels should beat in-sample ones (P3, P4, W29).**
   - Suggestion. The usual expectation is that in-sample predictions leak the training labels, so OOF pseudo labels were built with 5-fold training.
   - Result. P3 scored public LB 0.937 against 0.939 for in-sample P1 and 0.938 for W8.
9. **Drop the non-fat-suppressed series and give their slice budget to the FS series (V5A).**
   - Suggestion. Based on a third-party ablation (the knee-mri-raptor repository) that reported no loss from dropping the non-FS series.
   - Result. Gold 0.9086 against 0.9159, with the largest drops on Lateral OA (0.779) and Lateral Meniscus (0.845). Mixed into the ensemble at 0.15 it gave public LB 0.944, no gain.
10. **Widen the crop from 128 mm to 160 mm (W43, W44).**
    - Suggestion. The same repository reported that tighter crops lost AUC, so 128 mm might be cutting off surrounding structures.
    - Result. W43 gold 0.9033 against 0.9159, and the 160 mm group mixed at 0.2 gave public LB 0.943 against 0.944.
11. **Copy the training recipe of a public 0.949 model (W13).**
    - Suggestion. Gradient accumulation to an effective batch of 16 with 16 epochs, the main remaining difference from that model.
    - Result. Gold 0.9022 against 0.9159.
12. **Depth convolution transfers from KneeNet to WindowNet.**
    - Suggestion. Depth-conv raised the older KneeNet from about 0.841 to 0.856 on gold (L4, L5), so it was expected to help WindowNet too. Codex reached the same conclusion independently in item 3.
    - Result. See item 3. The context that depth-conv added to KneeNet seems to be already present in WindowNet's three-slice windows.

## What the pattern suggests

- The strongest refutations come from leaderboard probes (items 3, 4, 5 and 8). Several of the others rest on one gold run each and would need more seeds to be firm.
- Item 5 is the clearest case for the report. The best model on the 58 expert-labeled studies was among the worst on the leaderboard, which shows why a small expert set cannot rank models.
- Items 9 to 11 show that ablations reported by third parties did not transfer to a different pipeline, even when the setting looked close.
- Items 3 and 12 show two assistants agreeing on a hypothesis from the same earlier evidence and both being wrong, so agreement between AI assistants was not independent evidence.
