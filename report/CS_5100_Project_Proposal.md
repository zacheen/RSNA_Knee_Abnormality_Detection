# Multimodal learning for knee abnormality detection

Team members: Yujing Wang, Yu-Chang Shih

## 1. Introduction

The knee is among the most commonly imaged joints, and its injuries and degenerative disorders can substantially impair mobility and quality of life [1,2].
Common knee disorders affect a substantial proportion of adults: knee osteoarthritis was estimated to affect 654 million people aged 40 years and older worldwide in 2020 [2], while the Framingham Osteoarthritis Study found a meniscal tear, alone or with meniscal destruction, on right-knee MRI in 31% of 991 adults aged 50–90 years [3].
In that study, 61% of these tears occurred in participants without knee pain, aching or stiffness in the previous month [3].
Assessing these disorders requires detailed evaluation of the joint’s soft tissues and bone.
Magnetic resonance imaging (MRI) provides this anatomical detail without ionizing radiation and can reveal coexisting ligament injuries, meniscal tears, cartilage damage and bone lesions [1,4].

Interpreting these findings requires radiologists to follow structures across slices and integrate evidence from multiple planes and sequences.
Subtle lesions can resemble normal anatomical variants, contributing to diagnostic variability [1,5].
Deep learning systems now classify multiple knee abnormalities, and reader studies have shown that their predictions can improve diagnostic performance [4,5].
Performance across institutions and acquisition protocols remains variable, however, and assembling comprehensive expert annotations is labor-intensive.
Radiology reports offer a complementary source of supervision, but their diagnostic information must be used with attention to uncertainty and differences in labeling criteria.

Here, we develop a report-supervised model to detect twelve knee abnormalities using the Radiological Society of North America (RSNA) Knee Abnormality Detection dataset [6].
The targets span ligament and meniscal injuries, compartmental osteoarthritis, effusion, synovitis, Baker’s cyst, bone contusion and fracture.
Our approach combines labels extracted from multilingual radiology reports with a multisequence MRI classifier that integrates neighboring slices through finding-specific attention [13].
Reports provide supervision during training; prediction requires only images and acquisition metadata.
Building on the implemented system, we will examine which design choices improve performance and how reliably these improvements can be assessed with limited expert annotation.

## 2. Previous Work and Contributions

### Related work

Recent work has expanded MRI analysis from task-specific models to representations that transfer across anatomical regions and clinical applications.
MARS separates shared anatomical information from acquisition-dependent variation through multisequence pretraining and was evaluated across 44 tasks, including diagnosis, segmentation and report generation [7].
Hoyer et al. adapted SAM-family segmentation models to five musculoskeletal regions and used the resulting measurements for knee triage and outcome prediction [8].
These studies demonstrate the breadth of pretrained MRI analysis; their relevance to our project lies in representation learning and adaptation, while detection of individual abnormalities requires task-specific evaluation.

An early foundation for knee classification was MRNet, which combined ImageNet-pretrained AlexNet slice encoders, max pooling across the slices of each series and logistic regression across imaging planes [1].
Trained on labels manually extracted from reports and evaluated internally against the majority vote of three musculoskeletal radiologists, it detected general abnormalities, anterior cruciate ligament (ACL) tears and meniscal tears.
External evaluation underscored the importance of domain adaptation: the area under the receiver operating characteristic curve (AUC) for ACL injury increased from 0.824 to 0.911 after training on data from the external institution.
MRNet therefore provides a precedent for both transfer learning and report-derived supervision, as well as a warning against assuming that performance will transfer unchanged between datasets.

CoPAS extended this approach to twelve abnormalities through three-dimensional convolutional features and attention across planes and sequences [4].
Its five-center cohort comprised 1,748 patients, all of whom underwent knee arthroscopy, with arthroscopic reference standards for meniscal, ACL and cartilage lesions and MRI-based labels for the other nine findings.
Mean AUC was 0.812 at the development center and 0.721–0.726 at the external centers.
The multisequence architecture and reader study support learned integration of complementary views, but the surgical cohort and the exclusion of scans with motion or metal artifacts restrict applicability to routine examinations.
The authors also identify the need for scans in three planes as a limitation.
The twelve categories also differ from the RSNA targets, precluding a direct comparison of aggregate scores.

Xie et al. evaluated a larger cohort of 14,962 examinations from 13,419 patients across five centers [5].
Although described as a multi-task system, it localized the knee through three-dimensional meniscus segmentation and then applied nine independently trained classifiers, each using attention to refine relevant image regions.
External testing and staged reader studies showed improved diagnostic accuracy and shorter reading times with model assistance.
The reference standard nevertheless relied on MRI and report review without arthroscopic confirmation, and the retrospective Chinese cohorts leave prospective and international generalizability unestablished.
This study also provides an architectural alternative to a shared model: a separate classifier for each finding.

Beyond current abnormalities, MRI models have been used to predict future clinical outcomes.
MR-Transformer adapted the ImageNet-pretrained DeiT-Tiny vision transformer to volumetric MRI through positional embeddings and attention across patches [9].
For total knee replacement prediction, sevenfold nested cross-validation on case-control pairs from the Osteoarthritis Initiative and Multicenter Osteoarthritis Study, matched on sex, race, age and body mass index, yielded AUCs of 0.84–0.88 across four sequences.
The authors found no evidence of a difference from MRNet (P = .36–.92) despite substantially greater computational requirements.
These results support evaluating computational cost alongside discrimination and motivate our use of compact pretrained image encoders.
The prognostic endpoint and predominantly older, overweight, White cohorts limit direct extrapolation to abnormality detection.

Clinical text offers an additional source of supervision.
BiomedGPT jointly models images and language across biomedical tasks, although reported generation errors and uneven transfer across modalities emphasize the need for task-specific validation [10].
Our implementation instead uses a language model to derive training targets from reports, without a text encoder at inference.
Attention-based multiple instance learning provides the basis for aggregating images under examination-level supervision [11].
Self-supervised representations such as DINOv2 [12] offer another transfer-learning strategy, but the current repository centers on ImageNet-pretrained convolutional backbones.
The resulting approach combines report-derived supervision with efficient integration of multiple MRI sequences.

### Proposed contributions and methodology

The current pipeline trains on 4,349 report-labeled examinations and reserves 58 radiologist-labeled examinations from the 4,407-study dataset for local evaluation [13].
It has evolved from KneeNet, a slice encoder with attention pooling, to WindowNet, which combines neighboring slices and attends across sequences.
Our contribution is an empirical evaluation of report-derived supervision, input construction and ensemble diversity under limited expert annotation and computational resources.

Image representation and integration.
WindowNet selects up to five series: fluid-sensitive sagittal, coronal and axial images, plus non-fat-suppressed sagittal and coronal images.
The principal configuration samples 20 slices per series, applies a 128-mm physical center crop using DICOM pixel spacing and resizes images to 320 × 320 pixels.
Three consecutive sampled slices form a three-channel window for an ImageNet-pretrained encoder.
Series and relative-position embeddings inform finding-specific attention across windows, with missing series masked.
Training samples windows and spatial crops; inference uses all windows with center cropping and left–right mirror averaging.
The ten-member ensemble combines five EfficientNet models, four EfficientNetV2-B0 models and one ConvNeXt-Nano model [13].

Report-derived supervision.
A language model converts multilingual reports into soft targets for twelve findings, preserving graded and equivocal descriptions.
Unmentioned findings are assigned a missing value and excluded from binary cross-entropy.
The principal label set averages the project’s refined labels with seven public label sets, ignoring unavailable entries.
Training fine-tunes the encoder and attention layers with AdamW and a one-cycle learning-rate schedule; predictions are averaged over the final three epochs.
Experiments also compare hard targets and pseudo-labels formed by blending report targets with model predictions [13].
These are model-generated supervision signals, not calibrated clinical probabilities.

Evaluation.
We distinguish report-label agreement, image-model performance on the 58 expert-labeled studies and public leaderboard AUC.
The expert labels are excluded from image-model training but have informed prompt development and repeated model comparisons, so this cohort is a development set rather than an independent test set.
Prompt development initially used 29 studies, with the remaining 29 withheld for evaluation.
Fivefold pseudo-label experiments group identical normalized report texts within folds; this does not establish patient-level independence.
We will report macro and per-finding AUC, class counts and variation across seeds, and prioritize controlled comparisons of series selection, physical cropping, window construction and label sources.
Repeated leaderboard use also limits interpretation of small gains [13].

## 3. Expected Results and Preliminary Results

The repository records a public leaderboard macro AUC of 0.944 for the ten-member WindowNet ensemble [13].
An earlier ten-member KneeNet ensemble scored 0.893, whereas the first three-member WindowNet ensemble scored 0.939.
This transition changed series selection, physical cropping, neighboring-slice inputs and attention together; the improvement cannot be attributed to any one component.
Combining five EfficientNet members with four EfficientNetV2-B0 members and one ConvNeXt-Nano member increased the reported score from 0.943 to 0.944.
Expanding to sixteen members produced the same reported score.
These are repository-reported public leaderboard results, not independently reproduced estimates or evidence of clinical benefit.

Report-label evaluation improved from an AUC of 0.6496 for keyword rules to 0.8065 for severity-sensitive soft targets on the 29 studies withheld from prompt development.
Across all 58 expert-labeled studies, averaging the refined labels with seven public sources increased label agreement from 0.8655 to 0.8923 [13].
Better label agreement did not consistently improve image-model performance.
Local and leaderboard rankings also diverged: a full-frame model reached a local AUC of 0.9301 but a public score of 0.931, below a baseline model’s public score of 0.938.
These observations motivate a closer examination of validation reliability.

Further work will focus on separating the effects of input design from backbone and label choices, using matched configurations and repeated seeds.
The existing experiments suggest that incorporating local slice context and complementary sequences is useful, but do not establish the independent benefit of each component.
Out-of-fold pseudo-labels and additional ensemble members showed no clear advantage in the reported comparisons.
We will consolidate reproducible configurations, examine errors by finding and quantify uncertainty where predictions are available.
A larger independent evaluation cohort is needed to resolve small differences, and external validation and reader studies would be required before drawing conclusions about clinical utility.

## References

[1\. Bien, N. et al. Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet. PLOS Med. 15, e1002699 (2018).](https://doi.org/10.1371/journal.pmed.1002699)

[2\. Cui, A. et al. Global, regional prevalence, incidence and risk factors of knee osteoarthritis in population-based studies. EClinicalMedicine 29–30, 100587 (2020).](https://doi.org/10.1016/j.eclinm.2020.100587)

[3\. Englund, M. et al. Incidental meniscal findings on knee MRI in middle-aged and elderly persons. N. Engl. J. Med. 359, 1108–1115 (2008).](https://doi.org/10.1056/NEJMoa0800777)

[4\. Qiu, Z. et al. Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities. Nat. Commun. 15, 7637 (2024).](https://doi.org/10.1038/s41467-024-51888-4)

[5\. Xie, Z. et al. Development of a multi-task deep learning system for classification of nine common knee abnormalities on MRI: a large-scale, multicentre, stepwise validation study. EClinicalMedicine 89, 103534 (2025).](https://doi.org/10.1016/j.eclinm.2025.103534)

[6\. Radiological Society of North America. RSNA Knee Abnormality Detection AI Challenge (2026).](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)

[7\. Qiu, Z. et al. Large-scale multi-sequence pretraining for generalizable MRI analysis in versatile clinical applications. Nat. Biomed. Eng. (2026).](https://doi.org/10.1038/s41551-026-01740-5)

[8\. Hoyer, G. et al. Clinical utility of foundation models in musculoskeletal MRI for biomarker fidelity and predictive outcomes. npj Digit. Med. 9, 383 (2026).](https://doi.org/10.1038/s41746-026-02520-w)

[9\. Zhang, C. et al. MR-Transformer: A Vision Transformer-based Deep Learning Model for Total Knee Replacement Prediction Using MRI. Radiol. Artif. Intell. 7, e240373 (2025).](https://doi.org/10.1148/ryai.240373)

[10\. Zhang, K. et al. A generalist vision–language foundation model for diverse biomedical tasks. Nat. Med. 30, 3129–3141 (2024).](https://doi.org/10.1038/s41591-024-03185-2)

[11\. Ilse, M., Tomczak, J. & Welling, M. Attention-based Deep Multiple Instance Learning. Proc. Mach. Learn. Res. 80, 2127–2136 (2018).](https://proceedings.mlr.press/v80/ilse18a.html)

[12\. Oquab, M. et al. DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193 (2023; revised 2024).](https://arxiv.org/abs/2304.07193)

[13\. RSNA\_Knee\_Abnormality\_Detection project repository. Source code and experiment records, commit d834783 (2026).](https://github.com/zacheen/RSNA_Knee_Abnormality_Detection/tree/d8347834345feb57da2f2284ba88a93c8f8ff305)
