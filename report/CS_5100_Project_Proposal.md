# Detecting knee abnormalities on MRI from report-derived labels: 2.5D and 3D approaches

Team members: Yujing Wang, Yu-Chang Shih

## 1. Introduction

Knee osteoarthritis (OA) affected an estimated 654 million people aged 40 and older in 2020 [1], and in the Framingham Osteoarthritis Study a meniscal tear appeared on magnetic resonance imaging (MRI) in 31% of adults aged 50 to 90, most often in people without knee symptoms [2]. Reading knee MRI is demanding, because a radiologist follows small structures across dozens of slices in several sequences and subtle lesions can resemble normal anatomy [3,4]. Deep learning can help, but it usually needs large sets of expert image labels, which are slow and expensive to produce, while radiology reports already exist for nearly every examination.

This project uses the Radiological Society of North America (RSNA) Knee Abnormality Detection dataset [5]. Of its 4,407 knee MRI studies, only 58 carry radiologist labels for all twelve target findings, namely anterior cruciate ligament (ACL) and medial collateral ligament (MCL) injury, medial and lateral meniscal tears, medial, lateral and patellofemoral OA, effusion, synovitis, Baker's cyst, bone contusion and fracture. The others come with free-text reports in several languages, from which a language model extracts training labels. Yu-Chang Shih has developed a 2.5D model that reads three adjacent slices at a time (Approach A), and Yujing Wang will develop a 3D model, inspired by CoPAS [6], that reads each MRI series as a whole volume (Approach B).

Even the strongest competition entries fall well short of perfect ranking, and we see four reasons. The reference labels are imperfect, because radiologists disagree on subtle lesions [3,4]. Reports are written for clinical communication and often omit findings that a radiologist annotating the images would mark, so the training and reference labels follow different criteria. With only 58 expert-labeled examinations, the random seed alone can move the score as much as a change of method. The input is heterogeneous, with three to fourteen series per examination, varying protocols and missing sequences, and offline inference on about 1,300 examinations within nine hours limits model size and resolution. Our central question is therefore how much of the remaining error comes from the labels rather than from the model, and which design choices still help once label noise and evaluation noise are taken into account.

Our goal is to compare the 2.5D WindowNet with a 3D CNN when both are trained mainly on noisy report-derived labels, and to measure how reliably a difference between them can be detected with so few expert labels. Both are scored by macro-averaged area under the ROC curve (AUC) over the twelve findings, under one evaluation protocol. Any conclusion will apply to these two implementations rather than to 2.5D and 3D models in general.

## 2. Previous Work and Contributions

Four studies shaped our approach.

- **MRNet [3]**
  - *Data and labels.* 1,370 exams and 3 findings, with training labels from reports and validation labels from three radiologists.
  - *Method.* A 2D ImageNet CNN (AlexNet) per slice, max-pooled per series, with planes combined by logistic regression.
  - *Main result.* External ACL AUC rose from 0.824 to 0.911 after retraining on the new site.
  - *What we liked.* Simple and reproducible, and shows that 2D transfer learning and report labels work.
  - *Limitations.* Ignores slice order, covers only 3 findings and does not transfer unchanged between sites.
- **CoPAS [6]**
  - *Data and labels.* 1,748 patients and 12 findings, with arthroscopy-confirmed labels for key findings.
  - *Method.* 3D convolutional features per sequence, with attention across planes and sequences.
  - *Main result.* Mean AUC 0.812 at the development center and 0.721–0.726 at external centers, and improved radiologists' accuracy.
  - *What we liked.* Reads whole volumes, and its reader study ties the results to clinical use.
  - *Limitations.* A surgical cohort that excludes artifacts, needs all 3 planes and uses categories that differ from RSNA.
- **Xie et al. [4]**
  - *Data and labels.* 14,962 exams from 5 centers and 9 findings, labeled by MRI and report review.
  - *Method.* The knee is localized by meniscus segmentation, then nine independently trained classifiers are applied, although the system is described as multi-task.
  - *Main result.* Better reader accuracy and shorter reading time with assistance.
  - *What we liked.* Large multicenter scale and staged clinical validation.
  - *Limitations.* No arthroscopic confirmation, and separate models cannot share learning.
- **MR-Transformer [7]**
  - *Data and labels.* OAI and MOST cohorts, predicting knee replacement.
  - *Method.* An ImageNet-pretrained vision transformer (DeiT-Tiny) adapted to MRI volumes.
  - *Main result.* AUC 0.84–0.88.
  - *What we liked.* Careful nested cross-validation.
  - *Limitations.* Not significantly better than MRNet despite much higher compute.

Together, these studies suggest that ImageNet pretraining transfers well to knee MRI, that reading slices in context matters, and that results rarely carry over unchanged to new data. CoPAS in particular shows the value of 3D features, but under clean, surgery-confirmed labels. Whether that advantage survives noisy report labels is the question our comparison is built to answer.

Both approaches share the same labels and images, so that any difference between them comes from the model. A language model reads each report and outputs a probability for each finding, graded by the severity described, and findings that a report does not mention are left out of the loss. The principal label set averages these with public label sets. Images are converted once into a cache holding up to five series per study (fat-suppressed sagittal, coronal and axial, plus non-fat-suppressed sagittal and coronal), 20 slices each, cropped to 128 mm of anatomy and resized to 320 px [8].

Approach A, WindowNet, stacks three adjacent slices into one RGB image for an ImageNet-pretrained 2D encoder such as EfficientNet-B0. Each finding then has its own attention query over all windows of all series, an attention-based form of multiple-instance learning [9]. This design reuses strong 2D pretraining while adding the local slice context that MRNet-style pooling loses.

We will also run the ablations below, each changing one part of a single EfficientNet-B0 model in the principal configuration.

- **Three-slice windows against single slices.** A single grayscale slice, as in MRNet [3], gives a 2D encoder no context from neighbouring slices, which is the context a 3D encoder reads directly.
- **A residual convolution across windows.** Attention pooling ignores slice order, although the extent of a lesion across slices is diagnostic, and zero-initialized 3D residual blocks inside a 2D ResNet have added this context in knee MRI [14].
- **Attention against mean pooling.** Attention assumes that only a few windows carry the evidence for a finding [9], and plain mean pooling matched or beat it on several 3D neuroimaging tasks [15].
- **Five series against fat-suppressed series only.** Non-fat-suppressed series may show bone and degenerative change better, and CoPAS combines several sequences [6], although an unverified public repository reported little gain from them [19].
- **Down-weighting uncertain targets, and ignoring, zeroing or down-weighting unmentioned findings.** Targets near 0.5 come from equivocal wording, and a report may omit a finding that is present. CheXpert compared ways of mapping uncertain labels [16], report-derived labels can keep "not mentioned" apart from "absent" [17], and unobserved labels can be treated as noisy negatives [18].

We will also compare report-label agreement with expert labels and image-model AUC for each finding to look for patterns consistent with label-related limitations. This analysis alone cannot establish their cause.

Approach B replaces WindowNet's 2D window encoder with a 3D convolutional network and keeps everything else identical: cache, labels, per-finding attention head, training schedule and test-time mirroring. A 3D encoder can follow the ACL or a meniscus continuously through the slices, much as a radiologist does when scrolling. We will start with a small video-pretrained 3D ResNet (R3D-18 or R(2+1)D-18 [10]) on a single series. We will extend it to all five series and add, if time permits, a CoPAS-style attention layer across series. Pretrained 3D encoders are scarcer than 2D ones, so the choice of initialization matters:

- **Video-pretrained 3D CNN [10].** Easy to load and fast, but trained on natural videos.
- **Medical 3D pretraining (Med3D [11], MARS [12]).** Closer to our image domain, but trained on other anatomy and tasks, with varying availability and license.
- **2D ImageNet weights inflated to 3D [13].** Starts from the strongest 2D weights, but is heavier.
- **Training from scratch.** No domain mismatch, but likely too little data for a 3D model.

Because a stronger encoder could win for reasons unrelated to 3D design, we will report a matched version with an encoder of size comparable to EfficientNet-B0 and, if time allows, a strong version with the best encoder that fits the Kaggle budget of 9 hours on a T4 GPU for about 1,300 hidden studies. Before training at scale, we will benchmark cache building and model inference on the target GPU to check the projected runtime.

The comparison needs care, because the 58 expert-labeled studies have already been used to tune the label prompt and to compare many models, and changing only the random seed moves their macro AUC by about as much as most of the differences between methods [8]. All design choices are therefore made on a held-out 10% of the report-labeled training studies, with studies that share a report kept in the same fold. Each final model is scored once on the 58 expert-labeled studies, with per-finding AUC and paired bootstrap confidence intervals for the difference between the approaches. The main comparison uses Kaggle late submissions on the private split of the hidden test set, which no one probed during the competition. Each approach is trained with three seeds, and single models are compared with single models.

## 3. Expected and Preliminary Results

In preliminary experiments, better agreement between report-derived and expert labels did not yield a measurable improvement in image-model performance, and model rankings on the 58 expert-labeled studies did not consistently match the public leaderboard [8]. The transition to WindowNet changed series selection, cropping and slice windows together, so its gains cannot be attributed to a single component. These observations motivate the controlled ablations and evaluation protocol above.

We expect the two approaches to differ by type of finding. For thin structures that must be followed across slices, such as the ACL and the menisci, the 3D model has the best chance to help, while for diffuse findings such as effusion and OA, which are visible on single slices, we expect both to perform similarly. Some findings have very few expert positives, for example MCL with 9 of 58 and lateral OA with 11, so conclusions about them will rest on the Kaggle test set. Disagreement between report-derived and expert labels may affect both models' performance, making it difficult to isolate the effects of label quality from those of model architecture.

Overall, we expect a single 3D model to land near, and possibly below, a single WindowNet model, mainly because pretrained 3D encoders are weaker than 2D ImageNet models. We will also test whether an ensemble of both approaches beats either one alone.

<div class="page-break"></div>

## References

1. Cui, A. et al. Global, regional prevalence, incidence and risk factors of knee osteoarthritis in population-based studies. *EClinicalMedicine* 29–30, 100587 (2020). https://doi.org/10.1016/j.eclinm.2020.100587
2. Englund, M. et al. Incidental meniscal findings on knee MRI in middle-aged and elderly persons. *N. Engl. J. Med.* 359, 1108–1115 (2008). https://doi.org/10.1056/NEJMoa0800777
3. Bien, N. et al. Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet. *PLOS Med.* 15, e1002699 (2018). https://doi.org/10.1371/journal.pmed.1002699
4. Xie, Z. et al. Development of a multi-task deep learning system for classification of nine common knee abnormalities on MRI: a large-scale, multicentre, stepwise validation study. *EClinicalMedicine* 89, 103534 (2025). https://doi.org/10.1016/j.eclinm.2025.103534
5. Radiological Society of North America. RSNA Knee Abnormality Detection AI Challenge (2026). https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge
6. Qiu, Z. et al. Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities. *Nat. Commun.* 15, 7637 (2024). https://doi.org/10.1038/s41467-024-51888-4
7. Zhang, C. et al. MR-Transformer: A Vision Transformer-based Deep Learning Model for Total Knee Replacement Prediction Using MRI. *Radiol. Artif. Intell.* 7, e240373 (2025). https://doi.org/10.1148/ryai.240373
8. RSNA_Knee_Abnormality_Detection project repository. Source code and experiment records, commit d834783 (2026). https://github.com/zacheen/RSNA_Knee_Abnormality_Detection/tree/d8347834345feb57da2f2284ba88a93c8f8ff305
9. Ilse, M., Tomczak, J. & Welling, M. Attention-based Deep Multiple Instance Learning. *Proc. Mach. Learn. Res.* 80, 2127–2136 (2018). https://proceedings.mlr.press/v80/ilse18a.html
10. Tran, D. et al. A Closer Look at Spatiotemporal Convolutions for Action Recognition. *CVPR* (2018). https://arxiv.org/abs/1711.11248
11. Chen, S., Ma, K. & Zheng, Y. Med3D: Transfer Learning for 3D Medical Image Analysis. arXiv:1904.00625 (2019). https://arxiv.org/abs/1904.00625
12. Qiu, Z. et al. Large-scale multi-sequence pretraining for generalizable MRI analysis in versatile clinical applications. *Nat. Biomed. Eng.* (2026). https://doi.org/10.1038/s41551-026-01740-5
13. Carreira, J. & Zisserman, A. Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. *CVPR* (2017). https://arxiv.org/abs/1705.07750
14. Vuskov, R. et al. Comprehensive deep learning-assisted multi-condition analysis of knee MRI studies improves resident radiologist performance. *Eur. Radiol.* (2025). https://doi.org/10.1007/s00330-025-12052-8
15. Harvey, E. et al. A Multi-Dataset Benchmark of Multiple Instance Learning for 3D Neuroimage Classification. arXiv:2604.26807 (2026). https://arxiv.org/abs/2604.26807
16. Irvin, J. et al. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. *Proc. AAAI Conf. Artif. Intell.* 33, 590–597 (2019). https://doi.org/10.1609/aaai.v33i01.3301590
17. Bigolin Lanfredi, R., Mukherjee, P. & Summers, R. M. Enhancing chest X-ray datasets with privacy-preserving large language models and multi-type annotations: a data-driven approach for improved classification. *Med. Image Anal.* (2024). https://doi.org/10.1016/j.media.2024.103383
18. Kim, Y., Kim, J. M., Akata, Z. & Lee, J. Large Loss Matters in Weakly Supervised Multi-Label Classification. *CVPR* (2022). https://arxiv.org/abs/2206.03740
19. ibrahimdodo. knee-mri-raptor. GitHub repository for the RSNA Knee Abnormality Detection competition (2026). https://github.com/ibrahimdodo/knee-mri-raptor. Not peer reviewed. Its ablation results are self-reported in the repository README and have not been independently verified.
