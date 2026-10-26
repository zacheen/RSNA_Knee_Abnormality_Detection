# Detecting knee abnormalities on MRI from report-derived labels: 2.5D and 3D approaches

Team members: Yujing Wang, Yu-Chang Shih

## 1. Introduction

Knee disorders are common and often disabling. Knee osteoarthritis (OA) affected an estimated 654 million people aged 40 and older in 2020 [1], and in the Framingham Osteoarthritis Study a meniscal tear, alone or with meniscal destruction, appeared on magnetic resonance imaging (MRI) of the right knee in 31% of adults aged 50 to 90, most often in people without knee symptoms [2]. MRI shows ligaments, menisci, cartilage and bone without ionizing radiation, but reading it is demanding. A radiologist follows small structures across dozens of slices in several sequences, and subtle lesions can resemble normal anatomy [3,4]. Deep learning can help, yet training it usually needs large sets of expert image labels, which are slow and expensive to produce. Radiology reports, in contrast, already exist for nearly every examination.

This project works with the Radiological Society of North America (RSNA) Knee Abnormality Detection dataset [5], which reflects that situation closely. Of its 4,407 knee MRI studies, only 58 carry radiologist labels for all twelve target findings: anterior cruciate ligament (ACL) and medial collateral ligament (MCL) injury, medial and lateral meniscal tears, medial, lateral and patellofemoral OA, effusion, synovitis, Baker's cyst, bone contusion and fracture. The other 4,349 come with free-text reports in several languages, from which a language model extracts training labels. On this shared foundation, each of us builds a different image model. Yu-Chang Shih has developed a 2.5D model that reads three adjacent slices at a time (Approach A). Yujing Wang will develop a 3D model, inspired by CoPAS [6], that reads each MRI series as a whole volume (Approach B).

Our goal is to learn which way of reading the MRI volume works better when the only supervision is noisy report text, and how reliably such a difference can be measured with so few expert labels. We compare both approaches under one evaluation protocol, using macro-averaged area under the ROC curve (AUC) over the twelve findings. The questions reach beyond this dataset: they concern how to represent volumetric, multi-sequence images in medical image analysis, how to learn from text-derived weak labels, how to build decision support for radiology, and how to draw sound conclusions from small expert-labeled test sets.

Why the task is hard. Macro AUC is the probability that a randomly chosen positive examination is ranked above a randomly chosen negative one, averaged over the twelve findings, and even the strongest competition entries fall short of perfect ranking. We see five reasons for this gap, and each one shapes the questions we ask. First, the reference labels are themselves imperfect, because radiologists reading the same examination disagree, particularly on subtle lesions that resemble normal variants [3,4]. Second, the training labels and the reference labels follow different criteria, since a report is written for clinical communication and often omits findings that a radiologist annotating the images would mark. Third, only 58 examinations carry expert labels, so the variation caused by the random seed alone can be as large as the differences between methods. Fourth, the input is heterogeneous, with between three and fourteen series per examination, varying protocols, missing sequences and reports in several languages. Fifth, subtle findings call for fine spatial detail across many slices, while the competition requires offline inference on about 1,300 examinations within nine hours, which limits model size, resolution and ensemble size. Our central question is therefore how much of the remaining error comes from the labels rather than from the model, and which design choices still help once label noise and evaluation noise are taken into account.

## 2. Previous Work and Contributions

Four studies shaped our approach. The table summarizes each one, what we found valuable in it and where it falls short.

| Study | Data and labels | Method | Main result | What we liked | Limitations |
|---|---|---|---|---|---|
| MRNet [3] | 1,370 exams; 3 findings; training labels from reports, validation labels from three radiologists | 2D ImageNet CNN (AlexNet) per slice, max-pooled per series, planes combined by logistic regression | External ACL AUC rose from 0.824 to 0.911 after retraining on the new site | Simple and reproducible; shows 2D transfer learning and report labels work | Ignores slice order; only 3 findings; does not transfer unchanged between sites |
| CoPAS [6] | 1,748 patients; 12 findings; arthroscopy-confirmed labels for key findings | 3D convolutional features per sequence, attention across planes and sequences | Mean AUC 0.812 at the development center, 0.721–0.726 at external centers; improved radiologists' accuracy | Reads whole volumes; reader study ties results to clinical use | Surgical cohort; excludes artifacts; needs all 3 planes; categories differ from RSNA |
| Xie et al. [4] | 14,962 exams, 5 centers; 9 findings; MRI and report review | Knee localized by meniscus segmentation, then nine independently trained classifiers, although described as multi-task | Better reader accuracy and shorter reading time with assistance | Large multicenter scale, staged clinical validation | No arthroscopic confirmation; separate models cannot share learning |
| MR-Transformer [7] | OAI and MOST cohorts; knee replacement outcome | ImageNet-pretrained vision transformer (DeiT-Tiny) adapted to MRI volumes | AUC 0.84–0.88 | Careful nested cross-validation | Not significantly better than MRNet despite much higher compute |

Together, these studies suggest that ImageNet pretraining transfers well to knee MRI, that reading slices in context matters, and that results rarely carry over unchanged to new data. CoPAS in particular shows the value of 3D features, but under clean, surgery-confirmed labels. Whether that advantage survives noisy report labels is the question our comparison is built to answer.

Both approaches share the same labels and images, so that any difference between them comes from the model. A language model reads each report and outputs a probability for each finding, graded by the severity described; findings that a report does not mention are left out of the loss. The principal label set averages these with seven public label sets. Simpler options exist: keyword rules are fast and transparent but agreed poorly with radiologists, hard 0/1 labels discard the uncertainty that reports express, and manual image annotation is accurate but infeasible at this scale. Images are converted once into a compact cache holding up to five series per study (fat-suppressed sagittal, coronal and axial, plus non-fat-suppressed sagittal and coronal), 20 slices each, cropped to 128 mm of anatomy using DICOM pixel spacing and resized to 320 px [8].

Approach A, WindowNet, stacks three adjacent slices into one RGB image for an ImageNet-pretrained 2D encoder such as EfficientNet-B0. Each finding then has its own attention query over all windows of all series, an attention-based form of multiple-instance learning [9]. This design reuses strong 2D pretraining while adding local slice context. It improves on MRNet-style pooling, which loses slice order, and the ablations below test whether convolutions across slices add anything further.

Within Approach A, we will also run the ablations below. Each changes one part of the pipeline and starts from a single EfficientNet-B0 in the principal configuration.

| Ablation | Why we want to test it |
|---|---|
| Graded soft labels against three-state labels | Reports describe degree and certainty, which a present, absent or not-mentioned label discards |
| Blended label sources against a single extraction | Independently produced label sets are unlikely to make the same errors on the same reports |
| Soft against hard targets | Earlier comparisons changed the learning rate and regularization at the same time |
| Reduced weights for uncertain targets | Targets near 0.5 come from equivocal wording and are probably noisier than confident ones |
| Excluding, zeroing or down-weighting unmentioned findings | A report may omit a finding because it is absent or only because it was not worth reporting |
| In-sample against out-of-fold pseudo-labels | A model trained on thousands of reports may correct some report noise, but in-sample predictions risk reproducing memorized labels |
| Three-slice windows against single slices | A single grayscale slice gives a 2D encoder no context from neighbouring slices |
| 128 mm against 160 mm crops | A tighter crop may cut off peripheral structures such as the collateral ligaments |
| Five series against fat-suppressed series only | Non-fat-suppressed series may show bone and degenerative change better, although an external ablation found them of little use |
| Attention against mean pooling | Attention assumes that only a few windows carry the evidence for a finding |
| A residual convolution across windows | Attention pooling ignores slice order, although the extent of a lesion across slices is diagnostic |
| ResNet18 against DINOv2-small encoders, frozen or fine-tuned | ImageNet features come from natural photographs, and self-supervised features may transfer differently to MRI |
| Dropout, longer training, larger batches and higher resolution | A small labeled set may reward regularization, while subtle findings may need more capacity |
| Extra seeds against members from other architecture families | Different architectures should make less correlated errors than seeds of one recipe |
| Blending the predictions of related findings | Some findings co-occur for anatomical reasons, such as synovitis with effusion |
| Model against label agreement for each finding | Comparing the two separates findings limited by their labels from findings limited by the model |

Approach B replaces WindowNet's 2D window encoder with a 3D convolutional network that reads each series as a whole volume, and keeps everything else identical: cache, labels, per-finding attention head, training schedule and test-time mirroring. A 3D encoder can follow the ACL or a meniscus continuously through the slices, much as a radiologist does when scrolling. We will start with a small video-pretrained 3D ResNet (R3D-18 or R(2+1)D-18 [10]) on a single series. We will then extend it to all five series and add, if time permits, a CoPAS-style attention layer across series. Pretrained 3D encoders are scarcer than 2D ones, so the choice of initialization matters:

| Encoder initialization | Advantage | Limitation |
|---|---|---|
| Video-pretrained 3D CNN [10] | Easy to load, fast | Trained on natural videos, not MRI |
| Medical 3D pretraining (Med3D [11], MARS [12]) | Closer to our image domain | Other anatomy and tasks; availability and license vary |
| 2D ImageNet weights inflated to 3D [13] | Starts from the strongest 2D weights | Heavier, suits some architectures better than others |
| Training from scratch | No domain mismatch | Likely too little data for a 3D model |

Because a stronger encoder could win for reasons unrelated to 3D design, we will report a matched version, with an encoder of size comparable to EfficientNet-B0, and, if time allows, a strong version with the best encoder that fits the compute budget. That budget comes from Kaggle, which runs inference on about 1,300 hidden studies within 9 hours on a T4 GPU. Building the cache takes about 0.7 s per study [8], which leaves roughly 20 s per study for the model, a margin a small 3D CNN should meet easily; we will confirm this on a few studies before training at scale.

The comparison itself needs care, because the 58 expert-labeled studies have already been used to tune the label prompt and to compare many models, and changing only the random seed moves their macro AUC by about as much as most of the differences between methods [8]. We therefore give each data source one role. All design choices are made on a held-out 10% of the report-labeled training studies, scored against report labels, with studies that share a report kept in the same fold. Each final model is scored once on the 58 expert-labeled studies, with per-finding AUC, the number of positives per finding and paired bootstrap confidence intervals for the difference between the approaches. The main comparison uses Kaggle late submissions, which score models on the hidden test set of about 1,300 studies, specifically its private split, which no one probed during the competition. Each approach is trained with three seeds, and single models are compared with single models.

## 3. Expected and Preliminary Results

Yu-Chang Shih has already built the full pipeline that both approaches rely on. The raw data is about 570 GB, so the work is split across platforms: reports are labeled locally with a language model, DICOM files are converted into the cache on Kaggle CPU kernels where the data lives, models are trained on Kaggle GPUs, Colab and a local GPU, and inference runs offline in a Kaggle notebook within the 9-hour limit. Every checkpoint records the code version that trained it, so each result traces back to its exact code [8].

Early work with this pipeline also exposed two limitations, both partly expected given the small expert set. Better label agreement did not produce a measurable gain in the image models, and rankings on the 58 studies often disagreed with the leaderboard. In addition, the move from an earlier slice encoder with attention pooling to WindowNet changed series selection, cropping and slice windows at once, so its gain cannot be credited to any single component. These observations motivate the stricter evaluation protocol described above.

We expect the two approaches to differ by type of finding. For thin structures that must be followed across slices, such as the ACL and the menisci, the 3D model has the best chance to help. For diffuse findings such as effusion and OA, which are visible on single slices, we expect both to perform similarly, since 2D pretraining is strong. Some findings have very few expert positives, for example MCL with 9 of 58 and lateral OA with 11, so their confidence intervals on the expert set will be wide and conclusions about them will rest on the Kaggle test set. Where reports and images disagree, both models will be limited by label quality rather than architecture, and an early prompt already missed many radiologist positives for MCL, synovitis and effusion. Studies with missing series are masked by both models, but we expect lower accuracy on them and will report them separately.

Overall, we expect a single 3D model to land near, and possibly below, a single WindowNet model, mainly because pretrained 3D encoders are weaker than 2D ImageNet models. Either outcome is informative. If Approach B falls short, the natural next steps are inflated 2D initialization, attention across series and higher resolution. Because combining different model families helped Approach A, we will also test whether an ensemble of both approaches beats either one alone.

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
