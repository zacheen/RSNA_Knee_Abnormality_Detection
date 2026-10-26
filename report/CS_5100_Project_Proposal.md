# Report-supervised multisequence MRI classification for knee abnormality detection

Team members: Yujing Wang, Yu-Chang Shih

## 1. Introduction

Knee disorders are common and can seriously limit mobility and quality of life [1,2]. Knee osteoarthritis alone affected an estimated 654 million people aged 40 years and older worldwide in 2020 [2], and in the Framingham study MRI revealed meniscal tears in 31% of 991 adults aged 50–90 years, often in knees that caused no symptoms [3]. Because magnetic resonance imaging (MRI) depicts the soft tissue and bone of the joint in detail without ionizing radiation, a single examination can show coexisting injuries of the ligaments, menisci, cartilage and bone [1,4].

Interpreting such detailed examinations, however, is demanding. Radiologists must follow structures across slices and reconcile evidence from several planes and sequences, and because subtle lesions can mimic normal anatomical variants, interpretations can vary between readers [1,5]. Deep learning systems can now classify several knee abnormalities, and reader studies show that they can improve diagnostic performance [4,5]. Translating these advances into routine practice, however, faces two challenges. First, performance often degrades when a model meets data from a new institution or acquisition protocol [6]. Second, expert annotation of images is costly to obtain at scale. Radiology reports offer a way around the second challenge: in chest radiography and head CT, labels mined from reports have supported datasets of more than one hundred thousand images and models whose performance has been compared directly with that of radiologists [7–9]. This scale comes at a cost in label quality. Reports often express uncertainty and are written for clinical communication rather than annotation, so labels derived from them are best treated as noisy rather than definitive [8].

Here we ask how far report-derived supervision can take a knee MRI classifier when expert labels are scarce. Using the Radiological Society of North America (RSNA) Knee Abnormality Detection dataset [10], we train a model to detect twelve abnormalities: ligament and meniscal injuries, compartmental osteoarthritis, effusion, synovitis, Baker's cyst, bone contusion and fracture. Large language models can already convert free-text radiology reports in several languages into structured findings [11], and we use one to extract training labels from multilingual reports. These labels are needed only during training; at inference, the classifier receives images and acquisition metadata alone and integrates neighboring slices and multiple sequences through attention specific to each finding [12]. With this system in place, we will determine which design choices account for its performance and test how reliably such effects can be measured when only 58 examinations carry expert labels.

## 2. Previous Work and Contributions

### Related work

MRI analysis is moving from task-specific models towards representations that transfer across anatomical regions and clinical applications. MARS, for example, separates shared anatomical information from acquisition-dependent variation through multisequence pretraining and has been evaluated on 44 tasks spanning diagnosis, segmentation and report generation [13]. In a similar spirit, Hoyer et al. adapted SAM-family segmentation models to five musculoskeletal regions and used the resulting measurements for knee triage and outcome prediction [14]. These studies inform our work mainly through representation learning and adaptation, since detecting individual abnormalities still requires evaluation specific to that task.

Within knee MRI, MRNet set the pattern that later classifiers followed [1]. It combined ImageNet-pretrained slice encoders, max pooling within each series and logistic regression across planes, and, trained on labels extracted manually from reports, it detected general abnormalities, anterior cruciate ligament (ACL) tears and meniscal tears. Its external evaluation carries a lesson that still applies: the area under the receiver operating characteristic curve (AUC) for ACL injury rose from 0.824 to 0.911 only after the model was trained on data from the external institution. MRNet is therefore both a precedent for report-derived supervision and an early example of the gap between institutions that was later documented in chest radiography [6].

CoPAS broadened the task to twelve abnormalities, using three-dimensional convolutional features and attention across planes and sequences [4]. Its five-center cohort of 1,748 patients had arthroscopic reference standards for meniscal, ACL and cartilage lesions and MRI-based labels for other findings, and it reached a mean AUC of 0.812 internally and 0.721–0.726 externally. Although the architecture and reader study support learning how to combine complementary views, the surgical cohort, the exclusion of scans with artifacts and the requirement for all three planes limit its applicability to routine examinations. Its twelve categories also differ from the RSNA targets, so aggregate scores cannot be compared directly.

Xie et al. pursued scale instead, studying 14,962 examinations from 13,419 patients across five centers [5]. Their system located the knee by segmenting the menisci and then applied nine independently trained classifiers, each using attention to focus on relevant regions. In external testing and staged reader studies, model assistance improved diagnostic accuracy and shortened reading times. The reference standard, however, rested on MRI and report review without arthroscopic confirmation, and because the cohorts were retrospective and drawn entirely from China, prospective and international generalizability remains untested. The study nonetheless offers a useful contrast to a single shared model, namely one classifier per finding.

A related line of work predicts future outcomes rather than current findings. MR-Transformer adapted an ImageNet-pretrained vision transformer to volumetric MRI through positional embeddings and attention across patches [15]. When tested on predicting total knee replacement, using sevenfold nested cross-validation in matched cohorts from the Osteoarthritis Initiative and the Multicenter Osteoarthritis Study, it achieved AUCs of 0.84–0.88 across four sequences. These results did not differ significantly from those of MRNet, even though MR-Transformer requires far more computation; this argues for weighing computational cost alongside discrimination and motivates our choice of compact pretrained encoders. Its prognostic endpoint and its mostly older, overweight and White cohorts, however, limit extrapolation to abnormality detection.

Clinical text offers a further source of supervision, and it has been used in several ways. Generalist vision–language models such as BiomedGPT model images and language jointly across biomedical tasks, although reported generation errors and uneven transfer between modalities show that each task still needs its own validation [16]. Self-supervised training on paired chest X-rays and reports can even reach expert-level classification without explicit labels [17]. Most large imaging datasets, by contrast, convert reports into explicit labels, first with rule-based labelers [7–9] and more recently with large language models [11]. We follow this last route: a language model turns reports into training targets, so no text encoder is needed at inference. To aggregate many images under a single examination-level label, we build on attention-based multiple instance learning [18]. Self-supervised image representations such as DINOv2 [19] offer another route to transfer, but our current system uses ImageNet-pretrained convolutional backbones. Our approach thus combines report-derived supervision with efficient integration of multiple MRI sequences.

### Proposed contributions and methodology

Of the 4,407 examinations in the dataset, 58 carry radiologist labels. We reserve these for local evaluation and train on the remaining 4,349, which are labeled from reports alone [12]. The system began as KneeNet, a slice encoder with attention pooling, and has since evolved into WindowNet, which combines neighboring slices and attends across sequences. Our contribution is an empirical account of how report-derived supervision, the construction of model inputs and diversity among ensemble members behave when expert annotation and computing resources are both limited.

Image representation and integration. WindowNet reads up to five series: fluid-sensitive sagittal, coronal and axial images, together with non-fat-suppressed sagittal and coronal images. In the principal configuration, 20 slices are sampled from between 2% and 98% of each stack, cropped to the central 128 mm using the DICOM pixel spacing and stored at 320 × 320 pixels. Each run of three consecutive slices then forms a three-channel window for an ImageNet-pretrained encoder; because consecutive windows share two slices, each series yields 18 windows. Series and relative-position embeddings are added to the window features, after which a separate attention query for each finding pools over all windows while masking absent series. During training, 32 windows per study are chosen at random and a random 288 × 288 region is cropped, whereas at inference all windows are used with a central 288 × 288 crop and predictions are averaged with those for the left–right mirrored image. A full-frame variant instead uses the entire 320 × 320 image.

Rather than repeating one configuration, our best ensemble combines ten members of different kinds. Its five EfficientNet [20] members comprise two EfficientNet-B0 models in the principal configuration trained with different random seeds, one EfficientNet-B0 trained on pseudo-labels, one EfficientNet-B2 and one full-frame EfficientNet-B0. They are joined by four EfficientNetV2-B0 models [21] and one ConvNeXt-Nano model [22], and the configuration of every member is recorded in the project repository [12]. Averaging independently trained networks in this way is known to improve both accuracy and calibration [23].

Report-derived supervision. A language model converts the multilingual reports into soft targets for the twelve findings, preserving graded and equivocal language rather than forcing a binary decision. Findings that a report does not mention are treated as missing and excluded from the binary cross-entropy loss. The principal label set averages our refined labels with seven public label sets, ignoring missing entries. Training fine-tunes the encoder and attention layers with AdamW and a one-cycle learning-rate schedule, and predictions from the final three epochs are averaged. We also compare hard targets with two forms of pseudo-label, each an average of report targets and model predictions. In-sample pseudo-labels come from models trained on the same studies and therefore act as smoothed labels, whereas out-of-fold pseudo-labels come from fivefold predictions for studies withheld from each model [12]. All of these are training signals generated by models, not calibrated clinical probabilities.

Evaluation. We keep three measures distinct: agreement between report-derived and expert labels, image-model performance on the 58 expert-labeled studies, and AUC on the public leaderboard. The expert labels were never used to train the image models, but they did guide prompt development and many model comparisons, so the 58 studies serve as a development set rather than an independent test set. Small evaluation sets of this kind are a recognized source of over-optimistic and unstable results in medical imaging [24]. To limit this, prompt development used only 29 of the studies, and the other 29 were withheld for evaluation. In the fivefold pseudo-label experiments, folds are assigned from a hash of the lowercased report text, so the 183 studies that share a report with another study fall in the same fold; this does not, however, guarantee independence at the patient level. We will report macro and per-finding AUC together with the number of positive studies for each finding, treating per-finding AUC as descriptive where positives are few. Finally, the leaderboard itself becomes less informative with repeated use, because each submission reveals information about the hidden test set [25]. Although an analysis of more than one hundred Kaggle competitions found such overfitting to be modest in practice [26], small gains on a heavily queried leaderboard should still be interpreted with caution [12].

## 3. Expected Results and Preliminary Results

The ten-member WindowNet ensemble reaches a public leaderboard macro AUC of 0.944 [12]. Most of this performance came from a single change of architecture: a ten-member KneeNet ensemble scored 0.893, whereas the first three-member WindowNet ensemble scored 0.939. Because that transition altered series selection, physical cropping, neighboring-slice input and attention at once, the gain cannot be credited to any one of them. Subsequent changes added little. Adding four EfficientNetV2-B0 members and one ConvNeXt-Nano member to the five EfficientNet members raised the score from 0.943 to 0.944, and enlarging the ensemble to sixteen members left it unchanged. Moreover, since the EfficientNet group already contains a pseudo-label member and a full-frame member, the 0.944 ensemble does not isolate any single configuration. For context, the leading public score was 0.966 when the project first reached the leaderboard. Our models use no external imaging data, although the blended labels draw on seven public label sets. These figures are public leaderboard results recorded in the repository; they have not been independently reproduced and are not evidence of clinical benefit.

Report-derived labels, in turn, improved substantially with prompt design. On the 29 studies withheld from prompt development, macro AUC rose from 0.6496 with keyword rules to 0.8065 with soft targets graded by severity, and then to 0.8262 with the refined prompt used for the final labels. Across all 58 expert-labeled studies, averaging the refined labels with seven public label sets further raised agreement from 0.8655 to 0.8923 [12]. Better labels did not, however, reliably translate into better image models.

A more fundamental difficulty is that variation between random seeds on the 58 studies is as large as most of the method effects we set out to measure, which echoes broader evidence that variance from seeds and data sampling can overturn conclusions drawn from single runs [27]. A full-frame EfficientNetV2-B0, for instance, reached a local AUC of 0.9301, the best single-model score in the project, yet three further seeds of the same configuration scored only 0.9026, 0.9062 and 0.9120. On the leaderboard the same model scored 0.931, below the 0.938 of a single EfficientNet-B0 in the principal configuration, and local and leaderboard rankings diverged in several other comparisons as well. Measuring small effects reliably is therefore itself a central problem for this project.

The evidence so far suggests that local slice context and complementary sequences are useful, but it does not establish the separate contribution of each, and neither out-of-fold pseudo-labels nor larger ensembles showed a clear advantage. We will therefore run a controlled ablation that starts from a single EfficientNet-B0 in the principal configuration and changes one factor at a time:

- series: all five series versus the three fluid-sensitive series alone;
- cropping: the 128-mm physical crop versus the full frame;
- slice context: three-slice windows versus single slices;
- labels: blended soft targets versus the refined labels alone versus hard targets.

To separate these effects from seed noise, each configuration will be trained with the same three seeds. We will then report the mean and standard deviation of macro AUC across seeds on the 58 expert-labeled studies, together with a paired bootstrap 95% confidence interval for each difference from the reference configuration.

The seed variation observed so far, between 0.01 and 0.03 in macro AUC, implies that differences smaller than about 0.02 cannot be resolved on this cohort. We will therefore estimate the detectable difference from the new runs and state it before interpreting any comparison. Leaderboard submissions will be reserved for single-change tests of the largest local effects, and errors will be examined finding by finding. Beyond this project, resolving smaller differences will require a larger independent evaluation cohort, and any claim of clinical utility would require external validation and reader studies, reported according to current guidance for clinical prediction models that use machine learning [28].

## References

[1\. Bien, N. et al. Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet. PLOS Med. 15, e1002699 (2018).](https://doi.org/10.1371/journal.pmed.1002699)

[2\. Cui, A. et al. Global, regional prevalence, incidence and risk factors of knee osteoarthritis in population-based studies. EClinicalMedicine 29–30, 100587 (2020).](https://doi.org/10.1016/j.eclinm.2020.100587)

[3\. Englund, M. et al. Incidental meniscal findings on knee MRI in middle-aged and elderly persons. N. Engl. J. Med. 359, 1108–1115 (2008).](https://doi.org/10.1056/NEJMoa0800777)

[4\. Qiu, Z. et al. Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities. Nat. Commun. 15, 7637 (2024).](https://doi.org/10.1038/s41467-024-51888-4)

[5\. Xie, Z. et al. Development of a multi-task deep learning system for classification of nine common knee abnormalities on MRI: a large-scale, multicentre, stepwise validation study. EClinicalMedicine 89, 103534 (2025).](https://doi.org/10.1016/j.eclinm.2025.103534)

[6\. Zech, J. R. et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. PLOS Med. 15, e1002683 (2018).](https://doi.org/10.1371/journal.pmed.1002683)

[7\. Wang, X. et al. ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. 3462–3471 (2017).](https://doi.org/10.1109/CVPR.2017.369)

[8\. Irvin, J. et al. CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. Proc. AAAI Conf. Artif. Intell. 33, 590–597 (2019).](https://doi.org/10.1609/aaai.v33i01.3301590)

[9\. Chilamkurthy, S. et al. Deep learning algorithms for detection of critical findings in head CT scans: a retrospective study. Lancet 392, 2388–2396 (2018).](https://doi.org/10.1016/S0140-6736(18)31645-3)

[10\. Radiological Society of North America. RSNA Knee Abnormality Detection AI Challenge (2026).](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)

[11\. Adams, L. C. et al. Leveraging GPT-4 for post hoc transformation of free-text radiology reports into structured reporting: a multilingual feasibility study. Radiology 307, e230725 (2023).](https://doi.org/10.1148/radiol.230725)

[12\. RSNA\_Knee\_Abnormality\_Detection project repository. Source code and experiment records, commit 6087926 (2026).](https://github.com/zacheen/RSNA_Knee_Abnormality_Detection/tree/60879260572e6b195b24a92ff246b0a79168e582)

[13\. Qiu, Z. et al. Large-scale multi-sequence pretraining for generalizable MRI analysis in versatile clinical applications. Nat. Biomed. Eng. (2026).](https://doi.org/10.1038/s41551-026-01740-5)

[14\. Hoyer, G. et al. Clinical utility of foundation models in musculoskeletal MRI for biomarker fidelity and predictive outcomes. npj Digit. Med. 9, 383 (2026).](https://doi.org/10.1038/s41746-026-02520-w)

[15\. Zhang, C. et al. MR-Transformer: A Vision Transformer-based Deep Learning Model for Total Knee Replacement Prediction Using MRI. Radiol. Artif. Intell. 7, e240373 (2025).](https://doi.org/10.1148/ryai.240373)

[16\. Zhang, K. et al. A generalist vision–language foundation model for diverse biomedical tasks. Nat. Med. 30, 3129–3141 (2024).](https://doi.org/10.1038/s41591-024-03185-2)

[17\. Tiu, E. et al. Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning. Nat. Biomed. Eng. 6, 1399–1406 (2022).](https://doi.org/10.1038/s41551-022-00936-9)

[18\. Ilse, M., Tomczak, J. & Welling, M. Attention-based Deep Multiple Instance Learning. Proc. Mach. Learn. Res. 80, 2127–2136 (2018).](https://proceedings.mlr.press/v80/ilse18a.html)

[19\. Oquab, M. et al. DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193 (2023; revised 2024).](https://arxiv.org/abs/2304.07193)

[20\. Tan, M. & Le, Q. EfficientNet: Rethinking model scaling for convolutional neural networks. Proc. Mach. Learn. Res. 97, 6105–6114 (2019).](https://proceedings.mlr.press/v97/tan19a.html)

[21\. Tan, M. & Le, Q. EfficientNetV2: Smaller models and faster training. Proc. Mach. Learn. Res. 139, 10096–10106 (2021).](https://proceedings.mlr.press/v139/tan21a.html)

[22\. Liu, Z. et al. A ConvNet for the 2020s. Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. 11966–11976 (2022).](https://doi.org/10.1109/CVPR52688.2022.01167)

[23\. Lakshminarayanan, B., Pritzel, A. & Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles. Adv. Neural Inf. Process. Syst. 30 (2017).](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)

[24\. Varoquaux, G. & Cheplygina, V. Machine learning for medical imaging: methodological failures and recommendations for the future. npj Digit. Med. 5, 48 (2022).](https://doi.org/10.1038/s41746-022-00592-y)

[25\. Dwork, C. et al. The reusable holdout: Preserving validity in adaptive data analysis. Science 349, 636–638 (2015).](https://doi.org/10.1126/science.aaa9375)

[26\. Roelofs, R. et al. A meta-analysis of overfitting in machine learning. Adv. Neural Inf. Process. Syst. 32, 9179–9189 (2019).](https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning)

[27\. Bouthillier, X. et al. Accounting for variance in machine learning benchmarks. Proc. Mach. Learn. Syst. 3 (2021).](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html)

[28\. Collins, G. S. et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 385, e078378 (2024).](https://doi.org/10.1136/bmj-2023-078378)
