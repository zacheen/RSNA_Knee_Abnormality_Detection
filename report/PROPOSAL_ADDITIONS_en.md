# Additional experiments for the proposal

Text to add under "Proposed contributions and methodology". Each paragraph states the experiment and the reasoning that motivated it, without results.

**Graded soft labels from reports.** Radiology reports rarely state a finding as simply present or absent. They describe degree and certainty, such as a small joint effusion, a possible tear or mild degenerative change. A three-state label of present, absent or not mentioned discards this gradation, so we will compare it against probabilities assigned according to the described severity and certainty, with findings that are not mentioned excluded from the loss.

**Blending independent label sources.** A single LLM extraction makes systematic errors on ambiguous wording, but independently produced label sets are unlikely to make the same errors on the same reports. We will therefore average our extraction with publicly released label sets cell by cell, on the hypothesis that partly independent errors cancel and the blend agrees better with expert reading than any single source.

**Adjacent slices as color channels.** ImageNet-pretrained encoders expect three input channels, and a single grayscale slice gives them no information about the neighbouring anatomy. Stacking three adjacent slices into one RGB window supplies local through-plane context at the cost of a 2D encoder rather than a 3D one, so we will compare windowed input against single-slice input with the same per-finding attention over all windows.

**Cropping by physical size.** Field of view and pixel spacing vary across scanners and protocols, so a fixed pixel resize maps the same structure to different sizes in different examinations. We will crop a fixed physical extent around the knee using the DICOM pixel spacing before resizing, so that anatomy appears at a consistent scale, and we will vary that extent (128 mm and 160 mm) to test whether a tighter crop cuts off peripheral structures such as the collateral ligaments.

**Which series to include.** Fluid-sensitive, fat-suppressed series show edema, effusion and tears most clearly, while non-fat-suppressed series may show bone and degenerative change better. We will compare three fat-suppressed slots against five slots that add the non-fat-suppressed sagittal and coronal series, and a third arm that moves the same slice budget onto the fat-suppressed series alone, because an external ablation reported that the non-fat-suppressed series contributed little.

**Left-right mirroring.** Left and right knees are mirror images, and medial and lateral labels refer to anatomy rather than to image sides, so a mirrored left knee is a valid right knee with the same labels. We will use plane-aware mirroring as training augmentation and test-time averaging, flipping coronal and axial slices in width and sagittal stacks in depth.

**Pseudo labels from the model.** A model trained on thousands of reports may correct some of the noise in an individual report. We will retrain on a 50/50 blend of report labels and model predictions, comparing in-sample predictions with out-of-fold predictions from five folds grouped by report text, since in-sample predictions risk reproducing memorized labels.

**Soft against hard targets.** Earlier comparisons of soft and hard labels changed the learning rate and regularization at the same time. We will repeat the comparison with only the targets changed, thresholding the blended labels at 0.5 while keeping missing findings excluded.

**Order-aware pooling.** Attention pooling treats the slices of a series as an unordered set, although the extent of a lesion across consecutive slices is diagnostic. We will add a residual one-dimensional convolution over each series' windows before the attention, initialized to zero so that training starts from plain attention pooling.

**Regularization and capacity.** Small labelled datasets often reward regularization, while subtle findings may need more capacity. We will test dropout inside the finding-attention scorer, longer training, a larger effective batch through gradient accumulation, higher input resolution, more windows per training step, and larger backbones of the same family.

**Ensembles across architecture families.** Models from the same recipe with different seeds tend to make correlated errors, whereas different architectures attend to different image features. We will compare adding seeds of one recipe against adding members from other families (EfficientNet, EfficientNetV2 and ConvNeXt).

**Combining related findings.** Some findings co-occur for anatomical reasons, such as synovitis with effusion, and the weakest findings are those whose report labels agree least with expert reading. We will test whether blending the prediction for a weak finding with the prediction for a related, well-learned finding improves its ranking, with blend weights chosen on one half of the expert-labelled studies and scored on the other.

**Evaluation under a small expert set.** With only 58 expert-labelled studies, differences between configurations can be smaller than the variation caused by the random seed alone. We will therefore repeat key configurations across seeds, use five-fold cross-validation on the report-labelled studies as a second estimate, and, where the competition allows, submit single changes to the held-out leaderboard so that each submission tests one variable.
