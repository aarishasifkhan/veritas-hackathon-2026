# Experiment log

Everything we tried, in order, including what went wrong. Numbers are for finding **bleeds** (Bleeding vs Normal + Ischemia). "Missed" means a bleed the model did not flag; "false alarm" means a non-bleed it flagged.

## Data and splits
- Main data: Brain Stroke CT Dataset (Kaggle, ozguraslank), 6,650 slices: 4,427 Normal, 1,093 Bleeding, 1,130 Ischemia. No patient IDs, so same-source scores are an upper bound.
- Split 70/15/15, stratified by class, seed 42: train 4,655, validation 997 (164 bleeds), test 998 (164 bleeds).
- Second source (new hospital stand-in): Kitamura "Head CT - hemorrhage" (CC0), 100 normal and 100 hemorrhage. Split 50/50 into DEV (used to compare experiments and pick cut-offs) and FINAL (looked at once at the end).
- External_Test (same source as the main data, 200 slices, 0 ID overlap): Stroke yes/no only.

## 1. Baselines
| Model | Accuracy | Bleeds caught | Notes |
|---|---|---|---|
| Always "no bleed" | 83.6% | 0% | accuracy alone is misleading |
| Pixel logistic regression | 82.5% | 74.4% (42 missed) | 132 false alarms of 833 |

## 2. First ResNet18 (notebook 01)
- Pretrained ResNet18 (ImageNet), fine-tuned to 3 classes with class-weighted loss.
- Validation: 97.2% accuracy, 92.7% bleeds caught (12 of 164 missed), 16 false alarms.
- External_Test: 59 of 70 strokes caught (84.3%), 12 of 130 false alarms.
- Kitamura (all 200, first look): 100 of 100 bleeds caught, **59 of 100 normals flagged**.
- Lesson: it works on its own source but raised false alarms on a new source.

## 3. Single-seed experiments (seed 42)
| | Val bleeds caught | Val false-alarm | Val AUC | ExtTest stroke caught | ExtTest false-alarm | KitDev caught | KitDev false-alarm | KitDev AUC |
|---|---|---|---|---|---|---|---|---|
| E0 plain | 0.927 | 0.010 | 0.995 | 0.843 | 0.092 | 0.920 | 0.160 | 0.945 |
| E1 augmentation | 0.890 | 0.007 | n/a | 0.843 | 0.069 | 1.000 | 0.180 | 0.992 |
| E2 augmentation + contrast stretch | 0.945 | 0.012 | n/a | 0.957 | 0.362 | 0.940 | 0.140 | 0.972 |

- The first-run Kitamura false-alarm rate (59%) did not repeat on re-run (E0: 16% on KitDev). Same recipe, very different results, so we ran several seeds.

## 4. Decision rule
- Original rule: pick the best KitDev AUC. Problem found: it ignored that E2 made same-source false alarms much worse (ExtTest 0.362).
- **Amended rule (written before the seed runs):** choose the configuration with the best average KitDev AUC among those whose average validation bleed recall is within 2 points of E0's and whose average ExtTest false-alarm rate is not more than 5 points above E0's.

## 5. Three seeds per configuration (mean, min-max)
| | E0 plain | E1 augmentation | E2 aug + contrast |
|---|---|---|---|
| Val bleeds caught | 0.876 (0.848-0.896) | 0.927 (0.909-0.945) | 0.900 (0.854-0.945) |
| Val false-alarm | 0.001 | 0.012 | 0.008 |
| Val AUC | 0.995 | 0.992 | 0.993 |
| ExtTest stroke caught | 0.824 | 0.895 | 0.890 |
| ExtTest false-alarm | 0.023 (0.015-0.031) | 0.154 (0.077-0.208) | 0.123 (0.046-0.223) |
| KitDev bleeds caught | 0.927 | 0.980 | 0.927 |
| KitDev false-alarm | 0.160 (0.080-0.280) | 0.253 (0.120-0.380) | 0.227 (0.100-0.320) |
| KitDev AUC | 0.952 (0.939-0.966) | 0.962 (0.940-0.973) | 0.938 (0.883-0.977) |

- Result of the rule: only E0 passes (E1 and E2 are more than 5 points above E0's ExtTest false-alarm rate). **Chosen: E0, plain ResNet18.**
- Honest conclusion: augmentation and contrast stretching shifted the trade-off (more bleeds caught, more false alarms) but did not improve AUC, and the run-to-run range was as large as the differences.
- Demo model: E0_plain_s3 (highest validation AUC, 0.996, chosen on validation only).

## 6. Process mistake we made
- Cell H (test split + Kitamura FINAL) was first run on the placeholder model E1_aug_s1, before the choice was made. Its numbers are reported as E1_aug_s1's, not the final model's:
  - Test, cut-off 0.5: 87.2% caught (21 missed), 11 of 834 false alarms (1.3%).
  - Test, validation-chosen cut-off 0.214: 89.6% caught (17 missed), 35 of 834 false alarms (4.2%). The validation target of 95% was not reached on the test split, so validation numbers are optimistic.
  - Kitamura FINAL, 0.5: 100% caught, 15 of 50 false alarms (30%).
  - Kitamura FINAL, cut-off recalibrated on DEV (0.829): 96% caught, 11 of 50 false alarms (22%). AUC 0.94.
- We then evaluated the rule-chosen model (E0) and did not switch afterwards.

## 7. Final evaluation: four plain-ResNet runs (first look at test and Kitamura FINAL)
| | E0_plain | s1 | s2 | s3 (demo) | Mean (min-max) |
|---|---|---|---|---|---|
| Test bleeds caught @0.5 | 0.933 | 0.896 | 0.921 | 0.884 | 0.909 (0.884-0.933) |
| Test false-alarm @0.5 | 0.014 | 0.007 | 0.004 | 0.002 | 0.007 (0.002-0.014) |
| Test AUC | 0.992 | 0.995 | 0.996 | 0.996 | 0.995 |
| Kit bleeds caught @0.5 | 0.960 | 0.920 | 0.860 | 0.920 | 0.915 (0.860-0.960) |
| Kit false-alarm @0.5 | 0.180 | 0.140 | 0.100 | 0.200 | 0.155 (0.100-0.200) |
| Kit AUC | 0.945 | 0.958 | 0.951 | 0.937 | 0.948 |
| Kit recalibrated caught | 1.000 | 1.000 | 0.880 | 0.960 | 0.960 (0.880-1.000) |
| Kit recalibrated false-alarm | 0.480 | 0.360 | 0.140 | 0.220 | 0.300 (0.140-0.480) |

- Not overfitted in the usual sense: validation and test AUC agree (0.995).
- Cut-off-only recalibration was unreliable (false alarms 14-48%).
- The demo model s3 was the worst of the four on Kitamura false alarms. We kept it because we chose on validation only.

## 8. Our own question: how many local scans does a new hospital need?
Fine-tuned E0_plain_s3 on n labelled Kitamura-DEV scans (5 epochs, lr 1e-4), tested on Kitamura FINAL and the validation split. 3 seeds per n (n=0 is one run). Mean (min-max):

| Local scans | Kit caught | Kit false-alarm | Kit AUC | Main-source caught | Main-source false-alarm |
|---|---|---|---|---|---|
| 0 | 0.920 | 0.200 | 0.937 | 0.884 | 0.002 |
| 10 | 0.873 (0.840-0.900) | 0.067 (0.060-0.080) | 0.968 | 0.811 (0.774-0.835) | 0.001 |
| 20 | 0.913 (0.880-0.940) | 0.053 (0.040-0.080) | 0.981 | 0.774 (0.726-0.823) | 0.000 |
| 40 | 0.940 (0.900-0.960) | 0.013 (0.000-0.040) | 0.992 | 0.783 (0.756-0.823) | 0.000 |
| 100 | 0.953 (0.940-0.980) | 0.013 (0.000-0.020) | 0.997 | 0.868 (0.829-0.915) | 0.013 (0.006-0.024) |

- With ~100 local scans, new-source false alarms fell from 20% to about 1% and AUC rose to 0.997.
- Cost: recall on the original data dropped to ~77-78% at 20-40 scans and recovered to 87% at 100.
- Caveats: one new source, three seeds, 100 test scans.

## 9. Grad-CAM heatmaps (E0_plain_s3)
- Caught bleeds: a broad glow over the brain and the skull ring.
- Missed bleeds: heat at the back or bottom of the head and on the head-holder, not clearly on the brain.
- Correctly cleared normals: heat on image corners and overlay text.
- New-source false alarms: heat on the skull rim and posterior area.
- Conclusion: the heatmaps do not show the model finding the bleed; it may use non-bleed cues. Listed as a limitation.

## Problems we hit (and fixes)
- medmnist install failed (Python version out of range): used a Python 3.12 environment, then moved the practice to Kaggle.
- GitHub "nothing to compare" / no pull request: a pushed branch alone does not create a PR; make a real commit and open the PR manually.
- PhysioNet / AWS registry mix-up: the AWS entry was an unrelated, deprecated ECG registry.
- Hugging Face dataset: 429 rate limit and no labels; switched to the Kaggle original.
- Wrong Kitamura input (a notebook, not the dataset): re-created the notebook from the dataset page.
- README draft had wrong numbers and a "tumor" claim: corrected.
- Decision rule was incomplete (ignored same-source false alarms): amended before the seed runs.
- Cell H run on the placeholder model: documented in section 6.
- Single runs were misleading (59% vs 16% false alarms on the same recipe): switched to several seeds.
- Demo hosting: we built the first demo in Gradio and planned to host it on Hugging Face Spaces. Spaces asked for a paid plan for the Gradio SDK, which we could not afford. We rebuilt the demo in Streamlit and deployed it free on Streamlit Community Cloud. The model and results did not change, only the interface.
