# BleedCheck CT | flagging brain bleeds on head CT
**Track:** AI + Health
**Team:** Aarish Asif Khan, Aabir Asif Khan

> This project is a prototype and is not medical advice.

## The problem
When someone arrives at an emergency department with signs of a stroke, the doctor must quickly find out whether a head CT shows a bleed in the brain or a blocked/narrowed blood vessel (ischemia), because they need different treatment. At hospitals or on night shifts without a radiologist on site, that task can fall to a busy emergency doctor. Our user is that doctor, they need the most urgent scans flagged first. [Optional: one sentence of your own motivation.]

## What we built
A model that takes one head CT slice and returns the chance that it shows a bleed (and also the chance of ischemia or a normal scan). Input a brain CT image. Output would be "possible bleed" or "no bleed" with a probability. A doctor would use it to decide which scans to open first, not to make a diagnosis.

## What we added
Our project is our own idea, related to the suggested pneumonia X-ray project. Beyond a basic classifier we added:
- **Our own question:** does it still work on scans from a different source? We tested on a second dataset from a different publisher.
- **A harder "no bleed" class:** ischemia scans count as "no bleed", so the model must tell a bleed from a clot, not just from a healthy brain.
- **Misses and false alarms reported separately**, for each test set.
- **Experiments to reduce false alarms on new data** (augmentation, contrast normalisation, threshold choice). [Fill in after the experiments.]
- A written log of every experiment and mistake: `docs/experiment_log.md`.

## How it works
**Data.** Brain Stroke CT Dataset on Kaggle, a copy of the TEKNOFEST-2021 Turkish Ministry of Health stroke data: 6,650 anonymised head CT slices (4,427 normal, 1,093 bleeding, 1,130 ischemia), split 70/15/15 into 4,655 / 997 / 998 images. The dataset has no patient IDs, so slices from one patient can appear in both training and test; our scores on this data may be optimistic. Two outside sets: `External_Test` (200 slices, same source, "any stroke" label) and Head CT - hemorrhage (200 slices, a different publisher, 100 normal / 100 bleeding).
**Model.** ResNet18 pretrained on ImageNet, fine-tuned for three classes (normal, bleeding, ischemia) with class weights for the rarer classes. We keep the epoch with the best validation AUC. "Bleed" means the bleeding probability is at least [threshold].

## Results

All numbers are for finding **bleeds** (the Bleeding class vs everything else). We report missed bleeds and false alarms separately. Our final model is a plain ResNet18, chosen by a rule we wrote before looking at the test sets. We trained it four times with different seeds and report the mean and range.

| Data | Bleeds caught | False alarms | AUC |
|---|---|---|---|
| Same-source test split (998 slices, 164 bleeds) | 90.9% (88.4–93.3%) | 0.7% (0.2–1.4%) | 0.995 |
| New source: Kitamura, held-out half (50 bleeds, 50 normals), cut-off 0.5 | 91.5% (86–96%) | 15.5% (10–20%) | 0.948 |

The always-"no bleed" baseline scores 83.6% accuracy and catches no bleeds, so accuracy alone says little here.
Same-source scores are an upper bound: the dataset has no patient IDs, so neighbouring slices of one patient may appear in both train and test.

**Where it fails.** On the new source the model catches every bleed but raises many false alarms. [Add your explanation after looking at the images.] [Add one or two example images with what the model said.]
**After improvements:** [fill in the final table: first model vs. final model on all test sets, from the experiment log.]

## Where it fails

- **About 1 in 11 bleeds is missed** even on data like its training data (11–19 of 164 across our four runs).
- **On a new source the cut-off does not transfer.** At 0.5, 10–20% of normal scans from the new source were flagged. Changing only the cut-off did not fix this (false alarms 14–48%).
- **The heatmaps do not show it finding the bleed.** For missed bleeds, the highlighted area sits at the back of the head or on the head-holder; for some normal scans it sits on image corners and text overlays. The model may use cues that are not the bleed.
- Our test on the new source is small (100 scans from one source).

## What we added: our own question

*How many labelled scans does a new hospital need before the tool works on its scans?*
We fine-tuned our model on 10, 20, 40 or 100 labelled scans from the new source and tested on its other half (3 seeds each).

| Local scans | False alarms (new source) | Bleeds caught (new source) | AUC | Bleeds caught (original data) |
|---|---|---|---|---|
| 0 | 20.0% | 92.0% | 0.937 | 88.4% |
| 40 | 1.3% | 94.0% | 0.992 | 78.3% |
| 100 | 1.3% | 95.3% | 0.997 | 86.8% |

With about 100 local scans, false alarms on the new source fell from 20% to about 1%. The same fine-tuning cost recall on the original data (down to ~77% at 20–40 scans, back to 87% at 100). This is one new source and three seeds, so we treat it as a lead, not a guarantee.
Suggested use: a hospital checks the tool on its own labelled scans before relying on it.

## What we'd do next
- Use data with patient IDs, so we can split by patient and remove the leakage risk.
- Test on scans from more hospitals, and re-set the alert threshold on each hospital's own scans before using the tool.
- Check performance by bleed size and by patient age and sex, where the data allows.

## Sources
- Kaggle Brain Stroke CT Dataset (copy of the TEKNOFEST-2021 data): Koç U, Akçapınar Sezer E, Alper Özkaya Y, et al. Eurasian J Med. 2022;54(3):248-258. https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset. Rights: Republic of Türkiye Ministry of Health.
- Kitamura, F. C. (2018). Head CT - hemorrhage (CC0). https://doi.org/10.34740/KAGGLE/DSV/152137
- Tools: PyTorch and torchvision (ResNet18), scikit-learn, pandas, Kaggle notebooks, Gradio.
- AI tools used: Claude, ChatGPT, GitHub Copilot [add Cursor if used]. We read and understood all code before using it.
