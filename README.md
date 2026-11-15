# BleedCheck (working name): flagging brain bleeds on head CT
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
Validation set (997 slices, 164 bleeds). Baselines vs. first model:
| | Accuracy | Bleeds caught | Missed bleeds | False alarms (833 non-bleeds) |
|---|---|---|---|---|
| Always "no bleed" | 83.6% | 0% | 164 | 0 |
| Pixel logistic regression | 82.5% | 74.4% | 42 | 132 |
| ResNet18 (first model) | 97.2% | 92.7% | 12 | 16 |

Tests on other data (first model):
| Test set | Result |
|---|---|
| External_Test, same source (200) | caught 59 of 70 strokes, 12 false alarms on 130 normals |
| Kitamura, different source (200) | caught 100 of 100 bleeds, but flagged 59 of 100 normal scans |

**Where it fails.** On the new source the model catches every bleed but raises many false alarms. [Add your explanation after looking at the images.] [Add one or two example images with what the model said.]
**After improvements:** [fill in the final table: first model vs. final model on all test sets, from the experiment log.]

## What we'd do next
- Use data with patient IDs, so we can split by patient and remove the leakage risk.
- Test on scans from more hospitals, and re-set the alert threshold on each hospital's own scans before using the tool.
- Check performance by bleed size and by patient age and sex, where the data allows.

## Sources
- Kaggle Brain Stroke CT Dataset (copy of the TEKNOFEST-2021 data): Koç U, Akçapınar Sezer E, Alper Özkaya Y, et al. Eurasian J Med. 2022;54(3):248-258. https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset. Rights: Republic of Türkiye Ministry of Health.
- Kitamura, F. C. (2018). Head CT - hemorrhage (CC0). https://doi.org/10.34740/KAGGLE/DSV/152137
- Tools: PyTorch and torchvision (ResNet18), scikit-learn, pandas, Kaggle notebooks, Gradio.
- AI tools used: Claude, ChatGPT, GitHub Copilot [add Cursor if used]. We read and understood all code before using it.
