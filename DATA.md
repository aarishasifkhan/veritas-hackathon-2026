# Datasets

BleedCheck CT uses two public datasets, which we use as three separate sets. We never commit images to GitHub, and we never redistribute them (except a few CC0 example scans for the demo).

| Role | Dataset | Size | Used for |
| --- | --- | --- | --- |
| Main data | Brain Stroke CT Dataset (Kaggle) | 6,650 slices | Training, validation and same-source test |
| Same-source check | `External_Test` folder, same Kaggle dataset | 200 slices | Development check ("any stroke detected"), used in our model-choice rule |
| New source | Head CT - hemorrhage (Kitamura) | 200 slices | DEV half (100): comparing experiments, cut-off choice, local fine-tuning. FINAL half (100): one-time external test |

## 1. Main dataset (training, validation, test)

- **Name:** Brain Stroke CT Dataset (Kaggle), a copy of the TEKNOFEST-2021 Stroke data set.
- **Link:** https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset
- **Original source:** Republic of Türkiye Ministry of Health open data portal, https://acikveri.saglik.gov.tr/Home/DataSetDetail/1 (as the Kaggle page states).
- **Terms:** The Kaggle page says "All legal rights belong to the Republic of Türkiye Ministry of Health" and names no standard open licence. We use the data for a non-commercial hackathon prototype.
- **Citation:** Koç U, Akçapınar Sezer E, Alper Özkaya Y, et al. Artificial intelligence in healthcare competition (TEKNOFEST-2021): Stroke data set. *Eurasian J Med.* 2022;54(3):248-258.
- **Contents:** Anonymised head CT slices recorded in 2019–2020 and annotated by 7 radiologists. The original description lists 6,653 slices (4,428 no stroke, 1,131 ischemia, 1,094 bleeding). The copy we used has 6,650 slices in its class folders.

| Class | Slices |
| --- | --- |
| Normal (no stroke) | 4,427 |
| Ischemia | 1,130 |
| Bleeding | 1,093 |

- **Our split:** 70/15/15 (stratified by class) into 4,655 training, 997 validation and 998 test slices. The test split holds 164 bleeds. "Bleed" means the Bleeding class; "no bleed" means Normal or Ischemia.
- **Patient IDs (limitation):** The files have none. The DICOM headers show PatientID, StudyInstanceUID and SeriesInstanceUID as MISSING, so we cannot split by patient. Slices from one patient may appear in both training and test, which can make same-source scores look better than they are.
- **Age and sex:** We did not use either, so we could not check performance across age or sex groups.
- **Not used:** A Hugging Face copy (`iraqigold/brain-stroke-ct-dataset`). It had no labels and rate-limited our downloads.

## 2. Same-source held-out set

- **Where:** The `External_Test` folder in the same Kaggle dataset.
- **Contents:** 200 slices with a "Stroke" label of 0 or 1 (130 / 70). It does not separate bleeding from ischemia, so we score a slice as "stroke" when 1 minus P(Normal) is at least 0.5.
- **Overlap check:** None of its IDs appear in the main data (0 of 200).
- **Use:** A same-source development check. We used it, with validation and Kitamura DEV, to compare recipes and choose the final one. It is not part of the final test.

## 3. Second source (different publisher)

- **Name:** Head CT - hemorrhage, Felipe Campos Kitamura (Kaggle, 2018).
- **Link:** https://doi.org/10.34740/kaggle/dsv/152137
- **Licence:** CC0 (public domain).
- **Citation:** Kitamura, F. C. (2018). Head CT - hemorrhage. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/152137
- **Contents:** 200 slices, balanced at 100 normal and 100 hemorrhage. No patient IDs.
- **Use:** We split it 50/50, stratified by label, into two halves.
  - **DEV (100 slices):** compare experiments, choose the final recipe, choose the recalibrated cut-off, and supply the 10 to 100 local scans for the fine-tuning experiment.
  - **FINAL (100 slices, 50 bleed / 50 normal):** looked at once, after the recipe was chosen. It is the external test and the test set for the fine-tuning experiment.
