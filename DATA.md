# Datasets

## Main dataset (training, validation, test)
- Name: Brain Stroke CT Dataset (Kaggle), a copy of the TEKNOFEST-2021 Stroke data set
- Link: https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset
- Original source: Republic of Türkiye Ministry of Health open data portal, https://acikveri.saglik.gov.tr/Home/DataSetDetail/1 (as stated on the Kaggle page)
- Terms: the Kaggle page states "All legal rights belong to the Republic of Türkiye Ministry of Health" and names no standard open licence. We use it for a non-commercial hackathon prototype and do not redistribute the images (data is never committed to GitHub).
- Citation: Koç U, Akçapınar Sezer E, Alper Özkaya Y, et al. Artificial intelligence in healthcare competition (TEKNOFEST-2021): Stroke data set. Eurasian J Med. 2022;54(3):248-258.
- Contents: anonymised head CT slices recorded 2019-2020, annotated by 7 radiologists. Original description: 6,653 slices (4,428 no stroke, 1,131 ischemia, 1,094 bleeding). The copy we used has 6,650 in the class folders (4,427 / 1,130 / 1,093).
- Patient IDs: none. DICOM headers show PatientID, StudyInstanceUID and SeriesInstanceUID as MISSING, so a patient-level split is not possible (limitation).
- Not used: a Hugging Face copy (iraqigold/brain-stroke-ct-dataset), because it had no labels and was rate-limited.

## Same-source held-out set
- External_Test folder in the same Kaggle dataset: 200 slices, label "Stroke" 0/1 (130 / 70), no bleed vs ischemia type. None of its IDs appear in the main data (0 of 200).
- Used only for final evaluation, as "any stroke detected".

## Second source (different publisher)
- Name: Head CT - hemorrhage, Felipe Campos Kitamura (Kaggle, 2018)
- Link: https://doi.org/10.34740/kaggle/dsv/152137
- Licence: CC0 (public domain)
- Citation: Kitamura, F. C. (2018). Head CT - hemorrhage. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/152137
- Contents: 200 slices, balanced 100 normal / 100 hemorrhage. Used only as an external test.
