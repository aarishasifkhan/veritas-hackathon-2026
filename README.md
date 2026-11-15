# BleedCheck CT: a second look at head CT scans

**Track:** AI + Health

**Team:** Aarish Asif Khan, Aabir Asif Khan

**Live demo:** https://bleedcheck-ct.streamlit.app

**Notebook with outputs:** https://www.kaggle.com/code/aarishasifkhan/bleedcheck-ct-a-second-look-at-head-ct-scans

> This project is a prototype and is not medical advice.

## The problem

When a patient arrives with stroke symptoms, the emergency doctor must decide fast whether the head CT shows a **bleed** in the brain or a **blocked vessel** (ischemia). The two need very different treatment, and the wrong one can harm the patient. On night shifts and in hospitals without a radiologist, a busy emergency doctor often makes this call alone. That doctor needs the most urgent scans flagged first.

## What we built

BleedCheck CT reads one head CT slice and returns three probabilities: bleeding, ischemia and normal. It then shows one of two messages: **"Possible bleed"** (bleeding probability of 50% or higher) or **"No bleed flagged"**. The doctor uses it to choose which scans to open first. The tool does not diagnose anyone.

## What we added

Our project is our own idea, related to the suggested pneumonia X-ray project. We added:

- **Our own question.** Does the tool still work on scans from a different hospital source, and how many local scans does it need to adapt?
- **A harder "no bleed" class.** Ischemia scans count as "no bleed", so the model must tell a bleed from a clot, not only from a healthy brain.
- **Two kinds of error, reported separately.** We report missed bleeds and false alarms for every test set.
- **A second data source.** We tested on 200 scans from a different publisher.
- **A written log** of every experiment and mistake: `docs/experiment_log.md`.

## How it works

**Data.** We trained on the Brain Stroke CT Dataset from Kaggle, a copy of the TEKNOFEST-2021 stroke data from the Turkish Ministry of Health. It holds 6,650 anonymised head CT slices (4,427 normal, 1,093 bleeding, 1,130 ischemia). We split it 70/15/15 into 4,655 training, 997 validation and 998 test images. The data has no patient IDs, so slices from one patient can land in both training and test. Our same-source scores may therefore look better than they should. Full details are in [DATA.md](DATA.md).

**Model.** We fine-tuned a ResNet18 pretrained on ImageNet to sort slices into three classes. Class weights help it learn the rarer bleeding and ischemia classes. We keep the epoch with the best validation AUC. We chose this plain ResNet18 by a rule we wrote before looking at any test set. We trained it four times with different random seeds and report the mean and the range.

## Results

We measure how well the model finds **bleeds** (bleeding vs. everything else).

- **Bleeds caught** is the share of real bleeds the tool flags.
- **False alarms** is the share of normal or ischemia scans it flags by mistake.
- **AUC** scores how well the model ranks bleeds above non-bleeds (1.0 is perfect, 0.5 is a coin flip).

**Baseline.** A "model" that always says "no bleed" scores 83.6% accuracy and catches **0%** of bleeds. Accuracy alone says little here, so we use the measures above.

| Test data | Bleeds caught | False alarms | AUC |
| --- | --- | --- | --- |
| Same source (998 slices, 164 bleeds) | 90.9% (88.4–93.3%) | 0.7% (0.2–1.4%) | 0.995 |
| New source: Kitamura, held-out half (50 bleeds, 50 normal), cut-off 0.5 | 91.5% (86–96%) | 15.5% (10–20%) | 0.948 |

Ranges show the lowest and highest of four training runs. Same-source scores are an upper bound because of the patient-ID limit above.

## Where it fails

- **It misses about 1 in 11 bleeds,** even on data like its training data (11–19 of 164 across our four runs).
- **False alarms rise on a new source.** At a 0.5 cut-off, 10–20% of normal scans from the new source got flagged. Changing only the cut-off did not fix it (false alarms ranged from 14% to 48%).
- **The heatmaps do not show the model finding the bleed.** On missed bleeds, the highlighted area sits at the back of the head or on the head-holder. On some normal scans, it sits on image corners and text overlays. The model may lean on cues that are not the bleed.
- **Our new-source test is small:** 100 scans from one publisher.

## Our own question: how many local scans does a new hospital need?

We fine-tuned the model on 10, 20, 40 or 100 labelled scans from the new source, then tested on that source's other half (3 seeds each).

| Local scans | False alarms (new source) | Bleeds caught (new source) | AUC (new source) | Bleeds caught (original data) |
| --- | --- | --- | --- | --- |
| 0 | 20.0% | 92.0% | 0.937 | 88.4% |
| 40 | 1.3% | 94.0% | 0.992 | 78.3% |
| 100 | 1.3% | 95.3% | 0.997 | 86.8% |

With about 100 local scans, false alarms on the new source fell from 20% to about 1%. The same fine-tuning cost recall on the original data: it dropped to about 77% at 20–40 scans and recovered to 87% at 100. The 0-scan row uses a different half of the new source than the table above, so its numbers differ slightly. This is one new source and three seeds, so we treat the result as a lead, not a guarantee. We suggest that a hospital test the tool on its own labelled scans before relying on it.

## What we'd do next

- Use data with patient IDs, so we can split by patient and remove the leakage risk.
- Test scans from more hospitals, and set the alert cut-off on each hospital's own scans.
- Check performance by bleed size, patient age and sex, once the data allows it.

## Sources

- Brain Stroke CT Dataset (copy of the TEKNOFEST-2021 data): Koç U, Akçapınar Sezer E, Alper Özkaya Y, et al. *Eurasian J Med.* 2022;54(3):248-258. https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset. Rights: Republic of Türkiye Ministry of Health.
- Kitamura, F. C. (2018). Head CT - hemorrhage (CC0). https://doi.org/10.34740/KAGGLE/DSV/152137
- Tools: [PyTorch and torchvision](https://pytorch.org) (ResNet18), [scikit-learn](https://scikit-learn.org), [pandas](https://pandas.pydata.org), [Kaggle Notebooks](https://www.kaggle.com/code), [Streamlit](https://streamlit.io) and Streamlit Community Cloud (demo app).
- AI tools used: [Claude](https://claude.ai), [ChatGPT](https://chatgpt.com), [GitHub Copilot](https://github.com/features/copilot). We read, tested and can explain all code we used.
