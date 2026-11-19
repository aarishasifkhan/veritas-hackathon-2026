# BleedCheck CT: a second look at head CT scans

**Track:** AI + Health

**Team:** Aarish Asif Khan, Aabir Asif Khan

**Live demo:** https://bleedcheck-ct.streamlit.app

**Submission video:** https://youtu.be/ZO6M_Wn9Gow

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
- **A second data source.** We used 200 scans from a different publisher. We split them in half: 100 to compare experiments and fine-tune, and 100 kept for one final test.
- **Several training runs, not one.** A single run can mislead: in our first attempt one run flagged 59% of the new source's normal scans and a re-run of the same recipe flagged 16%. So every recipe is trained with four random seeds and we report the mean and the range.
- **A written log** of every experiment and mistake: `docs/experiment_log.md`.

## How it works

**Data.** We trained on the Brain Stroke CT Dataset from Kaggle, a copy of the TEKNOFEST-2021 stroke data from the Turkish Ministry of Health. It holds 6,650 anonymised head CT slices (4,427 normal, 1,093 bleeding, 1,130 ischemia). We split it 70/15/15 into 4,655 training, 997 validation and 998 test images. The data has no patient IDs, so slices from one patient can land in both training and test. Our same-source scores may therefore look better than they should. Full details are in [DATA.md](DATA.md).

**Model.** We fine-tuned a ResNet18 pretrained on ImageNet to sort slices into three classes. Class weights help it learn the rarer bleeding and ischemia classes. We keep the epoch with the best validation bleed AUC.

**How we chose the recipe.** We compared three recipes: plain, plus augmentation (random crops, rotations, brightness changes), and plus augmentation and contrast stretching. We chose by a rule we wrote before looking at the test split or the final Kitamura half: a recipe must stay within 2 points of the plain model's validation bleed recall and within 5 points of its same-source false-alarm rate, and among those we pick the best AUC on the first half of the new source. Both augmentation recipes lowered false alarms on the new source (about 31% down to about 11%) and raised its AUC (0.941 up to about 0.973). But they also lowered bleeds caught on the original data (92.4% down to 87.7%) and roughly tripled false alarms on the dataset's own held-out set (5.4% up to 16.7%), so the rule rejected them and the **plain ResNet18** was chosen. We trained it four times with different random seeds and report the mean and the range.

## Results

We measure how well the model finds **bleeds** (bleeding vs. everything else).

- **Bleeds caught** is the share of real bleeds the tool flags.
- **False alarms** is the share of normal or ischemia scans it flags by mistake.
- **AUC** scores how well the model ranks bleeds above non-bleeds (1.0 is perfect, 0.5 is a coin flip).

**Baseline.** A "model" that always says "no bleed" scores 83.6% accuracy and catches **0%** of bleeds. Accuracy alone says little here, so we use the measures above.

| Test data | Bleeds caught | False alarms | AUC |
| --- | --- | --- | --- |
| Same source (998 slices, 164 bleeds) | 92.1% (82.9–97.6%) | 1.7% (0.6–2.8%) | 0.996 (0.991–0.998) |
| New source: Kitamura, held-out half (50 bleeds, 50 normal), cut-off 0.5 | 98.5% (96–100%) | 27.5% (16–48%) | 0.946 (0.893–0.985) |

Ranges show the lowest and highest of four training runs. Same-source scores are an upper bound because of the patient-ID limit above.

## Where it fails

- **It misses about 1 in 13 bleeds on average,** even on data like its training data. Across our four runs it missed between 4 and 28 of 164 bleeds (2% to 17%), so one unlucky run can be much worse than another.
- **False alarms rise sharply on a new source.** At a 0.5 cut-off, 16–48% of normal scans from the new source got flagged (27.5% on average), compared with 1.7% on data like the training data. The ranking quality (AUC) also dropped from 0.996 to 0.946.
- **Moving the cut-off did not fix it.** We moved the cut-off so the model caught at least 95% of bleeds on the first half of the new source. On the held-out half, false alarms averaged 28% (14–44%). It helped some runs and hurt others, so it is not reliable from one run to the next.
- **The heatmaps do not show the model finding the bleed.** On missed bleeds, the highlighted area sits at the back of the head or on the head-holder. On some normal scans, it sits on image corners and text overlays. The model may lean on cues that are not the bleed.
- **Our new-source test is small:** 100 scans from one publisher.
- **Demo hosting limits.** We first built the demo in Gradio for Hugging Face Spaces, but the paid hosting was out of reach for us, so we moved it to Streamlit Community Cloud. Free hosting can sleep when unused, so the first load may take a minute.

## Our own question: how many local scans does a new hospital need?

We fine-tuned our demo model (one of the four runs) on 10, 20, 40 or 100 labelled scans from the first half of the new source, then tested on the other 100 scans from that source, which the model never trained on (3 random draws each). We also checked that the model did not forget the original source.

| Local scans | False alarms (new source) | Bleeds caught (new source) | AUC (new source) | Bleeds caught (original data, validation split) |
| --- | --- | --- | --- | --- |
| 0 | 16.0% | 100% | 0.985 | 97.0% |
| 10 | 4.0% | 90.7% | 0.989 | 88.8% |
| 20 | 6.0% | 95.3% | 0.988 | 89.0% |
| 40 | 1.3% | 95.3% | 0.998 | 86.4% |
| 100 | 2.0% | 96.0% | 0.999 | 90.9% |

With about 40 to 100 local scans, false alarms on the new source fell from 16% to about 1–2%, and even 10 scans cut them to 4%. This came with costs:

- **Bleeds caught on the new source dropped** from 100% to about 91–96%, so the model missed a few more bleeds as it became less trigger-happy.
- **The model got worse on the original data.** Bleeds caught there fell from 97.0% to 86.4% at 40 scans and recovered only to 90.9% at 100 scans. At 100 scans, false alarms on the original data also rose from 1.3% to 4.7%.

The 0-scan row is the single demo model, while the results table above averages all four runs, so the numbers differ. The demo model was chosen for its validation AUC, and it also happens to be the best of the four on the new source (AUC 0.985, the top of the 0.893–0.985 range), so its starting point looks better than the average. At 100 local scans the model sees every scan in the local half, so the three draws differ only in training order. This is one demo model, one new source and three draws, so we treat the result as a lead, not a guarantee. We suggest that a hospital test the tool on its own labelled scans before relying on it.

## What we'd do next

- Use data with patient IDs, so we can split by patient and remove the leakage risk.
- Test scans from more hospitals, and set the alert cut-off on each hospital's own scans.
- Mask the skull and image background so the model cannot lean on them.
- Check performance by bleed size, patient age and sex, once the data allows it.
- Test with clinicians.

## Sources

- Brain Stroke CT Dataset (copy of the TEKNOFEST-2021 data): Koç U, Akçapınar Sezer E, Alper Özkaya Y, et al. *Eurasian J Med.* 2022;54(3):248-258. https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset. Rights: Republic of Türkiye Ministry of Health.
- Kitamura, F. C. (2018). Head CT - hemorrhage (CC0). https://doi.org/10.34740/KAGGLE/DSV/152137
- Tools: [PyTorch and torchvision](https://pytorch.org) (ResNet18), [scikit-learn](https://scikit-learn.org), [pandas](https://pandas.pydata.org), [Kaggle Notebooks](https://www.kaggle.com/code), [Streamlit](https://streamlit.io) and Streamlit Community Cloud (demo app).
- AI tools used: Claude (Anthropic) was our main AI assistant: it helped us plan, explained concepts, and wrote and debugged much of the code, which we ran, checked and adjusted ourselves. ChatGPT was used only to brainstorm project names. We ran every experiment and made the decisions. No other AI tools were used.
