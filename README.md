# BleedCheck CT

**A second look at head CT scans: can a small AI model flag brain bleeds, and what happens when it meets scans from a different hospital?**

> **This project is a prototype and is not medical advice.** It is a student research project built over one weekend on public data. It is not a medical device and has not been tested on patients.

Veritas AI International High School Hackathon 2026, Health track.
Team: **Aarish Asif Khan** and **Aabir Asif Khan** (Islamabad, Pakistan).

| | |
|---|---|
| Live demo | https://bleedcheck-ct.streamlit.app/ |
| Notebook | https://www.kaggle.com/code/aarishasifkhan/bleedcheck-ct-a-second-look-at-head-ct-scans/ |
| Video | [FILL: video link] |
| Code and logs | this repository (`notebooks/`, `app/`, `docs/`) |

## The problem
A brain bleed is a medical emergency, and doctors look for it on a head CT scan. When many scans need reading, a tool that points to the scans that deserve a closer look first could help.

But a model that looks excellent on the data it was built on can behave very differently at another hospital, where scanners and settings are different. Most demos report a single accuracy number from one source and never test this.

We care about this personally: our father fell seriously ill in 2019, and it made us want to build something in health. [EDIT this sentence so it says exactly what you want to share.]

## Our question
> **Does a bleed-detection model still work on scans from a different hospital, and how many scans from that hospital does it need to adapt?**

## What we built
- A **ResNet18** (pretrained on ImageNet, fine-tuned by us) that scores a head CT slice as Normal, Bleeding or Ischemia.
- An **honest test setup**: a second public data set stands in for "a different hospital", split into a half for choosing settings (DEV) and a half we look at once at the end (FINAL).
- **Every recipe trained with several seeds**, because one run can mislead.
- A **rule written before the final test** for choosing the model.
- **Grad-CAM heatmaps** to see what drives the score, and a look at where the model fails.
- An experiment on **how many local scans a new hospital needs** to fix the model's false alarms.
- A **web demo** where anyone can upload a CT slice and see the score and a heatmap: [FILL: Streamlit link].

## Data
Full details and citations are in [`DATA.md`](DATA.md).

| | Main data | Second source ("a different hospital") |
|---|---|---|
| Name | Brain Stroke CT Dataset (Kaggle, ozguraslank), a copy of the TEKNOFEST-2021 stroke data set | Head CT - hemorrhage (Kaggle, Felipe Kitamura) |
| Size | 6,650 slices: 4,427 Normal, 1,093 Bleeding, 1,130 Ischemia, plus 200 held-out slices (`External_Test`) | 200 slices: 100 normal, 100 hemorrhage |
| Licence | Rights statement from the Republic of Türkiye Ministry of Health, no standard open licence | CC0 (public domain) |
| Our use | Train, validation and test | DEV half and FINAL half |

There are no patient IDs, so slices from one patient may land in different splits. **Same-source scores are therefore an optimistic upper bound.** We used only public data and no real patient records.

## How we tested
- "Bleed" means the Bleeding class; "no bleed" means Normal or Ischemia.
- We report **missed bleeds** and **false alarms** separately. Accuracy alone is misleading: a model that always says "no bleed" is about 84% accurate and catches no bleeds.
- Baselines (validation set): *always "no bleed"*: 83.6% accuracy, 0% of bleeds caught. *Logistic regression on 32×32 pixels*: 82.5% accuracy, 74.4% of bleeds caught, 132 false alarms. [CONFIRM against notebook Section 2.]

## Results
All numbers come from `notebooks/BleedCheck_CT_final.ipynb` (Section 9). They are averages over [FILL: number] training runs with different seeds, with the range in brackets.

| Test | Bleeds caught | False alarms | AUC |
|---|---|---|---|
| Same source as the training data (test split) | [FILL] | [FILL] | [FILL] |
| A different source (Kitamura FINAL, cut-off 0.5) | [FILL] | [FILL] | [FILL] |
| Different source, only the cut-off moved | [FILL] | [FILL] | n/a |
| Different source, after tuning on 100 local scans | [FILL] | [FILL] | [FILL] |

**What we found**
1. On data like its training data the model ranks scans very well, but it still misses [FILL: about 1 in N] bleeds.
2. On a second source it keeps catching most bleeds, **but flags many more normal scans**, and the ranking quality drops. This is dataset shift.
3. **Our attempted fixes did not clearly help.** Augmentation and contrast stretching mostly moved the trade-off (more bleeds caught, more false alarms) instead of improving it, and the difference between seeds was as large as the difference between recipes. [CONFIRM with notebook Section 5 and the rule's choice in Section 6.]
4. **Moving only the cut-off is not reliable** across seeds.
5. **A small amount of local data helps a lot.** After fine-tuning on [FILL: n] scans from the new source, false alarms there went from [FILL] to [FILL]. The cost: bleeds caught on the original data changed from [FILL] to [FILL]. This is one new source and 100 test scans, so we treat it as a lead, not a guarantee.

## Where it fails
- **It misses bleeds.** About [FILL: 1 in N] bleeds were missed even on data like its training data.
- **A new hospital's scans cause false alarms.** The cut-off from one source does not carry over to another.
- **The heatmaps do not show it finding the bleed.** For missed bleeds and false alarms, the highlighted area often sits on the skull edge, the back of the head, the head-holder, or image corners and text overlays. The model may use cues that are not the bleed. We are not radiologists and do not claim the heatmaps locate bleeds.
- **Small tests.** One extra source with 200 scans. A difference of a few scans is noise.

## What went wrong along the way
We kept an honest record in [`docs/experiment_log.md`](docs/experiment_log.md), including mistakes: an early test run used a placeholder model before the model was chosen; our first decision rule ignored an important condition and we fixed it before the multi-seed runs; and one run flagged 59% of the new source's normal scans while a re-run flagged 16%, which is why we use several seeds.

## Run it yourself
1. Open `notebooks/BleedCheck_CT_final.ipynb` on Kaggle.
2. Add the two data sets (Brain Stroke CT Dataset by ozguraslank; Head CT - hemorrhage by Felipe Kitamura).
3. Turn on a GPU and Internet, then **Save & Run All**. It takes about 30 minutes.

Repository layout:
```
README.md               this file
DATA.md                 data sources, licences and citations
notebooks/              our notebooks (final one: BleedCheck_CT_final.ipynb)
docs/experiment_log.md  everything we tried, including mistakes
app/                    the web demo (Streamlit)
```

## AI tools and credits
**AI tools.** We used **Claude** (Anthropic) as our main AI assistant: it helped us plan the project, explained concepts, and wrote and debugged much of the code, which we ran, checked and adjusted ourselves. We used **ChatGPT** only to brainstorm project names. Its suggestions promised more than our project delivers, so we asked Claude for alternatives and chose **BleedCheck CT** ourselves. We ran every experiment and made the decisions about what to keep or drop. Some of this write-up was drafted with Claude and edited by us. No other AI tools were used.

**Data.**
- Brain Stroke CT Dataset (Kaggle, ozguraslank), a copy of the TEKNOFEST-2021 stroke data set. Koç U, Akçapınar Sezer E, Alper Özkaya Y, et al. *Eurasian J Med.* 2022;54(3):248-258.
- Kitamura F. *Head CT - hemorrhage*, Kaggle (CC0). https://doi.org/10.34740/KAGGLE/DSV/152137

**Methods.** ResNet18 (He et al., 2016) via torchvision with ImageNet weights; Grad-CAM (Selvaraju et al., 2017); PyTorch, scikit-learn, pandas, matplotlib; demo built with Streamlit.

## Ethics and safety
- Public data only. No real patient records, and the demo asks people not to upload any.
- A flag means "look closer", never a diagnosis. A clear result does not rule out a bleed.
- This is a prototype. It has not been validated clinically.

**This project is a prototype and is not medical advice.**
