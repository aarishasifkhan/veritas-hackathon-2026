# Notebooks

**Start here:** `bleedcheck-ct-a-second-look-at-head-ct-scans.ipynb` is the final notebook. It holds the whole project from data to results, and it is the version we submitted. A copy with all outputs saved is on Kaggle:
https://www.kaggle.com/code/aarishasifkhan/bleedcheck-ct-a-second-look-at-head-ct-scans

## Files

| File | What it is |
| --- | --- |
| `bleedcheck-ct-a-second-look-at-head-ct-scans.ipynb` | Final notebook: data, baselines, model, experiments, the one-time final test, heatmaps, and the local fine-tuning study. |
| `01_data_and_model.ipynb` | Early work: loading the data and training the first model. |
| `02-second-source.ipynb` | Early work: testing on the second source (Kitamura). |
| `03-experiments.ipynb` | Early work: the experiments, before we merged everything into the final notebook. |

The three numbered notebooks are kept for the record. The final notebook replaces them.

## How to run the final notebook

1. Open it on Kaggle with a **GPU (T4)** and **Internet on**.
2. Add the two datasets as inputs:
   - Brain Stroke CT Dataset (ozguraslank)
   - Head CT - hemorrhage (Kitamura, CC0)
3. Click **Save & Run All**. It takes about 30 minutes.

Every number in the results section is computed by the notebook when it runs.

The settings for each experiment, and the mistakes we made, are in [`../docs/experiment_log.md`](../docs/experiment_log.md). Dataset details are in [`../DATA.md`](../DATA.md).

> This project is a prototype and is not medical advice.
