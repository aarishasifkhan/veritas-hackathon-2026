# Experiment log
Every run is scored the same way: validation set (997 slices), External_Test (200, same source, "any stroke") and half of the Kitamura set (KitDev, 100 slices, different source). The other Kitamura half (KitFinal) and our test split stay untouched until the end. "False-alarm rate" = share of non-bleeds flagged as bleeds.

## First model (notebook 01)
ResNet18, no augmentation. Validation: bleed recall 92.7% (12 of 164 missed), 16 false alarms of 833. External_Test: 59 of 70 strokes caught, 12 of 130 false alarms. Kitamura (all 200): 100 of 100 bleeds caught, 59 of 100 normals flagged.

## E0: same recipe re-run (notebook 03)
Validation: recall 92.7%, false-alarm rate 1.0%, AUC 0.995. External_Test: recall 84.3%, false alarms 9.2%. KitDev: recall 92%, false alarms 16%, AUC 0.945.
Learned: the same recipe gave very different false alarms on the new source (59% before, 16% now). Run-to-run noise is large, so we repeat each experiment with 3 more seeds.

## E1: augmentation (random crop, 10° rotation, brightness/contrast jitter)
Validation: recall 89.0% (18 missed), false alarms 0.7%, AUC 0.991. External_Test: recall 84.3%, false alarms 6.9%. KitDev: recall 100%, false alarms 18%, AUC 0.992.
Learned: best ranking on the new source and fewest same-source false alarms, but more missed bleeds in-source. One run only.

## E2: augmentation + per-image contrast stretching
Validation: recall 94.5% (9 missed), false alarms 1.2%, AUC 0.994. External_Test: recall 95.7% but false alarms 36.2%. KitDev: recall 94%, false alarms 14%, AUC 0.972.
Learned: more bleeds and strokes caught, but many more false alarms on same-source normals. Not a clear win.

## Decision rule
Before the experiments: pick the best KitDev AUC unless validation bleed recall drops over 2 points. After E0-E2 we saw the rule ignored a large same-source regression (E2), so we amended it, before looking at more results, to use averages over seeds and to limit the ExtTest false-alarm increase to 5 points.

## Problems we hit
- Hugging Face copy rate-limited (HTTP 429) and had no labels, so we switched to the original Kaggle dataset.
- We first attached Kitamura's notebook instead of his dataset.
- DICOM headers had no patient IDs, so we could not split by patient.
- Our first Kitamura result (59% false alarms) did not repeat when we retrained.
