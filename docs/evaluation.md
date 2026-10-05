# Evaluation: what the numbers establish

## Historical observations

Colab cells 30–31 align five prediction files and report aggregate metrics on 1,362 examinations. Cell 29 shows 1,382 rows reduced to 1,362 by keeping the first duplicate patient ID. Cell 33 fits five logit-blend weights using SLSQP and calculates metrics on the same rows:

| Model order | Weight (rounded notebook output) |
|---|---:|
| pred_25d_z4 | 0.2611 |
| pred_densenet121 | 0.0000 |
| pred_swin_rgb | 0.0606 |
| pred_swin_z2_sbr_tta | 0.4067 |
| pred_swin_z7 | 0.2716 |

Reported blend log loss: 0.2189. Reported AUROC: 0.9693. These are **weight-fitting diagnostics**. Rounded weights are evidence, not a lossless export of the optimizer artifact. They are not installed as default inference weights in this repository.

## Issues found during the refactor

1. **Duplicate provenance is unresolved.** Removing repeated OOF rows does not prove that a patient was absent from every relevant training fold. Duplicate volumes could also occur as `.nii` / `.nii.gz` copies. The package rejects duplicate IDs; it does not silently choose one.
2. **Blend fitting and evaluation reused rows.** Base predictions being OOF does not make a fitted meta-model's score on those same rows unbiased. The `fit-blend` command labels its metrics accordingly. `evaluate-blend` rejects IDs used to fit weights.
3. **Epoch selection reused validation folds.** Original early stopping selected a checkpoint on the same fold later reported as validation. Package OOF metrics retain this common development protocol and explicitly say so; they are not final test estimates.
4. **Preprocessing was mutable notebook state.** Later Colab cells redefine `load_volume_torchio` without resampling and redefine `prepare_model_input` to return three channels, while the selected ResNet18 scorer expects one. A clean run of the exported notebook cannot be assumed to reproduce saved outputs.
5. **Scorer provenance is incomplete.** A scorer trained on quality annotations from classifier-validation patients is not a strictly held-out upstream component, even if those annotations are not pathology labels. Establish its training patient set before claiming end-to-end OOF independence.
6. **TTA and metric clipping differ across experiments.** The source uses random training augmentations for some TTA runs and sometimes clips probabilities to `[0.01, 0.99]` when selecting checkpoints. The refactor reports ordinary log loss and uses an explicit deterministic `none` or horizontal-flip TTA policy. It therefore needs new experiment results.
7. **File names alone do not prove model identity.** Some Z7 output filenames mention SBR even when a Kaggle invocation sets `use_sbr=False`. Exact checkpoint-to-OOF mapping for every historical branch has not been established.

## Recommended independent evaluation

Reserve an outer patient holdout before tuning. If center/scanner metadata becomes available, additionally evaluate a center-separated holdout. Keep **all** outer-validation patients out of slice-scorer training, normalization fitting, base-model selection, hyperparameter tuning, blend weight fitting, and calibration fitting.

Within the outer training split:

1. Train the selector using only permitted training-side annotations, or use a separately sourced frozen selector with documented provenance.
2. Produce base-model OOF predictions using shared patient folds. Use an inner split for checkpoint selection if an unbiased outer-fold estimate is needed.
3. Fit ensemble weights on those training-side OOF predictions.
4. Freeze the pipeline and evaluate it once on the untouched outer holdout.

Meta-level cross-validation on a fixed OOF matrix is a useful development diagnostic, but it is not automatically nested end-to-end validation: the base models producing meta-training features may have trained on meta-validation patients. Independent claims require controlling the complete training dependency graph.

Report log loss, AUROC, calibration curves, cohort sizes, and patient-level uncertainty intervals. Include sensitivity to z offsets and compare fixed-center slices, learned selection, single branches, and equal-weight blends. No such new ablation or holdout result is claimed here.

The CLI can detect overlapping patient IDs between blend fitting and evaluation; it **cannot infer** whether base classifiers have previously seen an evaluation patient. That responsibility requires saved split provenance and an independently held-out cohort.
