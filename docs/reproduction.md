# Reproduction and artifact handling

## What is reproducible now

Software checks can run with no competition data: package installation, unit tests, synthetic volume processing, random-weight model loading, deterministic inference, and submission schema. The historical competition results are not yet reproduced by this package. Real training requires an authorized dataset, quality annotations, and sufficient compute.

## Environment

The README provides a CPU installation. Development validation uses Python 3.12 on Windows; CI targets Python 3.12 on Linux. Dependency ranges in `pyproject.toml` define supported installation bounds, not a promise of bitwise equivalence across versions. Record `python -m pip freeze`, platform, CUDA/cuDNN versions, device model, and code commit for each real run. A local verification record is in `reports/verification.md`.

## Selector lifecycle

Run patient-grouped selector CV first. Compare MSE with the training-mean baseline. Select hyperparameters and the final epoch count using training-side evidence. The example final refit uses 50 epochs because that invocation appears in Kaggle cell 66; it is not asserted to be optimal. Final refit runs a fixed number of epochs and creates no validation estimate.

The selected architecture matches the one-channel ResNet18 scorer's parameter names. `load_scorer` also accepts a raw state dictionary from that architecture, but a raw dictionary cannot prove preprocessing provenance. Verify that legacy weights were trained with the selected per-slice 50–99 percentile normalization. Global-normalization variants are not interchangeable.

## Train both example classifier branches

After the selector and Z7 preparation from the README:

```bash
dat-parkinsons prepare --nifti-dir data/niftis --labels data/train_labels.csv --scorer artifacts/selector/selector.pt --config configs/seresnext_z4.json --output data/prepared_z4 --device cuda
dat-parkinsons train-classifier --manifest data/prepared_z4/manifest.csv --folds artifacts/folds.csv --config configs/seresnext_z4.json --output runs/seresnext_z4 --device cuda
```

Use `artifacts/folds.csv` for both branches. Each run writes fold checkpoints, an OOF CSV, training history, configuration, metrics, and input provenance. The view sidecar must match the experiment configuration. Classifier datasets are unique by patient before splitting. Classifier folds are sorted by patient ID before assignment so manifest order does not affect membership; this may differ from historical notebook folds.

## Fit a blend, then build an inference bundle

```bash
dat-parkinsons fit-blend --tables configs/prediction_tables.example.json --output artifacts/blend.json
```

Copy `model_order` and `weights` from this artifact into a copy of `configs/bundle.example.json`. Preserve the same ordered branch names. Checkpoint and scorer paths are resolved relative to the bundle file. Each branch averages probabilities across its listed fold checkpoints before the logit blend combines branches. The model checkpoint stores its view, trained SBR medians, and inference TTA policy.

The example weights `[0.5, 0.5]` are an equal-weight baseline, not fitted weights. The original five-model rounded weights in `reports/historical-results.json` are documentation only; they do not belong to the two example branches.

For a disjoint holdout prediction-table specification:

```bash
dat-parkinsons evaluate-blend --tables artifacts/holdout_tables.json --blend artifacts/blend.json --output reports/holdout_metrics.json
```

This rejects patients used to fit the blend. Read `docs/evaluation.md`: base-model training and selection must also be independent of the holdout.

## Intentional changes from the notebooks

- No mutable global path configuration, broad fallback directory search, or automatic choice of the first checkpoint.
- No silent duplicate deletion or skipping of failed cases.
- Per-slice scorer normalization and 1 mm resampling are explicit.
- Metadata-rich classifier checkpoints replace bare weights. Existing classifier weights need a separately verified migration including architecture, training medians, view, and scorer provenance.
- Fixed deterministic horizontal-flip TTA replaces random augmentation TTA where enabled. This changes predictions.
- Classifier validation loss uses ordinary log loss instead of clipping all validation probabilities to `[0.01,0.99]`. Final logit blending retains the source's `[1e-4,1-1e-4]` output clipping.
- Slice MSE is sample-weighted across validation batches. Final selector training retains the final fixed-epoch weights rather than choosing an epoch by training loss.
- Grayscale mode is an explicitly three-channel classifier view; the broken three-channel scorer redefinition is excluded.

These changes improve the execution contract but require new measured results. A package refactor is not evidence that the scientific score improved.

## Competition runtime

The standalone `predict` command produces the official CSV column names. It has not been validated inside the DrivenData competition container and is not presented as a ready-to-submit competition ZIP. Follow the [official runtime specification](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/989/) for the adapter, mounted paths, dependencies, and smoke test before a compatible deployment.
