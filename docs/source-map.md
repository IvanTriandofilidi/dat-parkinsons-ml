# From exploratory notebooks to a package

Sources inspected on 2026-10-05:

- Kaggle: [Dat_Parkinson](https://www.kaggle.com/code/ivantriandofilidi/dat-parkinson). The current editor notebook was exported; the initial request pointed to run `342117009`, but equivalence of that historical run and the current draft is not established.
- Colab: [DaT Parkinson's Challenge](https://colab.research.google.com/drive/1qCBlGvt3ugkqL6cME0UFm_c7xQ_9ExmG). Python source exported without cell outputs; aggregate result values separately transcribed from saved outputs.

Raw notebooks are not distributed because their outputs include restricted data and exploratory material. Selected source hashes are recorded in `reports/source-fingerprints.json` for local traceability. No code was executed in the original Kaggle or Colab environments during this refactor.

## Mapping

Cell indices below are zero-based.

| Source | Original responsibility | Package location |
|---|---|---|
| Kaggle 57 | Raw rotated slice-quality annotation crops | `preprocessing.raw_scorer_crop`, annotation contract in `docs/data.md` |
| Kaggle 1 / 64 | ResNet18 slice-score regression and GroupKFold | `models.SliceScorerResNet18`, `selector_training.py` |
| Kaggle 1 / 64 | Per-slice percentile window, scoring over z | `preprocessing.scorer_input`, `pipeline.select_slice` |
| Kaggle 25 | Percentile / CLAHE / uptake-ratio RGB view | `preprocessing.multi_contrast` |
| Kaggle 15 / 35 | Z7 / Z4 slabs, brightness matching, fixed crop | `preprocessing.make_view` |
| Kaggle 15 / 25 | Geometric uptake and asymmetry features | `preprocessing.sbr_features` |
| Kaggle 19 / 26 / 41; Colab 25 | Classifier backbone, optional SBR head, CV | `models.ParkinsonClassifier`, `training.py` |
| Kaggle 43 | SE-ResNeXt Z4 training parameters | `configs/seresnext_z4.json` |
| Colab 27 | Swin Z7 training parameters, SBR enabled | `configs/swin_z7.json` |
| Colab 30 / 31 | OOF alignment and branch comparisons | `contracts.align_predictions`, historical result report |
| Colab 33 | Constrained five-model logit blending | `ensemble.py`, documented historical weights |

## Preserved ideas

Learned slice ranking; one-channel ResNet18 selection; 1 mm resampling; central half crop for the scorer; 200-pixel classifier crops; multi-contrast and 2.5D representations; optional uptake proxies; patient-aware folds; probability ensembles optimized for log loss.

## Added engineering

Explicit artifact metadata, common preparation and inference code, strict row identity validation, deterministic inference policy, train-only imputation, CLI entry points, package distribution, synthetic demo, and CI.

## Not claimed as preserved

Exact notebook execution state, every historical architecture/head combination, original random TTA draws, original fold ordering, every earlier exploratory segmentation or heuristic slicing method, and historical model weights. The cleaned walkthrough explains the chosen pipeline rather than embedding the entire experiment history.

The original method and experiments are Ivan Triandofilidi's work. Packaging, tests, and documentation were refactored with AI assistance. The author should review the code and understand the documented changes before presenting it in technical interviews.
