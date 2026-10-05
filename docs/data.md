# Data contracts

## Access and redistribution

The original challenge data license restricts sharing, third-party uploads, and use outside the competition. The repository does not distribute the training dataset, NIfTI volumes, annotations, patient-level OOF predictions, or trained competition weights. Two separately supplied reference illustrations appear in the README; they are not model outputs or a downloadable training dataset. Continued data use requires a separate applicable permission or license. Do not infer permission from the continued existence of a hosted dataset or a local archive.

## Volume and pathology labels

```text
data/
  niftis/<uid>.nii.gz
  train_labels.csv
```

`train_labels.csv` has `uid,is_pathologic` columns with unique nonempty IDs and binary labels (`0 = normal`, `1 = abnormal`). Preparation requires exactly the same ID set in labels and volumes. One examination per patient is assumed; if that assumption fails, adapt the grouping key before training.

Volumes must be finite, three-dimensional, single-channel NIfTI images. The pipeline resamples to 1 mm isotropic spacing with TorchIO and preserves native array orientation, matching the selected notebook path. The array's third axis is used as z. **Anatomical orientation must be verified from the affine before using the method with a new source.** Arbitrarily oriented scans are not automatically made anatomically equivalent. Canonical reorientation would change the historical input contract and requires validation or retraining.

## Slice-quality annotations

`z_slice_labels.csv` requires:

| Column | Meaning |
|---|---|
| patient_id | Patient grouping key |
| z | Slice index after the 1 mm resampling step |
| score | Manually assigned quality score in `[0, 1]` |
| npy_filename | Path relative to the CSV directory |

Each `.npy` stores the **raw** central 50% crop, rotated with `np.rot90`, before percentile normalization. This follows Kaggle annotation cell 57. The scorer applies a 50th–99th percentile window and resizes to 128 × 128. A score is an annotation target, not a calibrated confidence or a pathology probability. Annotation quality and inter-rater reliability have not been established.

## Prepared classifier manifest

The `prepare` command writes PNG images, `manifest.csv`, and `preprocessing.json` together. Image paths are relative to the manifest. Columns are `patient_id,label,image_path,best_z,slice_score,sbr_mean,sbr_asymmetry`. Missing uptake proxies are represented by NaN and imputed from each training fold only. The preprocessing sidecar records the view configuration and SHA-256 of the scorer checkpoint.

Do not edit the manifest, sidecar, or images independently. Rebuild preparation outputs after changing the selector or preprocessing. Existing output directories are rejected to prevent accidental mixing of runs.

## Prediction tables

Each model CSV requires `patient_id,label,pred_probability`, with optional `fold`. Every table must contain exactly one row for the same patients, consistent labels, finite probabilities in `[0,1]`, and matching fold assignments when present. Files are aligned by ID, not by their row order. No implicit inner join or duplicate removal is used.

## Output

`predict` writes exactly `uid,is_pathologic`, one row per discovered volume. It processes examinations independently; normalization is within an examination, and SBR medians are loaded from the training checkpoint. An invalid file fails the run instead of silently disappearing from the submission.
