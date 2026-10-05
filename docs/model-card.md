# Model card

**Owner:** Ivan Triandofilidi

**Task:** Probability that a DaT scan examination is abnormal

**Status:** Research prototype; code refactor, historical aggregate results, synthetic verification

**Distributed trained weights:** None

## Inputs and outputs

Input: a single 3D DaT NIfTI examination. Output: a probability in `[0,1]` written as `is_pathologic`. The pipeline uses an upstream slice-quality regressor and downstream image classifiers with optional geometric uptake features. It does not output a medical diagnosis, anatomical segmentation, or calibrated uncertainty.

## Intended use

Study of learned slice selection, multi-view medical imaging, patient-level validation, and software packaging. Reviewers can inspect the implementation and run a synthetic integration check without restricted data.

## Out-of-scope use

Clinical decision making, screening, treatment selection, diagnosis of Parkinson's disease, or claims about patient outcomes. External generalization and prospective performance have not been established.

## Training and evaluation context

The original challenge describes a multicenter French dataset. Repository historical metrics come from saved notebook outputs, not a new independent benchmark. Exact original fold provenance, duplicate origins, final submitted checkpoint composition, and private leaderboard results are not established. Details: [evaluation audit](evaluation.md).

## Limitations

- A single selected slice and nearby slabs can omit relevant 3D information.
- Slice-quality annotations are subjective and lack a documented inter-rater study.
- Native NIfTI orientation is preserved; new orientations need explicit validation.
- Intensity-based centering and geometric SBR proxies are heuristics, not clinical regional measurements.
- Scanner, reconstruction, institution, and population shifts may reduce performance.
- The observed blend score reused data for weight selection.
- No calibration study, demographic subgroup audit, clinical validation, or external test result is supplied.
- No claim is made that all original models can be reconstructed from their filenames alone.

## Data governance and licenses

The training dataset, patient-level predictions, labels, and trained competition checkpoints are not distributed. The README includes two separately supplied reference illustrations, not model predictions. Post-competition training requires appropriate data rights. Source code uses MIT; third-party images, weights, and data have separate terms.
