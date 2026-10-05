# Software verification

Local verification on 2026-10-05, Windows, Python 3.12.14, CPU PyTorch. Exact installed versions are recorded in `verification-environment.json`; `verification-requirements.txt` records the dependency snapshot.

| Check | Outcome |
|---|---|
| Ruff static checks | Passed |
| Pytest | 30 tests passed |
| Example SE-ResNeXt and Swin architecture forward passes | Passed with random weights and no downloads |
| Selector channel contract | One-channel input accepted; three-channel misuse rejected |
| Classifier optimization / checkpoint round-trip | Two folds, one epoch each, synthetic images; passed |
| Selector grouped CV / final refit | Synthetic annotations, one epoch; passed |
| Full CLI inference demo | Synthetic NIfTI → selector → view → classifier → blend → submission; passed |
| Inference repeatability | Identical repeated synthetic predictions |
| Distribution build | Source distribution and wheel built successfully |
| 2.5D comparison against extracted source functions | 18 synthetic cases; maximum difference 1 / 255; uptake proxies agree within float32 tolerance |

The source comparison covered two spatial shapes, boundary/middle slices, and offsets 2/4/7. Six of 2,160,000 channel values differed by one 8-bit step. This is rounding-tolerant agreement, not bitwise identity. Details are in `preprocessing-parity.json`.

The GitHub Actions workflow separately runs Linux CPU checks on each push. Its live status is shown by the README badge; local success is not a claim about a remote workflow that has not completed.

## Not verified by these checks

Historical notebook metrics, trained competition checkpoints, real-data training convergence, GPU reproducibility, clinical validity, external generalization, or execution in the DrivenData submission container. No competition dataset was used for these software checks.
