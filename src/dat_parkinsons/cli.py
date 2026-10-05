"""Command-line entry points. Heavy ML imports are loaded only when needed."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .contracts import align_predictions, metrics, patient_folds
from .ensemble import blend, fit_blend
from .io import patient_hashes, read_json, write_json


def load_tables(specification):
    path = Path(specification)
    paths = read_json(path)
    if not isinstance(paths, dict) or not paths:
        raise ValueError("Expected a JSON object mapping model names to prediction CSV paths")
    tables = {
        name: pd.read_csv(path.parent / filename, dtype={"patient_id": str})
        for name, filename in paths.items()
    }
    ids, labels, predictions = align_predictions(tables)
    return list(tables), ids, labels, predictions


def run_fit(args):
    names, ids, labels, predictions = load_tables(args.tables)
    weights = fit_blend(predictions, labels)
    artifact = {
        "model_order": names,
        "weights": weights.tolist(),
        "eps": 1e-6,
        "output_clip": 1e-4,
        "fit_patient_hashes": patient_hashes(ids),
        "fit_metrics": metrics(labels, blend(predictions, weights)),
        "scope": "Fitting diagnostic on the weight-selection data; not independent validation",
    }
    write_json(args.output, artifact)
    return {key: value for key, value in artifact.items() if key != "fit_patient_hashes"}


def run_evaluate(args):
    names, ids, labels, predictions = load_tables(args.tables)
    artifact = read_json(args.blend)
    if names != artifact["model_order"]:
        raise ValueError("Prediction table order differs from the blend artifact")
    if set(patient_hashes(ids)) & set(artifact["fit_patient_hashes"]):
        raise ValueError("Evaluation patients overlap blend-fitting patients")
    report = metrics(
        labels, blend(predictions, artifact["weights"], artifact["eps"], artifact["output_clip"])
    )
    report["scope"] = (
        "Disjoint from blend-fitting IDs; caller must also ensure independence from all base-model training and selection"
    )
    write_json(args.output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="DaT imaging research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare", help="Prepare classifier images using a trained slice scorer"
    )
    prepare.add_argument("--nifti-dir", required=True)
    prepare.add_argument("--labels", required=True)
    prepare.add_argument("--scorer", required=True)
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--device", default="cpu")
    folds = sub.add_parser("folds", help="Create a shared patient fold manifest")
    folds.add_argument("--manifest", required=True)
    folds.add_argument("--output", required=True)
    folds.add_argument("--n-splits", type=int, default=5)
    folds.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train-classifier")
    train.add_argument("--manifest", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--folds")
    train.add_argument("--device", default="cpu")
    selector = sub.add_parser("train-selector")
    selector.add_argument("--labels", required=True)
    selector.add_argument("--output", required=True)
    selector.add_argument("--epochs", type=int, default=40)
    selector.add_argument("--n-splits", type=int, default=5)
    selector.add_argument("--seed", type=int, default=42)
    selector.add_argument("--device", default="cpu")
    selector.add_argument(
        "--final",
        action="store_true",
        help="Fixed-epoch refit; does not produce validation metrics",
    )
    selector.add_argument("--no-pretrained", action="store_true")
    fit = sub.add_parser("fit-blend")
    fit.add_argument("--tables", required=True)
    fit.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate-blend")
    evaluate.add_argument("--tables", required=True)
    evaluate.add_argument("--blend", required=True)
    evaluate.add_argument("--output", required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--bundle", required=True)
    predict.add_argument("--nifti-dir", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", default="cpu")
    demo = sub.add_parser(
        "demo", help="Synthetic CPU inference check; random weights, no scientific claims"
    )
    demo.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = None
    if args.command == "prepare":
        from .pipeline import prepare_images
        from .preprocessing import ViewConfig

        prepare_images(
            args.nifti_dir,
            args.labels,
            args.scorer,
            args.output,
            ViewConfig(**read_json(args.config)["view"]),
            args.device,
        )
    elif args.command == "folds":
        frame = pd.read_csv(args.manifest, dtype={"patient_id": str})
        result = patient_folds(frame, args.n_splits, args.seed)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    elif args.command == "train-classifier":
        from .training import train_classifier

        report = train_classifier(args.manifest, args.config, args.output, args.device, args.folds)
    elif args.command == "train-selector":
        from .selector_training import train_selector

        train_selector(
            args.labels,
            args.output,
            args.epochs,
            args.n_splits,
            args.seed,
            args.device,
            args.final,
            not args.no_pretrained,
        )
    elif args.command == "fit-blend":
        report = run_fit(args)
    elif args.command == "evaluate-blend":
        report = run_evaluate(args)
    elif args.command == "predict":
        from .pipeline import Predictor

        report = {
            "examinations": len(
                Predictor(args.bundle, args.device).predict_directory(args.nifti_dir, args.output)
            )
        }
    elif args.command == "demo":
        from .demo import run_demo

        report = run_demo(args.output)
    if report is not None:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
