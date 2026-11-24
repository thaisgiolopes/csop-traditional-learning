"""Tune a LightGBM TDS regressor and export a validated ONNX package.

This focused model-building command reuses candidate samples from an existing
experiment run. It reserves the largest candidate sizes for a final temporal
test, tunes only on forward folds from smaller sizes, evaluates once, then
refits on all labeled samples for deployment.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
from pathlib import Path
import random
import shutil
from statistics import fmean, pstdev
import zipfile

import lightgbm as lgb
import numpy as np
import onnx
import onnxruntime as ort
import onnxmltools
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.dataset.dataset import Dataset
from src.dataset.sample import Sample
from src.dataset.splitter import (
    split_dataset_by_future_groups,
    split_dataset_by_ordered_groups,
)


FEATURE_NAMES = (
    "num_vertices",
    "num_edges",
    "degree_mean",
    "degree_max",
    "degree_min",
    "degree_std",
)
BASELINE_PARAMETERS = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "max_depth": -1,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "subsample": 1.0,
    "subsample_freq": 1,
    "colsample_bytree": 1.0,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and export a temporal-validation TDS ONNX model."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("experiments/runs/EXP_20261111_307605b45817"),
        help="Existing run directory containing dataset.csv and subgraphs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/tds-amazon-onnx"),
        help="Directory for model, metrics, manifest, and shareable ZIP.",
    )
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _load_dataset(experiment_root: Path) -> tuple[Dataset, list[int]]:
    csv_path = experiment_root / "datasets" / "dataset.csv"
    subgraphs_root = experiment_root / "subgraphs"
    if not csv_path.is_file() or not subgraphs_root.is_dir():
        raise FileNotFoundError(
            "Expected datasets/dataset.csv and subgraphs/ under "
            f"{experiment_root}."
        )

    table = pd.read_csv(csv_path)
    required = {
        "graph_id",
        "subgraph_id",
        "sample_id",
        "subgraph_size",
        "degree_mean",
        "degree_max",
        "degree_min",
        "degree_std",
        "target",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)!r}.")

    subgraph_files: dict[str, Path] = {}
    for path in subgraphs_root.rglob("*.json"):
        if path.stem in subgraph_files:
            raise ValueError(f"Duplicate candidate file for sample {path.stem!r}.")
        subgraph_files[path.stem] = path

    samples = []
    groups = []
    for row in table.itertuples(index=False):
        sample_id = str(row.sample_id)
        structure_path = subgraph_files.get(sample_id)
        if structure_path is None:
            raise FileNotFoundError(
                f"No stored subgraph structure was found for {sample_id}."
            )
        structure = json.loads(structure_path.read_text(encoding="utf-8"))
        num_vertices = int(row.subgraph_size)
        num_edges = len(structure["edges"])
        features = {
            "num_vertices": num_vertices,
            "num_edges": num_edges,
            "degree_mean": float(row.degree_mean),
            "degree_max": float(row.degree_max),
            "degree_min": float(row.degree_min),
            "degree_std": float(row.degree_std),
        }
        samples.append(
            Sample(
                graph_id=row.graph_id,
                subgraph_id=int(row.subgraph_id),
                sample_id=sample_id,
                features=features,
                target=float(row.target),
            )
        )
        groups.append(num_vertices)

    dataset = Dataset(samples)
    if tuple(dataset.feature_names) != FEATURE_NAMES:
        raise ValueError(
            f"Unexpected feature order: {dataset.feature_names!r}; "
            f"expected {FEATURE_NAMES!r}."
        )
    return dataset, groups


def _random_parameter_sets(trials: int, seed: int) -> list[dict[str, int | float]]:
    if trials < 1:
        raise ValueError("trials must be positive.")
    rng = random.Random(seed)
    candidates = [BASELINE_PARAMETERS.copy()]
    seen = {json.dumps(BASELINE_PARAMETERS, sort_keys=True)}
    choices = {
        "n_estimators": (120, 200, 350, 500, 700),
        "learning_rate": (0.02, 0.035, 0.05, 0.075, 0.1),
        "num_leaves": (7, 11, 15, 23, 31),
        "min_child_samples": (5, 10, 20, 30, 45),
        "max_depth": (3, 4, 5, 7, -1),
        "reg_alpha": (0.0, 0.05, 0.2, 0.8, 2.0),
        "reg_lambda": (0.0, 0.1, 1.0, 3.0, 8.0),
        "subsample": (0.75, 0.85, 1.0),
        "colsample_bytree": (0.7, 0.85, 1.0),
    }
    while len(candidates) < trials + 1:
        candidate: dict[str, int | float] = {
            key: rng.choice(values) for key, values in choices.items()
        }
        candidate["subsample_freq"] = 1
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    observed = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    correlation = spearmanr(observed, predicted).statistic
    k = max(1, math.ceil(len(observed) * 0.1))
    actual_top = set(np.argpartition(observed, -k)[-k:].tolist())
    predicted_top = set(np.argpartition(predicted, -k)[-k:].tolist())
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "spearman": float(correlation),
        "top_decile_overlap": float(len(actual_top & predicted_top) / k),
        "top_decile_best_value": float(np.max(observed[list(predicted_top)])),
        "optimal_best_value": float(np.max(observed)),
    }


def _index_by_id(dataset: Dataset) -> dict[str, int]:
    return {sample.sample_id: index for index, sample in enumerate(dataset.samples)}


def _cv_score(
    dataset: Dataset,
    groups: list[int],
    parameters: dict[str, int | float],
    n_splits: int,
) -> tuple[float, dict[str, dict[str, list[float]]]]:
    folds = split_dataset_by_ordered_groups(dataset, groups, n_splits=n_splits)
    index_by_id = _index_by_id(dataset)
    scores = {
        split: {metric: [] for metric in ("mae", "rmse", "r2")}
        for split in ("train", "validation")
    }
    for train_part, validation_part in folds:
        train_indices = [index_by_id[sample.sample_id] for sample in train_part.samples]
        validation_indices = [
            index_by_id[sample.sample_id] for sample in validation_part.samples
        ]
        model = lgb.LGBMRegressor(
            **parameters,
            objective="regression",
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(dataset.X.iloc[train_indices], dataset.y[train_indices])
        train_predictions = model.predict(dataset.X.iloc[train_indices])
        validation_predictions = model.predict(dataset.X.iloc[validation_indices])
        train_values = _metrics(dataset.y[train_indices], train_predictions)
        validation_values = _metrics(
            dataset.y[validation_indices], validation_predictions
        )
        for name in scores["train"]:
            scores["train"][name].append(train_values[name])
            scores["validation"][name].append(validation_values[name])
    return fmean(scores["validation"]["mae"]), scores


def _save_model_package(
    output_dir: Path,
    model: lgb.LGBMRegressor,
    dataset: Dataset,
    metrics: dict,
    parameters: dict,
    experiment_root: Path,
    test_count: int,
) -> tuple[Path, float, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    native_path = output_dir / "model.txt"
    model.booster_.save_model(str(native_path))

    feature_types = [("features", FloatTensorType([None, len(FEATURE_NAMES)]))]
    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=feature_types,
        target_opset=15,
        split=100 if model.n_estimators > 100 else None,
    )
    onnx_model.doc_string = (
        "TDS regressor trained on local candidate-subgraph features. "
        "Input is float32 in the order listed in model_manifest.json."
    )
    metadata = {
        "objective": "tds",
        "feature_names": json.dumps(FEATURE_NAMES),
        "feature_scope": "candidate_subgraph_local",
        "model_type": "LightGBMRegressor",
        "training_sample_count": str(dataset.num_samples),
    }
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(onnx_model)

    onnx_path = output_dir / "model.onnx"
    onnx.save_model(onnx_model, onnx_path)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Use up to 1000 records from the final test partition for converter parity.
    test_samples = [
        sample
        for sample in dataset.samples
        if sample.sample_id in set(metrics["test_sample_ids"])
    ]
    parity_features = pd.DataFrame(
        [sample.features for sample in test_samples[: min(test_count, 1000)]],
        columns=FEATURE_NAMES,
    )
    native_predictions = model.predict(parity_features)
    onnx_predictions = session.run(
        [output_name],
        {input_name: parity_features.to_numpy(dtype=np.float32)},
    )[0].reshape(-1)
    absolute_errors = np.abs(native_predictions - onnx_predictions)
    max_error = float(np.max(absolute_errors)) if len(absolute_errors) else 0.0
    mean_error = float(np.mean(absolute_errors)) if len(absolute_errors) else 0.0
    if not np.allclose(native_predictions, onnx_predictions, rtol=1e-4, atol=1e-5):
        raise RuntimeError(
            "ONNX Runtime predictions differ from LightGBM: "
            f"max_abs_error={max_error:.8g}."
        )

    manifest = {
        "model_file": "model.onnx",
        "native_lightgbm_file": "model.txt",
        "format": "ONNX",
        "onnx_opset": 15,
        "runtime": "ONNX Runtime CPU",
        "feature_names_in_order": list(FEATURE_NAMES),
        "input_name": input_name,
        "input_dtype": "float32",
        "input_shape": [None, len(FEATURE_NAMES)],
        "output_name": output_name,
        "target": "TDS = number_of_triangles / num_vertices",
        "features": {
            "num_vertices": "candidate subgraph vertex count",
            "num_edges": "candidate subgraph edge count",
            "degree_mean": "mean candidate-local degree",
            "degree_max": "maximum candidate-local degree",
            "degree_min": "minimum candidate-local degree",
            "degree_std": "population standard deviation of candidate-local degree",
        },
        "training_data": {
            "source_experiment": experiment_root.name,
            "source_dataset": str(
                (experiment_root / "datasets" / "dataset.csv").resolve()
            ),
        "training_sample_count": str(dataset.num_samples),
            "final_test_sample_count": test_count,
            "final_test_grouping": "largest ordered subgraph sizes",
            "fit_after_test_evaluation": True,
        },
        "selected_hyperparameters": parameters,
        "metrics": metrics["summary"],
        "onnx_parity": {
            "records_compared": len(parity_features),
            "max_absolute_error": max_error,
            "mean_absolute_error": mean_error,
            "allclose_rtol": 1e-4,
            "allclose_atol": 1e-5,
        },
    }
    (output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics["report"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_usage(output_dir)
    package_path = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("model.onnx", "model.txt", "model_manifest.json", "metrics.json", "USAGE.md"):
            archive.write(output_dir / name, arcname=name)
    return onnx_path, max_error, mean_error


def _write_usage(output_dir: Path) -> None:
    (output_dir / "USAGE.md").write_text(
        """# TDS ONNX model

Install `onnxruntime` and `numpy`, then pass one row per candidate subgraph as
a `float32` matrix. Keep the exact feature order in `model_manifest.json`.

```python
import json
import numpy as np
import onnxruntime as ort

manifest = json.load(open("model_manifest.json", encoding="utf-8"))
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
features = np.asarray([[10, 18, 3.6, 7, 1, 1.2]], dtype=np.float32)
prediction = session.run(
    [manifest["output_name"]],
    {manifest["input_name"]: features},
)[0][0, 0]
print(float(prediction))
```

Feature order: `num_vertices`, `num_edges`, `degree_mean`, `degree_max`,
`degree_min`, `degree_std`.
""",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if not 0 < args.test_ratio < 1:
        raise ValueError("--test-ratio must be between 0 and 1.")

    dataset, groups = _load_dataset(args.experiment_root)
    development, test = split_dataset_by_future_groups(
        dataset,
        groups,
        test_size=args.test_ratio,
        n_splits=args.folds,
    )
    index_by_id = _index_by_id(dataset)
    development_indices = [index_by_id[sample.sample_id] for sample in development.samples]
    test_indices = [index_by_id[sample.sample_id] for sample in test.samples]
    development_groups = [groups[index] for index in development_indices]
    candidates = _random_parameter_sets(args.trials, args.seed)

    scored = []
    for trial_number, parameters in enumerate(candidates, start=1):
        mean_mae, fold_metrics = _cv_score(
            development,
            development_groups,
            parameters,
            args.folds,
        )
        scored.append((mean_mae, parameters, fold_metrics))
        print(
            f"trial {trial_number}/{len(candidates)} "
            f"cv_mae={mean_mae:.6f} params={parameters}",
            flush=True,
        )
    scored.sort(key=lambda item: item[0])
    _, best_parameters, best_fold_metrics = scored[0]

    baseline_model = lgb.LGBMRegressor(
        **BASELINE_PARAMETERS,
        objective="regression",
        random_state=args.seed,
        n_jobs=1,
        verbosity=-1,
    )
    baseline_model.fit(
        dataset.X.iloc[development_indices], dataset.y[development_indices]
    )
    baseline_test_metrics = _metrics(
        dataset.y[test_indices],
        baseline_model.predict(dataset.X.iloc[test_indices]),
    )

    evaluation_model = lgb.LGBMRegressor(
        **best_parameters,
        objective="regression",
        random_state=args.seed,
        n_jobs=1,
        verbosity=-1,
    )
    evaluation_model.fit(
        dataset.X.iloc[development_indices], dataset.y[development_indices]
    )
    test_predictions = evaluation_model.predict(dataset.X.iloc[test_indices])
    test_metrics = _metrics(dataset.y[test_indices], test_predictions)
    test_ids = [test.samples[index].sample_id for index in range(test.num_samples)]

    report = {
        "model_selection": {
            "objective": "mean forward-fold validation MAE",
            "candidate_count": len(candidates),
            "selected_hyperparameters": best_parameters,
            "trials_ranked_by_validation_mae": [
                {
                    "rank": rank,
                    "validation_mae_mean": score,
                    "hyperparameters": candidate,
                }
                for rank, (score, candidate, _) in enumerate(scored, start=1)
            ],
            "folds": [
                {
                    "fold": index + 1,
                    "train": {
                        metric: best_fold_metrics["train"][metric][index]
                        for metric in best_fold_metrics["train"]
                    },
                    "validation": {
                        metric: best_fold_metrics["validation"][metric][index]
                        for metric in best_fold_metrics["validation"]
                    },
                }
                for index in range(args.folds)
            ],
            "train": {
                metric: {
                    "mean": fmean(values),
                    "std": pstdev(values),
                }
                for metric, values in best_fold_metrics["train"].items()
            },
            "validation": {
                metric: {
                    "mean": fmean(values),
                    "std": pstdev(values),
                }
                for metric, values in best_fold_metrics["validation"].items()
            },
        },
        "test": {
            "num_samples": len(test_indices),
            "size_distribution": {
                str(group): groups.count(group)
                for group in sorted(set(groups[index] for index in test_indices))
            },
            "metrics": test_metrics,
            "baseline_metrics_same_features": baseline_test_metrics,
        },
        "final_model": "refit_on_all_labeled_samples_after_temporal_test",
    }
    # Test scores above are sealed before the final all-data deployment fit.
    final_model = lgb.LGBMRegressor(
        **best_parameters,
        objective="regression",
        random_state=args.seed,
        n_jobs=1,
        verbosity=-1,
    )
    final_model.fit(dataset.X, dataset.y)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "selected_hyperparameters": best_parameters,
        "test_sample_ids": test_ids,
        "summary": {
            "validation_mae_mean": fmean(
                best_fold_metrics["validation"]["mae"]
            ),
            "validation_mae_std": pstdev(
                best_fold_metrics["validation"]["mae"]
            ),
            **{f"test_{key}": value for key, value in test_metrics.items()},
            **{
                f"baseline_test_{key}": value
                for key, value in baseline_test_metrics.items()
            },
        },
        "report": report,
    }
    metrics_bundle = {**report, **metadata}
    onnx_path, max_error, mean_error = _save_model_package(
        args.output_dir,
        final_model,
        dataset,
        metrics_bundle,
        best_parameters,
        args.experiment_root,
        len(test_indices),
    )
    print(f"ONNX model: {onnx_path}")
    print(f"Share package: {args.output_dir.with_suffix('.zip')}")
    print(f"ONNX parity max_abs_error={max_error:.8g} mean_abs_error={mean_error:.8g}")
    print(json.dumps(report["test"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
