"""Reusable runner for configured CSOP experiments."""

from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import replace
import csv
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
import traceback
from typing import Any

import networkx as nx

from src.dataset.builder import DatasetBuilder
from src.dataset.dataset import Dataset
from src.dataset.loader import DatasetLoader
from src.dataset.splitter import (
    split_dataset_by_ordered_groups,
    split_dataset_by_future_groups,
    split_dataset_stratified_holdout,
    split_dataset_stratified_kfold,
    split_dataset_three_way,
)
from src.evaluation.evaluator import Evaluator
from src.evaluation.result import EvaluationResult
from src.experiments.report import ExperimentReport
from src.features.base import FeatureScope
from src.features.engine import FeatureEngine
from src.features.graph_features import (
    NumEdgesFeature,
    NumVerticesFeature,
)
from src.features.node_features import (
    AverageNeighborDegreeFeature,
    BetweennessCentralityFeature,
    ClosenessCentralityFeature,
    DegreeFeature,
    EigenvectorCentralityFeature,
    KCoreNumberFeature,
    LocalClusteringCoefficientFeature,
    NormalizedDegreeFeature,
    PageRankFeature,
    SquareClusteringFeature,
    TriangleCountFeature,
)
from src.features.pooling import (
    MaxPooling,
    MeanPooling,
    MinPooling,
    StdPooling,
)
from src.graph.loader import GraphLoader
from src.model.lightgbm_model import LightGBMModel
from src.model.predictor import Predictor
from src.objectives.tds import TDSObjective
from src.pipeline.dataset_pipeline import DatasetPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.subgraphs.base import IndexedSubgraphGenerator
from src.subgraphs.connected_expansion_generator import (
    ConnectedExpansionSubgraphGenerator,
)
from src.subgraphs.networkx_generator import NetworkXSubgraphGenerator
from src.experiments.config import (
    ExperimentConfig,
    ModelConfig,
    SubgraphGenerationConfig,
)
from src.experiments.database import ExperimentDatabase, SampleRecord
from src.experiments.manager import ExperimentManager
from src.model.base import Model
from src.model.predictor import PredictionResult
from src.graph import Graph
from src.subgraphs.identity import (
    canonical_graph_structure,
    graph_fingerprint,
    graph_from_canonical_structure,
    sample_identity,
)

def _build_feature_set(
    feature_set_id: str,
    enabled_features: list[str] | None = None,
):
    """Resolve a configured feature-set ID to executable components."""
    if feature_set_id in {
        "graph-size-and-local-degree-v1",
        "graph-size-local-degree-and-subgraph-size-v2",
    }:
        unknown_features = set(enabled_features or ()) - {"degree"}
        if unknown_features:
            raise ValueError(
                "Unknown enabled features for this feature set: "
                f"{sorted(unknown_features)!r}."
            )
        include_degree = enabled_features is None or "degree" in enabled_features
        features = [NumVerticesFeature(), NumEdgesFeature()]
        if include_degree:
            features.append(DegreeFeature(scope=FeatureScope.LOCAL))
        feature_engine = FeatureEngine(
            features=features
        )
        pooling_strategies = {}
        if include_degree:
            pooling_strategies["degree"] = (
                MeanPooling(),
                MaxPooling(),
                MinPooling(),
                StdPooling(),
            )
        return feature_engine, pooling_strategies

    node_feature_scopes = {
        "graph-size-and-node-structure-local-v1": FeatureScope.LOCAL,
        "graph-size-and-node-structure-global-v1": FeatureScope.GLOBAL,
        "candidate-structure-local-v2": FeatureScope.LOCAL,
        "candidate-structure-global-v2": FeatureScope.GLOBAL,
    }
    legacy_subgraph_size_set = "-with-subgraph-size-v1" in feature_set_id
    # v2 exposes degree as a selectable node feature. The legacy suffix is
    # retained only so older experiment configurations remain loadable.
    include_degree = legacy_subgraph_size_set or feature_set_id.endswith("-v2")
    if legacy_subgraph_size_set:
        scope_key = feature_set_id.removeprefix(
            "graph-size-node-structure-"
        ).removesuffix("-with-subgraph-size-v1")
        if scope_key not in {"local", "global"}:
            raise ValueError(f"Unknown feature set: {feature_set_id!r}")
        node_feature_scopes[feature_set_id] = (
            FeatureScope.LOCAL if scope_key == "local" else FeatureScope.GLOBAL
        )
    if feature_set_id in node_feature_scopes:
        scope = node_feature_scopes[feature_set_id]
        available_node_features = [
            PageRankFeature(scope=scope),
            BetweennessCentralityFeature(scope=scope),
            ClosenessCentralityFeature(scope=scope),
            EigenvectorCentralityFeature(scope=scope),
            KCoreNumberFeature(scope=scope),
            TriangleCountFeature(scope=scope),
            LocalClusteringCoefficientFeature(scope=scope),
            SquareClusteringFeature(scope=scope),
            AverageNeighborDegreeFeature(scope=scope),
            NormalizedDegreeFeature(scope=scope),
        ]
        feature_by_id = {
            feature.name.removesuffix("_local").removesuffix("_global"): feature
            for feature in available_node_features
        }
        if include_degree:
            feature_by_id["degree"] = DegreeFeature(scope=scope)
        available_feature_ids = set(feature_by_id)
        requested_feature_ids = (
            list(feature_by_id)
            if enabled_features is None
            else list(enabled_features)
        )
        unknown_features = set(requested_feature_ids) - available_feature_ids
        if unknown_features:
            raise ValueError(
                "Unknown enabled features for this feature set: "
                f"{sorted(unknown_features)!r}."
            )
        node_features = [feature_by_id[name] for name in requested_feature_ids]
        graph_features = [NumVerticesFeature(), NumEdgesFeature()]
        feature_engine = FeatureEngine(features=[*graph_features, *node_features])
        pooling_strategies = {
            feature.name: (
                MeanPooling(),
                MaxPooling(),
                MinPooling(),
                StdPooling(),
            )
            for feature in node_features
        }
        return feature_engine, pooling_strategies

    raise ValueError(f"Unknown feature set: {feature_set_id!r}")


def _build_generator(
    config: SubgraphGenerationConfig,
) -> IndexedSubgraphGenerator:
    if config.method == "networkx_random_paths":
        return NetworkXSubgraphGenerator(
            num_subgraphs=config.sample_count,
            path_length=config.path_length or 0,
            seed=config.seed,
        )

    if config.method == "networkx_connected_expansion":
        subgraph_size = config.parameters.get("subgraph_size")
        if isinstance(subgraph_size, bool) or not isinstance(subgraph_size, int):
            raise ValueError(
                "networkx_connected_expansion requires integer "
                "parameters.subgraph_size."
            )
        max_attempts = _generation_attempt_limit(config)
        return ConnectedExpansionSubgraphGenerator(
            num_subgraphs=config.sample_count,
            subgraph_size=subgraph_size,
            seed=config.seed,
            max_attempts=max_attempts,
        )

    raise ValueError(f"Unknown subgraph generation method: {config.method!r}")


def _generation_attempt_limit(config: SubgraphGenerationConfig) -> int:
    """Resolve the bounded generation-attempt limit from method parameters."""
    default_limit = min(max(100, config.sample_count * 50), 100_000)
    if config.method != "networkx_connected_expansion":
        return default_limit

    max_attempts = config.parameters.get("max_attempts", default_limit)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise ValueError("subgraph generation max_attempts must be an integer.")
    if max_attempts <= 0:
        raise ValueError("subgraph generation max_attempts must be positive.")
    return max_attempts


def _sample_compatibility_key(
    graph_digest: str,
    config: SubgraphGenerationConfig,
) -> tuple[str, dict[str, Any]]:
    """Build the compatibility key for structural candidate samples."""
    parameters = config.parameters
    path_length = config.path_length
    if config.method == "networkx_connected_expansion":
        parameters = {
            name: value
            for name, value in config.parameters.items()
            if name != "max_attempts"
        }
        path_length = None

    metadata = {
        "graph_fingerprint": graph_digest,
        "method": config.method,
        "generator_version": 1,
        "networkx_version": nx.__version__,
        "seed": config.seed,
        "path_length": path_length,
        "parameters": parameters,
    }
    serialized = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(serialized).hexdigest(), metadata


def _resolve_candidate_samples(
    database: ExperimentDatabase,
    *,
    experiment_id: str,
    graph: Graph,
    graph_id: str,
    generation_config: SubgraphGenerationConfig,
    generator: IndexedSubgraphGenerator,
    experiment: ExperimentManager | None = None,
) -> tuple[list[SampleRecord], list[str]]:
    """Reuse compatible structures and deterministically generate missing ones."""
    requested_count = generation_config.sample_count
    fingerprint_stage = (
        experiment.track_stage("graph_fingerprinting_and_compatibility")
        if experiment is not None
        else nullcontext()
    )
    with fingerprint_stage:
        graph_digest = graph_fingerprint(graph)
        compatibility_key, generation_metadata = _sample_compatibility_key(
            graph_digest,
            generation_config,
        )

    retrieval_stage = (
        experiment.track_stage("candidate_retrieval")
        if experiment is not None
        else nullcontext()
    )
    with retrieval_stage:
        compatible = database.get_compatible_samples(
            compatibility_key,
            limit=requested_count,
        )

    available_ids = {record.sample_id for record in compatible}
    newly_generated_ids: set[str] = set()
    max_attempts = _generation_attempt_limit(generation_config)
    attempts = 0

    generation_stage = (
        experiment.track_stage("candidate_generation")
        if experiment is not None
        else nullcontext()
    )
    with generation_stage:
        while len(available_ids) < requested_count and attempts < max_attempts:
            sequence_index = next(
                iter(database.reserve_sample_indices(compatibility_key, 1))
            )
            attempts += 1
            candidate = generator.generate_at_index(graph, sequence_index)
            if candidate is None:
                continue

            sample_id = sample_identity(graph_digest, candidate)
            if sample_id in available_ids:
                continue

            inserted = database.register_generated_sample(
                sample_id=sample_id,
                graph_id=graph_id,
                graph_fingerprint=graph_digest,
                structure=canonical_graph_structure(candidate),
                compatibility_key=compatibility_key,
                sequence_index=sequence_index,
                generation_metadata={
                    **generation_metadata,
                    "sequence_index": sequence_index,
                },
            )
            if inserted:
                available_ids.add(sample_id)
                newly_generated_ids.add(sample_id)

    if len(available_ids) < requested_count:
        raise RuntimeError(
            f"Could generate only {len(available_ids)} unique compatible "
            f"samples out of {requested_count} requested after {attempts} "
            "deterministic attempts."
        )

    selection_stage = (
        experiment.track_stage("candidate_selection_and_association")
        if experiment is not None
        else nullcontext()
    )
    with selection_stage:
        all_compatible = database.get_compatible_samples(compatibility_key)
        selected = all_compatible[:requested_count]
        associations = [
            (
                record.sample_id,
                "generated"
                if record.sample_id in newly_generated_ids
                else "reused",
            )
            for record in selected
        ]
        database.associate_samples_with_experiment(
            experiment_id=experiment_id,
            compatibility_key=compatibility_key,
            requested_count=requested_count,
            samples=associations,
            total_available=len(all_compatible),
        )
    return selected, [origin for _, origin in associations]


ModelFactory = Callable[[ModelConfig], Model]
ModelSerializer = Callable[[Model], str]


def _metric_values(evaluation: EvaluationResult) -> dict[str, float | None]:
    """Convert evaluation metrics to finite JSON-compatible values."""
    return {
        name: float(value) if math.isfinite(value) else None
        for name, value in (
            ("mae", evaluation.mae),
            ("rmse", evaluation.rmse),
            ("r2", evaluation.r2),
        )
    }


def _dataset_to_csv(dataset: Dataset) -> str:
    dataframe = dataset.X.copy()
    dataframe.insert(0, "graph_id", dataset.graph_ids)
    dataframe.insert(1, "subgraph_id", dataset.subgraph_ids)
    if any(sample.sample_id is not None for sample in dataset.samples):
        dataframe.insert(
            2,
            "sample_id",
            [sample.sample_id for sample in dataset.samples],
        )
    dataframe["target"] = dataset.y.to_numpy()
    return dataframe.to_csv(index=False, lineterminator="\n")


def _predictions_to_csv(predictions: Iterable[PredictionResult]) -> str:
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "graph_id",
            "subgraph_id",
            "sample_id",
            "predicted_value",
            "actual_value",
            "prediction_error",
        )
    )

    for prediction in predictions:
        actual = prediction.true_target
        error = (
            prediction.predicted_target - actual
            if actual is not None
            else ""
        )
        writer.writerow(
            (
                prediction.graph_id,
                prediction.subgraph_id,
                prediction.sample_id
                or f"{prediction.graph_id}:{prediction.subgraph_id}",
                prediction.predicted_target,
                actual if actual is not None else "",
                error,
            )
        )

    return output.getvalue()


def _save_json(
    experiment: ExperimentManager,
    category: str,
    filename: str,
    data: dict[str, Any],
) -> Path:
    path = experiment.artifact_store.save_json(category, filename, data)
    experiment.register_artifact(category, filename, "application/json")
    return path


def _size_distribution(
    dataset: Dataset,
    sample_sizes: dict[str, int],
) -> dict[str, int]:
    """Count candidate subgraph sizes in a dataset partition."""
    counts: dict[str, int] = {}
    for sample in dataset.samples:
        if sample.sample_id not in sample_sizes:
            raise ValueError(
                f"Missing target-size metadata for sample {sample.sample_id!r}."
            )
        key = str(sample_sizes[sample.sample_id])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _run_size_cross_validation(
    *,
    experiment: ExperimentManager,
    config: ExperimentConfig,
    dataset: Dataset,
    sample_sizes: dict[str, int],
    model_factory: ModelFactory,
    model_serializer: ModelSerializer | None,
    artifact_name: Callable[[str], str],
    track_stage: Callable[[str], Any],
) -> None:
    """Cross-validate on development data and evaluate a held-out test set."""
    n_splits = config.dataset_split.parameters.get("n_splits", 5)
    strata = [sample_sizes[sample.sample_id] for sample in dataset.samples]
    test_ratio = config.dataset_split.test_ratio or 0.2
    with track_stage("external_test_holdout"):
        if config.dataset_split.strategy == "time_series_size":
            development_dataset, test_dataset = split_dataset_by_future_groups(
                dataset,
                strata,
                test_size=test_ratio,
                n_splits=n_splits,
            )
        else:
            development_dataset, test_dataset = split_dataset_stratified_holdout(
                dataset,
                strata,
                test_size=test_ratio,
                random_state=config.dataset_split.random_state,
            )
    development_strata = [
        sample_sizes[sample.sample_id] for sample in development_dataset.samples
    ]
    if config.dataset_split.strategy == "stratified_kfold":
        folds = split_dataset_stratified_kfold(
            development_dataset,
            development_strata,
            n_splits=n_splits,
            random_state=config.dataset_split.random_state,
        )
    elif config.dataset_split.strategy == "time_series_size":
        folds = split_dataset_by_ordered_groups(
            development_dataset,
            development_strata,
            n_splits=n_splits,
        )
    else:
        raise ValueError(
            f"Unsupported size-generalization strategy: "
            f"{config.dataset_split.strategy!r}."
        )
    fold_records = []
    out_of_fold_predictions: list[PredictionResult] = []
    validation_seen: list[str] = []

    with track_stage("split_characterization"):
        split_metadata = {
            "strategy": config.dataset_split.strategy,
            "grouping_key": "subgraph_size",
            "train_test_order": (
                "development_sizes_strictly_below_final_test_sizes"
                if config.dataset_split.strategy == "time_series_size"
                else "stratified_random_folds"
            ),
            "final_test_strategy": (
                "latest_ordered_groups"
                if config.dataset_split.strategy == "time_series_size"
                else "stratified_random_holdout"
            ),
            "n_splits": n_splits,
            "test_ratio": test_ratio,
            "holdout_random_state": config.dataset_split.random_state,
            "random_state": (
                config.dataset_split.random_state
                if config.dataset_split.strategy == "stratified_kfold"
                else None
            ),
            "total_sample_count": dataset.num_samples,
            "development_sample_count": development_dataset.num_samples,
            "test_sample_count": test_dataset.num_samples,
            "development_size_distribution": _size_distribution(
                development_dataset, sample_sizes
            ),
            "test_size_distribution": _size_distribution(test_dataset, sample_sizes),
            "test_subgraph_ids": test_dataset.subgraph_ids,
            "development_subgraph_ids": development_dataset.subgraph_ids,
            "folds": [],
        }
        for fold_index, (train_dataset, fold_validation_dataset) in enumerate(
            folds, start=1
        ):
            train_ids = set(train_dataset.subgraph_ids)
            validation_ids = set(fold_validation_dataset.subgraph_ids)
            if train_ids & validation_ids:
                raise RuntimeError(f"Fold {fold_index} train/test overlap.")
            validation_seen.extend(
                str(sample_id)
                for sample_id in fold_validation_dataset.subgraph_ids
            )
            split_metadata["folds"].append(
                {
                    "fold": fold_index,
                    "train_sample_count": train_dataset.num_samples,
                    "validation_sample_count": fold_validation_dataset.num_samples,
                    "train_size_distribution": _size_distribution(
                        train_dataset, sample_sizes
                    ),
                    "validation_size_distribution": _size_distribution(
                        fold_validation_dataset, sample_sizes
                    ),
                }
            )
        expected_ids = [
            str(sample_id) for sample_id in development_dataset.subgraph_ids
        ]
        if config.dataset_split.strategy == "stratified_kfold":
            expected_evaluation_ids = expected_ids
        else:
            tested_sizes = {
                int(size)
                for fold in split_metadata["folds"]
                for size in fold["validation_size_distribution"]
            }
            expected_evaluation_ids = [
                str(sample.subgraph_id)
                for sample in development_dataset.samples
                if sample_sizes[sample.sample_id] in tested_sizes
            ]
        if sorted(validation_seen) != sorted(expected_evaluation_ids):
            raise RuntimeError(
                "Cross-validation test partitions do not match the expected "
                "sample coverage."
            )
        _save_json(
            experiment,
            "datasets",
            artifact_name("split_metadata.json"),
            split_metadata,
        )

    for fold_index, (train_dataset, fold_validation_dataset) in enumerate(
        folds, start=1
    ):
        with track_stage(f"fold_{fold_index}_model_training"):
            fold_model = model_factory(config.model)
            fold_model = TrainingPipeline(
                dataset=train_dataset,
                model=fold_model,
            ).run()

        with track_stage(f"fold_{fold_index}_evaluation") as stage:
            result = PredictionPipeline(
                model=fold_model,
                dataset=fold_validation_dataset,
                predictor=Predictor(fold_model),
                evaluator=Evaluator(),
            ).run()
            values = _metric_values(result.evaluation)
            out_of_fold_predictions.extend(result.predictions)
            train_metrics = _metric_values(
                Evaluator().evaluate(
                    train_dataset.y,
                    fold_model.predict(train_dataset.X),
                )
            )
            record = {
                "fold": fold_index,
                "train": {
                    "num_samples": train_dataset.num_samples,
                    "metrics": train_metrics,
                },
                "validation": {
                    "num_samples": fold_validation_dataset.num_samples,
                    "metrics": values,
                },
            }
            fold_records.append(record)
        for metric_name, metric_value in values.items():
            if metric_value is not None:
                experiment.register_metric(
                    metric_name,
                    metric_value,
                    stage_id=stage.stage_id,
                    split=f"fold_{fold_index}_validation",
                )
        for metric_name, metric_value in train_metrics.items():
            if metric_value is not None:
                experiment.register_metric(
                    metric_name,
                    metric_value,
                    stage_id=stage.stage_id,
                    split=f"fold_{fold_index}_train",
                )

    metric_names = ("mae", "rmse", "r2")
    values_by_metric = {
        metric: [
            record["validation"]["metrics"][metric]
            for record in fold_records
            if record["validation"]["metrics"][metric] is not None
        ]
        for metric in metric_names
    }
    train_values_by_metric = {
        metric: [
            record["train"]["metrics"][metric]
            for record in fold_records
            if record["train"]["metrics"][metric] is not None
        ]
        for metric in metric_names
    }
    means = {
        metric: fmean(values) if values else None
        for metric, values in values_by_metric.items()
    }
    deviations = {
        metric: pstdev(values) if values else None
        for metric, values in values_by_metric.items()
    }
    train_means = {
        metric: fmean(values) if values else None
        for metric, values in train_values_by_metric.items()
    }
    train_deviations = {
        metric: pstdev(values) if values else None
        for metric, values in train_values_by_metric.items()
    }
    with track_stage("cross_validation_metric_aggregation") as aggregate_stage:
        for metric_name in metric_names:
            if means[metric_name] is not None:
                experiment.register_metric(
                    metric_name,
                    means[metric_name],
                    stage_id=aggregate_stage.stage_id,
                    split="cv_mean",
                )
                experiment.register_metric(
                    metric_name,
                    deviations[metric_name],
                    stage_id=aggregate_stage.stage_id,
                    split="cv_std",
                )

    # Train the evaluation model on development observations only; the
    # external test set remains untouched until its final evaluation.
    with track_stage("final_model_training_on_development_data"):
        final_model = model_factory(config.model)
        final_model = TrainingPipeline(
            dataset=development_dataset, model=final_model
        ).run()

    with track_stage("test_evaluation") as test_stage:
        test_result = PredictionPipeline(
            model=final_model,
            dataset=test_dataset,
            predictor=Predictor(final_model),
            evaluator=Evaluator(),
        ).run()
        test_metrics = _metric_values(test_result.evaluation)
    for metric_name, metric_value in test_metrics.items():
        if metric_value is not None:
            experiment.register_metric(
                metric_name,
                metric_value,
                stage_id=test_stage.stage_id,
                split="test",
            )

    with track_stage("development_training_predictions"):
        development_predictions = Predictor(final_model).predict(
            development_dataset
        )

    with track_stage("result_artifact_persistence"):
        experiment.artifact_store.save_text(
            "predictions",
            artifact_name("cv_predictions.csv"),
            _predictions_to_csv(out_of_fold_predictions),
        )
        experiment.register_artifact(
            "predictions", artifact_name("cv_predictions.csv"), "text/csv"
        )
        experiment.artifact_store.save_text(
            "datasets",
            artifact_name("test_dataset.csv"),
            _dataset_to_csv(test_dataset),
        )
        experiment.register_artifact(
            "datasets", artifact_name("test_dataset.csv"), "text/csv"
        )
        experiment.artifact_store.save_text(
            "predictions",
            artifact_name("predictions.csv"),
            _predictions_to_csv(test_result.predictions),
        )
        experiment.register_artifact(
            "predictions", artifact_name("predictions.csv"), "text/csv"
        )
        experiment.artifact_store.save_text(
            "predictions",
            artifact_name("train_predictions.csv"),
            _predictions_to_csv(development_predictions),
        )
        experiment.register_artifact(
            "predictions", artifact_name("train_predictions.csv"), "text/csv"
        )
        experiment.artifact_store.save_text(
            "predictions",
            artifact_name("validation_predictions.csv"),
            _predictions_to_csv(out_of_fold_predictions),
        )
        experiment.register_artifact(
            "predictions",
            artifact_name("validation_predictions.csv"),
            "text/csv",
        )
        _save_json(
            experiment,
            "evaluation",
            artifact_name("metrics.json"),
            {
                "evaluation_method": config.dataset_split.strategy,
                "n_splits": n_splits,
                "folds": fold_records,
                "train": {
                    "aggregation": "mean_and_std_across_fold_training_sets",
                    "fold_mean_metrics": train_means,
                    "fold_std_metrics": train_deviations,
                },
                "validation": {
                    "aggregation": "out_of_fold_mean_and_std",
                    "num_samples": len(out_of_fold_predictions),
                    "fold_mean_metrics": means,
                    "fold_std_metrics": deviations,
                },
                # Keep these aliases for consumers of the previous schema;
                # they refer to validation-fold aggregates.
                "mean": means,
                "std": deviations,
                "test": {
                    "num_samples": test_result.evaluation.num_samples,
                    "metrics": test_metrics,
                },
                "final_model": "deployment_model_refit_on_all_data_after_test",
            },
        )
        for prediction in development_predictions:
            actual = prediction.true_target
            experiment.register_prediction(
                sample_id=(
                    prediction.sample_id
                    or f"{prediction.graph_id}:{prediction.subgraph_id}"
                ),
                predicted_value=prediction.predicted_target,
                actual_value=actual,
                prediction_error=(
                    prediction.predicted_target - actual
                    if actual is not None
                    else None
                ),
                split="train",
            )
        for prediction in out_of_fold_predictions:
            actual = prediction.true_target
            experiment.register_prediction(
                sample_id=(
                    prediction.sample_id
                    or f"{prediction.graph_id}:{prediction.subgraph_id}"
                ),
                predicted_value=prediction.predicted_target,
                actual_value=actual,
                prediction_error=(
                    prediction.predicted_target - actual
                    if actual is not None
                    else None
                ),
                split="validation",
            )
        for prediction in test_result.predictions:
            actual = prediction.true_target
            error = (
                prediction.predicted_target - actual
                if actual is not None
                else None
            )
            experiment.register_prediction(
                sample_id=(
                    prediction.sample_id
                    or f"{prediction.graph_id}:{prediction.subgraph_id}"
                ),
                predicted_value=prediction.predicted_target,
                actual_value=actual,
                prediction_error=error,
                split="test",
            )

    # Test metrics above are from the model trained only on development data.
    # The deployment artifact then uses every labeled sample available.
    with track_stage("deployment_model_training_on_all_data"):
        deployment_model = model_factory(config.model)
        deployment_model = TrainingPipeline(
            dataset=dataset,
            model=deployment_model,
        ).run()
        if model_serializer is not None:
            model_text = model_serializer(deployment_model)
            experiment.artifact_store.save_text(
                "models", artifact_name("model.txt"), model_text
            )
            experiment.register_artifact(
                "models", artifact_name("model.txt"), "text/plain"
            )
            _save_json(
                experiment,
                "models",
                artifact_name("model_manifest.json"),
                {
                    "model_format": "lightgbm_booster_text",
                    "model_role": "deployment_refit_after_test_evaluation",
                    "objective": config.features.objective_name,
                    "feature_names_in_order": dataset.feature_names,
                    "training_sample_count": dataset.num_samples,
                    "target": "tds",
                },
            )

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_GRAPH_SOURCES = {
    "amazon": _REPOSITORY_ROOT / "data" / "raw" / "amazon",
    "citeseer": _REPOSITORY_ROOT / "data" / "raw" / "citeseer",
    "patents": _REPOSITORY_ROOT / "data" / "raw" / "patents",
}


def _resolve_graph_path(config: ExperimentConfig) -> Path:
    if config.graph.path is not None:
        path = Path(config.graph.path)
        path = path if path.is_absolute() else _REPOSITORY_ROOT / path
    elif config.graph.source in _GRAPH_SOURCES:
        path = _GRAPH_SOURCES[config.graph.source]
    else:
        raise ValueError(
            "GraphConfig requires a path or a supported source "
            "('amazon' or 'citeseer')."
        )

    if not path.is_dir():
        raise FileNotFoundError(f"Graph instance directory not found: {path}")
    if not (path / "metadata").is_file():
        raise FileNotFoundError(f"Graph metadata file not found: {path / 'metadata'}")
    if not (path / "adjlists").is_file() and not (path / "adjlist").is_file():
        raise FileNotFoundError(f"Graph adjacency-list file not found: {path}")
    return path


def _resolve_output_path(
    configured: str | Path | None,
    fallback: str | Path | None,
    key: str,
) -> Path:
    value = configured if configured is not None else fallback
    if value is None:
        raise ValueError(f"Missing output location: {key}.")
    path = Path(value)
    return path if path.is_absolute() else _REPOSITORY_ROOT / path


class _PrefixedStageTracker:
    """Add a configuration label to stages in grouped experiment records."""

    def __init__(self, experiment: ExperimentManager, prefix: str) -> None:
        self._experiment = experiment
        self._prefix = prefix

    def track_stage(self, stage_name: str):
        return self._experiment.track_stage(f"{self._prefix}:{stage_name}")


def run_experiment(
    config: ExperimentConfig,
    *,
    experiments_root: str | Path | None = None,
    database_path: str | Path | None = None,
    report_path: str | Path | None = None,
    processed_root: str | Path | None = None,
    model_factory: ModelFactory,
    model_serializer: ModelSerializer | None = None,
    _shared_experiment: ExperimentManager | None = None,
    _artifact_prefix: str | None = None,
    _store_graph_inputs: bool = True,
    _batch_configs: list[ExperimentConfig] | None = None,
) -> str:
    """Run one configuration or a batch with one model shared across sizes."""
    batch_configs = sorted(
        _batch_configs or [config],
        key=lambda item: item.subgraph_generation.parameters.get(
            "subgraph_size", 0
        ),
    )
    is_batch = len(batch_configs) > 1
    raw_graph_path = _resolve_graph_path(config)
    graph_label = config.graph.source or raw_graph_path.name
    graph_id = (
        f"{graph_label}_multi_size"
        if is_batch
        else config.graph.graph_id or raw_graph_path.name
    )
    split_config = config.dataset_split

    output_paths = {
        "experiments_root": _resolve_output_path(
            config.output.get("experiments_root"),
            experiments_root,
            "experiments_root",
        ),
        "database_path": _resolve_output_path(
            config.output.get("database_path"),
            database_path,
            "database_path",
        ),
        "report_path": _resolve_output_path(
            config.output.get("report_path"),
            report_path,
            "report_path",
        ),
        "processed_root": _resolve_output_path(
            config.output.get("processed_root"),
            processed_root,
            "processed_root",
        ),
    }
    processed_path = output_paths["processed_root"] / f"{graph_id}.csv"

    if config.features.objective_name != "tds":
        raise ValueError(
            f"Unsupported objective: {config.features.objective_name!r}"
        )

    if config.dataset_split.strategy not in {
        "random",
        "stratified_kfold",
        "time_series_size",
    }:
        raise ValueError(
            f"Unknown dataset split strategy: {config.dataset_split.strategy!r}"
        )

    feature_engine, pooling_strategies = _build_feature_set(
        config.features.feature_set_id,
        config.features.enabled_features,
    )
    subgraph_generator = _build_generator(config.subgraph_generation)

    effective_config = ExperimentConfig(
        graph=config.graph,
        subgraph_generation=config.subgraph_generation,
        features=config.features,
        dataset_split=config.dataset_split,
        model=config.model,
        name=(
            f"csop-tds-{graph_label}-shared-model-{len(batch_configs)}-sizes"
            if is_batch
            else config.name
        ),
        description=(
            "One LightGBM model trained on the combined candidate datasets "
            f"for target sizes {[item.subgraph_generation.parameters['subgraph_size'] for item in batch_configs]} on {graph_label}."
            if is_batch
            else config.description
        ),
        random_seed=config.random_seed,
        output={key: str(value) for key, value in output_paths.items()},
    )

    owns_experiment = _shared_experiment is None
    database_context = (
        ExperimentDatabase(output_paths["database_path"])
        if owns_experiment
        else nullcontext(_shared_experiment.database)
    )
    with database_context as database:
        experiment = _shared_experiment or ExperimentManager.create(
            config=effective_config,
            experiments_root=output_paths["experiments_root"],
            database=database,
            report_path=output_paths["report_path"],
            console_logging=False,
        )
        logger = experiment.logger

        if is_batch and owns_experiment:
            batch_config = {
                "kind": "csop_shared_model_size_batch",
                "graph": graph_label,
                "target_sizes": [
                    item.subgraph_generation.parameters["subgraph_size"]
                    for item in batch_configs
                ],
                "model_scope": "one model trained on combined sizes",
                "configurations": [item.to_dict() for item in batch_configs],
            }
            experiment.artifact_store.save_json(
                "config", "batch_config.json", batch_config
            )
            experiment.register_artifact(
                "config", "batch_config.json", "application/json"
            )

        artifact_prefix = (
            f"{_artifact_prefix.strip('/')}/"
            if _artifact_prefix
            else ""
        )

        def artifact_name(filename: str) -> str:
            return f"{artifact_prefix}{filename}"

        def track_stage(stage_name: str):
            if _artifact_prefix:
                stage_name = f"{_artifact_prefix}:{stage_name}"
            return experiment.track_stage(stage_name)

        try:
            lifecycle = experiment if owns_experiment else nullcontext(experiment)
            with lifecycle:
                logger.info("Experiment ID: %s", experiment.experiment_id)

                with track_stage("graph_loading"):
                    graph_loader = GraphLoader(raw_graph_path)
                    graph = graph_loader.load()

                    if _store_graph_inputs:
                        adjacency_filename = (
                            "adjlists"
                            if (raw_graph_path / "adjlists").is_file()
                            else "adjlist"
                        )
                        for filename in ("metadata", adjacency_filename):
                            source_path = raw_graph_path / filename
                            stored_path = experiment.artifact_store.copy_file(
                                "graphs",
                                f"{graph_id}/{filename}",
                                source_path,
                            )
                            experiment.register_artifact(
                                "graphs",
                                f"{graph_id}/{filename}",
                                "text/plain",
                            )
                            logger.info(
                                "Registered graph input: %s", stored_path
                            )

                        _save_json(
                            experiment,
                            "graphs",
                            f"{graph_id}/metadata.json",
                            {
                                "graph_id": graph_id,
                                "source_path": str(raw_graph_path),
                                "num_vertices": graph.num_vertices,
                                "num_edges": graph.num_edges,
                            },
                        )

                candidate_subgraphs = []
                sample_ids = []
                sample_sizes: dict[str, int] = {}
                for size_config in batch_configs:
                    size = size_config.subgraph_generation.parameters.get(
                        "subgraph_size"
                    )
                    size_prefix = f"k{size}" if is_batch else None
                    size_stage_tracker = (
                        _PrefixedStageTracker(experiment, size_prefix)
                        if size_prefix
                        else experiment
                    )
                    size_generator = _build_generator(
                        size_config.subgraph_generation
                    )
                    size_graph_id = (
                        size_config.graph.graph_id or raw_graph_path.name
                    )
                    records, origins = _resolve_candidate_samples(
                        database,
                        experiment_id=experiment.experiment_id,
                        graph=graph,
                        graph_id=size_graph_id,
                        generation_config=size_config.subgraph_generation,
                        generator=size_generator,
                        experiment=size_stage_tracker,
                    )
                    size_candidates = [
                        graph_from_canonical_structure(
                            {"vertices": record.vertices, "edges": record.edges}
                        )
                        for record in records
                    ]
                    candidate_subgraphs.extend(size_candidates)
                    size_sample_ids = [record.sample_id for record in records]
                    sample_ids.extend(size_sample_ids)
                    sample_sizes.update(
                        {
                            sample_id: candidate.num_vertices
                            for sample_id, candidate in zip(
                                size_sample_ids, size_candidates, strict=True
                            )
                        }
                    )

                    with track_stage(
                        f"{size_prefix + ':' if size_prefix else ''}candidate_metadata_persistence"
                    ):
                        request = database.get_experiment_sample_request(
                            experiment.experiment_id,
                            records[0].compatibility_key,
                        )
                        _save_json(
                            experiment,
                            "metadata",
                            artifact_name(
                                f"{size_prefix + '/' if size_prefix else ''}sample_request.json"
                            ),
                            {
                                "target_size": size,
                                "compatibility_key": request.compatibility_key,
                                "requested_count": request.requested_count,
                                "reused_count": request.reused_count,
                                "generated_count": request.generated_count,
                                "total_available": request.total_available,
                                "samples": [
                                    {
                                        "sample_id": record.sample_id,
                                        "sequence_index": record.sequence_index,
                                        "origin": origin,
                                    }
                                    for record, origin in zip(
                                        records, origins, strict=True
                                    )
                                ],
                            },
                        )

                    for record, subgraph, origin in zip(
                        records, size_candidates, origins, strict=True
                    ):
                        _save_json(
                            experiment,
                            "subgraphs",
                            artifact_name(
                                f"{size_prefix + '/' if size_prefix else ''}{size_graph_id}/{record.sample_id}.json"
                            ),
                            {
                                "graph_id": size_graph_id,
                                "target_size": size,
                                "sample_id": record.sample_id,
                                "sequence_index": record.sequence_index,
                                "origin": origin,
                                "vertices": sorted(subgraph.vertices),
                                "edges": sorted(
                                    [sorted(edge) for edge in subgraph.edges]
                                ),
                            },
                        )

                builder = DatasetBuilder(
                    graph_loader=graph_loader,
                    subgraph_generator=subgraph_generator,
                    feature_engine=feature_engine,
                    pooling_strategies=pooling_strategies,
                    objective_function=TDSObjective(),
                )
                experiment_feature_metadata = {
                    "feature_set_id": effective_config.features.feature_set_id,
                    "feature_names": effective_config.features.feature_names,
                    "enabled_features": effective_config.features.enabled_features,
                    "objective_name": effective_config.features.objective_name,
                    "target_sizes": [
                        item.subgraph_generation.parameters.get("subgraph_size")
                        for item in batch_configs
                    ],
                }
                expected_dataset_configuration = {
                    "feature_pipeline": builder.feature_configuration,
                    "experiment": experiment_feature_metadata,
                }
                built_dataset = DatasetPipeline(
                    dataset_builder=builder,
                    processed_dir=output_paths["processed_root"],
                ).run(
                    graph_id=graph_id,
                    output_path=processed_path,
                    complete_graph=graph,
                    candidate_subgraphs=candidate_subgraphs,
                    sample_ids=sample_ids,
                    experiment_metadata=experiment_feature_metadata,
                    stage_context_factory=track_stage,
                )

                if not processed_path.is_file():
                    raise RuntimeError("Dataset CSV was not created.")
                if len(candidate_subgraphs) != built_dataset.num_samples:
                    raise RuntimeError(
                        "Generated subgraphs and dataset sample counts differ."
                    )
                _save_json(
                    experiment,
                    "datasets",
                    artifact_name("sample_sizes.json"),
                    sample_sizes,
                )

                with track_stage("dataset_loading"):
                    dataset = DatasetLoader(
                        processed_path,
                        expected_configuration=expected_dataset_configuration,
                    ).load()

                    if dataset.feature_names != effective_config.features.feature_names:
                        raise ValueError(
                            "Configured feature_names do not match the generated "
                            f"dataset: {dataset.feature_names!r}"
                        )

                    stored_dataset = experiment.artifact_store.copy_file(
                        "datasets",
                        artifact_name("dataset.csv"),
                        processed_path,
                    )
                    experiment.register_artifact(
                        "datasets",
                        artifact_name("dataset.csv"),
                        "text/csv",
                    )
                    manifest_path = processed_path.with_suffix(".metadata.json")
                    experiment.artifact_store.copy_file(
                        "datasets",
                        artifact_name("dataset.metadata.json"),
                        manifest_path,
                    )
                    experiment.register_artifact(
                        "datasets",
                        artifact_name("dataset.metadata.json"),
                        "application/json",
                    )

                if split_config.strategy in {
                    "stratified_kfold",
                    "time_series_size",
                }:
                    _run_size_cross_validation(
                        experiment=experiment,
                        config=effective_config,
                        dataset=dataset,
                        sample_sizes=sample_sizes,
                        model_factory=model_factory,
                        model_serializer=model_serializer,
                        artifact_name=artifact_name,
                        track_stage=track_stage,
                    )
                    return experiment.experiment_id

                with track_stage("train_validation_test_split"):
                    train_dataset, validation_dataset, test_dataset = (
                        split_dataset_three_way(
                            dataset,
                            test_size=split_config.test_ratio,
                            validation_size=split_config.validation_size,
                            random_state=split_config.random_state,
                        )
                    )

                    split_ids = [
                        set(train_dataset.subgraph_ids),
                        set(validation_dataset.subgraph_ids),
                        set(test_dataset.subgraph_ids),
                    ]
                    if any(
                        split_ids[left] & split_ids[right]
                        for left in range(3)
                        for right in range(left + 1, 3)
                    ):
                        raise RuntimeError("Dataset splits overlap.")

                    if set.union(*split_ids) != set(dataset.subgraph_ids):
                        raise RuntimeError("Dataset split lost samples.")

                with track_stage("split_dataset_persistence"):
                    datasets_by_split = {
                        "train": train_dataset,
                        "validation": validation_dataset,
                        "test": test_dataset,
                    }

                    for split_name, split_part in datasets_by_split.items():
                        filename = artifact_name(f"{split_name}_dataset.csv")
                        experiment.artifact_store.save_text(
                            "datasets",
                            filename,
                            _dataset_to_csv(split_part),
                        )
                        experiment.register_artifact(
                            "datasets",
                            filename,
                            "text/csv",
                        )

                    _save_json(
                        experiment,
                        "datasets",
                        artifact_name("split_metadata.json"),
                        {
                            "strategy": split_config.strategy,
                            "test_ratio": split_config.test_ratio,
                            "validation_size": split_config.validation_size,
                            "random_state": split_config.random_state,
                            "train_ratio": (
                                (1 - split_config.test_ratio)
                                * (1 - split_config.validation_size)
                            ),
                            "validation_ratio": (
                                (1 - split_config.test_ratio)
                                * split_config.validation_size
                            ),
                            "train_subgraph_ids": train_dataset.subgraph_ids,
                            "train_size_distribution": _size_distribution(
                                train_dataset, sample_sizes
                            ),
                            "validation_subgraph_ids": (
                                validation_dataset.subgraph_ids
                            ),
                            "validation_size_distribution": _size_distribution(
                                validation_dataset, sample_sizes
                            ),
                            "test_subgraph_ids": test_dataset.subgraph_ids,
                            "test_size_distribution": _size_distribution(
                                test_dataset, sample_sizes
                            ),
                        },
                    )

                with track_stage("model_training"):
                    model = model_factory(config.model)
                    trained_model = TrainingPipeline(
                        dataset=train_dataset,
                        model=model,
                    ).run()

                datasets_by_split = {
                    "train": train_dataset,
                    "validation": validation_dataset,
                    "test": test_dataset,
                }
                results_by_split = {}
                metrics_by_split = {}

                for split_name, split_part in datasets_by_split.items():
                    with track_stage(
                        f"{split_name}_evaluation"
                    ) as stage:
                        result = PredictionPipeline(
                            model=trained_model,
                            dataset=split_part,
                            predictor=Predictor(trained_model),
                            evaluator=Evaluator(),
                        ).run()
                        results_by_split[split_name] = result
                        split_metrics = _metric_values(result.evaluation)
                        metrics_by_split[split_name] = {
                            "num_samples": result.evaluation.num_samples,
                            "metrics": split_metrics,
                        }

                    for metric_name, metric_value in split_metrics.items():
                        if metric_value is None:
                            logger.warning(
                                "Metric %s/%s is not finite and was not registered",
                                split_name,
                                metric_name,
                            )
                            continue
                        experiment.register_metric(
                            metric_name,
                            metric_value,
                            stage_id=stage.stage_id,
                            split=(
                                f"{_artifact_prefix}:{split_name}"
                                if _artifact_prefix
                                else split_name
                            ),
                        )

                per_size_test_metrics = {}
                if is_batch:
                    for size in (
                        item.subgraph_generation.parameters["subgraph_size"]
                        for item in batch_configs
                    ):
                        size_test_samples = [
                            sample
                            for sample in test_dataset.samples
                            if sample_sizes.get(sample.sample_id) == size
                        ]
                        if not size_test_samples:
                            per_size_test_metrics[str(size)] = {
                                "num_samples": 0,
                                "metrics": {},
                            }
                            continue
                        size_test_dataset = Dataset(
                            size_test_samples, metadata=test_dataset.metadata
                        )
                        size_split = f"test:k{size}"
                        with track_stage(
                            f"test_evaluation_k{size}_shared_model"
                        ) as stage:
                            size_result = PredictionPipeline(
                                model=trained_model,
                                dataset=size_test_dataset,
                                predictor=Predictor(trained_model),
                                evaluator=Evaluator(),
                            ).run()
                            values = _metric_values(size_result.evaluation)
                            per_size_test_metrics[str(size)] = {
                                "num_samples": size_result.evaluation.num_samples,
                                "metrics": values,
                            }
                        for metric_name, metric_value in values.items():
                            if metric_value is not None:
                                experiment.register_metric(
                                    metric_name,
                                    metric_value,
                                    stage_id=stage.stage_id,
                                    split=size_split,
                                )

                with track_stage("result_artifact_persistence"):
                    test_result = results_by_split["test"]
                    predictions_path = experiment.artifact_store.save_text(
                        "predictions",
                        artifact_name("predictions.csv"),
                        _predictions_to_csv(test_result.predictions),
                    )
                    experiment.register_artifact(
                        "predictions",
                        artifact_name(predictions_path.name),
                        "text/csv",
                    )

                    metrics_path = _save_json(
                        experiment,
                        "evaluation",
                        artifact_name("metrics.json"),
                        {
                            **metrics_by_split,
                            **(
                                {"test_by_size": per_size_test_metrics}
                                if is_batch
                                else {}
                            ),
                        },
                    )

                    for split_name, split_result in results_by_split.items():
                        for prediction in split_result.predictions:
                            actual = prediction.true_target
                            experiment.register_prediction(
                                sample_id=(
                                    prediction.sample_id
                                    or f"{prediction.graph_id}:{prediction.subgraph_id}"
                                ),
                                predicted_value=prediction.predicted_target,
                                actual_value=actual,
                                prediction_error=(
                                    prediction.predicted_target - actual
                                    if actual is not None
                                    else None
                                ),
                                split=split_name,
                            )

                    for split_name in ("train", "validation"):
                        split_result = results_by_split[split_name]
                        experiment.artifact_store.save_text(
                            "predictions",
                            artifact_name(f"{split_name}_predictions.csv"),
                            _predictions_to_csv(split_result.predictions),
                        )
                        experiment.register_artifact(
                            "predictions",
                            artifact_name(f"{split_name}_predictions.csv"),
                            "text/csv",
                        )

                    logger.info(
                        "Artifacts saved: dataset=%s predictions=%s metrics=%s",
                        stored_dataset,
                        predictions_path,
                        metrics_path,
                    )

                # Test metrics above come from the train-only model. Refit the
                # deployable artifact on all labeled samples after evaluation.
                with track_stage("deployment_model_training_on_all_data"):
                    deployment_model = model_factory(config.model)
                    deployment_model = TrainingPipeline(
                        dataset=dataset, model=deployment_model
                    ).run()
                    if model_serializer is not None:
                        model_text = model_serializer(deployment_model)
                        experiment.artifact_store.save_text(
                            "models", artifact_name("model.txt"), model_text
                        )
                        experiment.register_artifact(
                            "models", artifact_name("model.txt"), "text/plain"
                        )
                        _save_json(
                            experiment,
                            "models",
                            artifact_name("model_manifest.json"),
                            {
                                "model_format": "lightgbm_booster_text",
                                "model_role": "deployment_refit_after_test_evaluation",
                                "objective": config.features.objective_name,
                                "feature_names_in_order": dataset.feature_names,
                                "training_sample_count": dataset.num_samples,
                                "target": "tds",
                            },
                        )

            return experiment.experiment_id

        except Exception as error:
            error_path = _save_json(
                experiment,
                "metadata",
                artifact_name("error.json"),
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": "".join(
                        traceback.format_exception(
                            type(error),
                            error,
                            error.__traceback__,
                        )
                    ),
                },
            )
            logger.error(
                "Experiment failed; error details saved to %s",
                error_path,
            )

            if owns_experiment:
                try:
                    ExperimentReport(database).update_xlsx(
                        output_paths["report_path"],
                        experiment.experiment_id,
                    )
                except Exception:
                    logger.exception(
                        "Could not update report after experiment failure"
                    )

            raise


def run_experiment_batch(
    configs: Iterable[ExperimentConfig],
    *,
    model_factory: ModelFactory,
    model_serializer: ModelSerializer | None = None,
) -> str:
    """Train one model on the combined dataset from multiple target sizes."""
    selected = list(configs)
    if not selected:
        raise ValueError("An experiment batch must contain at least one config.")
    sizes = [
        config.subgraph_generation.parameters.get("subgraph_size")
        for config in selected
    ]
    if len(sizes) != len(set(sizes)):
        raise ValueError("An experiment batch cannot repeat target sizes.")

    first = selected[0]
    shared = first.to_dict()
    shared["graph"].pop("graph_id", None)
    shared["subgraph_generation"]["parameters"].pop("subgraph_size", None)
    shared["subgraph_generation"]["parameters"].pop("max_attempts", None)
    shared["name"] = None
    shared["description"] = None
    for config in selected[1:]:
        candidate = config.to_dict()
        candidate["graph"].pop("graph_id", None)
        candidate["subgraph_generation"]["parameters"].pop("subgraph_size", None)
        candidate["subgraph_generation"]["parameters"].pop("max_attempts", None)
        candidate["name"] = None
        candidate["description"] = None
        if candidate != shared:
            raise ValueError(
                "Configs in one experiment batch must share graph, sample "
                "count, generation identity, features, split, model, seed, "
                "and output settings; only target size and max_attempts may vary."
            )
    return run_experiment(
        first,
        model_factory=model_factory,
        model_serializer=model_serializer,
        _batch_configs=selected,
    )


def _build_transfer_dataset(
    *,
    experiment: ExperimentManager,
    database: ExperimentDatabase,
    configs: list[ExperimentConfig],
    role: str,
    output_paths: dict[str, Path],
    artifact_name: Callable[[str], str],
) -> tuple[Dataset, dict[str, int]]:
    """Generate and persist candidates and features for one transfer graph."""
    first = configs[0]
    graph_name = first.graph.source or first.graph.graph_id or role
    raw_graph_path = _resolve_graph_path(first)
    graph_loader = GraphLoader(raw_graph_path)

    with experiment.track_stage(f"{role}_{graph_name}_graph_loading"):
        graph = graph_loader.load()
        adjacency_filename = (
            "adjlists" if (raw_graph_path / "adjlists").is_file() else "adjlist"
        )
        for filename in ("metadata", adjacency_filename):
            relative = f"{role}/{graph_name}/{filename}"
            experiment.artifact_store.copy_file(
                "graphs", artifact_name(relative), raw_graph_path / filename
            )
            experiment.register_artifact(
                "graphs", artifact_name(relative), "text/plain"
            )

    candidates: list[Graph] = []
    sample_ids: list[str] = []
    sample_sizes: dict[str, int] = {}
    for config in sorted(
        configs,
        key=lambda item: item.subgraph_generation.parameters["subgraph_size"],
    ):
        size = config.subgraph_generation.parameters["subgraph_size"]
        graph_id = config.graph.graph_id or graph_name
        tracker = _PrefixedStageTracker(
            experiment, f"{role}_{graph_name}_k{size}"
        )
        records, origins = _resolve_candidate_samples(
            database,
            experiment_id=experiment.experiment_id,
            graph=graph,
            graph_id=graph_id,
            generation_config=config.subgraph_generation,
            generator=_build_generator(config.subgraph_generation),
            experiment=tracker,
        )
        size_candidates = [
            graph_from_canonical_structure(
                {"vertices": record.vertices, "edges": record.edges}
            )
            for record in records
        ]
        size_sample_ids = [record.sample_id for record in records]
        candidates.extend(size_candidates)
        sample_ids.extend(size_sample_ids)
        sample_sizes.update(
            {
                sample_id: candidate.num_vertices
                for sample_id, candidate in zip(
                    size_sample_ids, size_candidates, strict=True
                )
            }
        )
        request = database.get_experiment_sample_request(
            experiment.experiment_id,
            records[0].compatibility_key,
        )
        _save_json(
            experiment,
            "metadata",
            artifact_name(f"{role}/{graph_name}/k{size}/sample_request.json"),
            {
                "graph": graph_name,
                "target_size": size,
                "compatibility_key": request.compatibility_key,
                "requested_count": request.requested_count,
                "reused_count": request.reused_count,
                "generated_count": request.generated_count,
                "total_available": request.total_available,
                "samples": [
                    {
                        "sample_id": record.sample_id,
                        "sequence_index": record.sequence_index,
                        "origin": origin,
                    }
                    for record, origin in zip(records, origins, strict=True)
                ],
            },
        )
        for record, candidate, origin in zip(
            records, size_candidates, origins, strict=True
        ):
            _save_json(
                experiment,
                "subgraphs",
                artifact_name(
                    f"{role}/{graph_name}/k{size}/{record.sample_id}.json"
                ),
                {
                    "graph_id": graph_id,
                    "target_size": size,
                    "sample_id": record.sample_id,
                    "sequence_index": record.sequence_index,
                    "origin": origin,
                    "vertices": sorted(candidate.vertices),
                    "edges": sorted([sorted(edge) for edge in candidate.edges]),
                },
            )

    feature_engine, pooling_strategies = _build_feature_set(
        first.features.feature_set_id,
        first.features.enabled_features,
    )
    builder = DatasetBuilder(
        graph_loader=graph_loader,
        subgraph_generator=_build_generator(first.subgraph_generation),
        feature_engine=feature_engine,
        pooling_strategies=pooling_strategies,
        objective_function=TDSObjective(),
    )
    feature_metadata = {
        "feature_set_id": first.features.feature_set_id,
        "feature_names": first.features.feature_names,
        "enabled_features": first.features.enabled_features,
        "objective_name": first.features.objective_name,
        "target_sizes": sorted(
            item.subgraph_generation.parameters["subgraph_size"]
            for item in configs
        ),
    }
    expected_configuration = {
        "feature_pipeline": builder.feature_configuration,
        "experiment": feature_metadata,
    }
    processed_path = output_paths["processed_root"] / (
        f"{experiment.experiment_id}_{role}_{graph_name}.csv"
    )
    with experiment.track_stage(f"{role}_{graph_name}_feature_extraction"):
        built = DatasetPipeline(
            dataset_builder=builder,
            processed_dir=output_paths["processed_root"],
        ).run(
            graph_id=f"{role}_{graph_name}_multi_size",
            output_path=processed_path,
            complete_graph=graph,
            candidate_subgraphs=candidates,
            sample_ids=sample_ids,
            experiment_metadata=feature_metadata,
            stage_context_factory=experiment.track_stage,
        )
    if built.num_samples != len(sample_ids):
        raise RuntimeError(
            f"{role} graph dataset sample count differs from candidate count."
        )

    loaded = DatasetLoader(
        processed_path,
        expected_configuration=expected_configuration,
    ).load()
    if loaded.feature_names != first.features.feature_names:
        raise ValueError(
            f"Configured feature names do not match {role} graph dataset: "
            f"{loaded.feature_names!r}."
        )
    filename = f"{role}_{graph_name}_dataset.csv"
    experiment.artifact_store.copy_file(
        "datasets", artifact_name(filename), processed_path
    )
    experiment.register_artifact(
        "datasets", artifact_name(filename), "text/csv"
    )
    manifest_name = f"{role}_{graph_name}_dataset.metadata.json"
    experiment.artifact_store.copy_file(
        "datasets",
        artifact_name(manifest_name),
        processed_path.with_suffix(".metadata.json"),
    )
    experiment.register_artifact(
        "datasets", artifact_name(manifest_name), "application/json"
    )
    return loaded, sample_sizes


def run_graph_transfer_experiment(
    configs: Iterable[ExperimentConfig],
    *,
    train_graph: str,
    test_graph: str,
    model_factory: ModelFactory,
    model_serializer: ModelSerializer | None = None,
) -> str:
    """Train on one graph and evaluate on a separate, unseen input graph."""
    if train_graph == test_graph:
        raise ValueError("Training and test graphs must differ.")
    selected = list(configs)
    train_configs = sorted(
        [config for config in selected if config.graph.source == train_graph],
        key=lambda item: item.subgraph_generation.parameters["subgraph_size"],
    )
    test_configs = sorted(
        [config for config in selected if config.graph.source == test_graph],
        key=lambda item: item.subgraph_generation.parameters["subgraph_size"],
    )
    if not train_configs or not test_configs:
        raise ValueError("Configs must include both selected transfer graphs.")
    if [c.subgraph_generation.parameters["subgraph_size"] for c in train_configs] != [
        c.subgraph_generation.parameters["subgraph_size"] for c in test_configs
    ]:
        raise ValueError("Training and test graphs must use identical target sizes.")

    first = train_configs[0]
    shared_reference = first.to_dict()
    shared_reference["graph"] = {}
    shared_reference["name"] = None
    shared_reference["description"] = None
    shared_reference["subgraph_generation"]["parameters"].pop("subgraph_size", None)
    shared_reference["subgraph_generation"]["parameters"].pop("max_attempts", None)
    for config in train_configs + test_configs:
        candidate = config.to_dict()
        candidate["graph"] = {}
        candidate["name"] = None
        candidate["description"] = None
        candidate["subgraph_generation"]["parameters"].pop("subgraph_size", None)
        candidate["subgraph_generation"]["parameters"].pop("max_attempts", None)
        if candidate != shared_reference:
            raise ValueError(
                "Transfer configs must share sample count, generation method, "
                "features, model, seed, and output settings."
            )

    output_paths = {
        key: _resolve_output_path(first.output.get(key), None, key)
        for key in (
            "experiments_root",
            "database_path",
            "report_path",
            "processed_root",
        )
    }
    transfer_config = ExperimentConfig(
        graph=first.graph,
        subgraph_generation=first.subgraph_generation,
        features=first.features,
        dataset_split=replace(
            first.dataset_split,
            strategy="graph_transfer",
            test_ratio=None,
            validation_size=None,
            parameters={"train_graph": train_graph, "test_graph": test_graph},
        ),
        model=first.model,
        name=f"csop-tds-transfer-{train_graph}-to-{test_graph}",
        description=(
            f"Train on graph {train_graph}; evaluate on unseen graph {test_graph} "
            f"for sizes {[c.subgraph_generation.parameters['subgraph_size'] for c in train_configs]}."
        ),
        random_seed=first.random_seed,
        output={key: str(value) for key, value in output_paths.items()},
    )
    with ExperimentDatabase(output_paths["database_path"]) as database:
        experiment = ExperimentManager.create(
            config=transfer_config,
            experiments_root=output_paths["experiments_root"],
            database=database,
            report_path=output_paths["report_path"],
            console_logging=False,
        )
        batch_config = {
            "kind": "graph_transfer",
            "train_graph": train_graph,
            "test_graph": test_graph,
            "target_sizes": [
                config.subgraph_generation.parameters["subgraph_size"]
                for config in train_configs
            ],
            "model_scope": "fit on training graph only",
            "configurations": [config.to_dict() for config in selected],
            "transfer_split": transfer_config.dataset_split.to_dict()
            if hasattr(transfer_config.dataset_split, "to_dict")
            else {
                "strategy": "graph_transfer",
                "train_graph": train_graph,
                "test_graph": test_graph,
            },
        }
        experiment.artifact_store.save_json(
            "config", "transfer_config.json", batch_config
        )
        experiment.register_artifact(
            "config", "transfer_config.json", "application/json"
        )
        with experiment:
            train_dataset, train_sample_sizes = _build_transfer_dataset(
                experiment=experiment,
                database=database,
                configs=train_configs,
                role="train",
                output_paths=output_paths,
                artifact_name=lambda filename: filename,
            )
            test_dataset, test_sample_sizes = _build_transfer_dataset(
                experiment=experiment,
                database=database,
                configs=test_configs,
                role="test",
                output_paths=output_paths,
                artifact_name=lambda filename: filename,
            )
            if train_dataset.feature_names != test_dataset.feature_names:
                raise ValueError("Transfer graph feature schemas do not match.")
            train_sample_ids = {sample.sample_id for sample in train_dataset.samples}
            test_sample_ids = {sample.sample_id for sample in test_dataset.samples}
            if train_sample_ids & test_sample_ids:
                raise RuntimeError("Training and test graph samples overlap.")

            with experiment.track_stage("graph_transfer_split_characterization"):
                _save_json(
                    experiment,
                    "datasets",
                    "split_metadata.json",
                    {
                        "strategy": "graph_transfer",
                        "train_graph": train_graph,
                        "test_graph": test_graph,
                        "train_sample_count": train_dataset.num_samples,
                        "test_sample_count": test_dataset.num_samples,
                        "train_size_distribution": _size_distribution(
                            train_dataset, train_sample_sizes
                        ),
                        "test_size_distribution": _size_distribution(
                            test_dataset, test_sample_sizes
                        ),
                        "train_sample_ids": sorted(train_sample_ids),
                        "test_sample_ids": sorted(test_sample_ids),
                    },
                )

            with experiment.track_stage("graph_transfer_model_training"):
                model = model_factory(transfer_config.model)
                model = TrainingPipeline(
                    dataset=train_dataset,
                    model=model,
                ).run()
                if model_serializer is not None:
                    experiment.artifact_store.save_text(
                        "models", "model.txt", model_serializer(model)
                    )
                    experiment.register_artifact(
                        "models", "model.txt", "text/plain"
                    )

            with experiment.track_stage("unseen_graph_evaluation") as stage:
                result = PredictionPipeline(
                    model=model,
                    dataset=test_dataset,
                    predictor=Predictor(model),
                    evaluator=Evaluator(),
                ).run()
                metrics = _metric_values(result.evaluation)
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    experiment.register_metric(
                        metric_name,
                        metric_value,
                        stage_id=stage.stage_id,
                        split="test_unseen_graph",
                    )

            metrics_by_size = {}
            for size in sorted(set(test_sample_sizes.values())):
                size_samples = [
                    sample
                    for sample in test_dataset.samples
                    if test_sample_sizes[sample.sample_id] == size
                ]
                size_dataset = Dataset(size_samples, metadata=test_dataset.metadata)
                with experiment.track_stage(
                    f"unseen_graph_evaluation_k{size}"
                ) as size_stage:
                    size_result = PredictionPipeline(
                        model=model,
                        dataset=size_dataset,
                        predictor=Predictor(model),
                        evaluator=Evaluator(),
                    ).run()
                    size_metrics = _metric_values(size_result.evaluation)
                    metrics_by_size[str(size)] = {
                        "num_samples": size_dataset.num_samples,
                        "metrics": size_metrics,
                    }
                for metric_name, metric_value in size_metrics.items():
                    if metric_value is not None:
                        experiment.register_metric(
                            metric_name,
                            metric_value,
                            stage_id=size_stage.stage_id,
                            split=f"test_unseen_graph:k{size}",
                        )

            with experiment.track_stage("graph_transfer_result_persistence"):
                predictions_path = experiment.artifact_store.save_text(
                    "predictions",
                    "predictions.csv",
                    _predictions_to_csv(result.predictions),
                )
                experiment.register_artifact(
                    "predictions", "predictions.csv", "text/csv"
                )
                _save_json(
                    experiment,
                    "evaluation",
                    "metrics.json",
                    {
                        "evaluation_method": "graph_transfer",
                        "train_graph": train_graph,
                        "test_graph": test_graph,
                        "num_samples": test_dataset.num_samples,
                        "metrics": metrics,
                        "test_by_size": metrics_by_size,
                    },
                )
                for prediction in result.predictions:
                    actual = prediction.true_target
                    experiment.register_prediction(
                        sample_id=(
                            prediction.sample_id
                            or f"{prediction.graph_id}:{prediction.subgraph_id}"
                        ),
                        predicted_value=prediction.predicted_target,
                        actual_value=actual,
                        prediction_error=(
                            prediction.predicted_target - actual
                            if actual is not None
                            else None
                        ),
                        split="test_unseen_graph",
                    )
    return experiment.experiment_id
