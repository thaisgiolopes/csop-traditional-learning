"""Reusable runner for configured CSOP experiments."""

from collections.abc import Callable, Iterable
from contextlib import nullcontext
import csv
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import Path
import traceback
from typing import Any

import networkx as nx

from src.dataset.builder import DatasetBuilder
from src.dataset.dataset import Dataset
from src.dataset.loader import DatasetLoader
from src.dataset.splitter import split_dataset_three_way
from src.evaluation.evaluator import Evaluator
from src.evaluation.result import EvaluationResult
from src.experiments.report import ExperimentReport
from src.features.base import FeatureScope
from src.features.engine import FeatureEngine
from src.features.graph_features import NumEdgesFeature, NumVerticesFeature
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
    if feature_set_id == "graph-size-and-local-degree-v1":
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
    }
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
        available_feature_ids = {
            feature.name.removesuffix("_local").removesuffix("_global")
            for feature in available_node_features
        }
        requested_feature_ids = (
            available_feature_ids
            if enabled_features is None
            else set(enabled_features)
        )
        unknown_features = requested_feature_ids - available_feature_ids
        if unknown_features:
            raise ValueError(
                "Unknown enabled features for this feature set: "
                f"{sorted(unknown_features)!r}."
            )
        node_features = [
            feature
            for feature in available_node_features
            if feature.name.removesuffix("_local").removesuffix("_global")
            in requested_feature_ids
        ]
        feature_engine = FeatureEngine(
            features=[NumVerticesFeature(), NumEdgesFeature(), *node_features]
        )
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

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_GRAPH_SOURCES = {
    "amazon": _REPOSITORY_ROOT / "data" / "raw" / "amazon",
    "citeseer": _REPOSITORY_ROOT / "data" / "raw" / "citeseer",
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


def run_experiment(
    config: ExperimentConfig,
    *,
    experiments_root: str | Path | None = None,
    database_path: str | Path | None = None,
    report_path: str | Path | None = None,
    processed_root: str | Path | None = None,
    model_factory: ModelFactory,
    model_serializer: ModelSerializer | None = None,
) -> str:
    """Run one experiment described by ExperimentConfig and return its ID."""
    raw_graph_path = _resolve_graph_path(config)
    graph_id = config.graph.graph_id or raw_graph_path.name
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

    if config.dataset_split.strategy != "random":
        raise ValueError(
            f"Unknown dataset split strategy: {config.dataset_split.strategy!r}"
        )

    feature_engine, pooling_strategies = _build_feature_set(
        config.features.feature_set_id,
        config.features.enabled_features,
    )
    subgraph_generator = _build_generator(config.subgraph_generation)

    model = model_factory(config.model)

    effective_config = ExperimentConfig(
        graph=config.graph,
        subgraph_generation=config.subgraph_generation,
        features=config.features,
        dataset_split=config.dataset_split,
        model=config.model,
        name=config.name,
        description=config.description,
        random_seed=config.random_seed,
        output={key: str(value) for key, value in output_paths.items()},
    )

    with ExperimentDatabase(output_paths["database_path"]) as database:
        experiment = ExperimentManager.create(
            config=effective_config,
            experiments_root=output_paths["experiments_root"],
            database=database,
            report_path=output_paths["report_path"],
            console_logging=False,
        )
        logger = experiment.logger

        try:
            with experiment:
                logger.info("Experiment ID: %s", experiment.experiment_id)

                with experiment.track_stage("graph_loading"):
                    graph_loader = GraphLoader(raw_graph_path)
                    graph = graph_loader.load()

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
                        logger.info("Registered graph input: %s", stored_path)

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

                sample_records, sample_origins = _resolve_candidate_samples(
                    database,
                    experiment_id=experiment.experiment_id,
                    graph=graph,
                    graph_id=graph_id,
                    generation_config=effective_config.subgraph_generation,
                    generator=subgraph_generator,
                    experiment=experiment,
                )
                candidate_subgraphs = [
                    graph_from_canonical_structure(
                        {"vertices": record.vertices, "edges": record.edges}
                    )
                    for record in sample_records
                ]
                sample_ids = [record.sample_id for record in sample_records]

                with experiment.track_stage("candidate_metadata_persistence"):
                    sample_request = database.get_experiment_sample_request(
                        experiment.experiment_id
                    )
                    _save_json(
                        experiment,
                        "metadata",
                        "sample_request.json",
                        {
                            "compatibility_key": sample_request.compatibility_key,
                            "requested_count": sample_request.requested_count,
                            "reused_count": sample_request.reused_count,
                            "generated_count": sample_request.generated_count,
                            "total_available": sample_request.total_available,
                            "samples": [
                                {
                                    "sample_id": record.sample_id,
                                    "sequence_index": record.sequence_index,
                                    "origin": origin,
                                }
                                for record, origin in zip(
                                    sample_records,
                                    sample_origins,
                                    strict=True,
                                )
                            ],
                        },
                    )

                    for record, subgraph, origin in zip(
                        sample_records,
                        candidate_subgraphs,
                        sample_origins,
                        strict=True,
                    ):
                        _save_json(
                            experiment,
                            "subgraphs",
                            f"{graph_id}/{record.sample_id}.json",
                            {
                                "graph_id": graph_id,
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
                    candidate_subgraphs=candidate_subgraphs,
                    sample_ids=sample_ids,
                    experiment_metadata=experiment_feature_metadata,
                    stage_context_factory=experiment.track_stage,
                )

                if not processed_path.is_file():
                    raise RuntimeError("Dataset CSV was not created.")
                if len(candidate_subgraphs) != built_dataset.num_samples:
                    raise RuntimeError(
                        "Generated subgraphs and dataset sample counts differ."
                    )

                with experiment.track_stage("dataset_loading"):
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
                        "dataset.csv",
                        processed_path,
                    )
                    experiment.register_artifact(
                        "datasets",
                        "dataset.csv",
                        "text/csv",
                    )
                    manifest_path = processed_path.with_suffix(".metadata.json")
                    experiment.artifact_store.copy_file(
                        "datasets",
                        "dataset.metadata.json",
                        manifest_path,
                    )
                    experiment.register_artifact(
                        "datasets",
                        "dataset.metadata.json",
                        "application/json",
                    )

                with experiment.track_stage("train_validation_test_split"):
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

                with experiment.track_stage("split_dataset_persistence"):
                    datasets_by_split = {
                        "train": train_dataset,
                        "validation": validation_dataset,
                        "test": test_dataset,
                    }

                    for split_name, split_part in datasets_by_split.items():
                        filename = f"{split_name}_dataset.csv"
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
                        "split_metadata.json",
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
                            "validation_subgraph_ids": (
                                validation_dataset.subgraph_ids
                            ),
                            "test_subgraph_ids": test_dataset.subgraph_ids,
                        },
                    )

                with experiment.track_stage("model_training"):
                    trained_model = TrainingPipeline(
                        dataset=train_dataset,
                        model=model,
                    ).run()

                    if model_serializer is not None:
                        model_text = model_serializer(trained_model)
                        experiment.artifact_store.save_text(
                            "models",
                            "model.txt",
                            model_text,
                        )
                        experiment.register_artifact(
                            "models",
                            "model.txt",
                            "text/plain",
                        )

                datasets_by_split = {
                    "train": train_dataset,
                    "validation": validation_dataset,
                    "test": test_dataset,
                }
                results_by_split = {}
                metrics_by_split = {}

                for split_name, split_part in datasets_by_split.items():
                    with experiment.track_stage(
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
                            split=split_name,
                        )

                with experiment.track_stage("result_artifact_persistence"):
                    test_result = results_by_split["test"]
                    predictions_path = experiment.artifact_store.save_text(
                        "predictions",
                        "predictions.csv",
                        _predictions_to_csv(test_result.predictions),
                    )
                    experiment.register_artifact(
                        "predictions",
                        predictions_path.name,
                        "text/csv",
                    )

                    metrics_path = _save_json(
                        experiment,
                        "evaluation",
                        "metrics.json",
                        metrics_by_split,
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
                        )

                    logger.info(
                        "Artifacts saved: dataset=%s predictions=%s metrics=%s",
                        stored_dataset,
                        predictions_path,
                        metrics_path,
                    )

            return experiment.experiment_id

        except Exception as error:
            error_path = _save_json(
                experiment,
                "metadata",
                "error.json",
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

            try:
                ExperimentReport(database).update_xlsx(
                    output_paths["report_path"],
                    experiment.experiment_id,
                )
            except Exception:
                logger.exception("Could not update report after experiment failure")

            raise
