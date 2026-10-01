"""Run the CSOP pipeline and persist experiment artifacts and metadata."""

import csv
from io import StringIO
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.builder import DatasetBuilder
from src.dataset.dataset import Dataset
from src.dataset.loader import DatasetLoader
from src.dataset.splitter import split_dataset
from src.evaluation.evaluator import Evaluator
from src.features.base import FeatureScope
from src.features.engine import FeatureEngine
from src.features.graph_features import (
    NumEdgesFeature,
    NumVerticesFeature,
)
from src.features.node_features import DegreeFeature
from src.features.pooling import (
    MaxPooling,
    MeanPooling,
    MinPooling,
    StdPooling,
)
from src.graph.loader import GraphLoader
from src.model.base import Model
from src.model.lightgbm_model import LightGBMModel
from src.model.predictor import Predictor
from src.objectives.tds import TDSObjective
from src.pipeline.dataset_pipeline import DatasetPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.subgraphs.networkx_generator import NetworkXSubgraphGenerator
from src.experiments.config import (
    DatasetSplitConfig,
    ExperimentConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    SubgraphGenerationConfig,
)
from src.experiments.database import ExperimentDatabase
from src.experiments.manager import ExperimentManager


GRAPH_ID = "integration_test_graph"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments" / "runs"
DATABASE_PATH = PROJECT_ROOT / "database" / "experiments.sqlite3"
REPORT_PATH = PROJECT_ROOT / "experiments" / "reports" / "experiments.xlsx"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
RAW_GRAPH_PATH = PROJECT_ROOT / "data" / "raw" / "graph_test"

NUM_SUBGRAPHS = 8
PATH_LENGTH = 3
GENERATION_SEED = 42
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42

MODEL_PARAMETERS = {
    "n_estimators": 5,
    "learning_rate": 0.1,
    "num_leaves": 5,
    "min_child_samples": 1,
    "verbosity": -1,
    "random_state": 42,
}

FEATURE_NAMES = [
    "num_vertices",
    "num_edges",
    "degree_mean",
    "degree_max",
    "degree_min",
    "degree_std",
]


def make_experiment_config() -> ExperimentConfig:
    """Create the reproducibility configuration for this run."""
    return ExperimentConfig(
        name="CSOP TDS baseline",
        description="LightGBM prediction of TDS for candidate subgraphs.",
        random_seed=GENERATION_SEED,
        graph=GraphConfig(
            graph_id=GRAPH_ID,
            path=RAW_GRAPH_PATH,
            parameters={
                "format": "adjlist",
                "directed": False,
            },
        ),
        subgraph_generation=SubgraphGenerationConfig(
            method="networkx_random_paths",
            sample_count=NUM_SUBGRAPHS,
            path_length=PATH_LENGTH,
            seed=GENERATION_SEED,
        ),
        features=FeatureConfig(
            feature_set_id="graph-size-and-local-degree-v1",
            feature_names=FEATURE_NAMES,
            objective_name="tds",
        ),
        dataset_split=DatasetSplitConfig(
            strategy="random",
            train_ratio=1.0 - TEST_SIZE,
            test_ratio=TEST_SIZE,
            seed=SPLIT_RANDOM_STATE,
        ),
        model=ModelConfig(
            model_type="lightgbm",
            hyperparameters=MODEL_PARAMETERS,
        ),
    )


def dataset_to_csv(dataset: Dataset) -> str:
    """Serialize a Dataset using the processed CSV column layout."""
    dataframe = dataset.X.copy()
    dataframe.insert(0, "graph_id", dataset.graph_ids)
    dataframe.insert(1, "subgraph_id", dataset.subgraph_ids)
    dataframe["target"] = dataset.y.to_numpy()
    return dataframe.to_csv(index=False, lineterminator="\n")


def predictions_to_csv(predictions) -> str:
    """Serialize prediction records as UTF-8-compatible CSV text."""
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
        actual_value = prediction.true_target
        prediction_error = (
            prediction.predicted_target - actual_value
            if actual_value is not None
            else ""
        )
        writer.writerow(
            (
                prediction.graph_id,
                prediction.subgraph_id,
                f"{prediction.graph_id}:{prediction.subgraph_id}",
                prediction.predicted_target,
                actual_value if actual_value is not None else "",
                prediction_error,
            )
        )

    return output.getvalue()


def register_json_artifact(
    experiment: ExperimentManager,
    category: str,
    filename: str,
    data: dict,
) -> Path:
    """Save a JSON artifact and register its file metadata."""
    artifact_path = experiment.artifact_store.save_json(
        category,
        filename,
        data,
    )
    experiment.register_artifact(
        category,
        filename,
        "application/json",
    )
    return artifact_path


def main() -> None:
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    config = make_experiment_config()

    with ExperimentDatabase(DATABASE_PATH) as database:
        experiment = ExperimentManager.create(
            config=config,
            experiments_root=EXPERIMENTS_ROOT,
            database=database,
            report_path=REPORT_PATH,
            console_logging=False,
        )

        with experiment:
            logger = experiment.logger
            logger.info("Experiment ID: %s", experiment.experiment_id)

            with experiment.track_stage("graph_loading"):
                assert RAW_GRAPH_PATH.is_dir()
                assert (RAW_GRAPH_PATH / "metadata").is_file()
                assert (RAW_GRAPH_PATH / "adjlist").is_file()

                graph_loader = GraphLoader(RAW_GRAPH_PATH)
                graph = graph_loader.load()

                assert graph.num_vertices == 5
                assert graph.num_edges == 6

                for source_name in ("metadata", "adjlist"):
                    stored_path = experiment.artifact_store.copy_file(
                        "graphs",
                        f"{GRAPH_ID}/{source_name}",
                        RAW_GRAPH_PATH / source_name,
                    )
                    experiment.register_artifact(
                        "graphs",
                        f"{GRAPH_ID}/{source_name}",
                        "text/plain",
                    )
                    logger.info("Registered input graph file: %s", stored_path)

                register_json_artifact(
                    experiment,
                    "graphs",
                    f"{GRAPH_ID}/metadata.json",
                    {
                        "graph_id": GRAPH_ID,
                        "source_path": str(RAW_GRAPH_PATH),
                        "num_vertices": graph.num_vertices,
                        "num_edges": graph.num_edges,
                    },
                )

            with experiment.track_stage("dataset_construction"):
                subgraph_generator = NetworkXSubgraphGenerator(
                    num_subgraphs=NUM_SUBGRAPHS,
                    path_length=PATH_LENGTH,
                    seed=GENERATION_SEED,
                )

                # Persist the candidates using the existing Graph representation.
                # DatasetBuilder generates candidates independently with the same
                # configured seed when it builds the samples.
                candidate_subgraphs = subgraph_generator.generate(graph)
                for subgraph_id, subgraph in enumerate(candidate_subgraphs):
                    edges = sorted(
                        [sorted(edge) for edge in subgraph.edges]
                    )
                    register_json_artifact(
                        experiment,
                        "subgraphs",
                        f"{GRAPH_ID}/subgraph_{subgraph_id:06d}.json",
                        {
                            "graph_id": GRAPH_ID,
                            "subgraph_id": subgraph_id,
                            "vertices": sorted(subgraph.vertices),
                            "edges": edges,
                        },
                    )

                feature_engine = FeatureEngine(
                    features=[
                        NumVerticesFeature(),
                        NumEdgesFeature(),
                        DegreeFeature(scope=FeatureScope.LOCAL),
                    ]
                )
                pooling_strategies = {
                    "degree": (
                        MeanPooling(),
                        MaxPooling(),
                        MinPooling(),
                        StdPooling(),
                    ),
                }
                dataset_builder = DatasetBuilder(
                    graph_loader=graph_loader,
                    subgraph_generator=subgraph_generator,
                    feature_engine=feature_engine,
                    pooling_strategies=pooling_strategies,
                    objective_function=TDSObjective(),
                )

                processed_dataset_path = (
                    PROCESSED_ROOT / f"{GRAPH_ID}.csv"
                )
                built_dataset = DatasetPipeline(
                    dataset_builder=dataset_builder,
                    processed_dir=PROCESSED_ROOT,
                ).run(
                    graph_id=GRAPH_ID,
                    output_path=processed_dataset_path,
                )

                assert built_dataset.num_samples > 0
                assert processed_dataset_path.is_file()
                assert len(candidate_subgraphs) == built_dataset.num_samples

            with experiment.track_stage("dataset_loading"):
                dataset = DatasetLoader(processed_dataset_path).load()

                assert isinstance(dataset, Dataset)
                assert dataset.num_samples >= 2
                assert dataset.feature_names == FEATURE_NAMES
                assert all(
                    graph_id == GRAPH_ID
                    for graph_id in dataset.graph_ids
                )

                full_dataset_path = experiment.artifact_store.copy_file(
                    "datasets",
                    "dataset.csv",
                    processed_dataset_path,
                )
                experiment.register_artifact(
                    "datasets",
                    "dataset.csv",
                    "text/csv",
                )

            logger.info(
                "Loaded dataset: %d samples, %d features",
                dataset.num_samples,
                dataset.num_features,
            )

            with experiment.track_stage("train_test_split"):
                train_dataset, test_dataset = split_dataset(
                    dataset,
                    test_size=TEST_SIZE,
                    random_state=SPLIT_RANDOM_STATE,
                )

                train_ids = set(train_dataset.subgraph_ids)
                test_ids = set(test_dataset.subgraph_ids)

                assert train_ids.isdisjoint(test_ids)
                assert train_ids | test_ids == set(dataset.subgraph_ids)
                assert (
                    train_dataset.num_samples + test_dataset.num_samples
                    == dataset.num_samples
                )
                assert train_dataset.feature_names == dataset.feature_names
                assert test_dataset.feature_names == dataset.feature_names

            with experiment.track_stage("split_dataset_persistence"):
                for filename, dataset_part in (
                    ("train_dataset.csv", train_dataset),
                    ("test_dataset.csv", test_dataset),
                ):
                    experiment.artifact_store.save_text(
                        "datasets",
                        filename,
                        dataset_to_csv(dataset_part),
                    )
                    experiment.register_artifact(
                        "datasets",
                        filename,
                        "text/csv",
                    )

                register_json_artifact(
                    experiment,
                    "datasets",
                    "split_metadata.json",
                    {
                        "strategy": "random",
                        "train_ratio": 1.0 - TEST_SIZE,
                        "test_ratio": TEST_SIZE,
                        "seed": SPLIT_RANDOM_STATE,
                        "train_subgraph_ids": train_dataset.subgraph_ids,
                        "test_subgraph_ids": test_dataset.subgraph_ids,
                    },
                )

            logger.info(
                "Split completed: %d training samples, %d test samples",
                train_dataset.num_samples,
                test_dataset.num_samples,
            )

            with experiment.track_stage("model_training"):
                model = LightGBMModel(**MODEL_PARAMETERS)
                trained_model = TrainingPipeline(
                    dataset=train_dataset,
                    model=model,
                ).run()

                assert isinstance(trained_model, Model)
                assert trained_model is model

                # LightGBM's native text model format; store the model as a file.
                model_text = trained_model._estimator.booster_.model_to_string()
                experiment.artifact_store.save_text(
                    "models",
                    "lightgbm_model.txt",
                    model_text,
                )
                experiment.register_artifact(
                    "models",
                    "lightgbm_model.txt",
                    "text/plain",
                )

            with experiment.track_stage("test_prediction_and_evaluation"):
                pipeline_result = PredictionPipeline(
                    model=trained_model,
                    dataset=test_dataset,
                    predictor=Predictor(trained_model),
                    evaluator=Evaluator(),
                ).run()

                prediction_results = pipeline_result.predictions
                evaluation = pipeline_result.evaluation
                predicted_values = np.asarray(
                    [
                        result.predicted_target
                        for result in prediction_results
                    ]
                )

                assert len(prediction_results) == test_dataset.num_samples
                assert predicted_values.shape == (
                    test_dataset.num_samples,
                )
                assert np.issubdtype(predicted_values.dtype, np.number)
                assert np.isfinite(predicted_values).all()
                assert evaluation.num_samples == test_dataset.num_samples

                predictions_path = experiment.artifact_store.save_text(
                    "predictions",
                    "predictions.csv",
                    predictions_to_csv(prediction_results),
                )
                experiment.register_artifact(
                    "predictions",
                    predictions_path.name,
                    "text/csv",
                )

                metrics = {
                    metric_name: (
                        float(value)
                        if np.isfinite(value)
                        else None
                    )
                    for metric_name, value in (
                        ("mae", evaluation.mae),
                        ("rmse", evaluation.rmse),
                        ("r2", evaluation.r2),
                    )
                }
                metrics_path = register_json_artifact(
                    experiment,
                    "evaluation",
                    "metrics.json",
                    {
                        "num_samples": evaluation.num_samples,
                        "metrics": metrics,
                    },
                )

            for prediction in prediction_results:
                prediction_error = None
                if prediction.true_target is not None:
                    prediction_error = (
                        prediction.predicted_target - prediction.true_target
                    )

                experiment.register_prediction(
                    sample_id=(
                        f"{prediction.graph_id}:{prediction.subgraph_id}"
                    ),
                    predicted_value=prediction.predicted_target,
                    actual_value=prediction.true_target,
                    prediction_error=prediction_error,
                )

            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    experiment.register_evaluation(
                        metric_name,
                        metric_value,
                    )
                else:
                    logger.warning(
                        "Metric %s is not finite and was not registered",
                        metric_name,
                    )

            logger.info(
                "Evaluation complete: MAE=%s RMSE=%s R2=%s",
                evaluation.mae,
                evaluation.rmse,
                evaluation.r2,
            )
            logger.info(
                "Artifacts saved: dataset=%s predictions=%s metrics=%s",
                full_dataset_path,
                predictions_path,
                metrics_path,
            )
            logger.info(
                "Experiment artifacts stored under %s",
                experiment.experiment_root,
            )


if __name__ == "__main__":
    main()