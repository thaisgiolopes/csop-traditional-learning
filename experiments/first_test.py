"""
End-to-end architecture integration test for the CSOP project.

This script reads a graph instance from data/raw/graph_test, builds and
persists a machine-learning dataset, loads it again, trains LightGBM,
generates predictions, and evaluates them.

The metrics produced here are not experimental results. The same dataset is
used for training and prediction, so the values must not be interpreted as
evidence of model generalization.
"""

from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.builder import DatasetBuilder
from src.dataset.dataset import Dataset
from src.dataset.loader import DatasetLoader
from src.evaluation.evaluator import Evaluator
from src.evaluation.result import EvaluationResult
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
from src.model.predictor import PredictionResult, Predictor
from src.objectives.tds import TDSObjective
from src.pipeline.dataset_pipeline import DatasetPipeline
from src.pipeline.prediction_pipeline import (
    PredictionEvaluationResult,
    PredictionPipeline,
)
from src.pipeline.training_pipeline import TrainingPipeline
from src.subgraphs.networkx_generator import NetworkXSubgraphGenerator


GRAPH_ID = "integration_test_graph"


def main() -> None:
    """
    Execute the complete CSOP pipeline integration test.

    The raw graph is read from ``data/raw/graph_test`` and is never modified.
    The processed dataset is saved under ``data/processed``. All execution
    output is written to ``experiments/results/result_first_test``.
    """
    raw_instance_path = PROJECT_ROOT / "data" / "raw" / "graph_test"
    processed_root = PROJECT_ROOT / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    processed_dataset_path = processed_root / f"{GRAPH_ID}.csv"

    results_path = (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "result_first_test.txt"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", encoding="utf-8") as result_file:
        with redirect_stdout(result_file):
            print("[1] Loading existing test graph")

            assert raw_instance_path.is_dir(), (
                f"Raw graph directory not found: {raw_instance_path}"
            )
            assert (raw_instance_path / "metadata").is_file()
            assert (raw_instance_path / "adjlist").is_file()

            print("[2] Loading graph")

            graph_loader = GraphLoader(raw_instance_path)
            graph = graph_loader.load()

            assert graph.num_vertices == 5
            assert graph.num_edges == 6

            print("[3] Generating candidate subgraphs")

            subgraph_generator = NetworkXSubgraphGenerator(
                num_subgraphs=8,
                path_length=3,
                seed=42,
            )

            candidate_subgraphs = subgraph_generator.generate(graph)

            assert len(candidate_subgraphs) > 0

            print("[4] Building and saving processed dataset")

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

            dataset_pipeline = DatasetPipeline(
                dataset_builder=dataset_builder,
                processed_dir=processed_root,
            )

            built_dataset = dataset_pipeline.run(
                graph_id=GRAPH_ID,
                output_path=processed_dataset_path,
            )

            assert built_dataset.num_samples > 0
            assert processed_dataset_path.is_file()

            print(
                f"Processed dataset saved to: {processed_dataset_path}"
            )

            print("[5] Loading processed dataset")

            dataset_loader = DatasetLoader(processed_dataset_path)
            dataset = dataset_loader.load()

            assert isinstance(dataset, Dataset)
            assert dataset.num_samples > 0

            print("[6] Inspecting X, y, and samples")

            print(f"Number of samples: {dataset.num_samples}")
            print(f"Number of features: {dataset.num_features}")
            print(f"Feature names: {dataset.feature_names}")
            print("X:")
            print(dataset.X)
            print("y:")
            print(dataset.y)

            assert len(candidate_subgraphs) == dataset.num_samples
            assert len(dataset.X) == len(dataset.y)
            assert len(dataset.X) > 0
            assert len(dataset.y) > 0

            assert all(
                graph_id == GRAPH_ID
                for graph_id in dataset.graph_ids
            )
            assert dataset.subgraph_ids == list(
                range(dataset.num_samples)
            )

            expected_feature_names = [
                "num_vertices",
                "num_edges",
                "degree_mean",
                "degree_max",
                "degree_min",
                "degree_std",
            ]

            assert dataset.feature_names == expected_feature_names

            for sample_index, (sample, subgraph) in enumerate(
                zip(
                    dataset.samples,
                    candidate_subgraphs,
                    strict=True,
                )
            ):
                print()
                print(f"Sample {sample_index}")
                print(f"    graph_id: {sample.graph_id}")
                print(f"    subgraph_id: {sample.subgraph_id}")
                print(
                    f"    vertices: {sorted(subgraph.vertices)}"
                )
                print("    features:")

                for feature_name, feature_value in (
                    sample.features.items()
                ):
                    print(
                        f"        {feature_name}: {feature_value}"
                    )

                print(f"    target: {sample.target}")

            print("[7] Training LightGBM")

            model = LightGBMModel(
                n_estimators=5,
                learning_rate=0.1,
                num_leaves=5,
                min_child_samples=1,
                verbosity=-1,
                random_state=42,
            )

            trained_model = TrainingPipeline(
                dataset=dataset,
                model=model,
            ).run()

            assert isinstance(trained_model, Model)
            assert trained_model is model

            print("[8] Generating predictions")

            predictor = Predictor(trained_model)
            evaluator = Evaluator()

            print("[9] Evaluating predictions")

            pipeline_result = PredictionPipeline(
                model=trained_model,
                dataset=dataset,
                predictor=predictor,
                evaluator=evaluator,
            ).run()

            assert isinstance(
                pipeline_result,
                PredictionEvaluationResult,
            )
            assert isinstance(
                pipeline_result.evaluation,
                EvaluationResult,
            )

            prediction_results = pipeline_result.predictions
            evaluation_result = pipeline_result.evaluation

            assert len(prediction_results) == len(dataset.y)
            assert len(prediction_results) > 0

            assert all(
                isinstance(result, PredictionResult)
                for result in prediction_results
            )

            assert all(
                result.graph_id == graph_id
                and result.subgraph_id == subgraph_id
                for result, graph_id, subgraph_id in zip(
                    prediction_results,
                    dataset.graph_ids,
                    dataset.subgraph_ids,
                    strict=True,
                )
            )

            predicted_values = np.asarray(
                [
                    result.predicted_target
                    for result in prediction_results
                ]
            )

            assert predicted_values.shape == (
                dataset.num_samples,
            )
            assert np.issubdtype(
                predicted_values.dtype,
                np.number,
            )
            assert np.isfinite(predicted_values).all()

            assert (
                evaluation_result.num_samples
                == dataset.num_samples
            )
            assert isinstance(evaluation_result.mae, float)
            assert isinstance(evaluation_result.rmse, float)
            assert isinstance(evaluation_result.r2, float)

            print()
            print("True targets:")
            print(dataset.y.to_list())

            print("Predicted targets:")
            print(predicted_values.tolist())

            print("Evaluation metrics:")
            print(f"MAE: {evaluation_result.mae}")
            print(f"RMSE: {evaluation_result.rmse}")
            print(f"R2: {evaluation_result.r2}")

            print()
            print("[OK] Integration test completed successfully")
            print(
                "Summary: raw graph -> samples -> processed CSV "
                "-> Dataset -> trained model -> predictions "
                "-> evaluation."
            )


if __name__ == "__main__":
    main()