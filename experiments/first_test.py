"""
End-to-end experiment for the CSOP prediction pipeline.

Candidate subgraphs from one graph are split into training and test datasets.
Metrics are calculated only on test candidates and should be interpreted
cautiously because the current dataset is small.
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
from src.dataset.splitter import split_dataset
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
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42


def main() -> None:
    raw_instance_path = PROJECT_ROOT / "data" / "raw" / "graph_test"
    processed_root = PROJECT_ROOT / "data" / "processed"
    processed_dataset_path = processed_root / f"{GRAPH_ID}.csv"
    results_path = (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "result_first_test.txt"
    )

    processed_root.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", encoding="utf-8") as result_file:
        with redirect_stdout(result_file):
            assert raw_instance_path.is_dir(), (
                f"Raw graph directory not found: {raw_instance_path}"
            )
            assert (raw_instance_path / "metadata").is_file()
            assert (raw_instance_path / "adjlist").is_file()

            print("[1] Loading graph")
            graph_loader = GraphLoader(raw_instance_path)
            graph = graph_loader.load()

            assert graph.num_vertices == 5
            assert graph.num_edges == 6

            print("[2] Generating candidate subgraphs")
            subgraph_generator = NetworkXSubgraphGenerator(
                num_subgraphs=8,
                path_length=3,
                seed=42,
            )
            candidate_subgraphs = subgraph_generator.generate(graph)
            assert candidate_subgraphs

            print("[3] Building and saving dataset")
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

            print("[4] Loading processed dataset")
            dataset = DatasetLoader(processed_dataset_path).load()

            assert isinstance(dataset, Dataset)
            assert dataset.num_samples >= 2
            assert len(candidate_subgraphs) == dataset.num_samples
            assert dataset.subgraph_ids == list(
                range(dataset.num_samples)
            )
            assert all(
                graph_id == GRAPH_ID
                for graph_id in dataset.graph_ids
            )
            assert dataset.feature_names == [
                "num_vertices",
                "num_edges",
                "degree_mean",
                "degree_max",
                "degree_min",
                "degree_std",
            ]

            print(f"Total samples: {dataset.num_samples}")
            print(f"Number of features: {dataset.num_features}")
            print(f"Feature names: {dataset.feature_names}")
            print("Full feature matrix:")
            print(dataset.X)
            print("Full target vector:")
            print(dataset.y)

            for sample, subgraph in zip(
                dataset.samples,
                candidate_subgraphs,
                strict=True,
            ):
                print()
                print(f"Sample {sample.subgraph_id}")
                print(f"    graph_id: {sample.graph_id}")
                print(f"    vertices: {sorted(subgraph.vertices)}")
                print(f"    features: {sample.features}")
                print(f"    target: {sample.target}")

            print("[5] Splitting candidates into train and test")
            train_dataset, test_dataset = split_dataset(
                dataset,
                test_size=TEST_SIZE,
                random_state=SPLIT_RANDOM_STATE,
            )

            train_ids = set(train_dataset.subgraph_ids)
            test_ids = set(test_dataset.subgraph_ids)

            assert train_ids.isdisjoint(test_ids)
            assert train_ids | test_ids == set(dataset.subgraph_ids)
            assert train_dataset.num_samples + test_dataset.num_samples \
                == dataset.num_samples
            assert train_dataset.feature_names == dataset.feature_names
            assert test_dataset.feature_names == dataset.feature_names

            print(f"Training samples: {train_dataset.num_samples}")
            print(f"Training subgraph IDs: {train_dataset.subgraph_ids}")
            print(f"Test samples: {test_dataset.num_samples}")
            print(f"Test subgraph IDs: {test_dataset.subgraph_ids}")

            print("[6] Training LightGBM on training dataset")
            model = LightGBMModel(
                n_estimators=5,
                learning_rate=0.1,
                num_leaves=5,
                min_child_samples=1,
                verbosity=-1,
                random_state=42,
            )
            trained_model = TrainingPipeline(
                dataset=train_dataset,
                model=model,
            ).run()

            assert isinstance(trained_model, Model)
            assert trained_model is model

            print("[7] Predicting and evaluating on test dataset")
            predictor = Predictor(trained_model)
            evaluator = Evaluator()
            pipeline_result = PredictionPipeline(
                model=trained_model,
                dataset=test_dataset,
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

            assert len(prediction_results) == test_dataset.num_samples
            assert all(
                isinstance(result, PredictionResult)
                for result in prediction_results
            )
            assert all(
                result.graph_id == graph_id
                and result.subgraph_id == subgraph_id
                for result, graph_id, subgraph_id in zip(
                    prediction_results,
                    test_dataset.graph_ids,
                    test_dataset.subgraph_ids,
                    strict=True,
                )
            )

            predicted_values = np.asarray(
                [
                    result.predicted_target
                    for result in prediction_results
                ]
            )
            assert predicted_values.shape == (test_dataset.num_samples,)
            assert np.issubdtype(predicted_values.dtype, np.number)
            assert np.isfinite(predicted_values).all()

            assert evaluation_result.num_samples == test_dataset.num_samples
            assert np.isfinite(evaluation_result.mae)
            assert np.isfinite(evaluation_result.rmse)

            print()
            print("True test targets:")
            print(test_dataset.y.to_list())
            print("Predicted test targets:")
            print(predicted_values.tolist())
            print("Test metrics:")
            print(f"MAE: {evaluation_result.mae}")
            print(f"RMSE: {evaluation_result.rmse}")
            print(f"R2: {evaluation_result.r2}")

            print()
            print("[OK] Integration experiment completed")
            print(
                "Summary: graph -> candidates -> dataset -> "
                "train/test split -> training -> test prediction "
                "-> evaluation."
            )


if __name__ == "__main__":
    main()