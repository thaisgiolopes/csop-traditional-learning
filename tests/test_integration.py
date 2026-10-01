from pathlib import Path

import numpy as np

from src.dataset.builder import DatasetBuilder
from src.dataset.dataset import Dataset
from src.dataset.splitter import split_dataset
from src.evaluation.result import EvaluationResult
from src.features.base import FeatureScope
from src.features.engine import FeatureEngine
from src.features.graph_features import (
    NumEdgesFeature,
    NumVerticesFeature,
)
from src.features.node_features import DegreeFeature
from src.features.pooling import MaxPooling, MeanPooling
from src.graph.loader import GraphLoader
from src.model.base import Model
from src.model.lightgbm_model import LightGBMModel
from src.model.predictor import PredictionResult, Predictor
from src.objectives.tds import TDSObjective
from src.pipeline.prediction_pipeline import (
    PredictionEvaluationResult,
    PredictionPipeline,
)
from src.pipeline.training_pipeline import TrainingPipeline
from src.subgraphs.networkx_generator import NetworkXSubgraphGenerator
from src.evaluation.evaluator import Evaluator

def test_complete_csop_prediction_pipeline():
    """
    Test dataset construction, train/test splitting, training, prediction,
    and evaluation. The test checks integration, not scientific accuracy.
    """
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "simple_graph"
    )

    graph_loader = GraphLoader(fixture_path)
    subgraph_generator = NetworkXSubgraphGenerator(
        num_subgraphs=4,
        path_length=2,
        seed=42,
    )
    feature_engine = FeatureEngine(
        features=[
            NumVerticesFeature(),
            NumEdgesFeature(),
            DegreeFeature(scope=FeatureScope.LOCAL),
        ]
    )
    builder = DatasetBuilder(
        graph_loader=graph_loader,
        subgraph_generator=subgraph_generator,
        feature_engine=feature_engine,
        pooling_strategies={
            "degree": (MeanPooling(), MaxPooling()),
        },
        objective_function=TDSObjective(),
    )

    samples = builder.build(graph_id="integration_graph")
    assert samples

    dataset = Dataset(samples)

    assert dataset.num_samples == len(samples)
    assert dataset.num_samples >= 2
    assert dataset.num_features == 4
    assert list(dataset.X.columns) == [
        "num_vertices",
        "num_edges",
        "degree_mean",
        "degree_max",
    ]
    assert len(dataset.y) == dataset.num_samples
    assert dataset.graph_ids == [
        "integration_graph"
    ] * dataset.num_samples
    assert dataset.subgraph_ids == list(range(dataset.num_samples))

    train_dataset, test_dataset = split_dataset(
        dataset,
        test_size=0.2,
        random_state=42,
    )

    train_ids = set(train_dataset.subgraph_ids)
    test_ids = set(test_dataset.subgraph_ids)

    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(dataset.subgraph_ids)
    assert train_dataset.num_samples + test_dataset.num_samples \
        == dataset.num_samples
    assert train_dataset.feature_names == dataset.feature_names
    assert test_dataset.feature_names == dataset.feature_names

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

    pipeline_result = PredictionPipeline(
        model=trained_model,
        dataset=test_dataset,
        predictor=Predictor(trained_model),
        evaluator=Evaluator(),
    ).run()

    assert isinstance(pipeline_result, PredictionEvaluationResult)

    prediction_results = pipeline_result.predictions
    evaluation_result = pipeline_result.evaluation

    assert isinstance(evaluation_result, EvaluationResult)
    assert len(prediction_results) == test_dataset.num_samples
    assert evaluation_result.num_samples == test_dataset.num_samples
    assert all(
        isinstance(result, PredictionResult)
        for result in prediction_results
    )
    assert [
        (result.graph_id, result.subgraph_id)
        for result in prediction_results
    ] == list(zip(test_dataset.graph_ids, test_dataset.subgraph_ids))
    assert np.isfinite(
        [result.predicted_target for result in prediction_results]
    ).all()

    # R2 is undefined when the test set contains only one sample.
    assert np.isfinite(
        [evaluation_result.mae, evaluation_result.rmse]
    ).all()