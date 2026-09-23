from pathlib import Path

import numpy as np

from src.dataset.builder import DatasetBuilder
from src.dataset.dataset import Dataset
from src.evaluation.evaluator import Evaluator
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
from src.model.lightgbm_model import LightGBMModel
from src.model.predictor import PredictionResult, Predictor
from src.objectives.tds import TDSObjective
from src.subgraphs.networkx_generator import NetworkXSubgraphGenerator


def test_complete_csop_prediction_pipeline():
    """
    Test the complete pipeline from graph loading through evaluation.

    The test verifies structural integration and metadata preservation. It
    does not assert scientific model accuracy.
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
    assert dataset.num_features == 4
    assert list(dataset.X.columns) == [
        "num_vertices",
        "num_edges",
        "degree_mean",
        "degree_max",
    ]
    assert len(dataset.y) == dataset.num_samples
    assert dataset.graph_ids == ["integration_graph"] * dataset.num_samples
    assert dataset.subgraph_ids == list(range(dataset.num_samples))

    model = LightGBMModel(
        n_estimators=5,
        learning_rate=0.1,
        num_leaves=5,
        min_child_samples=1,
        verbosity=-1,
        random_state=42,
    )

    assert model.fit(dataset.X, dataset.y) is model

    prediction_results = Predictor(model).predict(dataset)

    assert len(prediction_results) == dataset.num_samples
    assert all(
        isinstance(result, PredictionResult)
        for result in prediction_results
    )
    assert [
        (result.graph_id, result.subgraph_id)
        for result in prediction_results
    ] == list(zip(dataset.graph_ids, dataset.subgraph_ids))
    assert np.isfinite(
        [result.predicted_target for result in prediction_results]
    ).all()

    evaluation_result = Evaluator().evaluate(
        dataset.y,
        [
            result.predicted_target
            for result in prediction_results
        ],
    )

    assert isinstance(evaluation_result, EvaluationResult)
    assert evaluation_result.num_samples == dataset.num_samples
    assert np.isfinite(
        [
            evaluation_result.mae,
            evaluation_result.rmse,
            evaluation_result.r2,
        ]
    ).all()