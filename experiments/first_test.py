"""Configuration for the first CSOP experiment."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.csop_experiment_runner import run_experiment
from src.experiments.config import (
    DatasetSplitConfig,
    ExperimentConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    SubgraphGenerationConfig,
)

from src.model.lightgbm_model import LightGBMModel

def make_config() -> ExperimentConfig:
    graph_id = "integration_test_graph"
    seed = 42

    return ExperimentConfig(
        name="CSOP TDS baseline",
        description="LightGBM prediction of TDS for candidate subgraphs.",
        random_seed=seed,
        graph=GraphConfig(
            graph_id=graph_id,
            path=PROJECT_ROOT / "data" / "raw" / "graph_test",
            parameters={"format": "adjlist", "directed": False},
        ),
        subgraph_generation=SubgraphGenerationConfig(
            method="networkx_random_paths",
            sample_count=8,
            path_length=3,
            seed=seed,
        ),
        features=FeatureConfig(
            feature_set_id="graph-size-and-local-degree-v1",
            feature_names=[
                "num_vertices",
                "num_edges",
                "degree_mean",
                "degree_max",
                "degree_min",
                "degree_std",
            ],
            objective_name="tds",
        ),
        dataset_split=DatasetSplitConfig(
            strategy="random",
            test_ratio=0.2,
            validation_size=0.25,
            random_state=seed,
        ),
        model=ModelConfig(
            model_type="lightgbm",
            hyperparameters={
                "n_estimators": 5,
                "learning_rate": 0.1,
                "num_leaves": 5,
                "min_child_samples": 1,
                "verbosity": -1,
                "random_state": seed,
            },
        ),
    )


if __name__ == "__main__":
    run_experiment(
        make_config(),
        experiments_root=PROJECT_ROOT / "experiments" / "runs",
        database_path=PROJECT_ROOT / "database" / "experiments.sqlite3",
        report_path=PROJECT_ROOT / "experiments" / "reports" / "experiments.xlsx",
        processed_root=PROJECT_ROOT / "data" / "processed",
                model_factory=lambda model_config: LightGBMModel(
            **model_config.hyperparameters
        ),
        model_serializer=lambda model: (
            model._estimator.booster_.model_to_string()
        ),
    )