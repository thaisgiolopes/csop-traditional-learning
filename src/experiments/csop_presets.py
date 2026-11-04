"""Reusable configurations for the initial CSOP TDS experiments."""

from collections.abc import Iterable, Mapping
from pathlib import Path

from .config import (
    DatasetSplitConfig,
    ExperimentConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    SubgraphGenerationConfig,
)

SUPPORTED_DATASETS = ("amazon", "citeseer")
SUPPORTED_SUBGRAPH_SIZES = tuple(range(10, 501, 10))

BASE_FEATURE_NAMES = [
    "num_vertices",
    "num_edges",
    "subgraph_size",
]
POOLING_NAMES = ("mean", "max", "min", "std")
AVAILABLE_NODE_FEATURES = (
    "degree",
    "pagerank",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
    "k_core_number",
    "triangle_count",
    "local_clustering_coefficient",
    "square_clustering",
    "average_neighbor_degree",
    "normalized_degree",
)


def configured_feature_names(
    node_features: Iterable[str], node_scope: str
) -> list[str]:
    """List the exact ordered feature columns produced by the runner."""
    if node_scope not in {"local", "global"}:
        raise ValueError("node_scope must be 'local' or 'global'.")
    selected = list(node_features)
    unknown = set(selected) - set(AVAILABLE_NODE_FEATURES)
    if unknown:
        raise ValueError(f"Unknown node features: {sorted(unknown)!r}.")
    if len(selected) != len(set(selected)):
        raise ValueError("Node features must not contain duplicates.")
    pooled_names = [
        f"{feature}_{pooling}"
        if feature == "degree"
        else f"{feature}_{node_scope}_{pooling}"
        for feature in selected
        for pooling in POOLING_NAMES
    ]
    if node_scope == "global":
        # Global node mappings are emitted with global graph features; the
        # candidate-local size feature is merged after those mappings.
        return ["num_vertices", "num_edges", *pooled_names, "subgraph_size"]
    return [*BASE_FEATURE_NAMES, *pooled_names]


def build_csop_configs(
    *,
    datasets: Iterable[str],
    subgraph_sizes: Iterable[int],
    sample_count: int = 100,
    seed: int = 42,
    max_attempts: int = 5_000,
    model_hyperparameters: Mapping[str, object] | None = None,
    node_features: Iterable[str] = ("degree",),
    node_scope: str = "local",
    n_splits: int = 5,
    output_paths: Mapping[str, str] | None = None,
) -> list[ExperimentConfig]:
    """Build one reproducible TDS experiment config per dataset and size."""
    dataset_names = list(datasets)
    sizes = list(subgraph_sizes)
    selected_features = list(node_features)
    feature_names = configured_feature_names(selected_features, node_scope)

    if not dataset_names:
        raise ValueError("At least one dataset must be selected.")
    if not sizes:
        raise ValueError("At least one subgraph size must be selected.")
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("Dataset selections must not contain duplicates.")
    if len(sizes) != len(set(sizes)):
        raise ValueError("Subgraph size selections must not contain duplicates.")
    if set(dataset_names) - set(SUPPORTED_DATASETS):
        raise ValueError(
            "Unsupported datasets: "
            f"{sorted(set(dataset_names) - set(SUPPORTED_DATASETS))}"
        )
    if set(sizes) - set(SUPPORTED_SUBGRAPH_SIZES):
        raise ValueError("Subgraph sizes must be selected from 10, 20, ..., 500.")
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts <= 0
    ):
        raise ValueError("max_attempts must be a positive integer.")

    configured_output = dict(
        output_paths
        or {
            "experiments_root": "experiments/runs",
            "database_path": "database/experiments.sqlite3",
            "report_path": "experiments/reports/experiments.xlsx",
            "processed_root": "data/processed",
        }
    )
    expected_output_keys = {
        "experiments_root",
        "database_path",
        "report_path",
        "processed_root",
    }
    if set(configured_output) != expected_output_keys:
        raise ValueError(
            "output_paths must define exactly: "
            f"{sorted(expected_output_keys)!r}."
        )

    configured_model = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": seed,
        "verbosity": -1,
    }
    if model_hyperparameters is not None:
        configured_model.update(model_hyperparameters)
    configured_model["random_state"] = seed

    configs = []
    for dataset in dataset_names:
        for size in sizes:
            graph_id = f"{dataset}_k{size}"
            configs.append(
                ExperimentConfig(
                    name=f"csop-tds-{dataset}-k{size}",
                    description=(
                        "LightGBM prediction of TDS for connected induced "
                        "subgraphs."
                    ),
                    random_seed=seed,
                    graph=GraphConfig(
                        graph_id=graph_id,
                        source=dataset,
                    ),
                    subgraph_generation=SubgraphGenerationConfig(
                        method="networkx_connected_expansion",
                        sample_count=sample_count,
                        seed=seed,
                        parameters={
                            "subgraph_size": size,
                            "max_attempts": max_attempts,
                        },
                    ),
                    features=FeatureConfig(
                        feature_set_id=(
                            "graph-size-node-structure-"
                            f"{node_scope}-with-subgraph-size-v1"
                        ),
                        feature_names=feature_names.copy(),
                        enabled_features=selected_features.copy(),
                        objective_name="tds",
                    ),
                    dataset_split=DatasetSplitConfig(
                        strategy="stratified_kfold",
                        test_ratio=None,
                        validation_size=None,
                        random_state=seed,
                        parameters={"n_splits": n_splits},
                    ),
                    model=ModelConfig(
                        model_type="lightgbm",
                        hyperparameters=configured_model.copy(),
                    ),
                    output=configured_output.copy(),
                )
            )
    return configs
