"""Reproducibly benchmark the ten node-level graph features.

Run from the repository root with:
    python -m experiments.benchmark_node_features --output /tmp/features.json
"""

import argparse
import random
from pathlib import Path

import networkx as nx

from src.experiments.feature_benchmark import FeatureExtractionBenchmark
from src.features.base import FeatureScope
from src.features.node_features import (
    AverageNeighborDegreeFeature,
    BetweennessCentralityFeature,
    ClosenessCentralityFeature,
    EigenvectorCentralityFeature,
    KCoreNumberFeature,
    LocalClusteringCoefficientFeature,
    NormalizedDegreeFeature,
    PageRankFeature,
    SquareClusteringFeature,
    TriangleCountFeature,
)
from src.graph import Graph


def _to_project_graph(networkx_graph: nx.Graph) -> Graph:
    return Graph(
        vertices=list(networkx_graph.nodes),
        edges=list(networkx_graph.edges),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/reports/node_feature_benchmark.json"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory", action="store_true")
    args = parser.parse_args()

    seed = 42
    networkx_graph = nx.erdos_renyi_graph(30, 0.16, seed=seed)
    graph = _to_project_graph(networkx_graph)
    chooser = random.Random(seed)
    candidate_subgraphs = []
    candidate_ids = []
    for candidate_index in range(4):
        vertices = set(chooser.sample(sorted(graph.vertices), 12))
        induced = networkx_graph.subgraph(vertices)
        candidate_subgraphs.append(_to_project_graph(induced))
        candidate_ids.append(f"candidate-{candidate_index}")

    features = [
        PageRankFeature(scope=FeatureScope.LOCAL),
        BetweennessCentralityFeature(scope=FeatureScope.LOCAL),
        ClosenessCentralityFeature(scope=FeatureScope.LOCAL),
        EigenvectorCentralityFeature(scope=FeatureScope.LOCAL),
        KCoreNumberFeature(scope=FeatureScope.LOCAL),
        TriangleCountFeature(scope=FeatureScope.LOCAL),
        LocalClusteringCoefficientFeature(scope=FeatureScope.LOCAL),
        SquareClusteringFeature(scope=FeatureScope.LOCAL),
        AverageNeighborDegreeFeature(scope=FeatureScope.LOCAL),
        NormalizedDegreeFeature(scope=FeatureScope.LOCAL),
    ]
    benchmark = FeatureExtractionBenchmark(
        graph_id="erdos-renyi-n30-p016-seed42",
        graph=graph,
        candidate_subgraphs=candidate_subgraphs,
        candidate_ids=candidate_ids,
        features=features,
        repeats=args.repeats,
        random_seed=seed,
        measure_memory=args.memory,
    )
    report = benchmark.run()
    output_path = report.save_json(args.output)

    print(f"Wrote {len(report.records)} measurements to {output_path}")
    for comparison in report.comparisons:
        print(
            f"{comparison['feature_name']}: original="
            f"{comparison['original_graph_compute_seconds']:.6f}s, "
            "candidate computations total="
            f"{comparison['candidate_subgraph_compute_seconds_total']:.6f}s, "
            "reuse filtering total="
            f"{comparison['original_mapping_reuse_seconds_total']:.6f}s"
        )


if __name__ == "__main__":
    main()
