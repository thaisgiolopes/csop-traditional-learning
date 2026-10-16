"""Opt-in benchmarks for graph feature computation and mapping reuse."""

from collections.abc import Collection, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

import networkx as nx

from ..features.base import Feature, FeatureContext, FeatureLevel
from ..graph import Graph
from ..subgraphs.identity import graph_fingerprint
from .resource_monitor import ProcessMemoryMonitor
from .timer import Timer


_REUSE_SEMANTICS = (
    "Reused original-graph feature values are restricted to candidate vertices; "
    "they are not mathematically equivalent to recomputing the feature on the "
    "candidate's induced topology."
)


def _json_value(value: Any) -> Any:
    """Convert feature configuration values to deterministic JSON values."""
    if isinstance(value, Enum):
        return value.name
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(
        "Feature configuration must be JSON-compatible; received "
        f"{type(value).__name__}."
    )


@dataclass(frozen=True)
class FeatureBenchmarkReport:
    """Benchmark records, reproducibility metadata, and cost comparisons."""

    metadata: dict[str, Any]
    records: list[dict[str, Any]]
    comparisons: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report structure."""
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        """Save this report as readable UTF-8 JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                self.to_dict(),
                output_file,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            output_file.write("\n")
        return output_path


class FeatureExtractionBenchmark:
    """Measure feature computation and the cost of reusing original values.

    Benchmarking is explicit and separate from normal feature extraction.
    Each feature is computed once on the original graph, once per candidate
    per repeat, and its original mapping is separately filtered and timed for
    reuse by every candidate. These original and candidate calculations have
    different graph scopes and may produce mathematically different values.
    """

    def __init__(
        self,
        graph_id: str,
        graph: Graph,
        candidate_subgraphs: Collection[Graph],
        features: Collection[Feature],
        *,
        candidate_ids: Sequence[str] | None = None,
        repeats: int = 1,
        random_seed: int | None = None,
        measure_memory: bool = False,
        memory_monitor: ProcessMemoryMonitor | None = None,
        timer_factory=Timer,
    ) -> None:
        if not isinstance(graph_id, str) or not graph_id.strip():
            raise ValueError("graph_id must be a non-empty string.")
        if not isinstance(graph, Graph):
            raise TypeError("graph must be a Graph instance.")
        if isinstance(repeats, bool) or not isinstance(repeats, int):
            raise TypeError("repeats must be an integer.")
        if repeats <= 0:
            raise ValueError("repeats must be positive.")
        if random_seed is not None and (
            isinstance(random_seed, bool) or not isinstance(random_seed, int)
        ):
            raise TypeError("random_seed must be an integer or None.")
        if not isinstance(measure_memory, bool):
            raise TypeError("measure_memory must be a boolean.")
        if memory_monitor is not None and not isinstance(
            memory_monitor,
            ProcessMemoryMonitor,
        ):
            raise TypeError("memory_monitor must be a ProcessMemoryMonitor.")
        if not callable(timer_factory):
            raise TypeError("timer_factory must be callable.")

        self._graph_id = graph_id
        self._graph = graph
        self._candidates = tuple(candidate_subgraphs)
        if not all(isinstance(candidate, Graph) for candidate in self._candidates):
            raise TypeError("candidate_subgraphs must contain Graph instances.")
        self._features = tuple(features)
        if not all(self._is_feature_like(feature) for feature in self._features):
            raise TypeError("features must contain Feature-compatible objects.")
        if any(feature.level is not FeatureLevel.NODE for feature in self._features):
            raise ValueError("features must be node-level for mapping reuse.")
        self._repeats = repeats
        self._random_seed = random_seed
        self._candidate_ids = tuple(
            candidate_ids
            if candidate_ids is not None
            else (f"candidate-{index}" for index in range(len(self._candidates)))
        )
        if len(self._candidate_ids) != len(self._candidates):
            raise ValueError("candidate_ids must match candidate_subgraphs length.")
        if not all(isinstance(item, str) and item for item in self._candidate_ids):
            raise ValueError("candidate_ids must contain non-empty strings.")
        self._memory_monitor = (
            memory_monitor
            if memory_monitor is not None
            else ProcessMemoryMonitor() if measure_memory else None
        )
        self._timer_factory = timer_factory

    def run(self) -> FeatureBenchmarkReport:
        """Run one original, candidate, and reuse measurement per feature."""
        records: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        feature_configuration = []

        for feature in self._features:
            feature.compute(
                FeatureContext(full_graph=self._graph, subgraph=self._graph)
            )
            if self._candidates:
                first_candidate = self._candidates[0]
                feature.compute(
                    FeatureContext(
                        full_graph=first_candidate,
                        subgraph=first_candidate,
                    )
                )

            feature_configuration.append(
                {
                    "name": feature.name,
                    "class": (
                        f"{type(feature).__module__}."
                        f"{type(feature).__qualname__}"
                    ),
                    "configured_scope": getattr(
                        feature.scope,
                        "name",
                        str(feature.scope),
                    ),
                    "level": getattr(feature.level, "name", str(feature.level)),
                    "parameters": _json_value(vars(feature)),
                }
            )

            full_values, record = self._measure(
                feature,
                self._graph,
                computation_scope="original_graph",
                result_reused=False,
                candidate_id=None,
                candidate=None,
                repeat_index=0,
            )
            records.append(record)
            candidate_compute_seconds = 0.0
            reuse_seconds = 0.0

            for candidate_index, candidate in enumerate(self._candidates):
                candidate_id = self._candidate_ids[candidate_index]
                for repeat_index in range(self._repeats):
                    _, candidate_record = self._measure(
                        feature,
                        candidate,
                        computation_scope="candidate_subgraph",
                        result_reused=False,
                        candidate_id=candidate_id,
                        candidate=candidate,
                        repeat_index=repeat_index,
                    )
                    records.append(candidate_record)
                    candidate_compute_seconds += candidate_record[
                        "elapsed_seconds"
                    ]

                    _, reuse_record = self._measure_reuse(
                        feature,
                        full_values,
                        candidate,
                        candidate_id=candidate_id,
                        repeat_index=repeat_index,
                    )
                    records.append(reuse_record)
                    reuse_seconds += reuse_record["elapsed_seconds"]

            comparisons.append(
                {
                    "feature_name": feature.name,
                    "original_graph_compute_seconds": record[
                        "elapsed_seconds"
                    ],
                    "candidate_subgraph_compute_seconds_total": (
                        candidate_compute_seconds
                    ),
                    "original_mapping_reuse_seconds_total": reuse_seconds,
                    "original_precompute_plus_reuse_seconds_total": (
                        record["elapsed_seconds"] + reuse_seconds
                    ),
                    "candidate_measurement_count": (
                        len(self._candidates) * self._repeats
                    ),
                    "reuse_measurement_count": (
                        len(self._candidates) * self._repeats
                    ),
                    "semantic_note": _REUSE_SEMANTICS,
                }
            )

        metadata = {
            "method": "perf_counter per operation; optional process RSS before/after",
            "warmup_policy": (
                "one unmeasured computation per feature on the original graph "
                "and first candidate graph"
            ),
            "graph_id": self._graph_id,
            "graph_fingerprint": graph_fingerprint(self._graph),
            "graph_num_vertices": self._graph.num_vertices,
            "graph_num_edges": self._graph.num_edges,
            "networkx_version": nx.__version__,
            "random_seed": self._random_seed,
            "repeats": self._repeats,
            "candidate_ids": list(self._candidate_ids),
            "candidate_sizes": [
                {
                    "candidate_id": candidate_id,
                    "graph_fingerprint": graph_fingerprint(candidate),
                    "num_vertices": candidate.num_vertices,
                    "num_edges": candidate.num_edges,
                }
                for candidate_id, candidate in zip(
                    self._candidate_ids,
                    self._candidates,
                    strict=True,
                )
            ],
            "memory_measurement_enabled": self._memory_monitor is not None,
            "feature_configuration": feature_configuration,
            "reuse_semantics": _REUSE_SEMANTICS,
        }
        return FeatureBenchmarkReport(metadata, records, comparisons)

    def run_and_track(
        self,
        experiment,
        *,
        artifact_filename: str = "feature_benchmark.json",
    ) -> FeatureBenchmarkReport:
        """Run under an experiment stage and register the JSON report."""
        with experiment.track_stage("feature_extraction_benchmark"):
            report = self.run()
            experiment.artifact_store.save_json(
                "evaluation",
                artifact_filename,
                report.to_dict(),
            )
        experiment.register_artifact(
            "evaluation",
            artifact_filename,
            "application/json",
        )
        return report

    def _measure(
        self,
        feature: Feature,
        graph: Graph,
        *,
        computation_scope: str,
        result_reused: bool,
        candidate_id: str | None,
        candidate: Graph | None,
        repeat_index: int,
    ) -> tuple[Any, dict[str, Any]]:
        stage_name = (
            f"feature_benchmark:{feature.name}:{computation_scope}:"
            f"compute:{candidate_id or 'original'}:{repeat_index}"
        )
        memory_context = (
            self._memory_monitor.measure(stage_name)
            if self._memory_monitor is not None
            else nullcontext(None)
        )
        with memory_context as memory:
            timer = self._timer_factory(stage_name)
            context = FeatureContext(full_graph=graph, subgraph=graph)
            with timer:
                values = feature.compute(context)
            elapsed_seconds = timer.elapsed_seconds

        return values, self._record(
            feature,
            computation_scope=computation_scope,
            result_reused=result_reused,
            elapsed_seconds=elapsed_seconds,
            candidate_id=candidate_id,
            candidate=candidate,
            repeat_index=repeat_index,
            memory=memory,
        )

    def _measure_reuse(
        self,
        feature: Feature,
        original_values: Any,
        candidate: Graph,
        *,
        candidate_id: str,
        repeat_index: int,
    ) -> tuple[dict[Any, Any], dict[str, Any]]:
        if not isinstance(original_values, dict):
            raise TypeError(
                f"Node feature {feature.name!r} must return a node mapping "
                "to benchmark candidate reuse."
            )

        stage_name = (
            f"feature_benchmark:{feature.name}:original_graph:reuse:"
            f"{candidate_id}:{repeat_index}"
        )
        memory_context = (
            self._memory_monitor.measure(stage_name)
            if self._memory_monitor is not None
            else nullcontext(None)
        )
        with memory_context as memory:
            timer = self._timer_factory(stage_name)
            with timer:
                reused_values = {
                    node: value
                    for node, value in original_values.items()
                    if node in candidate.vertices
                }
            elapsed_seconds = timer.elapsed_seconds

        return reused_values, self._record(
            feature,
            computation_scope="original_graph",
            result_reused=True,
            elapsed_seconds=elapsed_seconds,
            candidate_id=candidate_id,
            candidate=candidate,
            repeat_index=repeat_index,
            memory=memory,
        )

    def _record(
        self,
        feature: Feature,
        *,
        computation_scope: str,
        result_reused: bool,
        elapsed_seconds: float,
        candidate_id: str | None,
        candidate: Graph | None,
        repeat_index: int,
        memory,
    ) -> dict[str, Any]:
        return {
            "graph_id": self._graph_id,
            "graph_num_vertices": self._graph.num_vertices,
            "graph_num_edges": self._graph.num_edges,
            "candidate_id": candidate_id,
            "subgraph_num_vertices": (
                candidate.num_vertices if candidate is not None else None
            ),
            "subgraph_num_edges": (
                candidate.num_edges if candidate is not None else None
            ),
            "feature_name": feature.name,
            "configured_scope": getattr(
                feature.scope,
                "name",
                str(feature.scope),
            ),
            "computation_scope": computation_scope,
            "operation": "mapping_filter_reuse" if result_reused else "compute",
            "elapsed_seconds": elapsed_seconds,
            "result_reused": result_reused,
            "result_status": "reused" if result_reused else "newly_computed",
            "repeat_index": repeat_index,
            "random_seed": self._random_seed,
            "memory_before_bytes": (
                memory.memory_before_bytes if memory is not None else None
            ),
            "memory_after_bytes": (
                memory.memory_after_bytes if memory is not None else None
            ),
            "memory_delta_bytes": (
                memory.memory_delta_bytes if memory is not None else None
            ),
            "memory_unit": "bytes" if memory is not None else None,
            "measurement_method": (
                memory.measurement_method if memory is not None else None
            ),
        }

    @staticmethod
    def _is_feature_like(feature: object) -> bool:
        return all(
            hasattr(feature, attribute)
            for attribute in ("name", "scope", "level", "compute")
        )
