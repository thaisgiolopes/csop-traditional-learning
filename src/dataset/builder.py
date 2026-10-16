from collections.abc import Collection, Mapping, Sequence
from enum import Enum
from typing import Any

from ..features.base import FeatureLevel, FeatureContext
from ..features.engine import FeatureEngine
from ..features.pooling import Pooling
from ..graph import Graph
from ..graph.loader import GraphLoader
from ..objectives.base import ObjectiveFunction
from ..subgraphs.base import SubgraphGenerator
from .sample import Sample


class DatasetBuilder:
    """
    Orchestrates the construction of the CSOP training dataset.

    The builder combines a complete graph, candidate subgraphs, extracted
    features, pooling strategies, and objective values into dataset samples.

    The resulting learning problem is:

        X(G, S) -> objective(S)

    where G is the complete graph and S is a candidate subgraph.

    Global features are computed once from the complete graph and reused
    for all candidate subgraphs generated from that graph. Global node
    mappings are restricted to each candidate's vertices before pooling.
    Local features are computed independently for each candidate subgraph.

    This class does not implement feature calculations, subgraph
    generation, pooling algorithms, objective functions, machine learning,
    or dataset splitting.
    """

    def __init__(
        self,
        graph_loader: GraphLoader,
        subgraph_generator: SubgraphGenerator,
        feature_engine: FeatureEngine,
        pooling_strategies: Mapping[str, tuple[Pooling, ...]],
        objective_function: ObjectiveFunction,
    ):
        """
        Initialize the dataset builder.

        Args:
            graph_loader: Loads the complete graph instance.
            subgraph_generator: Generates candidate subgraphs.
            feature_engine: Computes configured global and local features.
            pooling_strategies: Maps feature names to pooling strategies.
                Node-level and edge-level feature results can be
                aggregated using the configured strategies.
            objective_function: Computes the target value for each
                candidate subgraph.
        """
        self._graph_loader = graph_loader
        self._subgraph_generator = subgraph_generator
        self._feature_engine = feature_engine
        self._pooling_strategies = pooling_strategies
        self._objective_function = objective_function

        self._features_by_name = {
            feature.name: feature
            for feature in feature_engine.features
        }

    @property
    def feature_configuration(self) -> dict[str, Any]:
        """Return the ordered, JSON-compatible configuration used for X."""
        return {
            "features": [
                {
                    "name": feature.name,
                    "class": (
                        f"{type(feature).__module__}."
                        f"{type(feature).__qualname__}"
                    ),
                    "scope": getattr(feature.scope, "name", str(feature.scope)),
                    "level": getattr(feature.level, "name", str(feature.level)),
                    "parameters": self._configuration_value(
                        vars(feature)
                    ),
                }
                for feature in self._feature_engine.features
            ],
            "pooling": [
                {
                    "feature_name": feature.name,
                    "strategies": [
                        {
                            "name": pooling.name,
                            "class": (
                                f"{type(pooling).__module__}."
                                f"{type(pooling).__qualname__}"
                            ),
                        }
                        for pooling in self._pooling_strategies.get(
                            feature.name,
                            (),
                        )
                    ],
                }
                for feature in self._feature_engine.features
            ],
        }

    @classmethod
    def _configuration_value(cls, value: Any) -> Any:
        """Convert feature parameters into deterministic JSON values."""
        if isinstance(value, Enum):
            return value.name
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): cls._configuration_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [cls._configuration_value(item) for item in value]
        raise TypeError(
            "Feature parameters must be JSON-compatible; received "
            f"{type(value).__name__}."
        )

    def build(
        self,
        graph_id: Any = None,
        *,
        candidate_subgraphs: Collection[Graph] | None = None,
        sample_ids: Sequence[str] | None = None,
    ) -> list[Sample]:
        """
        Build dataset samples for one complete graph instance.

        The construction process follows these steps:

        1. Load the complete graph.
        2. Compute global features once.
        3. Generate candidate subgraphs.
          4. Compute local features for each candidate subgraph.
          5. Restrict node mappings to candidate vertices and apply configured
              pooling strategies to node-level and edge-level feature results.
        6. Combine global and local features into a feature vector.
        7. Compute the objective value for the candidate subgraph.
        8. Create a Sample containing the feature vector and target.

        Args:
            graph_id: Optional identifier preserved in every sample. If it
                is omitted, the loader's instance directory name is used.

        Returns:
            list[Sample]: Dataset samples generated from the complete graph.
        """
        complete_graph = self._graph_loader.load()

        if graph_id is None:
            graph_id = self._graph_loader.instance_path.name

        # Global features depend only on the complete graph and therefore
        # must be computed once and reused for every candidate subgraph.
        global_context = FeatureContext(
            full_graph=complete_graph,
            subgraph=complete_graph,
        )

        global_results = self._feature_engine.compute_global(global_context)

        if candidate_subgraphs is None:
            candidates = list(self._subgraph_generator.generate(complete_graph))
        else:
            candidates = list(candidate_subgraphs)

        if sample_ids is not None and len(sample_ids) != len(candidates):
            raise ValueError(
                "sample_ids must have the same length as candidate_subgraphs."
            )

        samples = []

        for subgraph_id, subgraph in enumerate(candidates):
            # Local features are computed independently for each candidate
            # subgraph.
            local_context = FeatureContext(
                full_graph=complete_graph,
                subgraph=subgraph,
            )

            local_results = self._feature_engine.compute_local(local_context)
            candidate_vertices = getattr(subgraph, "vertices", None)
            global_features = self._prepare_features(
                global_results,
                candidate_vertices=candidate_vertices,
            )
            local_features = self._prepare_features(
                local_results,
                candidate_vertices=candidate_vertices,
            )

            feature_vector = {
                **global_features,
                **local_features,
            }

            target = self._objective_function.compute(subgraph)

            samples.append(
                Sample(
                    graph_id=graph_id,
                    subgraph_id=subgraph_id,
                    features=feature_vector,
                    target=target,
                    sample_id=(
                        sample_ids[subgraph_id]
                        if sample_ids is not None
                        else None
                    ),
                )
            )

        return samples

    def _prepare_features(
        self,
        results: Mapping[str, Any],
        *,
        candidate_vertices: Collection[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Prepare computed feature values for inclusion in a dataset sample.

        Graph-level features are preserved directly.

        Node-level and edge-level features are aggregated using the
        configured pooling strategies. Node values are restricted to the
        candidate's vertices before pooling, including values computed on
        the complete graph. A feature may have multiple pooling strategies,
        producing multiple dataset features.

        For example, applying mean and max pooling to a node-level feature
        named "degree" produces:

            degree_mean
            degree_max

        Args:
            results: Feature results keyed by feature name.
            candidate_vertices: Optional candidate node identifiers used to
                restrict node-level feature mappings before pooling.

        Returns:
            dict[str, Any]: Prepared feature values.
        """
        prepared = {}

        for feature_name, value in results.items():
            feature = self._features_by_name[feature_name]

            if feature.level == FeatureLevel.GRAPH:
                prepared[feature_name] = value
                continue

            if feature.level in (
                FeatureLevel.NODE,
                FeatureLevel.EDGE,
            ):
                if (
                    feature.level == FeatureLevel.NODE
                    and candidate_vertices is not None
                    and isinstance(value, dict)
                ):
                    value = {
                        vertex: node_value
                        for vertex, node_value in value.items()
                        if vertex in candidate_vertices
                    }

                poolings = self._pooling_strategies.get(
                    feature_name,
                    (),
                )

                for pooling in poolings:
                    prepared[
                        f"{feature_name}_{pooling.name}"
                    ] = pooling.compute(value)

        return prepared