from dataclasses import dataclass
from typing import Any


@dataclass
class Sample:
    """
    Represents one candidate-subgraph sample in the CSOP dataset.

    A sample associates a candidate subgraph with its graph identifier,
    feature representation, and objective-function target.

    This class is a data container only. It does not calculate features,
    generate subgraphs, perform pooling, calculate objectives, or run
    machine learning.
    """

    graph_id: Any
    subgraph_id: Any
    features: dict[str, Any]
    target: float