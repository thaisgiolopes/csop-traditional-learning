"""List, validate, and run explicitly selected CSOP TDS experiments.

Examples from the repository root::

    python -m experiments.run_csop_experiments pilot --output-root /tmp/csop-pilot
    python -m experiments.run_csop_experiments validate \
        --graph amazon --size 10 --samples 20
    python -m experiments.run_csop_experiments run \
        --graph amazon --size 10 --samples 100
    python -m experiments.run_csop_experiments run \
        --graph amazon --all-sizes --samples 100
    python -m experiments.run_csop_experiments run --all --samples 100

The ``--all`` option is required to execute the full graph-by-size batch.
"""

import argparse
from collections.abc import Sequence
from math import ceil
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.application.csop_experiment_runner import run_experiment_batch
from src.experiments.csop_presets import (
    SUPPORTED_DATASETS,
    SUPPORTED_SUBGRAPH_SIZES,
    build_csop_configs,
)
from src.experiments.config import ExperimentConfig
from src.model.lightgbm_model import LightGBMModel


# _PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PILOT_SAMPLE_COUNT = 20
_PILOT_SIZE = 10


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--graph",
        dest="graphs",
        action="append",
        choices=SUPPORTED_DATASETS,
        help="Select a graph; repeat to select multiple graphs.",
    )
    parser.add_argument(
        "--size",
        dest="sizes",
        action="append",
        type=int,
        choices=SUPPORTED_SUBGRAPH_SIZES,
        help="Select a target size; repeat to select multiple sizes.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Select all supported graph and target-size combinations.",
    )
    parser.add_argument(
        "--all-sizes",
        action="store_true",
        help="Select every supported size for the explicitly selected graph(s).",
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=5_000)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--model-jobs", type=int, default=1)


def _output_paths(root: Path) -> dict[str, str]:
    resolved_root = root.expanduser().resolve()
    return {
        "experiments_root": str(resolved_root / "runs"),
        "database_path": str(resolved_root / "experiments.sqlite3"),
        "report_path": str(resolved_root / "reports" / "experiments.xlsx"),
        "processed_root": str(resolved_root / "processed"),
    }


def _selection(
    args: argparse.Namespace,
    *,
    default_all: bool = False,
    default_graphs: Sequence[str] | None = None,
    default_sizes: Sequence[int] | None = None,
) -> tuple[list[str], list[int]]:
    if args.all and (args.graphs or args.sizes or args.all_sizes):
        raise ValueError("--all cannot be combined with --graph, --size, or --all-sizes.")
    if args.all_sizes and args.sizes:
        raise ValueError("--all-sizes cannot be combined with --size.")

    if args.all:
        graphs = list(SUPPORTED_DATASETS)
        sizes = list(SUPPORTED_SUBGRAPH_SIZES)
    else:
        graphs = list(args.graphs or default_graphs or ())
        sizes = list(
            SUPPORTED_SUBGRAPH_SIZES
            if args.all_sizes
            else args.sizes or default_sizes or ()
        )
        if default_all:
            graphs = graphs or list(SUPPORTED_DATASETS)
            sizes = sizes or list(SUPPORTED_SUBGRAPH_SIZES)

    if not graphs:
        raise ValueError("Select at least one graph with --graph or --all.")
    if not sizes:
        raise ValueError("Select at least one target size with --size or --all.")
    if len(graphs) != len(set(graphs)):
        raise ValueError("Graph selections must not contain duplicates.")
    if len(sizes) != len(set(sizes)):
        raise ValueError("Target-size selections must not contain duplicates.")
    return graphs, sizes


def _model_parameters(args: argparse.Namespace) -> dict[str, int | float]:
    values = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "n_jobs": args.model_jobs,
        "verbosity": -1,
    }
    if values["n_estimators"] <= 0:
        raise ValueError("--n-estimators must be positive.")
    if not 0 < values["learning_rate"] <= 1:
        raise ValueError("--learning-rate must be greater than 0 and at most 1.")
    if values["num_leaves"] < 2:
        raise ValueError("--num-leaves must be at least 2.")
    if values["min_child_samples"] <= 0:
        raise ValueError("--min-child-samples must be positive.")
    if values["n_jobs"] == 0 or values["n_jobs"] < -1:
        raise ValueError("--model-jobs must be -1 or a positive integer.")
    return values


def _build_configs(
    *,
    graphs: Sequence[str],
    sizes: Sequence[int],
    samples: int,
    seed: int,
    max_attempts: int,
    output_root: Path,
    model_parameters: dict[str, int | float],
) -> list[ExperimentConfig]:
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("--seed must be between 0 and 4294967295.")
    if samples < 10:
        raise ValueError(
            "--samples must be at least 10 so train, validation, and test "
            "splits each contain at least two samples for regression metrics."
        )
    if max_attempts <= 0:
        raise ValueError("--max-attempts must be positive.")
    return build_csop_configs(
        datasets=graphs,
        subgraph_sizes=sizes,
        sample_count=samples,
        seed=seed,
        max_attempts=max_attempts,
        model_hyperparameters=model_parameters,
        output_paths=_output_paths(output_root),
    )


def _validate_config(config: ExperimentConfig) -> tuple[int, int]:
    """Validate inputs without loading the potentially large graph into RAM."""
    graph_path = _PROJECT_ROOT / "data" / "raw" / str(config.graph.source)
    metadata_path = graph_path / "metadata"
    adjacency_path = graph_path / "adjlists"
    if not graph_path.is_dir():
        raise FileNotFoundError(f"Graph directory does not exist: {graph_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Graph metadata does not exist: {metadata_path}")
    if not adjacency_path.is_file() and not (graph_path / "adjlist").is_file():
        raise FileNotFoundError(f"Graph adjacency list is missing in {graph_path}")

    header = metadata_path.read_text(encoding="utf-8").split()
    if len(header) != 2:
        raise ValueError(f"Invalid graph metadata in {metadata_path}.")
    vertex_count, edge_count = (int(value) for value in header)
    target_size = config.subgraph_generation.parameters["subgraph_size"]
    if target_size > vertex_count:
        raise ValueError(
            f"Target size {target_size} exceeds {config.graph.source} "
            f"vertex count {vertex_count}."
        )

    sample_count = config.subgraph_generation.sample_count
    test_count = ceil(sample_count * config.dataset_split.test_ratio)
    development_count = sample_count - test_count
    validation_count = ceil(
        development_count * config.dataset_split.validation_size
    )
    training_count = development_count - validation_count
    if min(training_count, validation_count, test_count) < 2:
        raise ValueError(
            "The configured sample count must produce at least two samples "
            "in each train, validation, and test split."
        )
    return vertex_count, edge_count


def _describe(config: ExperimentConfig, vertex_count: int, edge_count: int) -> str:
    return (
        f"{config.name}: graph={config.graph.source} "
        f"(V={vertex_count}, E={edge_count}), "
        f"k={config.subgraph_generation.parameters['subgraph_size']}, "
        f"samples={config.subgraph_generation.sample_count}, "
        f"seed={config.random_seed}, model={config.model.model_type}"
        f"{config.model.hyperparameters}"
    )


def _run_configs(configs: Sequence[ExperimentConfig]) -> int:
    groups: dict[str, list[ExperimentConfig]] = {}
    for config in configs:
        groups.setdefault(config.graph.source, []).append(config)
    for index, (graph, graph_configs) in enumerate(groups.items(), start=1):
        sizes = sorted(
            config.subgraph_generation.parameters["subgraph_size"]
            for config in graph_configs
        )
        print(
            f"[{index}/{len(groups)}] Running one {graph} experiment "
            f"with one shared model across sizes {sizes}",
            flush=True,
        )
        experiment_id = run_experiment_batch(
            graph_configs,
            model_factory=lambda model_config: LightGBMModel(
                **model_config.hyperparameters
            ),
            model_serializer=lambda model: (
                model._estimator.booster_.model_to_string()
            ),
        )
        print(f"Completed {experiment_id}", flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("list", "validate", "run"):
        command_parser = subparsers.add_parser(name)
        _add_selection_arguments(command_parser)
        _add_run_arguments(command_parser)

    pilot_parser = subparsers.add_parser("pilot")
    pilot_parser.add_argument(
        "--graph",
        dest="graphs",
        action="append",
        choices=SUPPORTED_DATASETS,
        help="Select a graph; defaults to both Amazon and CiteSeer.",
    )
    _add_run_arguments(pilot_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        pilot = args.command == "pilot"
        if pilot:
            graphs = list(args.graphs or SUPPORTED_DATASETS)
            if len(graphs) != len(set(graphs)):
                raise ValueError("Graph selections must not contain duplicates.")
            sizes = [_PILOT_SIZE]
            samples = (
                args.samples
                if args.samples is not None
                else _PILOT_SAMPLE_COUNT
            )
            output_root = args.output_root or Path("experiments/csop-pilot")
        else:
            graphs, sizes = _selection(args, default_all=args.command == "list")
            samples = args.samples if args.samples is not None else 100
            output_root = args.output_root or Path("experiments/")
            if args.command == "run" and args.all is False and (
                not args.graphs or (not args.sizes and not args.all_sizes)
            ):
                raise ValueError(
                    "run requires explicit --graph and --size selections, "
                    "or --all for the full configured batch."
                )

        model_parameters = _model_parameters(args)
        configs = _build_configs(
            graphs=graphs,
            sizes=sizes,
            samples=samples,
            seed=args.seed,
            max_attempts=args.max_attempts,
            output_root=output_root,
            model_parameters=model_parameters,
        )

        validated = []
        for config in configs:
            vertex_count, edge_count = _validate_config(config)
            validated.append((config, vertex_count, edge_count))

        if args.command in {"list", "validate"}:
            for config, vertex_count, edge_count in validated:
                print(_describe(config, vertex_count, edge_count))
            action = "validated" if args.command == "validate" else "listed"
            print(
                f"{len(configs)} experiment configuration(s) {action}; "
                f"none executed. They form "
                f"{len({config.graph.source for config in configs})} graph "
                "group(s) when run."
            )
            return 0

        if pilot:
            print("Validated pilot configurations; starting selected pilot runs.")
        return _run_configs(configs)
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())
