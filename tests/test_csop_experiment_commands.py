from experiments import run_csop_experiments
from src.experiments.csop_presets import build_csop_configs
from src.application.csop_experiment_runner import _build_feature_set


def test_presets_expose_selected_factors_and_isolated_output_paths(tmp_path):
    output_paths = {
        "experiments_root": str(tmp_path / "runs"),
        "database_path": str(tmp_path / "db.sqlite3"),
        "report_path": str(tmp_path / "report.xlsx"),
        "processed_root": str(tmp_path / "processed"),
    }
    configs = build_csop_configs(
        datasets=["amazon", "citeseer"],
        subgraph_sizes=[10, 40],
        sample_count=20,
        seed=17,
        max_attempts=321,
        model_hyperparameters={"n_estimators": 12, "num_leaves": 7},
        output_paths=output_paths,
    )

    assert len(configs) == 4
    assert [config.graph.source for config in configs] == [
        "amazon",
        "amazon",
        "citeseer",
        "citeseer",
    ]
    for config in configs:
        assert (
            config.features.feature_set_id
            == "candidate-structure-local-v2"
        )
        assert "subgraph_size" not in config.features.feature_names
        assert config.subgraph_generation.sample_count == 20
        assert config.subgraph_generation.seed == 17
        assert config.random_seed == 17
        assert config.dataset_split.random_state == 17
        assert config.model.hyperparameters["random_state"] == 17
        assert config.model.hyperparameters["n_estimators"] == 12
        assert config.model.hyperparameters["num_leaves"] == 7
        assert config.output == output_paths


def test_global_feature_preset_matches_runner_column_order():
    config = build_csop_configs(
        datasets=["citeseer"],
        subgraph_sizes=[10],
        node_features=[
            "degree",
            "pagerank",
            "k_core_number",
            "average_neighbor_degree",
        ],
        node_scope="global",
    )[0]

    assert config.features.feature_names == [
        "degree_mean",
        "degree_max",
        "degree_min",
        "degree_std",
        "pagerank_global_mean",
        "pagerank_global_max",
        "pagerank_global_min",
        "pagerank_global_std",
        "k_core_number_global_mean",
        "k_core_number_global_max",
        "k_core_number_global_min",
        "k_core_number_global_std",
        "average_neighbor_degree_global_mean",
        "average_neighbor_degree_global_max",
        "average_neighbor_degree_global_min",
        "average_neighbor_degree_global_std",
        "num_vertices",
        "num_edges",
    ]


def test_list_and_validate_commands_do_not_execute_or_create_outputs(
    tmp_path,
    capsys,
    monkeypatch,
):
    monkeypatch.setattr(
        run_csop_experiments,
        "_run_configs",
        lambda configs: (_ for _ in ()).throw(AssertionError("executed")),
    )
    output_root = tmp_path / "must-not-be-created"

    assert run_csop_experiments.main(
        [
            "list",
            "--graph",
            "amazon",
            "--size",
            "10",
            "--samples",
            "20",
            "--output-root",
            str(output_root),
        ]
    ) == 0
    listed = capsys.readouterr().out
    assert "graph=amazon (V=334863, E=925872)" in listed
    assert "1 experiment configuration(s) listed; none executed." in listed

    assert run_csop_experiments.main(
        [
            "validate",
            "--graph",
            "citeseer",
            "--size",
            "10",
            "--samples",
            "20",
            "--output-root",
            str(output_root),
        ]
    ) == 0
    validated = capsys.readouterr().out
    assert "graph=citeseer (V=3312, E=4536)" in validated
    assert "1 experiment configuration(s) validated; none executed." in validated
    assert not output_root.exists()


def test_run_requires_explicit_single_selection_or_all(monkeypatch):
    called = []
    monkeypatch.setattr(
        run_csop_experiments,
        "_run_configs",
        lambda configs: called.extend(configs) or 0,
    )

    assert run_csop_experiments.main(
        [
            "run",
            "--graph",
            "citeseer",
            "--size",
            "20",
            "--samples",
            "20",
        ]
    ) == 0
    assert len(called) == 1
    assert called[0].graph.source == "citeseer"
    assert called[0].subgraph_generation.parameters["subgraph_size"] == 20


def test_cli_selects_a_subset_of_local_node_features(monkeypatch):
    captured = []
    monkeypatch.setattr(
        run_csop_experiments,
        "_run_configs",
        lambda configs: captured.extend(configs) or 0,
    )

    assert run_csop_experiments.main(
        [
            "run",
            "--graph", "citeseer",
            "--size", "10",
            "--size", "20",
            "--samples", "100",
            "--node-feature", "pagerank",
            "--node-feature", "triangle_count",
            "--node-scope", "local",
        ]
    ) == 0
    assert len(captured) == 2
    for config in captured:
        assert config.features.enabled_features == [
            "pagerank", "triangle_count"
        ]
        assert "pagerank_local_mean" in config.features.feature_names
        assert "triangle_count_local_std" in config.features.feature_names


def test_cli_can_use_global_node_features_or_disable_them(monkeypatch):
    captured = []
    monkeypatch.setattr(
        run_csop_experiments,
        "_run_configs",
        lambda configs: captured.extend(configs) or 0,
    )
    run_csop_experiments.main(
        [
            "run", "--graph", "citeseer", "--size", "10", "--samples", "20",
            "--node-feature", "pagerank", "--node-scope", "global",
        ]
    )
    assert captured[0].features.enabled_features == ["pagerank"]
    assert "pagerank_global_mean" in captured[0].features.feature_names

    captured.clear()
    run_csop_experiments.main(
        [
            "run", "--graph", "citeseer", "--size", "10", "--samples", "20",
            "--no-node-features",
        ]
    )
    assert captured[0].features.enabled_features == []
    assert captured[0].features.feature_names == [
        "num_vertices", "num_edges"
    ]


def test_selected_node_features_match_runner_output_schema():
    engine, pooling = _build_feature_set(
        "graph-size-node-structure-local-with-subgraph-size-v1",
        enabled_features=["pagerank", "triangle_count"],
    )
    generated_names = [feature.name for feature in engine.features[:2]]
    generated_names.extend(
        f"{feature.name}_{strategy.name}"
        for feature in engine.features[2:]
        for strategy in pooling[feature.name]
    )
    config = build_csop_configs(
        datasets=["citeseer"],
        subgraph_sizes=[10],
        node_features=["pagerank", "triangle_count"],
    )[0]
    assert config.features.feature_names == generated_names


def test_list_all_expands_full_configured_batch(monkeypatch, capsys):
    assert run_csop_experiments.main(["list", "--all", "--samples", "20"]) == 0
    output = capsys.readouterr().out
    assert output.count("csop-tds-") == 150
    assert "150 experiment configuration(s) listed; none executed." in output


def test_time_series_size_validation_requires_equal_spacing_and_more_sizes_than_folds(
    capsys,
):
    assert run_csop_experiments.main(
        [
            "validate",
            "--graph", "amazon",
            "--size", "10", "--size", "20", "--size", "30",
            "--size", "40", "--size", "50", "--size", "60",
            "--size", "70",
            "--samples", "20",
            "--split-strategy", "time_series_size",
            "--folds", "5",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "split=time_series_size({'n_splits': 5})" in output
    assert "7 experiment configuration(s) validated" in output


def test_transfer_validation_reads_metadata_without_running_experiment(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        run_csop_experiments,
        "run_graph_transfer_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("transfer executed during validation")
        ),
    )
    assert run_csop_experiments.main(
        [
            "transfer", "validate",
            "--train-graph", "amazon",
            "--test-graph", "patents",
            "--size", "10", "--size", "20",
            "--samples", "20",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "graph=patents (V=2745761, E=13965409)" in output
    assert "One model will train on amazon and evaluate on unseen graph patents" in output
    assert "4 graph/size configurations validated; none executed." in output
