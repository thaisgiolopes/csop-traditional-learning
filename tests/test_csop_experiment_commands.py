from experiments import run_csop_experiments
from src.experiments.csop_presets import build_csop_configs


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
            == "graph-size-local-degree-and-subgraph-size-v2"
        )
        assert "subgraph_size" in config.features.feature_names
        assert config.subgraph_generation.sample_count == 20
        assert config.subgraph_generation.seed == 17
        assert config.random_seed == 17
        assert config.dataset_split.random_state == 17
        assert config.model.hyperparameters["random_state"] == 17
        assert config.model.hyperparameters["n_estimators"] == 12
        assert config.model.hyperparameters["num_leaves"] == 7
        assert config.output == output_paths


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


def test_list_all_expands_full_configured_batch(monkeypatch, capsys):
    assert run_csop_experiments.main(["list", "--all", "--samples", "20"]) == 0
    output = capsys.readouterr().out
    assert output.count("csop-tds-") == 100
    assert "100 experiment configuration(s) listed; none executed." in output
