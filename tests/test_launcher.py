import src.launcher as launcher


def test_setup_environment_clears_generated_outputs(tmp_path, monkeypatch):
    root_results = tmp_path / "results"
    run_results = root_results / "local_results"
    plots = run_results / "plots"
    plots.mkdir(parents=True)
    (plots / "old.png").write_text("generated", encoding="utf-8")
    (run_results / "resultados_finales.csv").write_text("old", encoding="utf-8")
    (root_results / "resultados_parciales.csv").write_text("old", encoding="utf-8")
    calls = []
    monkeypatch.setattr(launcher, "RESULTS_DIR", root_results)
    monkeypatch.setattr(launcher, "db_setup", lambda rebuild: calls.append(rebuild))

    launcher.setup_environment(True, True, run_results)

    assert calls == [True]
    assert plots.is_dir()
    assert list(plots.iterdir()) == []
    assert not (run_results / "resultados_finales.csv").exists()
    assert not (root_results / "resultados_parciales.csv").exists()
