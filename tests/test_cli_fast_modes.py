import contextlib
import json
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

mock_shap = types.SimpleNamespace(
    Explainer=lambda *args, **kwargs: types.SimpleNamespace(__call__=lambda *_a, **_k: types.SimpleNamespace(values=[])),
    summary_plot=lambda *args, **kwargs: None,
)
sys.modules.setdefault("shap", mock_shap)

from oncotarget_lite.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def stub_workdir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def stub_mlflow(monkeypatch):
    class DummyMlflow:
        def __init__(self):
            self._uri = "file://./mlruns"

        def set_tracking_uri(self, uri: str) -> None:
            self._uri = uri

        def get_tracking_uri(self) -> str:
            return self._uri

        def set_experiment(self, *_args, **_kwargs) -> None:  # pragma: no cover - trivial stub
            return None

        def log_params(self, *_args, **_kwargs) -> None:
            return None

        def log_metrics(self, *_args, **_kwargs) -> None:
            return None

        def log_artifacts(self, path: str) -> None:
            # Create a placeholder so downstream consumers can read it safely.
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).touch()

        def set_tags(self, *_args, **_kwargs) -> None:
            return None

    stub = DummyMlflow()
    monkeypatch.setattr("oncotarget_lite.cli._mlflow", lambda: stub)

    @contextlib.contextmanager
    def fake_start_run(*_args, **_kwargs):
        yield types.SimpleNamespace(info=types.SimpleNamespace(run_id="stub-run"))

    monkeypatch.setattr("oncotarget_lite.cli._start_run", fake_start_run)
    monkeypatch.setattr("oncotarget_lite.cli._load_run_context", lambda: None)
    return stub


@pytest.fixture(autouse=True)
def stub_pipeline_components(monkeypatch, tmp_path):
    processed_dir = tmp_path / "data" / "processed"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    docs_dir = tmp_path / "docs"
    shap_dir = reports_dir / "shap"

    def fake_prepare_dataset(raw_dir, processed_dir, **_kwargs):
        processed_dir.mkdir(parents=True, exist_ok=True)
        (processed_dir / "features.parquet").write_text("features", encoding="utf-8")
        (processed_dir / "labels.parquet").write_text("labels", encoding="utf-8")
        (processed_dir / "splits.json").write_text(json.dumps({"dataset_hash": "hash"}), encoding="utf-8")
        return types.SimpleNamespace(dataset_fingerprint="hash")

    class DummyTrainConfig:  # pylint: disable=too-few-public-methods
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_train_model(processed_dir, models_dir, reports_dir, config):  # pylint: disable=unused-argument
        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "predictions.parquet").write_text("predictions", encoding="utf-8")
        pipeline = types.SimpleNamespace(named_steps={"clf": types.SimpleNamespace(n_features_in_=4)})
        metrics = {"auroc": 0.9, "ap": 0.8}
        return types.SimpleNamespace(
            pipeline=pipeline,
            dataset_hash="hash",
            train_metrics=metrics,
            test_metrics={"auroc": 0.85, "ap": 0.75},
        )

    def fake_evaluate_predictions(reports_dir, **_kwargs):
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "metrics.json").write_text(json.dumps({"auroc": 0.84}), encoding="utf-8")
        (reports_dir / "bootstrap.json").write_text(
            json.dumps({"auroc": {"mean": 0.84, "ci_half_width": 0.01}}),
            encoding="utf-8",
        )
        (reports_dir / "calibration.json").write_text(
            json.dumps({"mean_expected": 0.5, "mean_observed": 0.5, "bins": []}),
            encoding="utf-8",
        )
        metrics = types.SimpleNamespace(
            auroc=0.84,
            ap=0.77,
            brier=0.2,
            ece=0.1,
            accuracy=0.7,
            f1=0.6,
            overfit_gap=0.05,
        )
        return types.SimpleNamespace(metrics=metrics)

    def fake_generate_shap(**_kwargs):
        shap_dir.mkdir(parents=True, exist_ok=True)
        return types.SimpleNamespace(alias_map={"gene": "GENE001"})

    def fake_generate_scorecard(reports_dir, shap_dir, output_path):  # pylint: disable=unused-argument
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<html>scorecard</html>", encoding="utf-8")
        return output_path

    def fake_build_docs_index(reports_dir, docs_dir, model_card):  # pylint: disable=unused-argument
        docs_dir.mkdir(parents=True, exist_ok=True)
        output = docs_dir / "index.md"
        output.write_text("# Documentation", encoding="utf-8")
        return output

    def fake_capture_streamlit(output_path, timeout):  # pylint: disable=unused-argument
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("image-bytes", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "oncotarget_lite.cli._data_module",
        lambda: types.SimpleNamespace(
            prepare_dataset=fake_prepare_dataset,
            RAW_DIR=Path("data/raw"),
            PROCESSED_DIR=Path("data/processed"),
        ),
    )
    monkeypatch.setattr(
        "oncotarget_lite.cli._model_module",
        lambda: types.SimpleNamespace(
            TrainConfig=DummyTrainConfig,
            train_model=fake_train_model,
            MODELS_DIR=Path("models"),
        ),
    )
    monkeypatch.setattr(
        "oncotarget_lite.cli._eval_module",
        lambda: types.SimpleNamespace(evaluate_predictions=fake_evaluate_predictions),
    )
    monkeypatch.setattr(
        "oncotarget_lite.cli._explain_module",
        lambda: types.SimpleNamespace(generate_shap=fake_generate_shap, SHAP_DIR=Path("reports/shap")),
    )
    monkeypatch.setattr(
        "oncotarget_lite.cli._reporting_module",
        lambda: types.SimpleNamespace(
            generate_scorecard=fake_generate_scorecard,
            build_docs_index=fake_build_docs_index,
        ),
    )
    monkeypatch.setattr("oncotarget_lite.cli.capture_streamlit", fake_capture_streamlit)

    # Ensure downstream commands reference predictable directories
    return {
        "processed_dir": processed_dir,
        "models_dir": models_dir,
        "reports_dir": reports_dir,
        "docs_dir": docs_dir,
        "shap_dir": shap_dir,
    }


def test_prepare_runs_with_fast_profile(stub_pipeline_components):
    result = runner.invoke(
        app,
        [
            "prepare",
            "--fast",
            "--raw-dir",
            str(Path("data/raw")),
            "--processed-dir",
            str(stub_pipeline_components["processed_dir"]),
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_train_invokes_stub(stub_pipeline_components):
    result = runner.invoke(
        app,
        [
            "train",
            "--fast",
            "--processed-dir",
            str(stub_pipeline_components["processed_dir"]),
            "--models-dir",
            str(stub_pipeline_components["models_dir"]),
            "--reports-dir",
            str(stub_pipeline_components["reports_dir"]),
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_eval_supports_global_ci_flag(stub_pipeline_components):
    reports_dir = stub_pipeline_components["reports_dir"]
    result = runner.invoke(
        app,
        [
            "--ci",
            "eval",
            "--reports-dir",
            str(reports_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (reports_dir / "metrics.json").exists()


def test_explain_uses_fast_background(stub_pipeline_components):
    result = runner.invoke(
        app,
        [
            "explain",
            "--fast",
            "--processed-dir",
            str(stub_pipeline_components["processed_dir"]),
            "--models-dir",
            str(stub_pipeline_components["models_dir"]),
            "--shap-dir",
            str(stub_pipeline_components["shap_dir"]),
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_scorecard_generates_html(stub_pipeline_components):
    output_path = Path("reports/target_scorecard.html")
    result = runner.invoke(
        app,
        [
            "scorecard",
            "--reports-dir",
            str(stub_pipeline_components["reports_dir"]),
            "--shap-dir",
            str(stub_pipeline_components["shap_dir"]),
            "--output-path",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert output_path.exists()


def test_docs_command_produces_index(stub_pipeline_components, tmp_path):
    model_card = tmp_path / "model_card.md"
    model_card.write_text("# Model Card", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "docs",
            "--reports-dir",
            str(stub_pipeline_components["reports_dir"]),
            "--docs-dir",
            str(stub_pipeline_components["docs_dir"]),
            "--model-card",
            str(model_card),
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_snapshot_creates_placeholder_image():
    output = Path("reports/streamlit_demo.png")
    result = runner.invoke(app, ["snapshot", "--output-path", str(output)])
    assert result.exit_code == 0, result.stdout
    assert output.exists()


def test_all_pipeline_runs_in_ci_profile():
    result = runner.invoke(app, ["--fast", "all"])
    assert result.exit_code == 0, result.stdout
