from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from oncotarget_lite import cli

runner = CliRunner()
GENE_COUNT = 50
TOP_TARGETS = 3


def test_system_info_cli() -> None:
    result = runner.invoke(cli.app, ["system-info"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "python" in data


def test_validate_data_cli() -> None:
    result = runner.invoke(cli.app, ["validate-data"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["genes"] == GENE_COUNT


def test_train_evaluate_report_cli(tmp_path: Path) -> None:
    train_result = runner.invoke(
        cli.app,
        [
            "train",
            "--out",
            str(tmp_path),
            "--seed",
            "2024",
            "--device",
            "cpu",
        ],
    )
    assert train_result.exit_code == 0
    train_payload = json.loads(train_result.stdout)
    run_dir = Path(train_payload["artifacts"])
    assert run_dir.exists()

    evaluate_result = runner.invoke(cli.app, ["evaluate", "--run-dir", str(run_dir)])
    assert evaluate_result.exit_code == 0
    eval_payload = json.loads(evaluate_result.stdout)
    assert eval_payload["stored_test_metrics"]

    report_result = runner.invoke(
        cli.app, ["report", "--run-dir", str(run_dir), "--top-k", str(TOP_TARGETS)]
    )
    assert report_result.exit_code == 0
    report_payload = json.loads(report_result.stdout)
    assert len(report_payload["top_targets"]) == TOP_TARGETS
