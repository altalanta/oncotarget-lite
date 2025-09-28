from __future__ import annotations

from oncotarget_lite.reports import summarise_run


def test_summarise_run(pipeline_result) -> None:
    summary = summarise_run(pipeline_result.output_dir)
    assert summary["top_targets"]
    assert summary["test_metrics"]

