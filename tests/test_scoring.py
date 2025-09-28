from __future__ import annotations

import pandas as pd

from oncotarget_lite.scoring import explain_score, score_targets


def test_score_targets_and_explain() -> None:
    features = pd.DataFrame(
        {
            "log2fc_br": [2.0, 1.0],
            "log2fc_luad": [1.5, 0.5],
            "mean_dependency": [-0.2, -0.5],
            "min_normal_tpm": [0.1, 3.0],
            "ppi_degree": [30, 60],
            "signal_peptide": [1.0, 0.0],
            "ig_like_domain": [1.0, 0.0],
            "protein_length": [500, 700],
        },
        index=["GENE_A", "GENE_B"],
    )
    labels = pd.Series([True, False], index=features.index)
    scores = score_targets(features, labels)
    assert list(scores.index)[0] == "GENE_A"
    explanation = explain_score("GENE_A", scores)
    assert explanation["rank"] == 1
    assert "log2fc_component" in explanation

