import pandas as pd

from oncotarget.scoring import explain_gene_score, score_candidates


def test_score_monotonic_wrt_log2fc() -> None:
    features = pd.DataFrame(
        {
            "log2fc_brca": [0.5, 1.0],
            "log2fc_luad": [0.5, 1.0],
            "log2fc_coad": [0.5, 1.0],
            "min_normal_tpm": [1.0, 1.0],
            "mean_tumor_tpm": [2.0, 2.0],
            "mean_dependency": [-0.5, -0.5],
            "ppi_degree": [40, 40],
            "signal_peptide": [1, 1],
            "ig_like_domain": [0, 0],
            "protein_length": [600, 600],
        },
        index=["LOW", "HIGH"],
    )
    labels = pd.Series([False, False], index=features.index)
    scores = score_candidates(features, labels)
    assert scores.loc["HIGH", "score"] > scores.loc["LOW", "score"]


def test_explain_gene_score_returns_components() -> None:
    features = pd.DataFrame(
        {
            "log2fc_brca": [1.0],
            "log2fc_luad": [1.0],
            "log2fc_coad": [1.0],
            "min_normal_tpm": [0.5],
            "mean_tumor_tpm": [2.0],
            "mean_dependency": [-0.2],
            "ppi_degree": [45],
            "signal_peptide": [1],
            "ig_like_domain": [1],
            "protein_length": [500],
        },
        index=["TEST"],
    )
    labels = pd.Series([True], index=features.index)
    scores = score_candidates(features, labels)
    breakdown = explain_gene_score("TEST", scores)
    expected_keys = {
        "log2fc_component",
        "dependency_component",
        "surface_component",
        "ig_like_component",
        "ppi_component",
        "low_normal_component",
        "score",
        "rank",
    }
    assert expected_keys.issubset(breakdown.keys())
