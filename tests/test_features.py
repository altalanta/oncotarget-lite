import numpy as np

from oncotarget.features import create_feature_matrix
from oncotarget.io import merge_gene_feature_table


def test_feature_engineering_outputs_expected_values() -> None:
    merged = merge_gene_feature_table()
    feature_matrix = create_feature_matrix(merged)
    features = feature_matrix.features

    assert np.isclose(features.loc["MSLN", "log2fc_brca"], 1.511620, atol=1e-3)
    assert np.isclose(features.loc["MSLN", "min_normal_tpm"], 0.46, atol=1e-2)
    assert np.isclose(features.loc["MSLN", "mean_dependency"], -0.006625, atol=1e-4)
    assert not bool(features.loc["MSLN", "ig_like_domain"])
    assert np.isclose(features.loc["CEACAM5", "log2fc_coad"], 2.050125, atol=1e-3)


def test_labels_boolean_dtype() -> None:
    feature_matrix = create_feature_matrix(merge_gene_feature_table())
    assert feature_matrix.labels.dtype == bool
