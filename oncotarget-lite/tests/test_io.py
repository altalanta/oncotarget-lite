import pandas as pd

from oncotarget.io import load_raw_data, merge_gene_feature_table


def test_loaders_return_expected_gene_counts() -> None:
    bundle = load_raw_data()
    assert bundle.gtex["gene"].nunique() == 50
    assert bundle.tcga["gene"].nunique() == 50
    assert bundle.depmap.shape[0] == 50


def test_merge_gene_feature_table_has_expected_columns() -> None:
    merged = merge_gene_feature_table()
    assert merged.shape[0] == 50
    assert any(col.startswith("normal_") for col in merged.columns)
    assert any(col.startswith("tumor_") for col in merged.columns)
    assert any(col.startswith("dep_") for col in merged.columns)
    for col in ("is_cell_surface", "signal_peptide", "ppi_degree"):
        assert col in merged.columns


def test_merge_deterministic() -> None:
    merged_first = merge_gene_feature_table()
    merged_second = merge_gene_feature_table()
    pd.testing.assert_frame_equal(merged_first, merged_second)
