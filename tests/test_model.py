from pathlib import Path

import numpy as np
import torch

from oncotarget.features import create_feature_matrix
from oncotarget.io import merge_gene_feature_table
from oncotarget.model import MLPConfig, train_mlp


def test_train_mlp_reproducible(tmp_path) -> None:
    merged = merge_gene_feature_table()
    feature_matrix = create_feature_matrix(merged)
    config = MLPConfig(epochs=40, patience=10, seed=7)

    metrics_first, model_first = train_mlp(
        feature_matrix.features,
        feature_matrix.labels,
        config=config,
        output_dir=tmp_path,
    )
    metrics_second, model_second = train_mlp(
        feature_matrix.features,
        feature_matrix.labels,
        config=config,
        output_dir=tmp_path,
    )

    assert metrics_first == metrics_second
    model_path = Path(tmp_path) / "mlp_model.pt"
    assert model_path.exists()

    with torch.no_grad():
        tensor_inputs = torch.tensor(
            feature_matrix.features.astype("float32").values, dtype=torch.float32
        )
        logits_1 = model_first(tensor_inputs)
        logits_2 = model_second(tensor_inputs)
    np.testing.assert_allclose(logits_1.numpy(), logits_2.numpy(), rtol=1e-5, atol=1e-5)
