from __future__ import annotations

from pathlib import Path

import pytest

from oncotarget_lite.config import AppConfig, ConfigError, load_config

SMALL_EPOCHS = 5


def test_default_config_points_to_raw_data(default_config: AppConfig) -> None:
    assert default_config.data.raw_dir.exists()
    assert default_config.data.dataset_name == "synthetic"


def test_overrides_applied(tmp_path: Path) -> None:
    config = load_config(
        overrides={
            "training": {"device": "cpu", "max_epochs": SMALL_EPOCHS},
            "artifacts": {"base_dir": str(tmp_path)},
        }
    )
    assert config.training.max_epochs == SMALL_EPOCHS
    assert config.training.device == "cpu"
    assert config.artifacts.base_dir == tmp_path.resolve()


def test_invalid_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError):
        load_config(config_path)
