"""Schema validation for data manifest and other JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def validate_manifest_schema(manifest_data: Dict[str, Any]) -> List[str]:
    """Validate data manifest schema and return list of errors."""
    errors = []
    
    # Check required top-level fields
    required_fields = ["version", "generated_by", "description", "sources", "files"]
    for field in required_fields:
        if field not in manifest_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate version
    if "version" in manifest_data:
        version = manifest_data["version"]
        if not isinstance(version, str) or not version:
            errors.append("version must be a non-empty string")
    
    # Validate sources
    if "sources" in manifest_data:
        sources = manifest_data["sources"]
        if not isinstance(sources, dict):
            errors.append("sources must be a dictionary")
        else:
            for source_name, source_info in sources.items():
                if not isinstance(source_info, dict):
                    errors.append(f"Source '{source_name}' must be a dictionary")
                else:
                    if "description" not in source_info:
                        errors.append(f"Source '{source_name}' missing description")
    
    # Validate files
    if "files" in manifest_data:
        files = manifest_data["files"]
        if not isinstance(files, list):
            errors.append("files must be a list")
        else:
            for i, file_info in enumerate(files):
                if not isinstance(file_info, dict):
                    errors.append(f"File {i} must be a dictionary")
                    continue
                
                # Check required file fields
                required_file_fields = ["path", "type", "source", "sha256", "size_bytes"]
                for field in required_file_fields:
                    if field not in file_info:
                        errors.append(f"File {i} missing required field: {field}")
                
                # Validate SHA256 format
                if "sha256" in file_info:
                    sha256 = file_info["sha256"]
                    if not isinstance(sha256, str) or len(sha256) != 64:
                        errors.append(f"File {i} has invalid SHA256 hash format")
                    elif not all(c in '0123456789abcdef' for c in sha256.lower()):
                        errors.append(f"File {i} has invalid SHA256 hash characters")
                
                # Validate size
                if "size_bytes" in file_info:
                    size = file_info["size_bytes"]
                    if not isinstance(size, int) or size < 0:
                        errors.append(f"File {i} has invalid size_bytes (must be non-negative integer)")
    
    return errors


def validate_ablations_config_schema(config_data: Dict[str, Any]) -> List[str]:
    """Validate ablation config schema and return list of errors."""
    errors = []
    
    # Check required top-level fields
    required_fields = ["name", "model", "features", "training", "evaluation", "tags"]
    for field in required_fields:
        if field not in config_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate model section
    if "model" in config_data:
        model = config_data["model"]
        if not isinstance(model, dict):
            errors.append("model must be a dictionary")
        else:
            if "type" not in model:
                errors.append("model missing required field: type")
            elif model["type"] not in ["logreg", "mlp", "xgb"]:
                errors.append(f"Invalid model type: {model['type']} (must be logreg, mlp, or xgb)")
            
            if "params" not in model:
                errors.append("model missing required field: params")
    
    # Validate features section
    if "features" in config_data:
        features = config_data["features"]
        if not isinstance(features, dict):
            errors.append("features must be a dictionary")
        else:
            if "type" not in features:
                errors.append("features missing required field: type")
            elif features["type"] not in ["all_features", "clinical_only", "network_only"]:
                errors.append(f"Invalid feature type: {features['type']}")
    
    # Validate training section
    if "training" in config_data:
        training = config_data["training"]
        if not isinstance(training, dict):
            errors.append("training must be a dictionary")
        else:
            required_training_fields = ["seed", "test_size"]
            for field in required_training_fields:
                if field not in training:
                    errors.append(f"training missing required field: {field}")
    
    # Validate evaluation section
    if "evaluation" in config_data:
        evaluation = config_data["evaluation"]
        if not isinstance(evaluation, dict):
            errors.append("evaluation must be a dictionary")
        else:
            required_eval_fields = ["n_bootstrap", "ci", "bins"]
            for field in required_eval_fields:
                if field not in evaluation:
                    errors.append(f"evaluation missing required field: {field}")
    
    # Validate tags
    if "tags" in config_data:
        tags = config_data["tags"]
        if not isinstance(tags, list):
            errors.append("tags must be a list")
    
    return errors


def validate_file(file_path: Path, schema_type: str) -> List[str]:
    """Validate a JSON file against a schema."""
    errors = []
    
    if not file_path.exists():
        return [f"File does not exist: {file_path}"]
    
    try:
        with open(file_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON in {file_path}: {e}"]
    except Exception as e:
        return [f"Error reading {file_path}: {e}"]
    
    if schema_type == "manifest":
        errors = validate_manifest_schema(data)
    elif schema_type == "ablation_config":
        # For YAML configs, we'd need to load differently, but structure is similar
        errors = validate_ablations_config_schema(data)
    else:
        errors = [f"Unknown schema type: {schema_type}"]
    
    return errors


def validate_all_schemas() -> bool:
    """Validate all schemas in the project and print results."""
    all_valid = True
    
    # Validate data manifest
    manifest_file = Path("data/manifest.json")
    if manifest_file.exists():
        errors = validate_file(manifest_file, "manifest")
        if errors:
            print(f"❌ Data manifest validation failed:")
            for error in errors:
                print(f"  - {error}")
            all_valid = False
        else:
            print("✅ Data manifest schema valid")
    else:
        print("⚠️ Data manifest not found (run pipeline first)")
    
    # Check ablation configs exist
    ablations_dir = Path("configs/ablations")
    if ablations_dir.exists():
        config_files = list(ablations_dir.glob("*.yaml"))
        if config_files:
            print(f"✅ Found {len(config_files)} ablation config files")
        else:
            print("⚠️ No ablation config files found")
    else:
        print("⚠️ Ablation configs directory not found")
    
    # Check required directories exist
    required_dirs = [
        "oncotarget_lite/trainers",
        "scripts",
        "configs/ablations",
        "docs",
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ Required directory exists: {dir_path}")
        else:
            print(f"❌ Missing required directory: {dir_path}")
            all_valid = False
    
    return all_valid


if __name__ == "__main__":
    import sys
    success = validate_all_schemas()
    sys.exit(0 if success else 1)