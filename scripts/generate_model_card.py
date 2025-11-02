#!/usr/bin/env python3
"""
Generates a dynamic model card from a template and various project artifacts.
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
import yaml

def get_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "N/A"

def get_dvc_data_hash() -> str:
    """Get the data hash from dvc.lock for the processed data directory."""
    try:
        with open("dvc.lock") as f:
            lock_data = yaml.safe_load(f)
        return lock_data['stages']['prepare']['outs'][0]['md5']
    except (FileNotFoundError, KeyError, IndexError):
        return "N/A"

def main():
    """Main function to generate the model card."""
    print("🚀 Generating dynamic model card...")

    # Define paths
    template_path = Path("oncotarget_lite/model_card_template.md")
    output_path = Path("docs/model_card.md")
    metrics_path = Path("eval_results/evaluation_metrics.json")
    model_info_path = Path("eval_results/model_info.json")

    # Load data from artifacts
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Metrics file not found at {metrics_path}. Using placeholder data.")
        metrics = {k: "N/A" for k in ["auroc", "ap", "accuracy", "f1", "brier", "ece"]}

    try:
        with open(model_info_path) as f:
            model_info = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Model info file not found at {model_info_path}. Using placeholder data.")
        model_info = {"model_version": "N/A", "model_type": "N/A", "training_params": {}}

    # Prepare the context for the template
    context = {
        "model_version": model_info.get("model_version", "N/A"),
        "model_type": model_info.get("model_type", "N/A"),
        "release_date": datetime.now().strftime("%Y-%m-%d"),
        "git_commit": get_git_commit(),
        "dvc_data_hash": get_dvc_data_hash(),
        "auroc": f"{metrics.get('auroc', 0):.4f}",
        "ap": f"{metrics.get('ap', 0):.4f}",
        "accuracy": f"{metrics.get('accuracy', 0):.4f}",
        "f1": f"{metrics.get('f1', 0):.4f}",
        "brier": f"{metrics.get('brier', 0):.4f}",
        "ece": f"{metrics.get('ece', 0):.4f}",
    }

    # Format training parameters into a markdown table
    params = model_info.get("training_params", {})
    params_table = "\n".join(f"| {key} | `{value}` |" for key, value in params.items())
    context["training_parameters"] = params_table

    # Read the template
    try:
        with open(template_path) as f:
            template_content = f.read()
    except FileNotFoundError:
        print(f"Error: Model card template not found at {template_path}")
        return

    # Populate the template
    populated_content = template_content
    for key, value in context.items():
        placeholder = f"{{{{ {key} }}}}"
        populated_content = populated_content.replace(placeholder, str(value))

    # Write the output file
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        f.write(populated_content)

    print(f"✅ Model card successfully generated at: {output_path}")

if __name__ == "__main__":
    main()
