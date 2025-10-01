"""Renders docs/index.md section with benchmark metrics table."""

from __future__ import annotations

import json
from pathlib import Path


def load_metrics_summary() -> dict:
    """Load the metrics summary JSON file."""
    summary_path = Path("docs/metrics_summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"Metrics summary file not found: {summary_path}")
    
    with open(summary_path) as f:
        return json.load(f)


def format_metrics_table(summary: dict) -> str:
    """Format the metrics data as a Markdown table."""
    metrics = summary["metrics"]
    dataset = summary["dataset"]
    n = summary["n"]
    git_sha = summary["git_sha"]
    
    # Table header
    table_lines = [
        "| Metric | Mean | 95% CI | N | Dataset | Git SHA |",
        "|--------|------|--------|---|---------|---------|"
    ]
    
    # Table rows
    metric_names = ["auroc", "auprc", "accuracy", "ece"]
    metric_display = ["AUROC", "AUPRC", "Accuracy", "ECE"]
    
    for metric_name, display_name in zip(metric_names, metric_display):
        if metric_name in metrics:
            metric_data = metrics[metric_name]
            mean_val = metric_data["mean"]
            ci_low, ci_high = metric_data["ci95"]
            
            row = f"| {display_name} | {mean_val:.3f} | [{ci_low:.3f}, {ci_high:.3f}] | {n} | {dataset} | {git_sha} |"
            table_lines.append(row)
    
    return "\n".join(table_lines)


def update_docs_index(table_content: str) -> None:
    """Update or create docs/index.md with the metrics table."""
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    index_path = docs_dir / "index.md"
    
    # Markers for the metrics section
    begin_marker = "<!-- BEGIN: METRICS TABLE -->"
    end_marker = "<!-- END: METRICS TABLE -->"
    
    # New metrics section content
    new_section = f"""{begin_marker}

## Benchmark (deterministic CI)

{table_content}

{end_marker}"""
    
    if index_path.exists():
        # Read existing content
        with open(index_path) as f:
            content = f.read()
        
        # Check if markers exist
        if begin_marker in content and end_marker in content:
            # Replace existing section
            start_idx = content.find(begin_marker)
            end_idx = content.find(end_marker) + len(end_marker)
            
            new_content = content[:start_idx] + new_section + content[end_idx:]
        else:
            # Append new section at the end
            new_content = content.rstrip() + "\n\n" + new_section + "\n"
    else:
        # Create new file with just the metrics section
        new_content = f"""# oncotarget-lite

{new_section}
"""
    
    # Write updated content
    with open(index_path, "w") as f:
        f.write(new_content)
    
    print(f"Updated {index_path} with metrics table")


def main() -> None:
    """Render the metrics table in docs/index.md."""
    print("Rendering metrics documentation...")
    
    try:
        # Load metrics summary
        summary = load_metrics_summary()
        
        # Format as table
        table_content = format_metrics_table(summary)
        
        # Update docs
        update_docs_index(table_content)
        
        print("Metrics table rendered successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the benchmark first: python -m scripts.eval_small_benchmark")
        exit(1)
    except Exception as e:
        print(f"Error rendering metrics table: {e}")
        exit(1)


if __name__ == "__main__":
    main()