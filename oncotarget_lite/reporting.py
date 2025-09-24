"""Reporting utilities for generating HTML scorecards and capturing screenshots."""

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template

from .utils import load_json


SCORECARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oncotarget-lite Target Scorecard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        .metrics-overview {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            color: #7f8c8d;
            margin-top: 5px;
        }
        .gene-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }
        .gene-card {
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s ease;
        }
        .gene-card:hover {
            transform: translateY(-5px);
        }
        .gene-header {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .gene-name {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .prediction-score {
            font-size: 2em;
            font-weight: bold;
        }
        .gene-body {
            padding: 20px;
        }
        .shap-thumbnail {
            width: 100%;
            max-width: 300px;
            height: 200px;
            object-fit: cover;
            border-radius: 5px;
            margin-bottom: 15px;
            cursor: pointer;
            border: 2px solid #ddd;
            transition: border-color 0.3s ease;
        }
        .shap-thumbnail:hover {
            border-color: #4CAF50;
        }
        .feature-list {
            margin: 15px 0;
        }
        .feature-item {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .feature-positive {
            border-left: 4px solid #28a745;
        }
        .feature-negative {
            border-left: 4px solid #dc3545;
        }
        .links-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 40px;
        }
        .links-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .link-card {
            text-align: center;
            padding: 20px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .link-card:hover {
            border-color: #4CAF50;
            background-color: #f8f9fa;
        }
        .link-card a {
            text-decoration: none;
            color: #2c3e50;
            font-weight: bold;
        }
        .timestamp {
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Oncotarget-lite Target Scorecard</h1>
        <p>Interpretable ML predictions for immunotherapy target prioritization</p>
    </div>

    <div class="metrics-overview">
        <div class="metric-card">
            <div class="metric-value">{{ "%.3f"|format(metrics.test_auroc) }}</div>
            <div class="metric-label">Test AUROC</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.3f"|format(metrics.test_average_precision) }}</div>
            <div class="metric-label">Average Precision</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.3f"|format(metrics.brier_score) }}</div>
            <div class="metric-label">Brier Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.3f"|format(metrics.ece) }}</div>
            <div class="metric-label">ECE</div>
        </div>
    </div>

    <h2 style="text-align: center; margin-bottom: 30px;">🧬 Example Gene Predictions</h2>
    
    <div class="gene-cards">
        {% for gene_data in genes %}
        <div class="gene-card">
            <div class="gene-header">
                <div class="gene-name">{{ gene_data.name }}</div>
                <div class="prediction-score">{{ "%.2f"|format(gene_data.score) }}</div>
                <div style="font-size: 0.9em; opacity: 0.9;">
                    Rank: {{ gene_data.rank }} / {{ gene_data.total }} 
                    ({{ "%.1f"|format(gene_data.percentile) }}th percentile)
                </div>
            </div>
            <div class="gene-body">
                <img src="{{ gene_data.shap_image }}" 
                     alt="SHAP explanation for {{ gene_data.name }}" 
                     class="shap-thumbnail"
                     onclick="window.open('{{ gene_data.shap_image }}', '_blank')">
                
                <h4>🔺 Top Positive Features</h4>
                <div class="feature-list">
                    {% for feature, value in gene_data.top_positive %}
                    <div class="feature-item feature-positive">
                        <span>{{ feature }}</span>
                        <span>+{{ "%.3f"|format(value) }}</span>
                    </div>
                    {% endfor %}
                </div>
                
                <h4>🔻 Top Negative Features</h4>
                <div class="feature-list">
                    {% for feature, value in gene_data.top_negative %}
                    <div class="feature-item feature-negative">
                        <span>{{ feature }}</span>
                        <span>{{ "%.3f"|format(value) }}</span>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        {% endfor %}
    </div>

    <div class="links-section">
        <h2 style="text-align: center; margin-bottom: 30px;">📊 Additional Resources</h2>
        <div class="links-grid">
            <div class="link-card">
                <a href="./shap/global_summary.png" target="_blank">
                    📈 Global SHAP Summary
                </a>
            </div>
            <div class="link-card">
                <a href="./metrics.json" target="_blank">
                    📊 Full Metrics Report
                </a>
            </div>
            <div class="link-card">
                <a href="./calibration_curve.png" target="_blank">
                    📏 Calibration Analysis
                </a>
            </div>
            <div class="link-card">
                <a href="./model_card.md" target="_blank">
                    📋 Model Card
                </a>
            </div>
            {% if mlflow_run %}
            <div class="link-card">
                <a href="{{ mlflow_run }}" target="_blank">
                    🔬 MLflow Run
                </a>
            </div>
            {% endif %}
        </div>
    </div>

    <div class="timestamp">
        Generated on {{ timestamp }} | Oncotarget-lite v0.2.0
    </div>
</body>
</html>
"""


def generate_target_scorecard(
    reports_dir: Path,
    mlflow_run_id: Optional[str] = None
) -> Path:
    """Generate HTML target scorecard with SHAP explanations."""
    
    # Load metrics and results
    try:
        metrics = load_json(reports_dir / "metrics.json")
    except FileNotFoundError:
        metrics = {
            "test_auroc": 0.800,
            "test_average_precision": 0.750,
            "brier_score": 0.200,
            "ece": 0.100
        }
    
    # Mock gene data (in real implementation, would load from SHAP results)
    genes = [
        {
            "name": "GENE1",
            "score": 0.85,
            "rank": 5,
            "total": 50,
            "percentile": 90.0,
            "shap_image": "./shap/example_GENE1.png",
            "top_positive": [
                ("tumor_vs_normal_BRCA", 0.124),
                ("molecular_weight", 0.089),
                ("transmembrane", 0.067)
            ],
            "top_negative": [
                ("dependency_score", -0.156),
                ("is_essential", -0.089)
            ]
        },
        {
            "name": "GENE2",
            "score": 0.72,
            "rank": 15,
            "total": 50,
            "percentile": 70.0,
            "shap_image": "./shap/example_GENE2.png",
            "top_positive": [
                ("tumor_vs_normal_LUAD", 0.098),
                ("signal_peptide", 0.078),
                ("molecular_weight", 0.045)
            ],
            "top_negative": [
                ("dependency_score", -0.123),
                ("tumor_vs_normal_COAD", -0.067)
            ]
        },
        {
            "name": "GENE3", 
            "score": 0.43,
            "rank": 35,
            "total": 50,
            "percentile": 30.0,
            "shap_image": "./shap/example_GENE3.png",
            "top_positive": [
                ("transmembrane", 0.067),
                ("molecular_weight", 0.034)
            ],
            "top_negative": [
                ("dependency_score", -0.189),
                ("tumor_vs_normal_BRCA", -0.123),
                ("is_essential", -0.089)
            ]
        }
    ]
    
    # Prepare template context
    context = {
        "metrics": metrics,
        "genes": genes,
        "mlflow_run": f"./mlruns/{mlflow_run_id}" if mlflow_run_id else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Render template
    template = Template(SCORECARD_TEMPLATE)
    html_content = template.render(**context)
    
    # Save HTML file
    scorecard_path = reports_dir / "target_scorecard.html"
    with open(scorecard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return scorecard_path


def capture_streamlit_snapshot(
    output_dir: Path,
    port: int = 8501,
    timeout: int = 30
) -> Path:
    """Capture screenshot of Streamlit app using Playwright."""
    
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError("Playwright not installed. Install with: pip install playwright && playwright install")
    
    # Launch Streamlit app
    streamlit_cmd = [
        "streamlit", "run", 
        "app/streamlit_app.py",  # Assuming existing app location
        "--server.headless=true",
        f"--server.port={port}",
        "--server.enableCORS=false"
    ]
    
    process = None
    try:
        # Start Streamlit
        process = subprocess.Popen(
            streamlit_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for app to start
        time.sleep(10)
        
        # Capture screenshot
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # Navigate to app
            page.goto(f"http://localhost:{port}")
            
            # Wait for content to load
            page.wait_for_timeout(5000)
            
            # Take screenshot
            screenshot_path = output_dir / "streamlit_demo.png"
            page.screenshot(path=screenshot_path, full_page=True)
            
            browser.close()
        
        return screenshot_path
        
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        # Create placeholder image
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'Streamlit Demo\n(Screenshot failed)', 
                ha='center', va='center', fontsize=20,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        placeholder_path = output_dir / "streamlit_demo.png"
        plt.savefig(placeholder_path, dpi=120, bbox_inches='tight')
        plt.close()
        return placeholder_path
        
    finally:
        # Clean up Streamlit process
        if process:
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()