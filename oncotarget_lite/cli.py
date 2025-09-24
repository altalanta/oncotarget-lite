"""Enhanced command-line interface with configuration management and monitoring.

Provides production-ready CLI commands with comprehensive logging,
configuration validation, and performance monitoring.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import mlflow
import typer
from typing_extensions import Annotated

from .config import Config, load_config, get_output_dir
from .data import load_data, split_data
from .eval import comprehensive_evaluation
from .exceptions import OncotargetError
from .logging_utils import get_logger, log_config_summary, log_system_info
from .model import track_experiment, train_mlp, train_random_forest
from .performance import monitor_resources, optimize_memory, log_system_info
from .utils import ensure_dir, set_random_seed

app = typer.Typer(
    help="Production-ready oncotarget-lite ML pipeline with comprehensive monitoring and governance.",
    epilog="Visit https://github.com/your-org/oncotarget-lite for documentation and examples."
)


@app.command()
def config(
    show: Annotated[bool, typer.Option("--show", help="Show current configuration")] = False,
    validate: Annotated[bool, typer.Option("--validate", help="Validate configuration")] = False,
    config_file: Annotated[Optional[str], typer.Option(help="Configuration file path")] = None
) -> None:
    """Manage pipeline configuration."""
    try:
        config_path = Path(config_file) if config_file else None
        config_obj = load_config(config_path)
        
        if show or not validate:
            typer.echo("Current Configuration:")
            typer.echo("=" * 50)
            
            config_dict = config_obj.dict()
            for section, values in config_dict.items():
                typer.echo(f"[{section}]")
                if isinstance(values, dict):
                    for key, value in values.items():
                        typer.echo(f"  {key}: {value}")
                else:
                    typer.echo(f"  {values}")
                typer.echo()
        
        if validate:
            typer.echo("✓ Configuration is valid!", fg=typer.colors.GREEN)
            
    except Exception as e:
        typer.echo(f"Configuration error: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command()
def system_info() -> None:
    """Display comprehensive system information."""
    logger = get_logger(__name__)
    log_system_info(logger)
    
    typer.echo("System Information:")
    typer.echo("=" * 50)
    
    import torch
    import pandas as pd
    import numpy as np
    
    typer.echo(f"Python: {sys.version}")
    typer.echo(f"NumPy: {np.__version__}")
    typer.echo(f"Pandas: {pd.__version__}")
    typer.echo(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        typer.echo(f"CUDA: {torch.version.cuda}")
        typer.echo(f"GPU: {torch.cuda.get_device_name()}")
    else:
        typer.echo("CUDA: Not available")


@app.command()  
def validate_data(
    data_dir: Annotated[Optional[str], typer.Option(help="Data directory path")] = None
) -> None:
    """Validate input data integrity and format."""
    try:
        logger = get_logger(__name__)
        logger.info("Starting data validation...")
        
        data_path = Path(data_dir) if data_dir else None
        features, labels = load_data(data_path)
        
        typer.echo("✓ Data validation successful!", fg=typer.colors.GREEN)
        typer.echo(f"Features shape: {features.shape}")
        typer.echo(f"Label distribution: {dict(labels.value_counts())}")
        
    except Exception as e:
        typer.echo(f"Data validation failed: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command()
def train(
    model_type: Annotated[str, typer.Option(help="Model type (mlp, random_forest)")] = "random_forest",
    config_file: Annotated[Optional[str], typer.Option(help="Configuration file path")] = None,
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory")] = None,
    data_dir: Annotated[Optional[str], typer.Option(help="Data directory")] = None,
    enable_monitoring: Annotated[bool, typer.Option(help="Enable performance monitoring")] = True,
    optimize_memory: Annotated[bool, typer.Option(help="Optimize memory usage")] = True
) -> None:
    """Train a model with comprehensive monitoring and configuration management."""
    
    try:
        # Load configuration
        config_path = Path(config_file) if config_file else None
        config = load_config(config_path)
        
        # Setup logging
        logger = get_logger(__name__, config.logging)
        log_config_summary(logger, config)
        
        # Memory optimization
        if optimize_memory:
            memory_stats = optimize_memory()
            logger.info(f"Memory optimized: {memory_stats}")
        
        typer.echo(f"Training {model_type} model with enhanced monitoring...")
        
        # Determine output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = get_output_dir(config, model_type)
        
        ensure_dir(output_path)
        
        # Performance monitoring context
        monitor_context = monitor_resources(f"{model_type} training") if enable_monitoring else None
        
        with (monitor_context or typer.Context()):
            # Load and split data
            data_path = Path(data_dir) if data_dir else config.data.data_dir
            features, labels = load_data(data_path)
            X_train, X_test, y_train, y_test = split_data(
                features, labels, 
                test_size=config.data.test_size,
                random_state=config.data.random_state
            )
            
            # Train model based on configuration
            if model_type == "mlp":
                metrics, model = train_mlp(
                    X_train, y_train, X_test, y_test,
                    hidden_sizes=config.mlp.hidden_sizes,
                    dropout=config.mlp.dropout,
                    lr=config.mlp.lr,
                    epochs=config.mlp.epochs,
                    batch_size=config.mlp.batch_size,
                    patience=config.mlp.patience,
                    random_state=config.mlp.random_state
                )
                params = config.mlp.dict()
            else:  # random_forest
                metrics, model = train_random_forest(
                    X_train, y_train, X_test, y_test,
                    n_estimators=config.random_forest.n_estimators,
                    max_depth=config.random_forest.max_depth,
                    random_state=config.random_forest.random_state
                )
                params = config.random_forest.dict()
            
            # Comprehensive evaluation
            if enable_monitoring:
                eval_results = comprehensive_evaluation(
                    model, X_train, y_train, X_test, y_test, 
                    output_path, model_type
                )
                logger.info("Comprehensive evaluation completed")
            
            # Track experiment with MLflow
            if config.mlflow.enable_tracking:
                run_id = track_experiment(
                    model_type, features, labels, metrics, model, params, output_path
                )
                typer.echo(f"MLflow run ID: {run_id}")
        
        # Success output
        typer.echo("✓ Training completed successfully!", fg=typer.colors.GREEN)
        typer.echo(f"Model: {model_type}")
        typer.echo(f"Test AUROC: {metrics['test_auroc']:.4f}")
        typer.echo(f"Test AP: {metrics['test_average_precision']:.4f}")
        typer.echo(f"Overfitting gap: {metrics['auroc_gap']:.4f}")
        typer.echo(f"Results saved to: {output_path}")
        
    except OncotargetError as e:
        typer.echo(f"Pipeline error: {e}", err=True, fg=typer.colors.RED)
        if hasattr(e, 'details') and e.details:
            typer.echo(f"Details: {e.details}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command()
def evaluate(
    model_type: Annotated[str, typer.Option(help="Model type")] = "random_forest",
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "./reports",
    random_state: Annotated[int, typer.Option(help="Random seed")] = 42
) -> None:
    """Run comprehensive evaluation with bootstrap CIs and calibration."""
    
    typer.echo("Running comprehensive evaluation...")
    
    set_random_seed(random_state)
    output_path = Path(output_dir)
    ensure_dir(output_path)
    
    # Load and split data
    features, labels = load_data()
    X_train, X_test, y_train, y_test = split_data(features, labels, random_state=random_state)
    
    # Train model (in real scenario, would load from MLflow)
    if model_type == "mlp":
        _, model = train_mlp(X_train, y_train, X_test, y_test, random_state=random_state)
    else:
        _, model = train_random_forest(X_train, y_train, X_test, y_test, random_state=random_state)
    
    # Run evaluation
    results = comprehensive_evaluation(
        model, X_train, y_train, X_test, y_test, output_path, model_type
    )
    
    typer.echo("Evaluation complete!")
    typer.echo(f"Test AUROC: {results['metrics']['test_auroc']:.3f} "
               f"(95% CI: {results['bootstrap']['auroc']['ci_lower']:.3f}-"
               f"{results['bootstrap']['auroc']['ci_upper']:.3f})")
    typer.echo(f"Brier Score: {results['metrics']['brier_score']:.3f}")
    typer.echo(f"ECE: {results['metrics']['ece']:.3f}")


@app.command()
def explain(
    model_type: Annotated[str, typer.Option(help="Model type")] = "random_forest",
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "./reports",
    random_state: Annotated[int, typer.Option(help="Random seed")] = 42
) -> None:
    """Generate SHAP explanations."""
    
    typer.echo("Generating SHAP explanations...")
    
    set_random_seed(random_state)
    output_path = Path(output_dir)
    ensure_dir(output_path)
    
    # Load and split data
    features, labels = load_data()
    X_train, X_test, y_train, y_test = split_data(features, labels, random_state=random_state)
    
    # Train model (in real scenario, would load from MLflow)
    if model_type == "mlp":
        _, model = train_mlp(X_train, y_train, X_test, y_test, random_state=random_state)
    else:
        _, model = train_random_forest(X_train, y_train, X_test, y_test, random_state=random_state)
    
    # Generate explanations
    shap_results = generate_shap_explanations(
        model, X_train, X_test, output_path, model_type, random_state
    )
    
    # Log to MLflow if in a run context
    try:
        if mlflow.active_run():
            mlflow.log_artifacts(str(output_path / "shap"), "shap")
    except Exception:
        pass
    
    typer.echo("SHAP explanations generated!")
    typer.echo(f"Global summary: {shap_results['shap_summary_path']}")
    typer.echo(f"Example explanations: {len(shap_results['example_paths'])} files")


@app.command()
def scorecard(
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "./reports",
    random_state: Annotated[int, typer.Option(help="Random seed")] = 42
) -> None:
    """Generate target scorecard HTML."""
    
    typer.echo("Generating target scorecard...")
    
    output_path = Path(output_dir)
    ensure_dir(output_path)
    
    # Generate scorecard
    scorecard_path = generate_target_scorecard(output_path)
    
    typer.echo(f"Target scorecard generated: {scorecard_path}")


@app.command()
def snapshot(
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "./reports",
    port: Annotated[int, typer.Option(help="Streamlit port")] = 8501
) -> None:
    """Capture Streamlit demo snapshot."""
    
    typer.echo("Capturing Streamlit demo snapshot...")
    
    output_path = Path(output_dir)
    ensure_dir(output_path)
    
    try:
        from .reporting import capture_streamlit_snapshot
        snapshot_path = capture_streamlit_snapshot(output_path, port)
        typer.echo(f"Streamlit snapshot captured: {snapshot_path}")
    except ImportError:
        typer.echo("Playwright not available. Creating placeholder image...")
        # Create a placeholder image
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'Streamlit Demo\n(Snapshot not available)', 
                ha='center', va='center', fontsize=20, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(output_path / "streamlit_demo.png", dpi=120, bbox_inches='tight')
        plt.close()
        typer.echo(f"Placeholder image created: {output_path / 'streamlit_demo.png'}")


@app.command()
def all(
    model_type: Annotated[str, typer.Option(help="Model type")] = "random_forest",
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "./reports",
    random_state: Annotated[int, typer.Option(help="Random seed")] = 42
) -> None:
    """Run the complete pipeline: train -> eval -> explain -> scorecard -> snapshot."""
    
    typer.echo("Running complete pipeline...")
    
    # Set MLflow tracking
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(mlflow_uri)
    
    with mlflow.start_run() as run:
        typer.echo(f"Started MLflow run: {run.info.run_id}")
        
        # Train
        train(model_type=model_type, output_dir=output_dir, random_state=random_state)
        
        # Evaluate  
        evaluate(model_type=model_type, output_dir=output_dir, random_state=random_state)
        
        # Explain
        explain(model_type=model_type, output_dir=output_dir, random_state=random_state)
        
        # Scorecard
        scorecard(output_dir=output_dir, random_state=random_state)
        
        # Snapshot
        snapshot(output_dir=output_dir)
    
    typer.echo("Complete pipeline finished!")


if __name__ == "__main__":
    app()