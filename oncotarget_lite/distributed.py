"""
Distributed computing utilities for scaling ML workloads.

This module provides distributed computing capabilities for:
- Bootstrap computations
- SHAP value calculations
- Ablation studies
- Hyperparameter optimization
"""

from __future__ import annotations

import os
from functools import partial, wraps
from typing import Any, Callable, Iterable, List, Optional, TypeVar, Union

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .utils import set_seeds

T = TypeVar('T')

# Global cache for expensive computations
_COMPUTATION_CACHE: Dict[str, Any] = {}
_CACHE_ENABLED = True

# Global configuration for distributed computing
DISTRIBUTED_CONFIG = {
    'backend': 'joblib',  # 'joblib', 'dask', 'ray'
    'n_jobs': -1,  # -1 means use all available cores
    'verbose': 0,
    'prefer': 'processes',  # 'processes', 'threads'
    'memory_limit': None,  # Memory limit for dask workers
    'ray_address': None,  # Ray cluster address
}


def configure_distributed(
    backend: str = 'joblib',
    n_jobs: int = -1,
    verbose: int = 0,
    prefer: str = 'processes',
    memory_limit: Optional[str] = None,
    ray_address: Optional[str] = None,
) -> None:
    """
    Configure distributed computing settings.

    Args:
        backend: Computing backend ('joblib', 'dask', 'ray')
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Verbosity level for parallel operations
        prefer: Process/thread preference for joblib
        memory_limit: Memory limit for dask workers
        ray_address: Ray cluster address
    """
    DISTRIBUTED_CONFIG.update({
        'backend': backend,
        'n_jobs': n_jobs,
        'verbose': verbose,
        'prefer': prefer,
        'memory_limit': memory_limit,
        'ray_address': ray_address,
    })


def get_n_jobs() -> int:
    """Get the configured number of parallel jobs."""
    return DISTRIBUTED_CONFIG['n_jobs']


def is_distributed_enabled() -> bool:
    """Check if distributed computing is enabled."""
    return DISTRIBUTED_CONFIG['backend'] != 'sequential'


def parallel_map(
    func: Callable[..., T],
    items: Iterable[Any],
    **kwargs
) -> List[T]:
    """
    Apply function to items in parallel.

    Args:
        func: Function to apply
        items: Items to process
        **kwargs: Additional arguments for parallel execution

    Returns:
        List of results in the same order as input items
    """
    if not is_distributed_enabled():
        return [func(item) for item in items]

    backend = DISTRIBUTED_CONFIG['backend']
    n_jobs = kwargs.get('n_jobs', DISTRIBUTED_CONFIG['n_jobs'])
    verbose = kwargs.get('verbose', DISTRIBUTED_CONFIG['verbose'])

    if backend == 'joblib':
        prefer = kwargs.get('prefer', DISTRIBUTED_CONFIG['prefer'])
        return Parallel(
            n_jobs=n_jobs,
            verbose=verbose,
            prefer=prefer
        )(delayed(func)(item) for item in items)

    elif backend == 'dask':
        try:
            import dask
            from dask import delayed as dask_delayed
            from dask.distributed import Client, LocalCluster

            # Start local cluster if no address provided
            if not DISTRIBUTED_CONFIG['ray_address']:
                cluster = LocalCluster(
                    n_workers=n_jobs if n_jobs > 0 else os.cpu_count(),
                    threads_per_worker=1,
                    memory_limit=DISTRIBUTED_CONFIG['memory_limit'],
                    processes=False
                )
                client = Client(cluster)
            else:
                client = Client(DISTRIBUTED_CONFIG['ray_address'])

            # Create delayed tasks
            tasks = [dask_delayed(func)(item) for item in items]
            results = dask.compute(*tasks)

            if not DISTRIBUTED_CONFIG['ray_address']:
                client.close()
                cluster.close()

            return list(results)

        except ImportError:
            print("Dask not available, falling back to joblib")
            return Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(func)(item) for item in items
            )

    elif backend == 'ray':
        try:
            import ray

            if not ray.is_initialized():
                if DISTRIBUTED_CONFIG['ray_address']:
                    ray.init(address=DISTRIBUTED_CONFIG['ray_address'])
                else:
                    ray.init(num_cpus=n_jobs if n_jobs > 0 else os.cpu_count())

            @ray.remote
            def ray_func(item):
                return func(item)

            # Submit tasks
            futures = [ray_func.remote(item) for item in items]
            results = ray.get(futures)

            if not DISTRIBUTED_CONFIG['ray_address']:
                ray.shutdown()

            return results

        except ImportError:
            print("Ray not available, falling back to joblib")
            return Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(func)(item) for item in items
            )

    else:
        # Fallback to sequential
        return [func(item) for item in items]


def bootstrap_parallel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    n_jobs: Optional[int] = None,
) -> dict:
    """
    Compute bootstrap confidence intervals in parallel.

    This is a distributed version of the bootstrap computation used in eval.py.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric_fn: Metric function to compute
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level
        seed: Random seed
        n_jobs: Number of parallel jobs (uses config default if None)

    Returns:
        Dictionary with mean, lower, and upper confidence interval bounds
    """
    set_seeds(seed)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def bootstrap_sample(i):
        """Compute one bootstrap sample."""
        indices = rng.integers(0, n, size=n)
        sample_true = y_true[indices]
        sample_prob = y_prob[indices]
        try:
            return metric_fn(sample_true, sample_prob)
        except ValueError:
            return np.nan

    # Split bootstrap samples across workers
    sample_indices = np.array_split(np.arange(n_bootstrap), get_n_jobs())

    def compute_bootstrap_chunk(chunk_indices):
        """Compute bootstrap samples for a chunk."""
        return [bootstrap_sample(i) for i in chunk_indices]

    # Run bootstrap in parallel
    chunk_results = parallel_map(
        compute_bootstrap_chunk,
        sample_indices,
        n_jobs=n_jobs,
        verbose=0
    )

    # Combine results
    scores = []
    for chunk in chunk_results:
        scores.extend([s for s in chunk if not np.isnan(s)])

    if not scores:
        # Fallback to single computation
        value = metric_fn(y_true, y_prob)
        return {"mean": value, "lower": value, "upper": value}

    scores_arr = np.array(scores)
    value = float(scores_arr.mean())
    alpha = (1 - ci) / 2
    lower = float(np.quantile(scores_arr, alpha))
    upper = float(np.quantile(scores_arr, 1 - alpha))

    return {"mean": value, "lower": lower, "upper": upper}


def shap_parallel(
    explainer,
    data: Union[pd.DataFrame, np.ndarray],
    max_evals: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """
    Compute SHAP values in parallel for multiple samples.

    Args:
        explainer: SHAP explainer object
        data: Data to explain
        max_evals: Maximum evaluations (for sampling)
        n_jobs: Number of parallel jobs

    Returns:
        SHAP values array
    """
    if max_evals and len(data) > max_evals:
        # Sample data for SHAP computation
        indices = np.random.choice(len(data), max_evals, replace=False)
        data = data.iloc[indices] if hasattr(data, 'iloc') else data[indices]

    def compute_shap_values(idx):
        """Compute SHAP values for a single sample."""
        try:
            sample = data.iloc[[idx]] if hasattr(data, 'iloc') else data[[idx]]
            shap_values = explainer.shap_values(sample)
            return shap_values
        except Exception as e:
            print(f"Warning: SHAP computation failed for sample {idx}: {e}")
            return None

    # Compute SHAP values in parallel
    n_samples = len(data)
    sample_indices = list(range(n_samples))

    results = parallel_map(
        compute_shap_values,
        sample_indices,
        n_jobs=n_jobs,
        verbose=0
    )

    # Filter out failed computations and combine results
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        raise RuntimeError("All SHAP computations failed")

    # Combine results - this assumes SHAP returns arrays of consistent shape
    shap_values = np.stack(valid_results)

    return shap_values


def ablation_parallel(
    ablation_configs: List[Path],
    run_experiment_fn: Callable,
    processed_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    n_jobs: Optional[int] = None,
) -> List[dict]:
    """
    Run ablation experiments in parallel.

    Args:
        ablation_configs: List of ablation configuration file paths
        run_experiment_fn: Function to run a single experiment
        processed_dir: Processed data directory
        models_dir: Models output directory
        reports_dir: Reports output directory
        n_jobs: Number of parallel jobs

    Returns:
        List of experiment results
    """
    def run_single_experiment(config_path: Path) -> dict:
        """Run a single ablation experiment."""
        try:
            return run_experiment_fn(config_path, processed_dir, models_dir, reports_dir)
        except Exception as e:
            print(f"Warning: Ablation experiment {config_path.name} failed: {e}")
            return {"experiment": config_path.stem, "error": str(e)}

    # Run ablation experiments in parallel
    results = parallel_map(
        run_single_experiment,
        ablation_configs,
        n_jobs=n_jobs,
        verbose=1  # Show progress for long-running ablation studies
    )

    return results


class DistributedContext:
    """Context manager for distributed computing configuration."""

    def __init__(self, **config):
        self.config = config
        self.old_config = DISTRIBUTED_CONFIG.copy()

    def __enter__(self):
        configure_distributed(**self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        DISTRIBUTED_CONFIG.clear()
        DISTRIBUTED_CONFIG.update(self.old_config)


def enable_cache(enabled: bool = True) -> None:
    """Enable or disable computation caching."""
    global _CACHE_ENABLED
    _CACHE_ENABLED = enabled


def clear_cache() -> None:
    """Clear the computation cache."""
    global _COMPUTATION_CACHE
    _COMPUTATION_CACHE.clear()


def cached_computation(key: str, compute_func: Callable[[], T], force_recompute: bool = False) -> T:
    """
    Cache expensive computations with optional cache key.

    Args:
        key: Cache key for the computation
        compute_func: Function that performs the computation
        force_recompute: Force recomputation even if cached

    Returns:
        Cached or computed result
    """
    if not _CACHE_ENABLED or force_recompute or key not in _COMPUTATION_CACHE:
        _COMPUTATION_CACHE[key] = compute_func()

    return _COMPUTATION_CACHE[key]


def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    return {
        "cache_size": len(_COMPUTATION_CACHE),
        "cache_enabled": _CACHE_ENABLED,
    }
