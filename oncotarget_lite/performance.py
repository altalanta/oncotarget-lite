"""Performance monitoring and profiling utilities.

Provides tools for monitoring memory usage, execution time, and system resources
during model training and evaluation.
"""

import functools
import gc
import logging
import psutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch


class ResourceMonitor:
    """Monitor system resource usage during execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.reset()
    
    def reset(self) -> None:
        """Reset monitoring counters."""
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.measurements = []
    
    def measure(self) -> Dict[str, float]:
        """Take a measurement of current resource usage."""
        current_time = time.perf_counter()
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        measurement = {
            'elapsed_time': current_time - self.start_time,
            'memory_mb': memory_mb,
            'memory_delta': memory_mb - self.start_memory,
            'cpu_percent': self.process.cpu_percent()
        }
        
        self.peak_memory = max(self.peak_memory, memory_mb)
        self.measurements.append(measurement)
        
        return measurement
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for the monitoring period."""
        if not self.measurements:
            return {}
        
        return {
            'total_time': self.measurements[-1]['elapsed_time'],
            'peak_memory_mb': self.peak_memory,
            'memory_delta_mb': self.peak_memory - self.start_memory,
            'avg_cpu_percent': sum(m['cpu_percent'] for m in self.measurements) / len(self.measurements)
        }


@contextmanager
def monitor_resources(name: str = "operation", log_interval: float = 30.0):
    """Context manager for resource monitoring with periodic logging.
    
    Args:
        name: Name of the operation being monitored
        log_interval: Interval in seconds for periodic logging
        
    Example:
        with monitor_resources("model training") as monitor:
            # training code here
            pass
        summary = monitor.get_summary()
    """
    logger = logging.getLogger(__name__)
    monitor = ResourceMonitor()
    
    logger.info(f"Starting resource monitoring for {name}")
    monitor.reset()
    
    last_log_time = time.perf_counter()
    
    try:
        yield monitor
    finally:
        summary = monitor.get_summary()
        logger.info(f"Completed {name} - {summary}")


def profile_memory(func: Callable) -> Callable:
    """Decorator for memory profiling of functions.
    
    Logs memory usage before, during, and after function execution.
    Includes PyTorch GPU memory if available.
    
    Example:
        @profile_memory
        def train_model():
            # training code
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        # Initial memory measurement
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        
        gpu_memory_start = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_memory_start = torch.cuda.memory_allocated() / 1024 / 1024
        
        logger.info(f"Starting {func.__name__} - RAM: {start_memory:.1f} MB" + 
                   (f", GPU: {gpu_memory_start:.1f} MB" if gpu_memory_start else ""))
        
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
        finally:
            # Final memory measurement
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            gpu_memory_end = None
            if torch.cuda.is_available():
                gpu_memory_end = torch.cuda.memory_allocated() / 1024 / 1024
            
            logger.info(
                f"Completed {func.__name__} in {end_time - start_time:.2f}s - "
                f"RAM: {end_memory:.1f} MB (Δ{end_memory - start_memory:+.1f})" +
                (f", GPU: {gpu_memory_end:.1f} MB (Δ{gpu_memory_end - gpu_memory_start:+.1f})" 
                 if gpu_memory_end else "")
            )
        
        return result
    return wrapper


class PerformanceProfiler:
    """Detailed performance profiler for ML pipelines."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir
        self.metrics = {}
        self.start_times = {}
        
    def start_timer(self, name: str) -> None:
        """Start timing an operation."""
        self.start_times[name] = time.perf_counter()
        
    def end_timer(self, name: str) -> float:
        """End timing an operation and return duration."""
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")
        
        duration = time.perf_counter() - self.start_times[name]
        
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)
        
        del self.start_times[name]
        return duration
    
    @contextmanager
    def time_operation(self, name: str):
        """Context manager for timing operations."""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get summary metrics for all timed operations."""
        summary = {}
        
        for name, times in self.metrics.items():
            summary[name] = {
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'count': len(times)
            }
        
        return summary
    
    def save_profile(self, filepath: Optional[Path] = None) -> None:
        """Save profiling results to JSON file."""
        if filepath is None and self.output_dir:
            filepath = self.output_dir / "performance_profile.json"
        
        if filepath:
            import json
            with open(filepath, 'w') as f:
                json.dump(self.get_metrics(), f, indent=2)


def optimize_memory() -> Dict[str, float]:
    """Optimize memory usage by running garbage collection and clearing caches.
    
    Returns:
        Dictionary with memory statistics before and after optimization
    """
    process = psutil.Process()
    
    # Before optimization
    memory_before = process.memory_info().rss / 1024 / 1024
    gpu_memory_before = None
    
    if torch.cuda.is_available():
        gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024
    
    # Optimization steps
    gc.collect()  # Python garbage collection
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear PyTorch GPU cache
        torch.cuda.synchronize()  # Ensure all operations complete
    
    # After optimization
    memory_after = process.memory_info().rss / 1024 / 1024
    gpu_memory_after = None
    
    if torch.cuda.is_available():
        gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024
    
    stats = {
        'ram_before_mb': memory_before,
        'ram_after_mb': memory_after,
        'ram_freed_mb': memory_before - memory_after
    }
    
    if gpu_memory_before is not None:
        stats.update({
            'gpu_before_mb': gpu_memory_before,
            'gpu_after_mb': gpu_memory_after,
            'gpu_freed_mb': gpu_memory_before - gpu_memory_after
        })
    
    return stats


def log_system_info(logger: logging.Logger = None) -> None:
    """Log comprehensive system information for reproducibility."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # System information
    logger.info(f"System: {psutil.os.name} {psutil.os.path.basename(psutil.os.path.dirname(psutil.os.__file__))}")
    logger.info(f"CPU: {psutil.cpu_count()} cores, {psutil.cpu_count(logical=False)} physical")
    
    # Memory information
    memory = psutil.virtual_memory()
    logger.info(f"RAM: {memory.total / 1024**3:.1f} GB total, {memory.available / 1024**3:.1f} GB available")
    
    # GPU information
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("CUDA: Not available")
    
    # PyTorch version
    logger.info(f"PyTorch: {torch.__version__}")