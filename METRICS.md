# Checkpoint Metrics Documentation

This document describes the metrics collection features added to `checkpoint.py` for monitoring performance and cache efficiency.

## Overview

Two types of metrics are now collected:

1. **Tensor Loading Metrics** - Track time spent loading tensors in parallel
2. **Regex Cache Metrics** - Monitor cache hit/miss rates for regex pattern compilation

## Tensor Loading Metrics

### What is Tracked

When `load_tensors()` is called, the following metrics are automatically collected and logged:

- **Total loading time** - Wall-clock time for the entire operation
- **Per-tensor timing** - Individual load time for each tensor
- **Average time per tensor** - Mean loading time across all tensors
- **Min/Max times** - Fastest and slowest tensor loads

### Example Output

```
INFO - Loading tensors with max_workers=32, timeout=None
INFO - Tensor loading metrics: 1024 tensors loaded in 45.23s (avg: 0.044s/tensor, min: 0.012s, max: 0.156s)
```

### Use Cases

- **Performance monitoring** - Track loading performance across different hardware
- **Bottleneck identification** - Identify slow tensor loads via min/max times
- **Capacity planning** - Understand scaling characteristics with different worker counts

## Regex Cache Metrics

### What is Tracked

The `_compile_regex()` function uses `functools.lru_cache` to cache compiled regex patterns. Metrics track:

- **Total calls** - Number of times regex compilation was requested
- **Cache hits** - Number of times a cached pattern was reused
- **Cache misses** - Number of times a new pattern was compiled
- **Hit rate** - Percentage of cache hits (hits / total calls × 100)

### Example Output

```
INFO - Regex cache (processed 2048 paths) stats: 4096 total calls, 3584 hits, 512 misses, 87.5% hit rate
```

### How It Works

1. `_compile_regex(pattern)` - LRU-cached function that compiles patterns
2. `_compile_regex_with_metrics(pattern)` - Wrapper that tracks cache hits/misses
3. `RegexCacheMetrics` class - Stores and reports statistics

The wrapper checks the cache info before and after calling the cached function:
- If `hits` increased → cache hit
- If `misses` increased → cache miss

### Use Cases

- **Cache effectiveness** - Verify that regex caching provides benefits
- **Pattern analysis** - Understand which patterns are reused most
- **Performance validation** - Confirm optimization impact

## API Reference

### RegexCacheMetrics Class

```python
class RegexCacheMetrics:
    """Tracks regex cache hit/miss statistics for performance monitoring."""
    
    def record_call(self, is_hit: bool):
        """Record a cache access (hit or miss)."""
    
    def get_stats(self) -> dict:
        """Get current cache statistics.
        
        Returns:
            dict with keys: total_calls, cache_hits, cache_misses, hit_rate_percent
        """
    
    def reset(self):
        """Reset all counters to zero."""
    
    def log_stats(self, logger_instance=None, prefix="Regex cache"):
        """Log cache statistics."""
```

### Global Instance

```python
_regex_metrics = RegexCacheMetrics()
```

Access this instance to query metrics programmatically:

```python
from checkpoint import _regex_metrics

stats = _regex_metrics.get_stats()
print(f"Cache hit rate: {stats['hit_rate_percent']:.1f}%")
```

## Integration Points

### load_tensors()

Metrics are automatically logged at INFO level when tensor loading completes:

```python
loaded_tensors = load_tensors(
    shaped_arrays,
    directory,
    mesh_config,
    max_workers=32,  # Optional: control parallelism
    timeout=None,    # Optional: set timeout per tensor
)
# Metrics logged automatically
```

### replace_with_load_state()

Regex cache metrics are reset at the start and logged at the end:

```python
result = replace_with_load_state(
    init_state,
    load_state,
    load_rename_rules=[...],
    load_exclude_rules=[...],
)
# Regex cache stats logged automatically
```

## Performance Expectations

### Tensor Loading

- **Parallel speedup**: Near-linear scaling up to I/O bandwidth limits
- **Typical times**: 0.01-0.1s per tensor depending on size and storage
- **Bottlenecks**: Network I/O, disk speed, pickle deserialization

### Regex Caching

- **Expected hit rate**: 80-95% for typical checkpoint loading
- **Cache size**: 256 patterns (configurable via `maxsize` parameter)
- **Performance gain**: 10-100x faster for cache hits vs. recompilation

High hit rates (>80%) indicate effective caching. Lower rates may suggest:
- Highly diverse pattern sets
- Cache size too small (increase `maxsize`)
- Patterns not being reused

## Testing

Run the standalone test to verify metrics functionality:

```bash
python3 test_metrics_standalone.py
```

This demonstrates:
- Cache hit/miss tracking accuracy
- Timing metrics collection
- Performance comparison with/without caching

## Future Enhancements

Potential additions:
- Histogram of tensor load times
- Per-pattern cache statistics
- Metrics export to monitoring systems (Prometheus, etc.)
- Configurable metrics verbosity levels
