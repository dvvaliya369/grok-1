# Copyright 2024 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import contextlib
import functools
import logging
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import jax
import numpy as np
from jax.experimental import multihost_utils

from model import QuantizedWeight8bit

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")

# Needed for loading the checkpoint with pickle.
sys.modules['__main__'].QuantizedWeight8bit = QuantizedWeight8bit


class RegexCacheMetrics:
    """Tracks regex cache hit/miss statistics for performance monitoring."""
    
    def __init__(self):
        self.total_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def record_call(self, is_hit: bool):
        """Record a cache access (hit or miss)."""
        self.total_calls += 1
        if is_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_stats(self) -> dict:
        """Get current cache statistics."""
        hit_rate = (self.cache_hits / self.total_calls * 100) if self.total_calls > 0 else 0.0
        return {
            'total_calls': self.total_calls,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate,
        }
    
    def reset(self):
        """Reset all counters to zero."""
        self.total_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def log_stats(self, logger_instance=None, prefix="Regex cache"):
        """Log cache statistics."""
        stats = self.get_stats()
        log = logger_instance or logger
        log.info(
            "%s stats: %d total calls, %d hits, %d misses, %.1f%% hit rate",
            prefix,
            stats['total_calls'],
            stats['cache_hits'],
            stats['cache_misses'],
            stats['hit_rate_percent'],
        )


# Global instance for tracking regex cache metrics
_regex_metrics = RegexCacheMetrics()


# Configuration for ThreadPoolExecutor resource management
DEFAULT_MAX_WORKERS = 32
DEFAULT_LOAD_TIMEOUT = None  # No timeout by default (in seconds)


def _get_max_workers(max_workers: Optional[int] = None) -> int:
    """Get the maximum number of worker threads for parallel tensor loading.
    
    Priority: explicit parameter > environment variable > default constant.
    """
    if max_workers is not None:
        return max_workers
    env_workers = os.environ.get("CHECKPOINT_MAX_WORKERS")
    if env_workers:
        try:
            return int(env_workers)
        except ValueError:
            logger.warning(
                "Invalid CHECKPOINT_MAX_WORKERS='%s', using default %d",
                env_workers,
                DEFAULT_MAX_WORKERS,
            )
    return DEFAULT_MAX_WORKERS


def _get_timeout(timeout: Optional[float] = None) -> Optional[float]:
    """Get the timeout for individual tensor loading operations.
    
    Priority: explicit parameter > environment variable > default constant.
    Returns None for no timeout.
    """
    if timeout is not None:
        return timeout
    env_timeout = os.environ.get("CHECKPOINT_LOAD_TIMEOUT")
    if env_timeout:
        try:
            return float(env_timeout)
        except ValueError:
            logger.warning(
                "Invalid CHECKPOINT_LOAD_TIMEOUT='%s', using default %s",
                env_timeout,
                DEFAULT_LOAD_TIMEOUT,
            )
    return DEFAULT_LOAD_TIMEOUT


@contextlib.contextmanager
def copy_to_shm(file: str):
    if file.startswith("/dev/shm/"):
        # Nothing to do, the file is already in shared memory.
        yield file
        return

    tmp_dir = "/dev/shm/"
    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir)
    try:
        shutil.copyfile(file, tmp_path)
        yield tmp_path
    finally:
        os.remove(tmp_path)
        os.close(fd)


@contextlib.contextmanager
def copy_from_shm(file: str):
    tmp_dir = "/dev/shm/"
    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir)
    try:
        yield tmp_path
        shutil.copyfile(tmp_path, file)
    finally:
        os.remove(tmp_path)
        os.close(fd)


def fast_unpickle(path: str) -> Any:
    with copy_to_shm(path) as tmp_path:
        with open(tmp_path, "rb") as f:
            return pickle.load(f)


def fast_pickle(obj: Any, path: str) -> None:
    with copy_from_shm(path) as tmp_path:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f)


def load_tensors(
    shaped_arrays,
    directory,
    mesh_config,
    tensor_indices=None,
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
):
    """Loads a set of arrays with configurable parallelism and timeout.
    
    Args:
        shaped_arrays: Array shapes to load.
        directory: Directory containing tensor files.
        mesh_config: Mesh configuration for sharding.
        tensor_indices: Optional tensor indices to load.
        max_workers: Maximum number of worker threads. If None, uses
            CHECKPOINT_MAX_WORKERS env var or DEFAULT_MAX_WORKERS.
        timeout: Timeout in seconds for each tensor load. If None, uses
            CHECKPOINT_LOAD_TIMEOUT env var or no timeout.
    
    Returns:
        List of loaded tensors.
    
    Raises:
        RuntimeError: If any tensor fails to load.
    """
    actual_max_workers = _get_max_workers(max_workers)
    actual_timeout = _get_timeout(timeout)
    
    load_start_time = time.time()
    
    logger.info(
        "Loading tensors with max_workers=%d, timeout=%s",
        actual_max_workers,
        actual_timeout if actual_timeout is not None else "None",
    )
    
    fs = []
    tensor_metadata = []
    tensor_times = []
    num_tensors = 0
    num_replicas = 1
    data_model_shards = math.prod(mesh_config)
    if tensor_indices is None:
        iterator = enumerate(shaped_arrays)
    else:
        iterator = zip(tensor_indices, shaped_arrays)

    with ThreadPoolExecutor(max_workers=actual_max_workers) as pool:
        for i, t in iterator:
            if (i % num_replicas) == (
                (jax.process_index() // data_model_shards) % num_replicas
            ):
                idx = (
                    jax.process_index() // (num_replicas * data_model_shards) * data_model_shards
                    + jax.process_index() % data_model_shards
                )
                file_path = os.path.join(directory, f"tensor{i:05d}_{idx:03d}")
                fs.append(pool.submit(fast_unpickle, file_path))
                tensor_metadata.append((i, file_path))
                num_tensors += 1
            else:
                fs.append(pool.submit(np.zeros, t.shape, dtype=t.dtype))
                tensor_metadata.append((i, None))

        results = []
        errors = []
        for future_idx, future in enumerate(fs):
            tensor_idx, file_path = tensor_metadata[future_idx]
            tensor_start = time.time()
            try:
                results.append(future.result(timeout=actual_timeout))
                tensor_elapsed = time.time() - tensor_start
                tensor_times.append(tensor_elapsed)
            except TimeoutError as e:
                tb = traceback.format_exc()
                if file_path is not None:
                    logger.error(
                        "Timeout loading tensor %d from '%s' after %s seconds\n%s",
                        tensor_idx,
                        file_path,
                        actual_timeout,
                        tb,
                    )
                else:
                    logger.error(
                        "Timeout creating zero tensor %d after %s seconds\n%s",
                        tensor_idx,
                        actual_timeout,
                        tb,
                    )
                errors.append((tensor_idx, file_path, e))
            except Exception as e:
                tb = traceback.format_exc()
                if file_path is not None:
                    logger.error(
                        "Failed to load tensor %d from '%s': %s: %s\n%s",
                        tensor_idx,
                        file_path,
                        type(e).__name__,
                        e,
                        tb,
                    )
                else:
                    logger.error(
                        "Failed to create zero tensor %d: %s: %s\n%s",
                        tensor_idx,
                        type(e).__name__,
                        e,
                        tb,
                    )
                errors.append((tensor_idx, file_path, e))

        if errors:
            failed_indices = [idx for idx, _, _ in errors]
            logger.error(
                "Tensor loading failed for %d out of %d tensors. "
                "Failed tensor indices: %s",
                len(errors),
                len(fs),
                failed_indices,
            )
            raise RuntimeError(
                f"Failed to load {len(errors)} tensor(s). "
                f"First failure: tensor {errors[0][0]} "
                f"from '{errors[0][1]}': {errors[0][2]}"
            )
    
    # Log timing metrics
    total_load_time = time.time() - load_start_time
    if tensor_times:
        avg_time = sum(tensor_times) / len(tensor_times)
        min_time = min(tensor_times)
        max_time = max(tensor_times)
        logger.info(
            "Tensor loading metrics: %d tensors loaded in %.2fs "
            "(avg: %.3fs/tensor, min: %.3fs, max: %.3fs)",
            len(results),
            total_load_time,
            avg_time,
            min_time,
            max_time,
        )
    else:
        logger.info(
            "Tensor loading metrics: %d tensors loaded in %.2fs",
            len(results),
            total_load_time,
        )

    return results


def path_tuple_to_string(path: tuple) -> str:
    pieces = []
    for elem in path:
        if isinstance(elem, jax.tree_util.DictKey):
            pieces.append(elem.key)
        elif isinstance(elem, jax.tree_util.GetAttrKey):
            pieces.append(elem.name)
        else:
            assert isinstance(elem, (jax.tree_util.FlattenedIndexKey, jax.tree_util.SequenceKey))
    return "/".join(pieces)


@functools.lru_cache(maxsize=256)
def _compile_regex(pattern: str) -> re.Pattern:
    """Compile and cache a regex pattern to avoid redundant recompilation."""
    return re.compile(pattern)


def _compile_regex_with_metrics(pattern: str) -> re.Pattern:
    """Compile regex with cache hit/miss tracking for performance monitoring."""
    cache_info_before = _compile_regex.cache_info()
    result = _compile_regex(pattern)
    cache_info_after = _compile_regex.cache_info()
    
    # A cache hit occurred if the hit count increased
    is_hit = cache_info_after.hits > cache_info_before.hits
    _regex_metrics.record_call(is_hit)
    
    return result


def get_load_path_str(
    init_path_str: str,
    load_rename_rules: Optional[list[tuple[str, str]]] = None,
    load_exclude_rules: Optional[list[str]] = None,
) -> Optional[str]:
    # Exclusion
    if load_exclude_rules is not None:
        for search_pattern in load_exclude_rules:
            if _compile_regex_with_metrics(search_pattern).search(init_path_str):
                return None

    # Renaming
    load_path_str = init_path_str
    if load_rename_rules is not None:
        for search_pattern, replacement_pattern in load_rename_rules:
            compiled = _compile_regex_with_metrics(search_pattern)
            if compiled.search(load_path_str):
                load_path_str = compiled.sub(replacement_pattern, load_path_str)
                break

    return load_path_str


def replace_with_load_state(
    init_state: Any,
    load_state: Any,
    load_rename_rules: Optional[list[tuple[str, str]]] = None,
    load_exclude_rules: Optional[list[str]] = None,
    mesh_config: tuple = (1, 1),
) -> Any:
    # Reset regex cache metrics before processing
    _regex_metrics.reset()
    
    flatten_load, _ = jax.tree_util.tree_flatten_with_path(load_state)
    flatten_init, structure_init = jax.tree_util.tree_flatten_with_path(init_state)
    load_map = {path_tuple_to_string(path): tensor for path, tensor in flatten_load}

    replaced = []
    num_replicas = 1
    data_model_shards = math.prod(mesh_config)
    num_paths_processed = 0
    
    for i, (init_path, tensor) in enumerate(flatten_init):
        init_path_str = path_tuple_to_string(init_path)
        load_path_str = get_load_path_str(init_path_str, load_rename_rules, load_exclude_rules)
        num_paths_processed += 1
        
        if load_path_str is None:
            rank_logger.info(f"Excluded from restore: {init_path_str}.")
            replaced.append(tensor)
        elif load_path_str in load_map:
            if load_path_str == init_path_str:
                rank_logger.info(f"Restored from ckpt: {init_path_str}.")
            else:
                rank_logger.info(f"Restored from ckpt: {init_path_str} <-- {load_path_str}.")
            replaced.append(load_map[load_path_str])
        else:
            rank_logger.info(f"Not found in ckpt: {init_path_str}.")
            if (i % num_replicas) == ((jax.process_index() // data_model_shards) % num_replicas):
                replaced.append(tensor)
            else:
                replaced.append(np.zeros_like(tensor))
    
    # Log regex cache metrics after processing all paths
    _regex_metrics.log_stats(
        logger_instance=rank_logger,
        prefix=f"Regex cache (processed {num_paths_processed} paths)",
    )

    return jax.tree_util.tree_unflatten(structure_init, replaced)


def restore(
    checkpoint_path: str,
    state_shapes: Any,
    mesh,
    between_hosts_config,
    params_only,
    state_sharding,
    init_state: Optional[Any] = None,
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Restore model state from checkpoint with configurable resource limits.
    
    Args:
        checkpoint_path: Path to checkpoint directory.
        state_shapes: Expected state shapes.
        mesh: JAX mesh for distributed computation.
        between_hosts_config: Configuration for multi-host setup.
        params_only: Whether to return only parameters.
        state_sharding: Sharding specification for state.
        init_state: Optional initial state for validation.
        max_workers: Maximum worker threads for parallel loading.
        timeout: Timeout in seconds for each tensor load operation.
    
    Returns:
        Restored model state.
    """
    ckpt_path = os.path.join(checkpoint_path, "ckpt-0")

    rank_logger.info("Loading checkpoint at {}".format(ckpt_path))
    ckpt_shapes = state_shapes
    ckpt_shapes_with_path, structure = jax.tree_util.tree_flatten_with_path(ckpt_shapes)

    ckpt_shapes_flat = [elem[1] for elem in ckpt_shapes_with_path]
    loaded_tensors = load_tensors(
        ckpt_shapes_flat,
        ckpt_path,
        between_hosts_config,
        max_workers=max_workers,
        timeout=timeout,
    )

    state = jax.tree_util.tree_unflatten(structure, loaded_tensors)

    # Sanity check to give a better error message.
    ckpt_keys = set(state.params.keys())
    code_keys = set(state_sharding.params.keys())

    if ckpt_keys != code_keys and init_state is None:
        missing_in_ckpt = code_keys - ckpt_keys
        missing_locally = ckpt_keys - code_keys
        raise ValueError(
            "Parameters in the code are not matching checkpoint parameters.\n"
            "Params missing in checkpoint: {}\nParams missing in code: {}".format(
                missing_in_ckpt, missing_locally
            )
        )
    state_sharding = jax.tree_util.tree_map(
        lambda x: jax.sharding.PartitionSpec() if x is None else x,
        state_sharding,
        is_leaf=lambda x: x is None,
    )
    state = multihost_utils.host_local_array_to_global_array(state, mesh, state_sharding)
    if params_only:
        state = state.params
    return state
