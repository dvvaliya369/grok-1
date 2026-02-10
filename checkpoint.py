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


def load_tensors(shaped_arrays, directory, mesh_config, tensor_indices=None):
    """Loads a set of arrays."""
    fs = []
    tensor_metadata = []
    num_tensors = 0
    num_replicas = 1
    data_model_shards = math.prod(mesh_config)
    if tensor_indices is None:
        iterator = enumerate(shaped_arrays)
    else:
        iterator = zip(tensor_indices, shaped_arrays)

    with ThreadPoolExecutor(max_workers=32) as pool:
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
            try:
                results.append(future.result())
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


def get_load_path_str(
    init_path_str: str,
    load_rename_rules: Optional[list[tuple[str, str]]] = None,
    load_exclude_rules: Optional[list[str]] = None,
) -> Optional[str]:
    # Exclusion
    if load_exclude_rules is not None:
        for search_pattern in load_exclude_rules:
            if _compile_regex(search_pattern).search(init_path_str):
                return None

    # Renaming
    load_path_str = init_path_str
    if load_rename_rules is not None:
        for search_pattern, replacement_pattern in load_rename_rules:
            compiled = _compile_regex(search_pattern)
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
    flatten_load, _ = jax.tree_util.tree_flatten_with_path(load_state)
    flatten_init, structure_init = jax.tree_util.tree_flatten_with_path(init_state)
    load_map = {path_tuple_to_string(path): tensor for path, tensor in flatten_load}

    replaced = []
    num_replicas = 1
    data_model_shards = math.prod(mesh_config)
    for i, (init_path, tensor) in enumerate(flatten_init):
        init_path_str = path_tuple_to_string(init_path)
        load_path_str = get_load_path_str(init_path_str, load_rename_rules, load_exclude_rules)
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

    return jax.tree_util.tree_unflatten(structure_init, replaced)


def restore(
    checkpoint_path: str,
    state_shapes: Any,
    mesh,
    between_hosts_config,
    params_only,
    state_sharding,
    init_state: Optional[Any] = None,
) -> Any:
    ckpt_path = os.path.join(checkpoint_path, "ckpt-0")

    rank_logger.info("Loading checkpoint at {}".format(ckpt_path))
    ckpt_shapes = state_shapes
    ckpt_shapes_with_path, structure = jax.tree_util.tree_flatten_with_path(ckpt_shapes)

    ckpt_shapes_flat = [elem[1] for elem in ckpt_shapes_with_path]
    loaded_tensors = load_tensors(ckpt_shapes_flat, ckpt_path, between_hosts_config)

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
