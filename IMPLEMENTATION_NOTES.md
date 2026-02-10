# Skip Corrupted Tensors - Implementation Notes

## Overview
Modified the loading pipeline to gracefully skip corrupted tensor files while continuing to load the remaining valid ones.

## Changes Made

### 1. Configuration Options (`checkpoint.py`)

Added three new configuration helpers:

- **`DEFAULT_SKIP_CORRUPTED`**: Default behavior (False - fail on corruption)
- **`_get_skip_corrupted(skip_corrupted)`**: Helper function to get the skip_corrupted setting
  - Priority: explicit parameter > environment variable > default constant
  - Environment variable: `CHECKPOINT_SKIP_CORRUPTED` (accepts: true/1/yes/on)

### 2. Modified `load_tensors()` Function (`checkpoint.py`)

**New Parameter:**
- `skip_corrupted: Optional[bool] = None` - Whether to skip corrupted files

**New Behavior:**
- When `skip_corrupted=True`:
  - Catches both `TimeoutError` and general `Exception` during tensor loading
  - Logs detailed error information (tensor index, file path, error type)
  - Creates a zero tensor as fallback with the correct shape and dtype
  - Continues loading remaining tensors
  - Logs a summary of all skipped tensors at the end
  
- When `skip_corrupted=False` (default):
  - Original behavior - raises `RuntimeError` on any failure
  - Provides detailed error information about failed tensors

**Error Tracking:**
- `errors` list: Collects failures when skip_corrupted=False
- `skipped_corrupted` list: Tracks skipped tensors when skip_corrupted=True

### 3. Modified `restore()` Function (`checkpoint.py`)

**New Parameter:**
- `skip_corrupted: Optional[bool] = None` - Passed through to `load_tensors()`

**Updated Call:**
```python
loaded_tensors = load_tensors(
    ckpt_shapes_flat,
    ckpt_path,
    between_hosts_config,
    max_workers=max_workers,
    timeout=timeout,
    skip_corrupted=skip_corrupted,  # <-- New parameter
)
```

### 4. Modified `ModelRunner` Class (`runners.py`)

**New Field:**
- `skip_corrupted: bool = False` - Whether to skip corrupted tensor files

**Updated restore() call:**
```python
state = xai_checkpoint.restore(
    checkpoint_path=self.checkpoint_path,
    state_shapes=state_shapes,
    mesh=self.mesh,
    between_hosts_config=self.between_hosts_config,
    state_sharding=self.state_sharding,
    init_state=init_state,
    params_only=True,
    skip_corrupted=self.skip_corrupted,  # <-- New parameter
)
```

## Usage Examples

### Method 1: Environment Variable
```bash
export CHECKPOINT_SKIP_CORRUPTED=true
python run.py
```

### Method 2: ModelRunner Parameter
```python
from model import LanguageModelConfig, TransformerConfig
from runners import ModelRunner, InferenceRunner

runner = ModelRunner(
    model=grok_1_model,
    bs_per_device=0.125,
    checkpoint_path=CKPT_PATH,
    skip_corrupted=True,  # <-- Enable graceful skipping
)
```

### Method 3: Direct restore() Call
```python
import checkpoint

state = checkpoint.restore(
    checkpoint_path=path,
    state_shapes=shapes,
    mesh=mesh,
    between_hosts_config=config,
    state_sharding=sharding,
    params_only=True,
    skip_corrupted=True,  # <-- Enable graceful skipping
)
```

## Logging Output

### When Corrupted Tensors Are Skipped

```
ERROR:checkpoint:Failed to load tensor 42 from '/path/to/tensor00042_001': EOFError: Ran out of input
<traceback>

WARNING:checkpoint:Skipped corrupted tensor 42 from '/path/to/tensor00042_001' (EOFError: Ran out of input), using zero tensor as fallback

WARNING:checkpoint:Successfully loaded checkpoint with 1 corrupted tensor(s) skipped. Skipped tensor indices: [42]. These tensors were replaced with zero tensors.

INFO:checkpoint:Tensor loading metrics: 100 tensors loaded in 45.23s (avg: 0.452s/tensor, min: 0.123s, max: 2.345s)
```

### When Corrupted Tensors Cause Failure (Default)

```
ERROR:checkpoint:Failed to load tensor 42 from '/path/to/tensor00042_001': EOFError: Ran out of input
<traceback>

ERROR:checkpoint:Tensor loading failed for 1 out of 100 tensors. Failed tensor indices: [42]

RuntimeError: Failed to load 1 tensor(s). First failure: tensor 42 from '/path/to/tensor00042_001': Ran out of input
```

## Backward Compatibility

All changes are backward compatible:
- `skip_corrupted` parameter is optional with default value `None`
- Default behavior (`skip_corrupted=False`) matches original behavior
- Existing code continues to work without modification
- New functionality is opt-in via parameter or environment variable

## Testing

The implementation has been verified to:
1. ✅ Compile without syntax errors
2. ✅ Maintain backward compatibility
3. ✅ Support three configuration methods (parameter, env var, default)
4. ✅ Provide comprehensive logging for both success and failure cases
5. ✅ Use appropriate fallback (zero tensors) for corrupted files

## Error Handling

The implementation handles:
- **TimeoutError**: When tensor loading exceeds the configured timeout
- **General Exceptions**: Any other error during unpickling (EOFError, pickle.UnpicklingError, etc.)
- **Shape Preservation**: Fallback zero tensors match the expected shape and dtype
- **Index Tracking**: Correctly maps future indices to tensor indices for fallback creation
