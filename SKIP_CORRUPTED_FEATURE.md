# Skip Corrupted Tensors Feature

## Summary

The loading pipeline has been modified to gracefully skip corrupted tensor files while continuing to load the remaining valid ones. This feature is opt-in and maintains full backward compatibility.

## Key Features

✅ **Graceful Error Handling**: Corrupted tensors are replaced with zero tensors instead of failing the entire load
✅ **Comprehensive Logging**: Detailed warnings for each skipped tensor with error information
✅ **Multiple Configuration Methods**: Environment variable, function parameter, or class attribute
✅ **Backward Compatible**: Default behavior unchanged, existing code works without modification
✅ **Zero Tensor Fallback**: Corrupted tensors are replaced with zeros matching the expected shape and dtype

## Modified Files

1. **checkpoint.py**
   - Added `DEFAULT_SKIP_CORRUPTED` constant
   - Added `_get_skip_corrupted()` helper function
   - Modified `load_tensors()` to accept `skip_corrupted` parameter
   - Modified `restore()` to accept and pass through `skip_corrupted` parameter
   - Enhanced error handling to skip corrupted files when enabled

2. **runners.py**
   - Added `skip_corrupted` field to `ModelRunner` class
   - Updated `restore()` call to pass `skip_corrupted` parameter

## Configuration Methods

### 1. Environment Variable (Global)
```bash
export CHECKPOINT_SKIP_CORRUPTED=true
python run.py
```

Accepted values: `true`, `1`, `yes`, `on` (case-insensitive)

### 2. ModelRunner Class (Recommended)
```python
from runners import ModelRunner, InferenceRunner

runner = ModelRunner(
    model=grok_1_model,
    bs_per_device=0.125,
    checkpoint_path="./checkpoints/",
    skip_corrupted=True,  # Enable graceful skipping
)

inference_runner = InferenceRunner(
    runner=runner,
    # ... other parameters
)
```

### 3. Direct Function Call (Advanced)
```python
import checkpoint

state = checkpoint.restore(
    checkpoint_path="./checkpoints/",
    state_shapes=state_shapes,
    mesh=mesh,
    between_hosts_config=(1, 1),
    state_sharding=state_sharding,
    params_only=True,
    skip_corrupted=True,  # Enable graceful skipping
)
```

## Behavior

### When `skip_corrupted=True`

1. **Error Detection**: Catches `TimeoutError` and general exceptions during tensor loading
2. **Logging**: Logs detailed error information (ERROR level) for each corrupted file
3. **Fallback**: Creates a zero tensor with matching shape and dtype
4. **Warning**: Logs a warning for each skipped tensor
5. **Summary**: Logs a final summary of all skipped tensors
6. **Continuation**: Continues loading remaining valid tensors

**Example Log Output:**
```
INFO:checkpoint:Loading tensors with max_workers=32, timeout=None, skip_corrupted=True

ERROR:checkpoint:Failed to load tensor 42 from '/checkpoints/ckpt-0/tensor00042_001': EOFError: Ran out of input
[traceback details]

WARNING:checkpoint:Skipped corrupted tensor 42 from '/checkpoints/ckpt-0/tensor00042_001' (EOFError: Ran out of input), using zero tensor as fallback

WARNING:checkpoint:Successfully loaded checkpoint with 1 corrupted tensor(s) skipped. Skipped tensor indices: [42]. These tensors were replaced with zero tensors.

INFO:checkpoint:Tensor loading metrics: 100 tensors loaded in 45.23s (avg: 0.452s/tensor, min: 0.123s, max: 2.345s)
```

### When `skip_corrupted=False` (Default)

1. **Error Detection**: Catches errors during tensor loading
2. **Logging**: Logs detailed error information (ERROR level)
3. **Collection**: Collects all errors
4. **Failure**: Raises `RuntimeError` with details about failed tensors
5. **Termination**: Stops the loading process

**Example Log Output:**
```
INFO:checkpoint:Loading tensors with max_workers=32, timeout=None, skip_corrupted=False

ERROR:checkpoint:Failed to load tensor 42 from '/checkpoints/ckpt-0/tensor00042_001': EOFError: Ran out of input
[traceback details]

ERROR:checkpoint:Tensor loading failed for 1 out of 100 tensors. Failed tensor indices: [42]

RuntimeError: Failed to load 1 tensor(s). First failure: tensor 42 from '/checkpoints/ckpt-0/tensor00042_001': Ran out of input
```

## Use Cases

### 1. Development and Testing
When working with partially downloaded or experimental checkpoints:
```python
runner = ModelRunner(
    model=model_config,
    checkpoint_path="./partial_checkpoint/",
    skip_corrupted=True,  # Continue despite corrupted files
)
```

### 2. Checkpoint Recovery
When attempting to recover from a corrupted checkpoint:
```bash
export CHECKPOINT_SKIP_CORRUPTED=true
python run.py  # Will load valid tensors and skip corrupted ones
```

### 3. Production (Default)
In production, keep the default behavior to ensure data integrity:
```python
runner = ModelRunner(
    model=model_config,
    checkpoint_path="./production_checkpoint/",
    # skip_corrupted defaults to False
)
```

## Error Types Handled

The implementation handles various corruption scenarios:

- **EOFError**: Truncated or incomplete pickle files
- **pickle.UnpicklingError**: Corrupted pickle data
- **TimeoutError**: Files that take too long to load
- **OSError/IOError**: File system errors
- **Any Exception**: General catch-all for unexpected errors

## Technical Details

### Zero Tensor Creation
When a corrupted tensor is skipped, a zero tensor is created with:
- **Shape**: Matches the expected shape from `shaped_arrays`
- **Dtype**: Matches the expected dtype from `shaped_arrays`
- **Implementation**: `np.zeros(shape, dtype=dtype)`

### Index Mapping
The implementation correctly handles the mapping between:
- `future_idx`: Index in the futures list
- `tensor_idx`: Actual tensor index (may differ when `tensor_indices` is provided)
- `shape_idx`: Index in the `shaped_arrays` list

### Thread Safety
The implementation maintains thread safety:
- Errors are collected in thread-safe lists
- Each future is processed sequentially in the main thread
- No race conditions in error handling

## Testing

The implementation has been verified:

1. ✅ **Syntax Check**: Code compiles without errors
2. ✅ **Configuration**: All three configuration methods work correctly
3. ✅ **Backward Compatibility**: Existing code works without modification
4. ✅ **Error Handling**: Both skip and fail modes work as expected
5. ✅ **Logging**: Comprehensive logging at appropriate levels

## Migration Guide

### For Existing Code
No changes required! The default behavior is unchanged.

### To Enable Skipping
Add one line to your existing code:

**Before:**
```python
runner = ModelRunner(
    model=grok_1_model,
    bs_per_device=0.125,
    checkpoint_path=CKPT_PATH,
)
```

**After:**
```python
runner = ModelRunner(
    model=grok_1_model,
    bs_per_device=0.125,
    checkpoint_path=CKPT_PATH,
    skip_corrupted=True,  # <-- Add this line
)
```

## Limitations

1. **Zero Tensors**: Skipped tensors are replaced with zeros, which may affect model behavior
2. **No Partial Recovery**: Individual tensor values cannot be partially recovered
3. **Memory Usage**: Zero tensors still consume memory
4. **Model Accuracy**: Models loaded with skipped tensors may have degraded performance

## Recommendations

1. **Use Sparingly**: Only enable in development or recovery scenarios
2. **Monitor Logs**: Always check logs for skipped tensors
3. **Validate Results**: Test model outputs when using skip_corrupted=True
4. **Fix Corruption**: Address the root cause of corruption rather than relying on skipping
5. **Backup Checkpoints**: Maintain backup checkpoints for production use

## Future Enhancements

Potential improvements for future versions:

- [ ] Custom fallback strategies (e.g., random initialization, previous checkpoint values)
- [ ] Partial tensor recovery (load valid portions of corrupted files)
- [ ] Corruption detection before loading (file integrity checks)
- [ ] Automatic checkpoint repair tools
- [ ] Metrics tracking for corruption rates
