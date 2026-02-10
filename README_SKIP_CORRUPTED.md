# Skip Corrupted Tensors - Quick Start Guide

## What's New?

The checkpoint loading pipeline can now gracefully skip corrupted tensor files instead of failing completely. Corrupted tensors are replaced with zero tensors, allowing the model to load with partial data.

## Quick Usage

### Option 1: Environment Variable (Easiest)
```bash
export CHECKPOINT_SKIP_CORRUPTED=true
python run.py
```

### Option 2: Code Parameter (Recommended)
```python
from runners import ModelRunner, InferenceRunner

runner = ModelRunner(
    model=grok_1_model,
    bs_per_device=0.125,
    checkpoint_path="./checkpoints/",
    skip_corrupted=True,  # ← Add this line
)
```

### Option 3: Direct Function Call (Advanced)
```python
import checkpoint

state = checkpoint.restore(
    checkpoint_path="./checkpoints/",
    state_shapes=state_shapes,
    mesh=mesh,
    between_hosts_config=(1, 1),
    state_sharding=state_sharding,
    params_only=True,
    skip_corrupted=True,  # ← Add this parameter
)
```

## When to Use

✅ **Use `skip_corrupted=True` when:**
- Working with partially downloaded checkpoints
- Recovering from corrupted checkpoint files
- Testing with experimental or incomplete checkpoints
- Debugging checkpoint loading issues

❌ **Don't use `skip_corrupted=True` when:**
- Running production inference (data integrity is critical)
- Training models (corrupted weights will affect training)
- Benchmarking (results won't be accurate)

## What Happens?

### With `skip_corrupted=True`:
1. ✅ Corrupted files are logged as errors
2. ✅ Zero tensors replace corrupted data
3. ✅ Loading continues for valid tensors
4. ✅ Summary shows which tensors were skipped
5. ⚠️ Model may have degraded performance

### With `skip_corrupted=False` (default):
1. ❌ First corrupted file stops loading
2. ❌ RuntimeError is raised
3. ❌ No partial checkpoint is loaded
4. ✅ Data integrity is guaranteed

## Example Output

```
INFO:checkpoint:Loading tensors with max_workers=32, timeout=None, skip_corrupted=True

ERROR:checkpoint:Failed to load tensor 42 from '/checkpoints/ckpt-0/tensor00042_001': EOFError

WARNING:checkpoint:Skipped corrupted tensor 42, using zero tensor as fallback

WARNING:checkpoint:Successfully loaded checkpoint with 1 corrupted tensor(s) skipped.
Skipped tensor indices: [42]. These tensors were replaced with zero tensors.

INFO:checkpoint:Tensor loading metrics: 100 tensors loaded in 45.23s
```

## Important Notes

⚠️ **Limitations:**
- Corrupted tensors become zeros (may affect model accuracy)
- No partial recovery of tensor data
- Model behavior may be unpredictable with many skipped tensors

💡 **Best Practices:**
- Always check logs for skipped tensors
- Validate model outputs after loading
- Fix corrupted checkpoints rather than relying on skipping
- Keep backup checkpoints for production

## Documentation

For more details, see:
- **SKIP_CORRUPTED_FEATURE.md** - Complete feature documentation
- **IMPLEMENTATION_NOTES.md** - Technical implementation details
- **CHANGES_SUMMARY.md** - Summary of all changes

## Need Help?

1. Check the logs for detailed error messages
2. Review the documentation files listed above
3. Verify checkpoint file integrity
4. Try with `skip_corrupted=False` to see exact errors

## Default Behavior

**By default, `skip_corrupted=False`** - the original behavior is preserved for backward compatibility. You must explicitly enable this feature.
