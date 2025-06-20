# Actor Debug Simple

This repository contains debugging utilities for isolating and testing OpenRLHF actor model behavior without distributed training dependencies.

## What is replay_inputs.py?

`replay_inputs.py` replays previously captured actor model inputs to test model behavior in isolation. It's designed to help debug issues that occur during OpenRLHF training by reproducing the exact same forward passes outside of the distributed training environment.

### How it Works

1. **Loads Saved Inputs**: Reads pickled input data from the `input_dumps/` directory
2. **Initializes Actor**: Creates a standalone actor model instance using `init_actor_no_ds.py`
3. **Replays Forward Pass**: Feeds the saved inputs through the actor model
4. **Captures Results**: Records execution time, success/failure, and any errors

### Command Line Options

- `--flash-attn`: Enable Flash Attention 2 (disabled by default)
- `--compile`: Enable PyTorch torch.compile (disabled by default)
- `--packing`: Enable sequence packing (disabled by default)

### Usage Examples

```bash
# Most stable default - no optimizations enabled
python replay_inputs.py

# Enable Flash Attention for performance
python replay_inputs.py --flash-attn

# Enable PyTorch compilation
python replay_inputs.py --compile

# Enable sequence packing
python replay_inputs.py --packing

# Combine multiple features
python replay_inputs.py --flash-attn --packing
python replay_inputs.py --compile --packing
```

### Test Results

Based on comprehensive testing of all flag combinations:

| Configuration | Flash Attention | Compile | Packing | Result |
|---------------|-----------------|---------|---------|--------|
| Default | | | | ✅ Success |
| `--flash-attn` | ✅ | | | ✅ Success |
| `--compile` | | ✅ | | ✅ Success |
| `--packing` | | | ✅ | ✅ Success |
| `--flash-attn --compile` | ✅ | ✅ | | ❌ **FAIL** |
| `--flash-attn --packing` | ✅ | | ✅ | ✅ Success |
| `--compile --packing` | | ✅ | ✅ | ✅ Success |
| `--flash-attn --compile --packing` | ✅ | ✅ | ✅ | ✅ Success |


### Dependencies

- PyTorch with CUDA support
- Transformers library
- OpenRLHF installation in `../OpenRLHF/`
- Flash Attention 2 (optional, can be disabled)

