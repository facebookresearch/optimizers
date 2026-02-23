# GPAAdamW (Generalized Primal Averaging)

[![arXiv](https://img.shields.io/badge/arXiv-2512.17131-b31b1b.svg)](https://arxiv.org/abs/2512.17131)

GPAAdamW is an AdamW optimizer enhanced with Generalized Primal Averaging, which incorporates two generalizations of Nesterov momentum (Schedule-Free and GPA) in its primal averaging formulation. This optimizer can converge faster by leveraging advanced averaging techniques, achieving comparable or better model quality in fewer iterations. GPA unifies and generalizes recent averaging-based optimizers like single-worker DiLoCo and Schedule-Free, eliminating DiLoCo's memory-intensive two-loop structure while enabling smooth iterate averaging at every step with reduced memory overhead.

This implementation can be used directly as a drop-in replacement in distributed training frameworks such as DDP and FSDP, requiring no additional code changes.

## Contributors

- Aaron Defazio
- Konstantin Mishchenko
- Parameswaran Raman
- Hao-Jun Michael Shi
- Lin Xiao

## Key Features
- Schedule-Free mode (no learning rate scheduler needed, though warmup and schedulers can still help for LLMs)
- GPA mode for advanced primal averaging
- Memory-efficient implementation using in-place operations
- Seamless train/eval mode switching for proper evaluation
- Compatible with standard PyTorch checkpointing

## Background

The GPA optimizer maintains three parameter sequences:
- **z-sequence**: The primary sequence where gradient updates are applied
- **y-sequence**: The sequence where gradients are computed (used during training)
- **x-sequence**: The sequence where model evaluation should be performed (used during inference)

The update equations are:
```
y^{(t)} = μ_y · x^{(t)} + (1 - μ_y) · z^{(t)}
g^{(t)} = ∇f(y^{(t)})
m^{(t)} = β₁ · m^{(t-1)} + (1 - β₁) · g^{(t)}
v^{(t)} = β₂ · v^{(t-1)} + (1 - β₂) · (g^{(t)})²
z^{(t+1)} = z^{(t)} - α^{(t)} · m^{(t)} / (√v^{(t)} + ε)
x^{(t+1)} = μ_x · x^{(t)} + (1 - μ_x) · z^{(t+1)}
```

Where:
- `μ_y` is `train_interp_coeff` (weight for x in y-update)
- `μ_x` is `eval_interp_coeff` (weight for x in x-update)
- `α^{(t)}` is the learning rate with bias correction
- `β₁` and `β₂` are the inner optimizer (AdamW)'s hyperparameters
- `g^{(t)}` is the gradient coming from the inner optimizer

## Requirements

- PyTorch >= 2.0
- Python >= 3.10
- CUDA 11.x or 12.x (for GPU training)

## Installation

The GPAAdamW optimizer is available in the `gpa` package.

```python
from gpa.gpa_adamw import GPAAdamW
```

## Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
from gpa.gpa_adamw import GPAAdamW

# Create model
model = nn.Linear(10, 2)

# Create optimizer (GPA mode by default)
optimizer = GPAAdamW(
    model.parameters(),
    lr=0.001,
    train_interp_coeff=0.7,
    eval_interp_coeff=0.9967,  # GPA mode (equivalent to DiLoCo with 32 local steps)
)

# Training loop
for epoch in range(num_epochs):

    # IMPORTANT: Switch to train mode before training
    model.train()
    optimizer.train()

    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output)
        loss.backward()
        optimizer.step()

    # Switch to eval mode before evaluation (optional if y and x sequences are similar)
    model.eval()
    optimizer.eval()

    with torch.no_grad():
        for batch in val_loader:
            output = model(batch)

            # ... evaluate

# Save checkpoint in eval mode (optional if y and x sequences are similar)
optimizer.eval()
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint.pt')
```

### Critical Requirements

1. **Call `optimizer.train()` before training and `optimizer.eval()` before evaluation.** This switches the model parameters between the y-sequence (for gradient computation) and x-sequence (for evaluation). In practice, the training sequence (y) often performs well for evaluation too, so this switching is optional if y and x are similar.

2. **Saving checkpoints in eval mode is recommended but optional.** The x-sequence represents the averaged parameters. However, in practice the training sequence (y) often works well too. If the y and x sequences are not significantly different, saving in train mode is acceptable.

3. **Learning rate scheduling follows standard practices.** For Schedule-Free mode (`iterate_averaging_type=IterateAveragingType.SCHEDULE_FREE`), the original paper recommends no scheduler, though in practice warmup and schedulers can still help, especially for LLMs.

## Hyperparameter Guide

### Core Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|--------|
| `lr` | Learning rate. Must be >= 0 | 1e-3 |
| `train_interp_coeff` | μ_y: Weight for x in y-update. Higher = y closer to x. Must be in (0, 1] | 0.7 |
| `eval_interp_coeff` | μ_x: Weight for x in x-update (GPA mode only). Must be in [0, 1] | 0.9967 |
| `iterate_averaging_type` | Averaging mode: `IterateAveragingType.GPA` or `IterateAveragingType.SCHEDULE_FREE` | `GPA` |
| `beta1` | EMA coefficient for gradient (like Adam). Must be in [0, 1) | 0.9 |
| `beta2` | EMA coefficient for squared gradient (like Adam). Must be in [0, 1) | 0.999 |
| `eps` | Numerical stability term. Must be >= 0 | 1e-8 |
| `weight_decay` | L2 regularization applied to y-sequence. Must be >= 0 | 0.0 |
| `weight_pow_coeff` | Polynomial weighting power, r in the paper (Schedule-Free mode only). Must be >= 0 | 0.0 |
| `weight_lr_power` | Learning rate weighting power during warmup (Schedule-Free mode only). Must be >= 0 | 2.0 |

### Schedule-Free Mode vs GPA Mode

The `iterate_averaging_type` parameter controls which averaging mode is used:

#### Schedule-Free Mode (`IterateAveragingType.SCHEDULE_FREE`)

In this mode, the optimizer uses polynomial weighting for the x-update, eliminating the need for a learning rate scheduler. The `weight_pow_coeff` and `weight_lr_power` parameters control the weighting behavior.

```python
from gpa.gpa_types import IterateAveragingType

# Schedule-Free mode
optimizer = GPAAdamW(
    model.parameters(),
    lr=0.001,
    train_interp_coeff=0.9,
    iterate_averaging_type=IterateAveragingType.SCHEDULE_FREE,
)
```

**When to use Schedule-Free mode:**
- When you want to eliminate learning rate scheduling complexity
- For standard training scenarios
- When warmup is sufficient (no decay needed)

#### GPA Mode (`IterateAveragingType.GPA`)

In this mode (the default), the optimizer uses a fixed interpolation coefficient (`eval_interp_coeff` / μ_x) for the x-update.

```python

# GPA mode (default)
optimizer = GPAAdamW(
    model.parameters(),
    lr=0.001,
    train_interp_coeff=0.9,
    eval_interp_coeff=0.9967,  # GPA mode (equivalent to DiLoCo with 32 local steps)
    iterate_averaging_type=IterateAveragingType.GPA,  # This is the default
)
```

**Note:** This example uses `eval_interp_coeff=0.9967`, which is equivalent to `num_local_steps=32` in DiLoCo. In practice, this works well for many workloads, but depending on the model and task, some tuning may be needed. We have found that GPA `eval_interp_coeff` values corresponding to `num_local_steps` of 16, 32, or 64 (i.e., 0.9934, 0.9967, 0.9984) all work well and are good starting points before more aggressive tuning.

**When to use GPA mode:**
- When you want more control over the averaging behavior
- For research and experimentation
- When polynomial weighting is not desired

### Recommended Configurations

#### Configuration 1: GPA Mode (Default)

Best for most training scenarios. Equivalent to DiLoCo with 32 local steps.

```python
optimizer = GPAAdamW(
    model.parameters(),
    lr=0.001,
    train_interp_coeff=0.7,
    eval_interp_coeff=0.9967,  # GPA mode
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01,
)
```

#### Configuration 2: Schedule-Free

For scheduler-free training with polynomial weighting.

```python
from gpa.gpa_types import IterateAveragingType

optimizer = GPAAdamW(
    model.parameters(),
    lr=0.001,
    train_interp_coeff=0.9,
    iterate_averaging_type=IterateAveragingType.SCHEDULE_FREE,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01,
)
```

### Hyperparameter Tuning Tips

1. **Learning Rate (`lr`):**
   - Start with standard AdamW learning rates (1e-3 to 1e-4)
   - In Schedule-Free mode, you may need slightly higher learning rates
   - Use warmup if training is unstable

2. **train_interp_coeff (μ_y):**
   - Default 0.7 works well for most cases
   - Higher values (0.8-0.99): y is closer to x, more stable but slower
   - Lower values (0.5-0.7): y is closer to z, more aggressive updates

3. **eval_interp_coeff (μ_x):**
   - Default 0.9967 (recommended for GPA mode)
   - Only used in GPA mode (`iterate_averaging_type=IterateAveragingType.GPA`)
   - Use values > 0 for GPA mode (e.g., 0.9967 for DiLoCo-equivalent of 32 local steps)
   - GPA's `eval_interp_coeff` acts as a smooth, continuous hyperparameter similar to `num_local_steps` in DiLoCo. This allows GPA to play the role of smoothing DiLoCo-like methods.
   - The following table shows the approximate mapping between `eval_interp_coeff` and DiLoCo's `num_local_steps` (from Table 4 in the [GPA paper](https://arxiv.org/abs/2512.17131)):

   | DiLoCo num_local_steps | eval_interp_coeff (μ_x) |
   |------------------------|-------------------------|
   | 1                      | 0.9                     |
   | 4                      | 0.974                   |
   | 8                      | 0.9869                  |
   | 16                     | 0.9934                  |
   | 32                     | 0.9967                  |
   | 64                     | 0.9984                  |
   | 128                    | 0.9972                  |

4. **Weight Decay:**
   - Applied to the y-sequence (training sequence)
   - Standard values (0.01-0.1) work well
   - Reduce if training becomes unstable

5. **beta1 and beta2:**
   - Use same values as you would for AdamW
   - Default (0.9, 0.999) works for most cases
   - For transformers, consider (0.9, 0.95) or (0.9, 0.99)
   - Tune further if necessary

## Distributed Training

GPAAdamW works seamlessly with PyTorch's distributed training APIs including DDP and FSDP. No special configuration is required—simply wrap your model with DDP or FSDP and create the optimizer as usual.

## Checkpointing

### Saving Checkpoints

Save checkpoints in eval mode (optional if y and x sequences are similar):

```python

# Switch to eval mode before saving (optional if y and x sequences are similar)
model.eval()
optimizer.eval()

checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, 'checkpoint.pt')
```

### Loading Checkpoints

```python

# Create model and optimizer
model = create_model()
optimizer = GPAAdamW(model.parameters(), lr=0.001)

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']

# Resume training
model.train()
optimizer.train()

# ... continue training
```

## Running Tests

The test suite is organized into unit tests (`tests/`) and GPU tests (`gpu_tests/`).

### Test Organization

| Test File | Description | # Tests |
|-----------|-------------|---------|
| `tests/gpa_adamw_test.py` | Core GPAAdamW unit tests (initialization, step/mode, avg_coeff, state_dict) | 17 |
| `tests/gpa_equivalence_test.py` | Base optimizer equivalence tests | 1 |
| `gpu_tests/gpa_adamw_numerics_test.py` | GPU convergence and numerical tests (CPU + CUDA parameterized) | 12 |

### Run Tests

```bash
# Core GPAAdamW unit tests
python -m unittest gpa.tests.gpa_adamw_test -v

# Equivalence tests
python -m unittest gpa.tests.gpa_equivalence_test -v

# GPU tests (requires CUDA)
python -m unittest gpa.gpu_tests.gpa_adamw_numerics_test -v

# Run all tests in the tests/ directory
python -m unittest discover -s gpa.tests -v
```

## Common Issues and Troubleshooting

### RuntimeError: Optimizer was not in train mode

**Cause:** Calling `optimizer.step()` without first calling `optimizer.train()`.

**Solution:** Always call `optimizer.train()` before the training loop.

```python

optimizer.train()  # Add this before training

for batch in train_loader:
    optimizer.zero_grad()
    # ...
    optimizer.step()
```

### Incorrect evaluation results

**Cause:** Evaluating the model without switching to eval mode.

**Solution:** Call `optimizer.eval()` before evaluation (optional if y and x sequences are similar).

```python

model.eval()
optimizer.eval()  # Add this before evaluation (optional if y ≈ x)

with torch.no_grad():
    for batch in val_loader:
        output = model(batch)
        # ...
```

### Checkpoint incompatibility

**Cause:** Checkpoint saved in train mode contains y-sequence instead of x-sequence.

**Solution:** Save checkpoints after calling `optimizer.eval()` (optional if y and x sequences are similar).

### Training instability

**Possible solutions:**
1. Reduce learning rate
2. Add warmup
3. Reduce `train_interp_coeff` (e.g., from 0.9 to 0.8)
4. Increase `beta2` (e.g., from 0.999 to 0.9999)

## Advanced Options

### Custom Weighting in Schedule-Free Mode

In Schedule-Free mode (`iterate_averaging_type=IterateAveragingType.SCHEDULE_FREE`), the averaging coefficient is computed as:

```
weight = (k ^ weight_pow_coeff) * (lr_max ^ weight_lr_power)
avg_coeff = weight / weight_sum
```

Where `k` is the step number.

By default, `weight_pow_coeff=0.0` which means all steps have equal weight (constant weighting). You can customize this behavior:

```python

# Custom weighting: later steps get more weight (linear in step count)
optimizer = GPAAdamW(
    model.parameters(),
    lr=0.001,
    train_interp_coeff=0.9,
    iterate_averaging_type=IterateAveragingType.SCHEDULE_FREE,
    weight_pow_coeff=1.0,    # k^1 = linear weighting
    weight_lr_power=2.0,     # Weight by lr^2
    beta1=0.9,
    beta2=0.999,
)
```

| `weight_pow_coeff` | Behavior |
|--------------------|----------|
| 0.0 (default) | Constant weighting - all steps weighted equally |
| 1.0 | Linear weighting - later steps get proportionally more weight |
| 2.0 | Quadratic weighting - even more emphasis on recent steps |

This is an advanced option for research purposes. The default (`weight_pow_coeff=0.0`) is recommended for most use cases.

## Citing GPA-AdamW

If you use GPA-AdamW in your work, please use the following BibTeX entry.

```BibTeX
@misc{defazio2026smoothingdilocoprimalaveraging,
      title={Smoothing DiLoCo with Primal Averaging for Faster Training of LLMs},
      author={Aaron Defazio and Konstantin Mishchenko and Parameswaran Raman and Hao-Jun Michael Shi and Lin Xiao},
      year={2026},
      eprint={2512.17131},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.17131},
}
```

## References

1. [Smoothing DiLoCo with Primal Averaging for Faster Training of LLMs](https://arxiv.org/abs/2512.17131). Aaron Defazio, Konstantin Mishchenko, Parameswaran Raman, Hao-Jun Michael Shi, Lin Xiao. Tech report, 2025.
2. [The Road Less Scheduled](https://arxiv.org/abs/2405.15682). Aaron Defazio, Xingyu Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky. NeurIPS, 2024.
