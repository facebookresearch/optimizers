# Variance Metrics for Stochastic Optimization

`variance_metrics` is a lightweight, PyTorch-native Llama3 pretraining trainer instrumented to measure per-step gradient noise and to drive the global batch size adaptively from it. It logs gradient noise scale (GNS) and L1 / L2 / spectral (nuclear-norm) variance metrics across every transformer block, and can grow the batch size during training so that more data is processed once the gradient signal-to-noise ratio justifies it. All of this works across the standard parallelism layouts — Distributed Data Parallel (DDP), Fully Sharded Data Parallel (FSDP), Hybrid Sharded Data Parallel (HSDP), and Tensor Parallel (TP).

This is the companion code for:

[Naganuma, H., Gupta, S., Briki, Y., Mitliagkas, I., Rish, I., Raman, P., & Shi, H.-J. M. (2026). Adaptive Batch Sizes Using Non-Euclidean Gradient Noise Scales for Stochastic Sign and Spectral Descent. In Proceedings of the 43rd International Conference on Machine Learning (ICML).](https://openreview.net/forum?id=XMSaWRpEPS)

The repository is intentionally small: a single training loop, a flat package of swappable components, and a from-scratch Llama3 implementation. It is a lightweight version of [torchtitan](https://github.com/pytorch/torchtitan) [1], distilled down to the pieces needed for gradient-variance research and adaptive batch sizing. A bundled debug tokenizer and tiny `c4_test` fixture ship with the repo, so the default `train.py` runs end-to-end without any downloads — a 100-step debug-model run on 8 GPUs finishes in about 10 seconds.

Contributors:
- Shagun Gupta (Meta Platforms, Inc., shagun@meta.com)
- Hiroki Naganuma (Université de Montréal, naganuma.hiroki@mila.quebec)
- Parameswaran Raman (Meta Platforms, Inc., params@meta.com)
- Hao-Jun Michael Shi (Meta Platforms, Inc., hjmshi@meta.com)

## Features

Key features of this implementation include:
- **Per-step gradient-variance tracking** via PyTorch reduce-scatter hooks on every transformer block plus the embedding, final norm, and output projection — no manual instrumentation of the model.
- **Euclidean and non-Euclidean variance metrics**: L1, L2, and L∞ mean magnitudes and standard deviations, plus a spectral (nuclear-norm) path for 2D parameters.
- **GNS-driven adaptive batch sizing**: the global batch size is derived from the gradient noise scale and grows monotonically — local batch size grows first, then gradient accumulation takes over.
- **Three GNS estimators**: `l1_gns_batch_size`, `l2_gns_batch_size`, and `nuclear_gns_batch_size`, matching the L1 / L2 / spectral descent geometries.
- **Sign and spectral optimizers**: a `Signum` (SignSGD + momentum) and a `Muon` (Newton-Schulz orthogonalized momentum) implementation, plus any `torch.optim` optimizer (e.g. `AdamW`).
- **All standard parallelism layouts** via PyTorch-native APIs: DDP, FSDP, HSDP, and TP (with sequence and loss parallelism), selected from a single config.
- **Token-budget LR schedule** (warmup-stable-decay over tokens, not steps), with optional `sqrt(batch_scale)` learning-rate scaling for the adaptive-batch flow.
- **Offline-by-default quickstart**: bundled debug tokenizer + `c4_test` fixture; Hugging Face auth is only needed for real tokenizers and streamed datasets.
- **W&B logging** with stdout fallback.

## Requirements

The implementation relies on recent PyTorch distributed APIs (`replicate_with_fsdp`, `fully_shard` / FSDP2, custom reduce-scatter hooks, and `local_map`). We recommend:

- PyTorch >= 2.4 (a recent build is required for the custom reduce-scatter hooks used by the variance path);
- Python >= 3.10 (the code uses PEP 604 unions, `match`/`case`, and slotted dataclasses);
- CUDA-capable GPUs for the distributed paths.

Python dependencies (`requirements.txt`):

```
torch >= 2.4.0
torchdata >= 0.8.0
datasets >= 3.6.0, < 4.8.0
tokenizers >= 0.15.0
wandb >= 0.16.0
```

## Quickstart

A bundled debug tokenizer + tiny `c4_test` fixture ship with the repo, so the default `train.py` runs without any downloads.

```bash
git clone <this-repo> variance_metrics
cd variance_metrics
pip install -r requirements.txt
N_GPUS=${N_GPUS:-8}
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node="$N_GPUS" \
    --rdzv_backend=c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter=0 --role=rank --tee=3 \
    -m train
```

- `N_GPUS` (env var) — How many GPUs to use. Defaults to 8; set `N_GPUS=1` for a single-GPU test run.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` — Helps PyTorch reuse GPU memory as the batch size changes during a run, so it runs out of memory less often.
- `--nproc_per_node` — How many worker processes to start — one per GPU. Comes from `N_GPUS`.
- `--rdzv_backend=c10d` — How the workers find each other on a single machine. No extra service needed. Use `etcd` for multi-machine setups.
- `--rdzv_endpoint=localhost:0` — Where the workers connect to coordinate. Port `0` means "pick any free port" — fine for one machine. For multiple machines, point them all at `<rank0-host>:<fixed-port>`.
- `--local-ranks-filter=0` — Only show the first GPU's logs in the terminal; the rest go to log files.
- `--role=rank` — The label torchrun puts in front of each worker's log lines (`[rank]`).
- `--tee=3` — Send worker output to both the terminal and the log files.
- `-m train` — The program to run — `train.py` in the repo root.

GPU requirements: the bundled debug fixture runs on a single GPU with `N_GPUS=1` (also set `parallelism.data_parallel_replicate_degree=1` in `train.py`); DDP / FSDP / TP all need ≥2 GPUs; the spectral-variance path is DDP-only and the code raises an error under FSDP / HSDP / TP.

## How to Use

A run is fully described by a single `TrainerConfig` in `train.py`. Each swappable piece (optimizer, dataloader, LR scheduler, parallelism, variance metrics, adaptive batch sizing, logging) is a nested config; `config.build()` returns a `Trainer`, and `trainer.train()` runs the loop. The examples below show how to assemble that config for common scenarios — edit `train.py` and launch with the Quickstart `torchrun` command.

A few notes on configuration:

- **Adaptive batch sizing requires variance metrics.** `TrainerConfig.__post_init__` enforces `variance_metrics.enable=True` whenever `adaptive_batch_size.enable=True`.
- **Variance is measured before gradient clipping**, since clipping mutates the gradients.
- **The GNS estimator should match the optimizer geometry**: use `l1_gns_batch_size` / `l2_gns_batch_size` for sign/Euclidean methods (e.g. `Signum`), and `nuclear_gns_batch_size` for spectral methods (e.g. `Muon`). The nuclear path is DDP-only.
- **The model flavor** is chosen via `model_registry(...)`; available flavors are `debugmodel`, `160M`, `300M`, `660M`, `1B`, `3B`.

### Example 1: Baseline training (Signum)

A plain training run with the bundled debug model and offline fixture, no instrumentation:

```python
from components.configs import (
    TrainerConfig,
    OptimizerConfig,
    HFTextDataLoaderConfig,
    TrainingConfig,
    ParallelismConfig,
)
from model import model_registry
from optimizer.Signum import Signum

config = TrainerConfig(
    model_spec=model_registry("debugmodel"),
    hf_assets_path="./tests/assets/tokenizer/debug",
    optimizer=OptimizerConfig(
        optimizer_cls=Signum,
        optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.1, "beta": 0.9},
    ),
    dataloader=HFTextDataLoaderConfig(
        dataset="c4_test",
        dataset_path="./tests/assets/c4_test",
    ),
    training=TrainingConfig(
        local_batch_size=8, global_batch_size=64, seq_len=2048, steps=100,
    ),
    parallelism=ParallelismConfig(
        data_parallel_replicate_degree=8,   # 8-way DDP
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
    ),
)
```

Logged keys: `loss_metrics/{global_avg_loss,global_max_loss}`, `grad_norm`, `n_tokens_seen`, `lr`.

### Example 2: Logging gradient-variance metrics

Add a `VarianceMetricsConfig` to capture per-step gradient variance and mean magnitudes:

```python
from components.configs import VarianceMetricsConfig

config = TrainerConfig(
    ...,
    variance_metrics=VarianceMetricsConfig(
        enable=True,
        freq=1,                  # capture every step; bump to amortize cost at scale
        spectral_variance=False, # True = nuclear-norm path (DDP only)
    ),
)
```

This registers a capture hook on `tok_embeddings`, `norm`, `output`, and every transformer block. Logged keys:

- `variance_global_gradient/{variance,std_l1,std_linf,nuclear_norm}`
- `variance_minibatch_gradient/{variance,std_l1,std_linf,nuclear_norm}`
- `variance_example_gradient/{variance,std_l1,std_linf,nuclear_norm}`
- `mean_gradient_norms/{mean_l2_sq,mean_l1,mean_linf,nuc_norm}`

Cost is roughly one extra all-reduce of per-parameter scalars per captured step, so `freq=1` is fine on small models; use `freq=10..100` at scale.

### Example 3: GNS-driven adaptive batch sizing

Drive the global batch size from the gradient noise scale. The local batch size grows first (up to `largest_local_batch_size`), then gradient accumulation grows the global batch; the batch size is monotonically non-decreasing.

```python
from components.configs import (
    OptimizerConfig,
    VarianceMetricsConfig,
    AdaptiveBatchSizeConfig,
    LRSchedulerConfig,
)
from optimizer.Signum import Signum

config = TrainerConfig(
    ...,
    optimizer=OptimizerConfig(
        optimizer_cls=Signum,
        optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.1, "beta": 0.9},
    ),
    variance_metrics=VarianceMetricsConfig(enable=True, freq=100),
    adaptive_batch_size=AdaptiveBatchSizeConfig(
        enable=True,
        batch_size_method="l1_gns_batch_size",   # or l2_gns_batch_size / nuclear_gns_batch_size
        largest_local_batch_size=16,
        gns_batch_size_constant=0.6,
        var_ema_constant=0.9,
        gradient_ema_constant=0.9,
        batch_size_update_freq_gns=100,
        largest_global_batch_size=None,          # optional cap
        initial_constant_batch_steps=None,       # defaults to the LR-warmup length
    ),
    lr_scheduler=LRSchedulerConfig(
        scale_with_batch_size=True,              # LR *= sqrt(batch_scale)
    ),
)
```

Logged keys:

- `batch_size/{global_batch_size,local_batch_size,accumulation_steps,suggested_batch_size}`
- `adaptive_batch_metrics/{var_ema,mean_gradient_ema,gns_ema,suggested_batch_size}`

To run on a token budget instead of a fixed step count, set `training=TrainingConfig(n_token_limit=...)`; the stop criterion then switches from `steps` to `global_tokens_seen > n_token_limit`.

### Example 4: Spectral (nuclear-norm) variance with Muon

The spectral path measures variance in the nuclear norm over 2D parameters and pairs naturally with the `Muon` optimizer and the `nuclear_gns_batch_size` estimator. **This path is DDP-only** — it raises a `RuntimeError` under FSDP / HSDP / TP.

```python
from components.configs import (
    OptimizerConfig,
    VarianceMetricsConfig,
    AdaptiveBatchSizeConfig,
    ParallelismConfig,
)
from optimizer.Muon import Muon

config = TrainerConfig(
    ...,
    optimizer=OptimizerConfig(
        optimizer_cls=Muon,
        optimizer_kwargs={"lr": 2e-2, "momentum": 0.95, "weight_decay": 0.0},
    ),
    variance_metrics=VarianceMetricsConfig(enable=True, freq=100, spectral_variance=True),
    adaptive_batch_size=AdaptiveBatchSizeConfig(
        enable=True,
        batch_size_method="nuclear_gns_batch_size",
    ),
    parallelism=ParallelismConfig(
        data_parallel_replicate_degree=8,   # DDP only for the spectral path
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
    ),
)
```

### Example 5: Using a real tokenizer and streamed dataset

To swap the bundled debug tokenizer for a real one (e.g. Llama-3.1-8B) and stream the full C4 dataset:

```python
model_spec = model_registry("1B")
model_spec.model.vocab_size = 128256   # Llama 3 tokenizer vocab

config = TrainerConfig(
    model_spec=model_spec,
    hf_assets_path="./tests/assets/tokenizer/Llama-3.1-8B",  # holds tokenizer.json + tokenizer_config.json
    dataloader=HFTextDataLoaderConfig(dataset="c4"),         # streamed from HF — needs HF auth
    ...,
)
```

Download a gated tokenizer with:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "meta-llama/Llama-3.1-8B",
    allow_patterns=["tokenizer.json", "tokenizer_config.json"],
    local_dir="./tests/assets/tokenizer/Llama-3.1-8B",
)
```

`meta-llama/Llama-3.1-8B` is a gated repo — accept the license on the model's Hugging Face page, then configure auth (see [Hugging Face authentication](#hugging-face-authentication)).

### Example 6: Switching optimizers

Any `torch.optim` optimizer can be wired in via `optimizer_cls` / `optimizer_kwargs`. For example, AdamW:

```python
from torch.optim import AdamW

config = TrainerConfig(
    ...,
    optimizer=OptimizerConfig(
        optimizer_cls=AdamW,
        optimizer_kwargs={"lr": 1e-3, "betas": (0.9, 0.95), "weight_decay": 0.1},
    ),
)
```

The two bundled optimizers are:

- **`Signum`** — SignSGD with momentum. `Signum(params, lr=1e-1, weight_decay=0.1, beta=0.9)`. Decoupled weight decay; `beta=0` recovers vanilla SignSGD.
- **`Muon`** — Momentum orthogonalized by Newton-Schulz. `Muon(params, lr=2e-2, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0)`. 2D parameters are orthogonalized via a quintic Newton-Schulz iteration; non-2D parameters fall back to SGD with momentum.

## Distributed Training Support

All parallelism is configured through a single `ParallelismConfig`, which is compiled into a `DeviceMesh` by `ParallelDims`:

```python
ParallelismConfig(
    data_parallel_replicate_degree=1,   # DDP / replicate degree
    data_parallel_shard_degree=-1,      # FSDP shard degree (-1 = use leftover ranks)
    tensor_parallel_degree=1,           # TP degree
)
```

The product `data_parallel_replicate_degree * data_parallel_shard_degree * tensor_parallel_degree` must equal the world size (with `-1` auto-filling the shard degree). The strategy is then selected automatically:

| Layout | Configuration |
|---|---|
| **DDP** | `replicate_degree = N`, `shard_degree = 1`, `tp = 1` (uses `replicate_with_fsdp`) |
| **FSDP** | `replicate_degree = 1`, `shard_degree = N`, `tp = 1` (uses `fully_shard` / FSDP2) |
| **HSDP** | `replicate_degree > 1` and `shard_degree > 1` |
| **TP** | `tensor_parallel_degree > 1` (sequence parallel on norms, loss parallel on, combinable with the above) |

Mixed precision is `bfloat16` parameters with `float32` reductions. TP requires `seq_len % tp == 0` and `n_heads % tp == 0` (validated at config time). Variance metrics require a data-parallel degree > 1; the **spectral (nuclear-norm) variance path is DDP-only** and raises a `RuntimeError` under FSDP / HSDP / TP.

## Variance Metrics

The `variance_metrics/` package captures per-step (or per-`freq`-step) gradient variance and mean magnitudes by hooking the reduce-scatter of each parameter group:

- `hooks.py` — `ReplicateVarianceCaptureHook` (DDP / replicate, and the spectral path) and `FSDPVarianceCaptureHook` (FSDP / HSDP). The backend is selected automatically from the active device mesh: the replicate hook is used when `spectral_variance=True` or when FSDP is not enabled, otherwise the FSDP hook is used.
- `calculator.py` — `VarianceMetricsCalculator` aggregates per-parameter statistics into the per-step metrics that get logged, all-reducing across the appropriate DDP / FSDP / TP process groups.
- `types.py` — `MeanMetrics`, `VarianceMetrics`, and `AggregatedVarianceMetrics` (global / minibatch / per-example variance).

Configuration is via `VarianceMetricsConfig(enable, freq, spectral_variance)`. The default (elementwise) path computes L1 / L2 / L∞ statistics; the spectral path computes nuclear-norm statistics over 2D parameters only.

## Adaptive Batch Sizing

`components/adaptive_batch_size_manager.py` implements `BatchSizeManager`, which changes the global batch size during training, driven by GNS. It maintains EMAs of the gradient variance and squared mean gradient and computes a suggested global batch size proportional to their ratio, scaled by `gns_batch_size_constant`. Three GNS methods are exposed, matching the optimizer geometry:

| `batch_size_method` | Variance term | Gradient term |
|---|---|---|
| `l2_gns_batch_size` | `variance` | `mean_l2_sq` |
| `l1_gns_batch_size` | `std_l1²` | `mean_l1²` |
| `nuclear_gns_batch_size` | `nuclear_norm²` | `nuc_norm²` (DDP only) |

The batch size is monotonically non-decreasing, capped at `largest_global_batch_size`, and updated only after `initial_constant_batch_steps` and on multiples of `batch_size_update_freq_gns`. Local batch size grows first (up to `largest_local_batch_size`), then gradient accumulation handles the rest. Requires `variance_metrics.enable=True`.

## Logging (W&B)

Set `metrics=LoggerConfig(enable_wandb=True, log_freq=1)` in `train.py`. Run name, project, group, and tags are read from environment variables (standard `wandb` conventions); only rank 0 logs, and if `wandb` is unavailable the trainer falls back to stdout-only logging.

```bash
WANDB_PROJECT=my_project \
WANDB_RUN_NAME=my_run \
WANDB_RUN_GROUP=my_sweep \
WANDB_RUN_TAGS=experiment_a \
torchrun --nproc_per_node="$N_GPUS" ... -m train   # see "Quickstart"
```

Other recognized variables include `WANDB_TEAM`, `WANDB_RUN_ID`, `WANDB_RUN_NOTES`, and `WANDB_RUN_JOB_TYPE`. The default project is `variance_metrics_oss`; logs are written under `./outputs/wandb/`.

## Datasets

Three entries are pre-wired in `DATASETS` (`components/text_datasets.py`):

| name | path | network? |
|---|---|---|
| `c4` | `allenai/c4` | streamed from HF — needs HF auth |
| `c4_test` | `tests/assets/c4_test` | bundled local fixture — no network |
| `c4_validation` | `allenai/c4` | streamed from HF — needs HF auth |

The default `train.py` uses `c4_test` and works offline. To add a new dataset, register a `DatasetConfig` in `components/text_datasets.py`:

```python
def _process_my_text(sample: dict[str, Any]) -> str:
    return sample["my_field"]

DATASETS["my_dataset"] = DatasetConfig(
    path="org/my_dataset",                        # HF repo or local path
    loader=lambda path: load_dataset(path, split="train", streaming=True),
    sample_processor=_process_my_text,
)
```

Then set `dataloader=HFTextDataLoaderConfig(dataset="my_dataset")` in `train.py`.

## Hugging Face authentication

Only needed for streamed datasets (`c4` / `c4_validation`) and gated tokenizer downloads (e.g. Llama). The bundled debug quickstart needs none of this.

1. Create a Hugging Face account: <https://huggingface.co/join>
2. Generate a read token: <https://huggingface.co/settings/tokens>
3. For any gated model (e.g. `meta-llama/Llama-3.1-8B`), open the model page and click "Agree and access repository".
4. Make the token available to the training process — either `export HF_TOKEN=hf_xxx` in the shell you launch `torchrun` from, or run `huggingface-cli login` once (writes `~/.cache/huggingface/token`).

If neither is configured, the dataloader raises a `RuntimeError` at startup that points back here.

## File structure

```
variance_metrics/
├── train.py                     # run config + entrypoint
├── trainer.py                   # Trainer (DDP/FSDP/HSDP/TP, variance, adaptive bs)
├── requirements.txt
├── components/                  # flat package: building blocks
│   ├── adaptive_batch_size_manager.py
│   ├── configs.py               # all dataclass configs (one per component)
│   ├── configurable.py          # Configurable base + auto-wired build()
│   ├── dataloader.py            # BaseDataLoader
│   ├── logger.py                # MetricsProcessor (stdout + WandB)
│   ├── lr_scheduler.py
│   ├── parallel_dims.py         # ParallelDims + DeviceMesh wiring
│   ├── text_datasets.py         # HuggingFaceTextDataLoader + DATASETS registry
│   ├── tokenizer.py             # HuggingFaceTokenizer
│   ├── utils.py
│   └── validate.py              # mid-training validation loop
├── model/                       # Llama3 transformer + parallelize_fn
│   ├── module.py                # Module/ModuleDict + Linear/Embedding/RMSNorm
│   ├── attention.py             # Attention + RoPE + apply_rotary_emb_complex
│   ├── llama3.py                # FeedForward + TransformerBlock + Llama3Model
│   └── parallelize.py           # apply DDP/FSDP/HSDP/TP
├── optimizer/                   # Signum / Muon; wired directly from train.py
├── tests/assets/                # bundled debug tokenizer + c4_test fixture
└── variance_metrics/            # FSDP/Replicate variance hooks + calculator
```

## Citation

If you use this code in your work, please use the following BibTeX entry.

```bibtex
@inproceedings{naganuma2026adaptive,
  title     = {Adaptive Batch Sizes Using Non-Euclidean Gradient Noise Scales for Stochastic Sign and Spectral Descent},
  author    = {Naganuma, Hiroki and Gupta, Shagun and Briki, Youssef and Mitliagkas, Ioannis and Rish, Irina and Raman, Parameswaran and Shi, Hao-Jun Michael},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=XMSaWRpEPS},
}
```

## References

1. [torchtitan: One-stop PyTorch native solution for production-ready LLM pre-training](https://github.com/pytorch/torchtitan). Wanchao Liang, Tianyu Liu, Less Wright, Will Constable, Andrew Gu, Chien-Chin Huang, Iris Zhang, Wei Feng, Howard Huang, Junjie Wang, Sanket Purandare, Gokul Nadathur, and Stratos Idreos. ICLR, 2025.
