<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Parallelism Helper Library

A production-ready Python library for estimating optimal parallelization strategies for LLM training. Provides accurate memory estimates and configuration recommendations for Tensor Parallelism (TP), Pipeline Parallelism (PP), Data Parallelism (DP), Context Parallelism (CP), and Expert Parallelism (EP).

## Features

### Supported Architectures
- **Standard Transformers**: GPT, LLaMA, Mistral, Qwen, Phi
- **Mixture of Experts (MoE)**: Mixtral, DeepSeek-MoE, GPT-OSS
- **State-Space Models (Mamba)**: Pure Mamba, Hybrid Mamba-Transformer (e.g., Nemotron-Nano)
- **Grouped Query Attention (GQA/MQA)**: Automatic detection and parameter counting
- **Gated FFN**: SwiGLU, GeGLU (3x parameter multiplier vs 2x for standard FFN)
- **Sliding Window Attention**: Mistral, Gemma 2 (reduces KV cache memory)

### Memory Components Modeled
- **Model Parameters**: BF16 working weights (2 bytes/param)
- **Gradients**: FP32 gradients (4 bytes/param)
- **Optimizer States**: FP32 Adam (8 bytes/param for momentum + variance)
  - **Total: 14 bytes/param** for BF16 mixed precision training
  - Uses **Distributed Optimizer** (FSDP2) - optimizer states sharded across DP ranks
- **Activations**: Per-token activation memory with gradient checkpointing support
- **KV Cache**: Attention key-value cache (reduced for Mamba layers and sliding window)
- **LoRA/PEFT**: Frozen base model + trainable adapter memory

### Parallelization Strategies
- **TP (Tensor Parallel)**: Shards layers across GPUs (Megatron-LM/NeMo style)
- **PP (Pipeline Parallel)**: Splits model layers across pipeline stages
- **DP (Data Parallel)**: Replicates model and shards data (FSDP/ZeRO-style)
- **CP (Context Parallel)**: Shards sequence dimension for very long contexts (>8K tokens)
- **EP (Expert Parallel)**: Shards MoE experts across GPUs

### Validation
- Extensively tested against **NVIDIA NeMo's official H100 configurations**
- Parameter counting accuracy: **0.0-0.8% error** across all tested models
- Communication heuristic tuned to match NeMo's preferences (e.g., pure DP for small models)

## Quick Start

### Python Library Usage

```python
from nhx.core.models.parallelism.api import estimate_parallelization, find_minimum_gpus

# Estimate parallelization for a model
result = estimate_parallelization(
    model_id="meta-llama/Meta-Llama-3-8B",
    gpus=8,
    gpu_mem_gb=80,
    seq_len=8192,
    microbatch_size=4  # Optional: fix microbatch size
)

# Print top 3 configurations
for i, config in enumerate(result.configs[:3], 1):
    print(f"{i}. TP={config.tp} PP={config.pp} DP={config.dp} CP={config.cp} EP={config.ep}")
    print(f"   Memory: {config.total_memory_per_rank_gb:.1f}GB")
    print(f"   Microbatch: {config.microbatch_per_dp}")

# Find minimum GPUs needed
min_gpus, best_config = find_minimum_gpus(
    model_id="meta-llama/Meta-Llama-3-70B",
    gpu_mem_gb=80,
    seq_len=4096
)

if min_gpus:
    print(f"Minimum GPUs: {min_gpus}")
    print(f"Config: TP={best_config.tp} PP={best_config.pp} DP={best_config.dp}")
```

### Command-Line Usage

```bash
# Basic usage
./cli.py \
  --pretrained meta-llama/Meta-Llama-3-8B \
  --gpus 8 \
  --gpu-mem-gb 80 \
  --seq-len 8192

# With fixed microbatch size
./cli.py \
  --pretrained meta-llama/Meta-Llama-3-70B \
  --gpus 64 \
  --gpu-mem-gb 80 \
  --seq-len 4096 \
  --microbatch-size 1

# With LoRA fine-tuning
./cli.py \
  --pretrained meta-llama/Meta-Llama-3-8B \
  --gpus 8 \
  --gpu-mem-gb 80 \
  --seq-len 4096 \
  --lora \
  --lora-r 16

# For long context (enables CP)
./cli.py \
  --pretrained mistralai/Mistral-7B-v0.1 \
  --gpus 16 \
  --gpu-mem-gb 80 \
  --seq-len 32768 \
  --max-cp 8

# For MoE models (enables EP)
./cli.py \
  --pretrained mistralai/Mixtral-8x7B-v0.1 \
  --gpus 64 \
  --gpu-mem-gb 80 \
  --seq-len 4096 \
  --max-ep 8
```

## API Reference

### Core Functions

#### `estimate_parallelization()`

```python
def estimate_parallelization(
    model_id: str,
    gpus: int,
    gpu_mem_gb: float,
    seq_len: int,
    act_ckpt_ratio: float = 0.25,
    max_tp: int = 64,
    max_cp: int = 8,
    no_cp: bool = False,
    max_ep: int = 8,
    no_ep: bool = False,
    max_microbatch: int = 64,
    microbatch_size: Optional[int] = None,
    attn_scratch_factor: float = 0.0,
    lora: bool = False,
    lora_r: int = 16,
) -> ParallelizationRecommendation
```

**Parameters:**
- `model_id`: HuggingFace model ID (e.g., "meta-llama/Meta-Llama-3-8B")
- `gpus`: Number of GPUs available
- `gpu_mem_gb`: Memory per GPU in GB
- `seq_len`: Sequence length in tokens
- `act_ckpt_ratio`: Fraction of activations kept (0.0-1.0, default 0.25 = ~4x savings)
- `max_tp`: Maximum tensor parallelism degree (default 64)
- `max_cp`: Maximum context parallelism degree (default 8)
- `no_cp`: Disable context parallelism (default False)
- `max_ep`: Maximum expert parallelism degree for MoE (default 8)
- `no_ep`: Disable expert parallelism (default False)
- `max_microbatch`: Maximum microbatch size to search for (default 64)
- `microbatch_size`: Fixed microbatch size (if set, skips binary search; default None)
- `attn_scratch_factor`: Attention scratch memory factor (default 0.0)
- `lora`: Enable LoRA fine-tuning mode (default False)
- `lora_r`: LoRA rank if enabled (default 16)

**Returns:** `ParallelizationRecommendation` with sorted list of viable configurations

#### `find_minimum_gpus()`

```python
def find_minimum_gpus(
    model_id: str,
    gpu_mem_gb: float,
    seq_len: int,
    max_gpus: int = 128,
    **kwargs
) -> tuple[int, ParallelizationConfig] | tuple[None, None]
```

**Parameters:**
- `model_id`: HuggingFace model ID
- `gpu_mem_gb`: Memory per GPU in GB
- `seq_len`: Sequence length
- `max_gpus`: Maximum GPUs to try (default 128)
- `**kwargs`: Additional args passed to `estimate_parallelization`

**Returns:** `(min_gpus, best_config)` or `(None, None)` if not feasible

#### `infer_model_cfg_from_hf()`

```python
def infer_model_cfg_from_hf(pretrained_or_path: str) -> ModelConfig
```

Automatically detects model architecture from HuggingFace config. Returns a `ModelConfig` object with all architectural details.

### Pydantic Models

#### `EstimationParams`

Strongly-typed parameters for estimation:

```python
class EstimationParams(BaseModel):
    gpus: int
    gpu_mem_gb: float
    seq_len: int
    global_batch_size: Optional[int] = None
    microbatch_size: Optional[int] = None
    max_microbatch: int = 64
    act_ckpt_ratio: float = 0.25
    attn_scratch_factor: float = 0.0
    max_tp: int = 64
    max_cp: int = 8
    no_cp: bool = False
    max_ep: int = 8
    no_ep: bool = False
    lora: bool = False
    lora_r: int = 16
    lora_include_regex: Optional[list[str]] = None
    lora_exclude_regex: Optional[list[str]] = None
    pretrained: str = ""
```

#### `ModelConfig`

Model architecture configuration:

```python
class ModelConfig(BaseModel):
    model_name: str
    family: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    ffn_hidden_size: int
    vocab_size: int
    tied_embeddings: bool
    gated_mlp: bool
    moe_config: Optional[MoEConfig] = None
    mamba_config: Optional[MambaConfig] = None
    sliding_window_config: Optional[SlidingWindowConfig] = None
```

#### `ParallelizationConfig`

Single parallelization configuration:

```python
class ParallelizationConfig(BaseModel):
    tp: int  # Tensor Parallelism
    pp: int  # Pipeline Parallelism
    dp: int  # Data Parallelism
    cp: int  # Context Parallelism (1 = disabled)
    ep: int  # Expert Parallelism (1 = disabled)
    microbatch_per_dp: int
    per_rank_static_gb: float
    est_act_gb_per_mb1: float
    total_memory_per_rank_gb: float
    score: int  # Communication cost (lower is better)

    @property
    def total_gpus(self) -> int:
        return self.tp * self.pp * self.dp * self.cp * self.ep

    @property
    def global_batch_size(self) -> int:
        return self.microbatch_per_dp * self.dp
```

## Memory Model

### BF16 Mixed Precision Training (14 bytes/param)

```
Memory per Parameter:
├─ BF16 model weights:    2 bytes  (working copy on GPU)
├─ FP32 gradients:        4 bytes  (computed during backward)
├─ FP32 optimizer states: 8 bytes  (Adam momentum + variance)
│  └─ SHARDED by DP (distributed optimizer / FSDP2)
└─ Total:                14 bytes
```

**Note:** FP32 master copy is kept on disk, not in GPU memory during training.

### Memory Sharding

```
Parameters:     Sharded by TP * PP * EP
                (EP only shards expert parameters)

Gradients:      Sharded by TP * PP * EP
                (replicated across DP)

Optimizer:      Sharded by TP * PP * EP * DP
                (distributed optimizer)

Activations:    Sharded by TP * CP
                (sequence dimension sharded by CP)

KV Cache:       Sharded by TP * CP
                (reduced by sliding window if present)
```

### Memory Formula

```python
# Per-rank memory breakdown
static_memory = (
    bf16_params / (tp * pp * ep_for_experts) +
    fp32_grads / (tp * pp * ep_for_experts) +
    fp32_optimizer / (tp * pp * ep * dp)  # Distributed!
)

activation_memory = (
    activations_per_token * (microbatch * seq_len / cp) / tp
)

kv_cache_memory = (
    kv_per_token * (microbatch * effective_seq_len / cp) / tp
)

total_memory = static_memory + 1.1 * (activation_memory + kv_cache_memory)
# 10% overhead for temporary buffers
```

## Parallelization Constraints

The tool enforces the following constraints:

1. **TP must divide hidden_size** (for attention head splitting)
2. **PP must divide num_layers** (for even pipeline stages)
3. **EP must divide num_experts** (for even expert distribution)
4. **CP must divide seq_len** (for even sequence splitting)
5. **TP × PP × CP × EP must divide total GPUs** (for valid DP calculation)

## Communication Heuristic

The tool ranks configurations using a communication cost heuristic tuned to match NVIDIA NeMo's preferences:

### Key Principles

1. **Prefer DP for small models**: Pure DP=8 for models that fit easily (maximum throughput)
2. **TP is costly**: All-reduce per layer, use sparingly (prefer TP=1 when possible)
3. **PP is very costly**: Pipeline bubbles, only use when memory requires it
4. **CP for long contexts**: Based on sequence-to-parameter ratio
5. **EP for MoE**: Strong preference for EP=num_experts (1 expert per GPU)

### Cost Calculation

```python
# Simplified heuristic
tp_cost = n_layers * tp * d_model * 100  # Linear in model dimensions
  + penalty for tp > 4  # Diminishing returns

dp_cost = -1e8 * dp  # NEGATIVE = throughput benefit
  # Reduced bonus for MoE (EP more important)
  # Reduced bonus when using TP/PP (less critical)

pp_cost = 1e8 to 5e8 * (pp - 1)  # High penalty (pipeline bubbles)
  # Lower when TP > 1 (both have comm overhead)
  # Lower for MoE (useful for distributing experts)

cp_cost = based on seq_len / param_count ratio
  # Bonus when sequence memory dominates
  # Penalty when sequence memory is small

ep_cost = -5e8 when ep == num_experts  # HUGE bonus
  + 8e8 when ep == 1 (for MoE)  # HUGE penalty

balance_bonus = -3e8 when tp == pp  # Prefer balanced TP/PP
```

## PyTorch Implementation Mapping

### Data Parallelism (DP)
- Maps to **FSDP** (Fully Sharded Data Parallel) or **DDP**
- **FSDP/ZeRO**: Shards parameters, gradients, optimizer states across DP ranks
- Memory savings: ~1/DP for optimizer states (distributed optimizer)
- Communication: Gradient all-reduce (less frequent than TP)

### Tensor Parallelism (TP)
- Maps to **Megatron-LM** tensor model parallelism
- Column/row parallel for Linear layers, vocab parallel for embeddings
- Requires all-reduce per layer forward/backward (high communication cost)
- Memory savings: ~1/TP for parameters, activations, KV cache
- Best for high-bandwidth interconnects (NVLink, InfiniBand)

### Pipeline Parallelism (PP)
- Maps to **Megatron-LM** pipeline parallelism with microbatching
- Splits model layers across pipeline stages
- Introduces pipeline bubbles (idle time between stages)
- Memory savings: ~1/PP for parameters
- Communication: Only at stage boundaries (lower than TP)

### Context Parallelism (CP)
- Maps to **Megatron-LM** context parallelism for long sequences
- Distributes sequence length: `effective_seq_len = seq_len / CP`
- Memory savings: ~1/CP for activations and KV cache
- Communication: All-gather for attention across CP dimension
- Used for sequences >8K tokens (especially >32K)

### Expert Parallelism (EP)
- Maps to **Megatron-LM** expert parallelism for MoE
- Shards experts across GPUs with token dispatching
- Requires all-to-all communication for token routing
- Memory savings: ~1/EP for expert parameters only
- Only applicable to MoE models (Mixtral, DeepSeek-MoE, etc.)

## Supported Models

### Tested Models (with parameter count accuracy)

| Model | Type | Parameters | Accuracy | Notes |
|-------|------|------------|----------|-------|
| GPT-2 | Dense | 124M | 0.0% | Standard transformer |
| GPT-J-6B | Dense | 6B | 0.1% | Standard transformer |
| GPT-NeoX-20B | Dense | 20B | 0.1% | Standard transformer |
| Phi-2 | Dense | 2.7B | 0.2% | GQA, gated MLP |
| Phi-4 | Dense | 14B | 0.3% | GQA, gated MLP |
| LLaMA-3-8B | Dense | 8B | 0.1% | GQA, SwiGLU |
| LLaMA-3-70B | Dense | 70B | 0.2% | GQA, SwiGLU |
| Mistral-7B | Dense | 7B | 0.1% | GQA, sliding window |
| Mixtral-8x7B | MoE | 47B | 0.4% | 8 experts, 2 active |
| GPT-OSS-20B | MoE | 20B | 0.3% | 16 experts |
| GPT-OSS-120B | MoE | 117B | 0.5% | 16 experts, large |
| DeepSeek-67B | MoE | 67B | 0.6% | MoE architecture |
| DeepSeek-V3-671B | MoE | 671B | 0.8% | Very large MoE |
| Nemotron-Nano-9B | Hybrid | 9B | 0.7% | Mamba-Transformer hybrid |

### Architecture Detection

The tool automatically detects:
- **MoE**: Via `num_experts`, `num_experts_per_tok`, router config attributes
- **Mamba**: Via `ssm_state_size`, `layers_block_type`, hybrid detection
- **GQA/MQA**: Via `num_key_value_heads` < `num_attention_heads`
- **Gated MLP**: Via activation function names (swiglu, geglu, siglu)
- **Sliding Window**: Via `sliding_window` config attribute

## Limitations

### Not Modeled
- **Sequence Parallelism (SP)**: Would reduce activation memory further
- **Flash Attention**: Memory savings not explicitly modeled (use `attn_scratch_factor`)
- **Gradient Accumulation**: Assumes microbatch fits in memory
- **Communication Bandwidth**: Uses heuristic scoring, not actual profiling
- **Cross-node vs Intra-node**: Assumes homogeneous interconnect
- **Sequence Packing**: Improvements from packing not modeled

### Assumptions
- Fixed sequence length (no dynamic shapes)
- Gradient checkpointing controlled by `act_ckpt_ratio` (default 0.25 = ~4x savings)
- 10% temporary buffer overhead (may need tuning)
- BF16 mixed precision training (14 bytes/param)
- Adam optimizer (8 bytes/param for states)
- Distributed optimizer (FSDP2/ZeRO-style sharding)

## Usage Recommendations

### General Guidelines
1. **Start with DP**: For small models that fit, use pure DP for maximum throughput
2. **Use TP for large models**: When model doesn't fit, use minimal TP needed
3. **Use PP sparingly**: High pipeline bubble overhead, only when necessary
4. **Use CP for long contexts**: >8K tokens (especially >32K)
5. **Use EP for MoE**: Preferably EP=num_experts for optimal memory/compute

### Model-Specific Tips

**Small Models (< 10B params, < 8K seq):**
- Prefer: DP=8, TP=1, PP=1, CP=1
- Rationale: Maximize throughput with data parallelism

**Medium Models (10-70B params, ~4K seq):**
- Prefer: TP=2-4, PP=1, DP=remaining
- Rationale: Minimal TP to fit, maximize DP for throughput

**Large Models (70B+ params, ~4K seq):**
- Prefer: TP=4-8, PP=2-4 (balanced), DP=remaining
- Rationale: Balanced TP/PP for memory, communication efficiency

**MoE Models:**
- Prefer: EP=num_experts, TP=1-2, PP=1, DP=remaining
- Rationale: EP for memory efficiency, minimize TP/PP overhead

**Long Context (32K+ tokens):**
- Prefer: CP=4-8, TP=1-2, PP=1, DP=remaining
- Rationale: CP to shard sequence memory, reduce activation overhead

## Testing

The library includes comprehensive test suites:

### Unit Tests (`test_parallelism_models.py`)
- Pydantic model serialization/deserialization
- Model architecture detection for 8+ model families
- Edge cases for CP, EP, multi-dimensional parallelism

### Integration Tests (`test_nemo_validation.py`)
- Validates against NVIDIA NeMo's H100 configurations
- 31 test cases across LLaMA, Mistral, Mixtral, DeepSeek, Nemotron
- Tests parameter counting, memory estimation, parallelization strategies

```bash
# Run all tests
uv run pytest test_parallelism_models.py test_nemo_validation.py -v

# Run specific test
uv run pytest test_nemo_validation.py::test_nemo_config[pre_train_llama3_70b_bf16] -v
```

## Development

### Adding New Models

To add support for a new architecture:

1. **Update config inference** in `infer_model_cfg_from_hf()`:
   - Add detection logic for new architecture attributes
   - Create new Pydantic config model if needed

2. **Update parameter counting** in `param_count()`:
   - Add architecture-specific parameter calculations
   - Ensure accuracy with model card specs

3. **Add tests** in `test_parallelism_models.py`:
   - Add model to parametrized test suite
   - Verify parameter count accuracy

4. **Validate against baselines** in `test_nemo_validation.py`:
   - If available, compare against official configs

## Performance

The library is designed for fast inference:

- **Model introspection**: Uses `meta` device (no weight loading)
- **Configuration caching**: Reusable `ModelConfig` objects
- **Efficient search**: Binary search for microbatch, constraint filtering
- **Typical runtime**: 1-5 seconds per model on modern hardware

