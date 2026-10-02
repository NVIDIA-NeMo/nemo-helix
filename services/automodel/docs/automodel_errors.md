<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Automodel Error Table for nhx-automodel

This table maps Automodel errors to Custom Exception Classes for implementation.

**Implementation:** rules live in `src/nhx/automodel/tasks/training/errors/error_rules.yaml`; conversion runs in the training task runner via `create_error_details()`. Tests: `tests/tasks/training/test_errors.py`.

## Validation Status Legend

These markers indicate whether an error needs a rule in `error_rules.yaml`. Reviewed the code and categorized each potential error:

> **`[VALIDATED]`** = Pre-validated in nhx-automodel before Automodel execution (e.g., by `prepare_dataset()`, `validate_datasets()`, or API validation). These errors cannot reach the training backend.
>
> **`[ADD]`** = May occur at runtime and needs an error handling rule in `error_rules.yaml`. These are the errors we care about.
>
> **`[NEVER OCCUR]`** = Will never occur with current nhx-automodel configuration (e.g., uses Megatron dataset which we don't use, or NanoGPT which isn't supported).

---

## Custom Exception Classes

### Classes That NEED Implementation (have [ADD] errors)

| Exception Class | HTTP Status | Description | [ADD] Count |
|----------------|-------------|-------------|-------------|
| `DatasetFormatError` | 400 | Dataset has invalid format/schema | 1 |
| `ModelLoadError` | 500 | Failed to load/initialize model | 4 |
| `TrainingConfigError` | 400 | Invalid training config (parallelism, PEFT, etc.) | 6 |
| `CheckpointError` | 500 | Checkpoint save/load failure | 4 |
| `CudaError` | 500 | GPU/CUDA runtime error | 2 |
| `DistributedError` | 500 | Distributed training failure | 3 |
| `TrainingTimeoutError` | 500 | Training exceeded time limit | 1 |
| `InternalError` | 500 | Unexpected internal error | 5 |

---

### Classes That DON'T Need Implementation (all [VALIDATED] or [NEVER OCCUR])

| Exception Class | HTTP Status | Reason Not Needed |
|----------------|-------------|-------------------|
| `DatasetNotFoundError` | 404 | All errors pre-validated by `prepare_dataset()` or never occur (Megatron/NanoGPT not used) |
| `DatasetPermissionError` | 403 | Never occurs - Megatron not used, nhx-automodel creates files with correct permissions |
| `ModelNotFoundError` | 404 | All errors pre-validated by API before training starts |

---

## Error Mapping Table

### 1. DatasetNotFoundError (404)

| Automodel Error Raised | Validation Status | What It Means | Code Pointer |
|------------------------|-------------------|---------------|--------------|
| `FileNotFoundError(f"File not found: {fp}")` | `[VALIDATED]` `prepare_dataset()` creates files | A chat dataset file does not exist at the specified path | `/opt/Automodel/nemo_automodel/components/datasets/llm/chat_dataset.py:77` |
| `ValueError("Expected path to have a value.")` | `[NEVER OCCUR]` Megatron dataset not used | No dataset path was provided in the configuration | `/opt/Automodel/nemo_automodel/components/datasets/llm/megatron_dataset.py:306` |
| `ValueError("Expected path to be of string or Path type.")` | `[NEVER OCCUR]` Megatron dataset not used | Dataset path must be a string or Path object, got wrong type | `/opt/Automodel/nemo_automodel/components/datasets/llm/megatron_dataset.py:321` |
| `ValueError(f"No files matching glob {path} found")` | `[NEVER OCCUR]` Megatron dataset not used | The glob pattern for dataset files matched no files | `/opt/Automodel/nemo_automodel/components/datasets/llm/megatron_dataset.py:351` |
| `FileNotFoundError(f"Expected {str(file_path)} to exist.")` | `[NEVER OCCUR]` Megatron dataset not used | A specific dataset file does not exist at the given path | `/opt/Automodel/nemo_automodel/components/datasets/llm/megatron_dataset.py:338` |
| `RuntimeError("No data files provided")` | `[NEVER OCCUR]` nhx-automodel always provides files in config | No data files were specified for the chat dataset | `/opt/Automodel/nemo_automodel/components/datasets/llm/chat_dataset.py:70` |
| `ValueError("data_files entries must be strings")` | `[NEVER OCCUR]` nhx-automodel always provides strings | Data file paths must be strings, but got wrong type | `/opt/Automodel/nemo_automodel/components/datasets/llm/chat_dataset.py:45` |
| `FileNotFoundError(f"No files matched pattern {file_pattern}")` | `[NEVER OCCUR]` NanoGPT dataset not used | No files match the specified file pattern for NanoGPT dataset | `/opt/Automodel/nemo_automodel/components/datasets/llm/nanogpt_dataset.py:309` |

**User Message**: `Dataset not found: {details}. Please verify the dataset path exists and is accessible.`

---

### 2. DatasetFormatError (400)

| Automodel Error Raised | Validation Status | What It Means | Code Pointer |
|------------------------|-------------------|---------------|--------------|
| `ValueError("Each sample must contain a 'messages' list in OpenAI format")` | `[VALIDATED]` in `validate_datasets` | Dataset samples are not in the expected OpenAI chat format | `/opt/Automodel/nemo_automodel/components/datasets/llm/chat_dataset.py:171` |
| `RuntimeError(f"no sample to consume: {total_samples}")` | `[VALIDATED]` in `validate_batch_size` | The dataset has zero samples to train on | `/opt/Automodel/nemo_automodel/components/datasets/llm/megatron/sampler.py:59` |
| `DatasetFormatError("...")` | `[VALIDATED]` in `validate_datasets` | Dataset doesn't match expected JSON schema | nhx-automodel dataset validation |
| `DatasetFormatError("{file} is empty")` | `[VALIDATED]` in `validate_dataset` | Dataset file is empty | nhx-automodel dataset validation |
| `ValueError(f"Unsupported role in messages: {role}")` | `[ADD]` Schema only validates role is string, not value | A message has an invalid role (not system/user/assistant/tool) | `/opt/Automodel/nemo_automodel/components/datasets/llm/chat_dataset.py:114` |
| `ValueError("ChatDataset requires a tokenizer with chat template support.")` | `[NEVER OCCUR]` `set_chat_template()` provides or errors first | The tokenizer does not have a chat template defined | `/opt/Automodel/nemo_automodel/components/datasets/llm/chat_dataset.py:150` |
| `ValueError(f"Dataset sample is too long ({seq_len} > {packed_sequence_size})...")` | `[NEVER OCCUR]` Truncation to `packed_size` matches `packed_sequence_size` | A single sample exceeds the maximum allowed sequence length | `/opt/Automodel/nemo_automodel/components/datasets/llm/packed_sequence.py:259` |
| `ValueError("Tokenizer lacks a usable chat template...")` | `[NEVER OCCUR]` `set_chat_template()` provides or errors first | Tokenizer does not have a chat template for chat-format datasets | `/opt/Automodel/nemo_automodel/components/datasets/llm/formatting_utils.py:205` |
| `ValueError(f"Invalid JSON in blend file {path}: {e}")` | `[NEVER OCCUR]` Megatron dataset not used | Blend JSON file has invalid JSON syntax or wrong structure | `/opt/Automodel/nemo_automodel/components/datasets/llm/megatron_dataset.py:400,403` |
| `ValueError("Tokenizer is required")` | `[NEVER OCCUR]` Tokenizer always provided | Chat dataset was not provided a tokenizer during initialization | `/opt/Automodel/nemo_automodel/components/datasets/llm/chat_dataset.py:142` |
| `ValueError(f"Expected {n_bytes} to be equal to 2 (uint16) or 4 (uint32).")` | `[NEVER OCCUR]` NanoGPT dataset not used | Binary dataset uses unsupported byte size per token | `/opt/Automodel/nemo_automodel/components/datasets/llm/nanogpt_dataset.py:149` |
| `AssertionError("Expected answer to be in column_mapping")` | `[NEVER OCCUR]` nhx-automodel sets column mapping correctly | Column mapping is missing required fields | `/opt/Automodel/nemo_automodel/components/datasets/llm/column_mapped_text_instruction_dataset.py:213-229` |
| `ValueError("All elements must be strings")` | `[NEVER OCCUR]` nhx-automodel always provides strings | Dataset file paths must be strings or list of strings | `/opt/Automodel/nemo_automodel/components/datasets/llm/column_mapped_text_instruction_dataset.py:70,73` |
| `ValueError(f"Missing required fields: {missing}...")` | `[NEVER OCCUR]` Retrieval dataset not used | Dataset item is missing required fields for retrieval | `/opt/Automodel/nemo_automodel/components/datasets/llm/retrieval_dataset.py:127` |

**User Message**: `Dataset format error: {details}. Please check your dataset matches the expected schema.`

---

### 3. DatasetPermissionError (403)

| Automodel Error Raised | Validation Status | What It Means | Code Pointer |
|------------------------|-------------------|---------------|--------------|
| `PermissionError(f"Expected {str(path)} to be readable.")` | `[NEVER OCCUR]` Megatron not used; nhx-automodel files have correct permissions | Cannot read the dataset file due to permission issues | `/opt/Automodel/nemo_automodel/components/datasets/llm/megatron_dataset.py:328,333,340` |

**User Message**: `Dataset access denied: Cannot read {path}. Please check file permissions.`

---

### 4. ModelNotFoundError (404)

| Automodel Error Raised | Validation Status | What It Means | Code Pointer |
|------------------------|-------------------|---------------|--------------|
| `FileNotFoundError(f"Model path {model_path} does not exist")` | `[VALIDATED]` API validates model exists | The specified model directory or HuggingFace model ID does not exist | `/opt/Automodel/nemo_automodel/components/checkpoint/checkpointing.py:296` |
| `FileNotFoundError(f"No snapshot directories found in {snapshots_root}")` | `[VALIDATED]` model pre-downloaded | Model not found in HuggingFace cache | `/opt/Automodel/nemo_automodel/components/checkpoint/checkpointing.py:691` |
| `FileNotFoundError(file_path)` | `[NEVER OCCUR]` Config generated by nhx-automodel compiler, always valid | A required configuration file is missing | `/opt/Automodel/nemo_automodel/_cli/app.py:53,86` |

**User Message**: `Model not found: {path}. Please verify the model path is correct.`

---

### 5. ModelLoadError (500)

| Automodel Error Raised | Validation Status | What It Means | Code Pointer |
|------------------------|-------------------|---------------|--------------|
| `RuntimeError(f"_apply(): Couldn't swap {module._get_name()}.{key}")` | `[ADD]` May occur at runtime (model corruption) | Model weights could not be applied to a layer | `/opt/Automodel/nemo_automodel/components/checkpoint/checkpointing.py:817,837` |
| `RuntimeError("Failed to patch model")` | `[ADD]` May occur at runtime | Could not apply model optimizations/patches | `/opt/Automodel/nemo_automodel/_transformers/auto_model.py:124` |
| `AssertionError(f"Signature mismatch:\n  original: {sig_orig}\n  patched : {sig_patch}")` | `[ADD]` May occur at runtime | Method signature doesn't match expected signature | `/opt/Automodel/nemo_automodel/_transformers/auto_model.py:55` |
| `ValueError("lm_head.weight not found in model")` | `[ADD]` May occur at runtime (model corruption) | Model is missing the language model head weight | `/opt/Automodel/nemo_automodel/recipes/llm/train_ft.py:760` |
| `AssertionError("model_name is required when loading base model")` | `[NEVER OCCUR]` nhx-automodel always provides model_name | Model name must be specified when loading base model | `/opt/Automodel/nemo_automodel/components/checkpoint/checkpointing.py:367` |

**User Message**: `Model loading failed: {details}. The model may be corrupted or incompatible.`

---

### 6. TrainingConfigError (400)

#### 6a. Parallelism Config Errors

| Automodel Error Raised | Validation Status | What It Means | Code Pointer |
|------------------------|-------------------|---------------|--------------|
| `ValueError("Model '{model_name}' is not compatible with pipeline parallelism:\n\n1. tie_word_embeddings=True is not supported for pipelining. Use separate input/output embeddings.")` | `[ADD]` Model-dependent, may occur at runtime | **Tied embeddings**: Some models (e.g., GPT-2, some LLaMA variants) share the same weight matrix for input embeddings and output (lm_head) layer. Pipeline parallelism splits the model across stages, so if input embedding is on stage 0 and lm_head is on the last stage, they can't share weights. Models with `config.tie_word_embeddings=True` will fail. | `/opt/Automodel/nemo_automodel/components/distributed/pipelining/hf_utils.py:237-240` |
| `ValueError("Model '{model_name}' is not compatible with pipeline parallelism:\n\n1. Encoder-Decoder models with cross-attention are not supported yet for pipeline parallelism.")` | `[ADD]` Model-dependent, may occur at runtime | **Encoder-Decoder models**: Models like T5, BART, and mBART have separate encoder and decoder stacks with cross-attention between them. Pipeline parallelism can't properly handle the cross-attention communication patterns between encoder and decoder stages. Models with `config.is_encoder_decoder=True` will fail. | `/opt/Automodel/nemo_automodel/components/distributed/pipelining/hf_utils.py:241-242` |
| `AssertionError(f"pp_batch_size // pp_microbatch_size must be >= pp_size")` | `[ADD]` May occur with PP enabled | PP requires enough micro-batches to fill the pipeline | `/opt/Automodel/nemo_automodel/recipes/llm/train_ft.py:902-904` |
| `ValueError("Model does not support SDPA required for context parallelism")` | `[ADD]` Model-dependent, may occur at runtime | Context parallelism requires a model with SDPA support | `/opt/Automodel/nemo_automodel/_transformers/auto_model.py:211` |
| `ValueError(f"world_size ({self.world_size}) must be divisible by...")` | `[VALIDATED]` in `customizer_automodel_config.py` | Total GPUs cannot be evenly split across parallelism dimensions | `/opt/Automodel/nemo_automodel/components/distributed/fsdp2.py:179` |
| `AssertionError(f"{dp_cp_size=} must be a multiple of {self.ep_size=}")` | `[VALIDATED]` in `customizer_automodel_config.py` | For MoE, (dp_size × cp_size) must be divisible by expert_parallel_size | `/opt/Automodel/nemo_automodel/components/distributed/fsdp2.py:197` |
| `ValueError(...)` | `[VALIDATED]` in `customizer_automodel_config.py` | Data parallel size must be positive | `/opt/Automodel/nemo_automodel/components/distributed/fsdp2.py:176` |
| `AssertionError("dp_size must be a multiple of dp_replicate_size")` | `[NEVER OCCUR]` dp_replicate_size not configurable | Data parallel size must be evenly divisible by dp_replicate_size | `/opt/Automodel/nemo_automodel/components/distributed/fsdp2.py:192` |
| `AssertionError("Expected {name} to be an int...")` | `[NEVER OCCUR]` Config always produces valid integers | Parallelism dimension values must be positive integers | `/opt/Automodel/nemo_automodel/components/distributed/fsdp2.py:214-215` |
| `AssertionError("MegatronFSDPManager is not supported...")` | `[NEVER OCCUR]` Not used in nhx-automodel | MegatronFSDP cannot be used with pipeline parallelism | `/opt/Automodel/nemo_automodel/recipes/llm/train_ft.py:919-921` |
| `ValueError("Packed sequence is only supported with CP size 1")` | `[NEVER OCCUR]` nhx-automodel doesn't use CP with packing | Packed sequences cannot be used with context parallelism | `/opt/Automodel/nemo_automodel/_transformers/auto_model.py:201` |
| `ValueError("Student and teacher tokenizers have different vocab sizes...")` | `[NEVER OCCUR]` KD not used in finetune path | Student and teacher models have incompatible tokenizers | `/opt/Automodel/nemo_automodel/recipes/llm/kd.py:107,115,119` |
| `ValueError("Pipeline parallelism support will be added in the future...")` | `[NEVER OCCUR]` KD not used in finetune path | PP cannot be used with knowledge distillation | `/opt/Automodel/nemo_automodel/recipes/llm/kd.py:135` |

#### 6b. PEFT/LoRA Config Errors

| Automodel Error Raised | Validation Status | What It Means | Code Pointer |
|------------------------|-------------------|---------------|--------------|
| `ImportError("triton is not installed. Please install it with `pip install triton`.")` | `[ADD]` Environment-dependent, may occur | Triton library required for optimized LoRA kernels | `/opt/Automodel/nemo_automodel/components/_peft/lora_kernel.py:65,105,153,215,270` |
| `AssertionError("Incompatible X and LoRA A dimensions")` | `[ADD]` May occur at runtime | LoRA adapter dimensions don't match base model layer | `/opt/Automodel/nemo_automodel/components/_peft/lora_kernel.py:272-276` |
| `ValueError("QAT with PEFT is not supported in 25.11")` | `[NEVER OCCUR]` QAT not supported in nhx-automodel | Quantization-Aware Training cannot be used with PEFT | `/opt/Automodel/nemo_automodel/recipes/llm/train_ft.py:216` |
| `ValueError("PEFT checkpointing is not supported for torch_save format...")` | `[NEVER OCCUR]` nhx-automodel uses safetensors (hardcoded) | PEFT checkpoints must use safetensors format | `/opt/Automodel/nemo_automodel/recipes/llm/train_ft.py:377` |
| `ValueError("Expected match_all_linear to be true or target_modules...")` | `[NEVER OCCUR]` nhx-automodel uses match_all_linear=True | PEFT config must specify which modules to apply LoRA to | `/opt/Automodel/nemo_automodel/components/_peft/module_matcher.py:87` |
| `AssertionError("exclude_modules must be empty when target_modules is used.")` | `[NEVER OCCUR]` Config never uses both | Cannot use both target_modules and exclude_modules | `/opt/Automodel/nemo_automodel/components/_peft/module_matcher.py:108` |

#### 6c. Batch Config Errors (ALL VALIDATED/NEVER OCCUR)

| Automodel Error Raised | Validation Status | What It Means | Code Pointer |
|------------------------|-------------------|---------------|--------------|
| `AssertionError("warmup_steps < lr_decay_steps")` | `[VALIDATED]` in `customizer_automodel_config.py` | Warmup steps must be less than total training steps | lr_scheduler assertion |
| `DatasetFormatError("Batch size cannot be larger than...")` | `[VALIDATED]` in dataset validation | Batch size exceeds validation sample count | nhx-automodel dataset validation |
| `RuntimeError(f"micro_batch_size must be greater than 0...")` | `[NEVER OCCUR]` Megatron sampler not used | Micro batch size must be a positive number | `/opt/Automodel/nemo_automodel/components/datasets/llm/megatron/sampler.py:61` |
| `RuntimeError(f"global_batch_size ({gbs}) is not divisible by...")` | `[NEVER OCCUR]` Megatron sampler not used | Global batch size must be divisible by (micro_batch × dp_size) | `/opt/Automodel/nemo_automodel/components/datasets/llm/megatron/sampler.py:70` |
| `AssertionError(f"grad_acc_steps ({steps}) must be >= 1...")` | `[NEVER OCCUR]` nhx-automodel does not set grad_acc_steps | Gradient accumulation steps must be at least 1 | `/opt/Automodel/nemo_automodel/components/training/step_scheduler.py:74-76` |
| `AssertionError("epoch_len must be provided if max_steps is not provided")` | `[NEVER OCCUR]` nhx-automodel always provides max_steps | Cannot determine epoch length without max_steps | `/opt/Automodel/nemo_automodel/components/training/step_scheduler.py:92` |
| `AssertionError("num_epochs must be greater than 0")` etc. | `[NEVER OCCUR]` Epochs calculated in `customizer_automodel_config.py` | Training parameters have invalid values | `/opt/Automodel/nemo_automodel/components/training/step_scheduler.py:79,83,90,96` |

#### 6d. MoE Config Errors (ALL NEVER OCCUR)

| Automodel Error Raised | Validation Status | What It Means | Code Pointer |
|------------------------|-------------------|---------------|--------------|
| `ValueError(f"Invalid expert activation: {config.expert_activation}")` | `[NEVER OCCUR]` nhx-automodel uses model's default activation | MoE expert FFN activation function (gelu/relu/silu) is not supported | `/opt/Automodel/nemo_automodel/components/moe/layers.py:171,368` |
| `ValueError(f"{tensor_name} has shape {tensor.shape[0]} experts, expected {expected}")` | `[NEVER OCCUR]` nhx-automodel uses base model's expert count | Checkpoint expert count doesn't match model config (e.g., loading 8-expert weights into 4-expert model) | `/opt/Automodel/nemo_automodel/components/moe/state_dict_utils.py:181,187` |
| `ValueError("Two Different Datasets have the same corpus id...")` | `[NEVER OCCUR]` Retrieval dataset not used | Multiple datasets have same corpus ID but different paths | `/opt/Automodel/nemo_automodel/components/datasets/llm/retrieval_dataset.py:89` |

**User Message**: `Training configuration error: {details}. Please check your parallelism or PEFT settings.`

---

### 10. CheckpointError (500)

| Automodel Error Raised | Validation Status | What It Means | Code Pointer |
|------------------------|-------------------|---------------|--------------|
| `AssertionError(f"Checkpoint directory {path} already exists")` | `[ADD]` May occur at runtime | Checkpoint directory exists and overwrite is disabled | `/opt/Automodel/nemo_automodel/recipes/base_recipe.py:221` |
| `ValueError("Failed to validate global plan")` | `[ADD]` May occur at runtime | When saving/loading checkpoints across multiple GPUs, PyTorch creates a "global plan" that coordinates which GPU handles which model shards. This error occurs when the plan validation fails, typically due to: 1)Mismatch between GPU topology when saving vs loading (e.g., saved on 8 GPUs, loading on 4), 2)Corrupted checkpoint metadata 3)Inconsistent distributed state across ranks | `/opt/Automodel/nemo_automodel/components/checkpoint/_backports/default_planner.py:156` |
| `RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")` | `[ADD]` May occur at runtime | When loading in strict mode (default), every weight in the model must exist in the checkpoint. This error means the checkpoint is missing a weight the model expects, may be because of incomplete/corrupted checkpoint download | `/opt/Automodel/nemo_automodel/components/checkpoint/_backports/default_planner.py:462` |
| `RuntimeError(f"Expert weights missing from checkpoint...")` | `[ADD]` May occur at runtime | Specific to MoE (Mixture of Experts) models, the code validates that all expert weights exist. If any are missing, the checkpoint is likely corrupted or incomplete | `/opt/Automodel/nemo_automodel/components/moe/state_dict_mixin.py:105-110` |
| `AssertionError(f"Unsupported model save format: {format}")` | `[NEVER OCCUR]` nhx-automodel uses safetensors | Model save format not supported | `/opt/Automodel/nemo_automodel/components/checkpoint/checkpointing.py:101` |
| `Exception("Failed to write dataset materials to the data cache directory...")` | `[NEVER OCCUR]` Megatron dataset not used | Megatron dataset builder failed to write cache files due to disk full or permission issues | `/opt/Automodel/nemo_automodel/components/datasets/llm/megatron/builder.py:647` |

**User Message**: `Checkpoint error: {details}. The checkpoint may be corrupted or incompatible.`

---

### 11. CudaError (500)

| Automodel Error Raised | Validation Status | What It Means | Code Pointer |
|------------------------|-------------------|---------------|--------------|
| `torch.cuda.OutOfMemoryError` or "CUDA out of memory" | `[ADD]` May occur at runtime | GPU does not have enough memory for batch/model | Runtime |
| `RuntimeError` with "CUDA" in message | `[ADD]` May occur at runtime | General GPU error occurred | Runtime |

**User Message**: `GPU error: {details}. Try reducing batch_size or max_seq_length.`

---

### 12. DistributedError (500)

| Automodel Error Raised | Validation Status | What It Means | Code Pointer |
|------------------------|-------------------|---------------|--------------|
| `RuntimeError("torch.distributed not available")` | `[ADD]` Environment-dependent | PyTorch distributed package not installed or accessible | `/opt/Automodel/nemo_automodel/components/distributed/fsdp2.py:157` |
| `RuntimeError("expected torch.distributed to be initialized")` | `[ADD]` May occur at runtime | torch.distributed.init_process_group() was not called | `/opt/Automodel/nemo_automodel/components/distributed/fsdp2.py:160` |
| `TimeoutError` in distributed context | `[ADD]` May occur at runtime | PyTorch distributed ops (barrier/all_reduce) timed out waiting for workers - network issues or worker crash | PyTorch `torch.distributed` (not Automodel) |

**User Message**: `Distributed training error: {details}. Please try again.`

---

### 13. TrainingTimeoutError (500)

| Automodel Error Raised | Validation Status | What It Means | Code Pointer |
|------------------------|-------------------|---------------|--------------|
| `subprocess.TimeoutExpired` | `[ADD]` May occur at runtime | Training subprocess exceeded `training_timeout` from API config | nhx-automodel training runner (subprocess wait timeout; not Automodel) |

**User Message**: `Training exceeded time limit. Consider reducing training steps or increasing timeout.`

---

### 14. InternalError (500)

| Automodel Error Raised | Validation Status | What It Means | Code Pointer |
|------------------------|-------------------|---------------|--------------|
| `ValueError("You must provide either input_ids or inputs_embeds")` | `[ADD]` May occur at runtime | Pipeline stage has embed_tokens but received neither input_ids nor inputs_embeds | `/opt/Automodel/nemo_automodel/components/distributed/pipelining/hf_utils.py:47` |
| `ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")` | `[ADD]` May occur at runtime | Pipeline stage without embed_tokens layer didn't receive inputs_embeds from previous stage | `/opt/Automodel/nemo_automodel/components/distributed/pipelining/hf_utils.py:57` |
| `AssertionError("We only support 1D mesh for MoE")` | `[ADD]` May occur at runtime | MoE expert parallelism only supports 1D device mesh, got multi-dimensional mesh | `/opt/Automodel/nemo_automodel/components/moe/layers.py:245` |
| `ValueError(f"{tensor_name} has unsupported DTensor placement: {placement}. Expected Shard(dim=0) or Replicate for expert parallelism.")` | `[ADD]` May occur at runtime | DTensor has wrong placement type for expert parallelism - must be Shard(0) or Replicate | `/opt/Automodel/nemo_automodel/components/moe/state_dict_utils.py:196-198` |
| `ValueError("FusedLinearCrossEntropy requires the model to output hidden states. Set model.output_hidden_states=True in the config.")` | `[ADD]` May occur at runtime | Fused loss optimization requires hidden states output but model config doesn't enable it | `/opt/Automodel/nemo_automodel/recipes/llm/train_ft.py:1222` |
| `AssertionError("AutoPipeline configuration is required when pipeline parallelism is enabled")` | `[NEVER OCCUR]` nhx-automodel configures PP correctly | Pipeline parallelism requires autopipeline configuration | `/opt/Automodel/nemo_automodel/recipes/llm/train_ft.py:916-918` |
| `ImportError(f"Cannot resolve target (blocked or not found): {dotted_path}")` | `[NEVER OCCUR]` Config generated by nhx-automodel compiler | Config references a module/class that's blocked or doesn't exist | `/opt/Automodel/nemo_automodel/components/config/loader.py:246` |
| `ImportError("Access to private or dunder attributes is disabled by default. To allow out-of-tree code, set NEMO_ENABLE_USER_MODULES=1...")` | `[NEVER OCCUR]` Config generated by nhx-automodel compiler | Config tries to access private (_) or dunder (__) attributes - blocked for security | `/opt/Automodel/nemo_automodel/components/config/loader.py:210-213,234-237` |

**User Message**: `An internal error occurred: {details}.`

**More details on above InternalError errors:**

In Pipeline Parallelism (PP), the model is split across multiple GPUs where each GPU runs a "stage" of the model. For example:
GPU 0: Embedding layer + first few transformer layers
GPU 1: Middle transformer layers
GPU 2: Last transformer layers + LM head

- "You must provide either input_ids or inputs_embeds" -- This happens on the first stage (which has the embedding layer embed_tokens). The stage expects either raw token IDs (input_ids) to embed, OR pre-computed embeddings (inputs_embeds). If neither is provided, training can't proceed
- "inputs_embeds must be provided for pipeline stages without embed_tokens" -- This happens on middle/later stages that don't have the embedding layer. These stages expect to receive inputs_embeds (hidden states) from the previous stage. If the pipeline communication fails or is misconfigured, this stage won't receive the embeddings it needs


MoE (Mixture of Experts) models have expert parallelism (EP) where different experts are placed on different GPUs.

"We only support 1D mesh for MoE":
- PyTorch uses "device meshes" to organize GPUs for parallelism
- A 1D mesh is a simple linear arrangement: [GPU0, GPU1, GPU2, GPU3]
- A 2D mesh is a grid: [[GPU0, GPU1], [GPU2, GPU3]] (used for combining TP + DP)
- Automodel's MoE implementation currently only supports 1D mesh for expert parallelism
- If you try to use MoE with a 2D mesh (e.g., combining EP with other parallelism in a complex way), this error occurs