# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every rule in rules.yaml, checked against lines that training prints."""

import pytest
from nhx.customization_common.training.errors.classifier import FailureClassifier, get_classifier

# Lines that each rule must classify, as they appear in training output.
EXAMPLES: dict[str, list[str]] = {
    "training_timeout": ["subprocess.TimeoutExpired: Command '['torchrun']' timed out after 3600 seconds"],
    "host_out_of_memory": [
        "ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.",
        "(raylet) Worker exit detail: The process is killed by SIGKILL by OOM killer due to high memory usage.",
        "RuntimeError: [enforce fail at alloc_cpu.cpp:117] DefaultCPUAllocator: can't allocate memory: you tried to allocate 8GB",
        "MemoryError: ",
    ],
    "shared_memory_exhausted": [
        "RuntimeError: unable to allocate shared memory(shm) for file </torch_20843_1057569755_1664>: No space left on device (28)",
        "RuntimeError: unable to write to file </torch_123_456_7>: No space left on device (28)",
        "RuntimeError: DataLoader worker (pid 12) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory.",
        "misc/shmutils.cc:72 NCCL WARN Error while creating shared memory segment /dev/shm/nccl-Ab12 (size 9637888)",
        "ray.exceptions.ObjectStoreFullError: Failed to put object abc in object store because it is full.",
    ],
    "disk_full": [
        "OSError: [Errno 28] No space left on device: '/var/run/scratch/job/training/checkpoints/step_40'",
        "OSError: [Errno 122] Disk quota exceeded",
    ],
    "gpu_out_of_memory": [
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 79.25 GiB",
        "RuntimeError: CUDA error: out of memory",
        "torch.distributed.DistBackendError: NCCL error in: ProcessGroupNCCL.cpp:3866, unhandled cuda error, NCCL version 2.27.5 Last error: Cuda failure 'out of memory'",
        "(VllmWorker pid=9) CUDA out of memory. Tried to allocate 1.00 GiB",
    ],
    "vllm_kv_cache_too_small": [
        "ValueError: To serve at least one request with the model's max seq len (8192), (4.5 GiB KV cache is needed, which is larger than the available KV cache memory (2.1 GiB).",
        "ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.",
        "ValueError: Free memory on device (10.0/79.2 GiB) on startup is less than desired GPU memory utilization (0.6, 47.5 GiB).",
    ],
    "invalid_dataset": [
        "DatasetFormatError: Dataset path does not exist: /var/run/scratch/job/dataset",
        "TypeError: Dataset Schema Validation failed: missing 'chosen_response'",
    ],
    "unsupported_chat_role": ["ValueError: Unsupported role in messages: narrator"],
    "invalid_chat_dataset": [
        "ValueError: Each sample must contain a `messages` list (OpenAI format) or a `conversations` list",
        "AssertionError: At least one assistant message is required when answer_only_loss_mask=True",
        "ValueError: System role appeared in multiple messages. Only a single system message is supported.",
        "ValueError: assistant message `tool_calls[0].function.name` must be a non-empty string",
        "ValueError: `tools` must be a list or JSON-encoded list, got int",
    ],
    "invalid_json_dataset": ["pyarrow.lib.ArrowInvalid: JSON parse error: Invalid encoding in string. in row 20"],
    "dataset_generation_failed": [
        "datasets.exceptions.DatasetGenerationError: An error occurred while generating the dataset"
    ],
    "invalid_column_dataset": ["AssertionError: Expected at least two columns, got 1"],
    "answer_tokens_truncated": [
        "ValueError: All samples in the dataset produced labels that are entirely -100 (masked)."
    ],
    "invalid_preference_dataset": [
        "NotImplementedError: Ties are not supported yet.",
        "AssertionError: RM/DPO training supports only two completions",
    ],
    "invalid_retrieval_dataset": [
        "ValueError: Inline retrieval record must include 'query'",
        "ValueError: Recipe 'bi_encoder' requires a retrieval dataset, but the dataset is in 'chat' format.",
    ],
    "chat_template_missing": [
        "ValueError: ChatDataset requires a tokenizer with chat template support.",
        "ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!",
    ],
    "chat_template_error": [
        "jinja2.exceptions.TemplateError: Conversation roles must alternate user/assistant/user/assistant/...",
        "ValueError: Cannot build an answer-only loss mask from conversation prefixes for this template",
    ],
    "model_not_found": [
        "FileNotFoundError: Model path /var/run/scratch/job/model does not exist",
        "OSError: /model does not appear to have a file named model.safetensors.",
        "OSError: You are trying to access a gated repo.",
    ],
    "unsupported_model_architecture": [
        "ValueError: The checkpoint you are trying to load has model type `foo` but Transformers does not recognize this architecture.",
        "ValueError: Unsupported configuration for Qwen3MoeForCausalLM:",
    ],
    "model_weights_mismatch": [
        "RuntimeError: _apply(): Couldn't swap Linear.weight",
        "ValueError: Size mismatch between saved torch.Size([10]) and current: torch.Size([20]) for lm_head.weight",
    ],
    "tokenizer_error": ["IndexError: piece id is out of range."],
    "training_libraries_missing": [
        "ImportError: vLLM is not installed. Please check that the py_executable in the runtime_env is correct.",
        "AssertionError: Flash Attention is not available",
    ],
    "no_trainable_parameters": ["ValueError: optimizer received no trainable parameters"],
    "pipeline_parallel_config": [
        "ValueError: Model 'qwen' is not compatible with pipeline parallelism:",
        "AssertionError: pp_batch_size 2 // pp_microbatch_size 1 must be >= pp_size 4",
    ],
    "tensor_parallel_config": [
        "AssertionError: num_key_value_heads (2) must be divisible by TP size (4)",
        "ValueError: World size (4) is insufficient for the parallelism configuration",
    ],
    "context_parallel_config": [
        "ValueError: Model does not support SDPA required for context parallelism",
        "AssertionError: Context parallel is not supported for Gemma3ForCausalLM.",
        "ValueError: batching_strategy='sequence_packing' is not supported with context_parallel_size (2) > 1. Use 'dynamic' or 'static'.",
    ],
    "moe_parallel_config": [
        "ValueError: Tensor parallelism (tensor_parallel_size=2) is not supported for MoE models.",
        "AssertionError: We only support 1D mesh for MoE",
    ],
    "sequence_packing_config": [
        "ValueError: Dataset sample is too long (9000 > 8192). Please increase `packed_sequence_size`.",
        "ValueError: Cannot create 8 bins with only 3 sequences",
        "ValueError: train_mb_tokens (4096) is below max_seq_length (8192); a full-length rollout would not fit in any micro-batch.",
    ],
    "max_model_len_too_long": [
        "ValueError: User-specified max_model_len (65536) is greater than the derived max_model_len (max_position_embeddings=32768)."
    ],
    "grpo_sampling_config": ["AssertionError: Top k is not supported in the sampling params in NeMo-Gym path!"],
    "knowledge_distillation_config": [
        "ValueError: Knowledge distillation requires training.kd.teacher to be set. Ensure the job input includes a teacher model."
    ],
    "unsupported_training_option": [
        "ValueError: Unsupported optimizer_name 'lion'; expected 'auto' or one of ['adam', 'adamw'].",
        "RuntimeError: Unsloth backend only supports training_type='sft'",
    ],
    "grpo_dynamic_sampling_exhausted": [
        "ValueError: Dynamic sampling found no prompt group with non-zero reward standard deviation before the dataloader ran out.",
        "ValueError: Dynamic sampling has reached the maximum allowed number of batches (10).",
    ],
    "nemo_rl_config_rejected": ["pydantic_core._pydantic_core.ValidationError: 1 validation error for MasterConfig"],
    "invalid_environment_package": [
        "nhx.rl.tasks.environment.validate.EnvironmentPackageValidationError: Missing nemo-environment.yaml at environment root",
        "RuntimeError: Gym environment package is missing required manifest: /job/environment/nemo-environment.yaml",
    ],
    "invalid_gym_config": [
        "nemo_gym.config_types.ConfigPathNotFoundError: config_paths entry 'x.yaml' was not found",
        "ValueError: Rows reference agents not present in the running config: ['math_agent']",
    ],
    "gym_not_configured": [
        "ValueError: Sandboxed GRPO requires a Gym runtime image; gym.gym_runtime_image is unset.",
        "ValueError: GRPO jobs require gym configuration on TrainingStepConfig",
    ],
    "gym_host_failed": [
        "sandboxed_gym.host.models.GymHostBootstrapFailed: job host gym-1 failed to start: image pull error"
    ],
    "gym_rollout_unavailable": [
        "nemo_rl.experience.failures.GymTransportError: NeMo-Gym /run failed with HTTP 503: unavailable",
        "TimeoutError: job host gym-1 did not become ready within 600s",
    ],
    "gym_process_exited": ["RuntimeError: Process `math_resources_server` finished unexpectedly!"],
    "no_checkpoint_saved": [
        "ValueError: Training finished without saving a checkpoint.",
        "ValueError: No best checkpoint found",
        "FileNotFoundError: No epoch_step checkpoint directories found in /var/run/scratch/job/training/checkpoints",
    ],
    "resume_checkpoint_missing": [
        "nhx.customization_common.training.pause.ResumeCheckpointMissingError: Cannot resume: training paused at step 9, but no complete checkpoint was found in /var/run/scratch/job/training/checkpoints."
    ],
    "checkpoint_already_published": [
        "FileExistsError: Checkpoint directory /var/run/scratch/job/training/checkpoints/epoch_1_step_99 already holds a published checkpoint. Remove it, or point checkpoint.checkpoint_dir at a fresh directory, before re-running this step."
    ],
    "checkpoint_incompatible": [
        "RuntimeError: Missing key in checkpoint state_dict: model.layers.0.weight.",
        "ValueError: Cannot resume optimizer master weights: precision changed",
    ],
    "checkpoint_publish_failed": ["ValueError: Best checkpoint not found at /c/step_20"],
    "checkpoint_storage_failed": [
        "RuntimeError: Failed to publish checkpoint epoch_0_step_19; see the coordinator log for the underlying error.",
        "RuntimeError: Background checkpoint finalization failed",
    ],
    "generation_refit_failed": [
        "RuntimeError: ❌ Error: Updating weights for the generation policy failed during refit. This often indicates an issue with cuda-ipc"
    ],
    "generation_logprobs_nan": ["RuntimeError: Generation logprobs contain NaN"],
    "generation_placement_failed": ["ValueError: No placement groups available in the cluster"],
    "generation_internal_error": [
        "ValueError: Missing required keys for GenerationOutputSpec: {'logprobs'}",
        "RuntimeError: NeMo-Gym completed without producing a final prompt group",
    ],
    "generation_engine_failed": [
        "RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}",
        "vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.",
    ],
    "cuda_device_assert": ["RuntimeError: CUDA error: device-side assert triggered"],
    "cuda_error": [
        "RuntimeError: CUDA error: an illegal memory access was encountered",
        "RuntimeError: CUDA error: uncorrectable ECC error encountered",
    ],
    "ray_agent_port_conflict": [
        "RuntimeError: Ray agent failed to start because its port is already in use",
    ],
    "ray_agent_grpcio_mismatch": [
        "RuntimeError: Ray agent crashed during startup because grpcio does not match Ray's requirement",
    ],
    "ray_agent_killed": [
        "RuntimeError: Ray agent was killed by the operating system during startup",
    ],
    "ray_agent_startup_error": [
        "RuntimeError: Ray agent failed during startup: ValueError: worker failed to register",
    ],
    "ray_agent_fate_share": [
        'reason_message: "runtime_env_agent failed and raylet fate-shares with it."',
        'reason_message: "dashboard_agent failed and raylet fate-shares with it."',
        "RuntimeError: Ray agent failed and the raylet shut down with it",
    ],
    "cluster_resources_unavailable": [
        "ResourceInsufficientError: Maximum number of retries reached (6). Cluster resources may be insufficient.",
        "TimeoutError: Timed out waiting for all workers to connect after 600s.",
        "RuntimeError: Failed to start Ray head node",
    ],
    "ray_node_failed": ["ray.exceptions.ActorDiedError: The actor is dead because its node has died."],
    "ray_actor_died": [
        "ray.exceptions.ActorDiedError: The actor died unexpectedly before finishing this task.",
        "ray.exceptions.WorkerCrashedError: The worker died unexpectedly while executing this task.",
        "ray.exceptions.ActorUnavailableError: The actor is temporarily unavailable: RpcError: connection lost",
        "ray.exceptions.RaySystemError: System error: Failed to unpickle serialized exception",
    ],
    "nccl_error": [
        "torch.distributed.DistBackendError: NCCL error in: ProcessGroupNCCL.cpp:3866, unhandled system error, NCCL version 2.27.5"
    ],
    "distributed_connection_failed": [
        "torch.distributed.DistNetworkError: The client socket has failed to connect to [leader]:23456.",
        "torch.distributed.DistStoreError: Timed out after 901 seconds waiting for clients. 1/2 clients joined.",
    ],
    "distributed_timeout": [
        "TimeoutError: Timeout waiting for coordinator at barrier 'training_complete' after 600s",
        "RuntimeError: ENDED file detected during worker wait",
    ],
    "process_killed": [
        "RuntimeError: failed (exitcode: -9) local_rank: 0 (pid: 1057) of binary: /opt/venv/bin/python",
        "  traceback : Signal 9 (SIGKILL) received by PID 1061",
    ],
    "internal_tensor_error": ["ValueError: You must provide either input_ids or inputs_embeds"],
    "training_exit_code": ["RuntimeError: Training failed with exit code: 1"],
}


@pytest.fixture(scope="module")
def classifier() -> FailureClassifier:
    return get_classifier()


def test_every_rule_has_examples(classifier):
    assert {rule.id for rule in classifier.rules} == set(EXAMPLES)


@pytest.mark.parametrize(
    "rule_id,line",
    [(rule_id, line) for rule_id, lines in EXAMPLES.items() for line in lines],
)
def test_rule_classifies_its_examples(classifier, rule_id, line):
    error = classifier.classify_output([line], 1)

    assert error.cause == rule_id
    rule = next(rule for rule in classifier.rules if rule.id == rule_id)
    assert error.rerunnable == rule.rerunnable
    assert isinstance(error, rule.exception)


# Real failures from test runs. The cause is not the last line printed.


def test_rank_zero_oom_wins_over_the_peers_nccl_errors(classifier):
    lines = [
        "[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.",
        "[rank1]: torch.distributed.DistBackendError: NCCL error in: ProcessGroupNCCL.cpp:3866, remote process exited",
        "[rank1]: Watchdog caught collective operation timeout: WorkNCCL(SeqNum=12) ran for 600000 milliseconds",
    ]

    error = classifier.classify_output(lines, 1)

    assert error.cause == "gpu_out_of_memory"
    assert error.rerunnable is False


def test_published_checkpoint_conflict_wins_over_native_backtrace_and_peer_errors(classifier):
    lines = [
        "[rank1]:     raise FileExistsError(",
        "[rank1]: FileExistsError: Checkpoint directory /c/epoch_1_step_99 already holds a published checkpoint.",
        "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: ",
        "frame #2: c10::DistNetworkError::DistNetworkError(c10::SourceLocation, std::string) + 0x63",
        "[rank0]: torch.distributed.DistNetworkError: Connection reset by peer",
    ]

    assert classifier.classify_output(lines, 1).cause == "checkpoint_already_published"


def test_shared_memory_is_not_reported_as_a_full_disk(classifier):
    lines = [
        "Saving checkpoint for step 40...",
        "RuntimeError: unable to allocate shared memory(shm) for file </torch_20843_1057569755_1664>: No space left on device (28)",
        "2026-10-20 05:27:00 - nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap - INFO - Driver completed with exit code: 1",
    ]

    assert classifier.classify_output(lines, 1).cause == "shared_memory_exhausted"


def test_ray_host_oom_is_not_a_gpu_oom(classifier):
    lines = [
        "ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.",
        "ray.exceptions.ActorDiedError: The actor died unexpectedly before finishing this task.",
    ]

    error = classifier.classify_output(lines, 1)

    assert error.cause == "host_out_of_memory"


def test_vllm_root_cause_wins_over_the_engine_wrapper(classifier):
    lines = [
        "(EngineCore pid=88) ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`.",
        "RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}",
    ]

    assert classifier.classify_output(lines, 1).cause == "vllm_kv_cache_too_small"


def test_a_storage_error_wins_over_the_checkpoint_wrapper_after_it(classifier):
    lines = [
        "OSError: [Errno 28] No space left on device: '/var/run/scratch/job/training/checkpoints/epoch_0_step_19'",
        "RuntimeError: Failed to publish checkpoint epoch_0_step_19; see the coordinator log for the underlying error.",
    ]

    assert classifier.classify_output(lines, 1).cause == "disk_full"


def test_a_training_oom_wins_over_the_raylet_fate_share_notice(classifier):
    lines = [
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.",
        'reason_message: "runtime_env_agent failed and raylet fate-shares with it."',
    ]

    assert classifier.classify_output(lines, 1).cause == "gpu_out_of_memory"


def test_a_ray_port_conflict_wins_over_the_head_failing_to_start(classifier):
    lines = [
        "RuntimeError: Failed to start Ray head node",
        "RuntimeError: Ray agent failed to start because its port is already in use",
    ]

    assert classifier.classify_output(lines, 1).cause == "ray_agent_port_conflict"


def test_the_ray_hint_that_lists_every_cause_is_only_the_fate_share_notice(classifier):
    # Ray prints all three causes in one paragraph whenever the agent dies. That paragraph is not
    # itself a grpcio mismatch, a port conflict, or an out-of-memory kill.
    lines = [
        "The raylet fate shares with the agent. This can happen because",
        "- The version of `grpcio` doesn't follow Ray's requirement. Agent can segfault with the incorrect `grpcio` version.",
        "- The agent failed to start because of unexpected error or port conflict.",
        "- The agent is killed by the OS (e.g., out of memory).",
        'reason_message: "runtime_env_agent failed and raylet fate-shares with it."',
    ]

    assert classifier.classify_output(lines, 1).cause == "ray_agent_fate_share"


def test_a_gpu_oom_wins_over_the_worker_crash_after_it(classifier):
    lines = [
        "(DTensorPolicyWorker pid=412) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.",
        "ray.exceptions.WorkerCrashedError: The worker died unexpectedly while executing this task.",
    ]

    error = classifier.classify_output(lines, 1)

    assert error.cause == "gpu_out_of_memory"
    assert error.rerunnable is False


def test_a_gpu_oom_wins_over_the_refit_failure_after_it(classifier):
    lines = [
        "(VllmGenerationWorker pid=97) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.00 GiB.",
        "RuntimeError: ❌ Error: Updating weights for the generation policy failed during refit.",
    ]

    assert classifier.classify_output(lines, 1).cause == "gpu_out_of_memory"


def test_a_full_disk_wins_over_the_dataset_wrapper_after_it(classifier):
    lines = [
        "OSError: [Errno 28] No space left on device: '/root/.cache/huggingface/datasets/json/default'",
        "datasets.exceptions.DatasetGenerationError: An error occurred while generating the dataset",
    ]

    assert classifier.classify_output(lines, 1).cause == "disk_full"


def test_a_json_parse_error_is_reported_with_its_row_not_the_dataset_wrapper(classifier):
    lines = [
        "pyarrow.lib.ArrowInvalid: JSON parse error: Invalid encoding in string. in row 20",
        "datasets.exceptions.DatasetGenerationError: An error occurred while generating the dataset",
    ]

    error = classifier.classify_output(lines, 1)

    assert error.cause == "invalid_json_dataset"
    assert "in row 20" in error.message
    assert error.rerunnable is False


def test_an_in_process_dataset_wrapper_is_classified_by_the_error_it_wraps(classifier):
    class DatasetGenerationError(Exception):
        pass

    try:
        try:
            raise OSError(28, "No space left on device")
        except OSError as disk_error:
            raise DatasetGenerationError("An error occurred while generating the dataset") from disk_error
    except DatasetGenerationError as e:
        error = classifier.classify(e)

    assert error.cause == "disk_full"


def test_a_killed_torchrun_worker_is_not_reported_as_nccl(classifier):
    # torchrun prints no exception for a worker killed by a signal, only its root cause block.
    lines = [
        "2026-10-20 23:58:09 | INFO | root | Backend: nccl",
        "  backend: nccl",
        "INFO | root | step 31 | epoch 0 | loss 1.6779 | grad_norm 4.5590 | lr 8.51e-05 |",
        "E1020 23:59:41.295000 990 torch/distributed/elastic/multiprocessing/api.py:1002] failed (exitcode: -9) "
        "local_rank: 0 (pid: 1061) of binary: /opt/venv/bin/python",
        "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: ",
        "nhx.automodel.tasks.training.backends.finetune FAILED",
        "  exitcode  : -9 (pid: 1061)  (SIGKILL)",
        "  traceback : Signal 9 (SIGKILL) received by PID 1061",
    ]

    error = classifier.classify_output(lines, 1)

    assert error.cause == "process_killed"
    assert error.rerunnable is True


def test_our_dataset_errors_are_not_rerunnable(classifier):
    # Each service defines its own DatasetFormatError, a plain Exception raised in the runner process.
    class DatasetFormatError(Exception):
        pass

    error = classifier.classify(
        DatasetFormatError("Training file train.jsonl has 1 sample(s); at least 2 are required")
    )

    assert error.cause == "invalid_dataset"
    assert error.rerunnable is False
    assert "at least 2 are required" in error.message


def test_an_in_process_oom_is_not_rerunnable(classifier):
    class OutOfMemoryError(RuntimeError):
        pass

    error = classifier.classify(OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB"))

    assert error.cause == "gpu_out_of_memory"
    assert error.rerunnable is False
