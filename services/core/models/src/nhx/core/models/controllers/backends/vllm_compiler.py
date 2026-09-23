# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-agnostic vLLM compiler.

Turns a ``ModelDeploymentConfig`` whose ``engine`` is ``vllm`` into the image,
``vllm serve`` argument vector, and environment variables for a vLLM server.

These functions take a :class:`DeploymentConfigView` and a ``ModelEntity`` and
return plain data (arg vectors, env dicts, image tuples, TP sizing) -- they are
NOT specific to any service backend. The docker backend renders the result into
a ``docker run`` container; the k8s backend renders it into native Kubernetes
objects. Keep this module free of backend-specific imports so both can reuse it.
"""

from logging import getLogger
from typing import Optional

from nemo_helix_plugin.models.types import ModelEntity
from nhx.core.models.controllers.backends.common import DeploymentConfigView

logger = getLogger(__name__)

# Path the model weights are mounted at inside the container (populated by the puller).
MODEL_STORE_PATH = "/model-store"
# Automodel embedding outputs keep NIM's ONNX artifact at the root and the
# equivalent Hugging Face checkpoint in this engine-specific alternate.
VLLM_HF_ALTERNATE_PATH = f"{MODEL_STORE_PATH}/alternates/hf"
# Directory the LoRA adapter sidecar writes into and vLLM's filesystem resolver watches.
VLLM_LORA_CACHE_DIR = "/scratch/loras"
# vLLM's OpenAI-compatible server port (matches the NIM container port mapping).
VLLM_SERVER_PORT = 8000


def _served_model_name(view: DeploymentConfigView) -> str | None:
    """The OpenAI-compatible served model name (namespace/name or name).

    Returns ``None`` when no model name is set — this is not an error. The caller
    only injects ``--served-model-name`` when a name exists; otherwise vLLM falls
    back to its own default served name (e.g. for a generic container with no
    model identity).
    """
    if not view.model_name:
        return None
    if view.model_namespace:
        return f"{view.model_namespace}/{view.model_name}"
    return view.model_name


def _model_path(model_entity: Optional[ModelEntity]) -> str:
    """Select the Hugging Face alternate for customized embedding outputs."""
    if model_entity is None:
        return MODEL_STORE_PATH
    spec = getattr(model_entity, "spec", None)
    head_type = getattr(spec, "head_type", None)
    is_embedding = head_type == "embedding" or (
        head_type in (None, "unknown") and getattr(spec, "is_embedding_model", False)
    )
    if is_embedding and getattr(model_entity, "base_model", None) and getattr(model_entity, "finetuning_type", None):
        return VLLM_HF_ALTERNATE_PATH
    return MODEL_STORE_PATH


def _user_set_args(additional_args: list[str] | None) -> set[str]:
    """Flag names the user already supplied via additional_args (so we don't inject ours)."""
    flags: set[str] = set()
    for arg in additional_args or []:
        if isinstance(arg, str) and arg.startswith("--"):
            # Handle both "--flag value" and "--flag=value".
            flags.add(arg.split("=", 1)[0])
    return flags


def _divisors(n: int) -> list[int]:
    """All positive divisors of ``n`` in ascending order."""
    return [d for d in range(1, n + 1) if n % d == 0]


def compute_tensor_parallel_size(
    gpu: int,
    model_entity: Optional[ModelEntity],
    *,
    max_tp: int | None = None,
) -> int:
    """Compute ``--tensor-parallel-size`` from GPU count and model architecture.

    Mirrors the constraints the NeMo parallelism heuristic applies: a valid TP
    degree must divide the GPU count and the model's ``hidden_size``,
    ``num_attention_heads``, and ``num_kv_heads`` (critical for GQA/MQA). We pick
    the largest valid divisor so every allocated GPU is used. When ``gpu <= 1``
    (including CPU-only) TP is 1 and no computation runs.

    This intentionally re-implements the small divisor/divisibility check inline
    rather than importing ``nhx.core.models.parallelism`` -- that package pulls in
    torch/transformers, which are not installed in the models controller runtime.

    Falls back to ``1`` when the model entity spec lacks the architecture fields
    the constraints need.
    """
    if gpu <= 1:
        return 1

    spec = getattr(model_entity, "spec", None)
    hidden_size = getattr(spec, "hidden_size", None)
    num_heads = getattr(spec, "num_attention_heads", None)
    num_kv_heads = getattr(spec, "num_kv_heads", None)
    if spec is None or hidden_size is None or num_heads is None:
        logger.info(
            "Model entity spec missing architecture fields for TP auto-tune; defaulting tensor-parallel-size to 1"
        )
        return 1

    # num_kv_heads may be absent for non-GQA models; fall back to num_heads.
    if num_kv_heads is None:
        num_kv_heads = num_heads

    tp_cap = max_tp or gpu
    candidates = [
        tp
        for tp in _divisors(gpu)
        if tp <= tp_cap and hidden_size % tp == 0 and num_heads % tp == 0 and num_kv_heads % tp == 0
    ]
    if not candidates:
        logger.warning("No valid tensor-parallel-size candidates for model; defaulting to 1")
        return 1
    tp = max(candidates)
    logger.info("Computed tensor-parallel-size=%s for %s GPU(s)", tp, gpu)
    return tp


def compile_vllm_args(
    view: DeploymentConfigView,
    model_entity: Optional[ModelEntity],
) -> list[str]:
    """Build the ``vllm serve`` argument vector for the container.

    The image's entrypoint is ``vllm serve``; these are the args appended to it.
    ``additional_args`` are appended verbatim, and any flag the user set there is
    not auto-injected by the compiler.
    """
    user_flags = _user_set_args(view.additional_args)
    args: list[str] = [_model_path(model_entity)]

    served_name = _served_model_name(view)
    if served_name and "--served-model-name" not in user_flags:
        args.extend(["--served-model-name", served_name])

    if view.gpu >= 1 and "--tensor-parallel-size" not in user_flags:
        tp = compute_tensor_parallel_size(view.gpu, model_entity)
        if tp > 1:
            args.extend(["--tensor-parallel-size", str(tp)])

    if view.chat_template and "--chat-template" not in user_flags:
        args.extend(["--chat-template", view.chat_template])

    if view.lora_enabled and "--enable-lora" not in user_flags:
        args.append("--enable-lora")

    # vLLM has no env var for trust-remote-code; it is a `vllm serve` CLI flag
    # (EngineArgs.trust_remote_code -> --trust-remote-code).
    if model_entity and getattr(model_entity, "trust_remote_code", False) and "--trust-remote-code" not in user_flags:
        args.append("--trust-remote-code")

    # User-supplied raw args appended verbatim (highest precedence; e.g. --max-lora-rank).
    args.extend(view.additional_args or [])
    return args


def compile_vllm_env_vars(
    view: DeploymentConfigView,
) -> dict[str, str]:
    """Build the environment variables for the vLLM container."""
    env_vars: dict[str, str] = {}

    if view.lora_enabled:
        # LoRA hot-reload via the filesystem resolver plugin. The adapter sidecar
        # writes adapters into VLLM_LORA_RESOLVER_CACHE_DIR; vLLM lazy-loads on first
        # request. Mirrors the NIM sidecar contract (NIM_PEFT_SOURCE), differing only
        # in the directory env var.
        env_vars["VLLM_PLUGINS"] = "lora_filesystem_resolver"
        env_vars["VLLM_LORA_RESOLVER_CACHE_DIR"] = VLLM_LORA_CACHE_DIR
        env_vars["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    # Escape hatch: extra env vars (highest precedence).
    if view.additional_envs:
        env_vars.update({str(k): str(v) for k, v in view.additional_envs.items()})

    return env_vars


def resolve_vllm_image(view: DeploymentConfigView, default_image: str, default_tag: str) -> tuple[str, str]:
    """Resolve the vLLM image name and tag, falling back to the configured defaults."""
    image_name = view.image_name or default_image
    image_tag = view.image_tag or default_tag
    return image_name, image_tag
