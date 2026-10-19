# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from nhx.common.jobs.constants import NEMO_JOB_ID_ENVVAR

from .schemas import GPUInfo

logger = logging.getLogger(__name__)


def _get_architecture_name(major: int, minor: int) -> str:
    """Map CUDA compute capability to architecture name.

    https://developer.nvidia.com/cuda-gpus
    """
    if major == 3:
        return "Kepler"
    if major == 5:
        return "Maxwell"
    if major == 6:
        return "Pascal"
    if major == 7:
        # 7.0/7.2 = Volta, 7.5 = Turing
        if minor >= 5:
            return "Turing"
        return "Volta"
    if major == 8:
        return "Ampere"
    if major == 9:
        return "Hopper"
    if major == 10:
        return "Blackwell"
    return f"Unknown (sm_{major}{minor})"


def get_gpu_info() -> GPUInfo | None:
    """Capture GPU architecture information."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        device_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_id)
        major, minor = torch.cuda.get_device_capability(device_id)

        return GPUInfo(
            architecture=_get_architecture_name(major, minor),
            device_name=props.name,
            memory_gb=props.total_memory / (1024**3),
            cuda_version=str(torch.version.cuda),
        )
    except Exception as e:
        logger.warning(f"Failed to capture GPU info: {e}")
        return None


def generate_torchrun_flags_from_env() -> list[str]:
    """Generate torchrun flags for distributed training."""
    # These values are typically injected by the Volcano/PyTorch operator
    # or the Core Jobs Service when using DistributedGPUExecutionProvider.
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "23456")  # Default to port from volcano_job.py
    node_rank = os.environ.get("NODE_RANK", os.environ.get("RANK", "0"))
    num_nodes = os.environ.get("WORLD_SIZE", "1")
    gpus_per_node = os.environ.get("GPUS_PER_NODE")
    if gpus_per_node is None:
        try:
            import torch

            gpus_per_node = str(torch.cuda.device_count())
        except Exception as e:
            logger.warning(f"Failed to determine number of GPUs: {e}, using default of 1")
            gpus_per_node = "1"

    return [
        "--nnodes",
        num_nodes,
        "--nproc_per_node",
        gpus_per_node,
        "--node_rank",
        node_rank,
        "--rdzv_id",
        os.environ.get(NEMO_JOB_ID_ENVVAR, "customizer-rdzv"),
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        f"{master_addr}:{master_port}",
    ]
