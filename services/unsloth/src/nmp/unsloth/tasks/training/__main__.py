# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training container entrypoint.

Reads a :class:`~nmp.unsloth.app.jobs.training.schemas.TrainingStepConfig`
from the platform Jobs envelope (``NEMO_JOB_STEP_CONFIG_FILE_PATH``)
and runs :func:`~nmp.unsloth.tasks.training.backends.unsloth_sft.train_sft`
against the paths the file_io download step populated.

The container ENTRYPOINT bakes::

    ENTRYPOINT ["/opt/venv/bin/python"]
    CMD ["-m", "nmp.unsloth.tasks.training"]

Heavy ML imports (``unsloth``, ``torch``, ``transformers``) are
deferred to ``train_sft`` so the parent process (and pytest collection
on a CPU box without unsloth) can import this module.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _read_step_config() -> dict | None:
    """Load the JSON step config the platform Jobs runner injects."""
    env_var = "NEMO_JOB_STEP_CONFIG_FILE_PATH"
    config_path = os.environ.get(env_var)
    if not config_path:
        logger.error(
            f"{env_var} is not set. The training container expects the platform Jobs "
            "runner to mount the step config file path."
        )
        return None
    path = Path(config_path)
    if not path.is_file():
        logger.error(f"Step config file does not exist: {path}")
        return None
    return json.loads(path.read_text())


def main() -> int:
    """Run the unsloth training step inside a submit container."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    raw = _read_step_config()
    if raw is None:
        sys.stderr.write(
            "nmp.unsloth.tasks.training requires NEMO_JOB_STEP_CONFIG_FILE_PATH. "
            "Submit via `nemo customization unsloth submit <job.json>` so the "
            "platform Jobs runner populates this for you.\n",
        )
        return 2

    # Local imports so the parent process (e.g. CLI discovery, pytest
    # collection) does not pay the ML import cost.
    from nemo_platform_plugin.job_context import JobContext, StoragePaths
    from nmp.customization_common.service.path_utils import get_job_storage_path_from_env, remap_job_storage_path
    from nmp.unsloth.app.jobs.training.schemas import TrainingStepConfig
    from nmp.unsloth.tasks.training.backends.unsloth_sft import train_sft

    config = TrainingStepConfig.model_validate(raw)
    spec = config.spec

    persistent_root = get_job_storage_path_from_env()
    storage = StoragePaths(
        ephemeral=persistent_root / "ephemeral",
        persistent=persistent_root,
    )
    storage.ephemeral.mkdir(parents=True, exist_ok=True)
    storage.persistent.mkdir(parents=True, exist_ok=True)

    # Container runs don't currently publish ``ctx.results``; ``train_sft``
    # doesn't touch it today, so passing ``None`` is safe.
    ctx = JobContext(
        workspace=os.environ.get("NEMO_JOB_WORKSPACE", "default"),
        storage=storage,
        results=None,
        job_id=os.environ.get("NEMO_JOB_ID"),
    )

    if spec.hardware.gpus is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", spec.hardware.gpus)
        logger.info(f"Container: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Unsloth compiles patched modules into a cache dir that defaults to
    # ``unsloth_compiled_cache`` relative to the CWD. The container's WORKDIR
    # (/app) is root-owned and we run as a non-root user, so that write fails
    # (it falls back to a temp dir, but logs an error and loses cache reuse).
    # Point the compile cache and HF cache at the job's writable ephemeral
    # storage. ``unsloth_zoo`` reads ``UNSLOTH_COMPILE_LOCATION`` at import
    # time, so this must run before ``train_sft`` triggers ``import unsloth``.
    compile_cache = storage.ephemeral / "unsloth_compiled_cache"
    compile_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", str(compile_cache))
    hf_home = storage.ephemeral / "hf"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    logger.info(
        f"Container: UNSLOTH_COMPILE_LOCATION={os.environ['UNSLOTH_COMPILE_LOCATION']} HF_HOME={os.environ['HF_HOME']}"
    )

    def _storage_path(value: str | None) -> str | None:
        if value is None:
            return None
        return str(remap_job_storage_path(persistent_root, value))

    try:
        result = train_sft(
            spec,
            ctx,
            model_path=_storage_path(config.model_path),
            dataset_path=_storage_path(config.dataset_path),
            validation_path=_storage_path(config.validation_path),
            output_path=_storage_path(config.output_path),
        )
    except Exception:
        logger.exception("Unsloth training step failed")
        return 1

    logger.info(f"Training step completed: {json.dumps(result, default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
