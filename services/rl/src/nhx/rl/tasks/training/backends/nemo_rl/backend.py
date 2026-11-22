# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""TrainingBackend protocol implementation for NeMo RL (DPO)."""

import logging
import os
import signal
import time
from pathlib import Path
from typing import Any, Optional, cast

from nemo_helix_plugin.jobs.constants import PAUSE_EXIT_CODE
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nhx.customization_common.schemas.values import TrainingPhase
from nhx.customization_common.service.context import NHXJobContext
from nhx.customization_common.training.errors.classifier import classify_output
from nhx.customization_common.training.lora import merge_lora_adapter
from nhx.customization_common.training.nccl import get_nccl_ib_env
from nhx.customization_common.training.progress import JobsServiceProgressReporter
from nhx.rl.app.jobs.training.schemas import (
    CheckpointFormat,
    CheckpointInfo,
    TrainingMetrics,
    TrainingStepConfig,
    TrainingType,
)
from nhx.rl.app.jobs.training.schemas import (
    TrainingBackend as TrainingBackendEnum,
)
from nhx.rl.entities.values import FinetuningType
from nhx.rl.tasks.training.backends.nemo_rl.checkpoints import (
    LORA_ADAPTER_SEARCH_PATHS,
    convert_dcp_to_huggingface,
    copy_hf_full_weights,
    copy_lora_adapter,
    select_publication,
)
from nhx.rl.tasks.training.backends.nemo_rl.dpo_config import compile_dpo_config
from nhx.rl.tasks.training.backends.nemo_rl.grpo_config import compile_grpo_config, sandbox_host_provider_options
from nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap import create_bootstrap_from_env
from nhx.rl.tasks.training.chat_templates import apply_chat_template_to_checkpoint
from nhx.rl.tasks.training.integrations import find_logger_mlflow_run_id
from nhx.rl.tasks.training.protocol import LibraryConfig, TrainingBackend, TrainingPaused

logger = logging.getLogger(__name__)

# Path to driver scripts (relative to this module)
_DRIVER_DIR = Path(__file__).parent

# Upper bound on one sandbox sweep. A cancelled step pod gets a 30s termination grace period, and
# the driver wait that follows can use all of it, so the sweep has to leave room.
SANDBOX_REAP_TIMEOUT_S = 10.0

# One budget for everything a signal triggers: both sandbox sweeps and the driver termination. It
# stays under the default 30s grace so the process exits before the runtime sends SIGKILL.
SHUTDOWN_GRACE_S = 25.0

# ``terminate_driver`` can wait this long after it kills a driver that ignored the signal.
_DRIVER_KILL_WAIT_S = 5.0


class NemoRLBackend(TrainingBackend):
    """TrainingBackend implementation for NeMo RL (DPO and GRPO).

    This backend handles DPO and GRPO training using NeMo RL.

    Key responsibilities:
    - Run pre-training conversions (model to HF format) via injected converter
    - Compile TrainingStepConfig to NeMo RL YAML format
    - Bootstrap Ray cluster on Volcano-provisioned pods
    - Execute appropriate training driver (DPO)
    - Process checkpoints to standard output format

    Args:
        job_ctx: Job context with job metadata
    """

    def __init__(
        self,
        job_ctx: NHXJobContext,
    ) -> None:
        """Initialize the backend.

        Args:
            job_ctx: Job context with job metadata
        """
        self._job_ctx = job_ctx

    @property
    def backend_type(self) -> TrainingBackendEnum:
        return TrainingBackendEnum.NEMO_RL

    def compile_config(
        self,
        customizer_config: TrainingStepConfig,
        workspace_dir: Path,
    ) -> dict[str, Any]:
        """Compile TrainingStepConfig to NeMo RL YAML format.

        Args:
            customizer_config: The training step configuration
            workspace_dir: Directory for storing generated config files

        Returns:
            Configuration dict for NeMo RL (will be serialized to YAML)
        """
        training_type = customizer_config.training.training_type

        if training_type == TrainingType.DPO:
            return compile_dpo_config(customizer_config, self._job_ctx)

        if training_type == TrainingType.GRPO:
            return compile_grpo_config(customizer_config, self._job_ctx)

        raise NotImplementedError(
            f"NemoRLBackend does not support training type {training_type.value!r}. "
            f"Supported: {TrainingType.DPO.value!r}, {TrainingType.GRPO.value!r}."
        )

    def execute_training(
        self,
        customizer_config: TrainingStepConfig,
        library_config: LibraryConfig,
        progress: JobsServiceProgressReporter,
    ) -> TrainingMetrics:
        """Execute NeMo RL training via Ray bootstrap.

        Args:
            customizer_config: The training step configuration
            library_config: The compiled library configuration
            progress: Progress reporter for status updates

        Returns:
            TrainingMetrics with results from the training run

        Raises:
            TrainingPaused: The driver stopped for a pause after saving its checkpoint.
        """
        progress.report_running(TrainingPhase.TRAINING, backend=self.backend_type.value)

        # Get the workspace directory from config path
        workspace_dir = library_config.config_path.parent

        # Environment overrides the driver subprocess inherits. We snapshot and
        # restore them (below) so a reused worker process doesn't leak this run's
        # values — e.g. a stale MLFLOW_URI — into a subsequent run.
        env_overrides = {
            "BASE_LOG_DIR": str(workspace_dir),
            "GPUS_PER_NODE": str(customizer_config.parallelism.num_gpus_per_node),
        }
        # MLflow integration (if configured)
        if customizer_config.integrations and customizer_config.integrations.mlflow:
            mlflow_config = customizer_config.integrations.mlflow
            if mlflow_config.tracking_uri:
                env_overrides["MLFLOW_URI"] = mlflow_config.tracking_uri
        # Continue this job's MLflow run after a resume or rerun, instead of starting another.
        if mlflow_run_id := find_logger_mlflow_run_id(library_config.config_dict.get("logger") or {}):
            env_overrides["MLFLOW_RUN_ID"] = mlflow_run_id

        # Build driver arguments
        driver_path = self._get_driver_path(customizer_config)
        driver_args = [
            "--config",
            str(library_config.config_path),
            "--id",
            self._job_ctx.job_id,
            "--output-model",
            customizer_config.model.name or "output_model",
        ]

        # Bootstrap Ray cluster and run driver
        logger.info(f"Starting Ray cluster and running driver: {driver_path}")
        logger.info(f"Driver args: {driver_args}")

        bootstrap = create_bootstrap_from_env()
        bootstrap.progress = progress
        if customizer_config.parallelism.num_nodes > 1:
            # Each pod has its own RDMA devices, so each node's ray start gets this pod's list.
            bootstrap.node_env = get_nccl_ib_env()

        # Set up signal handler for cleanup — terminate the driver subprocess
        # explicitly so it doesn't become orphaned, then let SystemExit propagate
        # to trigger Ray cluster cleanup in the bootstrap's finally block.
        shutdown_deadline: float | None = None

        def cleanup(signum, frame):
            """Reap sandboxes, stop the driver, then exit; later signals take the default action."""
            nonlocal shutdown_deadline
            logger.warning(f"Signal {signum} received, terminating driver and cleaning up")
            # Put the default actions back before any blocking work, so a second signal ends the
            # process instead of re-entering this handler in the middle of the first teardown.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            shutdown_deadline = time.monotonic() + SHUTDOWN_GRACE_S
            # Reap before waiting on the driver, which can take the whole termination grace
            # period. The sandbox belongs to a Ray actor that is killed without running its
            # own teardown, and this process is the one the container runtime signals.
            self._reap_job_sandboxes(customizer_config, deadline=shutdown_deadline)
            driver_wait = max(1, int(shutdown_deadline - time.monotonic() - _DRIVER_KILL_WAIT_S))
            bootstrap.terminate_driver(signum, timeout=driver_wait)
            raise SystemExit(signum)

        # Snapshot env vars and signal handlers so we can restore them after the
        # run; otherwise stale state persists on workers reused across runs (a
        # leftover handler closure could even terminate a later, unrelated driver).
        saved_env = {key: os.environ.get(key) for key in env_overrides}
        previous_sigint = signal.getsignal(signal.SIGINT)
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        try:
            os.environ.update(env_overrides)
            signal.signal(signal.SIGINT, cleanup)
            signal.signal(signal.SIGTERM, cleanup)

            exit_code = bootstrap.run_with_driver(str(driver_path), driver_args)
        finally:
            # Whatever ended the driver (a cancel, a crash, an ordinary exit), no sandbox this
            # job created outlives it. A no-op when the Gym actor already tore its own down.
            self._reap_job_sandboxes(customizer_config, deadline=shutdown_deadline)
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
            for key, original in saved_env.items():
                if original is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original

        if exit_code == PAUSE_EXIT_CODE:
            logger.info("Training stopped for a pause after saving its checkpoint")
            raise TrainingPaused()
        if exit_code != 0:
            raise classify_output(bootstrap.driver_output, exit_code)

        logger.info("Training completed successfully")

        # Return empty metrics (actual metrics are logged during training)
        return TrainingMetrics(total_steps=0, total_epochs=0)

    def _reap_job_sandboxes(self, config: TrainingStepConfig, deadline: float | None = None) -> None:
        """Destroy every OpenSandbox resource labelled with this job's id. Never raises.

        ``deadline`` is a ``time.monotonic()`` instant. When set, the sweep gets only the time left
        before it, and is skipped once that has run out.

        Only GRPO jobs that run the Gym in a sandbox have anything to reap. Failures are logged
        rather than raised: this runs from a signal handler and a ``finally``, and an error here
        must not replace the training outcome or skip the rest of the teardown.
        """
        gym = getattr(config, "gym", None)
        if gym is None or not getattr(gym, "sandboxed", False):
            return
        timeout_s = SANDBOX_REAP_TIMEOUT_S
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(f"Shutdown budget spent; not reaping sandboxes for job {self._job_ctx.job_id}")
                return
            timeout_s = min(timeout_s, remaining)
        try:
            from sandboxed_gym.job_reaper import reap_job_sandboxes

            reap_job_sandboxes(
                self._job_ctx.job_id,
                host_provider="opensandbox",
                host_provider_options=sandbox_host_provider_options(gym),
                timeout_s=timeout_s,
            )
        except Exception:
            logger.exception(f"Could not reap sandboxes for job {self._job_ctx.job_id}")

    def _get_driver_path(self, config: TrainingStepConfig) -> Path:
        """Get the appropriate driver script path based on training type.

        Args:
            config: Training configuration with type information

        Returns:
            Path to the driver script (dpo_driver.py or grpo_driver.py)
        """
        training_type = config.training.training_type

        if training_type == TrainingType.DPO:
            return _DRIVER_DIR / "dpo_driver.py"

        if training_type == TrainingType.GRPO:
            return _DRIVER_DIR / "grpo_driver.py"

        raise NotImplementedError(
            f"No training driver available for training type {training_type.value!r}; "
            f"supported: {TrainingType.DPO.value!r}, {TrainingType.GRPO.value!r}."
        )

    def find_best_checkpoint(
        self,
        workspace_dir: Path,
        customizer_config: TrainingStepConfig,
        library_config: Optional[LibraryConfig] = None,
    ) -> Path:
        """Find the best checkpoint after training.

        NeMo RL driver converts the best checkpoint to HuggingFace format
        and saves it to {workspace_dir}/output. This method returns that path.

        Args:
            workspace_dir: Directory containing training artifacts
            customizer_config: Training configuration

        Returns:
            Path to the converted HF checkpoint
        """
        if library_config is None:
            raise ValueError("Library config is required to find the best checkpoint")

        checkpointing_config = library_config.config_dict["checkpointing"]
        if checkpointing_config is None or not isinstance(checkpointing_config, dict):
            raise ValueError("Checkpointing config is required to find the best checkpoint")

        checkpointing_config = cast(CheckpointingConfig, checkpointing_config)
        checkpointer = CheckpointManager(checkpointing_config)

        # get_best_checkpoint_path() handles the missing-metric case internally: it
        # filters out checkpoints lacking the metric (with a warning) and, if none
        # have it, returns the latest checkpoint. It only returns None when there
        # are no checkpoints at all
        best_checkpoint = checkpointer.get_best_checkpoint_path()

        if best_checkpoint is None:
            raise ValueError("No best checkpoint found")

        best_checkpoint_path = Path(best_checkpoint)
        if not best_checkpoint_path.exists():
            raise ValueError(f"Best checkpoint not found at {best_checkpoint_path}")

        return best_checkpoint_path

    def process_checkpoint(
        self,
        checkpoint_path: Path,
        output_path: Path,
        customizer_config: TrainingStepConfig,
        library_config: LibraryConfig | None = None,
    ) -> CheckpointInfo:
        """Process NeMo RL checkpoint to standard output format.

        ``lora`` copies the HF PEFT adapter. ``lora_merged`` folds that adapter into
        the base model and writes a HuggingFace model. Full-weight safetensors are
        copied. DCP is converted when ``.metadata`` is present.
        """
        logger.info("Processing created checkpoint")
        kind, root = select_publication(checkpoint_path, customizer_config.training.finetuning_type)
        if kind == "peft":
            assert root is not None
            logger.info("Copying LoRA adapter from %s to %s", root, output_path)
            copy_lora_adapter(checkpoint_path, root, output_path)
            return CheckpointInfo(
                path=str(output_path),
                format=CheckpointFormat.HF_PEFT,
                precision=customizer_config.model.precision,
            )
        if customizer_config.training.finetuning_type == FinetuningType.LORA and kind != "merge":
            logger.warning(
                "No adapter_config.json under %s (searched %s); falling back to full-weight publication",
                checkpoint_path,
                ", ".join(str(path) for path in LORA_ADAPTER_SEARCH_PATHS),
            )

        hf_checkpoint_path: Path | None = None
        if kind == "merge":
            if root is None:
                searched = ", ".join(str(path) for path in LORA_ADAPTER_SEARCH_PATHS)
                raise FileNotFoundError(
                    "lora_merged publication requires adapter_config.json "
                    f"under {checkpoint_path} (searched {searched})."
                )
            merge_lora_adapter(
                root,
                customizer_config.model.path,
                output_path,
                trust_remote_code=customizer_config.model.trust_remote_code,
                has_vision_tower=customizer_config.model.has_vision_tower,
            )
            hf_checkpoint_path = output_path
        elif kind == "hf":
            assert root is not None
            logger.info("Copying HuggingFace full-weight checkpoint from %s to %s", root, output_path)
            copy_hf_full_weights(checkpoint_path, root, output_path)
            hf_checkpoint_path = output_path
        elif kind == "dcp":
            hf_checkpoint_path = convert_dcp_to_huggingface(checkpoint_path, output_path)
        else:
            raise FileNotFoundError(
                f"No HuggingFace safetensors or DCP .metadata under {checkpoint_path}/policy/weights"
            )

        # Apply chat template if available (full-weight / merged HF trees only)
        chat_template = None
        if library_config and library_config.config_dict:
            chat_template = library_config.config_dict.get("policy", {}).get("tokenizer", {}).get("chat_template")

        if chat_template:
            apply_chat_template_to_checkpoint(hf_checkpoint_path, chat_template)
            logger.debug("Applied chat template to checkpoint")

        return CheckpointInfo(
            path=str(hf_checkpoint_path),
            format=CheckpointFormat.HF,
            precision=customizer_config.model.precision,
        )
