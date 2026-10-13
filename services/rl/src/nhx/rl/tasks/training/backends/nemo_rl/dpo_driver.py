# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DPO training driver (ray run entry point).

Entry point for DPO (Direct Preference Optimization) training, invoked via
ray run in a distributed environment.

Preference data handling lives in NeMo-RL itself (``setup_preference_data`` plus
the config-driven ``BinaryPreferenceDataset`` / ``PreferenceDataset`` loaders):
the ``data`` config emitted by ``dpo_config.compile_dpo_config`` drives the
built-in loaders, so no custom preprocessor is needed. On top of NeMo-RL's DPO
loop we add the ``NemoRLLogger`` that streams progress back to the NeMo Helix
Jobs service.
"""

import argparse
import logging
from typing import Any, cast

from nemo_rl.algorithms.dpo import MasterConfig, dpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_preference_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir
from nhx.customization_common.service.context import NHXJobContext
from nhx.rl.tasks.training.backends.nemo_rl.nemo_rl_logger import (
    DPO_DEFAULT_TIME_SERIES_METRICS,
    NemoRLLogger,
)
from nhx.rl.tasks.training.backends.nemo_rl.preference_datasets import register_preference_datasets
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run DPO training with configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--id", type=str, help="Customization ID")
    parser.add_argument("--output-model", type=str, help="Output Model")

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def main():
    """Main entry point."""
    args, overrides = parse_args()

    cfg = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        cfg = parse_hydra_overrides(cfg, overrides)

    # NeMo-RL's MasterConfig is a Pydantic BaseModel; setup()/dpo_train() read it
    # by attribute (e.g. master_config.dpo.seed). OmegaConf.to_container() returns
    # a plain dict, so build the model here — mirroring NeMo-RL's own run_dpo.py —
    # which validates the config up front and matches how the algorithm consumes
    # it. Only the top level is a model: the sub-configs (policy/logger/... ) stay
    # TypedDict dicts, so they are still accessed by subscript (config.policy["x"]).
    config = MasterConfig(**cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)))
    print("Applied CLI overrides")

    # Log only the top-level config section names. The resolved config carries
    # integration secrets (W&B / MLflow tokens, tracking URIs), so never dump the
    # full structure to stdout. model_fields is names-only — no values materialized.
    print(f"Config sections loaded: {sorted(type(config).model_fields)}")

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}")

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config.policy["tokenizer"])

    # Register our local-file-capable HelpSteer3 / Tulu3 datasets into NeMo-RL's
    # DATASET_REGISTRY before building data. Without this, setup_preference_data
    # resolves those two formats to NeMo-RL's built-in classes, which always
    # download from HuggingFace and ignore the uploaded local files.
    register_preference_datasets()

    # setup data — NeMo-RL builds the datasets from the `data` config (per-split
    # dataset specs). The compiler emits one of BinaryPreferenceDataset /
    # PreferenceDataset / HelpSteer3 / Tulu3Preference per detected schema, each
    # pointing at the prepared local training.jsonl / validation.jsonl.
    dataset, val_dataset = setup_preference_data(tokenizer, config.data)
    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        dpo_save_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    # Add NemoRLLogger for progress reporting if Jobs service is configured
    job_ctx = NHXJobContext.from_env()
    # Log only the non-sensitive job id; the full context carries service URLs
    # and identifiers that should not be dumped to stdout.
    print(f"Job context loaded (job_id={job_ctx.job_id})")
    customizer_logger: NemoRLLogger | None = None
    if job_ctx.jobs_url:
        customizer_logger = NemoRLLogger.for_schedule(
            job_ctx=job_ctx,
            max_steps=config.dpo.max_num_steps,
            num_epochs=config.dpo.max_num_epochs,
            val_period=config.dpo.val_period,
            # Extra (undeclared) DPOConfig fields, present only because the model
            # allows extras -- dpo_config.py puts them there. Read with getattr, not
            # attribute access: a config compiled elsewhere simply omits them, and
            # pydantic raises AttributeError for a missing extra. None is what
            # for_schedule's fallbacks take -- derive-from-the-schedule for
            # steps_per_epoch, and the shared default for the reporting budget.
            steps_per_epoch=getattr(config.dpo, "steps_per_epoch", None),
            time_series_metrics=getattr(config.dpo, "progress_time_series_metrics", None),
            min_report_interval_seconds=getattr(config.dpo, "progress_min_report_interval_seconds", None),
            default_time_series_metrics=DPO_DEFAULT_TIME_SERIES_METRICS,
            # `backend` is `nemo_rl` for both algorithms, so this is the only field
            # in a job's status that says which one ran. Set here as well as in the
            # GRPO driver: a field only one algorithm fills in looks like missing
            # data rather than like "not GRPO".
            run_facts={"training_type": "dpo"},
        )
        # The setup() logger is a composite with a `.loggers` list; guard in case
        # that internal shape changes.
        if hasattr(logger, "loggers"):
            logger.loggers.append(customizer_logger)
        else:
            print("WARNING: logger has no `.loggers`; NeMo Helix progress reporting disabled.")

    logger.log_hyperparams(config.model_dump())

    try:
        dpo_train(
            policy,
            train_dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            master_config,
            logger,
            checkpointer,
            dpo_save_state,
        )
    finally:
        # Flushes the final training step. NeMo-RL never closes the loggers it is
        # handed, so without this the only fallback is NemoRLLogger.__del__ at
        # interpreter shutdown, which does not run at all on an abnormal exit.
        if customizer_logger is not None:
            customizer_logger.close()


if __name__ == "__main__":
    main()
