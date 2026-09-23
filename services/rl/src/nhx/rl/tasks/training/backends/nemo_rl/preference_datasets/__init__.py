# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preference datasets for DPO training.

NeMo-RL's built-in ``HelpSteer3Dataset`` / ``Tulu3PreferenceDataset`` only download
their datasets from HuggingFace and ignore a local ``data_path``. This package
provides local-file-capable subclasses and registers them into NeMo-RL's
``DATASET_REGISTRY`` so the library's ``setup_preference_data`` /
``load_preference_dataset`` resolve a user-uploaded HelpSteer3/Tulu3 dataset to the
local file instead of silently training on the public HuggingFace dataset.

``BinaryPreferenceDataset`` / ``PreferenceDataset`` already load from a local path,
so they are re-exported unchanged.
"""

from nemo_rl.data.datasets.preference_datasets import (
    DATASET_REGISTRY,
    BinaryPreferenceDataset,
    PreferenceDataset,
)
from nhx.rl.tasks.training.backends.nemo_rl.preference_datasets.helpsteer3 import HelpSteer3Dataset
from nhx.rl.tasks.training.backends.nemo_rl.preference_datasets.tulu3 import Tulu3PreferenceDataset


def register_preference_datasets() -> None:
    """Override NeMo-RL's HF-only HelpSteer3 / Tulu3 with local-file-capable subclasses.

    NeMo-RL's ``load_preference_dataset`` resolves ``dataset_name`` via the
    module-level ``DATASET_REGISTRY`` dict and constructs ``cls(**data_config)``.
    Re-pointing the "HelpSteer3" / "Tulu3Preference" entries at our subclasses makes
    ``setup_preference_data`` honor the compiled ``data_path`` for those formats.

    Must be called before ``setup_preference_data`` (e.g. at driver start-up).
    Idempotent — safe to call more than once.
    """
    DATASET_REGISTRY["HelpSteer3"] = HelpSteer3Dataset
    DATASET_REGISTRY["Tulu3Preference"] = Tulu3PreferenceDataset


__all__ = [
    "BinaryPreferenceDataset",
    "HelpSteer3Dataset",
    "PreferenceDataset",
    "Tulu3PreferenceDataset",
    "register_preference_datasets",
]
