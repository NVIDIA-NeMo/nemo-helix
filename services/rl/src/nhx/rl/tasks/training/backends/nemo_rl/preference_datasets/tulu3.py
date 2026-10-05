# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tulu3 preference dataset with local-file support."""

from typing import Optional

from nemo_rl.data.datasets.preference_datasets.tulu3 import (
    Tulu3PreferenceDataset as BaseTulu3PreferenceDataset,
)
from nemo_rl.data.datasets.utils import load_dataset_from_path


class Tulu3PreferenceDataset(BaseTulu3PreferenceDataset):
    """Tulu3 preference dataset for DPO training, extended for local files.

    NeMo-RL's base ``Tulu3PreferenceDataset`` only downloads
    ``allenai/llama-3.1-tulu-3-8b-preference-mixture`` from HuggingFace and ignores
    any local path. This subclass adds local-file support: when ``data_path`` is
    provided it loads that JSONL (Tulu3 schema — ``chosen`` / ``rejected`` message
    lists) and reuses the base class's :meth:`format_data` to produce the canonical
    ``{context, completions, task_name}`` shape. With no ``data_path`` it falls back
    to the base HuggingFace download.

    NeMo-RL's ``load_preference_dataset`` instantiates the registered class via
    ``cls(**data_config)``, so ``__init__`` accepts the per-split spec keys
    (``data_path``, plus ``dataset_name`` and any others swallowed by ``**kwargs``).
    The ``task_spec`` is bound afterwards by the library via ``set_task_spec``.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        subset: Optional[str] = None,
        split: str = "train",
        **kwargs,
    ) -> None:
        if data_path is None:
            # No local file → keep the base HuggingFace download behavior. Forward
            # split for parity with the local branch and HelpSteer3Dataset (the base
            # currently hard-codes its split, but absorbs the kwarg harmlessly).
            super().__init__(split=split, **kwargs)
            return

        self.task_name = "Tulu3Preference"
        # Load from the local file (or HuggingFace) and apply the base formatting.
        self.dataset = load_dataset_from_path(data_path, subset, split)
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )
