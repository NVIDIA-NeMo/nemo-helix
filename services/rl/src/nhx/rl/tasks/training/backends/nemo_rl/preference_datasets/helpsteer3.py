# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HelpSteer3 preference dataset with local-file support."""

from typing import Optional

from nemo_rl.data.datasets.preference_datasets.helpsteer3 import (
    HelpSteer3Dataset as BaseHelpSteer3Dataset,
)
from nemo_rl.data.datasets.utils import load_dataset_from_path


class HelpSteer3Dataset(BaseHelpSteer3Dataset):
    """HelpSteer3 preference dataset for DPO training, extended for local files.

    NeMo-RL's base ``HelpSteer3Dataset`` only downloads ``nvidia/HelpSteer3`` from
    HuggingFace and ignores any local path. This subclass adds local-file support:
    when ``data_path`` is provided it loads that JSONL (HelpSteer3 schema —
    ``context`` / ``response1`` / ``response2`` / ``overall_preference``) and reuses
    the base class's :meth:`format_data` to produce the canonical
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
            # No local file → keep the base HuggingFace download behavior.
            super().__init__(split=split, **kwargs)
            return

        self.task_name = "HelpSteer3"
        # Load from the local file (or HuggingFace) and apply the base formatting.
        self.dataset = load_dataset_from_path(data_path, subset, split)
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )
