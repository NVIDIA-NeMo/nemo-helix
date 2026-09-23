# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The built-in task kinds; orchestrators may inject another mapping for tests."""

from collections.abc import Mapping

from nemo_evaluator.jobs.kinds.evaluator import EvaluatorTaskAdapter
from nemo_evaluator.jobs.kinds.harbor import HarborTaskAdapter
from nemo_evaluator.jobs.kinds.types import TaskKindAdapter

KIND_ADAPTERS: Mapping[str, TaskKindAdapter] = {
    adapter.kind: adapter for adapter in (EvaluatorTaskAdapter(), HarborTaskAdapter())
}


def get_adapter(kind: str, adapters: Mapping[str, TaskKindAdapter] = KIND_ADAPTERS) -> TaskKindAdapter:
    """Return the adapter that owns a resolved task kind.

    Args:
        kind: Task-definition discriminator to look up.
        adapters: Adapter registry; callers may inject a registry for tests or extensions.

    Returns:
        The adapter registered for ``kind``.

    Raises:
        ValueError: The registry contains no adapter for ``kind``.
    """
    try:
        return adapters[kind]
    except KeyError:
        raise ValueError(f"Unsupported task kind {kind!r}; known kinds: {sorted(adapters)}") from None
