# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage an optimize study's dataset reference to a local path.

Moved verbatim out of ``nemo_optimization.jobs.optimize`` (Task 9) so both the legacy
``OptimizeJob`` and the ``AgentOptimizeJob``-based ``nat`` strategy can share it without
either importing the other's job module.
"""

from __future__ import annotations

import contextlib
import copy
import re
from collections.abc import Iterator, Mapping
from typing import Any

from nemo_platform import NeMoPlatform
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.refs import FILESET_REF_PATTERN


@contextlib.contextmanager
def _staged_dataset(
    optimize_config: dict[str, Any],
    *,
    workspace: str,
    ctx: JobContext,
    sdk: NeMoPlatform | None,
) -> Iterator[dict[str, Any]]:
    """Yield *optimize_config* with a fileset dataset reference replaced by a local path.

    ``eval.general.dataset`` may be a plain host path (CLI runs) or a
    ``workspace/fileset#path`` reference (remote submitters, who have no host
    filesystem).  For the reference form the fileset is downloaded to a tempdir
    for the duration of the study and the config is rewritten in place, so
    everything downstream keeps seeing a plain readable path.
    """
    ref = _dataset_fileset_ref(optimize_config)
    if ref is None:
        yield optimize_config
        return

    # Soft dependency, mirroring nemo_optimization.agents' lazy imports.
    from nemo_agents_plugin.jobs.fileset_io import resolve_staged_config

    fileset_ref, _, object_path = ref.partition("#")
    with resolve_staged_config(
        object_path,
        fileset_ref,
        workspace=workspace,
        ctx=ctx,
        sdk=sdk,
        kind="optimize-dataset",
    ) as local_path:
        yield _with_dataset_path(optimize_config, str(local_path))


def _dataset_fileset_ref(optimize_config: Mapping[str, Any]) -> str | None:
    """Return ``eval.general.dataset`` when it is a ``workspace/fileset#path`` ref."""
    dataset = _dataset_node(optimize_config)
    value = dataset if isinstance(dataset, str) else None
    if isinstance(dataset, Mapping):
        candidate = dataset.get("file_path") or dataset.get("path")
        value = candidate if isinstance(candidate, str) else None
    if value is None or not re.match(FILESET_REF_PATTERN, value):
        return None
    return value


def _dataset_node(optimize_config: Mapping[str, Any]) -> Any:
    general = optimize_config.get("eval", {})
    general = general.get("general") if isinstance(general, Mapping) else None
    return general.get("dataset") if isinstance(general, Mapping) else None


def _with_dataset_path(optimize_config: dict[str, Any], local_path: str) -> dict[str, Any]:
    """Copy *optimize_config* with the dataset location swapped for *local_path*."""
    updated = copy.deepcopy(optimize_config)
    general = updated["eval"]["general"]
    dataset = general.get("dataset")
    general["dataset"] = {"file_path": local_path} if isinstance(dataset, str) else {**dataset, "file_path": local_path}
    return updated
