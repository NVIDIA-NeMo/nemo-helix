# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""References to persisted metrics and their resolution into executable bundles.

A :class:`MetricRef` lets an evaluation job point at a stored metric by
``workspace/name`` instead of inlining it. During spec resolution (``to_spec``),
references are loaded from storage and inline :class:`MetricInline` DTOs are
converted, so the canonical job spec and execution path only ever see runtime
:class:`MetricBundle` objects.
"""

from __future__ import annotations

# ``MetricRef`` / ``MetricRefOrInline`` are defined in ``api.schemas`` (next to ``MetricInline``) so
# entity/DTO modules can reference them without importing this module's entities-dependent resolution
# logic (which would create an import cycle). Imported here for use below and re-exported for the
# existing ``nemo_evaluator.metric_refs`` import sites.
from nemo_evaluator.api.schemas import MetricRef, MetricRefOrInline
from nemo_evaluator.entities import MetricBundleEntity
from nemo_evaluator.metric_storage import load_bundle
from nemo_evaluator.shared.metric_bundles.bundles import MetricBundle
from nemo_platform_plugin.entity_client import NemoEntityGetterProtocol, NemoEntityNotFoundError
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.refs import parse_entity_ref


def parse_metric_ref(root: str, default_workspace: str) -> tuple[str, str]:
    """Split a validated metric reference into ``(workspace, name)``.

    Thin alias over the platform's :func:`~nemo_platform_plugin.refs.parse_entity_ref` (all
    ``workspace/name`` refs split identically); kept for the existing ``metric_refs`` call sites,
    which want a tuple. A metric ref carries no ``#fragment`` — metrics are not revisioned — so the
    plain entity parser is the right one here.
    """
    parsed = parse_entity_ref(root, default_workspace)
    return parsed.workspace, parsed.name


async def resolve_metric_ref(
    ref: MetricRef,
    *,
    workspace: str,
    entity_client: NemoEntityGetterProtocol[MetricBundleEntity] | None,
    files_client: AsyncFilesClient | None,
) -> MetricBundle:
    """Load and reconstruct the stored metric a reference points at."""
    if entity_client is None or files_client is None:
        raise ValueError(
            "MetricRef metrics require a platform connection (entity store and Files service) to resolve; "
            "pass an inline metric instead."
        )
    ref_workspace, name = parse_metric_ref(ref.root, workspace)
    try:
        entity = await entity_client.get(MetricBundleEntity, name=name, workspace=ref_workspace)
    except NemoEntityNotFoundError as exc:
        raise ValueError(
            f"Metric reference '{ref.root}' not found. "
            f"Ensure a stored metric named '{name}' exists in workspace '{ref_workspace}', "
            "or pass an inline metric instead."
        ) from exc
    return await load_bundle(files_client, entity.bundle_ref, expected_digest=entity.payload_digest)


async def resolve_metric_specs(
    metrics: list[MetricRefOrInline],
    *,
    workspace: str,
    entity_client: NemoEntityGetterProtocol[MetricBundleEntity] | None,
    files_client: AsyncFilesClient | None,
) -> list[MetricBundle]:
    """Resolve a wire metric list into runtime bundles.

    References are loaded from storage; inline :class:`MetricInline` DTOs are
    converted to runtime bundles by JSON round-trip (which keeps the base64
    payload encoding consistent).
    """
    resolved: list[MetricBundle] = []
    for item in metrics:
        if isinstance(item, MetricRef):
            resolved.append(
                await resolve_metric_ref(
                    item,
                    workspace=workspace,
                    entity_client=entity_client,
                    files_client=files_client,
                )
            )
        else:
            resolved.append(MetricBundle.model_validate_json(item.model_dump_json()))
    return resolved
