# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-specific reference strings for the agents plugin.

This module defines the ref types that are unique to agent jobs:
:class:`AgentRef` and the :data:`AgentTarget` union it forms with
:class:`~nemo_helix_plugin.refs.EndpointURL`.  The generic ref machinery
(:class:`~nemo_helix_plugin.refs.StrRef`, :class:`~nemo_helix_plugin.refs.EndpointURL`,
:class:`~nemo_helix_plugin.refs.LocalDir`, :class:`~nemo_helix_plugin.refs.FilesetRef`,
:data:`~nemo_helix_plugin.refs.OutputTarget`,
:func:`~nemo_helix_plugin.refs.classify_output_target`) lives in
:mod:`nemo_helix_plugin.refs` so other plugins can reuse the same convention.

Both names from the generic module are re-exported here for a single
import surface at call sites that touch agent specs (the
``EvaluateAgentJob`` spec in particular declares an ``OutputTarget``
field alongside its ``AgentTarget`` field — sourcing both from
:mod:`nemo_agents_plugin.refs` keeps the two side-by-side in the
import block).
"""

from __future__ import annotations

from typing import ClassVar, Union

from nemo_helix_plugin.refs import (
    EndpointURL,
    FilesetRef,
    LocalDir,
    OutputTarget,
    StrRef,
    classify_output_target,
)


class AgentRef(StrRef):
    """A reference to a platform-managed agent (``"name"`` or ``"workspace/name"``).

    The job resolves this to the gateway URL
    ``{base_url}/apis/agents/v2/workspaces/{workspace}/agents/{name}/-`` at
    run time.  No validation is performed here — the resolver returns an
    actionable error if the agent (or its deployment) is missing.
    """

    __cli_metavar__: ClassVar[str | None] = "AGENT_REF"


# Documentary union alias — the wire shape is still ``str``.  The
# ``_spec_flags`` generator collapses this to a single ``--agent`` flag
# of type ``str``; the disambiguation between the two arms happens in
# :func:`classify_agent_target` at job-run time.
AgentTarget = Union[AgentRef, EndpointURL]
"""A reference to an agent: platform-managed agent name *or* explicit endpoint URL."""


def classify_agent_target(value: str) -> type[StrRef]:
    """Classify *value* as an :class:`EndpointURL` or :class:`AgentRef`.

    Dispatch is purely shape-based: the presence of ``"://"`` is the
    canonical "this is a URL" marker (matches what
    :func:`nemo_helix_ext.refs.parser.classify_input` does for the
    EXTERNAL_URL bucket).  Anything else is treated as a NeMo Helix agent
    reference.

    Returns the concrete subclass — callers can either compare against
    the class objects or wrap the returned class around *value* to get
    a typed instance.
    """
    if "://" in value:
        return EndpointURL
    return AgentRef


__all__ = [
    "AgentRef",
    "AgentTarget",
    "EndpointURL",
    "FilesetRef",
    "LocalDir",
    "OutputTarget",
    "StrRef",
    "classify_agent_target",
    "classify_output_target",
]
