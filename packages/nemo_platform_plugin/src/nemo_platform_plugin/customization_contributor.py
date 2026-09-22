# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contributor protocol for customization training backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from textwrap import wrap
from typing import ClassVar, Protocol, runtime_checkable

import typer
from nemo_platform_plugin.service import RouterSpec


class CustomizationContributorDiscoveryError(RuntimeError):
    """Raised when customization contributor discovery fails."""


CustomizationSDKResourceFactory = Callable[..., object]


@dataclass(frozen=True, slots=True)
class CustomizationContributorSDKResources:
    """Sync/async resource classes mounted under ``client.customization.<name>``."""

    # The customization hub owns the concrete context type and validates the
    # constructed resource shape before exposing a contributor SDK resource.
    sync_resource: CustomizationSDKResourceFactory | None = None
    async_resource: CustomizationSDKResourceFactory | None = None

    def __post_init__(self) -> None:
        if self.sync_resource is None and self.async_resource is None:
            raise ValueError("At least one of sync_resource or async_resource must be provided")


@dataclass(frozen=True, slots=True)
class CustomizationCLISummary:
    """One backend's entry in the ``nemo customization --help`` overview.

    The fields are fixed rather than free text so that every backend reads the
    same way in the overview. Keep each value to one or two short sentences. The
    full description belongs on the backend's own ``--help`` and ``submit --help``.
    """

    #: What the backend trains, for example ``"SFT or LoRA fine-tuning."``.
    trains: str
    #: Where training runs, plus any runtime limit.
    runs_on: str
    #: The shape of the job JSON the backend expects.
    job_json: str
    #: When to choose this backend over another one.
    use_when: str
    #: The command to run next, for example ``"nemo customization rl submit job.json"``.
    command: str

    def render(self, name: str, *, width: int = 78) -> str:
        """Return the summary as an indented block headed by *name*.

        Wrapping happens here rather than in the help formatter, because the
        router's help is printed through an 80-column console that would re-wrap
        long lines and drop their indent. The hanging indent keeps a wrapped
        field readable.
        """
        fields = (
            ("Trains", self.trains),
            ("Runs on", self.runs_on),
            ("Job JSON", self.job_json),
            ("Use it when", self.use_when),
            ("Submit", self.command),
        )
        lines = [name]
        for label, value in fields:
            lines.extend(wrap(f"{label}: {value}", width=width, initial_indent="  ", subsequent_indent="    "))
        return "\n".join(lines)


@runtime_checkable
class CustomizationCLISummaryProvider(Protocol):
    """Optional add-on to :class:`CustomizationContributor` for the CLI overview.

    Kept separate from the required contract on purpose. Discovery validates a
    loaded contributor against :class:`CustomizationContributor`, and a
    runtime-checkable protocol checks every member it declares, so folding this
    method into that contract would make a backend that predates it fail to load
    over missing help text.
    """

    def get_cli_summary(self) -> CustomizationCLISummary | None:
        """Short summary for the ``nemo customization --help`` overview.

        The router collects one summary per discovered backend, in name order, so
        that a user can choose a backend before reading any per-backend help.
        Return ``None`` to be listed by name only. The detailed description of the
        backend, including job JSON fields and constraints, belongs on the Typer
        app returned by ``get_cli()``.
        """


@runtime_checkable
class CustomizationContributor(Protocol):
    """One training backend mounted under ``/apis/customization``."""

    name: ClassVar[str]
    dependencies: ClassVar[list[str]]

    def get_routers(self) -> list[RouterSpec]:
        """HTTP routes for this backend (workspace-scoped prefix per backend)."""

    def get_cli(self) -> typer.Typer | None:
        """CLI subgroup mounted at ``nemo customization <name>``.

        HTTP authorization is **not** declared here: it is derived from the
        ``@path_rule``-decorated routes returned by :meth:`get_routers`, which the
        customization hub aggregates into its own ``nemo.services`` route surface.
        """

    def get_sdk_resources(self) -> CustomizationContributorSDKResources | None:
        """Return SDK resource classes for ``client.customization.<name>``.

        Return :class:`CustomizationContributorSDKResources` with sync and/or async
        resource classes. The Customizer hub owns the top-level
        ``client.customization`` SDK entry and passes each backend resource a
        typed Customizer SDK context containing the typed Customizer client, the
        typed Jobs client, and the active workspace. Do not register a separate
        ``nemo.sdk`` entry point; the Customizer hub composes contributors.
        """
        ...
