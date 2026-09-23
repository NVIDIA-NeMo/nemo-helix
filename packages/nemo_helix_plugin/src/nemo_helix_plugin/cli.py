# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin CLI interface — what plugin authors implement for CLI contributions.

Plugin authors subclass :class:`NemoCLI` and register the class under the
``nemo.cli`` entry-point group.  The platform instantiates each class at
startup, calls :meth:`get_cli`, and mounts the returned app as a subcommand
under ``nemo <plugin-name> <command>``.

Example::

    # my_plugin/cli.py
    import typer
    from nemo_helix_plugin.cli import NemoCLI

    class MyCLI(NemoCLI):
        name = "my-plugin"
        description = "My plugin commands."

        def get_cli(self) -> typer.Typer:
            app = typer.Typer(help=self.description)

            @app.command()
            def run(model: str) -> None:
                \"\"\"Run something.\"\"\"
                typer.echo(f"Running with {model}")

            return app

    # pyproject.toml:
    # [project.entry-points."nemo.cli"]
    # my-plugin = "my_plugin.cli:MyCLI"
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Literal

import typer
from nemo_helix_plugin._base import _NamedPlugin
from nemo_helix_plugin.cli_renderer import CLIRenderer
from nemo_helix_plugin.function import NemoFunction
from nemo_helix_plugin.job import NemoJob


class NemoCLI(_NamedPlugin):
    """Abstract base class for plugin-contributed CLI commands.

    Subclasses are registered directly as the ``nemo.cli`` entry-point value.
    The platform instantiates each class at startup, calls :meth:`get_cli`, and
    mounts the returned app as ``nemo <name> <command>``.  Plugin authors
    never instantiate the class themselves.

    Class variables:

    .. attribute:: name
        :type: str

        Unique kebab-case plugin name.  Must match the ``nemo.cli`` entry-point
        key in ``pyproject.toml``.

    .. attribute:: description
        :type: str

        Human-readable description.  Defaults to ``""``.

    Customization hooks
    -------------------

    Plugins that contribute :class:`~nemo_helix_plugin.function.NemoFunction` or
    :class:`~nemo_helix_plugin.job.NemoJob` primitives get an auto-generated CLI
    surface (``run`` / ``submit`` / ``explain`` verbs with one Typer flag per
    spec leaf). Override :meth:`update_function_cli` or :meth:`update_job_cli`
    to amend that surface — add a flag, drop a flag, replace the verb entirely,
    or anything else Typer permits. Both hooks default to no-ops, so plugins
    that don't override them get today's auto-generated surface unchanged.
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""

    @abstractmethod
    def get_cli(self) -> typer.Typer:
        """Return the Typer app contributed by this plugin."""

    def update_function_cli(self, fn_cls: type[NemoFunction], group: typer.Typer) -> None:
        """Customize the auto-generated function sub-CLI for *fn_cls*.

        Called once per :class:`~nemo_helix_plugin.function.NemoFunction` the plugin
        contributes, **after** the default ``run`` / ``submit`` verbs are
        registered on *group* and **before** the group is mounted on the
        plugin's CLI app. ``group.registered_commands`` carries one
        :class:`typer.models.CommandInfo` per verb, each with ``.name`` and
        ``.callback`` (the auto-generated function with its synthetic
        ``__signature__`` attached).

        Override by replacement: pluck the original callback, write a wrapper
        with the desired signature, and re-register under the same verb name —
        Typer's last-write-wins materialization rule
        (``typer/main.py``: ``commands[name] = command``) keeps the new
        registration. The default is a no-op.
        """

    def update_job_cli(self, job_cls: type[NemoJob], group: typer.Typer) -> None:
        """Customize the auto-generated job sub-CLI for *job_cls*.

        Same shape and override-by-replacement contract as
        :meth:`update_function_cli`, but for jobs (with ``run`` / ``submit`` /
        ``explain`` verbs). The default is a no-op.
        """

    def get_function_renderer(  # noqa: ARG002 — non-abstract default with named parameters
        self,
        fn_cls: type[NemoFunction],
        *,
        verb: Literal["run", "submit"],
    ) -> type[CLIRenderer] | None:
        """Return a :class:`~nemo_helix_plugin.cli_renderer.CLIRenderer` class for *fn_cls*'s *verb*.

        When this returns a class (and the global ``--output-format json``
        flag is not set), the framework instantiates it and drives its
        lifecycle around the streamed output instead of echoing each NDJSON
        frame as JSON to stdout. Return ``None`` to keep the default echo
        behavior. The default is to return ``None`` for every (function, verb)
        pair.

        Note the **delegate-to-use-the-renderer contract**: the renderer
        driver lives inside the framework's default ``run`` / ``submit``
        callback bodies, parameterized by this hook at invocation time. An
        :meth:`update_function_cli` wrapper that *delegates to the original
        callback* gets the renderer for free; a wrapper that takes over the
        verb body wholesale (does its own iteration without calling original)
        has implicitly claimed rendering responsibility too.
        """
        return None

    def get_job_renderer(  # noqa: ARG002 — non-abstract default with named parameters
        self,
        job_cls: type[NemoJob],
        *,
        verb: Literal["run", "submit"],
    ) -> type[CLIRenderer] | None:
        """Return a :class:`~nemo_helix_plugin.cli_renderer.CLIRenderer` class for *job_cls*'s *verb*.

        Same shape and delegate-to-use-the-renderer contract as
        :meth:`get_function_renderer`, but for jobs. ``explain`` is omitted
        from the verb literal because it prints schemas synchronously and
        doesn't stream — there's nothing for a renderer to drive.
        """
        return None
