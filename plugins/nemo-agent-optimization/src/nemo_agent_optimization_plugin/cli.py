# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agents optimize`` — the command group every optimization strategy shares.

The group is assembled here rather than generated from the job, because it has
to hold more than one job's worth of commands: ``run-strategy`` (this plugin's
router job), ``list-strategies`` (what is installed), and whatever companion
verbs installed plugins contribute through :data:`OPTIMIZE_CLI_GROUP` —
``prepare-fileset``, for one, which belongs to the plugin whose bundle format it
validates.
"""

from __future__ import annotations

import logging
from typing import Annotated, ClassVar, Optional

import typer
from nemo_agent_optimization_plugin.client import AgentOptimizationClient
from nemo_agent_optimization_plugin.jobs.run_strategy import RunStrategyJob
from nemo_helix_plugin.cli import NemoCLI
from nemo_helix_plugin.client.errors import NemoClientError
from nemo_helix_plugin.commands import add_job_commands
from nemo_helix_plugin.discovery import discover

logger = logging.getLogger(__name__)

#: How long to wait on the platform. Short on purpose: this is a listing command,
#: and an unreachable platform should cost a noticeable pause, not a hang.
STRATEGIES_TIMEOUT_SECONDS = 10.0

#: Same flag and env var as ``nemo_agents_plugin.cli_context.BaseUrlOption``. Declared
#: here rather than imported because typer resolves annotations against module globals,
#: and that module is deliberately not imported at module scope (see
#: :func:`_remote_strategy_names`).
BaseUrlOption = Annotated[
    Optional[str],
    typer.Option(
        "--base-url",
        envvar="NEMO_BASE_URL",
        help="Platform to ask. Defaults to the same target as every other `nemo` command.",
    ),
]

#: Entry-point group a plugin joins to hang its own verbs off this group.
#:
#: Each value is a plain ``def register(group: typer.Typer) -> None``, called with the
#: assembled ``nemo agents optimize`` app. Contributing is deliberately independent of
#: owning a strategy job: a plugin can add a verb here without shipping a strategy, and a
#: strategy needs nothing hung off its job class to add one.
#:
#: The group is shared with every installed contributor, so verb names must not collide:
#: prefix the owning strategy's name onto anything that is not plainly generic.
#: (The ``nat`` strategy holds the generic ``prepare-fileset``.)
OPTIMIZE_CLI_GROUP = "nemo.cli.agents.optimize"


class AgentOptimizeCLI(NemoCLI):
    """Contributes ``nemo agents optimize`` via the ``nemo.cli.agents`` group."""

    name: ClassVar[str] = "optimize"
    description: ClassVar[str] = "Optimize a platform agent with an installed optimization strategy."

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(name=self.name, help=self.description, no_args_is_help=True)

        @app.callback()
        def _root() -> None:
            """Force subcommand dispatch."""

        # The router job is mounted here, not by the generic `<plugin>.<job>` → CLI
        # machinery: its entry-point key lives under `agent-optimization`, which has no
        # CLI of its own, so this is the only place it appears.
        add_job_commands(app, {"agent-optimization.run-strategy": RunStrategyJob}, cli=self)

        @app.command("list-strategies")
        def list_strategies(base_url: BaseUrlOption = None) -> None:
            """List the strategies `--strategy` accepts, one name per line.

            Only the platform is asked, because only the platform runs the job: a
            client venv without a strategy plugin still submits to a server that
            has it, and vice versa. Answering from this environment instead would
            describe a different machine, so an unreachable platform is an error
            rather than a cue to guess.

            Stdout carries names and nothing else, so `for s in $(nemo agents optimize
            list-strategies)` is safe: an empty platform prints nothing there. The
            target is announced once on stderr by the shared base-URL resolver.
            """
            try:
                names, target = _remote_strategy_names(base_url)
            except NemoClientError as exc:
                # Covers all three failure modes the typed client distinguishes: transport,
                # HTTP status, and a response body that does not match the schema.
                typer.echo(f"Error: could not list the platform's strategies: {exc}", err=True)
                raise typer.Exit(code=1) from exc

            if not names:
                typer.echo(f"No optimization strategies are installed on {target}.", err=True)
                return
            for name in names:
                typer.echo(name)

        _register_contributed_subcommands(app)
        return app


def _register_contributed_subcommands(group: typer.Typer) -> None:
    """Let every plugin that joined :data:`OPTIMIZE_CLI_GROUP` hang its verbs off *group*.

    ``discover`` already isolates a contributor that fails to *import*; this covers one
    that fails to *run*. Either way a broken contributor must not take the whole ``nemo
    agents`` CLI down with it — losing one helper verb is recoverable, losing the command
    tree is not. Registration order is the discovery layer's documented name sort, so the
    rendered help is stable across environments.
    """
    try:
        contributors = discover(OPTIMIZE_CLI_GROUP)
    except Exception:  # noqa: BLE001 — a broken sibling plugin must not break the CLI
        logger.warning("Could not discover %r contributions", OPTIMIZE_CLI_GROUP, exc_info=True)
        return

    for name, register in contributors.items():
        try:
            register(group)
        except Exception:  # noqa: BLE001 — one bad contributor must not break the CLI
            logger.warning("Optimize CLI contribution %r failed to register", name, exc_info=True)


def _remote_strategy_names(base_url: str | None) -> tuple[list[str], str]:
    """Ask the platform for its installed strategies. Raises if it cannot answer.

    The target and auth headers resolve the way the rest of ``nemo agents`` does.
    ``nemo_agents_plugin.cli_context`` is imported lazily and is not a declared
    dependency: this group is only ever reached through the ``nemo.cli.agents``
    entry point, so the agents plugin is installed whenever this code runs, and
    declaring it would drag the whole agents stack into service-only installs.
    """
    from nemo_agents_plugin.cli_context import resolve_base_url, resolve_context_headers

    target = resolve_base_url(base_url)
    with AgentOptimizationClient(
        base_url=target,
        default_headers=resolve_context_headers(),
        timeout=STRATEGIES_TIMEOUT_SECONDS,
    ) as client:
        listing = client.list_strategies().data()
    return [strategy.name for strategy in listing.data], target
