# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo profile`` command group: the caller's own display name, time zone, and locale."""

from __future__ import annotations

from typing import Annotated

import typer
from nemo_helix_plugin.auth.profiles.client import ProfilesClient
from nemo_helix_plugin.auth.profiles.types import Me, ProfileUpdate
from nemo_helix_plugin.client.adapter import client_from_platform
from rich.console import Console
from rich.table import Table
from rich.text import Text

from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import format_output
from nemo_helix_ext.cli.core.help_formatter import create_typer_app

app = create_typer_app(name="profile", help="Manage your NeMo Helix user profile.")


def profiles_client(state: CLIContext) -> ProfilesClient:
    return client_from_platform(state.get_client(), ProfilesClient)


def print_me(me: Me, state: CLIContext) -> None:
    """Render ``/me`` as a property table, or as json/yaml for the other output formats."""
    output_format = state.get_output_format()
    if output_format != "table":
        format_output(me, output_format=output_format)
        return
    profile = me.profile
    table = Table(title="Who am I", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", overflow="fold")
    rows = [
        ("Account ID", me.account_id),
        ("Principal", me.principal_id or Text("none", style="dim")),
        ("Display Name", profile.display_name),
        ("IdP Name", profile.idp_display_name if profile.idp_display_name != profile.display_name else None),
        ("Name Override", profile.display_name_override),
        ("Email", me.email),
        ("Groups", ", ".join(me.groups) or None),
        ("Time Zone", profile.timezone),
        ("Locale", profile.locale),
        ("Auth Method", me.auth_method),
        ("Caller Kind", me.caller_kind),
        ("Can Edit Profile", "yes" if me.can_edit_profile else "no"),
        ("Platform Admin", "yes" if me.is_platform_admin else "no"),
    ]
    for label, value in rows:
        if value is not None:
            # Names and groups are user text: render them as-is, never as Rich markup.
            table.add_row(label, value if isinstance(value, Text) else Text(value))
    Console().print(table)


@app.command("set")
@handle_errors
def set_profile(
    ctx: typer.Context,
    display_name: Annotated[str | None, typer.Option("--display-name", help="Name shown to other users.")] = None,
    clear_display_name: Annotated[
        bool, typer.Option("--clear-display-name", help="Remove your name override and use the IdP name.")
    ] = False,
    timezone: Annotated[str | None, typer.Option("--timezone", help="IANA time zone, e.g. Europe/Paris.")] = None,
    clear_timezone: Annotated[bool, typer.Option("--clear-timezone", help="Remove your time zone.")] = False,
    locale: Annotated[str | None, typer.Option("--locale", help="BCP 47 language tag, e.g. en-US.")] = None,
    clear_locale: Annotated[bool, typer.Option("--clear-locale", help="Remove your locale.")] = False,
) -> None:
    """Update your profile. Only the options you pass change.

    Your email comes from your identity provider and cannot be changed here.

    Examples:
      nemo profile set --display-name "Alice A."
      nemo profile set --timezone Europe/Paris --locale fr-FR
      nemo profile set --clear-display-name
    """
    fields: dict[str, str | None] = {}
    for name, value, clear in (
        ("display_name", display_name, clear_display_name),
        ("timezone", timezone, clear_timezone),
        ("locale", locale, clear_locale),
    ):
        if value is not None and clear:
            raise typer.BadParameter(f"Pass either --{name.replace('_', '-')} or its --clear option, not both.")
        if value is not None or clear:
            fields[name] = value
    if not fields:
        raise typer.BadParameter("Pass at least one option to change.")

    state: CLIContext = ctx.obj
    me = profiles_client(state).update_my_profile(body=ProfileUpdate(**fields)).data()
    print_me(me, state)
