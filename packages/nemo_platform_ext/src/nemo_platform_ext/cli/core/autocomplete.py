# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from typer import Context


def autocomplete_workspace(
    ctx: Context,
    incomplete: str,
) -> list[tuple[str, str]]:
    # TODO
    return []


def autocomplete_model_entity(ctx: Context, incomplete: str) -> list[tuple[str, str]]:
    """
    Autocomplete model entity names from registered model entities.

    Used for model entity routing where IGW resolves the provider automatically.
    Supports both full format (workspace/model-name) and short format (model-name).
    """
    from nemo_platform_ext.cli.core.context import CLIContext

    workspace = ctx.params.get("workspace", None)

    try:
        # Context needs to be created here, as it's not created by default in the autocomplete flow.
        state: CLIContext = CLIContext()

        # Suppress logging during autocomplete
        logging.getLogger().setLevel(logging.CRITICAL)

        from nemo_platform_plugin.inference_gateway.client import InferenceGatewayClient

        models = state.typed_client(InferenceGatewayClient).list_openai_models(workspace=workspace).data()

        if models.data:
            results: list[tuple[str, str]] = []
            has_workspace_prefix = "/" in incomplete

            for model in models.data:
                model_id = model.id  # format: workspace/model-name

                if has_workspace_prefix:
                    # User is typing full format, match against full ID
                    if model_id.startswith(incomplete):
                        results.append((model_id, f"Model: {model_id}"))
                else:
                    # User is typing short format, match against model name part
                    if "/" in model_id:
                        _, model_name = model_id.split("/", 1)
                        if model_name.startswith(incomplete):
                            # Return short name since that's what user is typing
                            results.append((model_name, f"Model: {model_id}"))

            return results
    except Exception:
        pass

    return []
