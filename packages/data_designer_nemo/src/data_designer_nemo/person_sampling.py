# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import data_designer.config as dd
from data_designer_nemo.errors import NDDInternalError
from data_designer_nemo.nemotron_personas import get_resource_name_for_locale
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.files.client import AsyncFilesClient

logger = logging.getLogger(__name__)


async def ensure_nemotron_personas_filesets(config: dd.DataDesignerConfig, sdk: AsyncNeMoHelix) -> None:
    """Validate filesets exist for all locales required to execute the given config."""
    locales = _get_required_personas_locales(config)
    if not locales:
        return

    unreachable_locales = set()
    files = client_from_platform(sdk, AsyncFilesClient)

    for locale in locales:
        fileset_name = get_resource_name_for_locale(locale)
        try:
            await files.get_fileset(name=fileset_name, workspace="system")
        except NotFoundError:
            logger.error(
                f"Nemotron personas fileset {fileset_name!r} for locale {locale!r} is missing in workspace 'system'. "
                "Create it with `nemo data-designer personas make-fileset --locale <locale>`."
            )
            unreachable_locales.add(locale)
        except PermissionDeniedError:
            logger.error("Access denied to Nemotron personas filesets in workspace 'system'")
            unreachable_locales.add(locale)
        except Exception:
            logger.exception(f"Failed to verify Nemotron personas fileset for locale {locale!r}")
            unreachable_locales.add(locale)

    if unreachable_locales:
        raise NDDInternalError(f"Failed to access Nemotron personas filesets for locales: {unreachable_locales}")


def _get_required_personas_locales(config: dd.DataDesignerConfig) -> set[str]:
    """Extract all unique locales required by PersonSampler columns in the config.

    Returns:
        Set of locale strings (e.g., {"en_US", "ja_JP"}), empty if no PersonSampler columns
    """
    locales = set()
    for column in config.columns:
        if isinstance(column, dd.SamplerColumnConfig) and isinstance(column.params, dd.PersonSamplerParams):
            locales.add(column.params.locale)
    return locales
