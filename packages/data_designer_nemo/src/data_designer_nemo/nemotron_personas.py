# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from typing import Literal

from data_designer.config.utils.constants import NEMOTRON_PERSONAS_DATASET_SIZES
from nemo_platform_plugin.client.errors import ConflictError
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.storage_config import NGCStorageConfig
from nemo_platform_plugin.files.types import CreateFilesetRequest, FilesetPurpose
from nemo_platform_plugin.schema import SecretRef

logger = logging.getLogger(__name__)


@dataclass
class FilesetAsset:
    locale: str
    filename: str = ""

    def __post_init__(self):
        # By default, the parquet filename stem exactly matches the locale.
        # If that is ever *not* the case, set `filename` when creating the FilesetAsset.
        if not self.filename:
            self.filename = f"{self.locale}.parquet"


SUPPORTED_LOCALES = {locale: FilesetAsset(locale=locale) for locale in NEMOTRON_PERSONAS_DATASET_SIZES.keys()}

NGC_ORG = "nvidia"
NGC_TEAM = "nemotron-personas"

WORKSPACE = "system"

_CREATED = "created"
_EXISTS = "exists"
NemotronPersonasFilesetSyncResult = Literal["created", "exists"]


def get_resource_name_for_locale(locale: str) -> str:
    """Returns the resource name for the Nemotron personas dataset
    for the given locale. This resource name is both the name of
    the NGC resource and the name of the fileset pointing to it.
    """
    return f"nemotron-personas-dataset-{locale.lower()}"


def get_file_path_for_locale(locale: str) -> str:
    """Returns the file path for the given locale. This represents both
    the remote path in the NGC resource and the path inside a fileset.
    """
    return SUPPORTED_LOCALES[locale].filename


def get_locale_fileset_file_ref(locale: str) -> str:
    """Returns the fully qualified reference to the Nemotron personas dataset
    parquet file for the given locale.
    """
    fileset = get_resource_name_for_locale(locale)
    file_path = get_file_path_for_locale(locale)

    return f"{WORKSPACE}/{fileset}#{file_path}"


def sync_nemotron_personas_fileset(
    *,
    files: FilesClient,
    locale: str,
    api_key_secret: str,
) -> NemotronPersonasFilesetSyncResult:
    """
    Sync the NGC-backed fileset for a single Nemotron personas locale.

    Args:
        files: Files typed client.
        locale: Locale identifier (e.g. 'en_US').
        api_key_secret: Fully qualified secret reference for the NGC API key.

    Returns:
        "created" if fileset was created, "exists" if it already existed.
    """
    logger.info("Syncing Nemotron personas fileset", extra={"locale": locale})

    try:
        _create_fileset(files, locale, api_key_secret)
        logger.info("Successfully created Nemotron personas fileset", extra={"locale": locale})
        return _CREATED
    except ConflictError:
        logger.debug("Nemotron personas fileset already exists", extra={"locale": locale})
        return _EXISTS


def _create_fileset(files: FilesClient, locale: str, api_key_secret: str) -> None:
    files.create_fileset(
        workspace=WORKSPACE,
        body=CreateFilesetRequest(
            name=get_resource_name_for_locale(locale),
            description=f"Nemotron Personas dataset for locale: {locale!r}",
            purpose=FilesetPurpose.DATASET,
            storage=_get_storage_config_for_locale(locale, api_key_secret),
            cache=True,
        ),
    )


def _get_storage_config_for_locale(locale: str, api_key_secret: str) -> NGCStorageConfig:
    resource_name = get_resource_name_for_locale(locale)

    return NGCStorageConfig(
        api_key_secret=SecretRef(api_key_secret),
        org=NGC_ORG,
        team=NGC_TEAM,
        target=resource_name,
        target_type="resource",
    )
