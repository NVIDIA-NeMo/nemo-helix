# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect a locale's Nemotron Personas fileset and print its available fields.

Person sampling on NeMo Helix reads locale data from a fileset in the
``system`` workspace, so this script reads the same fileset the engine will.

Fields are split into two groups based on the with_synthetic_personas setting:
  - PII fields: always included in person sampling
  - SYNTHETIC PERSONA fields: only included when with_synthetic_personas=True

Usage: python get_person_object_schema.py <locale>
Example: python get_person_object_schema.py en_US
"""

from __future__ import annotations

import sys
from typing import NoReturn

import pyarrow.parquet as pq
from data_designer.engine.sampling_gen.entities.dataset_based_person_fields import PERSONA_FIELDS, PII_FIELDS
from data_designer_nemo.filesystem import make_filesystem
from data_designer_nemo.nemotron_personas import (
    SUPPORTED_LOCALES,
    WORKSPACE,
    get_locale_fileset_file_ref,
    get_resource_name_for_locale,
)
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.files.client import FilesClient


def _fail(message: str) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def _require_fileset(sdk: NeMoHelix, locale: str) -> None:
    """Exit with an actionable message when the locale's fileset isn't usable."""
    fileset_name = get_resource_name_for_locale(locale)
    files = client_from_platform(sdk, FilesClient)

    try:
        files.get_fileset(name=fileset_name, workspace=WORKSPACE)
    except NotFoundError:
        _fail(
            f"the Nemotron Personas fileset for locale {locale!r} does not exist yet.\n"
            f"  Expected fileset {WORKSPACE}/{fileset_name}.\n"
            "  Create it first (requires an NGC API key secret registered in NeMo Helix):\n"
            f"    nemo data-designer personas make-fileset --locale {locale} \\\n"
            "      --api-key-secret <workspace>/<secret-name>"
        )
    except PermissionDeniedError:
        _fail(
            f"access denied to workspace {WORKSPACE!r}, which holds the Nemotron Personas filesets.\n"
            "  Ask a platform administrator to grant read access, or to run:\n"
            f"    nemo data-designer personas make-fileset --locale {locale} "
            "--api-key-secret <workspace>/<secret-name>"
        )
    except Exception as exc:
        _fail(f"could not reach the NeMo Helix Files service to look up {WORKSPACE}/{fileset_name}: {exc}")


def main(locale: str) -> None:
    if locale not in SUPPORTED_LOCALES:
        _fail(f"unsupported locale {locale!r}; choose from {', '.join(sorted(SUPPORTED_LOCALES))}")

    try:
        sdk = NeMoHelix()
    except Exception as exc:
        _fail(
            f"could not connect to NeMo Helix: {exc}\n"
            "  Check that the CLI is configured and pointed at a reachable platform "
            "(`nemo config current-context`)."
        )

    _require_fileset(sdk, locale)

    file_ref = get_locale_fileset_file_ref(locale)
    try:
        with make_filesystem(sdk).open(file_ref, "rb") as f:
            schema = {field.name: str(field.type) for field in pq.read_schema(f)}
    except Exception as exc:
        _fail(f"could not read persona data at {file_ref!r}: {exc}")

    pii = {k: v for k, v in schema.items() if k in PII_FIELDS and v != "null"}
    persona = {k: v for k, v in schema.items() if k in PERSONA_FIELDS and v != "null"}

    print(f"=== {locale} PII fields (always included) ({len(pii)}) ===")
    for name, dtype in pii.items():
        print(f"  {name}: {dtype}")

    print(f"\n=== {locale} SYNTHETIC PERSONA fields (with_synthetic_personas=True) ({len(persona)}) ===")
    for name, dtype in persona.items():
        print(f"  {name}: {dtype}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <locale>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
