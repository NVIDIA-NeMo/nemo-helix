# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the Nemotron-Personas datasets live, and how they get there.

UserSim samples simulated users from one census-grounded dataset per locale. The plugin keeps them in its
own Data Designer home, and each job step points Data Designer at it through ``MANAGED_ASSETS_ENV``.
"""

import json
import os
from collections.abc import Iterable
from pathlib import Path

from data_designer.cli.repositories.persona_repository import PersonaRepository
from data_designer.cli.services.download_service import DownloadService
from nemo_helix_plugin.config import nhx_user_data_dir
from nemo_user_sim_plugin.config import UserSimConfig

# Data Designer reads this once, at import, so it must be in the task's environment before Python starts.
MANAGED_ASSETS_ENV = "DATA_DESIGNER_MANAGED_ASSETS_PATH"


class PersonasMissingError(RuntimeError):
    pass


def personas_home() -> Path:
    configured = UserSimConfig.get().personas_dir
    return Path(configured) if configured else nhx_user_data_dir() / "user-sim" / "personas"


def managed_assets_path() -> Path:
    """Inside a job step, the path the step was given; anywhere else, the plugin's own home."""
    return Path(os.environ.get(MANAGED_ASSETS_ENV) or personas_home() / "managed-assets")


def dataset_path(locale: str) -> Path:
    return managed_assets_path() / "datasets" / f"{locale}.parquet"


def require_personas(locale: str) -> None:
    if not dataset_path(locale).is_file():
        raise PersonasMissingError(
            f"Nemotron-Personas for {locale} isn't installed. Run: nemo user-sim setup --locale {locale}"
        )


def download_personas(locale: str) -> Path:
    """Download one locale from NGC with the NGC CLI's default config; returns the dataset path."""
    DownloadService(personas_home(), PersonaRepository()).download_persona_dataset(locale)
    return dataset_path(locale)


TRAITS = ("patience", "verbosity", "cooperativeness", "tech_literacy")


def kept_rows(stored: list[dict], *, removed: Iterable[str], users: int) -> list[dict]:
    """A preview's rows minus the personas a reviewer removed, and no more of them than the run asks for."""
    gone = set(removed)
    return [row for row in stored if row["persona_uuid"] not in gone][:users]


def top_up(kept: list[dict], drawn: list[dict], removed: Iterable[str]) -> list[dict]:
    """A preview's kept rows, then the newly drawn ones whose persona it hasn't shown yet."""
    seen = {row["persona_uuid"] for row in kept} | set(removed)
    fresh = []
    for row in drawn:
        if row["persona_uuid"] not in seen:
            seen.add(row["persona_uuid"])
            fresh.append(row)
    return kept + fresh


def persona_card(row: dict) -> dict:
    """Who one simulated user is, from a materialized UserSim row, for reviewing a draw before running it."""
    profile = row.get("behavioral_profile") or "{}"
    profile = json.loads(profile) if isinstance(profile, str) else profile
    return {
        "persona_uuid": row["persona_uuid"],
        "name": row.get("persona_name"),
        "age": row.get("persona_age_bin"),
        "education": row.get("persona_education_level"),
        "occupation": row.get("persona_occupation"),
        "location": row.get("persona_location"),
        "style": row.get("user_interaction_style"),
        "probe": row.get("probe_type"),
        "traits": {trait: profile.get(trait) for trait in TRAITS},
    }
