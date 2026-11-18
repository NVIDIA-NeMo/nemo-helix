# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for helix-operator: the ``helix-operator:`` section, or ``NEMO_HELIX_OPERATOR_*``."""

from typing import ClassVar

from nemo_helix_plugin.config import NemoConfig
from pydantic import Field


class HelixOperatorConfig(NemoConfig):
    """Configuration for building nhx-api images from an uploaded checkout."""

    plugin_name: ClassVar[str] = "helix-operator"
    plugin_description: ClassVar[str] = "Builds nhx-api images from an uploaded NeMo Helix checkout."

    build_set: str = Field(
        default="helix",
        description="The builder build set every build is submitted as. Its revisions number the builds.",
    )
    output_repository: str = Field(
        default="nhx-api",
        pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$",
        description=(
            "Repository the image is published to, under the builder's `<repository_prefix>/<workspace>/`. "
            "A single path segment: GitLab refuses to push deeper than two levels below the project."
        ),
    )
    platform: str = Field(default="linux/amd64", description="The platform to build, as `os/architecture`.")
    operator_url: str | None = Field(
        default=None,
        description=(
            "Where the CLI reaches the helix-operator service. Unset uses the global base URL, which only works "
            "if that also serves `/apis/helix-operator`."
        ),
    )
    base_image: str = Field(
        default="ghcr.io/nvidia-nemo/nemo-helix/nhx-api:70332d17b8562e2b1070da40d733210b19e43b70",
        description=(
            "The image `upload` writes into the generated Dockerfile's `FROM`. It must be pullable anonymously, "
            "and its dependencies must match the uploaded source."
        ),
    )
