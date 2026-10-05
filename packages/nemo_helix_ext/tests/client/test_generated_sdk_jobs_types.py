# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generated ``nemo_helix`` SDK re-exports the jobs types owned by ``nemo_helix_plugin``."""


def test_legacy_jobs_type_import_paths_resolve_to_source_owned_models() -> None:
    from nemo_helix.types import HelixJobStatusResponse as TopLevelStatusResponse
    from nemo_helix.types.jobs import HelixJobStep
    from nemo_helix.types.shared import HelixJobStatusResponse as SharedStatusResponse
    from nemo_helix_plugin.jobs.schemas import HelixJobStatusResponse
    from nemo_helix_plugin.jobs.types import HelixJobStepResponse

    assert HelixJobStep is HelixJobStepResponse
    assert SharedStatusResponse is HelixJobStatusResponse
    assert TopLevelStatusResponse is HelixJobStatusResponse
