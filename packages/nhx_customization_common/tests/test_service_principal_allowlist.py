# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every customization backend must authenticate as a service name the auth
service actually recognizes, or its job pods fail with a 502 the moment they
try to touch the Files or Jobs API (see ASTD-648).
"""

import pytest
from nhx.automodel.images import FILE_IO_TASK_COMMAND as AUTOMODEL_FILE_IO
from nhx.automodel.images import MODEL_ENTITY_TASK_COMMAND as AUTOMODEL_MODEL_ENTITY
from nhx.core.auth.app.account_resolution import _available_service_names
from nhx.core.auth.config import AuthServiceConfig
from nhx.rl.images import FILE_IO_TASK_COMMAND as RL_FILE_IO
from nhx.rl.images import MODEL_ENTITY_TASK_COMMAND as RL_MODEL_ENTITY
from nhx.unsloth.images import FILE_IO_TASK_COMMAND as UNSLOTH_FILE_IO
from nhx.unsloth.images import MODEL_ENTITY_TASK_COMMAND as UNSLOTH_MODEL_ENTITY


def _service_name(command: list[str]) -> str:
    return command[command.index("--service-name") + 1]


@pytest.mark.parametrize(
    "backend,command",
    [
        ("automodel file_io", AUTOMODEL_FILE_IO),
        ("automodel model_entity", AUTOMODEL_MODEL_ENTITY),
        ("unsloth file_io", UNSLOTH_FILE_IO),
        ("unsloth model_entity", UNSLOTH_MODEL_ENTITY),
        ("rl file_io", RL_FILE_IO),
        ("rl model_entity", RL_MODEL_ENTITY),
    ],
)
def test_backend_service_name_is_an_allowed_principal(backend: str, command: list[str]) -> None:
    allowed = _available_service_names(AuthServiceConfig())
    service_name = _service_name(command)
    assert service_name in allowed, (
        f"{backend} authenticates as service:{service_name!r}, which the auth "
        f"service does not recognize. Every job step call will 502. Use the "
        f"registered 'customization' identity instead."
    )
