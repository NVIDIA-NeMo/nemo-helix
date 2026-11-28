# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Models controller startup wiring."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nhx.core.models.controllers import main


def test_controller_uses_platform_qualified_puller_image() -> None:
    stop_signal = threading.Event()
    stop_signal.set()
    client = MagicMock()
    platform = SimpleNamespace(image_registry="registry.example.test/helix", image_tag="release")
    with (
        patch("nemo_helix_plugin.jobs.image.get_platform_config", return_value=platform),
        patch.object(main, "get_async_nemo_client", return_value=client),
        patch.object(main.BackendRegistry, "from_config") as from_config,
        patch.object(main, "wait_for_service_ready", return_value=False),
    ):
        main.run(parent_stop_signal=stop_signal)

    from_config.assert_called_once_with(
        client=client,
        backend_configs=main.backends,
        huggingface_model_puller="registry.example.test/helix/nhx-api:release",
    )
